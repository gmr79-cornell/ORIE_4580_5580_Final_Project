from __future__ import annotations

import random
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from policy_sim import simulate_p2c, simulate_ppb


# ---------------------------------------------------------
# 1. Helpers for single-run steady-state summary
# ---------------------------------------------------------

def _get_column(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    """
    Helper: return df[primary] if present, else df[fallback].
    Raises KeyError if neither exists.
    """
    if primary in df.columns:
        return df[primary]
    if fallback in df.columns:
        return df[fallback]
    raise KeyError(f"Expected column '{primary}' or '{fallback}' in DataFrame.")


def summarize_single_run(
    df: pd.DataFrame,
    lam: float,
    policy_name: str,
    warmup_jobs: int = 0,
) -> Dict[str, float] | None:
    """
    Given the full DataFrame for a single simulation run (one policy, one λ),
    drop an initial warm-up segment and compute steady-state metrics.

    Required columns: 'arrival_time', 'completion_time', 'TTFT',
    and either 'Latency' or 'completion_latency'. 'avg_TBT' is optional.
    """
    if df is None or df.empty:
        return None

    # Sort by completion time to define "jobs order"
    df_sorted = df.sort_values(by="completion_time").reset_index(drop=True)

    # Drop warm-up jobs if we have enough data
    if warmup_jobs > 0 and len(df_sorted) > warmup_jobs:
        df_ss = df_sorted.iloc[warmup_jobs:].reset_index(drop=True)
    else:
        df_ss = df_sorted

    if df_ss.empty:
        return None

    arrival_col = _get_column(df_ss, "arrival_time", "arrival")
    finish_col = _get_column(df_ss, "completion_time", "Finish")

    t_start = float(arrival_col.min())
    t_end = float(finish_col.max())
    total_time = t_end - t_start

    if total_time <= 0:
        return None

    ttft_vals = _get_column(df_ss, "TTFT", "ttft")
    lat_vals = _get_column(df_ss, "Latency", "completion_latency")

    mean_ttft = float(ttft_vals.mean())
    mean_latency = float(lat_vals.mean())
    p95_latency = float(lat_vals.quantile(0.95))
    throughput = len(df_ss) / total_time  # jobs per second

    result: Dict[str, float] = {
        "policy": policy_name,
        "lambda": lam,
        "mean_TTFT": mean_ttft,
        "mean_latency": mean_latency,
        "P95_latency": p95_latency,
        "throughput": throughput,
    }

    if "avg_TBT" in df_ss.columns:
        result["mean_avg_TBT"] = float(df_ss["avg_TBT"].mean())

    return result


# -------------------------------------------------------
# 2. Run multiple replications for each λ and each policy
# -------------------------------------------------------

def run_policy_grid_with_replications(
    lambdas: List[float],
    num_jobs_total: int,
    warmup_jobs: int,
    prompt_len: int,
    output_len: int,
    max_batch_size: int,
    replications: int = 20,
    seed: int = 123,
) -> pd.DataFrame:
    """
    For each λ in `lambdas` and each policy (P2C, PPB), run multiple
    independent replications and estimate steady-state metrics with
    95% confidence intervals.

    Returns
    -------
    pd.DataFrame
        One row per (policy, λ, metric) with columns:
        ['policy', 'lambda', 'metric', 'mean', 'ci_low', 'ci_high', 'n_rep']
    """
    meta_rng = random.Random(seed)
    rows: List[Dict[str, float]] = []

    for lam in lambdas:
        for policy_name, sim_fn, extra_kwargs in [
            ("P2C", simulate_p2c, {}),
            ("PPB", simulate_ppb, {"max_batch_size": max_batch_size}),
        ]:
            # Collect metric samples across replications
            samples: Dict[str, List[float]] = {
                "mean_TTFT": [],
                "mean_latency": [],
                "P95_latency": [],
                "throughput": [],
                # 'mean_avg_TBT' will be added dynamically if present
            }

            for _ in range(replications):
                rep_seed = meta_rng.randint(0, 10**9)

                df, _ = sim_fn(
                    lam=lam,
                    num_jobs=num_jobs_total,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    seed=rep_seed,
                    **extra_kwargs,
                )

                summary = summarize_single_run(
                    df=df,
                    lam=lam,
                    policy_name=policy_name,
                    warmup_jobs=warmup_jobs,
                )
                if summary is None:
                    continue

                # Push metrics into samples dict
                for key, value in summary.items():
                    if key in ("policy", "lambda"):
                        continue
                    if key not in samples:
                        samples[key] = []
                    samples[key].append(float(value))

            # Aggregate across replications with 95% CI
            for metric_name, metric_samples in samples.items():
                if not metric_samples:
                    continue

                arr = np.array(metric_samples, dtype=float)
                m = float(arr.mean())
                s = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                se = s / max(np.sqrt(len(arr)), 1.0)
                half_width = 1.96 * se

                rows.append(
                    {
                        "policy": policy_name,
                        "lambda": lam,
                        "metric": metric_name,
                        "mean": m,
                        "ci_low": m - half_width,
                        "ci_high": m + half_width,
                        "n_rep": len(arr),
                    }
                )

    return pd.DataFrame(rows)


# -------------------------------------------------------
# 3. Block-7 plots from the replicated summary
# -------------------------------------------------------

def plot_policy_replications(results_df: pd.DataFrame) -> None:
    """
    Produce the two Block-7 figures:

      Figure 1 (3 panels):
        - Mean TTFT vs λ
        - Mean total latency vs λ
        - Throughput vs λ (including offered load λ line)

      Figure 2:
        - Tail latency (P95) vs λ

    The input should be the long-format DataFrame returned by
    run_policy_grid_with_replications, with columns:
      ['policy', 'lambda', 'metric', 'mean', 'ci_low', 'ci_high', 'n_rep'].
    """
    if results_df is None or results_df.empty:
        print("No data to plot for Block 7.")
        return

    # ---- Figure 1: Mean TTFT, mean total latency, throughput ----
    metrics_for_panels = [
        ("mean_TTFT", "Mean TTFT vs arrival rate", "Mean TTFT (s)"),
        ("mean_latency", "Mean total latency vs arrival rate", "Mean Total Latency (s)"),
        ("throughput", "Throughput vs arrival rate", "Throughput (req/s)"),
    ]

    policies = sorted(results_df["policy"].unique())
    lambda_vals = sorted(results_df["lambda"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for idx, (metric_name, title, ylabel) in enumerate(metrics_for_panels):
        ax = axes[idx]
        for policy in policies:
            sub = results_df[
                (results_df["metric"] == metric_name)
                & (results_df["policy"] == policy)
            ].sort_values("lambda")
            if sub.empty:
                continue
            ax.plot(
                sub["lambda"].values,
                sub["mean"].values,
                marker="o",
                label=policy,
            )
        # Offered load line on throughput panel
        if metric_name == "throughput":
            ax.plot(
                lambda_vals,
                lambda_vals,
                linestyle=":",
                color="gray",
                label="Offered load (λ)",
            )
        ax.set_title(title)
        ax.set_xlabel("Arrival rate λ (req/s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(title="Policy")

    plt.tight_layout()
    plt.show()

    # ---- Figure 2: Tail latency (P95) vs λ ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

    for policy in policies:
        sub = results_df[
            (results_df["metric"] == "P95_latency")
            & (results_df["policy"] == policy)
        ].sort_values("lambda")
        if sub.empty:
            continue
        ax2.plot(
            sub["lambda"].values,
            sub["mean"].values,
            marker="o",
            label=policy,
        )

    ax2.set_title("Tail latency (P95) vs arrival rate")
    ax2.set_xlabel("Arrival rate λ (req/s)")
    ax2.set_ylabel("P95 total latency (s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="Policy")

    plt.tight_layout()
    plt.show()
