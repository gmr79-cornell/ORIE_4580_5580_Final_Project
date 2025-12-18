import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from service_model import batch_service_time_seconds


@dataclass
class SimpleRequest:
    """
    Simplified request object for policy comparison experiments.
    """
    req_id: int
    arrival_time: float
    prompt_len: int
    output_len: int
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    decoded_tokens: int = 0


def generate_homogeneous_requests(
    lam: float,
    num_jobs: int,
    prompt_len: int,
    output_len: int,
    seed: Optional[int] = None,
) -> List[SimpleRequest]:
    """
    Generate `num_jobs` Poisson arrivals with rate lam, all with the same
    prompt_len and output_len.
    """
    rng = random.Random(seed)
    t = 0.0
    requests: List[SimpleRequest] = []
    for j in range(num_jobs):
        t += rng.expovariate(lam)
        requests.append(SimpleRequest(j, t, prompt_len, output_len))
    return requests


# -------------------- Policy 1: Process-to-Completion (P2C) --------------------


def simulate_p2c(
    lam: float,
    num_jobs: int,
    prompt_len: int,
    output_len: int,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Single-server FCFS system.
    Each request runs to completion alone; modeled as a single batch
    with token load L + B.

    Returns
    -------
    df: per-request metrics
    sim_length: total simulated time (last_finish - first_arrival)
    """
    requests = generate_homogeneous_requests(
        lam=lam,
        num_jobs=num_jobs,
        prompt_len=prompt_len,
        output_len=output_len,
        seed=seed,
    )
    if not requests:
        return pd.DataFrame(), 0.0

    server_free_time = 0.0
    first_arrival = requests[0].arrival_time
    last_finish = first_arrival

    for r in requests:
        start_service = max(r.arrival_time, server_free_time)
        total_tokens = r.prompt_len + r.output_len
        service_time = batch_service_time_seconds(total_tokens)

        # Heuristic: TTFT = time to "prefill" the prompt fraction
        prefill_time = service_time * (r.prompt_len / total_tokens)
        r.first_token_time = start_service + prefill_time
        r.finish_time = start_service + service_time
        r.decoded_tokens = r.output_len

        server_free_time = r.finish_time
        last_finish = max(last_finish, r.finish_time)

    sim_length = max(last_finish - first_arrival, 1e-9)

    rows = []
    for r in requests:
        ttft = r.first_token_time - r.arrival_time
        latency = r.finish_time - r.arrival_time
        if r.output_len > 1:
            avg_tbt = (r.finish_time - r.first_token_time) / (r.output_len - 1)
        else:
            avg_tbt = 0.0

        rows.append(
            {
                "policy": "P2C",
                "req_id": r.req_id,
                "arrival_time": r.arrival_time,
                "completion_time": r.finish_time,
                "TTFT": ttft,
                "Latency": latency,
                "avg_TBT": avg_tbt,
            }
        )

    df = pd.DataFrame(rows)
    return df, sim_length


# -------- Policy 2: Prefill-Prioritizing Batched Decode (PPB) --------


def simulate_ppb(
    lam: float,
    num_jobs: int,
    prompt_len: int,
    output_len: int,
    max_batch_size: int = 8,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Prefill-prioritizing batched decode (Discrete Event Simulation).
    """
    requests = generate_homogeneous_requests(
        lam=lam,
        num_jobs=num_jobs,
        prompt_len=prompt_len,
        output_len=output_len,
        seed=seed,
    )

    if not requests:
        return pd.DataFrame(), 0.0

    t = 0.0
    arrival_index = 0
    N = len(requests)

    prefill_queue: List[SimpleRequest] = []
    decode_jobs: List[SimpleRequest] = []

    first_arrival = requests[0].arrival_time
    last_finish = first_arrival

    while arrival_index < N or prefill_queue or decode_jobs:
        # 1. Idle jump / Admit arrivals
        if not prefill_queue and not decode_jobs and arrival_index < N:
            t = max(t, requests[arrival_index].arrival_time)

        # Admit all arrivals up to time t
        while arrival_index < N and requests[arrival_index].arrival_time <= t:
            prefill_queue.append(requests[arrival_index])
            arrival_index += 1

        # Termination condition
        if not prefill_queue and not decode_jobs and arrival_index >= N:
            break

        if not prefill_queue and not decode_jobs and arrival_index < N:
            # Wait for more arrivals
            continue

        # 2. Scheduling logic: prioritize prefill
        if prefill_queue:
            # PREFILL phase
            batch = prefill_queue[:max_batch_size]
            prefill_queue = prefill_queue[max_batch_size:]
            prefill_tokens = sum(r.prompt_len for r in batch)
            step_time = batch_service_time_seconds(prefill_tokens)
            t += step_time

            for r in batch:
                r.decoded_tokens = 1  # first token generated
                if r.first_token_time is None:
                    r.first_token_time = t
                decode_jobs.append(r)

            # prefill is prioritized; loop back to see if more prefill work exists
            continue

        # 3. DECODE phase (only if prefill_queue is empty)
        if decode_jobs:
            active = [r for r in decode_jobs if r.decoded_tokens < r.output_len]
            if not active:
                decode_jobs = []
                continue

            active_batch = active[:max_batch_size]
            batch_tokens = len(active_batch)
            step_time = batch_service_time_seconds(batch_tokens)
            t += step_time

            for r in active_batch:
                r.decoded_tokens += 1
                if r.decoded_tokens >= r.output_len and r.finish_time is None:
                    r.finish_time = t
                    last_finish = max(last_finish, t)

            decode_jobs = [r for r in decode_jobs if r.finish_time is None]
            continue

    sim_length = max(last_finish - first_arrival, 1e-9)

    rows = []
    for r in requests:
        if r.finish_time is None or r.first_token_time is None:
            continue
        ttft = r.first_token_time - r.arrival_time
        latency = r.finish_time - r.arrival_time
        if r.output_len > 1:
            avg_tbt = (r.finish_time - r.first_token_time) / (r.output_len - 1)
        else:
            avg_tbt = 0.0

        rows.append(
            {
                "policy": "PPB",
                "req_id": r.req_id,
                "arrival_time": r.arrival_time,
                "completion_time": r.finish_time,
                "TTFT": ttft,
                "Latency": latency,
                "avg_TBT": avg_tbt,
            }
        )

    df = pd.DataFrame(rows)
    return df, sim_length


def run_policy_grid(
    lambdas: List[float],
    num_jobs: int = 2000,
    prompt_len: int = 100,
    output_len: int = 16,
    max_batch_size: int = 8,
    seed: int = 123,
) -> pd.DataFrame:
    """
    For each λ in lambdas, run P2C and PPB, collect summary metrics.
    """
    rows = []
    for lam in lambdas:
        seed_p2c = seed + int(1000 * lam)
        seed_ppb = seed + int(2000 * lam)

        df_p2c, len_p2c = simulate_p2c(
            lam=lam,
            num_jobs=num_jobs,
            prompt_len=prompt_len,
            output_len=output_len,
            seed=seed_p2c,
        )
        df_ppb, len_ppb = simulate_ppb(
            lam=lam,
            num_jobs=num_jobs,
            prompt_len=prompt_len,
            output_len=output_len,
            max_batch_size=max_batch_size,
            seed=seed_ppb,
        )

        for name, df, sim_len in [
            ("P2C", df_p2c, len_p2c),
            ("PPB", df_ppb, len_ppb),
        ]:
            if df.empty or sim_len <= 0 or len(df) < num_jobs * 0.5:
                continue

            mean_ttft = df["TTFT"].mean()
            p95_ttft = df["TTFT"].quantile(0.95)
            mean_lat = df["Latency"].mean()
            p95_lat = df["Latency"].quantile(0.95)
            mean_tbt = df["avg_TBT"].mean()
            throughput = len(df) / sim_len

            rows.append(
                {
                    "lambda": lam,
                    "policy": name,
                    "mean_TTFT": mean_ttft,
                    "p95_TTFT": p95_ttft,
                    "mean_Latency": mean_lat,
                    "p95_Latency": p95_lat,
                    "mean_avg_TBT": mean_tbt,
                    "Throughput": throughput,
                    "num_completed": len(df),
                }
            )

    return pd.DataFrame(rows)


def plot_policy_comparison(summary_df: pd.DataFrame) -> None:
    """
    Plot mean TTFT, mean Latency, and Throughput vs λ for each policy.
    """
    if summary_df.empty:
        print("No data to plot.")
        return

    policies = sorted(summary_df["policy"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Plot 1: Mean TTFT
    ax = axes[0]
    for pol in policies:
        sub = summary_df[summary_df["policy"] == pol]
        ax.plot(sub["lambda"], sub["mean_TTFT"], marker="o", label=pol)
    ax.set_xlabel(r"Arrival rate $\lambda$ (req/s)")
    ax.set_ylabel("Mean TTFT (s)")
    ax.set_title("Mean Time To First Token (TTFT)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean Latency
    ax = axes[1]
    for pol in policies:
        sub = summary_df[summary_df["policy"] == pol]
        ax.plot(sub["lambda"], sub["mean_Latency"], marker="o", label=pol)
    ax.set_xlabel(r"Arrival rate $\lambda$ (req/s)")
    ax.set_ylabel("Mean Total Latency (s)")
    ax.set_title("Mean Total Latency (E[T])")
    ax.grid(True, alpha=0.3)

    # Plot 3: Throughput
    ax = axes[2]
    for pol in policies:
        sub = summary_df[summary_df["policy"] == pol]
        ax.plot(sub["lambda"], sub["Throughput"], marker="o", label=pol)

    lam_vals = sorted(summary_df["lambda"].unique())
    ax.plot(
        lam_vals,
        lam_vals,
        linestyle=":",
        color="gray",
        label=r"Offered Load ($\lambda$)",
    )

    ax.set_xlabel(r"Arrival rate $\lambda$ (req/s)")
    ax.set_ylabel("Throughput (completed req/s)")
    ax.set_title("Achieved Throughput")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
