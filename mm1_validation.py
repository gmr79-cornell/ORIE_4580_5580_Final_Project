from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_mm1(
    lam: float,
    mu: float,
    num_jobs: int = 5000,
    seed: Optional[int] = None,
) -> float:
    """
    Simple M/M/1 FCFS simulation:
    - Arrivals ~ Poisson process with rate lam (Exp(lam) interarrivals)
    - Service times ~ Exp(mu)
    - Single server, FCFS

    Returns
    -------
    float
        Sample mean response time (arrival -> departure) over `num_jobs` customers.
    """
    if lam <= 0 or mu <= 0:
        raise ValueError("lam and mu must be positive")

    rng = random.Random(seed)

    # Generate arrival times
    arrival_times: List[float] = []
    t = 0.0
    for _ in range(num_jobs):
        t += rng.expovariate(lam)  # interarrival
        arrival_times.append(t)

    # Process jobs in FCFS order
    server_free_time = 0.0
    response_times: List[float] = []

    for a in arrival_times:
        # Start service cannot be earlier than either arrival or when server becomes free
        start_service = max(a, server_free_time)
        service_time = rng.expovariate(mu)
        finish_time = start_service + service_time

        server_free_time = finish_time
        response_times.append(finish_time - a)

    return float(np.mean(response_times))


def run_mm1_validation_demo(
    lam: float = 5.0,
    mu: float = 8.0,
    num_jobs: int = 5000,
    seed: int = 42,
) -> None:
    """
    Single-run comparison of theory vs simulation for one (lam, mu) pair.

    This reproduces the style of the notebook output, but note that a single
    run will not match theory perfectly because of Monte Carlo error.
    """
    if lam >= mu:
        print("Warning: For M/M/1 we require λ < μ for a stable system.")

    theo = 1.0 / (mu - lam)
    sim = simulate_mm1(lam=lam, mu=mu, num_jobs=num_jobs, seed=seed)

    print("=== M/M/1 Validation (Single Run) ===")
    print(f"lambda (arrival rate): {lam:.2f}")
    print(f"mu (service rate): {mu:.2f}")
    print(f"Theoretical E[T]: {theo:.4f} s")
    print(f"Simulated E[T]: {sim:.4f} s")
    print("Note: A single run may differ from theory due to sampling noise.\n")


def run_mm1_validation_with_ci(
    lam: float = 5.0,
    mu: float = 8.0,
    num_jobs: int = 5000,
    n_rep: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Improved validation for an M/M/1 queue:

    - Runs n_rep independent simulations with the specified lam, mu and num_jobs.
    - Computes the mean and 95% CI of the simulated E[T].
    - Compares this to the theoretical E[T] = 1 / (mu - lam).
    """
    if lam >= mu:
        print("Warning: For M/M/1 we require λ < μ for a stable system.\n")

    theo = 1.0 / (mu - lam)
    rng = random.Random(seed)

    samples = []
    for _ in range(n_rep):
        rep_seed = rng.randint(0, 10**9)
        sim_mean = simulate_mm1(lam=lam, mu=mu, num_jobs=num_jobs, seed=rep_seed)
        samples.append(sim_mean)

    samples_arr = np.array(samples)
    mean_sim = float(samples_arr.mean())
    std_sim = float(samples_arr.std(ddof=1)) if n_rep > 1 else 0.0
    se_sim = std_sim / max(np.sqrt(n_rep), 1.0)
    ci_low = mean_sim - 1.96 * se_sim
    ci_high = mean_sim + 1.96 * se_sim
    diff = mean_sim - theo

    print("=== M/M/1 Validation with CI ===")
    print(f"lambda (arrival rate): {lam:.2f}")
    print(f"mu (service rate): {mu:.2f}")
    print(f"num_jobs per replication: {num_jobs}")
    print(f"number of replications: {n_rep}")
    print(f"Theoretical E[T]: {theo:.4f} s")
    print(f"Simulated mean E[T]: {mean_sim:.4f} s")
    print(f"95% CI for simulated mean: [{ci_low:.4f}, {ci_high:.4f}] s")
    print(f"Difference (sim - theory): {diff:.4f} s\n")

    result_row = {
        "lambda": lam,
        "mu": mu,
        "num_jobs": num_jobs,
        "n_rep": n_rep,
        "theoretical_E_T": theo,
        "mean_sim_E_T": mean_sim,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "diff_sim_minus_theory": diff,
    }
    return pd.DataFrame([result_row])


def sweep_mm1_vs_lambda(
    mu: float = 8.0,
    lam_values: Optional[List[float]] = None,
    num_jobs: int = 3000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Sweep over a set of lambda values and compare theory vs simulated
    mean response time.

    For simplicity this uses a single run per λ (as in the notebook),
    but you can easily adapt it to use multiple replications by calling
    `run_mm1_validation_with_ci` inside the loop instead.
    """
    if lam_values is None:
        lam_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    rows = []
    rng = random.Random(seed)

    for lam in lam_values:
        if lam >= mu:
            # Skip unstable configurations
            continue

        theo = 1.0 / (mu - lam)
        rep_seed = rng.randint(0, 10**9)
        sim = simulate_mm1(lam=lam, mu=mu, num_jobs=num_jobs, seed=rep_seed)

        rows.append(
            {
                "lambda": lam,
                "mu": mu,
                "theoretical_E_T": theo,
                "simulated_E_T": sim,
            }
        )

    df = pd.DataFrame(rows)
    return df


def plot_mm1_validation(df: pd.DataFrame) -> None:
    """
    Plot theoretical vs simulated mean response time E[T] as a function of λ.
    """
    if df.empty:
        print("No data to plot for M/M/1 sweep.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(df["lambda"], df["theoretical_E_T"], marker="o", label="Theory 1/(μ-λ)")
    plt.plot(df["lambda"], df["simulated_E_T"], marker="x", linestyle="--", label="Simulation")
    plt.xlabel("λ (arrival rate)")
    plt.ylabel("Mean response time E[T] (s)")
    plt.title("M/M/1: Theory vs Simulation")
    plt.legend()
    plt.tight_layout()
    plt.show()
