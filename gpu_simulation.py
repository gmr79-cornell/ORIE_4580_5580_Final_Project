import random
from typing import List, Optional

import numpy as np
import pandas as pd

from service_model import (
    C_SETUP,
    A_MARGINAL,
    B0_THRESHOLD,
    batch_service_time_seconds,
)
from gpu_core import GPUServer, Request


def run_simulation(
    arrival_rate_lambda: float = 1.0,
    max_duration: float = 60.0,
    max_batch_size: int = 4,
    fixed_prompt_len: Optional[int] = None,
    fixed_output_len: Optional[int] = None,
    c_setup: float = C_SETUP,
    a_marginal: float = A_MARGINAL,
    b0_threshold: int = B0_THRESHOLD,
) -> pd.DataFrame:
    """
    Block 2 – Core discrete-event simulation for a single-GPU batched LLM server.

    Scheduling:
    - FCFS admission into batches up to max_batch_size.
    - Prefill: batched over all jobs that have not been prefills yet.
    - Decode: round-robin, 1 token per job in the active batch per step.

    Returns
    -------
    DataFrame with per-request metrics including TTFT and avg_TBT.
    """
    server = GPUServer(max_batch_size=max_batch_size)
    current_time = 0.0
    completed_requests: List[Request] = []

    # --- Generate arrivals (Poisson process) ---
    all_requests: List[Request] = []
    t = 0.0
    req_id = 0
    B_MAX = 32       # cap for output budget
    p_geom = 0.15    # geometric parameter for B_i

    while t < max_duration:
        dt = random.expovariate(arrival_rate_lambda)
        t += dt
        if t >= max_duration:
            break

        # Prompt length L_i
        prompt_len = (
            fixed_prompt_len if fixed_prompt_len is not None
            else random.randint(50, 200)
        )

        # Output budget B_i
        if fixed_output_len is not None:
            output_budget = fixed_output_len
        else:
            raw_length = np.random.geometric(p_geom)  # 1, 2, ...
            output_budget = int(min(max(raw_length, 1), B_MAX))

        r = Request(
            req_id=req_id,
            arrival_time=t,
            prompt_len=prompt_len,
            output_budget=output_budget,
        )
        all_requests.append(r)
        req_id += 1

    print(f"--- Simulation Started: {len(all_requests)} requests generated ---")

    arrival_index = 0

    # --- Main simulation loop ---
    while current_time < max_duration or server.active_batch:
        # 1) Admit arrivals whose arrival_time <= current_time
        while arrival_index < len(all_requests):
            next_req = all_requests[arrival_index]
            if next_req.arrival_time <= current_time:
                server.queue.append(next_req)
                arrival_index += 1
            else:
                break

        # 2) If system idle and more arrivals in future, jump to next arrival
        if (
            not server.active_batch
            and not server.queue
            and arrival_index < len(all_requests)
        ):
            current_time = all_requests[arrival_index].arrival_time
            continue

        # 3) If system empty and no more arrivals, stop
        if (
            not server.active_batch
            and not server.queue
            and arrival_index >= len(all_requests)
        ):
            break

        # 4) Fill active batch (FCFS) up to max_batch_size
        while server.queue and len(server.active_batch) < server.max_batch_size:
            req = server.queue.pop(0)
            if req.start_time < 0:
                req.start_time = current_time
            server.active_batch.append(req)

        batch = server.active_batch
        if not batch:
            continue

        # 5) Decide between PREFILL vs DECODE
        needs_prefill = any(r.tokens_decoded == 0 for r in batch)

        if needs_prefill:
            # --- PREFILL PHASE ---
            batch_load = sum(
                r.prompt_len for r in batch if r.tokens_decoded == 0
            )
            step_duration = batch_service_time_seconds(
                batch_load,
                c_setup=c_setup,
                a_marginal=a_marginal,
                b0_threshold=b0_threshold,
            )
            prefill_end_time = current_time + step_duration
            current_time = prefill_end_time

            for r in batch:
                if r.tokens_decoded == 0:
                    # First token becomes available right after prefill
                    r.tokens_decoded = 1
                    if r.first_token_time < 0:
                        r.first_token_time = prefill_end_time
        else:
            # --- DECODE PHASE ---
            batch_load = len(batch)  # one token per request
            step_duration = batch_service_time_seconds(
                batch_load,
                c_setup=c_setup,
                a_marginal=a_marginal,
                b0_threshold=b0_threshold,
            )
            current_time += step_duration

            finished_indices = []
            for i, r in enumerate(batch):
                r.tokens_decoded += 1
                if r.is_complete:
                    r.finish_time = current_time
                    completed_requests.append(r)
                    finished_indices.append(i)

            # Remove completed jobs from active batch
            for i in sorted(finished_indices, reverse=True):
                server.active_batch.pop(i)

    # --- Compile per-request metrics into a DataFrame ---
    results = []
    for r in completed_requests:
        if r.first_token_time > 0:
            ttft = r.first_token_time - r.arrival_time
        else:
            ttft = None

        completion_latency = r.finish_time - r.arrival_time

        if r.first_token_time > 0 and r.output_budget > 1:
            avg_tbt = (r.finish_time - r.first_token_time) / (r.output_budget - 1)
        else:
            avg_tbt = None

        results.append(
            {
                "request_id": r.req_id,
                "arrival_time": r.arrival_time,
                "first_token_time": r.first_token_time,
                "completion_time": r.finish_time,
                "completion_latency": completion_latency,
                "TTFT": ttft,
                "avg_TBT": avg_tbt,
                "output_tokens": r.output_budget,
                # Alias for Block 4 code that expects 'Latency'
                "Latency": completion_latency,
            }
        )

    return pd.DataFrame(results)


def run_gpu_mm1_validation() -> None:
    """
    Block 4 – GPU-based M/M/1-style validation using the same service model,
    specialized to:
    - single job type: L = 100, B = 1
    - no setup cost (c = 0)
    - threshold b0 = 0
    - batch_size = 1  (no batching)
    """
    print("\n=== VALIDATION: M/M/1 COMPARISON ===")

    lam = 5.0  # arrival rate (λ)
    prompt_len = 100
    decode_len = 1
    total_tokens = prompt_len + decode_len

    # Average service time per job (seconds) if S(b) = a * b
    avg_service_time = (A_MARGINAL * total_tokens) / 1000.0
    mu = 1.0 / avg_service_time

    print(f"Theory: lambda = {lam:.2f}, mu = {mu:.2f}")
    if lam >= mu:
        print("Warning: lambda >= mu, system is unstable in M/M/1 theory.")
    else:
        theo_response_time = 1.0 / (mu - lam)
        print(f"Expected mean response time (M/M/1): {theo_response_time:.4f} sec")

    # Simulate with c_setup = 0 and b0_threshold = 0, batch_size=1, fixed sizes
    df = run_simulation(
        arrival_rate_lambda=lam,
        max_duration=2000.0,
        max_batch_size=1,
        fixed_prompt_len=prompt_len,
        fixed_output_len=decode_len,
        c_setup=0.0,
        a_marginal=A_MARGINAL,
        b0_threshold=0,
    )

    if df.empty:
        print("Simulation produced no completed jobs.")
        print("=== VALIDATION COMPLETE ===\n")
        return

    sim_mean_latency = df["Latency"].mean()
    print(f"Simulated mean latency (N={len(df)}): {sim_mean_latency:.4f} sec")
    print("=== VALIDATION COMPLETE ===\n")
