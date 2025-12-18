from typing import List

# Core piecewise-linear service model parameters (Llama-3-70B example)
C_SETUP: float = 45.5      # ms – fixed setup cost per batch
A_MARGINAL: float = 0.30   # ms/token – marginal cost per token
B0_THRESHOLD: int = 64     # tokens – batch-size threshold b0


def batch_service_time_seconds(
    batch_token_load: int,
    c_setup: float = C_SETUP,
    a_marginal: float = A_MARGINAL,
    b0_threshold: int = B0_THRESHOLD,
) -> float:
    """
    Compute S(b) = c + a * max(0, b - b0) in SECONDS for a batch
    with token load `batch_token_load`.
    """
    if batch_token_load <= 0:
        return 0.0
    ms_time = c_setup + a_marginal * max(0, batch_token_load - b0_threshold)
    return ms_time / 1000.0  # ms -> seconds


def service_model_sanity_check() -> None:
    """
    Block 1 – Service Model Validation.
    Prints service times for some example batch loads.
    """
    server_loads = [10, 50, 100, 500]
    print("--- Service Model Validation ---")
    print(f"Parameters: c={C_SETUP}ms, a={A_MARGINAL}ms/tok, b0={B0_THRESHOLD}")
    for load in server_loads:
        duration = batch_service_time_seconds(load)
        print(f"Token Load: {load:3d} -> Service Time: {duration:.4f} sec")
