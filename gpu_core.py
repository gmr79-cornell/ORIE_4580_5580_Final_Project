from dataclasses import dataclass
from typing import List


@dataclass
class Request:
    """
    Represents a single LLM query (user job).

    All times (arrival_time, start_time, finish_time, first_token_time)
    are tracked in SECONDS in the simulation.
    """
    req_id: int
    arrival_time: float
    prompt_len: int          # L_i: tokens in the input prompt
    output_budget: int       # B_i: max response length (tokens)

    # State tracking
    start_time: float = -1.0
    finish_time: float = -1.0
    tokens_decoded: int = 0
    first_token_time: float = -1.0   # for TTFT (first decode token)

    @property
    def is_complete(self) -> bool:
        return self.tokens_decoded >= self.output_budget


class GPUServer:
    """
    Single GPU worker with batched processing.

    This class holds the waiting queue and the current active batch.
    The service-time physics are handled by the service_model module.
    """

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.queue: List[Request] = []
        self.active_batch: List[Request] = []
