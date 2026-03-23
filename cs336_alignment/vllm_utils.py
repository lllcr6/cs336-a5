from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def init_vllm(
    model_id: str,
    *,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> Iterator[Any]:
    """Initialize a vLLM instance for offline generation.

    Args:
        model_id: HuggingFace model id or local model path.
        device: Device string used by vLLM.
        seed: Random seed for reproducible sampling.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.

    Yields:
        A vLLM instance.
    """
    # TODO: instantiate vLLM with the desired runtime patches and cleanup semantics.
    raise NotImplementedError


def load_policy_into_vllm_instance(policy: Any, llm: Any) -> None:
    """Load policy weights into a live vLLM model instance.

    Args:
        policy: Policy model whose weights should be copied.
        llm: Live vLLM instance that should receive the weights.
    """
    # TODO: copy the policy weights into the underlying vLLM model.
    raise NotImplementedError


def build_sampling_params(
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    min_tokens: int | None = None,
    stop: list[str] | None = None,
    include_stop_str_in_output: bool = False,
) -> Any:
    """Build a vLLM SamplingParams object.

    Args:
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_tokens: Maximum number of generated tokens.
        min_tokens: Optional minimum number of generated tokens.
        stop: Optional stop strings.
        include_stop_str_in_output: Whether to keep stop strings in the output.

    Returns:
        A vLLM SamplingParams instance.
    """
    # TODO: construct and return SamplingParams.
    raise NotImplementedError

