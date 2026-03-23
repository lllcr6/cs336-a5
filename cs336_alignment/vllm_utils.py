from __future__ import annotations

from contextlib import contextmanager
import gc
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
    from vllm import LLM

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=device == "cpu",
    )
    try:
        yield llm
    finally:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def load_policy_into_vllm_instance(policy: Any, llm: Any) -> None:
    """Load policy weights into a live vLLM model instance.

    Args:
        policy: Policy model whose weights should be copied.
        llm: Live vLLM instance that should receive the weights.
    """
    state_dict = policy.state_dict()
    model = getattr(getattr(llm, "llm_engine", None), "model_executor", None)
    if model is None:
        raise ValueError("Could not locate vLLM model executor on the provided instance")

    target = None
    for attr_path in [
        ("driver_worker", "model_runner", "model"),
        ("driver_worker", "model"),
        ("model_runner", "model"),
    ]:
        obj = model
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            target = obj
            break

    if target is None:
        raise ValueError("Could not locate underlying model object on vLLM instance")

    target.load_state_dict(state_dict, strict=False)
    for param in target.parameters():
        param.requires_grad_(False)


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
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "include_stop_str_in_output": include_stop_str_in_output,
    }
    if min_tokens is not None:
        kwargs["min_tokens"] = min_tokens
    if stop is not None:
        kwargs["stop"] = stop
    return SamplingParams(**kwargs)
