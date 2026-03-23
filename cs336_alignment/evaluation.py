from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import EvalConfig


def evaluate_vllm(
    vllm_model: Any,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params: Any,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a language model on prompts and collect reward metrics.

    Args:
        vllm_model: vLLM instance used for generation.
        reward_fn: Callable that scores a response against a ground truth.
        prompts: Prompts to evaluate.
        eval_sampling_params: Sampling parameters to pass to vLLM.
        output_path: Optional path where results should be serialized.

    Returns:
        Dictionary containing generations, rewards, and summary metrics.
    """
    # TODO: run generation, score outputs, and optionally serialize the results.
    raise NotImplementedError


def log_generations(
    *,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_info: list[dict[str, float]],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create a logging payload for sample generations.

    Args:
        prompts: Input prompts shown to the model.
        responses: Model responses.
        ground_truths: Reference answers.
        reward_info: Reward dictionaries for each example.
        output_dir: Optional path for writing artifacts.

    Returns:
        A structured logging summary.
    """
    # TODO: compute summary statistics and optionally write logs to disk.
    raise NotImplementedError


def run_zero_shot_baseline(
    *,
    model_id: str,
    dataset_path: str | Path,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_config: EvalConfig,
) -> dict[str, Any]:
    """Scaffold entrypoint for the zero-shot baseline evaluation."""
    # TODO: load prompts, generate responses, and evaluate baseline performance.
    raise NotImplementedError

