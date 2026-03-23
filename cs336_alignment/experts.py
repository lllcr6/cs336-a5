from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import ExpertIterationConfig


def run_expert_iteration(
    *,
    model_id: str,
    train_dataset_path: str | Path,
    validation_dataset_path: str | Path,
    reward_fn: Callable[[str, str], dict[str, float]],
    config: ExpertIterationConfig,
) -> dict[str, Any]:
    """Scaffold entrypoint for expert iteration.

    Args:
        model_id: Base model identifier.
        train_dataset_path: Path to the training questions.
        validation_dataset_path: Path to the validation set.
        reward_fn: Reward function used to filter sampled outputs.
        config: Expert iteration configuration.

    Returns:
        Dictionary containing training state and evaluation summaries.
    """
    # TODO: sample rollouts, filter correct traces, and pass them into SFT rounds.
    raise NotImplementedError


def build_expert_iteration_dataset(
    *,
    prompts: list[str],
    responses: list[str],
    rewards: list[dict[str, float]],
) -> list[dict[str, str]]:
    """Filter rollout pairs into the SFT dataset used by expert iteration."""
    # TODO: retain only correct prompt/response pairs and normalize record structure.
    raise NotImplementedError

