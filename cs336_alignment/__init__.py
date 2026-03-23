"""Public scaffold exports for the CS336 Assignment 5 package."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CheckpointConfig",
    "DriveSyncConfig",
    "EvalConfig",
    "ExpertIterationConfig",
    "GRPOConfig",
    "SFTConfig",
    "WandbConfig",
    "latest_checkpoint",
    "load_checkpoint",
    "resolve_resume_checkpoint",
    "run_expert_iteration",
    "run_grpo_training",
    "run_sft_training",
    "run_zero_shot_baseline",
    "save_checkpoint",
    "sync_checkpoint_to_drive",
]

_EXPORTS = {
    "CheckpointConfig": ("cs336_alignment.config", "CheckpointConfig"),
    "DriveSyncConfig": ("cs336_alignment.config", "DriveSyncConfig"),
    "EvalConfig": ("cs336_alignment.config", "EvalConfig"),
    "ExpertIterationConfig": ("cs336_alignment.config", "ExpertIterationConfig"),
    "GRPOConfig": ("cs336_alignment.config", "GRPOConfig"),
    "SFTConfig": ("cs336_alignment.config", "SFTConfig"),
    "WandbConfig": ("cs336_alignment.config", "WandbConfig"),
    "latest_checkpoint": ("cs336_alignment.checkpointing", "latest_checkpoint"),
    "load_checkpoint": ("cs336_alignment.checkpointing", "load_checkpoint"),
    "resolve_resume_checkpoint": (
        "cs336_alignment.checkpointing",
        "resolve_resume_checkpoint",
    ),
    "run_expert_iteration": ("cs336_alignment.experts", "run_expert_iteration"),
    "run_grpo_training": ("cs336_alignment.grpo", "run_grpo_training"),
    "run_sft_training": ("cs336_alignment.sft", "run_sft_training"),
    "run_zero_shot_baseline": (
        "cs336_alignment.evaluation",
        "run_zero_shot_baseline",
    ),
    "save_checkpoint": ("cs336_alignment.checkpointing", "save_checkpoint"),
    "sync_checkpoint_to_drive": (
        "cs336_alignment.checkpointing",
        "sync_checkpoint_to_drive",
    ),
}


def __getattr__(name: str):
    """Lazily resolve scaffold exports so package import stays lightweight."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
