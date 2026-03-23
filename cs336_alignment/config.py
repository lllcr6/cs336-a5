"""Shared configuration scaffolds for Assignment 5 training and evaluation.

These dataclasses are intentionally lightweight and non-prescriptive. They
provide a stable surface for the scaffold modules and the Colab notebook to
share configuration without implementing the actual algorithms yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


LossType = Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]


@dataclass(slots=True)
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    project: str = "cs336-assignment5-alignment"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    log_dir: Path | None = None


@dataclass(slots=True)
class CheckpointConfig:
    """Configuration for local checkpoint persistence and resume."""

    output_dir: Path = Path("outputs/checkpoints")
    save_every_steps: int = 10
    max_checkpoints: int | None = 3
    resume_from_checkpoint: Path | None = None


@dataclass(slots=True)
class DriveSyncConfig:
    """Configuration for mirroring artifacts into Google Drive."""

    enabled: bool = False
    drive_root: Path | None = None
    per_run_subdir: str = "assignment5_alignment"
    checkpoint_dirname: str = "checkpoints"
    log_dirname: str = "logs"


@dataclass(slots=True)
class EvalConfig:
    """Configuration for evaluation and offline generation scaffolds."""

    prompts_path: Path | None = None
    output_dir: Path = Path("outputs/eval")
    batch_size: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    min_tokens: int | None = None
    max_tokens: int = 1024
    stop_tokens: list[str] = field(default_factory=list)
    include_stop_str_in_output: bool = False
    eval_every_steps: int = 10
    num_examples: int | None = None


@dataclass(slots=True)
class SFTConfig:
    """Configuration placeholder for supervised finetuning."""

    train_steps: int = 0
    learning_rate: float = 0.0
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    normalize_constant: float = 1.0
    eval_every_steps: int = 10
    save_every_steps: int = 10
    log_every_steps: int = 1
    clip_grad_norm: float | None = 1.0
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    drive_sync: DriveSyncConfig = field(default_factory=DriveSyncConfig)


@dataclass(slots=True)
class ExpertIterationConfig:
    """Configuration placeholder for expert iteration."""

    n_ei_steps: int = 0
    rollout_batch_size: int = 0
    rollouts_per_example: int = 0
    sft_epochs_per_round: int = 0
    eval_every_steps: int = 10
    save_every_steps: int = 10
    log_every_steps: int = 1
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    drive_sync: DriveSyncConfig = field(default_factory=DriveSyncConfig)


@dataclass(slots=True)
class GRPOConfig:
    """Configuration placeholder for GRPO training."""

    train_steps: int = 0
    learning_rate: float = 0.0
    rollout_batch_size: int = 256
    group_size: int = 8
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 1
    epochs_per_rollout_batch: int = 1
    loss_type: LossType = "reinforce_with_baseline"
    advantage_eps: float = 1e-6
    cliprange: float = 0.2
    normalize_by_std: bool = True
    eval_every_steps: int = 10
    save_every_steps: int = 10
    log_every_steps: int = 1
    clip_grad_norm: float | None = 1.0
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    drive_sync: DriveSyncConfig = field(default_factory=DriveSyncConfig)
