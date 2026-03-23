"""Checkpointing and Google Drive sync scaffolds for Assignment 5."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    state: dict[str, Any],
    step: int,
    max_checkpoints: int | None = None,
) -> Path:
    """Persist a training checkpoint to disk.

    Args:
        checkpoint_dir: Directory that should receive the checkpoint.
        state: Serializable payload containing training/model state.
        step: Training step associated with the saved checkpoint.
        max_checkpoints: Optional retention cap for older checkpoints.

    Returns:
        Path to the saved checkpoint directory.
    """
    # TODO: write the checkpoint payload, retain metadata about the step, and
    # rotate older checkpoints when ``max_checkpoints`` is configured.
    raise NotImplementedError


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a checkpoint payload from disk."""
    # TODO: deserialize the checkpoint contents and restore the expected state
    # dictionary used by the future SFT/EI/GRPO training loops.
    raise NotImplementedError


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the newest checkpoint found under ``checkpoint_dir``."""
    # TODO: scan the checkpoint directory and choose the highest-step or newest
    # checkpoint according to the eventual retention naming convention.
    raise NotImplementedError


def sync_checkpoint_to_drive(
    local_checkpoint_path: str | Path,
    drive_checkpoint_root: str | Path,
) -> Path:
    """Mirror a freshly written checkpoint into Google Drive.

    Args:
        local_checkpoint_path: Local checkpoint path created during training.
        drive_checkpoint_root: Drive directory used for long-lived checkpoint
            persistence across Colab disconnects.

    Returns:
        Path to the mirrored Drive checkpoint location.
    """
    # TODO: copy the local checkpoint tree into Drive after each save event and
    # return the destination path for logging/resume metadata.
    raise NotImplementedError


def resolve_resume_checkpoint(
    *,
    explicit_resume_path: str | Path | None,
    local_checkpoint_dir: str | Path | None = None,
    drive_checkpoint_dir: str | Path | None = None,
) -> Path | None:
    """Resolve the checkpoint path that a resumed run should load.

    Args:
        explicit_resume_path: User-provided checkpoint path, if any.
        local_checkpoint_dir: Local checkpoint directory to scan as fallback.
        drive_checkpoint_dir: Drive checkpoint directory to scan as fallback.

    Returns:
        Path to the checkpoint that should be loaded, or ``None`` if no resume
        candidate exists.
    """
    # TODO: honor an explicit resume path first, then prefer the newest Drive
    # checkpoint, then the newest local checkpoint, mirroring notebook behavior.
    raise NotImplementedError
