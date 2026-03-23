"""Checkpointing and Google Drive sync scaffolds for Assignment 5."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _checkpoint_step(path: Path) -> int | None:
    name = path.name
    if not path.is_dir() or not name.startswith("step-"):
        return None
    try:
        return int(name.removeprefix("step-"))
    except ValueError:
        return None


def _sorted_checkpoints(checkpoint_dir: Path) -> list[Path]:
    candidates = [p for p in checkpoint_dir.iterdir() if _checkpoint_step(p) is not None]
    return sorted(
        candidates,
        key=lambda p: (
            -1 if _checkpoint_step(p) is None else _checkpoint_step(p),
            p.stat().st_mtime,
        ),
    )


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
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"step-{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
    }
    torch.save(payload, checkpoint_path / "checkpoint.pt")

    with open(checkpoint_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"step": step, "saved_at": payload["saved_at"]}, f, indent=2)

    if max_checkpoints is not None and max_checkpoints > 0:
        checkpoints = _sorted_checkpoints(checkpoint_dir)
        while len(checkpoints) > max_checkpoints:
            to_remove = checkpoints.pop(0)
            shutil.rmtree(to_remove, ignore_errors=True)

    return checkpoint_path


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a checkpoint payload from disk."""
    checkpoint_path = Path(checkpoint_path)
    payload_path = checkpoint_path
    if checkpoint_path.is_dir():
        candidate = checkpoint_path / "checkpoint.pt"
        if candidate.exists():
            payload_path = candidate
        else:
            candidate = checkpoint_path / "checkpoint.pkl"
            if candidate.exists():
                payload_path = candidate

    if payload_path.suffix == ".pt":
        return dict(torch.load(payload_path, map_location="cpu"))

    import pickle

    with open(payload_path, "rb") as f:
        return pickle.load(f)


def latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the newest checkpoint found under ``checkpoint_dir``."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = _sorted_checkpoints(checkpoint_dir)
    if checkpoints:
        return checkpoints[-1]

    candidates = [p for p in checkpoint_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
    local_checkpoint_path = Path(local_checkpoint_path)
    drive_checkpoint_root = Path(drive_checkpoint_root)
    drive_checkpoint_root.mkdir(parents=True, exist_ok=True)

    destination = drive_checkpoint_root / local_checkpoint_path.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(local_checkpoint_path, destination)
    return destination


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
    if explicit_resume_path:
        return Path(explicit_resume_path)

    if drive_checkpoint_dir is not None:
        drive_latest = latest_checkpoint(drive_checkpoint_dir)
        if drive_latest is not None:
            return drive_latest

    if local_checkpoint_dir is not None:
        return latest_checkpoint(local_checkpoint_dir)

    return None
