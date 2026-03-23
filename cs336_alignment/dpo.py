from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizerBase


def run_compute_per_instance_dpo_loss(
    lm,
    lm_ref,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Any:
    """Compute the per-instance DPO loss for a single preference pair."""
    # TODO: tokenize prompt/continuations, score both models, and form the DPO loss.
    raise NotImplementedError
