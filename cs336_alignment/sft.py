"""SFT scaffolds for Assignment 5.

This module intentionally defines only typed function stubs and configuration
integration points. The concrete algorithmic implementation is left for later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from .config import EvalConfig, SFTConfig


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompts and responses into model-ready tensors.

    Args:
        prompt_strs: Prompt strings to be shown to the policy.
        output_strs: Target responses paired one-to-one with ``prompt_strs``.
        tokenizer: Tokenizer used to encode the prompt and response segments.

    Returns:
        Dictionary containing ``input_ids``, ``labels``, and ``response_mask``.

    Notes:
        The final implementation should tokenize prompts and outputs separately,
        concatenate them, shift labels by one token, and mark only the response
        positions in ``response_mask``.
    """
    # TODO: tokenize prompt and response segments, pad them into a batch, and
    # build the shifted labels plus response-only mask expected by SFT and RL.
    raise NotImplementedError


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute next-token entropies from model logits.

    Args:
        logits: Tensor of shape ``(batch_size, sequence_length, vocab_size)``.

    Returns:
        Tensor of shape ``(batch_size, sequence_length)`` containing per-token
        predictive entropy values.

    Notes:
        The eventual implementation should use a numerically stable formulation
        based on ``logsumexp`` or stable log-softmax operations.
    """
    # TODO: compute entropy over the vocabulary dimension in a numerically
    # stable way so it can be logged during SFT, EI, and GRPO.
    raise NotImplementedError


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Score a batch of prompt-response sequences with a causal LM.

    Args:
        model: Policy or reference model used to score token log-probabilities.
        input_ids: Prompt-response token ids excluding the final token.
        labels: Shifted token ids excluding the first token.
        return_token_entropy: Whether to also return per-token entropy values.

    Returns:
        Dictionary containing ``log_probs`` and optionally ``token_entropy``.

    Notes:
        The implementation will eventually run a forward pass, compute token
        log-probabilities aligned with ``labels``, and optionally call
        ``compute_entropy`` on the model logits.
    """
    # TODO: run the model, score each label token under the conditional next
    # token distribution, and optionally attach token entropy statistics.
    raise NotImplementedError


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Reduce a tensor over masked elements and divide by a constant.

    Args:
        tensor: Tensor containing values to reduce.
        mask: Boolean or 0/1 mask selecting which elements contribute.
        dim: Optional dimension to reduce over. If ``None``, reduce globally.
        normalize_constant: Scalar constant used for post-sum normalization.

    Returns:
        Tensor matching ``torch.sum`` shape semantics for the chosen reduction.

    Notes:
        This helper is used both for SFT loss aggregation and later GRPO
        length-normalization experiments.
    """
    # TODO: zero out masked elements, sum over the selected dimension, and
    # divide by ``normalize_constant`` without changing the expected shapes.
    raise NotImplementedError


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run a single SFT microbatch loss/backward scaffold step.

    Args:
        policy_log_probs: Per-token log-probabilities from the policy.
        response_mask: Mask that selects response tokens only.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        normalize_constant: Optional normalization constant for token reduction.

    Returns:
        Tuple of scalar loss tensor and auxiliary metadata.

    Notes:
        The eventual implementation should negate the response-token
        log-probabilities, reduce with ``masked_normalize``, scale for gradient
        accumulation, and call ``loss.backward()``.
    """
    # TODO: compute the SFT objective for the microbatch, backpropagate it, and
    # return both the loss tensor and any useful logging metadata.
    raise NotImplementedError


def should_save_checkpoint(step: int, save_every_steps: int) -> bool:
    """Return whether SFT should persist a checkpoint at ``step``."""
    # TODO: implement the shared step-based save cadence for SFT training.
    raise NotImplementedError


def should_run_evaluation(step: int, eval_every_steps: int) -> bool:
    """Return whether SFT should trigger evaluation at ``step``."""
    # TODO: implement the shared step-based evaluation cadence for SFT runs.
    raise NotImplementedError


def build_sft_run_name(base_name: str, train_steps: int) -> str:
    """Build a descriptive W&B run name for an SFT training job."""
    # TODO: fold the key SFT hyperparameters into a stable run naming scheme.
    raise NotImplementedError


def resolve_sft_output_dir(config: SFTConfig) -> Path:
    """Resolve the canonical local checkpoint/output directory for SFT."""
    # TODO: combine checkpoint settings and run metadata into the effective
    # output directory used by local checkpoints and Drive mirroring.
    raise NotImplementedError


def run_sft_training(
    *,
    model_id: str,
    dataset_path: str | Path,
    validation_dataset_path: str | Path | None,
    config: SFTConfig,
    eval_config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Scaffold entrypoint for full SFT training.

    Args:
        model_id: HuggingFace model id or local checkpoint path.
        dataset_path: Path to the SFT jsonl training dataset.
        validation_dataset_path: Optional validation set used for periodic eval.
        config: Shared SFT training configuration scaffold.
        eval_config: Optional evaluation settings reused during training.

    Returns:
        Dictionary describing the eventual training outputs, metrics, and
        checkpoint metadata.

    Notes:
        The eventual implementation should load the model/tokenizer, iterate
        through microbatches, log to W&B, save checkpoints frequently, and
        mirror artifacts into Drive when configured.
    """
    # TODO: load the policy, dataset, optimizer, and logging/checkpoint hooks;
    # then execute the SFT training loop described in the assignment handout.
    raise NotImplementedError
