"""SFT scaffolds for Assignment 5.

This module intentionally defines only typed function stubs and configuration
integration points. The concrete algorithmic implementation is left for later.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define an eos_token_id")

    prompt_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs
    ]
    output_token_ids = [
        tokenizer.encode(output, add_special_tokens=False) for output in output_strs
    ]

    full_sequences: list[list[int]] = []
    response_ranges: list[tuple[int, int]] = []
    max_len = 0
    for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids):
        full_ids = [*prompt_ids, *output_ids, eos_token_id]
        full_sequences.append(full_ids)
        response_start = max(len(prompt_ids) - 1, 0)
        response_end = len(prompt_ids) + len(output_ids) - 1
        response_ranges.append((response_start, response_end))
        max_len = max(max_len, len(full_ids))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    for full_ids, (response_start, response_end) in zip(full_sequences, response_ranges):
        padded = full_ids + [pad_token_id] * (max_len - len(full_ids))
        input_ids = padded[:-1]
        labels = padded[1:]
        mask = [False] * (max_len - 1)
        for idx in range(response_start, response_end):
            if 0 <= idx < len(mask):
                mask[idx] = True
        input_rows.append(input_ids)
        label_rows.append(labels)
        mask_rows.append(mask)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "response_mask": torch.tensor(mask_rows, dtype=torch.bool),
    }


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
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


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
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result: dict[str, torch.Tensor] = {"log_probs": token_log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


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
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / normalize_constant


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
    per_example_loss = masked_normalize(
        -policy_log_probs,
        response_mask,
        dim=1,
        normalize_constant=normalize_constant,
    )
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    metadata = {
        "microbatch_loss": loss.detach(),
        "response_token_count": response_mask.sum().detach(),
        "response_log_prob_sum": masked_normalize(
            policy_log_probs,
            response_mask,
            dim=1,
            normalize_constant=1.0,
        ).detach(),
    }
    return loss, metadata


def should_save_checkpoint(step: int, save_every_steps: int) -> bool:
    """Return whether SFT should persist a checkpoint at ``step``."""
    return save_every_steps > 0 and step > 0 and step % save_every_steps == 0


def should_run_evaluation(step: int, eval_every_steps: int) -> bool:
    """Return whether SFT should trigger evaluation at ``step``."""
    return eval_every_steps > 0 and step > 0 and step % eval_every_steps == 0


def build_sft_run_name(base_name: str, train_steps: int) -> str:
    """Build a descriptive W&B run name for an SFT training job."""
    return f"{base_name}-sft-{train_steps}steps"


def resolve_sft_output_dir(config: SFTConfig) -> Path:
    """Resolve the canonical local checkpoint/output directory for SFT."""
    output_dir = Path(config.checkpoint.output_dir)
    if config.wandb.run_name:
        return output_dir / config.wandb.run_name
    return output_dir


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
    from .checkpointing import save_checkpoint, sync_checkpoint_to_drive
    from .data import read_jsonl_dataset

    del validation_dataset_path, eval_config

    output_dir = resolve_sft_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.train_steps <= 0:
        return {
            "model_id": model_id,
            "output_dir": str(output_dir),
            "checkpoint_path": None,
            "steps": 0,
            "loss": None,
            "loss_history": [],
            "last_microbatch": None,
        }

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    records = read_jsonl_dataset(dataset_path)
    if not records:
        return {
            "model_id": model_id,
            "output_dir": str(output_dir),
            "checkpoint_path": None,
            "steps": 0,
            "loss": None,
            "loss_history": [],
            "last_microbatch": None,
        }

    def _record_prompt(record: dict[str, Any]) -> str:
        prompt = record.get("prompt")
        if prompt is None:
            prompt = record.get("instruction")
        if prompt is None:
            prompt = record.get("question")
        if prompt is None:
            raise ValueError("Each SFT record must contain a prompt/instruction/question")
        return str(prompt)

    def _record_response(record: dict[str, Any]) -> str:
        response = record.get("response")
        if response is None:
            response = record.get("output")
        if response is None:
            response = record.get("answer")
        if response is None:
            raise ValueError("Each SFT record must contain a response/output/answer")
        return str(response)

    drive_dir = None
    if config.drive_sync.enabled and config.drive_sync.drive_root is not None:
        drive_dir = (
            Path(config.drive_sync.drive_root)
            / config.drive_sync.per_run_subdir
            / config.drive_sync.checkpoint_dirname
        )
        drive_dir.mkdir(parents=True, exist_ok=True)

    losses: list[float] = []
    step = 0
    while step < config.train_steps:
        epoch_records = list(records)
        random.shuffle(epoch_records)
        for start in range(0, len(epoch_records), config.train_batch_size):
            if step >= config.train_steps:
                break
            batch_records = epoch_records[start : start + config.train_batch_size]
            prompt_strs = [_record_prompt(record) for record in batch_records]
            output_strs = [_record_response(record) for record in batch_records]
            batch = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            outputs = get_response_log_probs(
                model=model,
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                return_token_entropy=False,
            )
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=outputs["log_probs"],
                response_mask=batch["response_mask"],
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                normalize_constant=config.normalize_constant,
            )
            if config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1

            if should_save_checkpoint(step, config.save_every_steps):
                checkpoint_path = save_checkpoint(
                    output_dir,
                    state={
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "losses": losses,
                        "config": asdict(config),
                    },
                    step=step,
                    max_checkpoints=config.checkpoint.max_checkpoints,
                )
                if drive_dir is not None:
                    sync_checkpoint_to_drive(checkpoint_path, drive_dir)

    final_checkpoint = save_checkpoint(
        output_dir,
        state={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "losses": losses,
            "config": asdict(config),
        },
        step=step,
        max_checkpoints=config.checkpoint.max_checkpoints,
    )
    if drive_dir is not None:
        sync_checkpoint_to_drive(final_checkpoint, drive_dir)

    return {
        "model_id": model_id,
        "output_dir": str(output_dir),
        "checkpoint_path": str(final_checkpoint),
        "steps": step,
        "loss": losses[-1] if losses else None,
        "loss_history": losses,
        "last_microbatch": metadata,
    }
