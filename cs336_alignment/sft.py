"""SFT scaffolds for Assignment 5.

This module intentionally defines only typed function stubs and configuration
integration points. The concrete algorithmic implementation is left for later.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Callable
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizerBase

from .config import EvalConfig, SFTConfig
from .evaluation import log_generations
from .tensor_ops import masked_mean


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


def _aggregate_masked_metric(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Average a token-level metric over response tokens only."""
    return masked_mean(values, response_mask, dim=1)


def _split_batch_into_microbatches(
    batch_size: int,
    gradient_accumulation_steps: int,
) -> list[slice]:
    """Split a logical batch into up to ``gradient_accumulation_steps`` slices."""
    if batch_size <= 0:
        return []

    microbatch_count = max(1, min(batch_size, gradient_accumulation_steps))
    microbatch_size = math.ceil(batch_size / microbatch_count)
    return [
        slice(start, min(start + microbatch_size, batch_size))
        for start in range(0, batch_size, microbatch_size)
    ]


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


def _maybe_init_wandb(config: SFTConfig) -> Any | None:
    try:
        import wandb  # type: ignore
    except Exception:
        return None

    if getattr(wandb, "run", None) is not None:
        return wandb

    init_kwargs: dict[str, Any] = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "name": config.wandb.run_name,
        "tags": config.wandb.tags,
    }
    if config.wandb.log_dir is not None:
        init_kwargs["dir"] = str(config.wandb.log_dir)

    try:
        wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
    except Exception:
        return None
    return wandb


def _log_wandb_metrics(step: int, metrics: dict[str, Any], *, prefix: str = "train/") -> None:
    try:
        import wandb  # type: ignore
    except Exception:
        return

    if getattr(wandb, "run", None) is None:
        return

    payload: dict[str, Any] = {"step": step}
    for key, value in metrics.items():
        metric_name = key if key.startswith(("train/", "eval/", "val/", "step")) else f"{prefix}{key}"
        if isinstance(value, torch.Tensor):
            detached = value.detach()
            if detached.dtype.is_floating_point or detached.dtype.is_complex:
                payload[metric_name] = float(detached.mean().cpu().item())
            else:
                payload[metric_name] = int(detached.sum().cpu().item())
        elif isinstance(value, (int, float)):
            payload[metric_name] = float(value)
        else:
            payload[metric_name] = value

    wandb.log(payload, step=step)


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


def _record_ground_truth(record: dict[str, Any]) -> str:
    ground_truth = record.get("ground_truth")
    if ground_truth is None:
        ground_truth = record.get("answer")
    if ground_truth is None:
        ground_truth = record.get("response")
    if ground_truth is None:
        raise ValueError("Each SFT record must contain a ground truth/answer")
    return str(ground_truth)


def _evaluate_sft_validation(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    validation_records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    normalize_constant: float,
) -> dict[str, float]:
    if not validation_records:
        return {"eval/num_examples": 0.0}

    model.eval()
    total_loss = 0.0
    total_token_count = 0.0
    total_token_entropy = 0.0
    total_examples = 0.0
    num_batches = 0.0
    with torch.no_grad():
        for start in range(0, len(validation_records), max(1, batch_size)):
            batch_records = validation_records[start : start + max(1, batch_size)]
            prompt_strs = [_record_prompt(record) for record in batch_records]
            output_strs = [_record_response(record) for record in batch_records]
            batch = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = get_response_log_probs(
                model=model,
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                return_token_entropy=True,
            )
            token_log_probs = outputs["log_probs"]
            token_entropy = outputs["token_entropy"]
            response_mask = batch["response_mask"]
            token_count = float(response_mask.sum().item())
            batch_loss = float((-(token_log_probs * response_mask).sum().item()) / max(normalize_constant, 1e-12))
            batch_entropy = float(_aggregate_masked_metric(token_entropy, response_mask).mean().item())
            total_loss += batch_loss
            total_token_count += token_count
            total_token_entropy += batch_entropy
            total_examples += float(len(batch_records))
            num_batches += 1.0

    avg_loss = total_loss / max(num_batches, 1.0)
    avg_token_loss = total_loss / max(total_token_count, 1.0)
    avg_token_entropy = total_token_entropy / max(num_batches, 1.0)
    perplexity = math.exp(avg_token_loss) if avg_token_loss < 50 else float("inf")

    model.train()
    return {
        "eval/num_examples": total_examples,
        "eval/num_batches": num_batches,
        "eval/loss": avg_loss,
        "eval/token_loss": avg_token_loss,
        "eval/token_entropy": avg_token_entropy,
        "eval/perplexity": perplexity,
        "eval/response_token_count": total_token_count,
    }


def _evaluate_sft_generation_validation(
    *,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    validation_records: list[dict[str, Any]],
    device: torch.device,
    reward_fn: Callable[[str, str], dict[str, float]] | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    eval_output_dir: Path | None = None,
) -> dict[str, float]:
    if not validation_records:
        return {"eval/num_examples": 0.0}
    if reward_fn is None:
        return {"eval/num_examples": float(len(validation_records))}

    prompts: list[str] = []
    responses: list[str] = []
    ground_truths: list[str] = []
    reward_info: list[dict[str, float]] = []
    response_lengths: list[float] = []
    token_entropies: list[float] = []

    model.eval()
    with torch.no_grad():
        for record in validation_records:
            prompt = _record_prompt(record)
            ground_truth = _record_ground_truth(record)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            do_sample = temperature > 0.0
            generate_kwargs: dict[str, Any] = {
                "do_sample": do_sample,
                "top_p": top_p,
                "max_new_tokens": max_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
            generated = model.generate(
                **inputs,
                **generate_kwargs,
            )
            response = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            tokenized = tokenize_prompt_and_output([prompt], [response], tokenizer)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            outputs = get_response_log_probs(
                model=model,
                input_ids=tokenized["input_ids"],
                labels=tokenized["labels"],
                return_token_entropy=True,
            )

            prompts.append(prompt)
            responses.append(response)
            ground_truths.append(ground_truth)
            reward_info.append(reward_fn(response, ground_truth))
            response_lengths.append(float(generated[0].shape[-1] - inputs["input_ids"].shape[-1]))
            token_entropies.append(
                float(_aggregate_masked_metric(outputs["token_entropy"], tokenized["response_mask"]).mean().item())
            )

    summary: dict[str, float] = {
        "eval/num_examples": float(len(validation_records)),
        "eval/response_length": float(sum(response_lengths) / max(len(response_lengths), 1)),
        "eval/token_entropy": float(sum(token_entropies) / max(len(token_entropies), 1)),
    }
    keys = sorted({key for row in reward_info for key in row})
    for key in keys:
        values = torch.tensor([float(row.get(key, 0.0)) for row in reward_info], dtype=torch.float32)
        summary[f"eval/{key}"] = float(values.mean().item())
        summary[f"eval/{key}_mean"] = float(values.mean().item())
    if "answer_reward" in keys:
        summary["eval/reward_mean"] = summary["eval/answer_reward"]
    elif "reward" in keys:
        summary["eval/reward_mean"] = summary["eval/reward"]

    if eval_output_dir is not None:
        log_generations(
            prompts=prompts,
            responses=responses,
            ground_truths=ground_truths,
            reward_info=reward_info,
            output_dir=eval_output_dir,
        )

    model.train()
    return summary


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
    reward_fn: Callable[[str, str], dict[str, float]] | None = None,
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

    output_dir = resolve_sft_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _maybe_init_wandb(config)
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
    validation_records = read_jsonl_dataset(validation_dataset_path) if validation_dataset_path else []
    if eval_config is not None and eval_config.num_examples is not None:
        validation_records = validation_records[: eval_config.num_examples]
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

            microbatch_slices = _split_batch_into_microbatches(
                batch["input_ids"].shape[0],
                config.gradient_accumulation_steps,
            )
            accumulation_steps = len(microbatch_slices)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            total_response_tokens = torch.zeros((), device=device)
            total_response_log_prob_sum = torch.zeros((), device=device)
            total_entropy = torch.zeros((), device=device)
            total_examples = 0
            for microbatch_slice in microbatch_slices:
                microbatch = {
                    key: value[microbatch_slice]
                    for key, value in batch.items()
                }
                outputs = get_response_log_probs(
                    model=model,
                    input_ids=microbatch["input_ids"],
                    labels=microbatch["labels"],
                    return_token_entropy=True,
                )
                microbatch_loss, metadata = sft_microbatch_train_step(
                    policy_log_probs=outputs["log_probs"],
                    response_mask=microbatch["response_mask"],
                    gradient_accumulation_steps=accumulation_steps,
                    normalize_constant=config.normalize_constant,
                )
                microbatch_examples = microbatch["input_ids"].shape[0]
                microbatch_entropy = _aggregate_masked_metric(
                    outputs["token_entropy"], microbatch["response_mask"]
                ).mean()
                loss = loss + microbatch_loss.detach()
                total_response_tokens = total_response_tokens + metadata["response_token_count"]
                total_response_log_prob_sum = (
                    total_response_log_prob_sum + metadata["response_log_prob_sum"]
                )
                total_entropy = total_entropy + (
                    microbatch_entropy.detach() * microbatch_examples
                )
                total_examples += microbatch_examples

            train_token_entropy = total_entropy / max(total_examples, 1)
            if config.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip_grad_norm
                )
            else:
                grads = [
                    parameter.grad.detach().norm(2)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                grad_norm = torch.linalg.vector_norm(torch.stack(grads), ord=2) if grads else torch.tensor(0.0, device=device)
            metadata = {
                "microbatch_loss": loss.detach(),
                "response_token_count": total_response_tokens.detach(),
                "response_log_prob_sum": total_response_log_prob_sum.detach(),
                "token_entropy": train_token_entropy.detach(),
                "grad_norm": grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else torch.tensor(float(grad_norm), device=device),
            }
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1

            train_metrics = {
                "loss": loss.detach(),
                "microbatch_loss": metadata["microbatch_loss"],
                "response_token_count": metadata["response_token_count"],
                "response_log_prob_sum": metadata["response_log_prob_sum"],
                "token_loss": (
                    -metadata["response_log_prob_sum"]
                    / metadata["response_token_count"].clamp_min(1)
                ),
                "perplexity": torch.exp(
                    (-metadata["response_log_prob_sum"])
                    / metadata["response_token_count"].clamp_min(1)
                ),
                "token_entropy": train_token_entropy.detach(),
                "grad_norm": grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if step % max(1, config.log_every_steps) == 0 or step == 1 or step == config.train_steps:
                _log_wandb_metrics(step, train_metrics, prefix="train/")

            if validation_records and (
                should_run_evaluation(step, config.eval_every_steps) or step == config.train_steps
            ):
                validation_metrics = _evaluate_sft_validation(
                    model=model,
                    tokenizer=tokenizer,
                    validation_records=validation_records,
                    device=device,
                    batch_size=config.train_batch_size,
                    normalize_constant=config.normalize_constant,
                )
                if reward_fn is not None:
                    validation_output_dir = None
                    if eval_config is not None and eval_config.output_dir is not None:
                        validation_output_dir = Path(eval_config.output_dir) / f"step-{step}"
                    validation_metrics.update(
                        _evaluate_sft_generation_validation(
                            model=model,
                            tokenizer=tokenizer,
                            validation_records=validation_records,
                            device=device,
                            reward_fn=reward_fn,
                            temperature=eval_config.temperature if eval_config is not None else 0.0,
                            top_p=eval_config.top_p if eval_config is not None else 1.0,
                            max_tokens=eval_config.max_tokens if eval_config is not None else 64,
                            eval_output_dir=validation_output_dir,
                        )
                    )
                _log_wandb_metrics(step, validation_metrics, prefix="eval/")

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
