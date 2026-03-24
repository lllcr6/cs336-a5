"""GRPO scaffolds for Assignment 5.

This module intentionally exposes only typed placeholders and high-level
integration hooks. The actual GRPO algorithm is not implemented yet.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EvalConfig, GRPOConfig, LossType
from .tensor_ops import masked_mean
from .sft import get_response_log_probs, tokenize_prompt_and_output


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute raw rewards and group-normalized advantages for GRPO.

    Args:
        reward_fn: Callable used to score a response against a ground truth.
        rollout_responses: Rollout strings sampled from the policy.
        repeated_ground_truths: Ground truths repeated to match rollout layout.
        group_size: Number of rollouts sampled per prompt/question.
        advantage_eps: Small constant used to avoid division by zero.
        normalize_by_std: Whether to divide by the within-group standard
            deviation after subtracting the group mean.

    Returns:
        Tuple of ``(advantages, raw_rewards, metadata)``.

    Notes:
        The future implementation should call ``reward_fn`` for each rollout,
        aggregate by prompt group, compute the chosen normalization strategy,
        and expose useful logging statistics in ``metadata``.
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must match")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("rollout count must be divisible by group_size")

    reward_rows: list[dict[str, float]] = []
    raw_reward_values: list[float] = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_info = reward_fn(response, ground_truth)
        reward_rows.append(reward_info)
        raw_reward_values.append(float(reward_info.get("reward", 0.0)))

    raw_rewards = torch.tensor(raw_reward_values, dtype=torch.float32)
    advantages = torch.empty_like(raw_rewards)

    num_groups = len(raw_reward_values) // group_size
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = start + group_size
        group_rewards = raw_rewards[start:end]
        group_mean = group_rewards.mean()
        group_advantages = group_rewards - group_mean
        if normalize_by_std:
            if group_rewards.numel() > 1:
                group_std = group_rewards.std(unbiased=True)
            else:
                group_std = torch.zeros((), dtype=group_rewards.dtype)
            group_advantages = group_advantages / (group_std + advantage_eps)
        advantages[start:end] = group_advantages

    metadata: dict[str, float] = {
        "num_rollouts": float(len(raw_reward_values)),
        "num_groups": float(num_groups),
        "group_size": float(group_size),
        "reward_mean": float(raw_rewards.mean().item()) if raw_rewards.numel() else 0.0,
        "reward_std": float(raw_rewards.std(unbiased=False).item()) if raw_rewards.numel() else 0.0,
        "reward_min": float(raw_rewards.min().item()) if raw_rewards.numel() else 0.0,
        "reward_max": float(raw_rewards.max().item()) if raw_rewards.numel() else 0.0,
        "advantage_mean": float(advantages.mean().item()) if advantages.numel() else 0.0,
        "advantage_std": float(advantages.std(unbiased=False).item()) if advantages.numel() else 0.0,
    }
    if reward_rows:
        for key in sorted(reward_rows[0].keys()):
            values = torch.tensor(
                [float(row.get(key, 0.0)) for row in reward_rows],
                dtype=torch.float32,
            )
            metadata[f"{key}_mean"] = float(values.mean().item())
            metadata[f"{key}_std"] = float(values.std(unbiased=False).item())

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the unreduced naive policy-gradient loss.

    Args:
        raw_rewards_or_advantages: Per-example reward or advantage values.
        policy_log_probs: Per-token log-probabilities from the current policy.

    Returns:
        Per-token loss tensor with the same shape as ``policy_log_probs``.
    """
    weights = raw_rewards_or_advantages
    if weights.ndim == 1:
        weights = weights.unsqueeze(-1)
    weights = weights.to(
        device=policy_log_probs.device,
        dtype=policy_log_probs.dtype,
    ).expand_as(policy_log_probs)
    return -weights * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the unreduced GRPO-Clip objective and logging metadata.

    Args:
        advantages: Per-example advantages with shape ``(batch_size, 1)``.
        policy_log_probs: Per-token log-probabilities from the current policy.
        old_log_probs: Per-token log-probabilities from the rollout policy.
        cliprange: PPO-style clip parameter.

    Returns:
        Tuple of the per-token loss tensor and auxiliary metadata.
    """
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    advantages = advantages.to(
        device=policy_log_probs.device,
        dtype=policy_log_probs.dtype,
    ).expand_as(policy_log_probs)
    old_log_probs = old_log_probs.to(
        device=policy_log_probs.device,
        dtype=policy_log_probs.dtype,
    )

    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    objective = torch.minimum(unclipped_objective, clipped_objective)
    loss = -objective

    clip_mask = (ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))
    metadata = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio.std(unbiased=False).detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "clip_fraction": clip_mask.float().mean().detach(),
        "approx_kl": (old_log_probs - policy_log_probs).mean().detach(),
        "objective_mean": objective.mean().detach(),
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: LossType,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the requested policy-gradient loss variant.

    Args:
        policy_log_probs: Per-token log-probabilities from the current policy.
        loss_type: One of the supported GRPO loss modes.
        raw_rewards: Required for ``no_baseline``.
        advantages: Required for baseline and clipping-based objectives.
        old_log_probs: Required for ``grpo_clip``.
        cliprange: Required for ``grpo_clip``.

    Returns:
        Tuple of unreduced per-token loss and logging metadata.
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for no_baseline loss")
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {
            "raw_reward_mean": raw_rewards.mean().detach(),
            "raw_reward_std": raw_rewards.std(unbiased=False).detach(),
        }

    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required for reinforce_with_baseline loss")
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {
            "advantage_mean": advantages.mean().detach(),
            "advantage_std": advantages.std(unbiased=False).detach(),
        }

    if loss_type == "grpo_clip":
        if advantages is None:
            raise ValueError("advantages is required for grpo_clip loss")
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for grpo_clip loss")
        if cliprange is None:
            raise ValueError("cliprange is required for grpo_clip loss")
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        metadata["advantage_mean"] = advantages.mean().detach()
        metadata["advantage_std"] = advantages.std(unbiased=False).detach()
        return loss, metadata

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: LossType,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run a single GRPO microbatch loss/backward scaffold step.

    Args:
        policy_log_probs: Per-token log-probabilities for the microbatch.
        response_mask: Mask selecting response tokens in each sequence.
        gradient_accumulation_steps: Number of microbatches per optimizer step.
        loss_type: Selected GRPO loss mode.
        raw_rewards: Raw rewards for the no-baseline case.
        advantages: Advantages for the baseline and clipped cases.
        old_log_probs: Old policy log-probabilities for clipped GRPO.
        cliprange: Clip parameter for GRPO-Clip.

    Returns:
        Tuple of scalar loss tensor and auxiliary metadata.

    Notes:
        The implementation will eventually reduce per-token losses over the
        response mask with ``masked_mean``, scale for gradient accumulation,
        and call ``loss.backward()``.
    """
    unreduced_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = masked_mean(unreduced_loss, response_mask, dim=1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()

    metadata = dict(metadata)
    metadata.update(
        {
            "microbatch_loss": loss.detach(),
            "response_token_count": response_mask.sum().detach(),
            "masked_loss_mean": per_example_loss.mean().detach(),
        }
    )
    return loss, metadata


def should_save_checkpoint(step: int, save_every_steps: int) -> bool:
    """Return whether GRPO should persist a checkpoint at ``step``."""
    return save_every_steps > 0 and step > 0 and step % save_every_steps == 0


def should_run_evaluation(step: int, eval_every_steps: int) -> bool:
    """Return whether GRPO should trigger evaluation at ``step``."""
    return eval_every_steps > 0 and step > 0 and step % eval_every_steps == 0


def should_refresh_old_log_probs(
    step: int,
    epochs_per_rollout_batch: int,
) -> bool:
    """Return whether off-policy state should refresh cached old log-probs."""
    return epochs_per_rollout_batch > 0 and (step - 1) % epochs_per_rollout_batch == 0


def build_grpo_run_name(
    base_name: str,
    loss_type: LossType,
    rollout_batch_size: int,
    train_batch_size: int,
) -> str:
    """Build a descriptive W&B run name for a GRPO training job."""
    return (
        f"{base_name}-grpo-{loss_type}"
        f"-roll{rollout_batch_size}"
        f"-train{train_batch_size}"
    )


def resolve_grpo_output_dir(config: GRPOConfig) -> Path:
    """Resolve the canonical local checkpoint/output directory for GRPO."""
    output_dir = Path(config.checkpoint.output_dir)
    if config.wandb.run_name:
        return output_dir / config.wandb.run_name
    return output_dir


def _maybe_init_wandb(config: GRPOConfig) -> Any | None:
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


def log_grpo_metrics(
    *,
    step: int,
    metrics: dict[str, Any],
    config: GRPOConfig,
) -> None:
    """Placeholder hook for W&B metric logging during GRPO training."""
    del config
    _log_wandb_metrics(step, metrics, prefix="train/")


def _record_prompt(record: dict[str, Any]) -> str:
    prompt = record.get("prompt")
    if prompt is None:
        prompt = record.get("question")
    if prompt is None:
        raise ValueError("Each GRPO record must contain a prompt or question")
    return str(prompt)


def _record_ground_truth(record: dict[str, Any]) -> str:
    ground_truth = record.get("ground_truth")
    if ground_truth is None:
        ground_truth = record.get("answer")
    if ground_truth is None:
        ground_truth = record.get("response")
    if ground_truth is None:
        raise ValueError("Each GRPO record must contain a ground truth or answer")
    return str(ground_truth)


def _evaluate_grpo_validation(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    validation_records: list[dict[str, Any]],
    reward_fn: Callable[[str, str], dict[str, float]],
    device: torch.device,
    eval_config: EvalConfig | None,
) -> dict[str, float]:
    if not validation_records:
        return {"eval/num_examples": 0.0}

    total_rewards: list[float] = []
    reward_rows: list[dict[str, float]] = []
    answered = 0.0
    max_new_tokens = eval_config.max_tokens if eval_config is not None else 64
    temperature = eval_config.temperature if eval_config is not None else 1.0
    top_p = eval_config.top_p if eval_config is not None else 1.0

    model.eval()
    with torch.no_grad():
        for record in validation_records:
            prompt = _record_prompt(record)
            ground_truth = _record_ground_truth(record)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            reward_info = reward_fn(response, ground_truth)
            reward_rows.append(reward_info)
            reward = float(reward_info.get("reward", reward_info.get("answer_reward", 0.0)))
            total_rewards.append(reward)
            answered += 1.0 if reward > 0.0 else 0.0

    reward_tensor = torch.tensor(total_rewards, dtype=torch.float32)
    summary: dict[str, float] = {
        "eval/num_examples": float(len(validation_records)),
        "eval/reward_mean": float(reward_tensor.mean().item()),
        "eval/reward_std": float(reward_tensor.std(unbiased=False).item()) if reward_tensor.numel() > 1 else 0.0,
        "eval/reward_min": float(reward_tensor.min().item()),
        "eval/reward_max": float(reward_tensor.max().item()),
        "eval/pass_rate": answered / max(float(len(validation_records)), 1.0),
    }
    keys = sorted({key for row in reward_rows for key in row})
    for key in keys:
        values = torch.tensor([float(row.get(key, 0.0)) for row in reward_rows], dtype=torch.float32)
        summary[f"eval/{key}_mean"] = float(values.mean().item())
        summary[f"eval/{key}_std"] = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0

    model.train()
    return summary


def run_grpo_training(
    *,
    model_id: str,
    train_dataset_path: str | Path,
    validation_dataset_path: str | Path,
    reward_fn: Callable[[str, str], dict[str, float]],
    config: GRPOConfig,
    eval_config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Scaffold entrypoint for full GRPO training.

    Args:
        model_id: HuggingFace model id or local checkpoint path.
        train_dataset_path: Path to the GRPO training questions.
        validation_dataset_path: Path to the validation questions.
        reward_fn: Verified reward function used for train and eval rollouts.
        config: Shared GRPO training configuration scaffold.
        eval_config: Optional evaluation settings reused during training.

    Returns:
        Dictionary describing the eventual training outputs, metrics, and
        checkpoint metadata.

    Notes:
        The real implementation should sample rollout batches, compute rewards
        and advantages, optionally refresh old log-probs, optimize the policy,
        log to W&B, save checkpoints frequently, and mirror artifacts to Drive.
    """
    from .checkpointing import save_checkpoint, sync_checkpoint_to_drive
    from .data import read_jsonl_dataset

    output_dir = resolve_grpo_output_dir(config)
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
        }

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_records = read_jsonl_dataset(train_dataset_path)
    validation_records = read_jsonl_dataset(validation_dataset_path)
    if eval_config is not None and eval_config.num_examples is not None:
        validation_records = validation_records[: eval_config.num_examples]
    if not train_records:
        return {
            "model_id": model_id,
            "output_dir": str(output_dir),
            "checkpoint_path": None,
            "steps": 0,
            "loss": None,
            "loss_history": [],
        }

    group_size = max(1, config.group_size)
    rollout_batch_size = max(group_size, config.rollout_batch_size, config.train_batch_size, 1)
    prompt_batch_size = max(1, rollout_batch_size // group_size)
    epochs_per_rollout_batch = max(1, config.epochs_per_rollout_batch)

    def _prompt_from_record(record: dict[str, Any]) -> str:
        prompt = record.get("prompt")
        if prompt is None:
            prompt = record.get("question")
        if prompt is None:
            raise ValueError("Each GRPO record must contain a prompt or question")
        return str(prompt)

    def _ground_truth_from_record(record: dict[str, Any]) -> str:
        ground_truth = record.get("ground_truth")
        if ground_truth is None:
            ground_truth = record.get("answer")
        if ground_truth is None:
            ground_truth = record.get("response")
        if ground_truth is None:
            raise ValueError("Each GRPO record must contain a ground truth or answer")
        return str(ground_truth)

    def _generate_response(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=(eval_config.temperature if eval_config else 1.0),
                top_p=(eval_config.top_p if eval_config else 1.0),
                max_new_tokens=(eval_config.max_tokens if eval_config else 64),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )

    losses: list[float] = []
    cache: dict[str, Any] | None = None
    for step in range(1, config.train_steps + 1):
        if cache is None or should_refresh_old_log_probs(step, epochs_per_rollout_batch):
            start = ((step - 1) * prompt_batch_size) % len(train_records)
            batch_records = [
                train_records[(start + idx) % len(train_records)]
                for idx in range(prompt_batch_size)
            ]

            prompts: list[str] = []
            rollout_responses: list[str] = []
            repeated_ground_truths: list[str] = []
            for record in batch_records:
                prompt = _prompt_from_record(record)
                ground_truth = _ground_truth_from_record(record)
                for _ in range(group_size):
                    prompts.append(prompt)
                    rollout_responses.append(_generate_response(prompt))
                    repeated_ground_truths.append(ground_truth)

            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                reward_fn=reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=group_size,
                advantage_eps=config.advantage_eps,
                normalize_by_std=config.normalize_by_std,
            )

            tokenized = tokenize_prompt_and_output(prompts, rollout_responses, tokenizer)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            with torch.no_grad():
                rollout_log_probs = get_response_log_probs(
                    model=model,
                    input_ids=tokenized["input_ids"],
                    labels=tokenized["labels"],
                    return_token_entropy=False,
                )["log_probs"]

            cache = {
                "tokenized": tokenized,
                "raw_rewards": raw_rewards.unsqueeze(-1),
                "advantages": advantages.unsqueeze(-1),
                "old_log_probs": rollout_log_probs.detach(),
                "reward_metadata": reward_metadata,
            }

        optimizer.zero_grad(set_to_none=True)
        current_log_probs = get_response_log_probs(
            model=model,
            input_ids=cache["tokenized"]["input_ids"],
            labels=cache["tokenized"]["labels"],
            return_token_entropy=False,
        )["log_probs"]
        loss, loss_metadata = grpo_microbatch_train_step(
            policy_log_probs=current_log_probs,
            response_mask=cache["tokenized"]["response_mask"],
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            loss_type=config.loss_type,
            raw_rewards=cache["raw_rewards"] if config.loss_type == "no_baseline" else None,
            advantages=cache["advantages"] if config.loss_type != "no_baseline" else None,
            old_log_probs=cache["old_log_probs"] if config.loss_type == "grpo_clip" else None,
            cliprange=config.cliprange,
        )
        if config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        train_metrics = {
            "loss": loss.detach(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **cache["reward_metadata"],
            **loss_metadata,
        }
        if step % max(1, config.log_every_steps) == 0 or step == 1 or step == config.train_steps:
            log_grpo_metrics(step=step, metrics=train_metrics, config=config)

        if validation_records and (
            should_run_evaluation(step, config.eval_every_steps) or step == config.train_steps
        ):
            validation_metrics = _evaluate_grpo_validation(
                model=model,
                tokenizer=tokenizer,
                validation_records=validation_records,
                reward_fn=reward_fn,
                device=device,
                eval_config=eval_config,
            )
            _log_wandb_metrics(step, validation_metrics, prefix="eval/")

        if should_save_checkpoint(step, config.save_every_steps):
            checkpoint_path = save_checkpoint(
                output_dir,
                state={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "loss_history": losses,
                    "config": asdict(config),
                },
                step=step,
                max_checkpoints=config.checkpoint.max_checkpoints,
            )
            if config.drive_sync.enabled and config.drive_sync.drive_root is not None:
                drive_dir = (
                    Path(config.drive_sync.drive_root)
                    / config.drive_sync.per_run_subdir
                    / config.drive_sync.checkpoint_dirname
                )
                drive_dir.mkdir(parents=True, exist_ok=True)
                sync_checkpoint_to_drive(checkpoint_path, drive_dir)

    final_checkpoint = save_checkpoint(
        output_dir,
        state={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": config.train_steps,
            "loss_history": losses,
            "config": asdict(config),
        },
        step=config.train_steps,
        max_checkpoints=config.checkpoint.max_checkpoints,
    )
    if config.drive_sync.enabled and config.drive_sync.drive_root is not None:
        drive_dir = (
            Path(config.drive_sync.drive_root)
            / config.drive_sync.per_run_subdir
            / config.drive_sync.checkpoint_dirname
        )
        drive_dir.mkdir(parents=True, exist_ok=True)
        sync_checkpoint_to_drive(final_checkpoint, drive_dir)

    return {
        "model_id": model_id,
        "output_dir": str(output_dir),
        "checkpoint_path": str(final_checkpoint),
        "steps": config.train_steps,
        "loss": losses[-1] if losses else None,
        "loss_history": losses,
    }
