from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cs336_alignment.data import get_packed_sft_dataset as _get_packed_sft_dataset
from cs336_alignment.data import run_iterate_batches as _run_iterate_batches
from cs336_alignment.dpo import (
    run_compute_per_instance_dpo_loss as _run_compute_per_instance_dpo_loss,
)
from cs336_alignment.grpo import (
    compute_group_normalized_rewards as _compute_group_normalized_rewards,
    compute_grpo_clip_loss as _compute_grpo_clip_loss,
    compute_naive_policy_gradient_loss as _compute_naive_policy_gradient_loss,
    compute_policy_gradient_loss as _compute_policy_gradient_loss,
    grpo_microbatch_train_step as _grpo_microbatch_train_step,
)
from cs336_alignment.metrics import (
    run_parse_gsm8k_response as _run_parse_gsm8k_response,
    run_parse_mmlu_response as _run_parse_mmlu_response,
)
from cs336_alignment.sft import (
    compute_entropy as _compute_entropy,
    get_response_log_probs as _get_response_log_probs,
    masked_normalize as _masked_normalize,
    sft_microbatch_train_step as _sft_microbatch_train_step,
    tokenize_prompt_and_output as _tokenize_prompt_and_output,
)
from cs336_alignment.tensor_ops import masked_mean as _masked_mean


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    return _tokenize_prompt_and_output(
        prompt_strs=prompt_strs,
        output_strs=output_strs,
        tokenizer=tokenizer,
    )


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    return _compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    return _compute_entropy(logits)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    return _get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=return_token_entropy,
    )


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return _compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
    )


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def run_masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    return _masked_mean(tensor=tensor, mask=mask, dim=dim)


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _sft_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=normalize_constant,
    )


def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    return _masked_normalize(
        tensor=tensor,
        mask=mask,
        dim=dim,
        normalize_constant=normalize_constant,
    )


"""
The below adapters are used in the optional
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    return _get_packed_sft_dataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        seq_length=seq_length,
        shuffle=shuffle,
    )


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return _run_iterate_batches(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    return _run_parse_mmlu_response(
        mmlu_example=mmlu_example,
        model_output=model_output,
    )


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    return _run_parse_gsm8k_response(model_output=model_output)


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    return _run_compute_per_instance_dpo_loss(
        lm=lm,
        lm_ref=lm_ref,
        tokenizer=tokenizer,
        beta=beta,
        prompt=prompt,
        response_chosen=response_chosen,
        response_rejected=response_rejected,
    )
