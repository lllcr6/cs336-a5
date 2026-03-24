import torch

import cs336_alignment.grpo as grpo_module
from cs336_alignment.config import GRPOConfig

from .adapters import (
    run_compute_group_normalized_rewards as compute_group_normalized_rewards,
    run_compute_grpo_clip_loss as compute_grpo_clip_loss,
    run_compute_naive_policy_gradient_loss as compute_naive_policy_gradient_loss,
    run_compute_policy_gradient_loss as compute_policy_gradient_loss,
    run_grpo_microbatch_train_step as grpo_microbatch_train_step,
    run_masked_mean as masked_mean,
)


def test_compute_group_normalized_rewards_normalize_by_std(
    numpy_snapshot,
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    advantage_eps,
    group_size,
):
    normalized_rewards, raw_rewards, metadata = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=True,
    )
    output = {
        "normalized_rewards": normalized_rewards,
        "raw_rewards": raw_rewards,
    }
    numpy_snapshot.assert_match(output)


def test_compute_group_normalized_rewards_no_normalize_by_std(
    numpy_snapshot,
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    advantage_eps,
    group_size,
):
    normalized_rewards, raw_rewards, metadata = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=False,
    )
    output = {
        "normalized_rewards": normalized_rewards,
        "raw_rewards": raw_rewards,
    }
    numpy_snapshot.assert_match(output)


def test_compute_group_normalized_rewards_metadata_includes_reward_summaries(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    advantage_eps,
    group_size,
):
    _, _, metadata = compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=True,
    )

    assert {"reward_mean", "reward", "answer_reward", "format_reward"} <= metadata.keys()
    assert metadata["reward"] == metadata["reward_mean"]
    assert metadata["answer_reward"] == metadata["answer_reward_mean"]
    assert metadata["format_reward"] == metadata["format_reward_mean"]


def test_compute_naive_policy_gradient_loss(
    numpy_snapshot,
    raw_rewards_or_advantages,
    policy_log_probs,
):
    output = compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
    )
    numpy_snapshot.assert_match(output)


def test_compute_grpo_clip_loss_large_cliprange(
    numpy_snapshot,
    advantages,
    policy_log_probs,
    old_log_probs,
):
    output, _ = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=10.0,
    )
    numpy_snapshot.assert_match(output)


def test_compute_grpo_clip_loss_small_cliprange(
    numpy_snapshot,
    advantages,
    policy_log_probs,
    old_log_probs,
):
    output, _ = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=0.1,
    )
    numpy_snapshot.assert_match(output)


def test_compute_grpo_clip_loss_metadata_includes_clip_statistics(
    advantages,
    policy_log_probs,
    old_log_probs,
):
    _, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=0.1,
    )

    assert "clip_fraction" in metadata
    assert "approx_kl" in metadata


def test_compute_policy_gradient_loss_no_baseline(
    numpy_snapshot,
    policy_log_probs,
    raw_rewards,
    advantages,
    old_log_probs,
):
    output, _ = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type="no_baseline",
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=0.5,
    )
    numpy_snapshot.assert_match(output)


def test_compute_policy_gradient_loss_reinforce_with_baseline(
    numpy_snapshot,
    policy_log_probs,
    raw_rewards,
    advantages,
    old_log_probs,
):
    output, _ = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type="reinforce_with_baseline",
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=0.5,
    )
    numpy_snapshot.assert_match(output)


def test_compute_policy_gradient_loss_grpo_clip(
    numpy_snapshot,
    policy_log_probs,
    raw_rewards,
    advantages,
    old_log_probs,
):
    output, _ = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type="grpo_clip",
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=0.5,
    )
    numpy_snapshot.assert_match(output)


def test_masked_mean_dim0(numpy_snapshot, tensor, mask):
    output = masked_mean(
        tensor=tensor,
        mask=mask,
        dim=0,
    )
    numpy_snapshot.assert_match(output)


def test_masked_mean_dim1(numpy_snapshot, tensor, mask):
    output = masked_mean(
        tensor=tensor,
        mask=mask,
        dim=1,
    )
    numpy_snapshot.assert_match(output)


def test_masked_mean_dimlast(numpy_snapshot, tensor, mask):
    output = masked_mean(
        tensor=tensor,
        mask=mask,
        dim=-1,
    )
    numpy_snapshot.assert_match(output)


def test_masked_mean_dimNone(numpy_snapshot, tensor, mask):
    output = masked_mean(
        tensor=tensor,
        mask=mask,
    )
    numpy_snapshot.assert_match(output)


def test_grpo_microbatch_train_step_grpo_clip(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange,
):
    policy_log_probs.requires_grad = True

    loss, _ = grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type="grpo_clip",
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    output = {"loss": loss, "policy_log_probs_grad": policy_log_probs.grad}
    numpy_snapshot.assert_match(output)


def test_grpo_microbatch_train_step_grpo_clip_10_steps(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange,
):
    policy_log_probs.requires_grad = True

    loss_list = []
    grad_list = []
    for _ in range(10):
        loss, _ = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            loss_type="grpo_clip",
            raw_rewards=raw_rewards,
            advantages=advantages,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        loss_list.append(loss)
        grad_list.append(policy_log_probs.grad)

    output = {
        "loss": torch.stack(loss_list),
        "policy_log_probs_grad": torch.stack(grad_list),
    }
    numpy_snapshot.assert_match(output)


def test_log_grpo_metrics_keeps_reward_and_diagnostics(monkeypatch):
    captured = {}

    def fake_log_wandb_metrics(step, metrics, prefix="train/"):
        captured["step"] = step
        captured["metrics"] = metrics
        captured["prefix"] = prefix

    monkeypatch.setattr(grpo_module, "_log_wandb_metrics", fake_log_wandb_metrics)

    grpo_module.log_grpo_metrics(
        step=7,
        metrics={
            "loss": torch.tensor(3.0),
            "microbatch_loss": torch.tensor(2.0),
            "masked_loss_mean": torch.tensor(1.0),
            "reward_mean": torch.tensor(0.6),
            "answer_reward": torch.tensor(0.8),
            "format_reward": torch.tensor(0.9),
            "token_entropy": torch.tensor(1.2),
            "response_length": torch.tensor(42.0),
            "grad_norm": torch.tensor(0.3),
            "clip_fraction": torch.tensor(0.4),
            "approx_kl": torch.tensor(0.05),
        },
        config=GRPOConfig(),
    )

    assert captured["step"] == 7
    assert captured["prefix"] == "train/"
    assert "loss" not in captured["metrics"]
    assert "microbatch_loss" not in captured["metrics"]
    assert "masked_loss_mean" not in captured["metrics"]
    for key in [
        "reward_mean",
        "answer_reward",
        "format_reward",
        "token_entropy",
        "response_length",
        "grad_norm",
        "clip_fraction",
        "approx_kl",
    ]:
        assert key in captured["metrics"]
