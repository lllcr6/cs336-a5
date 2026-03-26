from __future__ import annotations

from cs336_alignment.evaluation import log_generations


def test_log_generations_emits_explicit_reward_means(tmp_path):
    payload = log_generations(
        prompts=["P"],
        responses=["R"],
        ground_truths=["G"],
        reward_info=[
            {
                "reward": 1.0,
                "answer_reward": 1.0,
                "format_reward": 0.5,
            }
        ],
        output_dir=tmp_path,
    )

    summary = payload["summary"]
    assert summary["reward_mean"] == 1.0
    assert summary["answer_reward_mean"] == 1.0
    assert summary["format_reward_mean"] == 0.5
    assert "reward" not in summary
    assert "answer_reward" not in summary
    assert "format_reward" not in summary
    assert "mean_reward" not in summary
