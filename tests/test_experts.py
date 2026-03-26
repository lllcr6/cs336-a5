from __future__ import annotations

import json
from pathlib import Path

import cs336_alignment.experts as experts_module
from cs336_alignment.config import ExpertIterationConfig, WandbConfig


class _FakeVllmContext:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_expert_iteration_runs_sft_update_on_filtered_dataset(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        train_path,
        [
            {"prompt": "P1", "answer": "A1"},
            {"prompt": "P2", "answer": "A2"},
        ],
    )
    _write_jsonl(
        validation_path,
        [
            {"prompt": "VP1", "answer": "VA1"},
        ],
    )

    train_results = {
        "prompts": ["P1", "P2"],
        "responses": ["R1", "R2"],
        "ground_truths": ["A1", "A2"],
        "reward_info": [
            {"reward": 1.0, "answer_reward": 1.0, "format_reward": 1.0},
            {"reward": 0.0, "answer_reward": 0.0, "format_reward": 0.0},
        ],
        "summary": {
            "answer_reward_mean": 0.5,
            "format_reward_mean": 0.5,
            "reward_mean": 0.5,
        },
    }
    validation_results = {
        "prompts": ["VP1"],
        "responses": ["VR1"],
        "ground_truths": ["VA1"],
        "reward_info": [{"reward": 1.0, "answer_reward": 1.0, "format_reward": 1.0}],
        "summary": {
            "answer_reward_mean": 1.0,
            "format_reward_mean": 1.0,
            "reward_mean": 1.0,
        },
    }

    captured = {}

    def fake_evaluate_vllm(llm, reward_fn, prompts, sampling_params, output_path=None):
        del llm, reward_fn, sampling_params, output_path
        if len(prompts) == 2:
            return train_results
        return validation_results

    def fake_run_sft_training(*, model_id, dataset_path, validation_dataset_path, config, eval_config=None, reward_fn=None):
        del model_id, validation_dataset_path, config, eval_config, reward_fn
        with open(dataset_path, "r", encoding="utf-8") as f:
            selected_records = [json.loads(line) for line in f if line.strip()]
        captured["selected_records"] = selected_records
        return {
            "steps": 2,
            "loss": 0.123,
            "loss_history": [0.456, 0.123],
            "last_microbatch": {"token_entropy": 2.5},
        }

    monkeypatch.setattr(experts_module, "init_vllm", lambda *args, **kwargs: _FakeVllmContext())
    monkeypatch.setattr(experts_module, "evaluate_vllm", fake_evaluate_vllm)
    monkeypatch.setattr(experts_module, "run_sft_training", fake_run_sft_training)
    monkeypatch.setattr(experts_module, "log_generations", lambda **kwargs: kwargs)
    monkeypatch.setattr(experts_module, "build_sampling_params", lambda **kwargs: object())

    result = experts_module.run_expert_iteration(
        model_id="fake-model",
        train_dataset_path=train_path,
        validation_dataset_path=validation_path,
        reward_fn=lambda response, ground_truth: {
            "reward": float(response == ground_truth),
            "answer_reward": float(response == ground_truth),
            "format_reward": float(response == ground_truth),
        },
        config=ExpertIterationConfig(
            n_ei_steps=1,
            rollout_batch_size=2,
            rollouts_per_example=1,
            sft_epochs_per_round=2,
            wandb=WandbConfig(run_name="ei-test"),
        ),
    )

    assert captured["selected_records"] == [{"prompt": "P1", "response": "R1"}]
    assert result["rounds"][0]["filter_keep_rate"] == 0.5
    assert result["rounds"][0]["sft_update"]["loss"] == 0.123
    assert result["rounds"][0]["sft_update"]["steps"] == 2
