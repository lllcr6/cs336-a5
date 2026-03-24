from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from .config import CheckpointConfig, DriveSyncConfig, EvalConfig, ExpertIterationConfig, SFTConfig
from .evaluation import evaluate_vllm, log_generations
from .vllm_utils import build_sampling_params, init_vllm
from .sft import run_sft_training


def _load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_prompt_and_ground_truth(example: dict[str, Any]) -> tuple[str, str]:
    prompt = example.get("prompt", example.get("question"))
    if prompt is None:
        raise KeyError("Example is missing a prompt/question field")

    ground_truth = example.get("ground_truth", example.get("answer"))
    if ground_truth is None:
        ground_truth = example.get("response", "")

    return str(prompt), str(ground_truth)


def run_expert_iteration(
    *,
    model_id: str,
    train_dataset_path: str | Path,
    validation_dataset_path: str | Path,
    reward_fn: Callable[[str, str], dict[str, float]],
    config: ExpertIterationConfig,
) -> dict[str, Any]:
    """Scaffold entrypoint for expert iteration.

    Args:
        model_id: Base model identifier.
        train_dataset_path: Path to the training questions.
        validation_dataset_path: Path to the validation set.
        reward_fn: Reward function used to filter sampled outputs.
        config: Expert iteration configuration.

    Returns:
        Dictionary containing training state and evaluation summaries.
    """
    train_examples = _load_jsonl_records(train_dataset_path)
    validation_examples = _load_jsonl_records(validation_dataset_path)

    if config.n_ei_steps < 0:
        raise ValueError("n_ei_steps must be non-negative")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampling_params = build_sampling_params(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop=None,
    )

    accumulated_dataset: list[dict[str, str]] = []
    accumulated_ground_truths: list[str] = []
    accumulated_reward_info: list[dict[str, float]] = []
    rounds: list[dict[str, Any]] = []

    if config.n_ei_steps == 0:
        config_n_rounds = 1
    else:
        config_n_rounds = config.n_ei_steps

    with init_vllm(model_id, device=device, seed=0) as llm:
        for round_idx in range(config_n_rounds):
            train_prompts_and_truths = [
                _extract_prompt_and_ground_truth(example) for example in train_examples
            ]
            validation_prompts_and_truths = [
                _extract_prompt_and_ground_truth(example)
                for example in validation_examples
            ]

            train_results = evaluate_vllm(
                llm,
                reward_fn,
                train_prompts_and_truths,
                sampling_params,
            )
            validation_results = evaluate_vllm(
                llm,
                reward_fn,
                validation_prompts_and_truths,
                sampling_params,
            )

            selected = build_expert_iteration_dataset(
                prompts=train_results["prompts"],
                responses=train_results["responses"],
                rewards=train_results["reward_info"],
            )
            keep_rate = len(selected) / max(len(train_results["responses"]), 1)
            for ground_truth, reward_info in zip(
                train_results["ground_truths"], train_results["reward_info"]
            ):
                reward = float(
                    reward_info.get("reward", reward_info.get("answer_reward", 0.0))
                )
                if reward >= 1.0:
                    accumulated_reward_info.append(reward_info)
                    accumulated_ground_truths.append(
                        "" if ground_truth is None else str(ground_truth)
                    )
            accumulated_dataset.extend(selected)

            sft_update_result: dict[str, Any] = {}
            with tempfile.TemporaryDirectory(prefix="ei-sft-") as tmpdir:
                tmpdir_path = Path(tmpdir)
                sft_dataset_path = tmpdir_path / "sft_dataset.jsonl"
                with open(sft_dataset_path, "w", encoding="utf-8") as f:
                    for example in selected:
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")

                sft_train_steps = max(1, config.sft_epochs_per_round) * max(1, len(selected))
                sft_update_result = run_sft_training(
                    model_id=model_id,
                    dataset_path=sft_dataset_path,
                    validation_dataset_path=validation_dataset_path,
                    config=SFTConfig(
                        train_steps=sft_train_steps,
                        learning_rate=1e-5,
                        train_batch_size=max(1, min(len(selected) or 1, config.rollout_batch_size or 1)),
                        gradient_accumulation_steps=1,
                        normalize_constant=1.0,
                        eval_every_steps=config.eval_every_steps,
                        save_every_steps=config.save_every_steps,
                        wandb=config.wandb,
                        checkpoint=CheckpointConfig(
                            output_dir=tmpdir_path / "checkpoints",
                            save_every_steps=0,
                            max_checkpoints=1,
                            resume_from_checkpoint=None,
                        ),
                        drive_sync=DriveSyncConfig(enabled=False),
                    ),
                    eval_config=EvalConfig(
                        output_dir=tmpdir_path / "eval",
                        batch_size=1,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=512,
                    ),
                    reward_fn=reward_fn,
                )

            rounds.append(
                {
                    "round": round_idx,
                    "train_summary": train_results["summary"],
                    "validation_summary": validation_results["summary"],
                    "selected_examples": len(selected),
                    "accumulated_examples": len(accumulated_dataset),
                    "filter_keep_rate": keep_rate,
                    "sft_update": {
                        "steps": sft_update_result.get("steps", 0),
                        "loss": sft_update_result.get("loss"),
                        "loss_history": sft_update_result.get("loss_history", []),
                    },
                }
            )

    logged = log_generations(
        prompts=[item["prompt"] for item in accumulated_dataset],
        responses=[item["response"] for item in accumulated_dataset],
        ground_truths=accumulated_ground_truths,
        reward_info=accumulated_reward_info,
    )

    return {
        "model_id": model_id,
        "train_dataset_path": str(train_dataset_path),
        "validation_dataset_path": str(validation_dataset_path),
        "rounds": rounds,
        "sft_dataset": accumulated_dataset,
        "sft_dataset_size": len(accumulated_dataset),
        "logged": logged,
    }


def build_expert_iteration_dataset(
    *,
    prompts: list[str],
    responses: list[str],
    rewards: list[dict[str, float]],
) -> list[dict[str, str]]:
    """Filter rollout pairs into the SFT dataset used by expert iteration."""
    if not (len(prompts) == len(responses) == len(rewards)):
        raise ValueError("prompts, responses, and rewards must have the same length")

    dataset: list[dict[str, str]] = []
    for prompt, response, reward_info in zip(prompts, responses, rewards):
        reward = float(reward_info.get("reward", reward_info.get("answer_reward", 0.0)))
        if reward >= 1.0:
            dataset.append({"prompt": str(prompt), "response": str(response)})
    return dataset
