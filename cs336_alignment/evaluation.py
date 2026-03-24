from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch

from .config import EvalConfig
from .vllm_utils import build_sampling_params, init_vllm


def _extract_prompt_and_ground_truth(example: Any) -> tuple[str, str | None]:
    if isinstance(example, str):
        return example, None
    if isinstance(example, dict):
        prompt = example.get("prompt")
        if prompt is None and "question" in example:
            prompt = example["question"]
        ground_truth = example.get("ground_truth", example.get("answer"))
        if prompt is None:
            raise KeyError("Example is missing a prompt/question field")
        return str(prompt), None if ground_truth is None else str(ground_truth)
    if isinstance(example, (tuple, list)) and len(example) >= 1:
        prompt = str(example[0])
        ground_truth = None if len(example) < 2 or example[1] is None else str(example[1])
        return prompt, ground_truth
    raise TypeError(f"Unsupported example type: {type(example)!r}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_rewards(reward_info: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {"num_examples": float(len(reward_info))}
    if not reward_info:
        return summary
    keys = sorted({key for item in reward_info for key in item})
    for key in keys:
        values = [item[key] for item in reward_info if key in item]
        if values:
            summary[key] = float(mean(values))
            summary[f"mean_{key}"] = float(mean(values))
    if "answer_reward" in summary:
        summary["reward_mean"] = summary["answer_reward"]
    elif "reward" in summary:
        summary["reward_mean"] = summary["reward"]
    return summary


def evaluate_vllm(
    vllm_model: Any,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params: Any,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a language model on prompts and collect reward metrics.

    Args:
        vllm_model: vLLM instance used for generation.
        reward_fn: Callable that scores a response against a ground truth.
        prompts: Prompts to evaluate.
        eval_sampling_params: Sampling parameters to pass to vLLM.
        output_path: Optional path where results should be serialized.

    Returns:
        Dictionary containing generations, rewards, and summary metrics.
    """
    prompts_and_truths = [_extract_prompt_and_ground_truth(example) for example in prompts]
    prompt_texts = [prompt for prompt, _ in prompts_and_truths]
    ground_truths = [ground_truth for _, ground_truth in prompts_and_truths]

    generations = vllm_model.generate(prompt_texts, eval_sampling_params)
    responses = [generation.outputs[0].text for generation in generations]

    reward_info: list[dict[str, float]] = []
    for response, ground_truth in zip(responses, ground_truths):
        if ground_truth is None:
            reward_info.append({})
        else:
            reward_info.append(reward_fn(response, ground_truth))

    rows = []
    for prompt, ground_truth, response, reward in zip(prompt_texts, ground_truths, responses, reward_info):
        rows.append(
            {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "response": response,
                "reward_info": reward,
            }
        )

    summary = _summarize_rewards(reward_info)
    output_data = {
        "prompts": prompt_texts,
        "ground_truths": ground_truths,
        "responses": responses,
        "reward_info": reward_info,
        "summary": summary,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path if output_path.suffix == ".jsonl" else output_path.with_suffix(".jsonl"), rows)
        with open(
            output_path if output_path.suffix == ".json" else output_path.with_suffix(".json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    return output_data


def log_generations(
    *,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_info: list[dict[str, float]],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create a logging payload for sample generations.

    Args:
        prompts: Input prompts shown to the model.
        responses: Model responses.
        ground_truths: Reference answers.
        reward_info: Reward dictionaries for each example.
        output_dir: Optional path for writing artifacts.

    Returns:
        A structured logging summary.
    """
    summary = {"num_examples": float(len(prompts))}
    if reward_info:
        keys = sorted({key for item in reward_info for key in item})
        for key in keys:
            values = [item[key] for item in reward_info if key in item]
            if values:
                summary[key] = float(mean(values))
                summary[f"mean_{key}"] = float(mean(values))
        if "answer_reward" in summary:
            summary["reward_mean"] = summary["answer_reward"]
        elif "reward" in summary:
            summary["reward_mean"] = summary["reward"]

    payload = {
        "prompts": prompts,
        "responses": responses,
        "ground_truths": ground_truths,
        "reward_info": reward_info,
        "summary": summary,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            output_dir / "generations.jsonl",
            [
                {
                    "prompt": prompt,
                    "response": response,
                    "ground_truth": ground_truth,
                    "reward_info": reward,
                }
                for prompt, response, ground_truth, reward in zip(
                    prompts, responses, ground_truths, reward_info
                )
            ],
        )
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return payload


def run_zero_shot_baseline(
    *,
    model_id: str,
    dataset_path: str | Path,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_config: EvalConfig,
) -> dict[str, Any]:
    """Scaffold entrypoint for the zero-shot baseline evaluation."""
    dataset_path = Path(dataset_path)
    examples: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    if eval_config.num_examples is not None:
        examples = examples[: eval_config.num_examples]

    prompts: list[str] = []
    ground_truths: list[str] = []
    for example in examples:
        prompt = example.get("prompt")
        if prompt is None and "question" in example:
            prompt = example["question"]
        if prompt is None:
            raise KeyError("Dataset example is missing a prompt/question field")
        prompts.append(str(prompt))
        ground_truth = example.get("ground_truth", example.get("answer", ""))
        ground_truths.append("" if ground_truth is None else str(ground_truth))

    sampling_params = build_sampling_params(
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        max_tokens=eval_config.max_tokens,
        min_tokens=eval_config.min_tokens,
        stop=eval_config.stop_tokens,
        include_stop_str_in_output=eval_config.include_stop_str_in_output,
    )

    results: dict[str, Any]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with init_vllm(model_id, device=device, seed=0) as llm:
        results = evaluate_vllm(
            llm,
            reward_fn,
            list(zip(prompts, ground_truths)),
            sampling_params,
            output_path=eval_config.output_dir / "zero_shot_results",
        )

    logged = log_generations(
        prompts=results["prompts"],
        responses=results["responses"],
        ground_truths=results["ground_truths"],
        reward_info=results["reward_info"],
        output_dir=eval_config.output_dir,
    )
    return {"results": results, "logged": logged, "output_dir": str(eval_config.output_dir)}
