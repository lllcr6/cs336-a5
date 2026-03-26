import torch

import cs336_alignment.sft as sft_module

from .adapters import (
    run_compute_entropy as compute_entropy,
    run_get_response_log_probs as get_response_log_probs,
    run_masked_normalize as masked_normalize,
    run_tokenize_prompt_and_output as tokenize_prompt_and_output,
    run_sft_microbatch_train_step as sft_microbatch_train_step,
)

def test_tokenize_prompt_and_output(numpy_snapshot, prompt_strs, output_strs, tokenizer):
    output = tokenize_prompt_and_output(
        prompt_strs=prompt_strs,
        output_strs=output_strs,
        tokenizer=tokenizer,
    )
    numpy_snapshot.assert_match(output)

def test_compute_entropy(numpy_snapshot, logits):
    output = compute_entropy(logits)
    numpy_snapshot.assert_match(output)


def test_get_response_log_probs(
    numpy_snapshot,
    model,
    input_ids,
    labels,
):
    output = get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=True,
    )
    numpy_snapshot.assert_match(output)

def test_masked_normalize_dim0(numpy_snapshot, tensor, mask, normalize_constant):
    output = masked_normalize(
        tensor=tensor,
        mask=mask,
        normalize_constant=normalize_constant,
        dim=0,
    )
    numpy_snapshot.assert_match(output)


def test_masked_normalize_dim1(numpy_snapshot, tensor, mask, normalize_constant):
    output = masked_normalize(
        tensor=tensor,
        mask=mask,
        normalize_constant=normalize_constant,
        dim=1,
    )
    numpy_snapshot.assert_match(output)


def test_masked_normalize_dimlast(numpy_snapshot, tensor, mask, normalize_constant):
    output = masked_normalize(
        tensor=tensor,
        mask=mask,
        normalize_constant=normalize_constant,
        dim=-1,
    )
    numpy_snapshot.assert_match(output)


def test_masked_normalize_dimNone(numpy_snapshot, tensor, mask, normalize_constant):
    output = masked_normalize(
        tensor=tensor,
        mask=mask,
        normalize_constant=normalize_constant,
    )
    numpy_snapshot.assert_match(output)

def test_sft_microbatch_train_step(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
):
    policy_log_probs.requires_grad = True
    loss, _ = sft_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=1.0,
    )
    output = {"loss": loss, "policy_log_probs_grad": policy_log_probs.grad}
    numpy_snapshot.assert_match(output)

def test_sft_microbatch_train_step_normalize(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    normalize_constant,
):
    policy_log_probs.requires_grad = True
    loss, _ = sft_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=normalize_constant,
    )
    output = {"loss": loss, "policy_log_probs_grad": policy_log_probs.grad}
    numpy_snapshot.assert_match(output)

def test_sft_microbatch_train_step_10_steps(
    numpy_snapshot,
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
):
    policy_log_probs.requires_grad = True

    loss_list = []
    grad_list = []
    for _ in range(10):
        loss, _ = sft_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            normalize_constant=1.0,
        )
        loss_list.append(loss)
        grad_list.append(policy_log_probs.grad)

    output = {
        "loss": torch.stack(loss_list),
        "policy_log_probs_grad": torch.stack(grad_list),
    }
    numpy_snapshot.assert_match(output)


def test_sft_generation_validation_logs_answer_reward_and_response_entropy(monkeypatch):
    rewards_seen: list[tuple[str, str]] = []

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, prompt, return_tensors="pt"):
            del prompt, return_tensors
            return {"input_ids": torch.tensor([[10, 11]], dtype=torch.long)}

        def decode(self, token_ids, skip_special_tokens=True):
            del token_ids, skip_special_tokens
            return "generated answer"

    class FakeModel:
        def eval(self):
            return self

        def train(self):
            return self

        def generate(self, **kwargs):
            del kwargs
            return torch.tensor([[10, 11, 12, 13]], dtype=torch.long)

    def fake_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
        del tokenizer
        assert prompt_strs == ["Question?"]
        assert output_strs == ["generated answer"]
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4]], dtype=torch.long),
            "response_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        }

    def fake_get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
        del model, input_ids, labels
        assert return_token_entropy is True
        return {
            "log_probs": torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float32),
            "token_entropy": torch.tensor([[9.0, 1.0, 3.0]], dtype=torch.float32),
        }

    def fake_reward_fn(response, ground_truth):
        rewards_seen.append((response, ground_truth))
        return {"reward": 0.25, "answer_reward": 0.75, "format_reward": 0.5}

    monkeypatch.setattr(sft_module, "tokenize_prompt_and_output", fake_tokenize_prompt_and_output)
    monkeypatch.setattr(sft_module, "get_response_log_probs", fake_get_response_log_probs)

    metrics = sft_module._evaluate_sft_generation_validation(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        validation_records=[{"prompt": "Question?", "answer": "42"}],
        device=torch.device("cpu"),
        reward_fn=fake_reward_fn,
        batch_size=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=4,
        eval_output_dir=None,
    )

    assert rewards_seen == [("generated answer", "42")]
    assert metrics["eval/num_examples"] == 1.0
    assert metrics["eval/answer_reward"] == 0.75
    assert metrics["eval/format_reward"] == 0.5
    assert metrics["eval/reward_mean"] == 0.75
    assert metrics["eval/token_entropy"] == 2.0
    assert metrics["eval/response_length"] == 2.0


def test_sft_teacher_forced_validation_logs_response_entropy(monkeypatch):
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [len(text), len(text) + 1]

    def fake_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
        del tokenizer
        assert prompt_strs == ["Question?"]
        assert output_strs == ["gold response"]
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4]], dtype=torch.long),
            "response_mask": torch.tensor([[False, True, True]], dtype=torch.bool),
        }

    def fake_get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
        del model, input_ids, labels
        assert return_token_entropy is True
        return {
            "log_probs": torch.tensor([[-1.0, -2.0, -3.0]], dtype=torch.float32),
            "token_entropy": torch.tensor([[9.0, 1.0, 3.0]], dtype=torch.float32),
        }

    monkeypatch.setattr(sft_module, "tokenize_prompt_and_output", fake_tokenize_prompt_and_output)
    monkeypatch.setattr(sft_module, "get_response_log_probs", fake_get_response_log_probs)

    class FakeModel:
        def eval(self):
            return self

        def train(self):
            return self

    metrics = sft_module._evaluate_sft_validation(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        validation_records=[{"prompt": "Question?", "response": "gold response"}],
        device=torch.device("cpu"),
        batch_size=1,
        normalize_constant=1.0,
    )

    assert metrics["eval/num_examples"] == 1.0
    assert metrics["eval/token_entropy"] == 2.0
    assert "eval/loss" in metrics
