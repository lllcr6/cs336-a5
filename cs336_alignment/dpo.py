from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def run_compute_per_instance_dpo_loss(
    lm,
    lm_ref,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> Any:
    """Compute the per-instance DPO loss for a single preference pair."""
    template_path = Path(__file__).resolve().parent / "prompts" / "alpaca_sft.prompt"
    template = template_path.read_text()

    prompt_text = template.format(instruction=prompt, response="").rstrip("\n") + "\n"
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids

    def _build_sequence(response_text: str) -> torch.Tensor:
        response_ids = tokenizer(
            response_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        if tokenizer.eos_token_id is not None:
            eos = torch.tensor([[tokenizer.eos_token_id]], dtype=response_ids.dtype)
            response_ids = torch.cat([response_ids, eos], dim=1)
        return torch.cat([prompt_ids, response_ids], dim=1)

    def _score(model: torch.nn.Module, response_text: str) -> torch.Tensor:
        sequence = _build_sequence(response_text)
        inputs = sequence[:, :-1].to(next(model.parameters()).device)
        labels = sequence[:, 1:].to(inputs.device)
        with torch.no_grad():
            logits = model(inputs).logits
            token_log_probs = logits.log_softmax(dim=-1).gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)
        start_index = prompt_ids.shape[1] - 1
        return token_log_probs[:, start_index:].sum()

    lm_was_training = lm.training
    lm_ref_was_training = lm_ref.training
    lm.eval()
    lm_ref.eval()
    try:
        chosen_logp = _score(lm, response_chosen)
        rejected_logp = _score(lm, response_rejected)
        chosen_ref_logp = _score(lm_ref, response_chosen)
        rejected_ref_logp = _score(lm_ref, response_rejected)
        logits = beta * ((chosen_logp - rejected_logp) - (chosen_ref_logp - rejected_ref_logp))
        return -F.logsigmoid(logits)
    finally:
        if lm_was_training:
            lm.train()
        if lm_ref_was_training:
            lm_ref.train()
