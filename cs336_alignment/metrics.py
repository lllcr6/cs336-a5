from __future__ import annotations

from typing import Any


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """Parse an MMLU response into a multiple-choice option label."""
    # TODO: extract a valid answer option from the model output.
    raise NotImplementedError


def run_parse_gsm8k_response(model_output: str) -> str | None:
    """Parse a GSM8K response into the final numeric answer."""
    # TODO: identify the last numeric answer in the model output.
    raise NotImplementedError

