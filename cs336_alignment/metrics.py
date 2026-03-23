from __future__ import annotations

import re
from typing import Any


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """Parse an MMLU response into a multiple-choice option label."""
    _ = mmlu_example
    matches = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", model_output.upper())
    return matches[-1] if matches else None


def run_parse_gsm8k_response(model_output: str) -> str | None:
    """Parse a GSM8K response into the final numeric answer."""
    matches = re.findall(r"(?<!\w)(-?\d[\d,]*(?:\.\d+)?)", model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")
