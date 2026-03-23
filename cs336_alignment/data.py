from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """Construct a packed SFT dataset of constant-length sequences.

    Args:
        tokenizer: Tokenizer used for packing examples.
        dataset_path: Path to the JSONL instruction-tuning dataset.
        seq_length: Fixed sequence length for packed examples.
        shuffle: Whether to shuffle documents before packing.

    Returns:
        PyTorch Dataset with `input_ids` and `labels` entries.
    """
    records = read_jsonl_dataset(dataset_path)
    if shuffle:
        random.shuffle(records)

    prompt_template = Path(__file__).resolve().parent / "prompts" / "alpaca_sft.prompt"
    template = prompt_template.read_text()

    packed_token_ids: list[int] = []

    for record in records:
        instruction = record["prompt"]
        response = record["response"]
        text = template.format(instruction=instruction, response=response).rstrip("\n")
        token_ids = tokenizer(text, add_special_tokens=True).input_ids
        if tokenizer.eos_token_id is not None:
            token_ids.append(tokenizer.eos_token_id)
        packed_token_ids.extend(token_ids)

    examples: list[dict[str, torch.Tensor]] = []
    window_size = seq_length + 1
    for start in range(0, len(packed_token_ids) - window_size + 1, seq_length):
        chunk = packed_token_ids[start : start + window_size]
        if len(chunk) < window_size:
            break
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        examples.append({"input_ids": input_ids, "labels": labels})

    class _PackedSFTDataset(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return examples[index]

    return _PackedSFTDataset()


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> Any:
    """Return a one-epoch iterable over batched dataset examples."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def read_jsonl_dataset(dataset_path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a JSONL dataset into memory."""
    records: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
