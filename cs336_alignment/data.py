from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
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
    # TODO: read, tokenize, pack, and return a dataset of fixed-length examples.
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> Any:
    """Return a one-epoch iterable over batched dataset examples."""
    # TODO: wrap the dataset in a DataLoader and return an epoch iterator.
    raise NotImplementedError


def read_jsonl_dataset(dataset_path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a JSONL dataset into memory."""
    # TODO: stream or load the dataset records.
    raise NotImplementedError
