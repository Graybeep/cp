"""Memmap-backed token dataset.

The corpus is pre-tokenized once (scripts/prepare_babylm.py) into flat
uint16 binaries (train.bin / val.bin) plus a meta.json. Training reads
non-overlapping (block_size + 1)-token windows from the memmap — map-style
Dataset so DataLoader shuffling stays deterministic under a seeded
generator, and safe under num_workers > 0 on Windows (each worker opens
its own memmap handle lazily).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenBlockDataset(Dataset):
    def __init__(self, bin_path: str | Path, block_size: int):
        self.bin_path = str(bin_path)
        self.block_size = block_size
        # Open once just to get the length; workers re-open lazily.
        n_tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r").shape[0]
        self.n_blocks = (n_tokens - 1) // block_size
        self._mm: np.memmap | None = None

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mm is None:
            self._mm = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        start = i * self.block_size
        chunk = torch.from_numpy(
            self._mm[start : start + self.block_size + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


def load_meta(tokenized_dir: str | Path) -> dict:
    return json.loads((Path(tokenized_dir) / "meta.json").read_text())


def make_loader(
    bin_path: str | Path,
    block_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    ds = TokenBlockDataset(bin_path, block_size)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def cycle(loader: DataLoader):
    """Infinite iterator — training length is governed by total_steps
    (exact step count, per the locked checkpoint convention), never epochs."""
    while True:
        yield from loader
