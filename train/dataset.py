from __future__ import annotations

import glob
import itertools
import json
import os
import random
from bisect import bisect_right
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch, Data

from utils.compression import decompress_position_data

BATCH_EXCLUDE_KEYS = [
    "legal_move_indices",
    "game_id",
    "fen",
    "rating",
    "ply",
    "y",
    "position_mate_n",
    "outcome",
]


def _load_shard(path: str) -> list:
    return torch.load(path, weights_only=False, map_location="cpu")


def _shard_sizes(split_dir: str, paths: List[str]) -> List[int]:
    manifest_path = os.path.join(split_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            num = int(meta["num_shards"])
            size = int(meta["shard_size"])
            total = int(meta["total"])
            sizes = [size] * (num - 1) + [total - size * (num - 1)]
            if num == len(paths) and sizes[-1] > 0 and len(_load_shard(paths[-1])) == sizes[-1]:
                return sizes
        except (KeyError, ValueError, OSError):
            pass
    return [len(_load_shard(p)) for p in paths]


class ChessShardDataset(Dataset):
    def __init__(self, split_dir: str, max_shards: Optional[int] = None) -> None:
        paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.pt")))
        if not paths:
            raise FileNotFoundError(f"Nessuno shard_*.pt trovato in {split_dir}")
        sizes = _shard_sizes(split_dir, paths)
        if max_shards is not None:
            paths, sizes = paths[:max_shards], sizes[:max_shards]

        self.split_dir = split_dir
        self._shard_paths = paths
        self._sizes = sizes
        self._starts = list(itertools.accumulate([0] + sizes[:-1]))
        self._total = sum(sizes)
        self._cache_idx: Optional[int] = None
        self._cache_records: Optional[list] = None

    def __len__(self) -> int:
        return self._total

    @property
    def shard_ranges(self) -> List[Tuple[int, int, int]]:
        return [(i, s, s + n) for i, (s, n) in enumerate(zip(self._starts, self._sizes))]

    def _shard(self, shard_idx: int) -> list:
        if self._cache_idx != shard_idx:
            self._cache_records = _load_shard(self._shard_paths[shard_idx])
            self._cache_idx = shard_idx
        return self._cache_records

    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        shard_idx = bisect_right(self._starts, idx) - 1
        offset = idx - self._starts[shard_idx]
        data = self._shard(shard_idx)[offset]

        if "x_binary" in data:
            data = decompress_position_data(data)
        if not hasattr(data, "outcome"):
            raise ValueError(f"Data senza 'outcome' in {self._shard_paths[shard_idx]} idx={offset}.")
        if data.legal_move_indices.dtype != torch.long:
            data.legal_move_indices = data.legal_move_indices.to(torch.long)
        return data


class ShardAwareSampler(Sampler[int]):
    def __init__(self, dataset: ChessShardDataset, shuffle: bool = True, seed: int = 42) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        ranges = list(self.dataset.shard_ranges)
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(ranges)
        for _, start, end in ranges:
            local = list(range(start, end))
            if self.shuffle:
                rng.shuffle(local)
            yield from local


def chess_collate(batch: List[Data]):
    batch_data = Batch.from_data_list(batch, exclude_keys=BATCH_EXCLUDE_KEYS)

    node_offsets: List[int] = []
    policy_targets: List[int] = []
    running = 0
    for d in batch:
        node_offsets.append(running)
        running += d.num_nodes
        matches = (d.legal_move_indices == d.y).nonzero(as_tuple=True)[0]
        policy_targets.append(int(matches[0]) if matches.numel() else -1)

    legal_move_indices = [d.legal_move_indices for d in batch]
    mate_targets = torch.stack([d.outcome for d in batch])
    return batch_data, node_offsets, legal_move_indices, policy_targets, mate_targets