from __future__ import annotations

import glob
import logging
import os
import threading
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from utils.compression import compress_position_data, decompress_position_data

logger = logging.getLogger(__name__)

_SHARD_TEMPLATE = "shard_{:08d}.pt"
_SHARD_GLOB = "shard_*.pt"


class SpoolError(RuntimeError):
    pass


def _game_id(data: Data) -> str:
    raw = data.game_id
    return raw if isinstance(raw, str) else str(raw.item() if hasattr(raw, "item") else raw)


class PositionSpool:
    """Spool su disco (shard = source of truth). Non e' un singleton: viene
    costruito una volta dalla pipeline e passato esplicitamente ai builder.

    Ogni record: {source_tag, group_key, game_id, data(compresso)}.
    """

    def __init__(self, spool_dir: str, shard_size: int = 5000) -> None:
        self._dir = spool_dir
        self._shard_size = max(1, shard_size)
        os.makedirs(self._dir, exist_ok=True)
        self._lock = threading.Lock()
        self._pending: List[dict] = []
        self._next_idx = self._compute_next_index()

    # ---- shard io ----
    def _paths(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self._dir, _SHARD_GLOB)))

    def _compute_next_index(self) -> int:
        idx = [
            int(os.path.basename(p)[len("shard_"):-len(".pt")])
            for p in self._paths()
        ]
        return max(idx) + 1 if idx else 0

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        path = os.path.join(self._dir, _SHARD_TEMPLATE.format(self._next_idx))
        tmp = path + ".tmp"
        torch.save(self._pending, tmp)
        os.replace(tmp, path)
        self._next_idx += 1
        self._pending = []

    def enqueue(self, source_tag: str, data: Data, group_key: int) -> None:
        if getattr(data, "game_id", None) is None:
            raise SpoolError(f"enqueue: game_id mancante (source_tag={source_tag}).")
        rec = {
            "source_tag": source_tag,
            "group_key": int(group_key),
            "game_id": _game_id(data),
            "data": compress_position_data(data),
        }
        with self._lock:
            self._pending.append(rec)
            if len(self._pending) >= self._shard_size:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _iter_shards(self) -> Iterator[List[dict]]:
        self.flush()
        for path in self._paths():
            try:
                records = torch.load(path, weights_only=False, map_location="cpu")
            except Exception as e:
                raise SpoolError(f"Shard illeggibile {path}: {e}") from e
            try:
                yield records
            finally:
                del records

    # ---- split ----
    def build_split_assignment(
        self, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
    ) -> Dict[str, str]:
        """game_id -> split, stratificato per (group_key, source_tag). Pass 1: solo metadati."""
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise SpoolError("ratios: 3 valori con somma 1.0.")

        strata: Dict[str, Tuple[int, str]] = {}
        for records in self._iter_shards():
            for r in records:
                key = (int(r["group_key"]), str(r["source_tag"]))
                prev = strata.setdefault(r["game_id"], key)
                if prev != key:
                    raise SpoolError(f"game_id={r['game_id']!r}: strato incoerente {prev} vs {key}.")
        if not strata:
            raise SpoolError("Spool vuoto.")

        groups: Dict[Tuple[int, str], List[str]] = defaultdict(list)
        for gid, s in strata.items():
            groups[s].append(gid)

        gen = torch.Generator().manual_seed(seed)
        n_tr_r, n_va_r, _ = ratios
        out: Dict[str, str] = {}
        for stratum in sorted(groups):
            gids = sorted(groups[stratum])
            n = len(gids)
            n_train = min(int(n_tr_r * n), n)
            n_val = min(int(n_va_r * n), n - n_train)
            order = [gids[i] for i in torch.randperm(n, generator=gen).tolist()]
            for g in order[:n_train]:
                out[g] = "train"
            for g in order[n_train:n_train + n_val]:
                out[g] = "val"
            for g in order[n_train + n_val:]:
                out[g] = "test"

        counts = {s: sum(1 for v in out.values() if v == s) for s in ("train", "val", "test")}
        logger.info("[spool] split: %s (%d strati, %d finestre)", counts, len(groups), len(out))
        return out

    def iter_positions(self, assignment: Dict[str, str]) -> Iterator[Tuple[str, Data]]:
        """Pass 2: (split, Data decompresso), uno shard in RAM per volta."""
        for records in self._iter_shards():
            for r in records:
                split = assignment.get(r["game_id"])
                if split is None:
                    raise SpoolError(f"game_id={r['game_id']!r} senza split: spool cambiato tra pass 1 e 2.")
                yield split, decompress_position_data(r["data"])

    def clear(self) -> None:
        with self._lock:
            self._pending = []
            for p in self._paths():
                try:
                    os.remove(p)
                except OSError as e:
                    logger.warning("rimozione shard fallita %s: %s", p, e)
            self._next_idx = 0
