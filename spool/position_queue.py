from __future__ import annotations

import glob
import logging
import os
import pickle
import threading
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from utils.compression import compress_position_data, decompress_position_data

logger = logging.getLogger(__name__)

_SHARD_TEMPLATE = "shard_{:08d}.pt"
_SHARD_GLOB = "shard_*.pt"
_META_SUFFIX = ".meta"

# (game_id, group_key, source_tag)
Meta = List[Tuple[str, int, str]]


class SpoolError(RuntimeError):
    pass


def _game_id(data: Data) -> str:
    raw = data.game_id
    return raw if isinstance(raw, str) else str(raw.item() if hasattr(raw, "item") else raw)


def _meta_path(shard_path: str) -> str:
    # shard_XXXXXXXX.pt -> shard_XXXXXXXX.meta (non matcha shard_*.pt)
    return shard_path[: -len(".pt")] + _META_SUFFIX


def _write_meta(shard_path: str, records: List[dict]) -> Meta:
    meta: Meta = [(r["game_id"], int(r["group_key"]), str(r["source_tag"])) for r in records]
    mp = _meta_path(shard_path)
    tmp = mp + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, mp)
    return meta


def _read_meta_if_fresh(shard_path: str) -> Optional[Meta]:
    """Legge il sidecar solo se esiste, non e' piu' vecchio dello shard e non e' corrotto."""
    mp = _meta_path(shard_path)
    try:
        if os.path.getmtime(mp) < os.path.getmtime(shard_path):
            return None
        with open(mp, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None


class PositionSpool:
    """Spool su disco (shard = source of truth). Non e' un singleton: viene
    costruito una volta dalla pipeline e passato esplicitamente ai builder.

    Ogni record: {source_tag, group_key, game_id, data(compresso)}.
    Per ogni shard esiste un sidecar .meta con (game_id, group_key, source_tag)
    di tutti i record, cosi' la pass 1 dello split non deserializza i tensori.
    Se il sidecar manca o e' piu' vecchio dello shard viene ricostruito.
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
        idx = [int(os.path.basename(p)[len("shard_"):-len(".pt")]) for p in self._paths()]
        return max(idx) + 1 if idx else 0

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        path = os.path.join(self._dir, _SHARD_TEMPLATE.format(self._next_idx))
        tmp = path + ".tmp"
        torch.save(self._pending, tmp)
        os.replace(tmp, path)
        _write_meta(path, self._pending)
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

    @staticmethod
    def _load_shard(path: str) -> List[dict]:
        try:
            return torch.load(path, weights_only=False, map_location="cpu")
        except Exception as e:
            raise SpoolError(f"Shard illeggibile {path}: {e}") from e

    def _iter_shards(self, desc: str = "") -> Iterator[List[dict]]:
        self.flush()
        for path in tqdm(self._paths(), desc=desc, unit="shard", disable=not desc):
            records = self._load_shard(path)
            try:
                yield records
            finally:
                del records

    def _iter_meta(self) -> Iterator[Meta]:
        """Metadati per shard. Fast path: sidecar. Fallback (una tantum): carica lo shard e lo scrive."""
        self.flush()
        for path in tqdm(self._paths(), desc="[spool] metadati shard", unit="shard"):
            meta = _read_meta_if_fresh(path)
            if meta is None:
                records = self._load_shard(path)
                meta = _write_meta(path, records)
                del records
            yield meta

    def approx_positions(self) -> int:
        """Upper bound del numero di record (shard * shard_size)."""
        self.flush()
        return len(self._paths()) * self._shard_size

    # ---- split ----
    def build_split_assignment(
        self, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
    ) -> Dict[str, str]:
        """game_id -> split, stratificato per (group_key, source_tag). Pass 1: solo metadati.
        Strato incoerente (resume): vince il primo record incontrato."""
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise SpoolError("ratios: 3 valori con somma 1.0.")

        strata: Dict[str, Tuple[int, str]] = {}
        conflicts = 0
        for meta in self._iter_meta():
            for gid, group_key, tag in meta:
                key = (group_key, tag)
                if gid in strata:
                    if strata[gid] != key:
                        conflicts += 1
                else:
                    strata[gid] = key

        if not strata:
            raise SpoolError("Spool vuoto.")
        if conflicts:
            logger.warning("[spool] %d record con strato incoerente (resume): vince il primo.", conflicts)

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
        logger.info("[spool] split: %s (%d strati, %d partite)", counts, len(groups), len(out))
        return out

    def iter_positions(self, assignment: Dict[str, str]) -> Iterator[Tuple[str, Data]]:
        """Pass 2: (split, Data decompresso), uno shard in RAM per volta."""
        for records in self._iter_shards(desc="[spool] pass 2 shard"):
            for r in records:
                split = assignment.get(r["game_id"])
                if split is None:
                    raise SpoolError(f"game_id={r['game_id']!r} senza split: spool cambiato tra pass 1 e 2.")
                yield split, decompress_position_data(r["data"])

    def clear(self) -> None:
        with self._lock:
            self._pending = []
            for p in self._paths():
                for q in (p, _meta_path(p)):
                    try:
                        os.remove(q)
                    except FileNotFoundError:
                        pass
                    except OSError as e:
                        logger.warning("rimozione fallita %s: %s", q, e)
            for q in glob.glob(os.path.join(self._dir, "*" + _META_SUFFIX)):
                try:
                    os.remove(q)
                except OSError:
                    pass
            self._next_idx = 0