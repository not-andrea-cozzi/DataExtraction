from __future__ import annotations

import logging
import os
from typing import List

from common.io import read_json
from train.dataset import ChessShardDataset

logger = logging.getLogger("evaluate.holdout")


def read_fens(holdout_dir: str) -> List[str]:
    """FEN di ogni posizione dello shard heldout, nell'ordine del dataset (= idx usato ovunque)."""
    ds = ChessShardDataset(holdout_dir)
    fens = [getattr(ds[i], "fen", None) for i in range(len(ds))]
    if any(f is None for f in fens):
        raise ValueError(f"Posizioni senza campo 'fen' in {holdout_dir}: rigenera l'heldout con heldout.py.")
    return fens


def check_mate_range(holdout_dir: str, mate_range) -> None:
    """Confronta il mate_range del checkpoint con quello con cui heldout.py ha costruito gli shard."""
    report = os.path.join(os.path.dirname(os.path.abspath(holdout_dir)), "report.json")
    if not os.path.exists(report):
        logger.warning("report.json non trovato accanto all'heldout: mate_range non verificato.")
        return
    shard_range = tuple((read_json(report, default={}) or {}).get("mate_range_shards", ()))
    if shard_range and shard_range != tuple(mate_range):
        raise RuntimeError(f"mate_range checkpoint {tuple(mate_range)} != heldout {shard_range}.")