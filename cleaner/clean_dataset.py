from __future__ import annotations

import gc
import json
import logging
import os
import time
from typing import Any, Dict, List

from common.progress import wrap_iter
import torch
from torch_geometric.data import Data
logger = logging.getLogger("clean")

KEEP_FIELDS = (
    "event_ids", "x", "edge_index", "edge_attr",
    "time", "y", "legal_move_indices", "position_mate_n", "outcome", "num_nodes",
    "fen",
)
SHARD_FILENAME_TEMPLATE = "shard_{:05d}.pt"
MANIFEST_FILENAME = "manifest.json"


def _clean_single(data: Data) -> Data:
    cleaned = Data()
    for key in KEEP_FIELDS:
        if hasattr(data, key):
            val = getattr(data, key)
            if val is not None:
                cleaned[key] = val
    return cleaned


def _clean_chunk(chunk: List[Data]) -> List[Data]:
    return [_clean_single(d) for d in chunk]


def _read_manifest(in_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(in_dir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest non trovato: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_sharded_directory(
    in_dir: str,
    out_dir: str,
    target_shard_size: int = 8000,
    workers: int = 0,
) -> Dict[str, Any]:
    manifest_in = _read_manifest(in_dir)
    num_input_shards = manifest_in["num_shards"]

    os.makedirs(out_dir, exist_ok=True)

    buffer: List[Data] = []
    output_shard_idx = 0
    total_cleaned = 0
    missing_outcome = 0
    missing_legal_moves = 0

    logger.info(
        f"[{os.path.basename(in_dir)}] Input: {num_input_shards} shard, "
        f"target_shard_size={target_shard_size}"
    )

    for shard_i in wrap_iter(
        range(num_input_shards),
        desc=f"[{os.path.basename(in_dir)}] Elaborazione shard",
        unit="shard",
    ):
        in_path = os.path.join(in_dir, SHARD_FILENAME_TEMPLATE.format(shard_i))
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Shard di input non trovato: {in_path}")

        data_list: List[Data] = torch.load(in_path, weights_only=False)

        for d in data_list:
            if not hasattr(d, "outcome"):
                missing_outcome += 1
            if not hasattr(d, "legal_move_indices"):
                missing_legal_moves += 1

        cleaned = _clean_chunk(data_list)
        del data_list
        gc.collect()

        buffer.extend(cleaned)
        total_cleaned += len(cleaned)
        del cleaned
        gc.collect()

        while len(buffer) >= target_shard_size:
            chunk = buffer[:target_shard_size]
            del buffer[:target_shard_size]

            out_path = os.path.join(
                out_dir, SHARD_FILENAME_TEMPLATE.format(output_shard_idx)
            )
            tmp_path = out_path + ".tmp"
            torch.save(chunk, tmp_path)
            os.replace(tmp_path, out_path)
            output_shard_idx += 1
            del chunk
            gc.collect()

    if buffer:
        out_path = os.path.join(
            out_dir, SHARD_FILENAME_TEMPLATE.format(output_shard_idx)
        )
        tmp_path = out_path + ".tmp"
        torch.save(buffer, tmp_path)
        os.replace(tmp_path, out_path)
        output_shard_idx += 1
        del buffer
        gc.collect()

    if missing_outcome or missing_legal_moves:
        logger.warning(
            f"[{os.path.basename(in_dir)}] Data incompleti nell'input: "
            f"{missing_outcome} senza 'outcome', {missing_legal_moves} senza 'legal_move_indices'. "
            f"Questi record risulteranno inutilizzabili dal training (ChessShardDataset solleva "
            f"ValueError su 'outcome' mancante)."
        )

    manifest_out = {
        "num_shards": output_shard_idx,
        "shard_size": target_shard_size,
        "total": total_cleaned,
        "missing_outcome": missing_outcome,
        "missing_legal_move_indices": missing_legal_moves,
    }
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2)
    os.replace(tmp_manifest, manifest_path)

    logger.info(
        f"[{os.path.basename(in_dir)}] Completato: {output_shard_idx} shard "
        f"({total_cleaned:,} Data) -> '{out_dir}'"
    )
    return manifest_out