"""Test 1: analisi dei modelli GNN (policy + value) sull'heldout.

Riusa collect_predictions / summarize / make_plots del training.
Output: <out_dir>/gnn/<variant>_predictions.csv, gnn/summary.csv, gnn/plots/*
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.io import read_json
from core.constants import NUM_NODE_FEATURES
from run_train import BACKBONES, resolve_device
from train import (
    ChessDualHeadModel, ChessGATConfig, chess_collate, collect_predictions, make_plots, summarize,
)
from train.dataset import ChessShardDataset

from .config import EvalConfig
from .holdout import check_mate_range, read_fens
from .stats_common import headless

logger = logging.getLogger("evaluate.gnn_heldout")


def load_model(path: str, device: str) -> Tuple[ChessDualHeadModel, Tuple[int, int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint non trovato: {path} (serve output.save_checkpoints: true).")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = ChessGATConfig(**ck["model_config"])
    backbone = BACKBONES[ck["variant"]](model_cfg, NUM_NODE_FEATURES)
    model = ChessDualHeadModel(backbone, model_cfg)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), tuple(ck["mate_range"])


def _summary_row(variant: str, s: Dict[str, Any]) -> Dict[str, Any]:
    row = {k: s[k] for k in ("n", "policy_skipped", "policy_acc", "value_acc")}
    row.update({f"policy_acc_n{n}": v for n, v in sorted(s["policy_acc_by_mate"].items())})
    return {"variant": variant, **row}


def run(cfg: EvalConfig) -> Dict[str, Any]:
    headless()
    device = resolve_device(cfg.gnn.device)
    ds = ChessShardDataset(cfg.paths.holdout_dir)
    fens = read_fens(cfg.paths.holdout_dir)
    loader = DataLoader(ds, batch_size=cfg.gnn.batch_size, shuffle=False,
                        collate_fn=chess_collate, num_workers=0)
    logger.info("heldout: %d posizioni | device=%s | varianti=%s", len(ds), device, cfg.gnn.variants)

    frames: Dict[str, pd.DataFrame] = {}
    histories: Dict[str, Any] = {}
    rows = []
    mate_range = None
    for variant in cfg.gnn.variants:
        model, mate_range = load_model(os.path.join(cfg.paths.runs_dir, variant, "model.pt"), device)
        check_mate_range(cfg.paths.holdout_dir, mate_range)

        frame = collect_predictions(model, loader, device, mate_range[0], cfg.gnn.use_amp)
        if len(frame) != len(fens):
            raise RuntimeError(f"[{variant}] {len(frame)} predizioni per {len(fens)} posizioni.")
        frame.insert(0, "fen", fens)
        frame.insert(0, "idx", range(len(frame)))

        path = cfg.gnn_csv(variant)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frame.to_csv(path, index=False)

        frames[variant] = frame
        rows.append(_summary_row(variant, summarize(frame)))
        history = read_json(os.path.join(cfg.paths.runs_dir, variant, "history.json"), default=None)
        if history:
            histories[variant] = history
        logger.info("[%s] policy_acc=%.4f value_acc=%.4f", variant, rows[-1]["policy_acc"], rows[-1]["value_acc"])
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_path = os.path.join(cfg.paths.out_dir, "gnn", "summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    make_plots(histories, frames, os.path.join(cfg.paths.out_dir, "gnn", "plots"), mate_range, "heldout", mate_order=cfg.mate_order)
    return {"summary_csv": summary_path, "variants": list(frames)}