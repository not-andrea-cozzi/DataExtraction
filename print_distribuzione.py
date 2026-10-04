"""Distribuzione di mate_n in train / val / test.

    python plot_mate_dist.py --config train.yaml
    python plot_mate_dist.py --config train.yaml --out-dir ../Plots --max-shards 5

Legge solo il campo `position_mate_n` (fallback: outcome + mate_range_min) di ogni record,
senza decomprimere i grafi. Directory e mate_range provengono dalla config di training.
Output: <out_dir>/mate_n_distribution.png e .csv
"""
from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from glob import glob
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import plotter
from common.progress import wrap_iter
from train.config import ConfigError, load_config
from train.dataset import _load_shard

logger = logging.getLogger("plot_mate_dist")


def count_mate_n(split_dir: str, mate_lo: int, desc: str, max_shards: Optional[int]) -> Counter:
    paths = sorted(glob(os.path.join(split_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"Nessuno shard_*.pt in {split_dir}")
    if max_shards is not None:
        paths = paths[:max_shards]

    counts: Counter = Counter()
    for p in wrap_iter(paths, desc=desc, unit="shard"):
        for d in _load_shard(p):
            pmn = getattr(d, "position_mate_n", None)
            n = int(pmn) if pmn is not None else int(d.outcome) + mate_lo
            counts[n] += 1
    return counts


def plot_distribution(table: pd.DataFrame, save_path: str) -> None:
    splits = list(table.columns)
    colors = plotter.model_colors(splits)
    mate_ns = list(table.index)
    x = np.arange(len(mate_ns))
    width = 0.8 / len(splits)
    totals = table.sum()
    frac = table / totals

    with plt.rc_context(plotter.STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for ax, data, ylabel, title in (
            (axes[0], table, "Posizioni", "Conteggi assoluti"),
            (axes[1], frac * 100, "% del relativo split", "Distribuzione normalizzata per split"),
        ):
            for i, s in enumerate(splits):
                ax.bar(
                    x + (i - (len(splits) - 1) / 2) * width, data[s].to_numpy(), width,
                    color=colors[s], label=f"{s} (n={int(totals[s]):,})",
                    edgecolor="white", linewidth=0.6,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([str(m) for m in mate_ns])
            ax.set_xlabel("Mate in n")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        axes[0].legend()
        fig.tight_layout()
        plotter.save_figure(fig, save_path)
        plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Grafico distribuzione mate_n per train/val/test")
    ap.add_argument("--config", default="train.yaml")
    ap.add_argument("--out-dir", default="../Plots")
    ap.add_argument("--max-shards", type=int, default=None, help="Limita gli shard letti per split (test rapido).")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        logger.error("Errore di configurazione: %s", e)
        return 2

    dirs: Dict[str, Optional[str]] = {
        "train": cfg.data.train_dir, "val": cfg.data.val_dir, "test": cfg.data.test_dir,
    }
    per_split: Dict[str, Counter] = {}
    for name, d in dirs.items():
        if not d:
            logger.warning("[%s] directory non configurata: saltato.", name)
            continue
        per_split[name] = count_mate_n(d, cfg.data.mate_range_min, f"[{name}] mate_n", args.max_shards)
        logger.info("[%s] %d posizioni: %s", name, sum(per_split[name].values()), dict(sorted(per_split[name].items())))

    table = pd.DataFrame(per_split).fillna(0).astype(int).sort_index()
    table.index.name = "mate_n"

    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "mate_n_distribution.csv"))
    out = os.path.join(args.out_dir, "mate_n_distribution.png")
    plot_distribution(table, out)
    logger.info("Salvato: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())