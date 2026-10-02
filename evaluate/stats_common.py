from __future__ import annotations

import logging
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from common import plotter 

from .config import StatsSection

logger = logging.getLogger("evaluate.stats")


def headless() -> None:
    plt.switch_backend("Agg")


def _verdict(p: float, acc_a: float, acc_b: float, label_a: str, label_b: str, alpha: float) -> str:
    if p >= alpha:
        return "n.s."
    return label_a if acc_a > acc_b else label_b


def _pooled_row(df: pd.DataFrame, col_a: str, col_b: str, label_a: str, label_b: str, alpha: float) -> Dict[str, Any]:
    a = df[col_a].astype(bool).to_numpy()
    b = df[col_b].astype(bool).to_numpy()
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = plotter.mcnemar_exact(only_a, only_b)  # test singolo: nessuna correzione
    acc_a, acc_b = float(a.mean()), float(b.mean())
    return {
        "mate_n": "all", "n": int(len(df)), "k_a": int(a.sum()), "k_b": int(b.sum()),
        "acc_a": acc_a, "acc_b": acc_b, "only_a": only_a, "only_b": only_b,
        "p_value": p, "p_adjusted": p, "verdict": _verdict(p, acc_a, acc_b, label_a, label_b, alpha),
    }


def compare(
    df: pd.DataFrame, col_a: str, col_b: str, label_a: str, label_b: str,
    tag: str, title: str, out_dir: str, stats: StatsSection,
) -> Dict[str, Any]:
    """McNemar per mate_n (con correzione) + riga pooled 'all'. Scrive <tag>.csv e <tag>.png."""
    if df.empty:
        raise ValueError(f"[{tag}] nessuna coppia di predizioni confrontabile.")
    os.makedirs(out_dir, exist_ok=True)
    colors = plotter.model_colors([label_a, label_b])
    _, table = plotter.plot_paired_comparison(
        df, "mate_n", col_a, col_b, label_a=label_a, label_b=label_b,
        color_a=colors[label_a], color_b=colors[label_b], correction=stats.correction,
        xlabel="Mate in n", title=title, save_path=os.path.join(out_dir, f"{tag}.png"),
    )
    plt.close("all")

    table = table.copy()
    table["verdict"] = [
        _verdict(r.p_adjusted, r.acc_a, r.acc_b, label_a, label_b, stats.alpha) for r in table.itertuples()
    ]
    pooled = _pooled_row(df, col_a, col_b, label_a, label_b, stats.alpha)
    out = pd.concat([table, pd.DataFrame([pooled])], ignore_index=True)
    out = out.rename(columns={
        "k_a": f"k_{label_a}", "k_b": f"k_{label_b}", "acc_a": f"acc_{label_a}", "acc_b": f"acc_{label_b}",
        "only_a": f"only_{label_a}", "only_b": f"only_{label_b}",
    })
    path = os.path.join(out_dir, f"{tag}.csv")
    out.to_csv(path, index=False)

    logger.info("[%s] pooled: %s=%.3f %s=%.3f p=%.3g (%s) | per n: %s", tag, label_a, pooled["acc_a"],
                label_b, pooled["acc_b"], pooled["p_value"], pooled["verdict"],
                dict(zip(table["mate_n"], table["verdict"])))
    return {"tag": tag, "csv": path, "pooled_verdict": pooled["verdict"]}