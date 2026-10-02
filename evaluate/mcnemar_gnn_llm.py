"""Test 3: McNemar esatto (policy) GNN vs LLM, stratificato per n.

Dati appaiati per posizione (idx + FEN). Le posizioni con errore API dell'LLM sono escluse
(non sono errori del modello); predizioni vuote/illegali dell'LLM contano come errate.
Output (tutti CSV + PNG) in <out_dir>/mcnemar_gnn_vs_llm/:
  <variant>_vs_llm.{csv,png}   test McNemar per n + riga pooled 'all'
  accuracy_by_n.{csv,png}      accuracy con intervalli di Wilson per ogni modello e per l'LLM
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from common import plotter

from . import stats_common
from .config import EvalConfig

logger = logging.getLogger("evaluate.mcnemar_gnn_vs_llm")


def paired(gnn: pd.DataFrame, llm: pd.DataFrame, variant: str) -> Tuple[pd.DataFrame, int]:
    m = gnn.merge(llm, on="idx", suffixes=("_gnn", "_llm"))
    if m.empty:
        raise RuntimeError(f"[{variant}] nessun idx in comune tra GNN e LLM.")
    if (m["fen_gnn"] != m["fen_llm"]).any() or (m["mate_n_gnn"] != m["mate_n_llm"]).any():
        raise RuntimeError(f"[{variant}] FEN/mate_n non allineati tra GNN e LLM: heldout cambiato tra i due run?")
    excluded = int(m["api_error"].sum())
    m = m[m["policy_valid"] & ~m["api_error"]]
    return pd.DataFrame({
        "mate_n": m["mate_n_gnn"].to_numpy(),
        f"{variant}_correct": m["policy_correct"].astype(bool).to_numpy(),
        "llm_correct": m["correct"].astype(bool).to_numpy(),
    }), excluded


def run(cfg: EvalConfig) -> Dict[str, Any]:
    stats_common.headless()
    if not os.path.exists(cfg.llm_csv):
        logger.warning("[mcnemar gnn-vs-llm] %s assente (LLM disabilitato o fallito): skip.", cfg.llm_csv)
        return {"skipped": "no llm predictions"}

    llm = pd.read_csv(cfg.llm_csv)
    out_dir = cfg.report_dir("mcnemar_gnn_vs_llm")
    os.makedirs(out_dir, exist_ok=True)

    results: Dict[str, Any] = {}
    long_parts = []
    for i, variant in enumerate(cfg.gnn.variants):
        gnn = pd.read_csv(cfg.gnn_csv(variant))
        df, excluded = paired(gnn, llm, variant)
        results[variant] = stats_common.compare(
            df, f"{variant}_correct", "llm_correct", variant, "llm",
            tag=f"{variant}_vs_llm", title=f"Policy: {variant} vs llm (McNemar)",
            out_dir=out_dir, stats=cfg.stats,
        )
        results[variant]["excluded_api_errors"] = excluded
        long_parts.append(pd.DataFrame({"model": variant, "mate_n": df["mate_n"], "correct": df[f"{variant}_correct"]}))
        if i == 0:
            long_parts.append(pd.DataFrame({"model": "llm", "mate_n": df["mate_n"], "correct": df["llm_correct"]}))

    # Riuso di plotter: accuracy + Wilson per modello e per n (grafico a barre del progetto).
    long = pd.concat(long_parts, ignore_index=True)
    plotter.accuracy_table(long, "mate_n").to_csv(os.path.join(out_dir, "accuracy_by_n.csv"), index=False)
    plotter.plot_accuracy_by_group(
        long, "mate_n", xlabel="Mate in n", ylabel="Policy accuracy",
        title="Policy accuracy by mate depth (heldout)", save_path=os.path.join(out_dir, "accuracy_by_n.png"),
    )
    plt.close("all")
    return results