"""Test 4: McNemar esatto, ablation timing: gat_time_decay (con tempo) vs gat_no_time (senza).

Stesse posizioni heldout per entrambi; confronto su policy e su value, stratificato per n.
Riusa paired_frame di train.reporting per la policy.
Output: <out_dir>/mcnemar_timing/{policy,value}_<A>_vs_<B>.{csv,png}
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from train.reporting import NO_TIME, TIME_DECAY, paired_frame

from . import stats_common
from .config import EvalConfig

logger = logging.getLogger("evaluate.mcnemar_timing")


def run(cfg: EvalConfig) -> Dict[str, Any]:
    stats_common.headless()
    a, b = TIME_DECAY, NO_TIME
    if a not in cfg.gnn.variants or b not in cfg.gnn.variants:
        logger.warning("[mcnemar timing] servono entrambe le varianti %s e %s: skip.", a, b)
        return {"skipped": f"variants must include {a} and {b}"}

    fa = pd.read_csv(cfg.gnn_csv(a)).sort_values("idx").reset_index(drop=True)
    fb = pd.read_csv(cfg.gnn_csv(b)).sort_values("idx").reset_index(drop=True)
    if not fa["fen"].equals(fb["fen"]):
        raise RuntimeError(f"Le predizioni di {a} e {b} non sono sulle stesse posizioni (FEN diverse).")

    policy = paired_frame(fa, fb, a, b)
    if policy is None:
        raise RuntimeError(f"paired_frame: mate_n/n_legal non allineati tra {a} e {b}.")
    value = pd.DataFrame({
        "mate_n": fa["mate_n"].to_numpy(),
        f"{a}_correct": (fa["value_true"] == fa["value_pred"]).to_numpy(),
        f"{b}_correct": (fb["value_true"] == fb["value_pred"]).to_numpy(),
    })

    out_dir = cfg.report_dir("mcnemar_timing")
    results: Dict[str, Any] = {}
    for name, df in (("policy", policy), ("value", value)):
        results[name] = stats_common.compare(
            df, f"{a}_correct", f"{b}_correct", a, b,
            tag=f"{name}_{a}_vs_{b}", title=f"{name.capitalize()}: {a} vs {b} (McNemar)",
            out_dir=out_dir, stats=cfg.stats,
        )
    return results