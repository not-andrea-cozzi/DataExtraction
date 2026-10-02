"""Test 2: valutazione LLM sull'heldout (usa il modulo EvaluateLLM).

Output: <out_dir>/llm/llm_predictions.csv, llm_metrics_per_n.csv, summary.csv.
La cache delle risposte e' un JSON interno a EvaluateLLM (contiene raw_text/error): non e' un output.

Due protezioni rispetto all'uso diretto di EvaluateLLM:
 - la cache del modulo e' indicizzata per posizione (idx), non per FEN: il nome del file include
   un hash delle FEN, cosi' un heldout rigenerato non riusa risposte di altre posizioni;
 - il modulo non ritenta mai le voci in cache che contengono un errore API: qui vengono purgate
   prima di ogni run.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, Dict

import pandas as pd

from common.io import atomic_write_json, read_json

from . import EvaluateLLM as E
from .config import ConfigError, EvalConfig
from .holdout import read_fens

logger = logging.getLogger("evaluate.llm_heldout")


def _purge_errors(cache_path: str) -> int:
    cache = read_json(cache_path, default=None)
    if not isinstance(cache, dict):
        return 0
    kept = {k: v for k, v in cache.items() if not (isinstance(v, dict) and v.get("error"))}
    removed = len(cache) - len(kept)
    if removed:
        atomic_write_json(cache_path, kept)
        logger.info("[llm] rimosse %d voci con errore API dalla cache (verranno ritentate).", removed)
    return removed


def run(cfg: EvalConfig) -> Dict[str, Any]:
    llm = cfg.llm
    if not llm.get("enabled", True):
        logger.info("[llm] enabled=false: skip.")
        return {"skipped": "llm disabled"}

    api_key = os.environ.get(llm["llm_api_key_env"])
    if not api_key:
        raise ConfigError(f"Variabile d'ambiente {llm['llm_api_key_env']} non impostata.")

    fens = read_fens(cfg.paths.holdout_dir)
    tag = hashlib.sha1("\n".join(fens).encode("utf-8")).hexdigest()[:8]
    model_slug = re.sub(r"\W+", "_", str(llm["llm_model"])).strip("_")
    out_dir = os.path.join(cfg.paths.out_dir, "llm")
    cache_path = os.path.join(out_dir, f"llm_cache_{model_slug}_{tag}.json")
    os.makedirs(out_dir, exist_ok=True)
    _purge_errors(cache_path)

    res = E.evaluate_llm_on_holdout(
        holdout_dir=cfg.paths.holdout_dir,
        solver_factory=E._build_solver_factory_from_cfg(llm, api_key),
        cache_path=cache_path,
        max_n=int(llm.get("max_n", 10)),
        limit=llm.get("limit"),
        max_workers=int(llm.get("max_workers", 1)),
    )
    if len(res["move_correct"]) == 0:
        raise RuntimeError("Nessuna posizione valutata dall'LLM.")

    cache = read_json(cache_path, default={}) or {}
    pids = [str(p) for p in res["problem_id"]]
    frame = pd.DataFrame({
        "idx": [int(p) for p in pids],
        "fen": [fens[int(p)] for p in pids],
        "mate_n": res["mate_n"],
        "best_move_uci": res["best_move_uci"],
        "pred_move_uci": res["pred_move_uci"],
        "correct": res["move_correct"],
        "api_error": [bool((cache.get(p) or {}).get("error")) for p in pids],
    })
    frame.to_csv(cfg.llm_csv, index=False)

    valid = frame[~frame["api_error"]]
    summary = pd.DataFrame([{
        "model": llm["llm_model"],
        "n": len(frame),
        "api_errors": int(frame["api_error"].sum()),
        "empty_predictions": int((valid["pred_move_uci"] == "").sum()),
        "acc_excl_api_errors": float(valid["correct"].mean()) if len(valid) else float("nan"),
    }])
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    E.save_per_n_csv(res["per_n"], os.path.join(out_dir, "llm_metrics_per_n.csv"))
    logger.info("[llm] %s", summary.iloc[0].to_dict())
    return {"summary_csv": summary_path, "predictions_csv": cfg.llm_csv}