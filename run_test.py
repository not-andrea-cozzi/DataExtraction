"""Lancia tutti i test di valutazione sull'heldout.

    python -m evaluate.all_test --config evaluate.yaml
    python -m evaluate.all_test --config evaluate.yaml --only mcnemar_timing
    python -m evaluate.all_test --config evaluate.yaml --skip llm_heldout --limit 20

Ordine (i test McNemar leggono i CSV scritti dai primi due):
    gnn_heldout -> llm_heldout -> mcnemar_gnn_vs_llm -> mcnemar_timing
Riepilogo: <out_dir>/all_test_summary.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from evaluate.config import EvalConfig, load_config
from pipeline.config import ConfigError
from pipeline.main import setup_logging  

from evaluate import gnn_heldout, llm_heldout, mcnemar_gnn_llm, mcnemar_timing

logger = logging.getLogger("evaluate.all_test")

TESTS: Dict[str, Callable[[EvalConfig], Dict[str, Any]]] = {
    "gnn_heldout": gnn_heldout.run,
    "llm_heldout": llm_heldout.run,
    "mcnemar_gnn_llm": mcnemar_gnn_llm.run,
    "mcnemar_timing": mcnemar_timing.run,
}


def run_all(cfg: EvalConfig, only: Optional[List[str]] = None, skip: Optional[List[str]] = None) -> pd.DataFrame:
    names = [n for n in TESTS if (not only or n in only) and n not in (skip or [])]
    os.makedirs(cfg.paths.out_dir, exist_ok=True)

    rows = []
    for name in names:
        logger.info("=" * 70)
        logger.info("TEST: %s", name)
        logger.info("=" * 70)
        t0 = time.time()
        row = {"test": name, "status": "ok", "detail": ""}
        try:
            result = TESTS[name](cfg)
            if "skipped" in result:
                row.update(status="skipped", detail=result["skipped"])
        except KeyboardInterrupt:
            logger.warning("Interrotto durante '%s'.", name)
            rows.append({**row, "status": "interrupted", "seconds": round(time.time() - t0, 1)})
            break
        except Exception as e:
            logger.exception("Test '%s' fallito.", name)
            row.update(status="failed", detail=f"{type(e).__name__}: {e}")
        rows.append({**row, "seconds": round(time.time() - t0, 1)})

    summary = pd.DataFrame(rows, columns=["test", "status", "seconds", "detail"])
    summary.to_csv(os.path.join(cfg.paths.out_dir, "all_test_summary.csv"), index=False)
    logger.info("Esiti: %s", dict(zip(summary["test"], summary["status"])))
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging("INFO")
    ap = argparse.ArgumentParser(description="Lancia tutti i test di valutazione sull'heldout")
    ap.add_argument("--config", default="evaluate.yaml")
    ap.add_argument("--only", nargs="+", choices=list(TESTS), default=None)
    ap.add_argument("--skip", nargs="+", choices=list(TESTS), default=None)
    ap.add_argument("--limit", type=int, default=None, help="Limita le posizioni valutate dall'LLM (smoke test).")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        logger.error("Errore di configurazione: %s", e)
        return 2
    setup_logging("INFO", os.path.join(cfg.paths.out_dir, "all_test.log"))
    if args.limit is not None:
        cfg.llm["limit"] = args.limit

    summary = run_all(cfg, args.only, args.skip)
    return 1 if summary["status"].isin(["failed", "interrupted"]).any() else 0


if __name__ == "__main__":
    sys.exit(main())