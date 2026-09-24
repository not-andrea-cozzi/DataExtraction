from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from .config import Config, ConfigError, load_config
from .state import PipelineState
from .steps import STEPS, Context

logger = logging.getLogger("dataset_main")


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers, force=True,
    )


def resolve_steps(step: str) -> List[str]:
    if step in ("", "all", None):
        return list(STEPS)
    if step not in STEPS:
        raise ConfigError(f"pipeline.step sconosciuto: '{step}'. Validi: {list(STEPS)}")
    return [step]


def run(cfg: Config) -> Dict[str, Any]:
    setup_logging(cfg.pipeline.log_level, cfg.pipeline.log_file)
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    ctx = Context(cfg, PipelineState(cfg.state_path))

    logger.info("=" * 70)
    logger.info("PIPELINE: %s", " -> ".join(STEPS))
    logger.info("=" * 70)

    results: Dict[str, Any] = {}
    for name in resolve_steps(cfg.pipeline.step):
        logger.info("--- step: %s ---", name)
        results[name] = STEPS[name](ctx)

    logger.info("PIPELINE COMPLETATA")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="chessmate dataset pipeline")
    ap.add_argument("--config", default="main.yaml")
    ap.add_argument("--step", default=None, help="Override pipeline.step")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
        if args.step:
            cfg.pipeline.step = args.step
        run(cfg)
        return 0
    except ConfigError as e:
        logging.getLogger("dataset_main").error("Errore di configurazione: %s", e)
        return 2
    except KeyboardInterrupt:
        logging.getLogger("dataset_main").warning("Interrotto. Rilancia lo stesso comando per riprendere.")
        return 130
    except Exception as e:
        logging.getLogger("dataset_main").exception("Interruzione imprevista: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
