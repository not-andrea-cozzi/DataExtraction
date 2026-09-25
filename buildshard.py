from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

from pipeline.config import ConfigError, load_config
from pipeline.main import setup_logging
from pipeline.state import PipelineState
from pipeline.steps import Context, step_finalize, step_games

logger = logging.getLogger("buildshard")


def _reset(ctx: Context) -> None:
    cfg = ctx.cfg
    ctx.spool.clear()
    for path in (
        cfg.resume_path,
        os.path.join(cfg.games_dir, "games_debug_records.pending.csv"),
    ):
        if os.path.exists(path):
            os.remove(path)
            logger.info("[buildshard] rimosso %s", path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build solo shard games")
    ap.add_argument("--config", default="main.yaml")
    ap.add_argument("--no-finalize", action="store_true", help="Ferma dopo lo spool (niente split/shard finali).", default=True)
    ap.add_argument("--force", action="store_true", help="Ignora stato/resume e riparte da zero.", default=False)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
        setup_logging(cfg.pipeline.log_level, cfg.pipeline.log_file)
        os.makedirs(cfg.dataset_dir, exist_ok=True)

        cfg.pipeline.use_existing_games = False
        if args.workers:
            cfg.games_pipeline.workers = args.workers

        ctx = Context(cfg, PipelineState(cfg.state_path))

        if args.force:
            cfg.pipeline.force_recompute = True
            _reset(ctx)

        step_games(ctx)
        games_ran = ctx.games_builder is not None

        if args.no_finalize:
            logger.info("[buildshard] --no-finalize: spool pronto in %s.", cfg.spool_dir)
            return 0
        if not games_ran:
            logger.warning(
                "[buildshard] games_pipeline gia' completato: nulla da finalizzare. Usa --force per rifare."
            )
            return 0

        cfg.pipeline.force_recompute = True
        meta = step_finalize(ctx)
        logger.info("[buildshard] shard scritti in %s: %s", cfg.merged_dir, meta)
        return 0

    except ConfigError as e:
        logger.error("Errore di configurazione: %s", e)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrotto. Rilancia lo stesso comando per riprendere (resume attivo).")
        return 130
    except Exception as e:
        logger.exception("Interruzione imprevista: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())