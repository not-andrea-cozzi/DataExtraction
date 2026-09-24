from __future__ import annotations

import logging
import os

from TimeStatBuilder.clock_builder import ClockSampler

from .config import PuzzleBuilderConfig

logger = logging.getLogger("puzzle_builder")


def build_clock_sampler(cfg: PuzzleBuilderConfig) -> ClockSampler:
    kwargs = dict(
        mode=cfg.clock_mode,
        condition_on_mate_n=cfg.clock_condition_on_mate_n,
        min_seconds=cfg.clock_min_seconds,
        cap_seconds=cfg.clock_cap_seconds,
    )
    if cfg.clock_stats_path and os.path.exists(cfg.clock_stats_path):
        logger.info("Clock puzzle: statistiche reali da %s (mode=%s).", cfg.clock_stats_path, cfg.clock_mode)
        return ClockSampler.from_json(cfg.clock_stats_path, **kwargs)
    if cfg.avg_time_by_rating:
        logger.warning(
            "clock_stats_path assente o non trovato: fallback su avg_time_by_rating (sigma fissa)."
        )
        return ClockSampler.from_avg_time(cfg.avg_time_by_rating, **kwargs)
    raise ValueError(
        "PuzzleBuilder: servono clock_stats_path (da ClockStatsBuilder) oppure avg_time_by_rating."
    )