from __future__ import annotations

import gc
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from builders.games import (
    ClockConfig, EngineConfig, GamesBuilder, GamesBuilderConfig, SamplingConfig, SourceSpec,
)
from builders.puzzles import PuzzleBuilder, PuzzleBuilderConfig, PuzzleClockConfig
from spool.position_queue import PositionSpool
from stats import ClockStatsBuilder, TimeStatsBuilder, load_avg_time_by_rating
from utils.filters import HeaderFilterConfig, QualityConfig

from .config import Config, ConfigError
from .state import PipelineState

logger = logging.getLogger("pipeline")

# Colonne fisse dei CSV di debug (games e puzzle condividono lo stesso schema).
DEBUG_FIELDS = [
    "problem_id", "game_id", "fen", "best_move_uci", "mate_n", "mate_n_window",
    "ply", "source", "clock_source", "clock_seconds", "clock_is_real", "rating",
]


class Context:
    """Stato condiviso tra step: config, state, spool (una sola istanza)."""

    def __init__(self, cfg: Config, state: PipelineState) -> None:
        self.cfg = cfg
        self.state = state
        self._spool: Optional[PositionSpool] = None
        self._avg_time: Optional[Dict[int, float]] = None
        self.games_builder: Optional[GamesBuilder] = None
        self.puzzle_builder: Optional[PuzzleBuilder] = None

    @property
    def spool(self) -> PositionSpool:
        if self._spool is None:
            self._spool = PositionSpool(self.cfg.spool_dir, shard_size=self.cfg.games_pipeline.shard_size)
        return self._spool

    @property
    def avg_time(self) -> Dict[int, float]:
        if self._avg_time is None:
            p = self.cfg.time_stats_path
            self._avg_time = load_avg_time_by_rating(p) if os.path.exists(p) else {}
            if not self._avg_time:
                logger.warning("avg_time_by_rating assente (%s): fallback su default_move_seconds.", p)
        return self._avg_time

    def skip_if_done(self, step: str) -> bool:
        return self.state.is_done(step, force=self.cfg.pipeline.force_recompute)


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise ConfigError(f"{label} non trovato: {path}")


def _guard(ctx: Context, step: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return fn()
    except Exception as e:
        ctx.state.mark_failed(step, f"{type(e).__name__}: {e}")
        raise


def _scalars(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str))}


# ---------------------------------------------------------------- time_stats
def step_time_stats(ctx: Context) -> Dict[str, Any]:
    cfg, ts, raw = ctx.cfg, ctx.cfg.time_stats, ctx.cfg.raw_data
    out = cfg.time_stats_path
    if ctx.skip_if_done("time_stats") and os.path.exists(out):
        logger.info("[time_stats] gia' completato: skip.")
        return ctx.state.meta("time_stats")
    if not raw.games_zst:
        logger.warning("[time_stats] raw_data.games_zst vuoto: skip.")
        return {}
    _require_file(raw.games_zst, "raw_data.games_zst")

    def run():
        stats = TimeStatsBuilder(raw.games_zst, ts.max_games, ts.bucket_size).build_and_save(out)
        meta = {"output": out, "buckets": len(stats)}
        ctx.state.mark_done("time_stats", **meta)
        return meta

    return _guard(ctx, "time_stats", run)


# ------------------------------------------------------------------- games
def _game_sources(cfg: Config) -> List[SourceSpec]:
    raw, g = cfg.raw_data, cfg.games_pipeline
    specs = [
        ("lichess", raw.games_zst, g.skip_games_part1, g.max_games, raw.games_source_tag, "pgn"),
        ("fics", raw.fics_pgn, g.fics_skip_games, g.fics_max_games, raw.fics_source_tag, "pgn"),
        ("club", raw.club_csv, g.club_skip_games, g.club_max_games, raw.club_source_tag, g.club_pgn_col),
    ]
    sources: List[SourceSpec] = []
    for kind, path, skip, mx, tag, col in specs:
        if not path:
            continue
        _require_file(path, f"raw_data ({kind})")
        sources.append(SourceSpec(kind=kind, path=path, pgn_col=col, skip_games=skip,
                                  max_games=mx or None, tag=tag))
    if not sources:
        raise ConfigError("games_pipeline: nessuna sorgente in raw_data.")
    return sources


def make_games_config(cfg: Config, avg_time: Dict[int, float]) -> GamesBuilderConfig:
    g, e = cfg.games_pipeline, cfg.engine
    if not e.stockfish_path or not (os.path.exists(e.stockfish_path) and os.access(e.stockfish_path, os.X_OK)):
        raise ConfigError(f"engine.stockfish_path non valido/eseguibile: {e.stockfish_path!r}")
    os.makedirs(cfg.games_dir, exist_ok=True)

    return GamesBuilderConfig(
        sources=_game_sources(cfg),
        engine=EngineConfig(
            stockfish_path=e.stockfish_path, threads=e.threads, hash_mb=e.hash_mb, multipv=2,
            syzygy_path=e.syzygy_path, analysis_time=g.analysis_time, search_depth=g.search_depth,
            retry_attempts=g.stockfish_retry_attempts, retry_backoff_seconds=g.stockfish_retry_backoff_seconds,
        ),
        mate_range=(g.mate_range_min, g.mate_range_max),
        header=HeaderFilterConfig(
            only_decisive_games=g.only_decisive_games, skip_time_forfeit=g.skip_time_forfeit,
            min_rating=g.min_rating, max_rating=g.max_rating,
        ),
        quality=QualityConfig(
            min_material_for_mate_attempt=g.min_material_for_mate_attempt,
            min_material_diff_for_mate_attempt=g.min_material_diff_for_mate_attempt,
            require_heavy_piece=g.require_heavy_piece, skip_trivial_endgame=g.skip_trivial_endgame,
            max_piece_count=g.max_piece_count, candidate_min_legal_moves=g.candidate_min_legal_moves,
            candidate_max_legal_moves=g.candidate_max_legal_moves, skip_if_in_check=g.skip_if_in_check,
            skip_forced_moves=g.skip_forced_moves,
        ),
        sampling=SamplingConfig(
            min_game_plies=g.min_game_plies, min_ply=g.min_ply, ply_sample_step=g.ply_sample_step,
            dense_tail_plies=g.dense_tail_plies, ply_sample_step_tail=g.ply_sample_step_tail,
            max_positions_per_game=g.max_positions_per_game, dedupe_positions=g.dedupe_positions,
        ),
        clock=ClockConfig(
            require_clock=g.require_clock, drop_zero_clock=g.drop_zero_clock,
            default_move_seconds=g.default_move_seconds, avg_time_by_rating=avg_time,
        ),
        workers=g.workers, pool_join_timeout=g.pool_join_timeout,
        resume_state_path=cfg.resume_path, auto_resume=True, resume_checkpoint_every=g.checkpoint_every,
        debug_dir=cfg.games_dir, save_debug_jsonl=g.save_debug_jsonl,
        dedupe_cross_file=g.dedupe_cross_file, game_id_store_path=cfg.game_id_store_path,
    )


def step_games(ctx: Context) -> Dict[str, Any]:
    cfg = ctx.cfg
    if cfg.pipeline.use_existing_games:
        logger.info("[games] use_existing_games=true: skip.")
        return {}
    if ctx.skip_if_done("games_pipeline"):
        logger.info("[games] gia' completato: skip.")
        return ctx.state.meta("games_pipeline")

    def run():
        builder = GamesBuilder(make_games_config(cfg, ctx.avg_time), ctx.spool)
        ctx.games_builder = builder
        result = builder.run()
        ctx.state.mark_done("games_pipeline", **_scalars(result))
        logger.info("[games] processed=%s accepted=%s enqueued=%s",
                    result["processed_games"], result["accepted_games"], result["enqueued_positions"])
        return result

    return _guard(ctx, "games_pipeline", run)


# -------------------------------------------------------------- clock_stats
def step_clock_stats(ctx: Context) -> Dict[str, Any]:
    cfg, cs = ctx.cfg, ctx.cfg.clock_stats
    if not cs.enabled:
        logger.info("[clock_stats] disabilitato: skip.")
        return {}
    if ctx.skip_if_done("clock_stats") and os.path.exists(cfg.clock_stats_path):
        logger.info("[clock_stats] gia' completato: skip.")
        return ctx.state.meta("clock_stats")

    paths = [os.path.join(cfg.games_dir, n)
             for n in ("games_debug.csv", "games_debug_records.pending.csv")]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        logger.warning("[clock_stats] nessun games_debug*.csv in %s: skip.", cfg.games_dir)
        return {}

    def run():
        stats = ClockStatsBuilder(paths, cs.bucket_size, cs.min_count, cs.max_seconds
                                  ).build_and_save(cfg.clock_stats_path)
        meta = {"output": cfg.clock_stats_path, "samples": int(stats["global"][2]),
                "cells": len(stats["by_rating_mate"])}
        ctx.state.mark_done("clock_stats", **meta)
        return meta

    return _guard(ctx, "clock_stats", run)


# ------------------------------------------------------------------ puzzles
def _decompress_puzzles(cfg: Config) -> Optional[str]:
    raw, p = cfg.raw_data, cfg.puzzle_pipeline
    if not raw.puzzles_zst:
        return None
    _require_file(raw.puzzles_zst, "raw_data.puzzles_zst")
    out = cfg.path(p.decompressed_csv_filename)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out

    import zstandard as zstd
    logger.info("[puzzles] decompressione %s -> %s", raw.puzzles_zst, out)
    tmp = out + ".tmp"
    with open(raw.puzzles_zst, "rb") as fin, open(tmp, "wb") as fout:
        zstd.ZstdDecompressor().copy_stream(fin, fout, read_size=p.chunk_size_bytes, write_size=p.chunk_size_bytes)
    os.replace(tmp, out)
    return out


def make_puzzle_config(cfg: Config, csv_path: str, avg_time: Dict[int, float]) -> PuzzleBuilderConfig:
    p, g = cfg.puzzle_pipeline, cfg.games_pipeline
    os.makedirs(cfg.puzzles_dir, exist_ok=True)
    return PuzzleBuilderConfig(
        csv_path=csv_path,
        mate_range=(g.mate_range_min, g.mate_range_max),
        max_puzzles=p.max_puzzles, max_puzzles_per_theme=p.max_puzzles_per_theme,
        chunksize=50_000, source_tag=cfg.raw_data.puzzles_source_tag,
        min_rating=p.min_rating, max_rating=p.max_rating, dedupe_positions=p.dedupe_positions,
        quality=QualityConfig(
            min_material_for_mate_attempt=p.min_material_for_mate_attempt,
            min_material_diff_for_mate_attempt=p.min_material_diff_for_mate_attempt,
            require_heavy_piece=p.require_heavy_piece, skip_trivial_endgame=p.skip_trivial_endgame,
            max_piece_count=p.max_piece_count,
        ),
        clock=PuzzleClockConfig(
            stats_path=cfg.clock_stats_path, mode=p.clock_mode,
            condition_on_mate_n=p.clock_condition_on_mate_n, min_seconds=p.clock_min_seconds,
            cap_seconds=p.clock_cap_seconds, avg_time_by_rating=avg_time,
        ),
        debug_dir=cfg.puzzles_dir, save_debug_jsonl=p.save_debug_jsonl,
    )


def step_puzzles(ctx: Context) -> Dict[str, Any]:
    cfg = ctx.cfg
    if ctx.skip_if_done("puzzle_pipeline"):
        logger.info("[puzzles] gia' completato: skip.")
        return ctx.state.meta("puzzle_pipeline")
    csv_path = _decompress_puzzles(cfg)
    if csv_path is None:
        logger.warning("[puzzles] raw_data.puzzles_zst vuoto: skip.")
        return {}
    if not os.path.exists(cfg.clock_stats_path):
        logger.warning("[puzzles] %s assente: fallback su avg_time_by_rating.", cfg.clock_stats_path)

    def run():
        builder = PuzzleBuilder(make_puzzle_config(cfg, csv_path, ctx.avg_time), ctx.spool)
        ctx.puzzle_builder = builder
        result = builder.run()
        ctx.state.mark_done("puzzle_pipeline", **_scalars(result))
        logger.info("[puzzles] processed=%s accepted=%s enqueued=%s",
                    result["processed_puzzles"], result["accepted_puzzles"], result["enqueued_positions"])
        return result

    return _guard(ctx, "puzzle_pipeline", run)


# ----------------------------------------------------------------- finalize
class _ShardWriter:
    def __init__(self, out_dir: str, split: str, shard_size: int) -> None:
        self.dir = os.path.join(out_dir, split)
        self._reset_dir()
        self.shard_size = max(1, int(shard_size))
        self.buf: List[Any] = []
        self.idx = 0
        self.files: List[str] = []
        self.total = 0

    def _reset_dir(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        for n in os.listdir(self.dir):
            if n.startswith("shard_") or n in ("manifest.json",):
                os.remove(os.path.join(self.dir, n))

    def append(self, rec: Any) -> None:
        self.buf.append(rec)
        self.total += 1
        if len(self.buf) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        import torch
        path = os.path.join(self.dir, f"shard_{self.idx:05d}.pt")
        torch.save(self.buf, path + ".tmp")
        os.replace(path + ".tmp", path)
        self.files.append(path)
        self.buf = []
        self.idx += 1

    def close(self) -> List[str]:
        from common.io import atomic_write_json
        self._flush()
        atomic_write_json(os.path.join(self.dir, "manifest.json"),
                          {"num_shards": self.idx, "shard_size": self.shard_size, "total": self.total})
        return self.files


def step_finalize(ctx: Context, mark_done: bool = True) -> Dict[str, Any]:
    cfg = ctx.cfg
    if ctx.skip_if_done("finalize_splits"):
        logger.info("[finalize] gia' completato: skip.")
        return ctx.state.meta("finalize_splits")

    def run():
        spool, sp = ctx.spool, cfg.splits
        os.makedirs(cfg.merged_dir, exist_ok=True)

        logger.info("[finalize] pass 1/2: assegnazione split.")
        assignment = spool.build_split_assignment(sp.ratios, cfg.pipeline.seed)

        logger.info("[finalize] pass 2/2: scrittura shard (size=%d).", sp.output_shard_size)
        writers = {n: _ShardWriter(cfg.merged_dir, n, sp.output_shard_size) for n in ("train", "val", "test")}
        for split, data in spool.iter_positions(assignment):
            writers[split].append(data)

        meta: Dict[str, Any] = {}
        for name, w in writers.items():
            files = w.close()
            meta[name] = w.total
            logger.info("[finalize] %s: %d posizioni in %d shard.", name, w.total, len(files))

        _finalize_debug(cfg, ctx, assignment)
        spool.clear()
        if mark_done: 
            ctx.state.mark_done("finalize_splits", **meta)
        return meta

    return _guard(ctx, "finalize_splits", run)


def _finalize_debug(cfg: Config, ctx: Context, assignment: Dict[str, str]) -> None:
    from common.io import finalize_csv
    for d, pending, final in (
        (cfg.games_dir, "games_debug_records.pending.csv", "games_debug.csv"),
        (cfg.puzzles_dir, "puzzle_debug_records.pending.csv", "puzzle_debug.csv"),
    ):
        finalize_csv(os.path.join(d, pending), os.path.join(d, final), assignment, DEBUG_FIELDS)


# -------------------------------------------------------------------- clean
def step_clean(ctx: Context) -> Dict[str, Any]:
    cfg, c = ctx.cfg, ctx.cfg.clean
    if not c.enabled:
        logger.info("[clean] disabilitato: skip.")
        return {}

    from Cleaner.CleanDataset import clean_sharded_directory  # esterno al package

    pairs = [("train", c.input_dir_train, c.output_dir_train), ("val", c.input_dir_val, c.output_dir_val)]
    if c.input_dir_test and c.output_dir_test:
        pairs.append(("test", c.input_dir_test, c.output_dir_test))
    for name, din, dout in pairs:
        if not din or not dout:
            raise ConfigError(f"[clean] input/output {name} mancante.")
        _require_file(os.path.join(din, "manifest.json"), f"[clean] manifest {name}")

    ready = all(os.path.exists(os.path.join(o, "manifest.json")) for _, _, o in pairs)
    if ctx.skip_if_done("clean") and ready:
        logger.info("[clean] gia' completato: skip.")
        return ctx.state.meta("clean")

    def run():
        meta: Dict[str, Any] = {"target_shard_size": c.target_shard_size}
        for name, din, dout in pairs:
            logger.info("[clean][%s] %s -> %s", name, din, dout)
            m = clean_sharded_directory(din, dout, c.target_shard_size, c.workers)
            meta[f"{name}_total"] = m.get("total", 0)
            meta[f"{name}_shards"] = m.get("num_shards", 0)
            _free_memory()
        ctx.state.mark_done("clean", **meta)
        return meta

    return _guard(ctx, "clean", run)


STEPS: Dict[str, Callable[[Context], Dict[str, Any]]] = {
    "time_stats": step_time_stats,
    "games_pipeline": step_games,
    "clock_stats": step_clock_stats,
    "puzzle_pipeline": step_puzzles,
    "finalize_splits": step_finalize,
    "clean": step_clean,
}