from __future__ import annotations

import copy
import dataclasses
import logging
import multiprocessing as mp
import os
import signal
import time
from collections import defaultdict
from typing import Any, Dict, Generator, Optional, Tuple

from common.game_id_store import GameIdStore
from common.io import CsvAppender, finalize_csv
from common.progress import wrap_iter
from spool.position_queue import PositionSpool
from utils.ipc import decode_from_ipc, harden_process_for_ipc

from .config import GamesBuilderConfig
from .pgn_io import iter_source_with_site_id
from .resume import ResumeTracker
from .worker import Task, pool_initializer, worker_entry

logger = logging.getLogger(__name__)

_DEBUG_FIELDS = ["problem_id", "game_id", "fen", "best_move_uci", "mate_n", "mate_n_window",
                  "ply", "source", "clock_source", "clock_seconds", "clock_is_real", "rating"]


def _kill_pool_processes(pool) -> None:
    for proc in getattr(pool, "_pool", []):
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _shutdown_pool(pool, graceful: bool, join_timeout: float) -> None:
    try:
        if graceful:
            deadline = time.monotonic() + join_timeout
            for p in pool._pool:
                p.join(timeout=max(deadline - time.monotonic(), 0.0))
        pool.terminate()
        deadline = time.monotonic() + 5.0
        for p in pool._pool:
            p.join(timeout=max(deadline - time.monotonic(), 0.1))
        for p in pool._pool:
            if p.is_alive():
                logger.warning("[games] worker pid=%s vivo dopo terminate(): SIGKILL.", p.pid)
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for p in pool._pool:
            p.join(timeout=2.0)
    except Exception:
        logger.exception("[games] errore shutdown pool: SIGKILL su tutti.")
        _kill_pool_processes(pool)


class GamesBuilder:
    def __init__(self, config: GamesBuilderConfig, spool: PositionSpool) -> None:
        config = dataclasses.replace(config, sources=[copy.copy(s) for s in config.sources])
        config.validate()
        self.config = config
        self.spool = spool
        self._workers = config.workers or max(1, (os.cpu_count() or 2) - 1)
        self._resume = ResumeTracker(config.resume_state_path, config.sources, enabled=config.auto_resume)

        self._dedup: Optional[GameIdStore] = None
        if config.dedupe_cross_file:
            self._dedup = GameIdStore(config.game_id_store_path)

        self.debug: Optional[CsvAppender] = None
        if config.save_debug_jsonl:
            d = config.debug_dir or os.path.dirname(os.path.abspath(config.resume_state_path))
            os.makedirs(d, exist_ok=True)
            self.debug = CsvAppender(
                os.path.join(d, "games_debug_records.pending.csv"),
                os.path.join(d, "games_debug.csv"),
                _DEBUG_FIELDS,
            )

    def _iter_tasks(self) -> Generator[Task, None, None]:
        skipped_dupe = 0
        for src in self.config.sources:
            for local_id, text, site_id in iter_source_with_site_id(src):
                if self._dedup is not None and site_id is not None:
                    if self._dedup.contains(site_id):
                        skipped_dupe += 1
                        continue
                    self._dedup.add(site_id)
                yield (local_id, text, src.tag, src.resume_key)
        if self._dedup is not None and skipped_dupe:
            logger.info("[games] dedup cross-file: %d partite gia' viste, scartate senza analisi.", skipped_dupe)

    def _estimate(self) -> Optional[int]:
        total = 0
        for s in self.config.sources:
            if s.max_games is None:
                return None
            total += s.max_games
        return total

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        harden_process_for_ipc()

        processed = accepted = enqueued = parent_errors = 0
        mate_counts: Dict[int, int] = defaultdict(int)
        source_counts: Dict[str, int] = defaultdict(int)
        clock_counts: Dict[str, int] = defaultdict(int)

        pool = mp.Pool(self._workers, initializer=pool_initializer, initargs=(cfg,))
        interrupted = {"flag": False}

        def _sigint(signum, frame):
            if interrupted["flag"]:
                _kill_pool_processes(pool)
                os._exit(1)
            interrupted["flag"] = True
            raise KeyboardInterrupt

        prev_sigint = signal.signal(signal.SIGINT, _sigint)
        last_flush = time.monotonic()

        try:
            results = pool.imap_unordered(worker_entry, self._iter_tasks(), chunksize=8)
            for local_id, resume_key, payload in wrap_iter(
                results, desc="[GamesBuilder] Analisi partite", unit="game", total=self._estimate()
            ):
                processed += 1
                self._resume.mark_done(resume_key, local_id)

                if processed % cfg.resume_checkpoint_every == 0:
                    if self.debug:
                        self.debug.persist()
                    if cfg.auto_resume:
                        self._resume.persist()
                    if self._dedup is not None:
                        self._dedup.commit()
                if cfg.flush_every_seconds and time.monotonic() - last_flush >= cfg.flush_every_seconds:
                    self.spool.flush()
                    last_flush = time.monotonic()

                try:
                    records = decode_from_ipc(payload)
                except Exception:
                    parent_errors += 1
                    logger.error("[games] decode fallito (%s, #%d).", resume_key, processed, exc_info=True)
                    continue
                if not records:
                    continue

                try:
                    for rec in records:
                        dbg = rec["debug"]
                        self.spool.enqueue(dbg["source"], rec["data"], dbg["mate_n_window"])
                        enqueued += 1
                        source_counts[dbg["source"]] += 1
                        clock_counts[dbg["clock_source"]] += 1
                        mate_counts[dbg["mate_n"]] += 1
                        if self.debug:
                            self.debug.add(dbg)
                    accepted += 1
                except Exception:
                    parent_errors += 1
                    logger.error("[games] enqueue fallito (%s, #%d).", resume_key, processed, exc_info=True)

        except KeyboardInterrupt:
            print("\n[WARNING] Interruzione: arresto worker...")
            _shutdown_pool(pool, graceful=False, join_timeout=cfg.pool_join_timeout)
            raise
        except Exception:
            logger.exception("[games] errore fatale: arresto worker.")
            _shutdown_pool(pool, graceful=False, join_timeout=cfg.pool_join_timeout)
            raise
        else:
            pool.close()
            _shutdown_pool(pool, graceful=True, join_timeout=cfg.pool_join_timeout)
        finally:
            self._resume.persist()
            if self._dedup is not None:
                self._dedup.close()
            signal.signal(signal.SIGINT, prev_sigint)

        self.spool.flush()
        if self.debug:
            self.debug.persist()
        if parent_errors:
            logger.warning("[games] %d game scartati per errori lato padre.", parent_errors)

        return {
            "processed_games": processed,
            "accepted_games": accepted,
            "enqueued_positions": enqueued,
            "mate_n_counts": dict(mate_counts),
            "source_counts": dict(source_counts),
            "clock_source_counts": dict(clock_counts),
            "skipped_games_on_parent_error": parent_errors,
        }

    def finalize_debug(self, assignment: Dict[str, str]) -> Optional[str]:
        if not self.debug:
            return None
        self.debug.persist()
        return finalize_csv(self.debug.pending_path, self.debug.final_path, assignment, self.debug.fieldnames)