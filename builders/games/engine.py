from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
import time
from typing import Optional

import chess
import chess.engine
import chess.syzygy

from utils.ipc import harden_process_for_ipc

from .config import EngineConfig

logger = logging.getLogger(__name__)
_WATCHDOG_POLL = 1.0
_WATCHDOG_MARGIN = 3.0
_DEPTH_TIMEOUT = 10.0


class Engine:
    """Stockfish per-worker con watchdog e RESTART automatico se killato."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._pid: Optional[int] = None
        self._tb: Optional["chess.syzygy.Tablebase"] = None
        self._lock = threading.Lock()
        self._deadline: Optional[float] = None
        self._killed = False
        self._stop = threading.Event()
        self._start_engine()
        if cfg.syzygy_path:
            try:
                self._tb = chess.syzygy.open_tablebase(cfg.syzygy_path)
            except Exception as e:
                logger.warning("Syzygy non aperto (%s): disabilitato.", e)
        self._thread = threading.Thread(target=self._watchdog, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _start_engine(self) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(self.cfg.stockfish_path)
        self._engine.configure({"Threads": self.cfg.threads, "Hash": self.cfg.hash_mb})
        try:
            self._pid = self._engine.transport.get_pid()
        except Exception:
            self._pid = None

    def _kill_pid(self) -> None:
        if self._pid is None:
            return
        try:
            os.kill(self._pid, signal.SIGKILL)
        except Exception:
            pass

    def _watchdog(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                overdue = self._deadline is not None and time.monotonic() > self._deadline
                if overdue:
                    self._kill_pid()
                    self._killed = True
                    self._deadline = None
            self._stop.wait(_WATCHDOG_POLL)

    def _restart_if_killed(self) -> bool:
        with self._lock:
            if not self._killed:
                return self._engine is not None
            self._killed = False
        try:
            if self._engine is not None:
                try:
                    self._engine.close()
                except Exception:
                    pass
            self._start_engine()
            logger.warning("Stockfish riavviato dopo timeout watchdog.")
            return True
        except Exception as e:
            logger.error("Restart Stockfish fallito: %s", e)
            self._engine = None
            return False

    def is_ready(self) -> bool:
        return self._engine is not None

    def analyse(self, board: "chess.Board"):
        if not self._restart_if_killed():
            return None
        cfg = self.cfg
        multipv = max(2, cfg.multipv)
        for attempt in range(1, cfg.retry_attempts + 1):
            try:
                if cfg.analysis_time is not None:
                    limit = chess.engine.Limit(time=cfg.analysis_time)
                    timeout = cfg.analysis_time
                else:
                    limit = chess.engine.Limit(depth=cfg.search_depth)
                    timeout = _DEPTH_TIMEOUT
                with self._lock:
                    self._deadline = time.monotonic() + timeout + _WATCHDOG_MARGIN
                return self._engine.analyse(board, limit, multipv=multipv)
            except Exception:
                if not self._restart_if_killed() or attempt >= cfg.retry_attempts:
                    return None
                time.sleep(cfg.retry_backoff_seconds * attempt)
            finally:
                with self._lock:
                    self._deadline = None
        return None

    def syzygy_says_no_mate(self, board: "chess.Board") -> bool:
        if self._tb is None or board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
            return False
        try:
            wdl = self._tb.probe_wdl(board)
        except Exception:
            return False
        return wdl is not None and wdl <= 0

    def close(self) -> None:
        self._stop.set()
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
        if self._tb is not None:
            try:
                self._tb.close()
            except Exception:
                pass
            self._tb = None


def second_line_ties_mate(info, mate_n: int) -> bool:
    if len(info) < 2:
        return False
    score = info[1].get("score")
    if score is None:
        return False
    rel = score.relative
    if not rel.is_mate():
        return False
    m = rel.mate()
    return m is not None and 0 < m <= mate_n


def install_worker_signals(engine_getter) -> None:
    """SIGINT ignorato (lo gestisce il padre); SIGTERM chiude Stockfish e esce."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _term(signum, frame):
        eng = engine_getter()
        if eng is not None:
            eng.close()
        os._exit(0)

    signal.signal(signal.SIGTERM, _term)
    try:
        os.setpgrp()
    except AttributeError:
        pass


def init_process() -> None:
    harden_process_for_ipc()
