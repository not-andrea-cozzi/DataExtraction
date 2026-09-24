from __future__ import annotations

import atexit
import os
import signal
import threading
import time
from typing import Optional

import chess
import chess.engine
import chess.syzygy

from Utils.ipc_safe_data import harden_process_for_ipc

from .config import GamesBuilderConfig

_engine: Optional[chess.engine.SimpleEngine] = None
_engine_pid: Optional[int] = None
_tablebase: Optional["chess.syzygy.Tablebase"] = None

_watchdog_lock = threading.Lock()
_watchdog_deadline: Optional[float] = None
_watchdog_stop = threading.Event()
_watchdog_thread: Optional[threading.Thread] = None
_WATCHDOG_POLL_SECONDS = 1.0
_WATCHDOG_MARGIN_SECONDS = 3.0


def is_engine_ready() -> bool:
    return _engine is not None


def _watchdog_arm(time_limit: float, margin: Optional[float] = None) -> None:
    global _watchdog_deadline
    eff_margin = _WATCHDOG_MARGIN_SECONDS if margin is None else margin
    with _watchdog_lock:
        _watchdog_deadline = time.monotonic() + time_limit + eff_margin


def _watchdog_disarm() -> None:
    global _watchdog_deadline
    with _watchdog_lock:
        _watchdog_deadline = None


def _watchdog_loop() -> None:
    global _engine, _engine_pid, _watchdog_deadline
    while not _watchdog_stop.is_set():
        with _watchdog_lock:
            deadline = _watchdog_deadline
        if deadline is not None and time.monotonic() > deadline:
            pid = _engine_pid
            if pid is not None:
                try:
                    import psutil
                    psutil.Process(pid).kill()
                except Exception:
                    try:
                        os.kill(pid, 9)
                    except Exception:
                        pass
            _engine = None
            _engine_pid = None
            with _watchdog_lock:
                _watchdog_deadline = None
        _watchdog_stop.wait(_WATCHDOG_POLL_SECONDS)


def _close_engine() -> None:
    global _engine, _engine_pid, _tablebase
    _watchdog_stop.set()
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        finally:
            _engine = None
            _engine_pid = None
    if _tablebase is not None:
        try:
            _tablebase.close()
        except Exception:
            pass
        finally:
            _tablebase = None


def _worker_sigterm_handler(signum, frame) -> None:
    _watchdog_stop.set()
    pid = _engine_pid
    if pid is not None:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    if _tablebase is not None:
        try:
            _tablebase.close()
        except Exception:
            pass
    os._exit(0)


def init_worker(stockfish_path: str, threads: int, hash_mb: int, syzygy_path: Optional[str]) -> None:
    global _engine, _engine_pid, _tablebase, _watchdog_thread

    harden_process_for_ipc()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _worker_sigterm_handler)

    try:
        os.setpgrp()
    except AttributeError:
        pass

    try:
        _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        _engine.configure({"Threads": threads, "Hash": hash_mb})
        try:
            _engine_pid = _engine.transport.get_pid()
        except Exception:
            _engine_pid = None
    except Exception as e:
        _engine = None
        _engine_pid = None
        raise RuntimeError(f"Impossibile avviare Stockfish: {e}")

    if syzygy_path:
        try:
            _tablebase = chess.syzygy.open_tablebase(syzygy_path)
        except Exception:
            _tablebase = None

    atexit.register(_close_engine)

    _watchdog_stop.clear()
    _watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
    _watchdog_thread.start()


def syzygy_says_no_mate(board: "chess.Board") -> bool:
    if _tablebase is None:
        return False
    if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
        return False
    try:
        wdl = _tablebase.probe_wdl(board)
    except Exception:
        return False
    return wdl is not None and wdl <= 0


def analyse_position(board: "chess.Board", cfg: GamesBuilderConfig):
    multipv = max(2, cfg.multipv)

    for attempt in range(1, cfg.stockfish_retry_attempts + 1):
        if _engine is None:
            return None
        try:
            if cfg.analysis_time is not None:
                limit = chess.engine.Limit(time=cfg.analysis_time)
                _watchdog_arm(cfg.analysis_time)
            else:
                limit = chess.engine.Limit(depth=cfg.search_depth)
                _watchdog_arm(10.0)
            return _engine.analyse(board, limit, multipv=multipv)
        except Exception:
            if attempt >= cfg.stockfish_retry_attempts:
                return None
            if cfg.stockfish_retry_backoff_seconds > 0:
                time.sleep(cfg.stockfish_retry_backoff_seconds * attempt)
            continue
        finally:
            _watchdog_disarm()
    return None


def second_line_ties_mate(info, mate_n: int) -> bool:
    if len(info) < 2:
        return False
    second_score = info[1].get("score")
    if second_score is None:
        return False
    second_rel = second_score.relative
    if not second_rel.is_mate():
        return False
    second_mate = second_rel.mate()
    return second_mate is not None and 0 < second_mate <= mate_n