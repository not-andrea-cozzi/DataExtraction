from __future__ import annotations

import atexit
import csv
import dataclasses
import functools
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chess
import numpy as np

from core.move_codec import decode_move
from train.dataset import ChessShardDataset

logger = logging.getLogger("evaluate_llm")


# ============================================================================
# Config hardcodata
# ============================================================================
@dataclass
class LLMConfig:
    enabled: bool = True
    holdout_dir: str = "../Dataset/HeldOut/heldout_clean"
    out_dir: str = "../Dataset/runs/evaluate/llm"
    max_n: int = 10
    limit: Optional[int] = None
    max_workers: int = 1

    llm_base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key_env: str = "GROQ_API_KEY"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 32
    llm_reasoning_effort: Optional[str] = None
    llm_reasoning_format: Optional[str] = None
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3
    llm_retry_backoff_seconds: float = 2.0
    llm_request_delay_seconds: float = 0.5


def as_config(cfg: Union[LLMConfig, Dict[str, Any]]) -> LLMConfig:
    """Accetta LLMConfig o dict (compat con EvalConfig.llm)."""
    if isinstance(cfg, LLMConfig):
        return cfg
    valid = {f.name for f in dataclasses.fields(LLMConfig)}
    return LLMConfig(**{k: v for k, v in (cfg or {}).items() if k in valid})


# ============================================================================
# Utils
# ============================================================================
def _atomic_json_dump(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
_SAN_RE = re.compile(r"\b(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b")


def _build_llm_prompt(fen: str) -> str:
    return (
        "You are a chess engine. Analyze the position given below (in FEN "
        "notation) and find the best move for the side to move. Respond "
        "with ONLY the move in UCI notation: the origin square followed by "
        "the destination square (both as file-letter + rank-number, "
        "lowercase), optionally followed by a promotion piece letter if "
        "promoting a pawn. Do NOT use algebraic/SAN notation (no piece "
        "letters like 'R' or 'N', no '+', no '#', no 'x'). Do not repeat "
        "this instruction or give an example: compute the move for THIS "
        "exact position and output only that move, nothing else.\n\n"
        f"FEN: {fen}\n"
        "Best move (UCI):"
    )


def _extract_uci(text: str) -> Optional[str]:
    if not text:
        return None
    m = _UCI_RE.search(text.strip())
    return m.group(1).lower() if m else None


def _try_parse_move(text: str, board: "chess.Board") -> str:
    if not text:
        return ""
    cand = _extract_uci(text)
    if cand:
        try:
            mv = chess.Move.from_uci(cand)
            if mv in board.legal_moves:
                return mv.uci()
        except ValueError:
            pass
    for san in _SAN_RE.findall(text):
        try:
            return board.parse_san(san).uci()
        except ValueError:
            continue
    return ""


# ============================================================================
# Solver
# ============================================================================
class GroqLLMSolver:
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.0,
                 max_tokens: int = 32, reasoning_effort: Optional[str] = None,
                 reasoning_format: Optional[str] = None, timeout_seconds: float = 30.0,
                 max_retries: int = 3, retry_backoff_seconds: float = 2.0,
                 request_delay_seconds: float = 0.0) -> None:
        import requests

        self._requests = requests
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning_format = reasoning_format
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.request_delay_seconds = request_delay_seconds

    def solve(self, fen: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": _build_llm_prompt(fen)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_format:
            payload["reasoning_format"] = self.reasoning_format
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._requests.post(self.base_url, headers=headers, json=payload,
                                           timeout=self.timeout_seconds)
                if resp.status_code == 429:
                    raise RuntimeError(f"rate limited (429): {resp.text[:200]}")
                resp.raise_for_status()
                message = resp.json()["choices"][0]["message"]
                raw_text = message.get("content", "") or ""
                pred = _extract_uci(raw_text)
                if not pred:
                    reasoning = message.get("reasoning", "") or ""
                    if reasoning:
                        raw_text, pred = reasoning, _extract_uci(reasoning)
                if self.request_delay_seconds > 0:
                    time.sleep(self.request_delay_seconds)
                return {"raw_text": raw_text, "pred_move_uci": pred or "", "error": None}
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.debug("[solve] attempt %d/%d fallita: %s", attempt, self.max_retries, last_err)
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
        return {"raw_text": "", "pred_move_uci": "", "error": last_err or "unknown error"}


def _build_solver_factory_from_cfg(cfg: Union[LLMConfig, Dict[str, Any]], api_key: str
                                   ) -> Callable[[], GroqLLMSolver]:
    c = as_config(cfg)
    return functools.partial(
        GroqLLMSolver,
        base_url=c.llm_base_url, model=c.llm_model, api_key=api_key,
        temperature=c.llm_temperature, max_tokens=c.llm_max_tokens,
        reasoning_effort=c.llm_reasoning_effort, reasoning_format=c.llm_reasoning_format,
        timeout_seconds=c.llm_timeout_seconds, max_retries=c.llm_max_retries,
        retry_backoff_seconds=c.llm_retry_backoff_seconds,
        request_delay_seconds=c.llm_request_delay_seconds,
    )


# ============================================================================
# Worker multiprocessing
# ============================================================================
_W_SOLVER: Optional[GroqLLMSolver] = None
_W_CACHE: Dict[str, Dict[str, Any]] = {}
_W_CACHE_PATH: Optional[str] = None
_W_CACHE_DIRTY = 0
_FLUSH_EVERY = 5


def _flush_worker_cache() -> None:
    global _W_CACHE_DIRTY
    if not _W_CACHE_PATH or _W_CACHE_DIRTY == 0:
        return
    try:
        _atomic_json_dump(f"{_W_CACHE_PATH}.{os.getpid()}", _W_CACHE)
        _W_CACHE_DIRTY = 0
    except Exception:
        logger.warning("[worker %s] flush cache fallito", os.getpid(), exc_info=True)


def _init_llm_worker(solver_factory: Optional[Callable[[], GroqLLMSolver]],
                     cache_path: Optional[str]) -> None:
    global _W_SOLVER, _W_CACHE, _W_CACHE_PATH, _W_CACHE_DIRTY
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    atexit.register(_flush_worker_cache)
    _W_CACHE, _W_CACHE_PATH, _W_CACHE_DIRTY = {}, cache_path, 0

    if solver_factory is not None:
        try:
            _W_SOLVER = solver_factory()
        except Exception:
            logger.exception("[worker %s] inizializzazione solver FALLITA", os.getpid())
            _W_SOLVER = None
    else:
        _W_SOLVER = None

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                _W_CACHE = json.load(f)
        except Exception:
            _W_CACHE = {}


def _worker_evaluate_one(args: Tuple[str, str]) -> Tuple[str, str]:
    global _W_CACHE_DIRTY
    pid, fen = args

    if _W_SOLVER is None:
        _W_CACHE[pid] = {"raw_text": "", "pred_move_uci": "", "error": "solver_not_initialized_in_worker"}
        _W_CACHE_DIRTY += 1
        if _W_CACHE_DIRTY >= _FLUSH_EVERY:
            _flush_worker_cache()
        return pid, ""

    result = _W_CACHE.get(pid)
    if not isinstance(result, dict):
        try:
            result = _W_SOLVER.solve(fen)
        except Exception as e:
            result = {"raw_text": "", "pred_move_uci": "", "error": f"{type(e).__name__}: {e}"}
        _W_CACHE[pid] = result
        _W_CACHE_DIRTY += 1
        if _W_CACHE_DIRTY >= _FLUSH_EVERY:
            _flush_worker_cache()

    pred = result.get("pred_move_uci") or ""
    if not pred:
        raw = result.get("raw_text", "") or ""
        if raw:
            try:
                pred = _try_parse_move(raw, chess.Board(fen))
            except Exception:
                pred = ""
    if not pred:
        logger.warning("[worker %s] pid=%s pred vuota | error=%r | raw[:100]=%r",
                       os.getpid(), pid, result.get("error"), (result.get("raw_text") or "")[:100])
    return pid, pred


def _merge_worker_caches(cache_path: str, base: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    files = [p for p in glob.glob(f"{cache_path}.*") if not p.endswith((".tmp", ".corrupt"))]
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                merged.update(json.load(f))
            os.remove(path)
        except Exception as e:
            logger.warning("[LLM] Impossibile leggere %s: %s", path, e)
    return merged


# ============================================================================
# Valutazione
# ============================================================================
def evaluate_llm_on_holdout(
    holdout_dir: str,
    solver_factory: Optional[Callable[[], GroqLLMSolver]],
    cache_path: Optional[str] = None,
    max_n: int = 10,
    limit: Optional[int] = None,
    max_workers: int = 1,
    pool_join_timeout: float = 60.0,
) -> Dict[str, Any]:
    if not os.path.exists(os.path.join(holdout_dir, "manifest.json")):
        raise FileNotFoundError(f"manifest.json non trovato in '{holdout_dir}': esegui heldout.py.")

    ds = ChessShardDataset(holdout_dir)

    base_cache: Dict[str, Any] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                base_cache = json.load(f)
            logger.info("Cache LLM: %d risposte da %s.", len(base_cache), cache_path)
        except Exception as e:
            logger.warning("Cache illeggibile (%s): spostata in .corrupt.", e)
            try:
                os.replace(cache_path, cache_path + ".corrupt")
            except OSError:
                pass

    items: List[Tuple[str, str, str, int]] = []
    missing_fen = 0
    n_total = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n_total):
        data = ds[i]
        fen = getattr(data, "fen", None)
        if fen is None:
            missing_fen += 1
            continue
        frm, to, promo = decode_move(int(data.y))
        true_uci = chess.Move(frm, to, promotion=promo).uci()
        items.append((str(i), fen, true_uci, int(getattr(data, "position_mate_n", 0))))
    if missing_fen:
        logger.warning("[LLM] %d posizioni senza 'fen' saltate.", missing_fen)

    results_by_pid: Dict[str, str] = {}
    total = len(items)

    # ---- sequenziale ----
    if solver_factory is None or max_workers <= 1:
        solver = solver_factory() if solver_factory is not None else None
        cache = dict(base_cache)
        for n_done, (pid, fen, _, _) in enumerate(items, 1):
            if solver is None:
                pred = ""
            else:
                result = cache.get(pid)
                if not isinstance(result, dict):
                    try:
                        result = solver.solve(fen)
                    except Exception as e:
                        result = {"raw_text": "", "pred_move_uci": "", "error": f"{type(e).__name__}: {e}"}
                    cache[pid] = result
                    if cache_path and n_done % 10 == 0:
                        _atomic_json_dump(cache_path, cache)
                pred = result.get("pred_move_uci") or ""
                if not pred:
                    try:
                        pred = _try_parse_move(result.get("raw_text", "") or "", chess.Board(fen))
                    except Exception:
                        pred = ""
            results_by_pid[pid] = pred
            if n_done % 5 == 0 or n_done == total:
                logger.info("[LLM] valutati %d/%d.", n_done, total)
        if cache_path:
            _atomic_json_dump(cache_path, cache)

    # ---- multiprocessing ----
    else:
        workers = min(max_workers, os.cpu_count() or 2)
        logger.info("[LLM] Pool con %d worker.", workers)
        pool = mp.Pool(processes=workers, initializer=_init_llm_worker,
                       initargs=(solver_factory, cache_path))
        prev_sigint = signal.signal(signal.SIGINT, signal.default_int_handler)
        done = 0
        graceful = False
        try:
            for pid, pred in pool.imap_unordered(_worker_evaluate_one,
                                                 ((p, f) for p, f, _, _ in items), chunksize=1):
                results_by_pid[pid] = pred
                done += 1
                if done % 5 == 0 or done == total:
                    logger.info("[LLM] valutati %d/%d.", done, total)
            graceful = True
        except KeyboardInterrupt:
            logger.warning("[LLM] Interruzione (%d/%d): salvo risultati parziali.", done, total)
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            try:
                if graceful:
                    pool.close()
                    pool.join()
                else:
                    pool.terminate()
                    deadline = time.monotonic() + 5.0
                    for p in pool._pool:
                        p.join(timeout=max(deadline - time.monotonic(), 0.1))
                    for p in pool._pool:
                        if p.is_alive():
                            try:
                                os.kill(p.pid, signal.SIGKILL)
                            except Exception:
                                pass
            except Exception:
                logger.exception("[LLM] Errore shutdown pool.")

        if cache_path:
            base_cache = _merge_worker_caches(cache_path, base_cache)
            for pid, pred in results_by_pid.items():
                if pid not in base_cache:
                    base_cache[pid] = {"pred_move_uci": pred}
            _atomic_json_dump(cache_path, base_cache)

    # ---- metriche ----
    pids = [p for p, _, _, _ in items]
    mate_arr = np.array([m for _, _, _, m in items], dtype=np.int64)
    best = [t for _, _, t, _ in items]
    pred = [results_by_pid.get(p, "") for p in pids]
    correct = np.array([a == b for a, b in zip(pred, best)], dtype=bool)

    n_empty = sum(1 for p in pred if not p)
    if n_empty:
        logger.error("[LLM] %d/%d predizioni VUOTE: controlla 'error'/'raw_text' in cache.", n_empty, len(pred))

    per_n = []
    for n in range(1, max_n + 1):
        mask = mate_arr == n
        cnt = int(mask.sum())
        per_n.append({"mate_n": n, "count": cnt,
                      "move_accuracy": float(correct[mask].mean()) if cnt else None})

    return {
        "problem_id": np.array(pids, dtype=object),
        "mate_n": mate_arr,
        "best_move_uci": np.array(best, dtype=object),
        "pred_move_uci": np.array(pred, dtype=object),
        "move_correct": correct,
        "per_n": per_n,
    }


def save_per_n_csv(per_n_rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mate_n", "count", "move_accuracy"])
        w.writeheader()
        w.writerows(per_n_rows)
    logger.info("Salvato %s", path)