from __future__ import annotations

import asyncio
import csv
import dataclasses
import functools
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chess
import httpx
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
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key_env: str = "GROQ_API_KEY"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_reasoning_effort: Optional[str] = "low"
    llm_reasoning_format: Optional[str] = None
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 10
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
# Solver (async, httpx)
# ============================================================================
class GroqLLMSolver:
    """Contenitore dei parametri di chiamata; la chiamata vera e' in _solve_async."""

    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.0,
                 max_tokens: int = 32, reasoning_effort: Optional[str] = None,
                 reasoning_format: Optional[str] = None, timeout_seconds: float = 30.0,
                 max_retries: int = 3, retry_backoff_seconds: float = 2.0,
                 request_delay_seconds: float = 0.0) -> None:
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


_MAX_PAUSE_SECONDS = 300.0  # Retry-After oltre questa soglia = quota giornaliera, non TPM: si interrompe


def _retry_after(resp: "httpx.Response", fallback: float) -> float:
    try:
        return float(resp.headers.get("retry-after", "")) + 0.5
    except ValueError:
        return fallback


def _parse_duration(v: str) -> float:
    """'7.66s' / '1m2.5s' -> secondi. Formato non riconosciuto (es. 'ms'): 5 s prudenziali."""
    m = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", (v or "").strip())
    if not m or not (m.group(1) or m.group(2)):
        return 5.0
    return int(m.group(1) or 0) * 60 + float(m.group(2) or 0)


class _Throttle:
    """Rate limiter condiviso: distanzia l'inizio delle richieste e permette pause globali."""

    def __init__(self, interval: float) -> None:
        self.interval = max(0.0, interval)
        self._lock = asyncio.Lock()
        self._next = 0.0
        self._avg_tokens: Optional[float] = None
        self.aborted: Optional[str] = None

    def record(self, total_tokens: float) -> None:
        """Media mobile dei token realmente consumati per richiesta."""
        self._avg_tokens = total_tokens if self._avg_tokens is None else 0.8 * self._avg_tokens + 0.2 * total_tokens

    def cost(self, default: float) -> float:
        return default if self._avg_tokens is None else self._avg_tokens * 1.5

    async def wait(self) -> None:
        async with self._lock:
            while True:
                delay = self._next - time.monotonic()
                if delay <= 0:
                    break
                await asyncio.sleep(delay)
            self._next = time.monotonic() + self.interval

    def pause(self, seconds: float) -> None:
        self._next = max(self._next, time.monotonic() + seconds)

    def observe(self, resp: "httpx.Response", cost: float) -> None:
        """Se i token residui nella finestra sono meno del costo stimato, pausa fino al reset."""
        try:
            remaining = float(resp.headers.get("x-ratelimit-remaining-tokens", ""))
        except ValueError:
            return
        if remaining < cost:
            self.pause(_parse_duration(resp.headers.get("x-ratelimit-reset-tokens", "")) + 0.5)


async def _solve_async(s: GroqLLMSolver, client: "httpx.AsyncClient",
                       sem: asyncio.Semaphore, fen: str, throttle: _Throttle) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": s.model,
        "temperature": s.temperature,
        "max_tokens": s.max_tokens,
        "messages": [{"role": "user", "content": _build_llm_prompt(fen)}],
    }
    if s.reasoning_effort:
        payload["reasoning_effort"] = s.reasoning_effort
    if s.reasoning_format:
        payload["reasoning_format"] = s.reasoning_format
    headers = {"Authorization": f"Bearer {s.api_key}"}

    err: Optional[str] = None
    raw_last = ""
    reasoning_last = ""
    finish: Optional[str] = None
    async with sem:
        for attempt in range(1, s.max_retries + 1):
            if throttle.aborted:
                return {"raw_text": "", "pred_move_uci": "", "error": f"aborted: {throttle.aborted}"}
            try:
                await throttle.wait()
                resp = await client.post(s.base_url, headers=headers, json=payload)
                throttle.observe(resp, throttle.cost(s.max_tokens + 300))
                if resp.status_code == 429:
                    err = "429: rate limited"
                    wait = _retry_after(resp, s.retry_backoff_seconds * attempt)
                    logger.warning(
                        "[LLM] 429, attesa %.0fs | remaining-tokens=%s remaining-requests=%s | %s",
                        wait, resp.headers.get("x-ratelimit-remaining-tokens"),
                        resp.headers.get("x-ratelimit-remaining-requests"), resp.text[:300],
                    )
                    if wait > _MAX_PAUSE_SECONDS:
                        throttle.aborted = (f"429 con Retry-After {wait:.0f}s "
                                            f"(probabile quota giornaliera): {resp.text[:200]}")
                        return {"raw_text": "", "pred_move_uci": "", "error": throttle.aborted}
                    throttle.pause(wait)
                    continue
                if resp.status_code in (400, 401, 403, 404):
                    # errore di configurazione (modello/chiave/payload): inutile ritentare
                    return {"raw_text": "", "pred_move_uci": "",
                            "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                message = choice["message"]
                total = (data.get("usage") or {}).get("total_tokens")
                if total:
                    throttle.record(float(total))
                # La mossa si legge SOLO dal content: il reasoning contiene mosse candidate, non la risposta.
                raw = message.get("content") or ""
                pred = _extract_uci(raw)
                err = None
                if pred:
                    return {"raw_text": raw, "pred_move_uci": pred, "error": None}
                # content vuoto (tipicamente finish_reason='length': token esauriti nel ragionamento).
                # Con temperature 0 e' deterministico: inutile ritentare.
                raw_last = raw
                finish = choice.get("finish_reason")
                reasoning_last = (message.get("reasoning") or "")[:1500]
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                logger.debug("[solve] attempt %d/%d fallita: %s", attempt, s.max_retries, err)
                await asyncio.sleep(s.retry_backoff_seconds * attempt)
    return {"raw_text": raw_last, "pred_move_uci": "", "error": err,
            "finish_reason": finish, "reasoning": reasoning_last}


async def solve_all_async(solver: GroqLLMSolver, items: List[Tuple[str, str, str, int]],
                          cache: Dict[str, Any], concurrency: int,
                          cache_path: Optional[str] = None, save_every: int = 10) -> None:
    sem = asyncio.Semaphore(max(1, concurrency))
    todo = [(p, f) for p, f, _, _ in items
            if not (isinstance(cache.get(p), dict)
                    and cache[p].get("pred_move_uci") and not cache[p].get("error"))]
    total = len(todo)
    logger.info("[LLM] %d da chiamare, %d gia' in cache (concorrenza=%d).",
                total, len(items) - total, concurrency)
    done = 0

    async with httpx.AsyncClient(timeout=solver.timeout_seconds) as client:
        throttle = _Throttle(solver.request_delay_seconds)

        async def one(pid: str, fen: str) -> None:
            nonlocal done
            cache[pid] = await _solve_async(solver, client, sem, fen, throttle)
            done += 1
            if done % save_every == 0 or done == total:
                logger.info("[LLM] valutati %d/%d.", done, total)
                if cache_path:
                    _atomic_json_dump(cache_path, cache)

        await asyncio.gather(*(one(p, f) for p, f in todo))

    if throttle.aborted:
        raise RuntimeError(f"[LLM] run interrotto, cache salvata: {throttle.aborted}")

    truncated = sum(1 for p, _ in todo if cache[p].get("finish_reason") == "length")
    if truncated:
        logger.warning("[LLM] %d/%d risposte troncate (finish_reason=length): alza llm_max_tokens.",
                       truncated, total)


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

    cache = dict(base_cache)
    try:
        if solver_factory is not None and items:
            solver = solver_factory()
            asyncio.run(solve_all_async(solver, items, cache, max(1, max_workers), cache_path))
    finally:
        if cache_path:
            _atomic_json_dump(cache_path, cache)

    results_by_pid: Dict[str, str] = {}
    for pid, fen, _, _ in items:
        r = cache.get(pid) if isinstance(cache.get(pid), dict) else {}
        pred = r.get("pred_move_uci") or ""
        if not pred:
            try:
                pred = _try_parse_move(r.get("raw_text", "") or "", chess.Board(fen))
            except Exception:
                pred = ""
        results_by_pid[pid] = pred

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