"""Costruisce il set held-out di problemi "mate in n" classici (n=1..10).

Input : uno o piu' file CSV / JSONL / PGN con FEN (+ soluzione opzionale) di problemi esterni
        al dataset Lichess (Chess.com, Kaggle, raccolte di problemi in PGN, ...).
Output: <out_dir>/heldout.jsonl, heldout.csv     -> tutti gli n selezionati (per baseline LLM / analisi)
        <out_dir>/heldout_clean/shard_*.pt       -> solo n in mate_range, stesso formato del train
        <out_dir>/report.json                    -> conteggi per stadio di filtro

Ogni problema e' verificato con Stockfish: matto forzato, distanza == n dichiarato (se presente),
prima mossa unica (stesso criterio del train) e uguale alla prima mossa della soluzione fornita.
I duplicati e le posizioni gia' presenti nei CSV di debug del train vengono scartati.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import chess
import chess.pgn
import pandas as pd
import torch
from tqdm import tqdm

from builders.games.config import EngineConfig
from builders.games.engine import Engine, init_process, install_worker_signals, second_line_ties_mate
from common.io import atomic_write_json
from core.schema import build_position_data
from pipeline.config import ConfigError, load_config
from pipeline.main import setup_logging
from stats.clock_stats import ClockSampler
from utils.edge_weighting import DEFAULT_EDGE_TIME_FACTORS
from utils.pgn_time import parse_rating

logger = logging.getLogger("heldoutbuild")

DEFAULT_QUOTA = {1: 30, 2: 30, 3: 30, 4: 30, 5: 30, 6: 10, 7: 10, 8: 6, 9: 4, 10: 4}
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


# ------------------------------------------------------------------ parsing
def _key(fen: str) -> str:
    return " ".join(fen.split(" ")[:4])


def _to_n(v: Any) -> Optional[int]:
    if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
        return None
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def _tokens(text: str) -> List[str]:
    text = re.sub(r"\{[^}]*\}|\([^)]*\)|\$\d+", " ", text)
    out: List[str] = []
    for t in text.split():
        t = re.sub(r"^\d+\.+", "", t).rstrip("!?")
        if t in ("0-0", "0-0-0"):
            t = t.replace("0", "O")
        if t and t not in _RESULTS:
            out.append(t)
    return out


def _parse_move(board: "chess.Board", tok: str) -> Optional["chess.Move"]:
    try:
        mv = chess.Move.from_uci(tok)
        if mv in board.legal_moves:
            return mv
    except ValueError:
        pass
    try:
        return board.parse_san(tok)
    except ValueError:
        return None


def _first(low: Dict[str, str], names: List[str]) -> Optional[str]:
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _safe_id(s: Any) -> str:
    return re.sub(r"\W+", "_", str(s)).strip("_")


def read_table(path: str, a: argparse.Namespace) -> List[Dict[str, Any]]:
    df = pd.read_json(path, lines=True) if path.lower().endswith((".jsonl", ".json")) else pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    fen_c = _first(low, [a.fen_col] if a.fen_col else ["fen"])
    if fen_c is None:
        raise ValueError(f"{path}: colonna FEN non trovata (colonne: {list(df.columns)}). Usa --fen-col.")
    sol_c = _first(low, [a.solution_col] if a.solution_col else ["moves", "solution", "solution_uci", "pv"])
    n_c = _first(low, [a.n_col] if a.n_col else ["mate_in", "mate_n", "n"])
    th_c = _first(low, ["themes"])
    r_c = _first(low, [a.rating_col] if a.rating_col else ["rating", "elo"])
    id_c = _first(low, [a.id_col] if a.id_col else ["id", "puzzleid", "problem_id"])
    base = _safe_id(os.path.splitext(os.path.basename(path))[0])

    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(df.to_dict("records")):
        fen = rec.get(fen_c)
        if not isinstance(fen, str) or not fen.strip():
            continue
        claimed = _to_n(rec.get(n_c)) if n_c else None
        if claimed is None and th_c:
            m = re.search(r"mateIn(\d+)", str(rec.get(th_c, "")))
            claimed = int(m.group(1)) if m else None
        sol = rec.get(sol_c) if sol_c else None
        if isinstance(sol, list):
            sol = " ".join(map(str, sol))
        sol = "" if sol is None or (not isinstance(sol, str) and pd.isna(sol)) else str(sol)
        rows.append({
            "id": f"{base}_{_safe_id(rec.get(id_c)) if id_c else i}",
            "fen": fen.strip(),
            "solution": sol,
            "claimed_n": claimed,
            "rating": parse_rating(rec.get(r_c)) if r_c else None,
            "source": base,
        })
    return rows


def read_pgn(path: str) -> List[Dict[str, Any]]:
    base = _safe_id(os.path.splitext(os.path.basename(path))[0])
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        i = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            i += 1
            fen = game.headers.get("FEN")
            if not fen:
                continue
            try:
                sol = " ".join(m.uci() for m in game.mainline_moves())
            except Exception:
                continue
            m = re.search(r"[Mm]ate in (\d+)", game.headers.get("Event", ""))
            rows.append({
                "id": f"{base}_{i}", "fen": fen, "solution": sol,
                "claimed_n": int(m.group(1)) if m else None,
                "rating": None, "source": base,
            })
    return rows


def read_input(path: str, a: argparse.Namespace) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise ConfigError(f"input non trovato: {path}")
    return read_pgn(path) if path.lower().endswith(".pgn") else read_table(path, a)


def load_exclusions(paths: List[str]) -> Set[str]:
    keys: Set[str] = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        before = len(keys)
        for chunk in pd.read_csv(p, usecols=["fen"], chunksize=500_000):
            keys.update(_key(f) for f in chunk["fen"].dropna())
        logger.info("[leak] %s: %d posizioni uniche aggiunte.", p, len(keys) - before)
    return keys


# ------------------------------------------------------------------- worker
_ENGINE: Optional[Engine] = None
_OPTS: Dict[str, Any] = {}


def _init_worker(ecfg: EngineConfig, opts: Dict[str, Any]) -> None:
    global _ENGINE, _OPTS
    init_process()
    _OPTS = opts
    _ENGINE = Engine(ecfg)
    install_worker_signals(lambda: _ENGINE)


def _verify(cand: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    o = _OPTS
    cid = cand["id"]
    try:
        board = chess.Board(cand["fen"])
    except ValueError:
        return cid, "invalid_fen", None
    if not board.is_valid() or board.is_game_over():
        return cid, "invalid_fen", None

    toks = _tokens(cand["solution"])
    if o["skip_first_move"] and toks:
        mv = _parse_move(board, toks[0])
        if mv is None:
            return cid, "bad_solution", None
        board.push(mv)
        toks = toks[1:]
        if board.is_game_over():
            return cid, "invalid_fen", None

    line: List["chess.Move"] = []
    tmp = board.copy(stack=False)
    for t in toks:
        mv = _parse_move(tmp, t)
        if mv is None:
            break
        line.append(mv)
        tmp.push(mv)
    if toks and not line:
        return cid, "bad_solution", None

    if _ENGINE is None or not _ENGINE.is_ready():
        return cid, "engine_fail", None
    info = _ENGINE.analyse(board)
    if not info:
        return cid, "engine_fail", None
    best = info[0]
    score, pv = best.get("score"), best.get("pv")
    if score is None or not pv:
        return cid, "engine_fail", None
    rel = score.relative
    if not rel.is_mate():
        return cid, "not_mate", None
    m = rel.mate()
    if m is None or m <= 0:
        return cid, "not_mate", None
    if m > o["max_n"]:
        return cid, "too_deep", None
    claimed = cand["claimed_n"]
    if claimed is not None and claimed != m and not o["relabel"]:
        return cid, "n_mismatch", None
    if second_line_ties_mate(info, int(m)):
        return cid, "not_unique", None
    if line and line[0] != pv[0]:
        return cid, "first_move_mismatch", None

    fen = board.fen()
    return cid, None, {
        "id": cid,
        "fen": fen,
        "key": _key(fen),
        "n": int(m),
        "claimed_n": claimed,
        "best_move_uci": pv[0].uci(),
        "pv_uci": " ".join(mv.uci() for mv in pv[: 2 * int(m) - 1]),
        "solution_uci": " ".join(mv.uci() for mv in line),
        "rating": cand["rating"],
        "source": cand["source"],
    }


# --------------------------------------------------------------------- main
def _parse_quota(s: Optional[str]) -> Dict[int, int]:
    if not s:
        return dict(DEFAULT_QUOTA)
    out: Dict[int, int] = {}
    for part in s.split(","):
        n, q = part.split(":")
        out[int(n)] = int(q)
    return out


def _write_lines(path: str, records: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build(a: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(a.config)
    setup_logging(cfg.pipeline.log_level, cfg.pipeline.log_file)

    stockfish = a.stockfish or cfg.engine.stockfish_path
    if not stockfish or not (os.path.exists(stockfish) and os.access(stockfish, os.X_OK)):
        raise ConfigError(f"stockfish non valido/eseguibile: {stockfish!r}")
    lo, hi = cfg.games_pipeline.mate_range_min, cfg.games_pipeline.mate_range_max
    out_dir = a.out_dir or cfg.path("HeldOut")
    quota = _parse_quota(a.quota)
    quota = {n: q for n, q in quota.items() if 1 <= n <= a.max_n}
    rng = random.Random(a.seed)

    candidates: List[Dict[str, Any]] = []
    for p in a.input:
        rows = read_input(p, a)
        logger.info("[input] %s: %d candidati.", p, len(rows))
        candidates += rows
    if not candidates:
        raise ConfigError("nessun candidato letto dagli input.")
    n_read = len(candidates)

    rng.shuffle(candidates)
    kept: List[Dict[str, Any]] = []
    per_n: Counter = Counter()
    for c in candidates:
        n = c["claimed_n"]
        if n is not None:
            if n not in quota or per_n[n] >= quota[n] * a.oversample:
                continue
            per_n[n] += 1
        kept.append(c)
    kept = kept[: a.max_candidates]
    logger.info("[verify] %d candidati da analizzare (letti: %d).", len(kept), n_read)

    exclude_paths = a.exclude_csv if a.exclude_csv is not None else [
        os.path.join(cfg.games_dir, n) for n in ("games_debug.csv", "games_debug_records.pending.csv")
    ] + [
        os.path.join(cfg.puzzles_dir, n) for n in ("puzzle_debug.csv", "puzzle_debug_records.pending.csv")
    ]
    leaked = load_exclusions(exclude_paths)
    if not leaked:
        logger.warning("[leak] nessuna posizione di train caricata: leakage NON verificato.")

    ecfg = EngineConfig(
        stockfish_path=stockfish, threads=a.threads, hash_mb=a.hash_mb, multipv=2,
        analysis_time=a.analysis_time, retry_attempts=2, retry_backoff_seconds=0.5,
    )
    opts = {"max_n": a.max_n, "relabel": a.relabel, "skip_first_move": a.skip_first_move}
    workers = a.workers or max(1, (os.cpu_count() or 2) - 1)

    stats: Counter = Counter()
    valid: List[Dict[str, Any]] = []
    pool = mp.Pool(workers, initializer=_init_worker, initargs=(ecfg, opts))
    try:
        for _, reason, rec in tqdm(pool.imap_unordered(_verify, kept, chunksize=1),
                                   total=len(kept), desc="Verifica Stockfish"):
            if reason:
                stats[reason] += 1
            else:
                valid.append(rec)
    finally:
        pool.terminate()
        pool.join()

    valid.sort(key=lambda r: r["id"])
    seen: Set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for r in valid:
        if r["key"] in seen:
            stats["duplicate"] += 1
            continue
        seen.add(r["key"])
        if r["key"] in leaked:
            stats["leaked_train"] += 1
            continue
        uniq.append(r)

    by_n: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in uniq:
        by_n[r["n"]].append(r)
    selected: List[Dict[str, Any]] = []
    shortfall: Dict[int, str] = {}
    for n in sorted(quota):
        avail = by_n.get(n, [])
        k = min(quota[n], len(avail))
        if k < quota[n]:
            shortfall[n] = f"{k}/{quota[n]}"
            logger.warning("n=%d: disponibili %d, richiesti %d.", n, len(avail), quota[n])
        selected += rng.sample(avail, k)
    selected.sort(key=lambda r: (r["n"], r["id"]))

    sampler: Optional[ClockSampler] = None
    clock_stats = a.clock_stats or cfg.clock_stats_path
    if a.clock_seconds is None and os.path.exists(clock_stats):
        sampler = ClockSampler.from_json(
            clock_stats, mode="lognormal", condition_on_mate_n=a.condition_on_mate_n,
            min_seconds=0.5, cap_seconds=300.0,
        )

    data_list = []
    for r in selected:
        r.pop("key", None)
        given = r["rating"] is not None
        rating = r["rating"] if given else a.default_rating
        r["rating"], r["rating_is_given"] = rating, given
        if a.clock_seconds is not None:
            clock, src = float(a.clock_seconds), "constant"
        elif sampler is not None:
            clock, src = sampler.sample(float(rating), r["n"], f"{r['id']}:0"), "synthetic"
        else:
            clock, src = 15.0, "default_constant"
        r["clock_seconds"], r["clock_source"], r["shard_index"] = round(float(clock), 3), src, -1

        if lo <= r["n"] <= hi:
            try:
                d = build_position_data(
                    board=chess.Board(r["fen"]), best_move=chess.Move.from_uci(r["best_move_uci"]),
                    clock_seconds=clock, rating=float(rating), game_id=f"heldout_{r['id']}", ply=0,
                    mate_n=r["n"], edge_time_factors=DEFAULT_EDGE_TIME_FACTORS, mate_range=(lo, hi),
                )
                d.fen = r["fen"]
                r["shard_index"] = len(data_list)
                data_list.append(d)
            except ValueError as e:
                stats["build_fail"] += 1
                logger.warning("[build] %s scartato dagli shard (%s).", r["id"], e)

    os.makedirs(out_dir, exist_ok=True)
    shard_dir = os.path.join(out_dir, "heldout_clean")
    os.makedirs(shard_dir, exist_ok=True)
    for name in os.listdir(shard_dir):
        if name.startswith("shard_") or name == "manifest.json":
            os.remove(os.path.join(shard_dir, name))
    if data_list:
        path = os.path.join(shard_dir, "shard_00000.pt")
        torch.save(data_list, path + ".tmp")
        os.replace(path + ".tmp", path)
    atomic_write_json(
        os.path.join(shard_dir, "manifest.json"),
        {"num_shards": 1 if data_list else 0, "shard_size": max(len(data_list), 1), "total": len(data_list)},
    )

    _write_lines(os.path.join(out_dir, "heldout.jsonl"), selected)
    pd.DataFrame(selected).to_csv(os.path.join(out_dir, "heldout.csv"), index=False)

    report = {
        "read": n_read,
        "analysed": len(kept),
        "verified": len(valid),
        "unique_not_leaked": len(uniq),
        "selected": len(selected),
        "selected_by_n": dict(sorted(Counter(r["n"] for r in selected).items())),
        "available_by_n": {n: len(v) for n, v in sorted(by_n.items())},
        "shortfall": shortfall,
        "in_shards": len(data_list),
        "mate_range_shards": [lo, hi],
        "rejected": dict(stats),
        "clock": {"mode": "constant" if a.clock_seconds is not None else ("stats" if sampler else "default"),
                  "condition_on_mate_n": a.condition_on_mate_n, "default_rating": a.default_rating},
        "leak_reference_positions": len(leaked),
        "analysis_time": a.analysis_time,
    }
    atomic_write_json(os.path.join(out_dir, "report.json"), report)
    logger.info("[heldout] %s", json.dumps(report))
    logger.info("[heldout] scritto in %s (shard: %s).", out_dir, shard_dir)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build del set held-out di problemi mate-in-n classici")
    ap.add_argument("--config", default="main.yaml")
    ap.add_argument("--input", nargs="+", required=True, help="CSV / JSONL / PGN con FEN e (opz.) soluzione")
    ap.add_argument("--out-dir", default=None, help="Default: <dataset_dir>/HeldOut")
    ap.add_argument("--stockfish", default=None)
    ap.add_argument("--quota", default=None, help='es. "1:30,2:30,3:30,4:30,5:30,6:10,7:10,8:6,9:4,10:4"')
    ap.add_argument("--max-n", type=int, default=10)
    ap.add_argument("--analysis-time", type=float, default=3.0, help="secondi di Stockfish per posizione")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash-mb", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oversample", type=int, default=4, help="candidati analizzati per n = quota * oversample (se n dichiarato)")
    ap.add_argument("--max-candidates", type=int, default=4000)
    ap.add_argument("--relabel", action="store_true", help="accetta l'n di Stockfish se diverso da quello dichiarato")
    ap.add_argument("--skip-first-move", action="store_true", help="input stile Lichess: la prima mossa e' dell'avversario")
    ap.add_argument("--fen-col", default=None)
    ap.add_argument("--solution-col", default=None)
    ap.add_argument("--n-col", default=None)
    ap.add_argument("--rating-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--default-rating", type=int, default=1500, help="rating per il clock sintetico se assente")
    ap.add_argument("--clock-seconds", type=float, default=None, help="clock costante (ablazione timing)")
    ap.add_argument("--clock-stats", default=None)
    ap.add_argument("--condition-on-mate-n", action="store_true")
    ap.add_argument("--exclude-csv", nargs="*", default=None, help="CSV di debug con colonna 'fen' da escludere")
    args = ap.parse_args(argv)

    try:
        build(args)
        return 0
    except (ConfigError, ValueError) as e:
        logging.getLogger("heldoutbuild").error("Errore: %s", e)
        return 2
    except KeyboardInterrupt:
        logging.getLogger("heldoutbuild").warning("Interrotto.")
        return 130
    except Exception as e:
        logging.getLogger("heldoutbuild").exception("Interruzione imprevista: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())