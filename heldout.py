from __future__ import annotations

import io
import json
import logging
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
from utils.filters import (
    QualityConfig, candidate_legal_moves, has_mate_potential, has_mating_material,
    is_trivially_drawn_endgame, mover_has_heavy_piece,
)
from utils.pgn_time import compute_move_duration, parse_clk, parse_emt, parse_rating, parse_time_control

logger = logging.getLogger("heldoutbuild")

SOURCE_TAG = "chesscom"
NORMAL_RULES = {"chess", "normal", "standard"}
TRUE_VALUES = {"true", "1", "yes"}


@dataclass
class HeldoutConfig:
    config_path: str = "main.yaml"
    input_csvs: List[str] = field(default_factory=lambda: ["/home/coco/Downloads/chesscom_games.csv"])
    pgn_col: str = "pgn"
    out_dir: Optional[str] = "../Heldout/Data/"
    stockfish_path: Optional[str] = None
    quota: Dict[int, int] = field(
        default_factory=lambda: {1: 30, 2: 30, 3: 30, 4: 30, 5: 30, 6: 10, 7: 10, 8: 6, 9: 4, 10: 4}
    )
    max_n: int = 10
    max_games: int = 10000
    time_classes: List[str] = field(default_factory=lambda: ["bullet", "blitz", "rapid"])
    rated_only: bool = False
    tail_plies: int = 30
    ply_step: int = 1
    max_cand_per_game: int = 6
    max_per_game: int = 1
    oversample: int = 4
    screen_time: float = 0.25
    screen_cp: int = 400
    analysis_time: float = 3.0
    workers: Optional[int] = None
    threads: int = 1
    hash_mb: int = 8
    seed: int = 42
    clock_seconds: Optional[float] = None
    allow_synthetic_clock: bool = False
    clock_stats_path: Optional[str] = None
    condition_on_mate_n: bool = False
    exclude_csvs: Optional[List[str]] = None


def _key(fen: str) -> str:
    return " ".join(fen.split(" ")[:4])


def _safe_id(s: Any) -> str:
    return re.sub(r"\W+", "_", str(s)).strip("_")


def _nn(v: Any) -> Any:
    return None if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)) else v


def _quality(cfg) -> QualityConfig:
    g = cfg.games_pipeline
    return QualityConfig(
        min_material_for_mate_attempt=g.min_material_for_mate_attempt,
        min_material_diff_for_mate_attempt=g.min_material_diff_for_mate_attempt,
        require_heavy_piece=g.require_heavy_piece, skip_trivial_endgame=g.skip_trivial_endgame,
        max_piece_count=g.max_piece_count, candidate_min_legal_moves=g.candidate_min_legal_moves,
        candidate_max_legal_moves=g.candidate_max_legal_moves, skip_if_in_check=g.skip_if_in_check,
        skip_forced_moves=g.skip_forced_moves, require_mate_potential=g.require_mate_potential,
        mate_potential_min_attackers=g.mate_potential_min_attackers,
        mate_potential_max_escapes=g.mate_potential_max_escapes,
    )


def read_games(path: str, h: HeldoutConfig) -> pd.DataFrame:
    if not os.path.exists(path):
        raise ConfigError(f"input non trovato: {path}")
    pgn_col = h.pgn_col.lower()
    wanted = {pgn_col, "white_rating", "black_rating", "time_control", "time_class", "rules", "rated"}
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in wanted)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if pgn_col not in df.columns:
        raise ValueError(f"{path}: colonna PGN '{pgn_col}' non trovata.")
    if pgn_col != "pgn":
        df = df.rename(columns={pgn_col: "pgn"})

    base = _safe_id(os.path.splitext(os.path.basename(path))[0])
    df["gid"] = [f"{base}_{i}" for i in df.index]
    n0 = len(df)

    df = df[df["pgn"].notna() & (df["pgn"].astype(str).str.strip() != "")]
    if "rules" in df.columns:
        df = df[df["rules"].astype(str).str.strip().str.lower().isin(NORMAL_RULES)]
    else:
        logger.warning("[input] %s: colonna 'rules' assente, varianti non filtrate dalla colonna.", path)
    classes = [t.lower() for t in h.time_classes]
    if classes and "time_class" in df.columns:
        df = df[df["time_class"].astype(str).str.strip().str.lower().isin(classes)]
    if h.rated_only and "rated" in df.columns:
        df = df[df["rated"].astype(str).str.strip().str.lower().isin(TRUE_VALUES)]
    for col in ("white_rating", "black_rating", "time_control", "time_class"):
        if col not in df.columns:
            df[col] = None
    logger.info("[input] %s: %d partite lette, %d dopo i filtri di riga.", path, n0, len(df))
    return df


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


class _Engines:
    def __init__(self, screen_cfg: EngineConfig, verify_cfg: EngineConfig) -> None:
        self.screen = Engine(screen_cfg)
        self.verify = Engine(verify_cfg)

    def is_ready(self) -> bool:
        return self.screen.is_ready() and self.verify.is_ready()

    def close(self) -> None:
        self.screen.close()
        self.verify.close()


_ENG: Optional[_Engines] = None
_OPTS: Dict[str, Any] = {}


def _init_worker(screen_cfg: EngineConfig, verify_cfg: EngineConfig, opts: Dict[str, Any]) -> None:
    global _ENG, _OPTS
    init_process()
    _OPTS = opts
    _ENG = _Engines(screen_cfg, verify_cfg)
    install_worker_signals(lambda: _ENG)


def _evaluate_position(
    board: "chess.Board", ply: int, rating: int, duration: Optional[float], is_real: bool,
    gid: str, time_class: Optional[str], st: Counter,
) -> Optional[Dict[str, Any]]:
    o, q = _OPTS, _OPTS["quality"]

    if candidate_legal_moves(board, q) is None:
        st["q_legal"] += 1
        return None
    if q.require_heavy_piece and not mover_has_heavy_piece(board):
        st["q_heavy"] += 1
        return None
    if not has_mating_material(board, q):
        st["q_material"] += 1
        return None
    if q.skip_trivial_endgame and is_trivially_drawn_endgame(board):
        st["q_trivial"] += 1
        return None
    if q.require_mate_potential and not has_mate_potential(
        board, q.mate_potential_min_attackers, q.mate_potential_max_escapes
    ):
        st["q_potential"] += 1
        return None
    if _ENG.verify.syzygy_says_no_mate(board):
        st["syzygy_no_mate"] += 1
        return None

    st["screened"] += 1
    info = _ENG.screen.analyse(board)
    if not info or info[0].get("score") is None:
        st["engine_fail"] += 1
        return None
    rel = info[0]["score"].relative
    if rel.is_mate():
        m = rel.mate()
        if m is None or m <= 0:
            st["screen_no_win"] += 1
            return None
    else:
        cp = rel.score()
        if cp is None or cp < o["screen_cp"]:
            st["screen_no_win"] += 1
            return None

    st["verified_try"] += 1
    info = _ENG.verify.analyse(board)
    if not info:
        st["engine_fail"] += 1
        return None
    best = info[0]
    score, pv = best.get("score"), best.get("pv")
    if score is None or not pv:
        st["engine_fail"] += 1
        return None
    rel = score.relative
    m = rel.mate() if rel.is_mate() else None
    if m is None or m <= 0:
        st["not_mate"] += 1
        return None
    if m > o["max_n"]:
        st["too_deep"] += 1
        return None
    if second_line_ties_mate(info, int(m)):
        st["not_unique"] += 1
        return None
    if pv[0] not in board.legal_moves:
        st["engine_fail"] += 1
        return None

    fen = board.fen()
    return {
        "id": f"{gid}_{ply}",
        "game_id": gid,
        "ply": int(ply),
        "fen": fen,
        "key": _key(fen),
        "n": int(m),
        "best_move_uci": pv[0].uci(),
        "pv_uci": " ".join(mv.uci() for mv in pv[: 2 * int(m) - 1]),
        "rating": int(rating),
        "source": SOURCE_TAG,
        "time_class": time_class,
        "clock_seconds": float(duration) if is_real else None,
        "clock_is_real": bool(is_real),
    }


def _analyse_game(task: Tuple[str, str, Any, Any, Any, Any]) -> Tuple[str, Dict[str, int], List[Dict[str, Any]]]:
    gid, pgn, white_rating, black_rating, tc_col, time_class = task
    o = _OPTS
    st: Counter = Counter()
    recs: List[Dict[str, Any]] = []

    if _ENG is None or not _ENG.is_ready():
        st["engine_fail"] += 1
        return gid, dict(st), recs
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
    except Exception:
        st["bad_pgn"] += 1
        return gid, dict(st), recs
    if game is None:
        st["bad_pgn"] += 1
        return gid, dict(st), recs

    hd = game.headers
    if hd.get("SetUp") == "1" or hd.get("Variant", "Standard").lower() not in ("standard", "normal", "chess"):
        st["variant"] += 1
        return gid, dict(st), recs
    end_ply = game.end().ply()
    if end_ply < o["min_game_plies"]:
        st["short_game"] += 1
        return gid, dict(st), recs

    ratings = {
        chess.WHITE: parse_rating(white_rating) or parse_rating(hd.get("WhiteElo")),
        chess.BLACK: parse_rating(black_rating) or parse_rating(hd.get("BlackElo")),
    }
    base, inc = parse_time_control(str(tc_col) if tc_col is not None else None)
    if base <= 0:
        base, inc = parse_time_control(hd.get("TimeControl"))
    prev = {chess.WHITE: base or None, chess.BLACK: base or None}

    tail = o["tail_plies"]
    start = max(o["min_ply"], end_ply - tail) if tail > 0 else o["min_ply"]

    board = game.board()
    node = game
    try:
        while node.variations:
            nxt = node.variation(0)
            ply = node.ply()
            color = board.turn
            comment = nxt.comment or ""

            emt = parse_emt(comment)
            cur = parse_clk(comment)
            duration = emt if emt is not None else compute_move_duration(prev[color], cur, inc)
            is_real = duration is not None
            if cur is not None:
                prev[color] = cur

            if ply >= start and (ply - start) % o["ply_step"] == 0:
                rating = ratings[color]
                if rating is None:
                    st["no_rating"] += 1
                elif o["require_real"] and not is_real:
                    st["no_clock"] += 1
                elif is_real and duration <= 0.0:
                    st["zero_clock"] += 1
                else:
                    rec = _evaluate_position(board, ply, rating, duration, is_real, gid, time_class, st)
                    if rec is not None:
                        recs.append(rec)
                        if len(recs) >= o["max_cand"]:
                            break

            board.push(nxt.move)
            node = nxt
    except Exception as e:
        st["game_exception"] += 1
        logger.warning("[heldout] eccezione su %s (%s: %s): %d posizioni mantenute.",
                       gid, type(e).__name__, e, len(recs))
    return gid, dict(st), recs


def _write_lines(path: str, records: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def build(h: HeldoutConfig) -> Dict[str, Any]:
    cfg = load_config(h.config_path)
    setup_logging(cfg.pipeline.log_level, cfg.pipeline.log_file)

    stockfish = h.stockfish_path or cfg.engine.stockfish_path
    if not stockfish or not (os.path.exists(stockfish) and os.access(stockfish, os.X_OK)):
        raise ConfigError(f"stockfish non valido/eseguibile: {stockfish!r}")
    lo, hi = cfg.games_pipeline.mate_range_min, cfg.games_pipeline.mate_range_max
    out_dir = h.out_dir or cfg.path("HeldOut")
    quota = {n: q for n, q in h.quota.items() if 1 <= n <= h.max_n}
    rng = random.Random(h.seed)

    games = pd.concat([read_games(p, h) for p in h.input_csvs], ignore_index=True)
    if games.empty:
        raise ConfigError("nessuna partita utilizzabile dopo i filtri di riga.")
    n_read = len(games)
    tasks = [
        (r["gid"], str(r["pgn"]), _nn(r["white_rating"]), _nn(r["black_rating"]),
         _nn(r["time_control"]), _nn(r["time_class"]))
        for r in games.to_dict("records")
    ]
    rng.shuffle(tasks)
    tasks = tasks[: h.max_games]
    logger.info("[verify] %d partite da analizzare (dopo filtri: %d).", len(tasks), n_read)

    exclude_paths = h.exclude_csvs if h.exclude_csvs is not None else [
        os.path.join(cfg.games_dir, n) for n in ("games_debug.csv", "games_debug_records.pending.csv")
    ] + [
        os.path.join(cfg.puzzles_dir, n) for n in ("puzzle_debug.csv", "puzzle_debug_records.pending.csv")
    ]
    leaked = load_exclusions(exclude_paths)
    if not leaked:
        logger.warning("[leak] nessuna posizione di train caricata: leakage NON verificato.")

    verify_cfg = EngineConfig(
        stockfish_path=stockfish, threads=h.threads, hash_mb=h.hash_mb, multipv=2,
        syzygy_path=cfg.engine.syzygy_path, analysis_time=h.analysis_time,
        retry_attempts=2, retry_backoff_seconds=0.5,
    )
    screen_cfg = EngineConfig(
        stockfish_path=stockfish, threads=h.threads, hash_mb=max(8, h.hash_mb // 4), multipv=2,
        syzygy_path=None, analysis_time=h.screen_time, retry_attempts=2, retry_backoff_seconds=0.5,
    )
    require_real = h.clock_seconds is None and not h.allow_synthetic_clock
    opts = {
        "quality": _quality(cfg), "max_n": h.max_n, "screen_cp": h.screen_cp,
        "tail_plies": h.tail_plies, "ply_step": max(1, h.ply_step),
        "min_ply": cfg.games_pipeline.min_ply, "min_game_plies": cfg.games_pipeline.min_game_plies,
        "require_real": require_real, "max_cand": max(1, h.max_cand_per_game),
    }
    workers = h.workers or max(1, (os.cpu_count() or 2) - 1)

    stats: Counter = Counter()
    valid: List[Dict[str, Any]] = []
    found: Counter = Counter()
    games_done = 0
    stopped_early = False
    pool = mp.Pool(workers, initializer=_init_worker, initargs=(screen_cfg, verify_cfg, opts))
    try:
        for _, st, recs in tqdm(pool.imap_unordered(_analyse_game, tasks, chunksize=2),
                                total=len(tasks), desc="Analisi partite"):
            games_done += 1
            stats.update(st)
            for r in recs:
                valid.append(r)
                found[r["n"]] += 1
            if quota and all(found[n] >= q * h.oversample for n, q in quota.items()):
                stopped_early = True
                logger.info("[verify] quote coperte (x%d) dopo %d partite: stop anticipato.", h.oversample, games_done)
                break
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
        if r["n"] in quota:
            by_n[r["n"]].append(r)
    used_games: Counter = Counter()
    selected: List[Dict[str, Any]] = []
    shortfall: Dict[int, str] = {}
    for n in sorted(quota, reverse=True):
        avail = list(by_n.get(n, []))
        rng.shuffle(avail)
        take: List[Dict[str, Any]] = []
        for r in avail:
            if len(take) >= quota[n]:
                break
            if used_games[r["game_id"]] >= h.max_per_game:
                continue
            used_games[r["game_id"]] += 1
            take.append(r)
        if len(take) < quota[n]:
            shortfall[n] = f"{len(take)}/{quota[n]}"
            logger.warning("n=%d: selezionati %d (disponibili %d), richiesti %d.",
                           n, len(take), len(avail), quota[n])
        selected += take
    selected.sort(key=lambda r: (r["n"], r["id"]))

    sampler: Optional[ClockSampler] = None
    clock_stats = h.clock_stats_path or cfg.clock_stats_path
    if h.clock_seconds is None and h.allow_synthetic_clock and os.path.exists(clock_stats):
        sampler = ClockSampler.from_json(
            clock_stats, mode="lognormal", condition_on_mate_n=h.condition_on_mate_n,
            min_seconds=0.5, cap_seconds=300.0,
        )

    data_list = []
    for r in selected:
        r.pop("key", None)
        if h.clock_seconds is not None:
            clock, src = float(h.clock_seconds), "constant"
        elif r["clock_is_real"]:
            clock, src = float(r["clock_seconds"]), "real"
        elif sampler is not None:
            clock, src = sampler.sample(float(r["rating"]), r["n"], f"{r['id']}:0"), "synthetic"
        else:
            clock, src = 15.0, "default_constant"
        r["clock_seconds"], r["clock_source"], r["shard_index"] = round(float(clock), 3), src, -1

        if lo <= r["n"] <= hi:
            try:
                d = build_position_data(
                    board=chess.Board(r["fen"]), best_move=chess.Move.from_uci(r["best_move_uci"]),
                    clock_seconds=clock, rating=float(r["rating"]), game_id=f"heldout_{r['id']}", ply=r["ply"],
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
        "source": SOURCE_TAG,
        "games_after_row_filters": n_read,
        "games_planned": len(tasks),
        "games_analysed": games_done,
        "stopped_early": stopped_early,
        "candidates_verified": len(valid),
        "verified_by_n": dict(sorted(found.items())),
        "unique_not_leaked": len(uniq),
        "selected": len(selected),
        "selected_by_n": dict(sorted(Counter(r["n"] for r in selected).items())),
        "available_by_n": {n: len(v) for n, v in sorted(by_n.items())},
        "shortfall": shortfall,
        "in_shards": len(data_list),
        "mate_range_shards": [lo, hi],
        "rejected": dict(stats),
        "clock": {
            "mode": "constant" if h.clock_seconds is not None else ("real" if require_real else "real+fallback"),
            "condition_on_mate_n": h.condition_on_mate_n,
            "real_clock_selected": int(sum(1 for r in selected if r["clock_source"] == "real")),
        },
        "params": {
            "screen_time": h.screen_time, "screen_cp": h.screen_cp, "analysis_time": h.analysis_time,
            "tail_plies": h.tail_plies, "ply_step": h.ply_step, "max_per_game": h.max_per_game,
            "max_cand_per_game": h.max_cand_per_game, "time_classes": h.time_classes,
        },
        "leak_reference_positions": len(leaked),
    }
    atomic_write_json(os.path.join(out_dir, "report.json"), report)
    logger.info("[heldout] %s", json.dumps(report))
    logger.info("[heldout] scritto in %s (shard: %s).", out_dir, shard_dir)
    return report


def main() -> int:
    try:
        build(HeldoutConfig())
        return 0
    except (ConfigError, ValueError) as e:
        logger.error("Errore: %s", e)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrotto.")
        return 130
    except Exception as e:
        logger.exception("Interruzione imprevista: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())