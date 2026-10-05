from __future__ import annotations

import io
import json
import logging
import multiprocessing as mp
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, fields
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
from utils.filters import QualityConfig, candidate_legal_moves, position_passes_quality
from utils.pgn_time import compute_move_duration, parse_clk, parse_emt, parse_rating, parse_time_control

logger = logging.getLogger("heldoutbuild")

SOURCE_TAG = "chesscom"
NORMAL_RULES = {"chess", "normal", "standard"}
TRUE_VALUES = {"true", "1", "yes"}
OPTIONAL_COLS = ("white_rating", "black_rating", "time_control", "time_class")


@dataclass
class HeldoutConfig:
    config_path: str = "main.yaml"
    input_csvs: List[str] = field(default_factory=lambda: ["/home/coco/Downloads/chesscom_games.csv"])
    pgn_col: str = "pgn"
    out_dir: Optional[str] = "../Heldout/Data/"
    stockfish_path: Optional[str] = None
    quota: Dict[int, int] = field(
        default_factory=lambda: {**{n: 30 for n in range(1, 6)}, **{n: 10 for n in range(6, 11)}}
    )
    stop_n_max: int = 5
    max_games: int = 1000
    time_classes: List[str] = field(default_factory=lambda: ["bullet", "blitz", "rapid"])
    rated_only: bool = False
    tail_plies: int = 30
    ply_step: int = 1
    max_cand_per_game: int = 4
    max_per_game: int = 1
    screen_time: float = 0.25
    screen_cp: int = 400
    analysis_time: float = 3.0
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


def _write_lines(path: str, records: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


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


def _isin(df: pd.DataFrame, col: str, values) -> pd.Series:
    return df[col].astype(str).str.strip().str.lower().isin(values)


def read_games(path: str, h: HeldoutConfig) -> pd.DataFrame:
    if not os.path.exists(path):
        raise ConfigError(f"input non trovato: {path}")
    pgn_col = h.pgn_col.lower()
    wanted = {pgn_col, "rules", "rated", *OPTIONAL_COLS}
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in wanted)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if pgn_col not in df.columns:
        raise ValueError(f"{path}: colonna PGN '{pgn_col}' non trovata.")
    df = df.rename(columns={pgn_col: "pgn"})
    df["gid"] = _safe_id(os.path.splitext(os.path.basename(path))[0]) + "_" + df.index.astype(str)
    n0 = len(df)

    mask = df["pgn"].notna() & df["pgn"].astype(str).str.strip().ne("")
    if "rules" in df.columns:
        mask &= _isin(df, "rules", NORMAL_RULES)
    classes = {t.lower() for t in h.time_classes}
    if classes and "time_class" in df.columns:
        mask &= _isin(df, "time_class", classes)
    if h.rated_only and "rated" in df.columns:
        mask &= _isin(df, "rated", TRUE_VALUES)
    df = df[mask]
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = None
    logger.info("[input] %s: %d lette, %d dopo i filtri.", path, n0, len(df))
    return df[["gid", "pgn", *OPTIONAL_COLS]]


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

    if candidate_legal_moves(board, q) is None or not position_passes_quality(board, q):
        st["quality"] += 1
        return None
    if _ENG.verify.syzygy_says_no_mate(board):
        st["syzygy_no_mate"] += 1
        return None

    st["screened"] += 1
    info = _ENG.screen.analyse(board)
    score = info[0].get("score") if info else None
    if score is None:
        st["engine_fail"] += 1
        return None
    rel = score.relative
    if rel.is_mate():
        m = rel.mate()
        if m is None or m <= 0:
            st["screen_reject"] += 1
            return None
    elif (rel.score() or 0) < o["screen_cp"]:
        st["screen_reject"] += 1
        return None

    st["verified_try"] += 1
    info = _ENG.verify.analyse(board)
    best = info[0] if info else None
    score = best.get("score") if best else None
    pv = best.get("pv") if best else None
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

    return {
        "id": f"{gid}_{ply}",
        "game_id": gid,
        "ply": int(ply),
        "fen": board.fen(),
        "n": int(m),
        "best_move_uci": pv[0].uci(),
        "pv_uci": " ".join(mv.uci() for mv in pv[: 2 * int(m) - 1]),
        "rating": int(rating),
        "source": SOURCE_TAG,
        "time_class": time_class,
        "clock_seconds": float(duration) if is_real else None,
        "clock_is_real": bool(is_real),
    }


def _analyse_game(task: Tuple[str, str, Any, Any, Any, Any]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    gid, pgn, white_rating, black_rating, tc_col, time_class = task
    o = _OPTS
    st: Counter = Counter()
    recs: List[Dict[str, Any]] = []

    if _ENG is None or not _ENG.is_ready():
        st["engine_fail"] += 1
        return dict(st), recs
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
    except Exception:
        game = None
    if game is None:
        st["bad_pgn"] += 1
        return dict(st), recs

    hd = game.headers
    if hd.get("SetUp") == "1" or hd.get("Variant", "Standard").lower() not in ("standard", "normal", "chess"):
        st["variant"] += 1
        return dict(st), recs
    end_ply = game.end().ply()
    if end_ply < o["min_game_plies"]:
        st["short_game"] += 1
        return dict(st), recs

    ratings = {
        chess.WHITE: parse_rating(white_rating) or parse_rating(hd.get("WhiteElo")),
        chess.BLACK: parse_rating(black_rating) or parse_rating(hd.get("BlackElo")),
    }
    base, inc = parse_time_control(str(tc_col) if tc_col is not None else None)
    if base <= 0:
        base, inc = parse_time_control(hd.get("TimeControl"))
    prev = {chess.WHITE: base or None, chess.BLACK: base or None}

    start = max(o["min_ply"], end_ply - o["tail_plies"]) if o["tail_plies"] > 0 else o["min_ply"]

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
        logger.warning("[heldout] %s (%s: %s): %d posizioni mantenute.", gid, type(e).__name__, e, len(recs))
    return dict(st), recs


def _clock(r: Dict[str, Any], h: HeldoutConfig, sampler: Optional[ClockSampler]) -> Tuple[float, str]:
    if h.clock_seconds is not None:
        return float(h.clock_seconds), "constant"
    if r["clock_is_real"]:
        return float(r["clock_seconds"]), "real"
    if sampler is not None:
        return sampler.sample(float(r["rating"]), r["n"], f"{r['id']}:0"), "synthetic"
    return 15.0, "default_constant"


def build(h: HeldoutConfig) -> Dict[str, Any]:
    cfg = load_config(h.config_path)
    setup_logging(cfg.pipeline.log_level, cfg.pipeline.log_file)

    stockfish = h.stockfish_path or cfg.engine.stockfish_path
    if not stockfish or not (os.path.exists(stockfish) and os.access(stockfish, os.X_OK)):
        raise ConfigError(f"stockfish non valido/eseguibile: {stockfish!r}")

    g = cfg.games_pipeline
    lo, hi = g.mate_range_min, g.mate_range_max
    out_dir = h.out_dir or cfg.path("HeldOut")
    quota = dict(sorted(h.quota.items()))
    shallow = [n for n in quota if n <= h.stop_n_max]
    quality = QualityConfig(**{f.name: getattr(g, f.name) for f in fields(QualityConfig) if hasattr(g, f.name)})

    games = pd.concat([read_games(p, h) for p in h.input_csvs], ignore_index=True)
    if games.empty:
        raise ConfigError("nessuna partita utilizzabile dopo i filtri di riga.")
    n_read = len(games)
    games = games.sample(n=min(h.max_games, n_read), random_state=h.seed).astype(object)
    games = games.where(games.notna(), None)
    tasks = list(zip(
        games["gid"], games["pgn"].astype(str), games["white_rating"], games["black_rating"],
        games["time_control"], games["time_class"],
    ))
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
        "quality": quality, "max_n": max(quota), "screen_cp": h.screen_cp,
        "tail_plies": h.tail_plies, "ply_step": max(1, h.ply_step),
        "min_ply": g.min_ply, "min_game_plies": g.min_game_plies,
        "require_real": require_real, "max_cand": max(1, h.max_cand_per_game),
    }
    workers = max(1, (os.cpu_count() or 2) - 1)

    stats: Counter = Counter()
    per_n: Counter = Counter()
    used_games: Counter = Counter()
    seen: Set[str] = set()
    selected: List[Dict[str, Any]] = []
    games_done = 0
    stopped_early = False

    pool = mp.Pool(workers, initializer=_init_worker, initargs=(screen_cfg, verify_cfg, opts))
    try:
        for st, recs in tqdm(pool.imap_unordered(_analyse_game, tasks, chunksize=2),
                             total=len(tasks), desc="Analisi partite"):
            games_done += 1
            stats.update(st)
            for r in sorted(recs, key=lambda x: -x["n"]):
                n = r["n"]
                key = _key(r["fen"])
                if n not in quota or per_n[n] >= quota[n]:
                    stats["quota_full"] += 1
                elif key in seen:
                    stats["duplicate"] += 1
                elif key in leaked:
                    stats["leaked_train"] += 1
                elif used_games[r["game_id"]] >= h.max_per_game:
                    stats["game_cap"] += 1
                else:
                    seen.add(key)
                    used_games[r["game_id"]] += 1
                    per_n[n] += 1
                    selected.append(r)
            if shallow and all(per_n[n] >= quota[n] for n in shallow):
                stopped_early = True
                logger.info("[verify] quote n<=%d coperte dopo %d partite.", h.stop_n_max, games_done)
                break
    finally:
        pool.terminate()
        pool.join()

    shortfall = {n: f"{per_n[n]}/{q}" for n, q in quota.items() if per_n[n] < q}
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
        clock, src = _clock(r, h, sampler)
        r["clock_seconds"], r["clock_source"], r["shard_index"] = round(float(clock), 3), src, -1
        if not lo <= r["n"] <= hi:
            continue
        try:
            d = build_position_data(
                board=chess.Board(r["fen"]), best_move=chess.Move.from_uci(r["best_move_uci"]),
                clock_seconds=clock, rating=float(r["rating"]), game_id=f"heldout_{r['id']}", ply=r["ply"],
                mate_n=r["n"], edge_time_factors=DEFAULT_EDGE_TIME_FACTORS, mate_range=(lo, hi),
            )
        except ValueError as e:
            stats["build_fail"] += 1
            logger.warning("[build] %s scartato dagli shard (%s).", r["id"], e)
            continue
        d.fen = r["fen"]
        r["shard_index"] = len(data_list)
        data_list.append(d)

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
        "selected": len(selected),
        "selected_by_n": dict(sorted(per_n.items())),
        "shortfall": shortfall,
        "in_shards": len(data_list),
        "mate_range_shards": [lo, hi],
        "rejected": dict(stats),
        "clock_real_selected": sum(1 for r in selected if r["clock_source"] == "real"),
        "leak_reference_positions": len(leaked),
    }
    atomic_write_json(os.path.join(out_dir, "report.json"), report)
    logger.info("[heldout] %s", report)
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