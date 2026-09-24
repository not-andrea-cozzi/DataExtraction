from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn

from core.schema import build_position_data
from utils.filters import candidate_legal_moves, has_mating_material, headers_are_eligible, \
    is_trivially_drawn_endgame, mover_has_heavy_piece
from utils.edge_weighting import DEFAULT_EDGE_TIME_FACTORS
from utils.ipc import encode_for_ipc
from utils.pgn_time import (
    closest_bucket_time, compute_move_duration, parse_clk, parse_emt, parse_rating, parse_time_control,
)

from .config import GamesBuilderConfig
from .engine import Engine, install_worker_signals, init_process, second_line_ties_mate

logger = logging.getLogger(__name__)

_CFG: Optional[GamesBuilderConfig] = None
_ENGINE: Optional[Engine] = None

Task = Tuple[int, str, str, str]           # (local_id, pgn, source_tag, resume_key)
Result = Tuple[int, str, bytes]            # (local_id, resume_key, payload)


def pool_initializer(cfg: GamesBuilderConfig) -> None:
    global _CFG, _ENGINE
    init_process()
    _CFG = cfg
    _ENGINE = Engine(cfg.engine)
    install_worker_signals(lambda: _ENGINE)


def worker_entry(task: Task) -> Result:
    return analyse_game(_CFG, _ENGINE, task)


def _resolve_duration(
    emt: Optional[float], prev_clock: Optional[float], cur_clock: Optional[float], inc: float
) -> Optional[float]:
    if emt is not None:
        return emt
    return compute_move_duration(prev_clock, cur_clock, inc)


def analyse_game(cfg: GamesBuilderConfig, engine: Optional[Engine], task: Task) -> Result:
    local_id, pgn_text, source_tag, resume_key = task
    empty = encode_for_ipc([])

    if engine is None or not engine.is_ready():
        return local_id, resume_key, empty

    try:
        stream = io.StringIO(pgn_text)
        headers = chess.pgn.read_headers(stream)
        if headers is None or headers.get("Variant", "Standard").lower() not in ("standard", "normal"):
            return local_id, resume_key, empty
        if not headers_are_eligible(headers, cfg.header):
            return local_id, resume_key, empty
        stream.seek(0)
        game = chess.pgn.read_game(stream)
        if game is None or game.end().ply() < cfg.sampling.min_game_plies:
            return local_id, resume_key, empty
    except Exception as e:
        logger.warning("[games] PGN illeggibile id=%s (%s: %s).", local_id, type(e).__name__, e)
        return local_id, resume_key, empty

    records = _analyse_moves(cfg, engine, game, f"{source_tag}_{local_id}", source_tag)

    try:
        return local_id, resume_key, encode_for_ipc(records)
    except Exception as e:
        logger.error("[games] serializzazione fallita id=%s (%s): %d posizioni perse.", local_id, e, len(records))
        return local_id, resume_key, empty


def _analyse_moves(
    cfg: GamesBuilderConfig, engine: Engine, game: "chess.pgn.Game", full_game_id: str, source_tag: str,
) -> List[Dict[str, Any]]:
    s, c = cfg.sampling, cfg.clock
    base_time, inc = parse_time_control(game.headers.get("TimeControl"))
    rating = {
        chess.WHITE: parse_rating(game.headers.get("WhiteElo")),
        chess.BLACK: parse_rating(game.headers.get("BlackElo")),
    }
    prev_clock = {chess.WHITE: base_time or None, chess.BLACK: base_time or None}
    end_ply = game.end().ply()

    records: List[Dict[str, Any]] = []
    seen: set = set()
    analysed = 0
    window_key: Optional[int] = None
    board = game.board()
    node = game

    try:
        while node.variations:
            nxt = node.variation(0)
            move = nxt.move
            ply = node.ply()
            color = board.turn
            comment = nxt.comment or ""

            emt = parse_emt(comment)
            cur_clock = parse_clk(comment)
            duration = _resolve_duration(emt, prev_clock[color], cur_clock, inc)
            is_real = duration is not None
            if cur_clock is not None:
                prev_clock[color] = cur_clock

            tail = s.dense_tail_plies > 0 and (end_ply - ply) <= s.dense_tail_plies
            step = s.ply_sample_step_tail if tail else s.ply_sample_step
            skip = ply < s.min_ply or (ply - s.min_ply) % step != 0 or (c.require_clock and not is_real)

            if not skip:
                if s.max_positions_per_game is not None and analysed >= s.max_positions_per_game:
                    break
                if s.dedupe_positions:
                    key = " ".join(board.fen().split(" ")[:4])
                    if key in seen:
                        skip = True
                    else:
                        seen.add(key)

            if not skip:
                did_analyse, rec = _evaluate(
                    cfg, engine, board, ply, rating[color], is_real, duration,
                    full_game_id, source_tag, window_key,
                )
                analysed += did_analyse
                if rec is not None:
                    records.append(rec)
                    if window_key is None:
                        window_key = rec["debug"]["mate_n_window"]

            board.push(move)
            node = nxt
    except Exception as e:
        logger.warning("[games] eccezione %s (%s: %s): %d posizioni mantenute.",
                       full_game_id, type(e).__name__, e, len(records), exc_info=True)
    return records


def _evaluate(
    cfg: GamesBuilderConfig, engine: Engine, board: "chess.Board", ply: int,
    rating: Optional[int], is_real: bool, duration: Optional[float],
    game_id: str, source_tag: str, window_key: Optional[int],
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """(analysed 0|1, record|None)."""
    q = cfg.quality
    if rating is None:
        return 0, None
    legal = candidate_legal_moves(board, q)
    if legal is None:
        return 0, None
    if q.require_heavy_piece and not mover_has_heavy_piece(board):
        return 0, None
    if not has_mating_material(board, q):
        return 0, None
    if q.skip_trivial_endgame and is_trivially_drawn_endgame(board):
        return 0, None
    if engine.syzygy_says_no_mate(board):
        return 0, None

    info = engine.analyse(board)
    if not info:
        return 1, None
    best = info[0]
    score = best.get("score")
    if score is None or not score.relative.is_mate():
        return 1, None

    mate_n = score.relative.mate()
    lo, hi = cfg.mate_range
    if mate_n is None or not (mate_n > 0 and lo <= mate_n <= hi):
        return 1, None
    if second_line_ties_mate(info, int(mate_n)):
        return 1, None

    pv = best.get("pv")
    if not pv or pv[0] not in legal:
        return 1, None
    best_move = pv[0]

    if is_real:
        clock_seconds, clock_source = duration, "real"
    else:
        bucket = closest_bucket_time(rating, cfg.clock.avg_time_by_rating)
        if bucket is not None:
            clock_seconds, clock_source = bucket, "rating_bucket"
        else:
            clock_seconds, clock_source = cfg.clock.default_move_seconds, "default_constant"
    if cfg.clock.drop_zero_clock and clock_seconds == 0.0 and not is_real:
        return 1, None

    try:
        data = build_position_data(
            board=board, best_move=best_move, clock_seconds=clock_seconds, rating=float(rating),
            game_id=game_id, ply=ply, mate_n=int(mate_n), edge_time_factors=DEFAULT_EDGE_TIME_FACTORS,
        )
    except ValueError:
        return 1, None

    debug = {
        "problem_id": f"{game_id}_{ply}",
        "game_id": game_id,
        "fen": board.fen(),
        "best_move_uci": best_move.uci(),
        "mate_n": int(mate_n),
        "mate_n_window": window_key if window_key is not None else int(mate_n),
        "ply": int(ply),
        "source": source_tag,
        "clock_source": clock_source,
        "clock_seconds": float(clock_seconds),
        "clock_is_real": bool(is_real),
        "rating": rating,
    }
    return 1, {"data": data, "debug": debug}
