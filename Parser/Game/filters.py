from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess

from .config import GamesBuilderConfig
from .pgn_utils import parse_rating

PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def headers_are_eligible(headers, cfg: GamesBuilderConfig) -> bool:
    if cfg.only_decisive_games:
        result = headers.get("Result", "")
        if result not in ("1-0", "0-1"):
            return False

    if cfg.skip_time_forfeit:
        termination = headers.get("Termination", "")
        if "Time forfeit" in termination:
            return False

    if cfg.min_rating is not None or cfg.max_rating is not None:
        white_elo = parse_rating(headers.get("WhiteElo", ""))
        black_elo = parse_rating(headers.get("BlackElo", ""))
        ratings = [r for r in (white_elo, black_elo) if r is not None]
        if ratings:
            best_rating = max(ratings)
            if cfg.min_rating is not None and best_rating < cfg.min_rating:
                return False
            if cfg.max_rating is not None and min(ratings) > cfg.max_rating:
                return False
    return True


def get_candidate_legal_moves(
    board: "chess.Board", cfg: GamesBuilderConfig
) -> Optional[List["chess.Move"]]:
    if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
        return None
    if cfg.max_piece_count is not None and len(board.piece_map()) > cfg.max_piece_count:
        return None

    n_legal = board.legal_moves.count()
    if n_legal < cfg.candidate_min_legal_moves or n_legal > 255:
        return None
    if cfg.candidate_max_legal_moves is not None and n_legal > cfg.candidate_max_legal_moves:
        return None
    if cfg.skip_if_in_check and board.is_check():
        return None
    return list(board.legal_moves)


def material_by_color(board: "chess.Board") -> Tuple[int, int]:
    white_mat = black_mat = 0
    for p in board.piece_map().values():
        val = PIECE_VALUES.get(p.piece_type, 0)
        if p.color == chess.WHITE:
            white_mat += val
        else:
            black_mat += val
    return white_mat, black_mat


def has_mating_material(board: "chess.Board", cfg: GamesBuilderConfig) -> bool:
    mover = board.turn
    white_mat, black_mat = material_by_color(board)
    mover_mat = white_mat if mover == chess.WHITE else black_mat
    opp_mat = black_mat if mover == chess.WHITE else white_mat

    if mover_mat < cfg.min_material_for_mate_attempt:
        return False
    if (mover_mat - opp_mat) < cfg.min_material_diff_for_mate_attempt:
        return False
    return True


def mover_has_heavy_piece(board: "chess.Board") -> bool:
    mover = board.turn
    for piece_type in (chess.QUEEN, chess.ROOK):
        if board.pieces(piece_type, mover):
            return True
    return False


def is_trivially_drawn_endgame(board: "chess.Board") -> bool:
    piece_map = board.piece_map()
    has_heavy_or_pawn = any(
        p.piece_type in (chess.QUEEN, chess.ROOK, chess.PAWN) for p in piece_map.values()
    )
    if has_heavy_or_pawn:
        return False
    white_minors = sum(
        1 for p in piece_map.values()
        if p.color == chess.WHITE and p.piece_type in (chess.BISHOP, chess.KNIGHT)
    )
    black_minors = sum(
        1 for p in piece_map.values()
        if p.color == chess.BLACK and p.piece_type in (chess.BISHOP, chess.KNIGHT)
    )
    return white_minors <= 1 and black_minors <= 1