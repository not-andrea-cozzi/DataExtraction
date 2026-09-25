from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess

from core.constants import PIECE_VALUES
from utils.pgn_time import parse_rating


@dataclass(frozen=True)
class QualityConfig:
    """Filtri di qualita' condivisi da Games e Puzzle. Tutti opzionali/neutri di default."""
    min_material_for_mate_attempt: int = 0
    min_material_diff_for_mate_attempt: int = 0
    require_heavy_piece: bool = False
    skip_trivial_endgame: bool = False
    max_piece_count: Optional[int] = None
    candidate_min_legal_moves: int = 1
    candidate_max_legal_moves: Optional[int] = None
    skip_if_in_check: bool = False
    skip_forced_moves: bool = False


def material_by_color(board: chess.Board) -> Tuple[int, int]:
    white = black = 0
    for p in board.piece_map().values():
        v = PIECE_VALUES.get(p.piece_type, 0)
        if p.color == chess.WHITE:
            white += v
        else:
            black += v
    return white, black


def has_mating_material(board: chess.Board, q: QualityConfig) -> bool:
    white, black = material_by_color(board)
    mover, opp = (white, black) if board.turn == chess.WHITE else (black, white)
    return mover >= q.min_material_for_mate_attempt and (mover - opp) >= q.min_material_diff_for_mate_attempt


def mover_has_heavy_piece(board: chess.Board) -> bool:
    return any(board.pieces(pt, board.turn) for pt in (chess.QUEEN, chess.ROOK))


def is_trivially_drawn_endgame(board: chess.Board) -> bool:
    pm = board.piece_map()
    if any(p.piece_type in (chess.QUEEN, chess.ROOK, chess.PAWN) for p in pm.values()):
        return False
    minors = {chess.WHITE: 0, chess.BLACK: 0}
    for p in pm.values():
        if p.piece_type in (chess.BISHOP, chess.KNIGHT):
            minors[p.color] += 1
    return minors[chess.WHITE] <= 1 and minors[chess.BLACK] <= 1


def candidate_legal_moves(board: chess.Board, q: QualityConfig) -> Optional[List[chess.Move]]:
    """Mosse legali se la posizione e' analizzabile, altrimenti None."""
    if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
        return None
    if q.max_piece_count is not None and len(board.piece_map()) > q.max_piece_count:
        return None
    if q.skip_if_in_check and board.is_check():
        return None
    moves = list(board.legal_moves)
    n = len(moves)
    if n < q.candidate_min_legal_moves or n > 255:
        return None
    if q.candidate_max_legal_moves is not None and n > q.candidate_max_legal_moves:
        return None
    if q.skip_forced_moves and n == 1:
        return None
    return moves


def position_passes_quality(board: chess.Board, q: QualityConfig) -> bool:
    """Filtri di posizione senza enumerare le mosse (usato dal Puzzle)."""
    if q.max_piece_count is not None and len(board.piece_map()) > q.max_piece_count:
        return False
    if not has_mating_material(board, q):
        return False
    if q.require_heavy_piece and not mover_has_heavy_piece(board):
        return False
    if q.skip_trivial_endgame and is_trivially_drawn_endgame(board):
        return False
    return True


# ---- header-level (solo Games) ----
@dataclass(frozen=True)
class HeaderFilterConfig:
    only_decisive_games: bool = True
    skip_time_forfeit: bool = True
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    require_both_ratings: bool = True



"""
    Funzione che 
"""
def headers_are_eligible(headers, h: HeaderFilterConfig) -> bool:
    """Filtra le partite per esito, terminazione e rating minimi/massimi."""
    if h.only_decisive_games and headers.get("Result", "") not in ("1-0", "0-1"):
        return False

    termination = headers.get("Termination", "") or ""
    if h.skip_time_forfeit and "Time forfeit" in termination:
        return False
    if termination != "Normal":
        return False

    white = parse_rating(headers.get("WhiteElo"))
    black = parse_rating(headers.get("BlackElo"))
    if h.require_both_ratings and (white is None or black is None):
        return False

    ratings = [r for r in (white, black) if r is not None]
    if ratings:
        if h.min_rating is not None and max(ratings) < h.min_rating:
            return False
        if h.max_rating is not None and min(ratings) > h.max_rating:
            return False
    return True