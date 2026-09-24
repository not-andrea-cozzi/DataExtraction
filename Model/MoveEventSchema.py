from __future__ import annotations

from typing import Optional, Tuple

import chess

_PROMOTION_TYPES: Tuple[Optional[int], ...] = (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
_PROMOTION_OFFSET = {pt: i for i, pt in enumerate(_PROMOTION_TYPES)}
NUM_PROMOTION_SLOTS = len(_PROMOTION_TYPES)

MOVE_VOCAB_SIZE = 64 * 64 * NUM_PROMOTION_SLOTS

CASE_COL = "game_id"
EVENT_COL = "move_uci_id"
TIME_COL = "event_time"
RATING_COL = "rating"
CLOCK_COL = "clock_seconds"
MATE_N_COL = "mate_n"
PLY_COL = "ply"

NUM_EVENT_FEATURES = [RATING_COL, CLOCK_COL]
SEQ_FEATURES = [MATE_N_COL]


def encode_move(move: "chess.Move") -> int:
    base = move.from_square * 64 + move.to_square
    promo_slot = _PROMOTION_OFFSET[move.promotion]
    return base * NUM_PROMOTION_SLOTS + promo_slot


def decode_move(move_id: int) -> Tuple[int, int, Optional[int]]:
    promo_slot = move_id % NUM_PROMOTION_SLOTS
    base = move_id // NUM_PROMOTION_SLOTS
    from_square = base // 64
    to_square = base % 64
    return from_square, to_square, _PROMOTION_TYPES[promo_slot]


def move_id_to_uci(move_id: int, board: "chess.Board") -> str:
    from_square, to_square, promotion = decode_move(move_id)
    move = chess.Move(from_square, to_square, promotion=promotion)
    if move not in board.legal_moves:
        raise ValueError(f"move_id={move_id} non legale sulla posizione data (fen={board.fen()}).")
    return move.uci()