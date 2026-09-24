from __future__ import annotations

from typing import Optional, Tuple

import chess

from .constants import NUM_PROMOTION_SLOTS, PROMOTION_OFFSET, PROMOTION_TYPES


def encode_move(move: "chess.Move") -> int:
    base = move.from_square * 64 + move.to_square
    return base * NUM_PROMOTION_SLOTS + PROMOTION_OFFSET[move.promotion]


def decode_move(move_id: int) -> Tuple[int, int, Optional[int]]:
    promo_slot = move_id % NUM_PROMOTION_SLOTS
    base = move_id // NUM_PROMOTION_SLOTS
    return base // 64, base % 64, PROMOTION_TYPES[promo_slot]


def move_id_to_uci(move_id: int, board: "chess.Board") -> str:
    frm, to, promo = decode_move(move_id)
    move = chess.Move(frm, to, promotion=promo)
    if move not in board.legal_moves:
        raise ValueError(f"move_id={move_id} illegale (fen={board.fen()}).")
    return move.uci()
