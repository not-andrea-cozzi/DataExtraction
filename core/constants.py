from __future__ import annotations

from typing import Dict, Optional, Tuple

import chess

PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

PROMOTION_TYPES: Tuple[Optional[int], ...] = (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMOTION_OFFSET: Dict[Optional[int], int] = {pt: i for i, pt in enumerate(PROMOTION_TYPES)}
NUM_PROMOTION_SLOTS = len(PROMOTION_TYPES)
MOVE_VOCAB_SIZE = 64 * 64 * NUM_PROMOTION_SLOTS

EVENT_ID_EMPTY = 0
NUM_EVENT_ID_CATEGORIES = 15

EDGE_LEGAL_MOVE = 0
EDGE_ATTACK = 1
EDGE_PIN = 2
NUM_EDGE_TYPES = 3

NUM_NODE_FEATURES = 3
EDGE_DIM_BASIC = NUM_EDGE_TYPES
TIME_EDGE_DIM = 1

CLOCK_CAP_SECONDS = 600.0
MATE_N_MAX = 255
