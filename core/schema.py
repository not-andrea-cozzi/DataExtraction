from __future__ import annotations

import math
from typing import List, Optional, Tuple

import chess
import torch
from torch_geometric.data import Data

from .constants import (
    CLOCK_CAP_SECONDS,
    EDGE_ATTACK,
    EDGE_LEGAL_MOVE,
    EDGE_PIN,
    EVENT_ID_EMPTY,
    MATE_N_MAX,
    MOVE_VOCAB_SIZE,
    NUM_EDGE_TYPES,
)
from .labels import mate_in_n_label
from .move_codec import encode_move


def build_legal_move_indices(board: "chess.Board") -> torch.Tensor:
    indices = [encode_move(move) for move in board.legal_moves]
    return torch.tensor(indices, dtype=torch.int16)


def legal_move_mask_from_indices(indices: torch.Tensor, vocab_size: int = MOVE_VOCAB_SIZE) -> torch.Tensor:
    mask = torch.zeros(1, vocab_size, dtype=torch.bool)
    if indices.numel():
        mask[0, indices.long()] = True
    return mask


def build_legal_move_mask(board: "chess.Board") -> torch.Tensor:
    return legal_move_mask_from_indices(build_legal_move_indices(board))


def encode_square_event_id(board: "chess.Board", square: int) -> int:
    piece = board.piece_at(square)
    if piece is None:
        return EVENT_ID_EMPTY
    return piece.piece_type * 2 + (1 if piece.color == chess.WHITE else 0) + 1


def _spatial_edges(board: "chess.Board") -> Tuple[List[int], List[int], List[int]]:
    src: List[int] = []
    dst: List[int] = []
    typ: List[int] = []

    for move in board.legal_moves:
        src.append(move.from_square)
        dst.append(move.to_square)
        typ.append(EDGE_LEGAL_MOVE)

    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        for target in board.attacks(sq):
            src.append(sq)
            dst.append(target)
            typ.append(EDGE_ATTACK)

        pin_ray = board.pin(piece.color, sq)
        if len(pin_ray) < 64:
            for ray_sq in pin_ray:
                attacker = piece_map.get(ray_sq)
                if (
                    attacker
                    and attacker.color != piece.color
                    and attacker.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)
                ):
                    src.append(ray_sq)
                    dst.append(sq)
                    typ.append(EDGE_PIN)
    return src, dst, typ


def encode_edge_type_onehot(edge_type: List[int]) -> torch.Tensor:
    if not edge_type:
        return torch.zeros((0, NUM_EDGE_TYPES), dtype=torch.float)
    t = torch.tensor(edge_type, dtype=torch.long)
    return torch.nn.functional.one_hot(t, num_classes=NUM_EDGE_TYPES).float()


def clock_norm(clock_seconds: float, cap_seconds: float = CLOCK_CAP_SECONDS) -> float:
    denom = math.log1p(cap_seconds)
    if denom <= 0:
        return 0.0
    return min(math.log1p(max(clock_seconds, 0.0)) / denom, 1.0)


def build_position_data(
    board: "chess.Board",
    best_move: "chess.Move",
    clock_seconds: float,
    rating: float,
    game_id: str,
    ply: int,
    mate_n: Optional[int] = None,
    edge_time_factors: Optional[dict] = None,
    mate_range: Optional[Tuple[int, int]] = None,
) -> Data:
    if best_move not in board.legal_moves:
        raise ValueError(f"best_move={best_move.uci()} illegale (fen={board.fen()}).")
    if mate_n is not None and not (0 <= mate_n <= MATE_N_MAX):
        raise ValueError(f"mate_n={mate_n} fuori da [0,{MATE_N_MAX}].")

    mover = board.turn
    c = clock_norm(clock_seconds)

    event_ids = torch.tensor(
        [[encode_square_event_id(board, sq)] for sq in range(64)], dtype=torch.long
    )

    x_rows = []
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            x_rows.append([0.0, 0.0, c])
        elif piece.color == mover:
            x_rows.append([1.0, 0.0, c])
        else:
            x_rows.append([0.0, 1.0, c])
    x = torch.tensor(x_rows, dtype=torch.float)

    src, dst, typ = _spatial_edges(board)
    if not src:
        raise ValueError(f"nessun arco spaziale (fen={board.fen()}).")

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = encode_edge_type_onehot(typ)

    time_tensor = torch.full((len(src),), float(c), dtype=torch.float)
    if edge_time_factors:
        types_t = torch.tensor(typ, dtype=torch.long)
        factors = torch.ones(NUM_EDGE_TYPES, dtype=torch.float)
        for t_id, f in edge_time_factors.items():
            if 0 <= t_id < NUM_EDGE_TYPES:
                factors[t_id] = float(f)
        time_tensor = time_tensor * factors[types_t]

    data = Data(event_ids=event_ids, x=x, edge_index=edge_index, num_nodes=64)
    data.edge_attr = edge_attr
    data.time = time_tensor
    data.y = torch.tensor(encode_move(best_move), dtype=torch.long)
    data.legal_move_indices = build_legal_move_indices(board)
    data.rating = torch.tensor(float(rating), dtype=torch.float16)
    data.game_id = game_id
    data.ply = torch.tensor(int(ply), dtype=torch.int64)
    if mate_n is not None:
        data.position_mate_n = torch.tensor(int(mate_n), dtype=torch.uint8)
        if mate_range is not None:
            data.outcome = torch.tensor(
                mate_in_n_label(int(mate_n), mate_range), dtype=torch.long
            )
    return data