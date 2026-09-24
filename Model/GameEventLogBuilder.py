from __future__ import annotations

from typing import List, Optional, Tuple

import chess
import chess.pgn
import pandas as pd

from Model.MoveEventSchema import (
    CASE_COL,
    CLOCK_COL,
    EVENT_COL,
    MATE_N_COL,
    PLY_COL,
    RATING_COL,
    TIME_COL,
    encode_move,
)


def build_game_event_log(
    game: "chess.pgn.Game",
    game_id: str,
    mover_rating: dict,
    clock_lookup,
    mate_n: Optional[int],
    start_unix_time: float = 0.0,
) -> pd.DataFrame:
    rows: List[dict] = []
    node = game
    cumulative_time = start_unix_time

    while node.variations:
        next_node = node.variation(0)
        board = node.board()
        move = next_node.move
        mover_color = board.turn

        clock_seconds = clock_lookup(node, next_node, mover_color)
        cumulative_time += clock_seconds

        rows.append({
            CASE_COL: game_id,
            EVENT_COL: encode_move(move),
            TIME_COL: cumulative_time,
            RATING_COL: mover_rating.get(mover_color),
            CLOCK_COL: clock_seconds,
            MATE_N_COL: mate_n if mate_n is not None else -1,
            PLY_COL: next_node.ply(),
        })

        node = next_node

    return pd.DataFrame(rows)


def build_puzzle_event_log(
    board: "chess.Board",
    uci_moves: List[str],
    game_id: str,
    rating: float,
    clock_sampler,
    mate_n_initial: int,
    start_unix_time: float = 0.0,
) -> pd.DataFrame:
    rows: List[dict] = []
    cumulative_time = start_unix_time
    working_board = board.copy()

    for ply_idx, uci in enumerate(uci_moves, start=1):
        move = chess.Move.from_uci(uci)
        if move not in working_board.legal_moves:
            break

        clock_seconds = clock_sampler(rating, mate_n_initial, f"{game_id}:{ply_idx}")
        cumulative_time += clock_seconds

        current_mate_n = max(1, mate_n_initial - ((ply_idx - 1) // 2))

        rows.append({
            CASE_COL: game_id,
            EVENT_COL: encode_move(move),
            TIME_COL: cumulative_time,
            RATING_COL: rating,
            CLOCK_COL: clock_seconds,
            MATE_N_COL: current_mate_n,
            PLY_COL: ply_idx,
        })

        working_board.push(move)

    return pd.DataFrame(rows)