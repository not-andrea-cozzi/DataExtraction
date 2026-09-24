from __future__ import annotations

import io
import logging
from collections import defaultdict
from typing import Dict

import chess.pgn
import zstandard as zstd

from common.io import atomic_write_json, read_json
from utils.pgn_time import (
    compute_move_duration, parse_clk, parse_emt, parse_rating, parse_time_control,
)

logger = logging.getLogger(__name__)


class TimeStatsBuilder:
    """Media tempo/mossa per bucket di rating da un .pgn.zst Lichess."""

    def __init__(self, zst_path: str, max_games: int = 50_000, bucket_size: int = 100) -> None:
        self.zst_path = zst_path
        self.max_games = max_games
        self.bucket_size = bucket_size

    def _bucket(self, rating: int) -> int:
        return int(round(rating / self.bucket_size) * self.bucket_size)

    def build(self) -> Dict[int, float]:
        sums: Dict[int, float] = defaultdict(float)
        counts: Dict[int, int] = defaultdict(int)
        games = 0

        with open(self.zst_path, "rb") as raw:
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            while games < self.max_games:
                game = chess.pgn.read_game(text)
                if game is None:
                    break
                games += 1
                base, inc = parse_time_control(game.headers.get("TimeControl"))
                ratings = {True: parse_rating(game.headers.get("WhiteElo")),
                           False: parse_rating(game.headers.get("BlackElo"))}
                prev = {True: base or None, False: base or None}
                board = game.board()
                node = game
                while node.variations:
                    nxt = node.variation(0)
                    color = board.turn
                    emt = parse_emt(nxt.comment)
                    clk = parse_clk(nxt.comment)
                    dur = emt if emt is not None else compute_move_duration(prev[color], clk, inc)
                    if clk is not None:
                        prev[color] = clk
                    r = ratings[color]
                    if dur is not None and r is not None:
                        b = self._bucket(r)
                        sums[b] += dur
                        counts[b] += 1
                    board.push(nxt.move)
                    node = nxt

        if not counts:
            raise ValueError("TimeStatsBuilder: nessun dato di tempo trovato.")
        return {b: sums[b] / counts[b] for b in sorted(counts)}

    def build_and_save(self, out_json: str) -> Dict[int, float]:
        stats = self.build()
        atomic_write_json(out_json, {str(k): v for k, v in stats.items()}, sort_keys=True)
        logger.info("[time_stats] %d bucket -> %s", len(stats), out_json)
        return stats


def load_avg_time_by_rating(path: str) -> Dict[int, float]:
    raw = read_json(path, default={}) or {}
    return {int(k): float(v) for k, v in raw.items()}
