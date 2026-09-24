from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, Optional

import chess
import pandas as pd
from tqdm import tqdm

from common.io import JsonlAppender, finalize_jsonl
from core.schema import build_position_data
from spool.position_queue import PositionSpool
from stats.clock_stats import ClockSampler
from utils.edge_weighting import DEFAULT_EDGE_TIME_FACTORS
from utils.filters import position_passes_quality

from .config import PuzzleBuilderConfig
from .loader import extract_mate_n, load_rows

logger = logging.getLogger(__name__)

_DEFAULT_PUZZLE_RATING = 1500.0


def build_clock_sampler(cfg: PuzzleBuilderConfig) -> ClockSampler:
    c = cfg.clock
    kw = dict(mode=c.mode, condition_on_mate_n=c.condition_on_mate_n,
              min_seconds=c.min_seconds, cap_seconds=c.cap_seconds)
    if c.stats_path and os.path.exists(c.stats_path):
        logger.info("Clock puzzle: statistiche reali da %s.", c.stats_path)
        return ClockSampler.from_json(c.stats_path, **kw)
    if c.avg_time_by_rating:
        logger.warning("clock_stats assente: fallback su avg_time_by_rating (sigma fissa).")
        return ClockSampler.from_avg_time(c.avg_time_by_rating, **kw)
    raise ValueError("PuzzleBuilder: servono clock stats o avg_time_by_rating.")


class PuzzleBuilder:
    def __init__(self, config: PuzzleBuilderConfig, spool: PositionSpool) -> None:
        config.validate()
        self.config = config
        self.spool = spool
        self._sampler = build_clock_sampler(config)

        self.debug: Optional[JsonlAppender] = None
        if config.save_debug_jsonl:
            d = config.debug_dir or "."
            os.makedirs(d, exist_ok=True)
            self.debug = JsonlAppender(
                os.path.join(d, "puzzle_debug_records.pending.jsonl"),
                os.path.join(d, "puzzle_debug.jsonl"),
            )

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        rows = load_rows(cfg)

        processed = accepted = enqueued = quality_filtered = deduped = 0
        mate_counts: Dict[int, int] = defaultdict(int)
        source_counts: Dict[int, int] = defaultdict(int)

        for row in tqdm(rows, desc="Costruzione posizioni puzzle"):
            processed += 1
            n = self._process_row(row, mate_counts, source_counts)
            if n is None:
                continue
            e, qf, dd = n
            quality_filtered += qf
            deduped += dd
            if e:
                accepted += 1
                enqueued += e

        self.spool.flush()
        if self.debug:
            self.debug.persist()

        logger.info("Puzzle: processed=%d accepted=%d enqueued=%d quality_filtered=%d deduped=%d",
                    processed, accepted, enqueued, quality_filtered, deduped)
        logger.info("Sorgente per mate_n: %s", dict(sorted(source_counts.items())))
        logger.info("Posizioni per mate_n: %s", dict(sorted(mate_counts.items())))

        return {
            "processed_puzzles": processed,
            "accepted_puzzles": accepted,
            "enqueued_positions": enqueued,
            "mate_n_counts": dict(mate_counts),
            "source_mate_n_counts": dict(source_counts),
            "quality_filtered_positions": quality_filtered,
            "deduped_positions": deduped,
        }

    def _process_row(self, row: Dict, mate_counts, source_counts):
        """Ritorna (enqueued, quality_filtered, deduped) oppure None se scartato."""
        cfg = self.config
        puzzle_id = row.get("PuzzleId")
        if not puzzle_id:
            logger.warning("Riga puzzle senza PuzzleId: scartata.")
            return None

        uci_moves = str(row["Moves"]).split()
        mate_initial = extract_mate_n(str(row.get("Themes", "")))
        if len(uci_moves) < 2 or mate_initial <= 0:
            return None

        try:
            board = chess.Board(row["FEN"])
        except ValueError as e:
            logger.warning("PuzzleId=%s: FEN non valido (%s).", puzzle_id, e)
            return None

        try:
            first = chess.Move.from_uci(uci_moves[0])
        except ValueError:
            return None
        if first not in board.legal_moves:
            return None
        board.push(first)
        source_counts[mate_initial] += 1

        rating_raw = row.get("Rating")
        rating = float(rating_raw) if pd.notna(rating_raw) else _DEFAULT_PUZZLE_RATING
        game_id = f"{cfg.source_tag}_{puzzle_id}"

        enqueued = quality_filtered = deduped = 0
        seen: set = set()

        for ply_idx, uci in enumerate(uci_moves[1:], start=1):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break

            is_solver_move = ply_idx % 2 == 1
            if is_solver_move:
                key = " ".join(board.fen().split(" ")[:4]) if cfg.dedupe_positions else None
                if key is not None and key in seen:
                    deduped += 1
                elif not position_passes_quality(board, cfg.quality):
                    quality_filtered += 1
                else:
                    if key is not None:
                        seen.add(key)
                    if self._emit(board, move, game_id, ply_idx, rating, mate_initial, puzzle_id, mate_counts):
                        enqueued += 1
            board.push(move)

        return enqueued, quality_filtered, deduped

    def _emit(self, board, move, game_id, ply_idx, rating, mate_initial, puzzle_id, mate_counts) -> bool:
        cfg = self.config
        current_mate = max(1, mate_initial - (ply_idx // 2))
        clock = self._sampler.sample(rating, mate_initial, f"{game_id}:{ply_idx}")
        try:
            data = build_position_data(
                board=board, best_move=move, clock_seconds=clock, rating=rating, game_id=game_id,
                ply=ply_idx, mate_n=current_mate, edge_time_factors=DEFAULT_EDGE_TIME_FACTORS,
            )
        except ValueError as e:
            logger.warning("PuzzleId=%s ply=%d scartata (%s).", puzzle_id, ply_idx, e)
            return False

        self.spool.enqueue(cfg.source_tag, data, mate_initial)
        mate_counts[current_mate] += 1
        if self.debug:
            self.debug.add({
                "problem_id": f"{game_id}_{ply_idx}",
                "puzzle_id": puzzle_id,
                "fen": board.fen(),
                "best_move_uci": move.uci(),
                "mate_n": current_mate,
                "mate_n_window": mate_initial,
                "rating": rating,
                "ply": ply_idx,
                "clock_seconds": float(clock),
                "clock_is_real": False,
                "game_id": game_id,
                "source": cfg.source_tag,
            })
        return True

    def finalize_debug(self, assignment: Dict[str, str]) -> Optional[str]:
        if not self.debug:
            return None
        self.debug.persist()
        return finalize_jsonl(self.debug.pending_path, self.debug.final_path, assignment)
