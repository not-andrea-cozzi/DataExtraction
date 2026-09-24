from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import chess
import pandas as pd
from tqdm import tqdm

from Model.PositionGraphSchema import build_position_data
from PositionQueue import PositionQueueRegistry
from Utils.time_edge_weighting import apply_edge_type_time_weighting

from . import debug_io
from .clock import build_clock_sampler
from .config import PuzzleBuilderConfig
from .csv_loader import load_filtered_rows
from .filters import extract_mate_n, make_quality_config, position_passes_quality_filters

logger = logging.getLogger("puzzle_builder")


class PuzzleBuilder:
    def __init__(self, config: PuzzleBuilderConfig):
        self.config = config
        config.validate()

        self._registry = PositionQueueRegistry.instance(
            state_path=config.queue_state_path,
            shard_size=config.shard_size,
        )

        self._clock_sampler = build_clock_sampler(config)
        self._quality_cfg = make_quality_config(config)

        self._debug_records: List[Dict] = []
        self._debug_jsonl_path = None
        self._debug_records_raw_path = None
        if config.save_debug_jsonl:
            if config.debug_jsonl_dir:
                os.makedirs(config.debug_jsonl_dir, exist_ok=True)
                debug_dir = config.debug_jsonl_dir
            else:
                debug_dir = os.path.dirname(config.queue_state_path) if config.queue_state_path else "."
                os.makedirs(debug_dir, exist_ok=True)
            self._debug_jsonl_path = os.path.join(debug_dir, "puzzle_debug.jsonl")
            self._debug_records_raw_path = os.path.join(debug_dir, "puzzle_debug_records.pending.jsonl")

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        all_rows = load_filtered_rows(cfg)
        processed = 0
        accepted_puzzles = 0
        enqueued_positions = 0
        mate_n_counts: Dict[int, int] = defaultdict(int)
        source_mate_n_counts: Dict[int, int] = defaultdict(int)
        quality_filtered_positions = 0
        deduped_positions = 0

        for row in tqdm(all_rows, desc="Costruzione posizioni puzzle"):
            processed += 1
            uci_moves = str(row["Moves"]).split()
            if not uci_moves:
                continue

            try:
                board = chess.Board(row["FEN"])
            except ValueError as e:
                logger.warning(f"PuzzleId={row.get('PuzzleId')}: FEN non valido ({e}), scartato.")
                continue

            mate_n_iniziale = extract_mate_n(str(row.get("Themes", "")))
            if mate_n_iniziale <= 0:
                continue

            source_mate_n_counts[mate_n_iniziale] += 1

            puzzle_id_raw = row.get("PuzzleId")
            if not puzzle_id_raw:
                logger.warning("Riga puzzle senza PuzzleId, scartata.")
                continue

            rating_raw = row.get("Rating")
            puzzle_rating = float(rating_raw) if pd.notna(rating_raw) else 1500.0

            first_move = chess.Move.from_uci(uci_moves[0])
            if first_move not in board.legal_moves:
                continue
            board.push(first_move)

            game_id = f"{cfg.source_tag}_{puzzle_id_raw}"
            window_group_key = mate_n_iniziale

            puzzle_enqueued = 0
            seen_positions: set = set()
            for ply_idx, uci in enumerate(uci_moves[1:], start=1):
                move = chess.Move.from_uci(uci)

                if ply_idx % 2 == 0:
                    if move not in board.legal_moves:
                        break
                    board.push(move)
                    continue

                if move not in board.legal_moves:
                    break

                if cfg.dedupe_positions:
                    position_key = " ".join(board.fen().split(" ")[:4])
                    if position_key in seen_positions:
                        deduped_positions += 1
                        board.push(move)
                        continue
                    seen_positions.add(position_key)

                if not position_passes_quality_filters(board, cfg, self._quality_cfg):
                    quality_filtered_positions += 1
                    board.push(move)
                    continue

                current_mate_n = max(1, mate_n_iniziale - (ply_idx // 2))
                clock_seconds = self._clock_sampler.sample(
                    puzzle_rating, mate_n_iniziale, f"{game_id}:{ply_idx}"
                )

                try:
                    data = build_position_data(
                        board=board,
                        best_move=move,
                        clock_seconds=clock_seconds,
                        rating=puzzle_rating,
                        game_id=game_id,
                        ply=ply_idx,
                        mate_n=int(current_mate_n),
                    )
                    data = apply_edge_type_time_weighting(data)
                except ValueError as e:
                    logger.warning(
                        f"PuzzleId={row.get('PuzzleId')} ply={ply_idx}: scarto la posizione ({e})."
                    )
                    board.push(move)
                    continue

                self._registry.enqueue(
                    source_tag=cfg.source_tag,
                    data=data,
                    group_key=window_group_key,
                )
                puzzle_enqueued += 1
                mate_n_counts[current_mate_n] += 1

                if cfg.save_debug_jsonl:
                    self._debug_records.append({
                        "puzzle_id": row.get("PuzzleId"),
                        "fen": board.fen(),
                        "best_move_uci": move.uci(),
                        "mate_n": current_mate_n,
                        "mate_n_window": window_group_key,
                        "rating": puzzle_rating,
                        "ply_idx": ply_idx,
                        "clock_seconds": float(clock_seconds),
                        "game_id": game_id,
                        "source": cfg.source_tag,
                    })

                board.push(move)

            if puzzle_enqueued > 0:
                accepted_puzzles += 1
                enqueued_positions += puzzle_enqueued

            if (
                cfg.max_positions_per_puzzle is not None
                and enqueued_positions >= cfg.max_positions_per_puzzle
            ):
                break

        self._registry.flush()

        if cfg.save_debug_jsonl and self._debug_records_raw_path:
            self._persist_pending_debug_records()

        self._log_summary(
            processed, accepted_puzzles, enqueued_positions, mate_n_counts,
            source_mate_n_counts, quality_filtered_positions, deduped_positions,
        )

        return {
            "processed_puzzles": processed,
            "accepted_puzzles": accepted_puzzles,
            "enqueued_positions": enqueued_positions,
            "mate_n_counts": dict(mate_n_counts),
            "source_mate_n_counts": dict(source_mate_n_counts),
            "quality_filtered_positions": quality_filtered_positions,
            "deduped_positions": deduped_positions,
        }

    @staticmethod
    def _log_summary(
        processed: int,
        accepted: int,
        enqueued: int,
        mate_n_counts: Dict[int, int],
        source_mate_n_counts: Dict[int, int],
        quality_filtered: int,
        deduped: int,
    ) -> None:
        logger.info(
            "Puzzle: processed=%d accepted=%d enqueued=%d quality_filtered=%d deduped=%d",
            processed, accepted, enqueued, quality_filtered, deduped,
        )
        logger.info("Sorgente per mate_n: %s", dict(sorted(source_mate_n_counts.items())))
        logger.info("Posizioni per mate_n: %s", dict(sorted(mate_n_counts.items())))

    def _persist_pending_debug_records(self) -> None:
        debug_io.persist_pending(self._debug_records, self._debug_records_raw_path)

    @staticmethod
    def write_debug_jsonl_from_pending(
        pending_path: str,
        output_path: str,
        split_assignment: Dict[str, str],
    ) -> Optional[str]:
        return debug_io.write_from_pending(pending_path, output_path, split_assignment)

    def write_debug_jsonl(self, split_assignment: Dict[str, str]) -> Optional[str]:
        if not self.config.save_debug_jsonl or not self._debug_jsonl_path:
            return None
        return debug_io.write_from_memory(self._debug_records, self._debug_jsonl_path, split_assignment)