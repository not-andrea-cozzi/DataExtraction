from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class PuzzleBuilderConfig:
    csv_path: str
    mate_range: Tuple[int, int] = (1, 5)
    max_puzzles: Optional[int] = None
    max_puzzles_per_theme: Optional[int] = None
    avg_time_by_rating: Dict[int, float] = field(default_factory=dict)
    chunksize: int = 50_000

    queue_state_path: Optional[str] = None
    shard_size: int = 500

    save_debug_jsonl: bool = True
    debug_jsonl_dir: Optional[str] = None

    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 42

    max_positions_per_puzzle: Optional[int] = None
    source_tag: str = "puzzle"

    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    max_piece_count: Optional[int] = None
    min_material_for_mate_attempt: int = 0
    min_material_diff_for_mate_attempt: int = 0
    require_heavy_piece: bool = False
    skip_trivial_endgame: bool = False
    dedupe_positions: bool = True

    clock_stats_path: Optional[str] = None
    clock_mode: str = "lognormal"
    clock_condition_on_mate_n: bool = True
    clock_min_seconds: float = 0.5
    clock_cap_seconds: float = 300.0

    def validate(self) -> None:
        if self.mate_range[0] < 1:
            raise ValueError("mate_range deve iniziare da almeno 1.")
        if self.mate_range[1] < self.mate_range[0]:
            raise ValueError("mate_range non valido.")
        if not os.path.exists(self.csv_path):
            raise ValueError(f"CSV puzzle non trovato: {self.csv_path}.")
        if self.max_puzzles is not None and self.max_puzzles < 1:
            raise ValueError("max_puzzles deve essere >= 1 se specificato.")
        if self.max_puzzles_per_theme is not None and self.max_puzzles_per_theme < 1:
            raise ValueError("max_puzzles_per_theme deve essere >= 1 se specificato.")
        if self.chunksize < 1:
            raise ValueError("chunksize deve essere >= 1.")
        if self.min_rating is not None and self.max_rating is not None and self.min_rating > self.max_rating:
            raise ValueError("min_rating non puo' essere maggiore di max_rating.")
        if self.max_piece_count is not None and self.max_piece_count < 2:
            raise ValueError("max_piece_count deve essere >= 2 se specificato.")
        if self.clock_mode not in ("lognormal", "constant"):
            raise ValueError("clock_mode deve essere 'lognormal' o 'constant'.")