from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from utils.filters import QualityConfig


@dataclass(frozen=True)
class PuzzleClockConfig:
    stats_path: Optional[str] = None
    mode: str = "lognormal"
    condition_on_mate_n: bool = True
    min_seconds: float = 0.5
    cap_seconds: float = 300.0
    avg_time_by_rating: Dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PuzzleBuilderConfig:
    csv_path: str
    mate_range: Tuple[int, int] = (1, 5)
    max_puzzles: Optional[int] = None
    max_puzzles_per_theme: Optional[int] = None
    chunksize: int = 50_000
    source_tag: str = "puzzle"

    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    dedupe_positions: bool = True
    quality: QualityConfig = field(default_factory=QualityConfig)
    clock: PuzzleClockConfig = field(default_factory=PuzzleClockConfig)

    debug_dir: Optional[str] = None
    save_debug_jsonl: bool = True

    def validate(self) -> None:
        lo, hi = self.mate_range
        if lo < 1 or hi < lo:
            raise ValueError(f"mate_range non valido: {self.mate_range}")
        if not os.path.exists(self.csv_path):
            raise ValueError(f"CSV puzzle non trovato: {self.csv_path}")
        for name in ("max_puzzles", "max_puzzles_per_theme"):
            v = getattr(self, name)
            if v is not None and v < 1:
                raise ValueError(f"{name} deve essere >= 1.")
        if self.chunksize < 1:
            raise ValueError("chunksize deve essere >= 1.")
        if self.min_rating is not None and self.max_rating is not None and self.min_rating > self.max_rating:
            raise ValueError("min_rating > max_rating.")
        if self.clock.mode not in ("lognormal", "constant"):
            raise ValueError("clock.mode deve essere 'lognormal' o 'constant'.")
