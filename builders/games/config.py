from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.filters import HeaderFilterConfig, QualityConfig


@dataclass
class SourceSpec:
    kind: str
    path: str
    pgn_col: str = "pgn"
    skip_games: int = 0
    max_games: Optional[int] = None
    tag: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in ("lichess", "fics", "club"):
            raise ValueError(f"SourceSpec.kind non valido: {self.kind}")
        if self.tag is None:
            self.tag = self.kind

    @property
    def resume_key(self) -> str:
        return f"{self.kind}:{self.path}"


@dataclass(frozen=True)
class EngineConfig:
    stockfish_path: str
    threads: int = 1
    hash_mb: int = 8
    multipv: int = 2
    syzygy_path: Optional[str] = None
    analysis_time: Optional[float] = 0.5
    search_depth: int = 16
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.5


@dataclass(frozen=True)
class SamplingConfig:
    min_game_plies: int = 20
    min_ply: int = 8
    ply_sample_step: int = 6
    dense_tail_plies: int = 24
    ply_sample_step_tail: int = 3
    max_positions_per_game: Optional[int] = 5
    dedupe_positions: bool = True


@dataclass(frozen=True)
class ClockConfig:
    require_clock: bool = True
    drop_zero_clock: bool = True
    default_move_seconds: float = 15.0
    avg_time_by_rating: Dict[int, float] = field(default_factory=dict)


@dataclass
class GamesBuilderConfig:
    sources: List[SourceSpec]
    engine: EngineConfig
    mate_range: Tuple[int, int] = (1, 5)
    header: HeaderFilterConfig = field(default_factory=HeaderFilterConfig)
    quality: QualityConfig = field(default_factory=lambda: QualityConfig(
        min_material_for_mate_attempt=0, min_material_diff_for_mate_attempt=0,
        require_heavy_piece=False, skip_trivial_endgame=True, max_piece_count=None,
    ))
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    clock: ClockConfig = field(default_factory=ClockConfig)

    workers: Optional[int] = None
    pool_join_timeout: float = 20.0

    resume_state_path: str = "games_builder_resume.json"
    auto_resume: bool = True
    resume_checkpoint_every: int = 2000
    flush_every_seconds: Optional[float] = None

    debug_dir: Optional[str] = None
    save_debug_jsonl: bool = True

    def validate(self) -> None:
        lo, hi = self.mate_range
        if lo < 1 or hi < lo:
            raise ValueError(f"mate_range non valido: {self.mate_range}")
        tags = [s.tag for s in self.sources]
        if len(tags) != len(set(tags)):
            raise ValueError(f"Tag sorgente duplicati: {tags}")
        if not self.sources:
            raise ValueError("Nessuna sorgente.")
