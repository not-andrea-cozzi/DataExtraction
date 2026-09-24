from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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


@dataclass
class GamesBuilderConfig:
    sources: List[SourceSpec]
    stockfish_path: str
    mate_range: Tuple[int, int] = (1, 5)
    search_depth: int = 16
    analysis_time: Optional[float] = 0.5

    workers: Optional[int] = None
    threads: int = 1
    hash_mb: int = 8
    multipv: int = 2
    syzygy_path: Optional[str] = None

    stockfish_retry_attempts: int = 2
    stockfish_retry_backoff_seconds: float = 0.5

    candidate_min_legal_moves: int = 1
    candidate_max_legal_moves: Optional[int] = None
    skip_if_in_check: bool = False
    max_piece_count: Optional[int] = 18
    min_material_for_mate_attempt: int = 4
    min_material_diff_for_mate_attempt: int = 3
    require_heavy_piece: bool = True
    skip_forced_moves: bool = False
    skip_trivial_endgame: bool = True
    dedupe_positions: bool = True

    require_clock: bool = True
    default_move_seconds: float = 15.0
    avg_time_by_rating: Dict[int, float] = field(default_factory=dict)
    drop_zero_clock: bool = True
    min_rating: Optional[int] = 1200
    max_rating: Optional[int] = None

    min_ply: int = 8
    ply_sample_step: int = 6

    dense_tail_plies: int = 24
    ply_sample_step_tail: int = 3

    max_positions_per_game: Optional[int] = 5

    only_decisive_games: bool = True
    skip_time_forfeit: bool = True
    min_game_plies: int = 20

    queue_state_path: Optional[str] = None
    shard_size: int = 500

    save_debug_jsonl: bool = True
    debug_jsonl_dir: Optional[str] = None

    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 42

    pool_join_timeout: Optional[float] = 20.0

    auto_resume: bool = True
    resume_state_path: Optional[str] = None
    resume_checkpoint_every: int = 2000
    flush_every_seconds: Optional[float] = None