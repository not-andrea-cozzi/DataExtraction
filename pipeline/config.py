from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

T = TypeVar("T")


class ConfigError(Exception):
    pass


def _build(cls: Type[T], raw: Optional[Dict[str, Any]], section: str) -> T:
    """Costruisce un dataclass da dict, rifiutando chiavi sconosciute (typo -> errore)."""
    raw = raw or {}
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(f"[{section}] chiavi sconosciute: {sorted(unknown)}. Valide: {sorted(valid)}")
    return cls(**raw)


@dataclass
class PipelineSection:
    dataset_dir: str = "Dataset"
    puzzles_subfolder: str = "Puzzles"
    merged_subfolder: str = "Train"
    games_subfolder: str = "Games"
    state_file: str = "pipeline_state.json"
    force_recompute: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = None
    step: str = "all"
    use_existing_games: bool = False
    seed: int = 42


@dataclass
class EngineSection:
    stockfish_path: str = ""
    threads: int = 1
    hash_mb: int = 8
    syzygy_path: Optional[str] = None


@dataclass
class RawDataSection:
    games_zst: str = ""
    puzzles_zst: str = ""
    fics_pgn: str = ""
    club_csv: str = ""
    games_source_tag: str = "games_lichess"
    fics_source_tag: str = "fics"
    club_source_tag: str = "club"
    puzzles_source_tag: str = "puzzle"


@dataclass
class TimeStatsSection:
    output_filename: str = "avg_time_by_rating.json"
    max_games: int = 50_000
    bucket_size: int = 100


@dataclass
class ClockStatsSection:
    enabled: bool = True
    output_filename: str = "clock_stats.json"
    bucket_size: int = 100
    min_count: int = 30
    max_seconds: float = 300.0


@dataclass
class GamesSection:
    max_games: Optional[int] = None
    skip_games_part1: int = 0
    fics_max_games: Optional[int] = None
    fics_skip_games: int = 0
    club_max_games: Optional[int] = None
    club_skip_games: int = 0
    club_pgn_col: str = "pgn"

    analysis_time: Optional[float] = 0.5
    search_depth: int = 16
    mate_range_min: int = 1
    mate_range_max: int = 5
    stockfish_retry_attempts: int = 2
    stockfish_retry_backoff_seconds: float = 0.5

    # header
    only_decisive_games: bool = True
    skip_time_forfeit: bool = True
    min_rating: Optional[int] = 1200
    max_rating: Optional[int] = None

    # quality
    max_piece_count: Optional[int] = 18
    min_material_for_mate_attempt: int = 4
    min_material_diff_for_mate_attempt: int = 3
    require_heavy_piece: bool = True
    skip_trivial_endgame: bool = True
    skip_forced_moves: bool = False
    candidate_min_legal_moves: int = 1
    candidate_max_legal_moves: Optional[int] = None
    skip_if_in_check: bool = False
    require_mate_potential: bool = False
    mate_potential_min_attackers: int = 1
    mate_potential_max_escapes: int = 3

    # sampling
    min_game_plies: int = 20
    min_ply: int = 8
    ply_sample_step: int = 6
    dense_tail_plies: int = 24
    ply_sample_step_tail: int = 3
    max_positions_per_game: Optional[int] = 5
    dedupe_positions: bool = True

    # clock
    require_clock: bool = True
    drop_zero_clock: bool = True
    default_move_seconds: float = 15.0

    workers: Optional[int] = None
    checkpoint_every: int = 2000
    pool_join_timeout: float = 20.0
    shard_size: int = 5000
    save_debug_jsonl: bool = True

    # dedup cross-file: evita di ri-analizzare la stessa partita lichess
    # (identificata dall'header [Site]) se compare in file/mesi diversi.
    dedupe_cross_file: bool = False
    game_id_store_filename: str = "lichess_seen_ids.sqlite3"


@dataclass
class PuzzleSection:
    decompressed_csv_filename: str = "lichess_puzzles.csv"
    chunk_size_bytes: int = 1_048_576
    max_puzzles: Optional[int] = None
    max_puzzles_per_theme: Optional[int] = None
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    max_piece_count: Optional[int] = None
    min_material_for_mate_attempt: int = 0
    min_material_diff_for_mate_attempt: int = 0
    require_heavy_piece: bool = False
    skip_trivial_endgame: bool = False
    dedupe_positions: bool = True
    clock_mode: str = "lognormal"
    clock_condition_on_mate_n: bool = True
    clock_min_seconds: float = 0.5
    clock_cap_seconds: float = 300.0
    save_debug_jsonl: bool = True


@dataclass
class SplitsSection:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    output_shard_size: int = 20_000

    @property
    def ratios(self) -> Tuple[float, float, float]:
        return (self.train_ratio, self.val_ratio, self.test_ratio)

    def validate(self) -> None:
        if abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ConfigError(f"[splits] i rapporti devono sommare a 1.0: {self.ratios}")


@dataclass
class CleanSection:
    enabled: bool = True
    input_dir_train: str = ""
    input_dir_val: str = ""
    input_dir_test: Optional[str] = None
    output_dir_train: str = ""
    output_dir_val: str = ""
    output_dir_test: Optional[str] = None
    target_shard_size: int = 8000
    workers: int = 0


@dataclass
class Config:
    pipeline: PipelineSection = field(default_factory=PipelineSection)
    engine: EngineSection = field(default_factory=EngineSection)
    raw_data: RawDataSection = field(default_factory=RawDataSection)
    time_stats: TimeStatsSection = field(default_factory=TimeStatsSection)
    clock_stats: ClockStatsSection = field(default_factory=ClockStatsSection)
    games_pipeline: GamesSection = field(default_factory=GamesSection)
    puzzle_pipeline: PuzzleSection = field(default_factory=PuzzleSection)
    splits: SplitsSection = field(default_factory=SplitsSection)
    clean: CleanSection = field(default_factory=CleanSection)

    # ---- derived paths ----
    @property
    def dataset_dir(self) -> str:
        return self.pipeline.dataset_dir

    def path(self, *parts: str) -> str:
        return os.path.join(self.dataset_dir, *parts)

    @property
    def games_dir(self) -> str:
        return self.path(self.pipeline.games_subfolder)

    @property
    def puzzles_dir(self) -> str:
        return self.path(self.pipeline.puzzles_subfolder)

    @property
    def merged_dir(self) -> str:
        return self.path(self.pipeline.merged_subfolder)

    @property
    def spool_dir(self) -> str:
        return self.path("spool")

    @property
    def resume_path(self) -> str:
        return self.path("games_builder_resume.json")

    @property
    def time_stats_path(self) -> str:
        return self.path(self.time_stats.output_filename)

    @property
    def clock_stats_path(self) -> str:
        return self.path(self.clock_stats.output_filename)

    @property
    def game_id_store_path(self) -> str:
        return self.path(self.games_pipeline.game_id_store_filename)

    @property
    def state_path(self) -> str:
        return self.path(self.pipeline.state_file)


_SECTIONS = {
    "pipeline": PipelineSection, "engine": EngineSection, "raw_data": RawDataSection,
    "time_stats": TimeStatsSection, "clock_stats": ClockStatsSection,
    "games_pipeline": GamesSection, "puzzle_pipeline": PuzzleSection,
    "splits": SplitsSection, "clean": CleanSection,
}


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise ConfigError(f"Config non trovata: {path}")
    if yaml is None:
        raise ConfigError("pyyaml non installato.")
    with open(path, "r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"YAML non valido ({path}): {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError("Il YAML deve essere un dizionario.")

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"Sezioni sconosciute: {sorted(unknown)}. Valide: {sorted(_SECTIONS)}")

    cfg = Config(**{name: _build(cls, raw.get(name), name) for name, cls in _SECTIONS.items()})
    cfg.splits.validate()
    return cfg