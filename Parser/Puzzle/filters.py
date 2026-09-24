from __future__ import annotations

from typing import Dict, List, Optional

import chess

from Utils.compatibility_filters import (
    QualityFilterConfig,
    has_mating_material,
    is_trivially_drawn_endgame,
    mover_has_heavy_piece,
    parse_rating_strict,
)

from .config import PuzzleBuilderConfig


def make_quality_config(cfg: PuzzleBuilderConfig) -> QualityFilterConfig:
    return QualityFilterConfig(
        min_material_for_mate_attempt=cfg.min_material_for_mate_attempt,
        min_material_diff_for_mate_attempt=cfg.min_material_diff_for_mate_attempt,
        require_heavy_piece=cfg.require_heavy_piece,
        skip_trivial_endgame=cfg.skip_trivial_endgame,
    )


def row_passes_rating_filter(row: Dict, cfg: PuzzleBuilderConfig) -> bool:
    if cfg.min_rating is None and cfg.max_rating is None:
        return True
    rating = parse_rating_strict(row.get("Rating"))
    if rating is None:
        return False
    if cfg.min_rating is not None and rating < cfg.min_rating:
        return False
    if cfg.max_rating is not None and rating > cfg.max_rating:
        return False
    return True


def extract_theme_tag(themes: str, themes_wanted: List[str]) -> Optional[str]:
    tokens = set(themes.split())
    for t in themes_wanted:
        if t in tokens:
            return t
    return None


def extract_mate_n(themes: str) -> int:
    for t in themes.split():
        if t.startswith("mateIn"):
            return int(t.replace("mateIn", ""))
    return 0


def position_passes_quality_filters(
    board: "chess.Board",
    cfg: PuzzleBuilderConfig,
    quality_cfg: QualityFilterConfig,
) -> bool:
    if cfg.max_piece_count is not None and len(board.piece_map()) > cfg.max_piece_count:
        return False
    if not has_mating_material(board, quality_cfg):
        return False
    if cfg.require_heavy_piece and not mover_has_heavy_piece(board):
        return False
    if cfg.skip_trivial_endgame and is_trivially_drawn_endgame(board):
        return False
    return True