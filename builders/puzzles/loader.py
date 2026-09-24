from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from utils.pgn_time import parse_rating

from .config import PuzzleBuilderConfig

logger = logging.getLogger(__name__)


def extract_theme_tag(themes: str, wanted: List[str]) -> Optional[str]:
    tokens = set(themes.split())
    return next((t for t in wanted if t in tokens), None)


def extract_mate_n(themes: str) -> int:
    for t in themes.split():
        if t.startswith("mateIn"):
            try:
                return int(t[len("mateIn"):])
            except ValueError:
                return 0
    return 0


def _rating_ok(row: Dict, cfg: PuzzleBuilderConfig) -> bool:
    if cfg.min_rating is None and cfg.max_rating is None:
        return True
    r = parse_rating(row.get("Rating"))
    if r is None:
        return False
    return (cfg.min_rating is None or r >= cfg.min_rating) and (cfg.max_rating is None or r <= cfg.max_rating)


def load_rows(cfg: PuzzleBuilderConfig) -> List[Dict]:
    lo, hi = cfg.mate_range
    wanted = [f"mateIn{n}" for n in range(lo, hi + 1)]
    pattern = "|".join(wanted)
    reader = pd.read_csv(cfg.csv_path, chunksize=cfg.chunksize)
    if cfg.max_puzzles_per_theme is not None:
        return _load_stratified(cfg, reader, wanted, pattern)
    return _load_flat(cfg, reader, pattern)


def _load_flat(cfg: PuzzleBuilderConfig, reader, pattern: str) -> List[Dict]:
    rows: List[Dict] = []
    with tqdm(desc="CSV puzzle (flat)", unit=" righe") as pbar:
        for chunk in reader:
            for rec in chunk[chunk["Themes"].str.contains(pattern, na=False)].to_dict("records"):
                if _rating_ok(rec, cfg):
                    rows.append(rec)
                    pbar.update(1)
            if cfg.max_puzzles and len(rows) >= cfg.max_puzzles:
                return rows[: cfg.max_puzzles]
    return rows


def _load_stratified(cfg: PuzzleBuilderConfig, reader, wanted: List[str], pattern: str) -> List[Dict]:
    cap = cfg.max_puzzles_per_theme
    by_theme: Dict[str, List[Dict]] = {t: [] for t in wanted}
    with tqdm(desc="CSV puzzle (stratificato)", unit=" righe") as pbar:
        for chunk in reader:
            for rec in chunk[chunk["Themes"].str.contains(pattern, na=False)].to_dict("records"):
                if not _rating_ok(rec, cfg):
                    continue
                theme = extract_theme_tag(str(rec.get("Themes", "")), wanted)
                if theme is None or len(by_theme[theme]) >= cap:
                    continue
                by_theme[theme].append(rec)
                pbar.update(1)
            if all(len(v) >= cap for v in by_theme.values()):
                break

    rows: List[Dict] = []
    for t in wanted:
        if len(by_theme[t]) < cap:
            logger.warning("Tema '%s': %d/%d puzzle trovati.", t, len(by_theme[t]), cap)
        rows.extend(by_theme[t])
    if cfg.max_puzzles is not None and len(rows) > cfg.max_puzzles:
        rows = rows[: cfg.max_puzzles]
    logger.info("Puzzle per tema: %s", {t: len(by_theme[t]) for t in wanted})
    return rows
