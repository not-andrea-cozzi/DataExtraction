from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .config import PuzzleBuilderConfig
from .filters import extract_theme_tag, row_passes_rating_filter

logger = logging.getLogger("puzzle_builder")


def load_filtered_rows(cfg: PuzzleBuilderConfig) -> List[Dict]:
    lo, hi = cfg.mate_range
    themes_wanted = [f"mateIn{n}" for n in range(lo, hi + 1)]
    theme_pattern = "|".join(themes_wanted)

    reader = pd.read_csv(cfg.csv_path, chunksize=cfg.chunksize)

    if cfg.max_puzzles_per_theme is not None:
        return _load_stratified(cfg, reader, themes_wanted, theme_pattern)
    return _load_flat(cfg, reader, theme_pattern)


def _load_flat(cfg: PuzzleBuilderConfig, reader, theme_pattern: str) -> List[Dict]:
    rows: List[Dict] = []
    pbar = tqdm(desc="Lettura CSV puzzle (flat)", unit=" righe valide")
    for chunk in reader:
        mask = chunk["Themes"].str.contains(theme_pattern, na=False)
        filtered = chunk[mask]
        for record in filtered.to_dict("records"):
            if not row_passes_rating_filter(record, cfg):
                continue
            rows.append(record)
            pbar.update(1)
        if cfg.max_puzzles and len(rows) >= cfg.max_puzzles:
            rows = rows[: cfg.max_puzzles]
            break
    pbar.close()
    return rows


def _load_stratified(
    cfg: PuzzleBuilderConfig, reader, themes_wanted: List[str], theme_pattern: str
) -> List[Dict]:
    cap = cfg.max_puzzles_per_theme
    rows_by_theme: Dict[str, List[Dict]] = {t: [] for t in themes_wanted}

    pbar = tqdm(desc="Lettura CSV puzzle (stratificato)", unit=" righe valide")
    for chunk in reader:
        mask = chunk["Themes"].str.contains(theme_pattern, na=False)
        filtered = chunk[mask]
        if filtered.empty:
            continue

        for record in filtered.to_dict("records"):
            if not row_passes_rating_filter(record, cfg):
                continue

            theme_found = extract_theme_tag(str(record.get("Themes", "")), themes_wanted)
            if theme_found is None:
                continue
            bucket = rows_by_theme[theme_found]
            if len(bucket) < cap:
                bucket.append(record)
                pbar.update(1)

        if all(len(v) >= cap for v in rows_by_theme.values()):
            break
    pbar.close()

    all_rows: List[Dict] = []
    for theme in themes_wanted:
        found = len(rows_by_theme[theme])
        if found < cap:
            logger.warning(f"Tema '{theme}': solo {found}/{cap} puzzle trovati nel CSV.")
        all_rows.extend(rows_by_theme[theme])

    if cfg.max_puzzles is not None and len(all_rows) > cfg.max_puzzles:
        logger.info(
            f"Campionamento stratificato ha prodotto {len(all_rows)} righe, "
            f"troncate a max_puzzles={cfg.max_puzzles}."
        )
        all_rows = all_rows[: cfg.max_puzzles]

    logger.info(
        "Distribuzione puzzle sorgente per tema: "
        + ", ".join(f"{t}={len(rows_by_theme[t])}" for t in themes_wanted)
    )
    return all_rows