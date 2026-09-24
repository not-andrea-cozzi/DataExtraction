from __future__ import annotations

import bz2
import io
import zipfile
from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple

import pandas as pd
import zstandard as zstd

from .config import SourceSpec


@contextmanager
def _open_text(path: str, kind: str):
    if kind == "lichess":
        raw = open(path, "rb")
        try:
            text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace")
            try:
                yield text
            finally:
                text.close()
        finally:
            raw.close()
    elif kind == "fics":
        f = (bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
             if path.lower().endswith(".bz2")
             else open(path, "r", encoding="utf-8", errors="replace"))
        try:
            yield f
        finally:
            f.close()
    else:
        raise ValueError(f"kind non testuale: {kind}")


def _iter_pgn_texts(stream, skip: int, max_games: Optional[int]) -> Generator[Tuple[int, str], None, None]:
    """Yield (local_id assoluto 1-based, pgn). Salta le prime `skip` partite."""
    local_id = yielded = 0
    current: List[str] = []

    def emit():
        nonlocal yielded
        if local_id > skip:
            yielded += 1
            return (local_id, "".join(current))
        return None

    for line in stream:
        if line.startswith("[Event ") and current:
            local_id += 1
            item = emit()
            if item:
                yield item
                if max_games is not None and yielded >= max_games:
                    return
            current = [line]
        else:
            current.append(line)
    if current:
        local_id += 1
        if max_games is None or yielded < max_games:
            item = emit()
            if item:
                yield item


def _iter_club_csv(src: SourceSpec) -> Generator[Tuple[int, str], None, None]:
    """local_id = posizione ordinale (1-based) tra le righe valide -> contiguo,
    compatibile con il resume basato su id contigui."""
    if src.path.endswith(".zip"):
        with zipfile.ZipFile(src.path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"Nessun CSV in {src.path}")
            with zf.open(names[0]) as f:
                df = pd.read_csv(f, usecols=[src.pgn_col])
    else:
        df = pd.read_csv(src.path, usecols=[src.pgn_col])

    series = df[src.pgn_col].dropna().reset_index(drop=True)
    end = None if src.max_games is None else src.skip_games + src.max_games
    for i in range(src.skip_games, len(series) if end is None else min(end, len(series))):
        yield i + 1, series.iloc[i]


def iter_source(src: SourceSpec) -> Generator[Tuple[int, str], None, None]:
    if src.kind == "club":
        yield from _iter_club_csv(src)
        return
    with _open_text(src.path, src.kind) as stream:
        yield from _iter_pgn_texts(stream, src.skip_games, src.max_games)
