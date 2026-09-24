from __future__ import annotations

import bz2
import io
import re
import zipfile
from typing import Generator, List, Optional, Tuple

import pandas as pd
import zstandard as zstd

from .config import SourceSpec

_CLK_RE = re.compile(r"\[\s*%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")
_EMT_RE = re.compile(r"\[\s*%emt\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]")


class _ClosingStream:
    def __init__(self, text_stream, raw_file):
        self._text_stream = text_stream
        self._raw_file = raw_file

    def __enter__(self):
        return self._text_stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._text_stream.close()
        finally:
            if self._raw_file is not None:
                self._raw_file.close()
        return False


def _parse_hms(regex: "re.Pattern", comment: str) -> Optional[float]:
    if not comment:
        return None
    m = regex.search(comment)
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def parse_clk(comment: str) -> Optional[float]:
    return _parse_hms(_CLK_RE, comment)


def parse_emt(comment: str) -> Optional[float]:
    return _parse_hms(_EMT_RE, comment)


def parse_time_control(time_control: str) -> Tuple[float, float]:
    if not time_control or time_control == "-":
        return 0.0, 0.0
    m = re.match(r"^(\d+)\+(\d+)$", time_control)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^(\d+)$", time_control)
    if m:
        return float(m.group(1)), 0.0
    return 0.0, 0.0


def compute_move_duration(
    previous_clock: Optional[float], current_clock: Optional[float], increment: float
) -> Optional[float]:
    if previous_clock is None or current_clock is None:
        return None
    return max(0.0, previous_clock - current_clock + increment)


def parse_rating(raw: str) -> Optional[int]:
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in raw if ch.isdigit())
        return int(digits) if digits else None


def open_pgn_text_stream(path: str, kind: str):
    if kind == "lichess":
        raw_file = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(raw_file)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        return _ClosingStream(text_stream, raw_file)

    if kind == "fics":
        if path.lower().endswith(".bz2"):
            raw_file = bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
            return _ClosingStream(raw_file, None)
        raw_file = open(path, "r", encoding="utf-8", errors="replace")
        return _ClosingStream(raw_file, None)

    raise ValueError(f"open_pgn_text_stream non applicabile a kind={kind}")


def iter_pgn_texts(
    text_stream, skip_games: int, max_games: Optional[int]
) -> Generator[Tuple[int, str], None, None]:
    local_id = 0
    yielded = 0
    current_game: List[str] = []

    for line in text_stream:
        if line.startswith("[Event ") and current_game:
            local_id += 1
            if local_id > skip_games:
                yield (local_id, "".join(current_game))
                yielded += 1
                if max_games is not None and yielded >= max_games:
                    return
            current_game = [line]
        else:
            current_game.append(line)

    if current_game:
        local_id += 1
        if local_id > skip_games:
            if max_games is None or yielded < max_games:
                yield (local_id, "".join(current_game))


def iter_club_csv(src: SourceSpec) -> Generator[Tuple[int, str], None, None]:
    if src.path.endswith(".zip"):
        with zipfile.ZipFile(src.path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(src.path)

    series = df[src.pgn_col].dropna().iloc[src.skip_games:]
    if src.max_games is not None:
        series = series.iloc[: src.max_games]
    for local_id, pgn_text in series.items():
        yield (int(local_id) + 1, pgn_text)


def iter_source(src: SourceSpec) -> Generator[Tuple[int, str], None, None]:
    if src.kind == "club":
        yield from iter_club_csv(src)
        return
    with open_pgn_text_stream(src.path, src.kind) as text_stream:
        yield from iter_pgn_texts(text_stream, src.skip_games, src.max_games)