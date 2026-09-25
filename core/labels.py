from __future__ import annotations

from typing import Tuple


def mate_in_n_num_classes(mate_range: Tuple[int, int]) -> int:
    """Numero di classi per la label mate-in-n dato un range (lo, hi) inclusivo."""
    lo, hi = mate_range
    if lo < 1 or hi < lo:
        raise ValueError(f"mate_range non valido: {mate_range}")
    return hi - lo + 1


def mate_in_n_label(mate_n: int, mate_range: Tuple[int, int]) -> int:
    lo, hi = mate_range
    if lo < 1 or hi < lo:
        raise ValueError(f"mate_range non valido: {mate_range}")
    if not (lo <= mate_n <= hi):
        raise ValueError(f"mate_n={mate_n} fuori da mate_range={mate_range}.")
    return int(mate_n) - lo