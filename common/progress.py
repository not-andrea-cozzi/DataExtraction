from __future__ import annotations

from typing import Iterable, Optional

from tqdm import tqdm


def wrap_iter(it: Iterable, desc: str = "", unit: str = "it", total: Optional[int] = None):
    return tqdm(it, desc=desc, unit=unit, total=total)
