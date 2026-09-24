from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

_HMS = r"\[\s*%{tag}\s+(\d+):(\d+):(\d+(?:\.\d+)?)\s*\]"
_CLK_RE = re.compile(_HMS.format(tag="clk"))
_EMT_RE = re.compile(_HMS.format(tag="emt"))
_TC_INC_RE = re.compile(r"^(\d+)\+(\d+)$")
_TC_BASE_RE = re.compile(r"^(\d+)$")


def _parse_hms(regex: "re.Pattern", comment: Optional[str]) -> Optional[float]:
    if not comment:
        return None
    m = regex.search(comment)
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def parse_clk(comment: Optional[str]) -> Optional[float]:
    """%clk = tempo RIMANENTE (s)."""
    return _parse_hms(_CLK_RE, comment)


def parse_emt(comment: Optional[str]) -> Optional[float]:
    """%emt = tempo SPESO sulla mossa (s)."""
    return _parse_hms(_EMT_RE, comment)


def parse_time_control(tc: Optional[str]) -> Tuple[float, float]:
    """(base_s, inc_s); (0,0) se assente/non riconosciuto."""
    if not tc or tc == "-":
        return 0.0, 0.0
    m = _TC_INC_RE.match(tc)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = _TC_BASE_RE.match(tc)
    if m:
        return float(m.group(1)), 0.0
    return 0.0, 0.0


def compute_move_duration(
    prev_clock: Optional[float], cur_clock: Optional[float], increment: float
) -> Optional[float]:
    if prev_clock is None or cur_clock is None:
        return None
    return max(0.0, prev_clock - cur_clock + increment)


def parse_rating(raw) -> Optional[int]:
    """Strict: solo interi positivi. Accetta int/float/str numerici
    (es. 1500, 1500.0, '1500'); rifiuta '1500?', '?', ''."""
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return None
            value = int(float(raw)) if "." in raw else int(raw)
        else:
            value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def closest_bucket_time(rating: Optional[int], avg_time_by_rating: Dict[int, float]) -> Optional[float]:
    if rating is None or not avg_time_by_rating:
        return None
    closest = min(avg_time_by_rating, key=lambda b: abs(b - rating))
    return avg_time_by_rating[closest]
