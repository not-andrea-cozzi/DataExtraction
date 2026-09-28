from __future__ import annotations

import logging
import math
import random
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from common.io import atomic_write_json, iter_csv, read_json

logger = logging.getLogger(__name__)

_MIN_SIGMA = 0.3
_TRUE_STRINGS = {"true", "1", "yes"}

# Stat = (mu, sigma, n) nello spazio log. Invariato rispetto alla versione
# precedente: chi consuma clock_stats.json non deve cambiare nulla.
Stat = Tuple[float, float, int]

FallbackLevel = str  # "rating_mate" | "rating" | "global"


def _bucket(rating: float, size: int) -> int:
    return int(round(rating / size) * size)


def _finalize(n: int, s: float, ss: float) -> Stat:
    mu = s / n
    return mu, math.sqrt(max(ss / n - mu * mu, 0.0)), n


def _to_bool(v) -> bool:
    """CSV restituisce sempre stringhe: 'True'/'False' vanno castate esplicitamente."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in _TRUE_STRINGS


def _to_float_or_none(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(v) -> Optional[int]:
    f = _to_float_or_none(v)
    return None if f is None else int(f)


def _iqr_bounds(values: List[float], k: float) -> Tuple[float, float]:
    """Bound di Tukey (k * IQR) sui valori dati (qui: log-tempi di una cella).
    Con <4 osservazioni i quartili non sono affidabili: nessun taglio."""
    if len(values) < 4:
        return float("-inf"), float("inf")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    if iqr <= 0:
        return float("-inf"), float("inf")
    return q1 - k * iqr, q3 + k * iqr


class ClockStatsBuilder:

    def __init__(self, csv_paths: Iterable[str], bucket_size: int = 100, min_count: int = 30,
                 max_seconds: float = 300.0, min_seconds: float = 0.05,
                 iqr_k: float = 3.0) -> None:
        self.paths = list(csv_paths)
        self.bucket_size = bucket_size
        self.min_count = min_count
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds
        self.iqr_k = iqr_k

    def _read_valid_log_times(self) -> Tuple[Dict[Tuple[int, int], List[float]], Dict[int, List[float]]]:
        by_rm: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        by_r: Dict[int, List[float]] = defaultdict(list)
        seen: set = set()

        for rec in iter_csv(self.paths):
            pid = rec.get("problem_id")
            if pid is not None and pid != "":
                if pid in seen:
                    continue
                seen.add(pid)
            if not _to_bool(rec.get("clock_is_real")):
                continue

            clock = _to_float_or_none(rec.get("clock_seconds"))
            rating = _to_float_or_none(rec.get("rating"))
            mate_n = _to_int_or_none(rec.get("mate_n"))
            if clock is None or rating is None or mate_n is None:
                continue
            if not (self.min_seconds <= clock <= self.max_seconds):
                continue

            lv = math.log(clock)
            b = _bucket(rating, self.bucket_size)
            by_rm[(b, mate_n)].append(lv)
            by_r[b].append(lv)

        return by_rm, by_r

    def build(self) -> dict:
        by_rm, by_r = self._read_valid_log_times()
        if not by_r:
            raise ValueError("ClockStatsBuilder: nessun record con clock reale.")

        # Filtro IQR per cella (bucket, mate_n) e per bucket puro, indipendenti tra loro.
        rm_clean: Dict[Tuple[int, int], List[float]] = {}
        rm_dropped = 0
        for key, vals in by_rm.items():
            lo, hi = _iqr_bounds(vals, self.iqr_k)
            kept = [v for v in vals if lo <= v <= hi]
            rm_dropped += len(vals) - len(kept)
            if kept:
                rm_clean[key] = kept

        r_clean: Dict[int, List[float]] = {}
        r_dropped = 0
        for key, vals in by_r.items():
            lo, hi = _iqr_bounds(vals, self.iqr_k)
            kept = [v for v in vals if lo <= v <= hi]
            r_dropped += len(vals) - len(kept)
            if kept:
                r_clean[key] = kept

        if rm_dropped or r_dropped:
            logger.info("[clock_stats] outlier IQR (k=%.1f) scartati: by_rating_mate=%d, by_rating=%d",
                        self.iqr_k, rm_dropped, r_dropped)

        all_log_times = [v for vals in by_r.values() for v in vals]
        n_g = len(all_log_times)
        s_g = sum(all_log_times)
        ss_g = sum(v * v for v in all_log_times)

        def _finalize_map(clean: dict) -> dict:
            out = {}
            for key, vals in clean.items():
                n = len(vals)
                if n < self.min_count:
                    continue
                s = sum(vals)
                ss = sum(v * v for v in vals)
                out[key] = _finalize(n, s, ss)
            return out

        by_rating = _finalize_map(r_clean)
        by_rating_mate = _finalize_map(rm_clean)

        return {
            "bucket_size": self.bucket_size,
            "iqr_k": self.iqr_k,
            "global": list(_finalize(n_g, s_g, ss_g)),
            "by_rating": {str(b): list(v) for b, v in sorted(by_rating.items())},
            "by_rating_mate": {f"{b}|{m}": list(v) for (b, m), v in sorted(by_rating_mate.items())},
        }

    def build_and_save(self, out_json: str) -> dict:
        stats = self.build()
        atomic_write_json(out_json, stats, sort_keys=True)
        logger.info("[clock_stats] n=%d, rating=%d, rating x mate=%d -> %s",
                    stats["global"][2], len(stats["by_rating"]), len(stats["by_rating_mate"]), out_json)
        return stats


class ClockSampler:
    def __init__(self, global_stats: Stat, by_rating: Dict[int, Stat],
                 by_rating_mate: Dict[Tuple[int, int], Stat], bucket_size: int = 100,
                 mode: str = "lognormal", condition_on_mate_n: bool = True,
                 min_seconds: float = 0.5, cap_seconds: float = 300.0,
                 max_bucket_distance: int = 300) -> None:
        if mode not in ("lognormal", "constant"):
            raise ValueError(f"ClockSampler mode non valido: {mode}")
        self._global = global_stats
        self._by_rating = by_rating
        self._by_rating_mate = by_rating_mate
        self.bucket_size = bucket_size
        self.mode = mode
        self.condition_on_mate_n = condition_on_mate_n
        self.min_seconds = min_seconds
        self.cap_seconds = cap_seconds
        self.max_bucket_distance = max_bucket_distance
        self._buckets_by_mate: Dict[int, List[int]] = defaultdict(list)
        for (b, m) in by_rating_mate:
            self._buckets_by_mate[m].append(b)
        self._rating_buckets = sorted(by_rating)

    @classmethod
    def from_json(cls, path: str, **kw) -> "ClockSampler":
        raw = read_json(path)
        if raw is None:
            raise FileNotFoundError(path)
        by_rating = {int(k): tuple(v) for k, v in raw["by_rating"].items()}
        by_rm = {}
        for k, v in raw["by_rating_mate"].items():
            b, m = k.split("|")
            by_rm[(int(b), int(m))] = tuple(v)
        return cls(tuple(raw["global"]), by_rating, by_rm, bucket_size=int(raw.get("bucket_size", 100)), **kw)

    @classmethod
    def from_avg_time(cls, avg_time_by_rating: Dict[int, float], sigma: float = 0.9, **kw) -> "ClockSampler":
        if not avg_time_by_rating:
            raise ValueError("from_avg_time: dizionario vuoto.")
        by_rating = {int(b): (math.log(max(float(v), 1e-3)) - sigma * sigma / 2.0, sigma, 0)
                     for b, v in avg_time_by_rating.items()}
        mu = sum(s[0] for s in by_rating.values()) / len(by_rating)
        return cls((mu, sigma, 0), by_rating, {}, **kw)

    def _lookup(self, rating: float, mate_n: Optional[int]) -> Tuple[Stat, FallbackLevel]:
        b = _bucket(rating, self.bucket_size)
        if self.condition_on_mate_n and mate_n is not None:
            cands = self._buckets_by_mate.get(int(mate_n))
            if cands:
                nearest = min(cands, key=lambda c: abs(c - b))
                if abs(nearest - b) <= self.max_bucket_distance:
                    return self._by_rating_mate[(nearest, int(mate_n))], "rating_mate"
        if self._rating_buckets:
            return self._by_rating[min(self._rating_buckets, key=lambda c: abs(c - b))], "rating"
        return self._global, "global"

    def sample_with_source(self, rating: float, mate_n: Optional[int], seed_key: str) -> Tuple[float, FallbackLevel]:
        if self.mode == "constant":
            value = math.exp(self._global[0])
            source: FallbackLevel = "global"
        else:
            (mu, sigma, _), source = self._lookup(float(rating), mate_n)
            value = math.exp(random.Random(seed_key).gauss(mu, max(sigma, _MIN_SIGMA)))
        return min(max(value, self.min_seconds), self.cap_seconds), source

    def sample(self, rating: float, mate_n: Optional[int], seed_key: str) -> float:
        value, _source = self.sample_with_source(rating, mate_n, seed_key)
        return value