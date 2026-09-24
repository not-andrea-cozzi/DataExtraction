from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from common.io import atomic_write_json, iter_jsonl, read_json

logger = logging.getLogger(__name__)

_MIN_SIGMA = 0.3
Stat = Tuple[float, float, int]


def _bucket(rating: float, size: int) -> int:
    return int(round(rating / size) * size)


def _finalize(n: int, s: float, ss: float) -> Stat:
    mu = s / n
    return mu, math.sqrt(max(ss / n - mu * mu, 0.0)), n


class ClockStatsBuilder:
    def __init__(self, jsonl_paths: Iterable[str], bucket_size: int = 100, min_count: int = 30,
                 max_seconds: float = 300.0, min_seconds: float = 0.05) -> None:
        self.paths = list(jsonl_paths)
        self.bucket_size = bucket_size
        self.min_count = min_count
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds

    def build(self) -> dict:
        acc_rm: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        acc_r: Dict[int, List[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        acc_g = [0, 0.0, 0.0]
        seen: set = set()

        for rec in iter_jsonl(self.paths):
            pid = rec.get("problem_id")
            if pid is not None:
                if pid in seen:
                    continue
                seen.add(pid)
            if not rec.get("clock_is_real"):
                continue
            clock, rating, mate_n = rec.get("clock_seconds"), rec.get("rating"), rec.get("mate_n")
            if clock is None or rating is None or mate_n is None:
                continue
            clock = float(clock)
            if not (self.min_seconds <= clock <= self.max_seconds):
                continue
            lv = math.log(clock)
            b = _bucket(float(rating), self.bucket_size)
            for cell in (acc_rm[(b, int(mate_n))], acc_r[b], acc_g):
                cell[0] += 1
                cell[1] += lv
                cell[2] += lv * lv

        if acc_g[0] == 0:
            raise ValueError("ClockStatsBuilder: nessun record con clock reale.")

        return {
            "bucket_size": self.bucket_size,
            "global": list(_finalize(int(acc_g[0]), acc_g[1], acc_g[2])),
            "by_rating": {str(b): list(_finalize(int(c[0]), c[1], c[2]))
                          for b, c in sorted(acc_r.items()) if c[0] >= self.min_count},
            "by_rating_mate": {f"{b}|{m}": list(_finalize(int(c[0]), c[1], c[2]))
                               for (b, m), c in sorted(acc_rm.items()) if c[0] >= self.min_count},
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

    def _lookup(self, rating: float, mate_n: Optional[int]) -> Stat:
        b = _bucket(rating, self.bucket_size)
        if self.condition_on_mate_n and mate_n is not None:
            cands = self._buckets_by_mate.get(int(mate_n))
            if cands:
                nearest = min(cands, key=lambda c: abs(c - b))
                if abs(nearest - b) <= self.max_bucket_distance:
                    return self._by_rating_mate[(nearest, int(mate_n))]
        if self._rating_buckets:
            return self._by_rating[min(self._rating_buckets, key=lambda c: abs(c - b))]
        return self._global

    def sample(self, rating: float, mate_n: Optional[int], seed_key: str) -> float:
        if self.mode == "constant":
            value = math.exp(self._global[0])
        else:
            mu, sigma, _ = self._lookup(float(rating), mate_n)
            value = math.exp(random.Random(seed_key).gauss(mu, max(sigma, _MIN_SIGMA)))
        return min(max(value, self.min_seconds), self.cap_seconds)
