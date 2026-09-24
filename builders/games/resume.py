from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Set

from common.io import atomic_write_json, read_json

from .config import SourceSpec

logger = logging.getLogger(__name__)


class ResumeTracker:
    """Avanza il contatore per sorgente solo su id CONTIGUI (sicuro con imap_unordered).
    Stato persistito = numero di partite completate in modo contiguo, per sorgente."""

    def __init__(self, path: str, sources: List[SourceSpec], enabled: bool = True) -> None:
        self.path = path
        saved = self._load() if enabled else {}
        self._base: Dict[str, int] = {}
        self._confirmed: Dict[str, int] = defaultdict(int)
        self._done: Dict[str, Set[int]] = defaultdict(set)
        self._next: Dict[str, int] = {}
        for src in sources:
            key = src.resume_key
            already = saved.get(key, 0)
            if enabled and already:
                src.skip_games += already
                logger.info("[resume] %s: skip_games=%d (%d gia' processate).", key, src.skip_games, already)
            self._base[key] = src.skip_games
            self._next[key] = src.skip_games + 1

    def _load(self) -> Dict[str, int]:
        raw = read_json(self.path, default={}) or {}
        try:
            return {str(k): int(v) for k, v in raw.items()}
        except (TypeError, ValueError):
            logger.warning("[resume] stato illeggibile in %s: riparto da 0.", self.path)
            return {}

    def mark_done(self, key: str, local_id: int) -> None:
        done = self._done[key]
        done.add(local_id)
        nxt = self._next[key]
        while nxt in done:
            done.discard(nxt)
            nxt += 1
            self._confirmed[key] += 1
        self._next[key] = nxt

    def persist(self) -> None:
        try:
            merged = self._load()
            for key, base in self._base.items():
                merged[key] = base + self._confirmed.get(key, 0)
            atomic_write_json(self.path, merged)
        except Exception as e:
            logger.warning("[resume] salvataggio fallito: %s", e)
