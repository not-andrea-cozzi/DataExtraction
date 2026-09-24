from __future__ import annotations

import functools
import logging
import os
import time

from common.io import atomic_write_json, read_json

logger = logging.getLogger("pipeline")


class PipelineState:
    def __init__(self, path: str) -> None:
        self.path = path
        self._data: dict = read_json(path, default={}) or {}

    def _save(self) -> None:
        atomic_write_json(self.path, self._data, sort_keys=True)

    def is_done(self, step: str, force: bool = False) -> bool:
        return (not force) and bool(self._data.get(step, {}).get("done", False))

    def mark_done(self, step: str, **meta) -> None:
        self._data[step] = {"done": True, **meta}
        self._save()

    def mark_failed(self, step: str, error: str) -> None:
        entry = self._data.get(step, {})
        entry.update({"done": False, "last_error": error, "last_attempt_ts": time.time()})
        self._data[step] = entry
        self._save()

    def meta(self, step: str) -> dict:
        return self._data.get(step, {})


def retry(max_attempts: int = 3, base_delay: float = 2.0, exceptions=(Exception,)):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            last = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*a, **kw)
                except exceptions as e:
                    last = e
                    if attempt == max_attempts:
                        break
                    delay = base_delay * 2 ** (attempt - 1)
                    logger.warning("%s fallito (%d/%d): %s. Riprovo tra %.1fs.", fn.__name__, attempt, max_attempts, e, delay)
                    time.sleep(delay)
            raise RuntimeError(f"{fn.__name__} fallito dopo {max_attempts} tentativi") from last
        return wrapper
    return deco
