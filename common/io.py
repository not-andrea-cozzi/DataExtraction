from __future__ import annotations

import json
import os
from typing import Any, Iterable, List


def atomic_write_json(path: str, obj: Any, **kw) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, **kw)
    os.replace(tmp, path)


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


class JsonlAppender:
    """Debug JSONL condiviso da Games/Puzzle: append di batch su .pending,
    poi consolidamento con lo split assegnato."""

    def __init__(self, pending_path: str, final_path: str) -> None:
        self.pending_path = pending_path
        self.final_path = final_path
        os.makedirs(os.path.dirname(os.path.abspath(pending_path)) or ".", exist_ok=True)
        self._buf: List[dict] = []

    def add(self, rec: dict) -> None:
        self._buf.append(rec)

    def persist(self) -> None:
        if not self._buf:
            return
        with open(self.pending_path, "a", encoding="utf-8") as f:
            for r in self._buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._buf.clear()


def finalize_jsonl(pending_path: str, final_path: str, assignment: dict) -> str | None:
    """Aggiunge 'split' a ogni record, scrive final_path, rimuove pending."""
    if not os.path.exists(pending_path):
        return None
    tmp = final_path + ".tmp"
    wrote = missing = 0
    with open(pending_path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            split = assignment.get(rec.get("game_id"))
            if split is None:
                missing += 1
                continue
            rec["split"] = split
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1
    if not wrote:
        os.remove(tmp)
        return None
    os.replace(tmp, final_path)
    os.remove(pending_path)
    return final_path


def iter_jsonl(paths: Iterable[str]):
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
