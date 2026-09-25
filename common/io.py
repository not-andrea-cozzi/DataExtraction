from __future__ import annotations

import csv
import json
import os
from typing import Any, Iterable, List, Optional


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


class CsvAppender:
    """Debug CSV: append di batch su .pending, poi consolidamento con lo split assegnato.
    fieldnames fissi -> ordine colonne stabile."""

    def __init__(self, pending_path: str, final_path: str, fieldnames: List[str]) -> None:
        self.pending_path = pending_path
        self.final_path = final_path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(os.path.abspath(pending_path)) or ".", exist_ok=True)
        self._buf: List[dict] = []

    def add(self, rec: dict) -> None:
        self._buf.append(rec)

    def persist(self) -> None:
        if not self._buf:
            return
        is_new = not os.path.exists(self.pending_path)
        with open(self.pending_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if is_new:
                w.writeheader()
            for r in self._buf:
                w.writerow(r)
        self._buf.clear()


def finalize_csv(pending_path: str, final_path: str, assignment: dict, fieldnames: List[str]) -> Optional[str]:
    """Aggiunge 'split' a ogni record, scrive final_path, rimuove pending."""
    if not os.path.exists(pending_path):
        return None
    tmp = final_path + ".tmp"
    wrote = 0
    with open(pending_path, "r", encoding="utf-8", newline="") as src, \
            open(tmp, "w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames + ["split"])
        writer.writeheader()
        for rec in reader:
            split = assignment.get(rec.get("game_id"))
            if split is None:
                continue
            rec["split"] = split
            writer.writerow(rec)
            wrote += 1
    if not wrote:
        os.remove(tmp)
        return None
    os.replace(tmp, final_path)
    os.remove(pending_path)
    return final_path


def iter_csv(paths: Iterable[str]):
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8", newline="") as f:
            for rec in csv.DictReader(f):
                yield rec