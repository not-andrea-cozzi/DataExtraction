from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Iterable, Optional, Set

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_ids (
    game_id TEXT PRIMARY KEY
) WITHOUT ROWID;
"""


class GameIdStore:
    """Set persistente su disco (SQLite) con in-memory caching, thread-safety e logging verboso degli errori."""

    def __init__(self, path: str, batch_commit_every: int = 500) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.RLock()

        try:
            self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
                self._conn.execute(_SCHEMA)

                cur = self._conn.execute("SELECT game_id FROM seen_ids;")
                self._seen: Set[str] = {row[0] for row in cur.fetchall()}
                logger.info("[game_id_store] DB caricato: %d ID gia' presenti in %s", len(self._seen), self.path)
        except Exception as e:
            logger.exception("[game_id_store] ERRORE CRITICO nell'inizializzazione del DB (%s): %s", self.path, e)
            raise e

        self._batch_commit_every = max(1, batch_commit_every)
        self._pending: Set[str] = set()

    def contains(self, game_id: str) -> bool:
        with self._lock:
            return game_id in self._seen

    def add(self, game_id: str) -> None:
        with self._lock:
            if game_id not in self._seen:
                self._seen.add(game_id)
                self._pending.add(game_id)
                if len(self._pending) >= self._batch_commit_every:
                    self.commit()

    def add_many(self, game_ids: Iterable[str]) -> None:
        with self._lock:
            for gid in game_ids:
                if gid not in self._seen:
                    self._seen.add(gid)
                    self._pending.add(gid)
            self.commit()

    def commit(self) -> None:
        with self._lock:
            if not self._pending:
                return

            try:
                if not self._conn.in_transaction:
                    self._conn.execute("BEGIN IMMEDIATE;")

                self._conn.executemany(
                    "INSERT OR IGNORE INTO seen_ids (game_id) VALUES (?);",
                    ((gid,) for gid in self._pending),
                )
                self._conn.execute("COMMIT;")
                self._pending.clear()
            except Exception as e:
                logger.exception("[game_id_store] ERRORE durante il commit su %s: %s", self.path, e)
                try:
                    self._conn.execute("ROLLBACK;")
                except sqlite3.Error as rb_err:
                    logger.exception("[game_id_store] ERRORE durante il ROLLBACK su %s: %s", self.path, rb_err)
                raise e

    def count(self) -> int:
        with self._lock:
            return len(self._seen)

    def close(self) -> None:
        with self._lock:
            try:
                if self._pending:
                    self.commit()
                self._conn.close()
            except Exception as e:
                logger.exception("[game_id_store] ERRORE durante la chiusura del DB (%s): %s", self.path, e)

    def __enter__(self) -> "GameIdStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            logger.exception("[game_id_store] Eccezione intercettata nel blocco context: %s", exc)
        self.close()


def extract_lichess_site_id(pgn_head: str) -> Optional[str]:
    """Estrae l'ID lichess dal tag Header PGN '[Site "https://lichess.org/AbCdEfGh"]'."""
    idx = pgn_head.find('[Site "')
    if idx == -1:
        return None
    start = idx + len('[Site "')
    end = pgn_head.find('"]', start)
    if end == -1:
        return None
    url = pgn_head[start:end]
    site_id = url.rsplit("/", 1)[-1].strip()
    if not site_id or "/" in site_id:
        return None
    return site_id