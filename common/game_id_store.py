from __future__ import annotations

import logging
import os
import sqlite3
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_ids (
    game_id TEXT PRIMARY KEY
) WITHOUT ROWID;
"""


class GameIdStore:
    """Set persistente su disco (SQLite) di ID partita gia' visti.

    Pensato per un solo processo scrivente (il padre, prima di accodare i task
    al pool): niente concorrenza multiprocesso da gestire. Lookup O(log n)
    tramite PRIMARY KEY, nessun caricamento completo in RAM.

    Uso tipico:
        store = GameIdStore(path)
        if not store.contains(site_id):
            store.add(site_id)
            # ... accoda il task ...
        store.close()
    """

    def __init__(self, path: str, batch_commit_every: int = 500) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # check_same_thread=False: l'iterazione (tqdm/imap_unordered) puo' toccare
        # il generator da un thread diverso da quello di __init__. Nessuna vera
        # concorrenza reale: un solo thread logico alla volta usa la connessione.
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(_SCHEMA)
        self._batch_commit_every = max(1, batch_commit_every)
        self._pending = 0
        self._conn.execute("BEGIN;")

    def contains(self, game_id: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM seen_ids WHERE game_id = ? LIMIT 1;", (game_id,))
        return cur.fetchone() is not None

    def add(self, game_id: str) -> None:
        """Idempotente: INSERT OR IGNORE, non solleva su duplicati."""
        self._conn.execute("INSERT OR IGNORE INTO seen_ids (game_id) VALUES (?);", (game_id,))
        self._pending += 1
        if self._pending >= self._batch_commit_every:
            self.commit()

    def add_many(self, game_ids: Iterable[str]) -> None:
        self._conn.executemany(
            "INSERT OR IGNORE INTO seen_ids (game_id) VALUES (?);", ((g,) for g in game_ids)
        )
        self.commit()

    def commit(self) -> None:
        self._conn.execute("COMMIT;")
        self._conn.execute("BEGIN;")
        self._pending = 0

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM seen_ids;")
        return int(cur.fetchone()[0])

    def close(self) -> None:
        try:
            self.commit()
        except sqlite3.Error as e:
            logger.warning("[game_id_store] commit finale fallito (%s): %s", self.path, e)
        finally:
            self._conn.close()

    def __enter__(self) -> "GameIdStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def extract_lichess_site_id(pgn_head: str) -> Optional[str]:
    """Estrae l'ID lichess da '[Site \"https://lichess.org/AbCdEfGh\"]'.

    pgn_head puo' essere l'intero testo pgn o solo le prime righe (l'header
    Site e' sempre tra i primi tag). Ritorna None se non trovato/non lichess.
    """
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