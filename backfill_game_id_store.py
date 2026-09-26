from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from common.game_id_store import GameIdStore, extract_lichess_site_id
from pipeline.config import ConfigError, load_config
from pipeline.main import setup_logging

logger = logging.getLogger("backfill")


def _iter_lichess_site_ids(path: str):
    """Scandisce il .pgn.zst leggendo solo gli header, senza parsing scacchistico
    ne' Stockfish: stesso costo di iter_source, ordini di grandezza piu' veloce
    dell'analisi originale (nessun engine coinvolto)."""
    import io
    import zstandard as zstd

    with open(path, "rb") as raw:
        text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace")
        current: List[str] = []
        for line in text:
            if line.startswith("[Event ") and current:
                yield "".join(current)
                current = [line]
            else:
                current.append(line)
        if current:
            yield "".join(current)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Backfill dello store dedup con gli ID delle partite lichess gia' processate "
                    "(run precedenti fatte PRIMA di attivare dedupe_cross_file). Non rianalizza nulla: "
                    "legge solo gli header via zstd, nessun coinvolgimento di Stockfish."
    )
    ap.add_argument("--config", default="main.yaml")
    ap.add_argument("--path", default=None,
                     help="Path del .pgn.zst da scandire. Default: raw_data.games_zst dalla config.")
    ap.add_argument("--limit", type=int, default=None,
                     help="Numero di partite da scandire dall'inizio del file (default: tutto il file). "
                          "Usa questo se sai gia' quante ne hai processate finora, es. 155000.")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Errore di configurazione: {e}", file=sys.stderr)
        return 2

    setup_logging(cfg.pipeline.log_level, cfg.pipeline.log_file)

    path = args.path or cfg.raw_data.games_zst
    if not path:
        logger.error("Nessun path: passa --path o valorizza raw_data.games_zst in %s.", args.config)
        return 2

    store_path = cfg.game_id_store_path

    logger.info("[backfill] scandisco %s -> store %s (limit=%s)", path, store_path, args.limit)

    store = GameIdStore(store_path)
    seen = added = no_site = 0
    try:
        for pgn in _iter_lichess_site_ids(path):
            seen += 1
            if args.limit is not None and seen > args.limit:
                seen -= 1
                break
            site_id = extract_lichess_site_id(pgn)
            if site_id is None:
                no_site += 1
                continue
            if not store.contains(site_id):
                store.add(site_id)
                added += 1
            if seen % 10000 == 0:
                logger.info("[backfill] %d partite scandite, %d nuove aggiunte allo store...", seen, added)
    finally:
        store.close()

    logger.info("[backfill] completato: scandite=%d aggiunte=%d senza_site_header=%d totale_store=%s",
                seen, added, no_site, "vedi store")
    return 0


if __name__ == "__main__":
    sys.exit(main())