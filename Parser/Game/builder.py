from __future__ import annotations

import copy
import dataclasses
import io
import json
import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple

import chess
import chess.pgn

from Common.progress import wrap_iter
from Model.PositionGraphSchema import build_position_data
from PositionQueue import PositionQueueRegistry
from DatasetPipeline.Utils.ipc_safe_data import (
    encode_for_ipc,
    decode_from_ipc,
    harden_process_for_ipc,
)
from DatasetPipeline.Utils.time_edge_weighting import apply_edge_type_time_weighting

from . import engine
from .config import GamesBuilderConfig, SourceSpec
from .filters import (
    get_candidate_legal_moves,
    has_mating_material,
    headers_are_eligible,
    is_trivially_drawn_endgame,
    mover_has_heavy_piece,
)
from .pgn_utils import (
    compute_move_duration,
    iter_source,
    parse_clk,
    parse_emt,
    parse_rating,
    parse_time_control,
)

logger = logging.getLogger(__name__)

_WORKER_CFG: Optional[GamesBuilderConfig] = None


def _pool_initializer(cfg: GamesBuilderConfig) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg
    engine.init_worker(cfg.stockfish_path, cfg.threads, cfg.hash_mb, cfg.syzygy_path)


def _worker_entry(args: Tuple[int, str, str, str]) -> Tuple[int, str, bytes]:
    return _analyse_game(_WORKER_CFG, args)


def _closest_bucket_time(cfg: GamesBuilderConfig, rating: Optional[int]) -> Optional[float]:
    if rating is None or not cfg.avg_time_by_rating:
        return None
    closest = min(cfg.avg_time_by_rating.keys(), key=lambda b: abs(b - rating))
    return cfg.avg_time_by_rating[closest]


def _analyse_game(cfg: GamesBuilderConfig, args: Tuple[int, str, str, str]) -> Tuple[int, str, bytes]:
    game_id, pgn_text, source_tag, resume_key = args

    empty_payload = encode_for_ipc([])
    if not engine.is_engine_ready():
        return game_id, resume_key, empty_payload

    try:
        pgn_io = io.StringIO(pgn_text)
        headers = chess.pgn.read_headers(pgn_io)
    except Exception as e:
        logger.warning(
            "[GamesBuilder] Worker: PGN illeggibile per game_id locale=%s (%s: %s), partita scartata.",
            game_id, type(e).__name__, e,
        )
        return game_id, resume_key, empty_payload

    if headers is None or headers.get("Variant", "Standard").lower() not in ("standard", "normal"):
        return game_id, resume_key, empty_payload
    if not headers_are_eligible(headers, cfg):
        return game_id, resume_key, empty_payload

    pgn_io.seek(0)
    try:
        game = chess.pgn.read_game(pgn_io)
    except Exception:
        return game_id, resume_key, empty_payload
    if game is None:
        return game_id, resume_key, empty_payload

    try:
        game_end_ply = game.end().ply()
    except Exception:
        return game_id, resume_key, empty_payload
    if game_end_ply < cfg.min_game_plies:
        return game_id, resume_key, empty_payload

    base_time, increment = parse_time_control(game.headers.get("TimeControl", ""))
    mover_rating = {
        chess.WHITE: parse_rating(game.headers.get("WhiteElo", "")),
        chess.BLACK: parse_rating(game.headers.get("BlackElo", "")),
    }
    previous_clock = {
        chess.WHITE: base_time if base_time > 0 else None,
        chess.BLACK: base_time if base_time > 0 else None,
    }

    records: List[Dict[str, Any]] = []
    mate_lo, mate_hi = cfg.mate_range
    positions_analysed = 0
    seen_positions: set = set()
    full_game_id = f"{source_tag}_{game_id}"
    window_group_key: Optional[int] = None

    board = game.board()
    node = game

    try:
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            ply = node.ply()
            mover_color = board.turn
            comment = next_node.comment or ""

            emt_seconds = parse_emt(comment)
            current_clock = parse_clk(comment)

            if emt_seconds is not None:
                move_duration = emt_seconds
                duration_is_real = True
                clock_source = "real_emt"
            else:
                move_duration = compute_move_duration(previous_clock[mover_color], current_clock, increment)
                duration_is_real = move_duration is not None
                clock_source = "real_clk" if duration_is_real else None

            if current_clock is not None:
                previous_clock[mover_color] = current_clock

            is_in_dense_tail = cfg.dense_tail_plies > 0 and (game_end_ply - ply) <= cfg.dense_tail_plies
            effective_step = cfg.ply_sample_step_tail if is_in_dense_tail else cfg.ply_sample_step

            skip = (
                ply < cfg.min_ply
                or (ply - cfg.min_ply) % effective_step != 0
                or (cfg.require_clock and not duration_is_real)
            )

            if not skip:
                if cfg.max_positions_per_game is not None and positions_analysed >= cfg.max_positions_per_game:
                    break

                if cfg.dedupe_positions:
                    position_key = " ".join(board.fen().split(" ")[:4])
                    if position_key in seen_positions:
                        skip = True
                    else:
                        seen_positions.add(position_key)

            if not skip:
                rec = _evaluate_position(
                    cfg, board, move, ply, mover_rating[mover_color],
                    duration_is_real, move_duration, clock_source,
                    full_game_id, source_tag, (mate_lo, mate_hi),
                    window_group_key,
                )
                positions_analysed += rec["analysed"]
                if rec["record"] is not None:
                    records.append(rec["record"])
                    if window_group_key is None:
                        window_group_key = rec["record"]["debug"]["mate_n_window"]

            board.push(move)
            node = next_node

    except Exception as e:
        logger.warning(
            "[GamesBuilder] Worker: eccezione durante l'analisi di game_id=%s (%s: %s); "
            "partita troncata, %d posizioni gia' raccolte mantenute.",
            full_game_id, type(e).__name__, e, len(records),
            exc_info=True,
        )

    try:
        payload = encode_for_ipc(records)
    except Exception as e:
        logger.error(
            "[GamesBuilder] Worker: impossibile serializzare i risultati per game_id=%s "
            "(%s: %s); partita scartata (%d posizioni perse).",
            full_game_id, type(e).__name__, e, len(records),
            exc_info=True,
        )
        payload = empty_payload

    return game_id, resume_key, payload


def _evaluate_position(
    cfg: GamesBuilderConfig,
    board: "chess.Board",
    _played_move,
    ply: int,
    mover_rating_val: Optional[int],
    duration_is_real: bool,
    move_duration: Optional[float],
    clock_source: Optional[str],
    full_game_id: str,
    source_tag: str,
    mate_range: Tuple[int, int],
    window_group_key: Optional[int],
) -> Dict[str, Any]:
    """Ritorna {"analysed": 0|1, "record": dict|None}."""
    none = {"analysed": 0, "record": None}
    mate_lo, mate_hi = mate_range

    legal_moves = get_candidate_legal_moves(board, cfg)
    if legal_moves is None:
        return none
    if cfg.skip_forced_moves and len(legal_moves) == 1:
        return none
    if cfg.require_heavy_piece and not mover_has_heavy_piece(board):
        return none
    if not has_mating_material(board, cfg):
        return none
    if cfg.skip_trivial_endgame and is_trivially_drawn_endgame(board):
        return none
    if mover_rating_val is None:
        return none
    if engine.syzygy_says_no_mate(board):
        return none

    info = engine.analyse_position(board, cfg)
    analysed = {"analysed": 1, "record": None}
    if not info:
        return analysed

    best_info = info[0]
    score = best_info.get("score")
    if score is None:
        return analysed

    relative_score = score.relative
    if not relative_score.is_mate():
        return analysed

    mate_n = relative_score.mate()
    if mate_n is None or not (mate_n > 0 and mate_lo <= mate_n <= mate_hi):
        return analysed
    if engine.second_line_ties_mate(info, int(mate_n)):
        return analysed

    pv = best_info.get("pv")
    if not pv:
        return analysed
    best_move = pv[0]
    if best_move not in legal_moves:
        return analysed

    if duration_is_real:
        clock_seconds = move_duration
    else:
        bucket_time = _closest_bucket_time(cfg, mover_rating_val)
        if bucket_time is not None:
            clock_seconds = bucket_time
            clock_source = "rating_bucket"
        else:
            clock_seconds = cfg.default_move_seconds
            clock_source = "default_constant"

    if cfg.drop_zero_clock and clock_seconds == 0.0 and not duration_is_real:
        return analysed

    try:
        data = build_position_data(
            board=board,
            best_move=best_move,
            clock_seconds=clock_seconds,
            rating=float(mover_rating_val),
            game_id=full_game_id,
            ply=ply,
            mate_n=int(mate_n),
        )
        data = apply_edge_type_time_weighting(data)
    except ValueError:
        return analysed

    debug_entry = {
        "problem_id": f"{full_game_id}_{ply}",
        "fen": board.fen(),
        "best_move_uci": best_move.uci(),
        "mate_n": int(mate_n),
        "mate_n_window": window_group_key if window_group_key is not None else int(mate_n),
        "ply": int(ply),
        "source": source_tag,
        "clock_source": clock_source or "unknown",
        "clock_seconds": float(clock_seconds),
        "clock_is_real": bool(duration_is_real),
        "rating": mover_rating_val,
        "game_id": full_game_id,
    }
    return {"analysed": 1, "record": {"data": data, "debug": debug_entry}}


class GamesBuilder:
    def __init__(self, config: GamesBuilderConfig):
        config = dataclasses.replace(config, sources=[copy.copy(src) for src in config.sources])
        self.config = config
        self._validate_config()

        self._registry = PositionQueueRegistry.instance(
            state_path=config.queue_state_path,
            shard_size=config.shard_size,
        )

        cpu_count = os.cpu_count() or 2
        self._workers = config.workers or max(1, cpu_count - 1)

        self._debug_records: List[Dict] = []
        self._debug_persisted_count = 0
        self._debug_jsonl_path = None
        self._debug_records_raw_path = None
        if config.save_debug_jsonl:
            if config.debug_jsonl_dir:
                os.makedirs(config.debug_jsonl_dir, exist_ok=True)
                debug_dir = config.debug_jsonl_dir
            else:
                debug_dir = os.path.dirname(config.queue_state_path) if config.queue_state_path else "."
                os.makedirs(debug_dir, exist_ok=True)
            self._debug_jsonl_path = os.path.join(debug_dir, "games_debug.jsonl")
            self._debug_records_raw_path = os.path.join(debug_dir, "games_debug_records.pending.jsonl")

        self._resume_state_path = config.resume_state_path
        if self._resume_state_path is None:
            state_dir = os.path.dirname(config.queue_state_path) if config.queue_state_path else "."
            os.makedirs(state_dir, exist_ok=True)
            self._resume_state_path = os.path.join(state_dir, "games_builder_resume.json")

        saved_progress = self._load_resume_state() if config.auto_resume else {}
        self._resume_base: Dict[str, int] = {}
        self._resume_confirmed: Dict[str, int] = defaultdict(int)
        self._resume_done_ids: Dict[str, set] = defaultdict(set)
        self._resume_next_id: Dict[str, int] = {}
        for src in config.sources:
            key = self._resume_key(src)
            already_done = saved_progress.get(key, 0)
            if config.auto_resume and already_done:
                src.skip_games += already_done
                logger.info(
                    "[GamesBuilder] Resume attivo per %s: skip_games portato a %d "
                    "(%d gia' processate in run precedenti).",
                    key, src.skip_games, already_done,
                )
            self._resume_base[key] = src.skip_games
            self._resume_next_id[key] = src.skip_games + 1

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if "_registry" in state:
            del state["_registry"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._registry = PositionQueueRegistry.instance(
            state_path=self.config.queue_state_path,
            shard_size=self.config.shard_size,
        )

    def _validate_config(self) -> None:
        tags_seen: Dict[str, SourceSpec] = {}
        for src in self.config.sources:
            if src.tag in tags_seen:
                raise ValueError(
                    f"Tag sorgente duplicato: '{src.tag}' usato sia da "
                    f"'{tags_seen[src.tag].path}' che da '{src.path}'."
                )
            tags_seen[src.tag] = src

    def _mark_done(self, key: str, local_id: int) -> None:
        """Avanza il contatore solo su id contigui: sicuro con imap_unordered."""
        done = self._resume_done_ids[key]
        done.add(local_id)
        nxt = self._resume_next_id[key]
        while nxt in done:
            done.discard(nxt)
            nxt += 1
            self._resume_confirmed[key] += 1
        self._resume_next_id[key] = nxt

    @staticmethod
    def _resume_key(src: SourceSpec) -> str:
        return f"{src.kind}:{src.path}"

    def _load_resume_state(self) -> Dict[str, int]:
        if not os.path.exists(self._resume_state_path):
            return {}
        try:
            with open(self._resume_state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {str(k): int(v) for k, v in raw.items()}
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            logger.warning(
                "[GamesBuilder] Stato di resume in %s illeggibile (%s), riparto senza resume.",
                self._resume_state_path, e,
            )
            return {}

    def _persist_resume_state(self) -> None:
        try:
            merged = self._load_resume_state()
            for key, base in self._resume_base.items():
                merged[key] = base + self._resume_confirmed.get(key, 0)

            state_dir = os.path.dirname(os.path.abspath(self._resume_state_path)) or "."
            os.makedirs(state_dir, exist_ok=True)
            tmp_path = self._resume_state_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._resume_state_path)
        except Exception as e:
            logger.warning("[GamesBuilder] Impossibile salvare lo stato di resume: %s", e)

    def _iter_all_tasks(self) -> Generator[Tuple[int, str, str, str], None, None]:
        for src in self.config.sources:
            resume_key = self._resume_key(src)
            for local_id, pgn_text in iter_source(src):
                yield (local_id, pgn_text, src.tag, resume_key)

    def _count_tasks_estimate(self) -> Optional[int]:
        total = 0
        for src in self.config.sources:
            if src.max_games is not None:
                total += src.max_games
            else:
                return None
        return total

    def run(self) -> Dict[str, Any]:
        cfg = self.config
        harden_process_for_ipc()

        processed_games = accepted_games = enqueued_positions = 0
        mate_n_counts: Dict[int, int] = defaultdict(int)
        source_counts: Dict[str, int] = defaultdict(int)
        clock_source_counts: Dict[str, int] = defaultdict(int)
        skipped_games_on_parent_error = 0

        pool = mp.Pool(
            processes=self._workers,
            initializer=_pool_initializer,
            initargs=(cfg,),
        )

        estimate = self._count_tasks_estimate()
        last_flush_time = time.monotonic()
        shutdown_in_progress = threading.Event()

        def _panic_kill() -> None:
            for proc in getattr(pool, "_pool", []):
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            os._exit(1)

        def _sigint_handler(signum, frame) -> None:
            if shutdown_in_progress.is_set():
                _panic_kill()
            shutdown_in_progress.set()
            raise KeyboardInterrupt

        previous_sigint = signal.signal(signal.SIGINT, _sigint_handler)

        def _shutdown_pool(graceful_first: bool) -> None:
            try:
                if graceful_first:
                    timeout = cfg.pool_join_timeout if cfg.pool_join_timeout is not None else 15.0
                    deadline = time.monotonic() + timeout
                    for proc in pool._pool:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        proc.join(timeout=remaining)

                pool.terminate()

                kill_deadline = time.monotonic() + 5.0
                for proc in pool._pool:
                    remaining = kill_deadline - time.monotonic()
                    proc.join(timeout=max(remaining, 0.1))

                for proc in pool._pool:
                    if proc.is_alive():
                        logger.warning(
                            "[GamesBuilder] Worker pid=%s ancora vivo dopo terminate(): invio SIGKILL diretto.",
                            proc.pid,
                        )
                        try:
                            os.kill(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except Exception as e:
                            logger.warning("[GamesBuilder] SIGKILL su pid=%s fallito: %s", proc.pid, e)

                for proc in pool._pool:
                    proc.join(timeout=2.0)
            except Exception:
                logger.exception(
                    "[GamesBuilder] Errore imprevisto nello shutdown del pool: forzo SIGKILL su tutti i worker."
                )
                for proc in getattr(pool, "_pool", []):
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass

        try:
            task_stream = self._iter_all_tasks()
            results = pool.imap_unordered(_worker_entry, task_stream, chunksize=8)

            for local_game_id, resume_key, payload in wrap_iter(
                results,
                desc="[GamesBuilder] Analisi partite (multi-sorgente)",
                unit="game",
                total=estimate,
            ):
                processed_games += 1

                self._mark_done(resume_key, local_game_id)
                if processed_games % cfg.resume_checkpoint_every == 0:
                    if cfg.save_debug_jsonl and self._debug_records_raw_path:
                        self._persist_pending_debug_records()
                    if cfg.auto_resume:
                        self._persist_resume_state()

                if cfg.flush_every_seconds and (time.monotonic() - last_flush_time) >= cfg.flush_every_seconds:
                    self._registry.flush()
                    last_flush_time = time.monotonic()

                try:
                    records: List[Dict[str, Any]] = decode_from_ipc(payload)
                except Exception:
                    skipped_games_on_parent_error += 1
                    logger.error(
                        "[GamesBuilder] decode_from_ipc fallito per resume_key=%s "
                        "(game processato #%d): payload scartato, run() continua.",
                        resume_key, processed_games,
                        exc_info=True,
                    )
                    continue

                if not records:
                    continue

                try:
                    accepted_games += 1

                    for rec in records:
                        data = rec["data"]
                        debug_entry = rec["debug"]
                        group_key = debug_entry["mate_n_window"]

                        self._registry.enqueue(
                            source_tag=debug_entry["source"],
                            data=data,
                            group_key=group_key,
                        )
                        enqueued_positions += 1

                        source_counts[debug_entry["source"]] += 1
                        clock_source_counts[debug_entry["clock_source"]] += 1
                        mate_n_counts[debug_entry["mate_n"]] += 1

                        if cfg.save_debug_jsonl:
                            self._debug_records.append(debug_entry)
                except Exception:
                    skipped_games_on_parent_error += 1
                    logger.error(
                        "[GamesBuilder] Errore nell'enqueue dei record per resume_key=%s "
                        "(game processato #%d): questo game viene scartato, run() continua.",
                        resume_key, processed_games,
                        exc_info=True,
                    )
                    continue

        except KeyboardInterrupt:
            print("\n[WARNING] Interruzione richiesta: arresto forzato dei worker in corso...")
            _shutdown_pool(graceful_first=False)
            raise
        except Exception:
            logger.exception(
                "[GamesBuilder] Errore FATALE e non recuperabile durante l'analisi: "
                "arresto forzato dei worker in corso."
            )
            print("\n[WARNING] Errore durante l'analisi: arresto forzato dei worker in corso...")
            _shutdown_pool(graceful_first=False)
            raise
        else:
            pool.close()
            _shutdown_pool(graceful_first=True)
        finally:
            self._persist_resume_state()
            signal.signal(signal.SIGINT, previous_sigint)

        self._registry.flush()

        if self.config.save_debug_jsonl and self._debug_records_raw_path:
            self._persist_pending_debug_records()

        if skipped_games_on_parent_error:
            logger.warning(
                "[GamesBuilder] %d game scartati per errori lato padre (decode/enqueue) durante questa run.",
                skipped_games_on_parent_error,
            )

        return {
            "processed_games": processed_games,
            "accepted_games": accepted_games,
            "enqueued_positions": enqueued_positions,
            "mate_n_counts": dict(mate_n_counts),
            "source_counts": dict(source_counts),
            "clock_source_counts": dict(clock_source_counts),
            "skipped_games_on_parent_error": skipped_games_on_parent_error,
        }

    def _persist_pending_debug_records(self) -> None:
        if not self._debug_records:
            return
        with open(self._debug_records_raw_path, "a", encoding="utf-8") as f:
            for rec in self._debug_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._debug_persisted_count += len(self._debug_records)
        self._debug_records.clear()

    @staticmethod
    def write_debug_jsonl_from_pending(
        pending_path: str,
        output_path: str,
        split_assignment: Dict[str, str],
    ) -> Optional[str]:
        if not os.path.exists(pending_path):
            return None

        missing_game_ids = set()
        tmp_path = output_path + ".tmp"
        wrote_any = False
        with open(pending_path, "r", encoding="utf-8") as src, \
             open(tmp_path, "w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                game_id = rec["game_id"]
                split_name = split_assignment.get(game_id)
                if split_name is None:
                    missing_game_ids.add(game_id)
                    continue
                rec["split"] = split_name
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote_any = True

        if not wrote_any:
            os.remove(tmp_path)
            return None

        os.replace(tmp_path, output_path)
        os.remove(pending_path)

        if missing_game_ids:
            logger.warning(
                "[GamesBuilder] %d game_id presenti nel debug JSONL ma assenti dallo split_assignment reale: esclusi.",
                len(missing_game_ids),
            )

        return output_path

    def write_debug_jsonl(self, split_assignment: Dict[str, str]) -> Optional[str]:
        if not self.config.save_debug_jsonl or not self._debug_jsonl_path:
            return None
        self._persist_pending_debug_records()
        return self.write_debug_jsonl_from_pending(
            self._debug_records_raw_path, self._debug_jsonl_path, split_assignment
        )