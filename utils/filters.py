from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess

from core.constants import PIECE_VALUES
from utils.pgn_time import parse_rating


# ---------------------------------------------------------------------------
# Diagnostica globale (opzionale): conta i motivi di scarto.
# ---------------------------------------------------------------------------
REJECTS: Counter = Counter()


def report_rejects(logger=None) -> None:
    """Stampa un riepilogo dei motivi di scarto accumulati e azzera il contatore."""
    total = sum(REJECTS.values())
    if total == 0:
        msg = "Reject reasons: nessuno scarto registrato."
    else:
        parts = [f"{k}={v} ({v / total:.1%})" for k, v in REJECTS.most_common()]
        msg = f"Reject reasons (tot={total}): " + " | ".join(parts)
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    REJECTS.clear()


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityConfig:
    """
    Filtri di qualita' condivisi da Games e Puzzle.

    I default sono pensati per i puzzle di matto (mateIn1..mateIn5):
    - min_material_for_mate_attempt = 5   -> pavimento assoluto
    - min_material_diff_for_mate_attempt = -3 -> tollera il solver sotto di 3 punti
      se ha ancora pezzi pesanti (copre i sacrifici tipici dei puzzle).
    """
    min_material_for_mate_attempt: int = 5
    min_material_diff_for_mate_attempt: int = -3
    require_heavy_piece: bool = False
    skip_trivial_endgame: bool = False
    max_piece_count: Optional[int] = None
    candidate_min_legal_moves: int = 1
    candidate_max_legal_moves: Optional[int] = None
    skip_if_in_check: bool = False
    skip_forced_moves: bool = False
    require_mate_potential: bool = False
    mate_potential_min_attackers: int = 1
    mate_potential_max_escapes: int = 3


@dataclass(frozen=True)
class HeaderFilterConfig:
    """Filtri a livello di header PGN (solo Games)."""
    only_decisive_games: bool = True
    skip_time_forfeit: bool = True
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    require_both_ratings: bool = True


# ---------------------------------------------------------------------------
# Materiale
# ---------------------------------------------------------------------------
def material_by_color(board: chess.Board) -> Tuple[int, int]:
    """Ritorna (materiale_bianco, materiale_nero) usando PIECE_VALUES."""
    white = black = 0
    for piece in board.piece_map().values():
        v = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white += v
        else:
            black += v
    return white, black


def mover_has_heavy_piece(board: chess.Board) -> bool:
    """True se il lato al tratto ha almeno una Donna o una Torre."""
    return any(board.pieces(pt, board.turn) for pt in (chess.QUEEN, chess.ROOK))


def has_mating_material(board: chess.Board, q: QualityConfig) -> bool:
    """
    Decide se il lato al tratto ha materiale credibile per forzare matto.

    Regola:
      1. pavimento assoluto: mover >= min_material_for_mate_attempt
      2. se diff >= min_material_diff_for_mate_attempt -> ok
      3. altrimenti (solver sotto materiale): accetta solo se ha un pezzo pesante
    """
    white, black = material_by_color(board)
    mover, opp = (white, black) if board.turn == chess.WHITE else (black, white)
    diff = mover - opp

    if mover < q.min_material_for_mate_attempt:
        return False

    if diff >= q.min_material_diff_for_mate_attempt:
        return True

    # solver in svantaggio: teniamolo solo se ha un pezzo pesante
    return mover_has_heavy_piece(board)


def is_trivially_drawn_endgame(board: chess.Board) -> bool:
    """Re+Re, Re+minore vs Re, Re vs Re+minore: posizioni morte."""
    pm = board.piece_map()
    if any(p.piece_type in (chess.QUEEN, chess.ROOK, chess.PAWN) for p in pm.values()):
        return False
    minors = {chess.WHITE: 0, chess.BLACK: 0}
    for p in pm.values():
        if p.piece_type in (chess.BISHOP, chess.KNIGHT):
            minors[p.color] += 1
    return minors[chess.WHITE] <= 1 and minors[chess.BLACK] <= 1


# ---------------------------------------------------------------------------
# Mosse legali analizzabili
# ---------------------------------------------------------------------------
def candidate_legal_moves(board: chess.Board, q: QualityConfig) -> Optional[List[chess.Move]]:
    """Ritorna le mosse legali se la posizione e' analizzabile, altrimenti None."""
    if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
        return None
    if q.max_piece_count is not None and len(board.piece_map()) > q.max_piece_count:
        return None
    if q.skip_if_in_check and board.is_check():
        return None

    moves = list(board.legal_moves)
    n = len(moves)
    if n < q.candidate_min_legal_moves or n > 255:
        return None
    if q.candidate_max_legal_moves is not None and n > q.candidate_max_legal_moves:
        return None
    if q.skip_forced_moves and n == 1:
        return None
    return moves


# ---------------------------------------------------------------------------
# Potenziale di matto (king hunt)
# ---------------------------------------------------------------------------
def king_hunt_score(board: chess.Board) -> Tuple[int, int]:
    """
    Ritorna (numero_attaccanti_sul_re, numero_case_di_fuga_libere).

    - attaccanti: pezzi del lato al tratto che attaccano la casa del re avversario
    - fuga libera: casa adiacente al re avversario, non occupata da un pezzo
      dello stesso colore e non attaccata dal lato al tratto
    """
    opp = not board.turn
    king_sq = board.king(opp)
    if king_sq is None:
        return 0, 8

    attackers = len(board.attackers(board.turn, king_sq))

    escapes = 0
    for sq in chess.SQUARES:
        if chess.square_distance(sq, king_sq) != 1:
            continue
        occupant = board.piece_at(sq)
        if occupant is not None and occupant.color == opp:
            continue
        if board.is_attacked_by(board.turn, sq):
            continue
        escapes += 1

    return attackers, escapes


def has_mate_potential(
    board: chess.Board,
    min_attackers: int = 1,
    max_escapes: int = 3,
) -> bool:
    """Euristica: il re avversario e' sotto pressione sufficiente per un matto."""
    if board.is_check():
        return True
    attackers, escapes = king_hunt_score(board)
    return attackers >= min_attackers and escapes <= max_escapes


# ---------------------------------------------------------------------------
# Gate principale
# ---------------------------------------------------------------------------
def position_passes_quality(board: chess.Board, q: QualityConfig) -> bool:
    """
    Applica tutti i filtri di qualita' opzionali. Ritorna True se la posizione
    e' utilizzabile per il training.
    """
    # 1) tetto al numero di pezzi sulla scacchiera
    if q.max_piece_count is not None and len(board.piece_map()) > q.max_piece_count:
        REJECTS["max_piece_count"] += 1
        return False

    # 2) materiale minimo per tentare il matto
    if not has_mating_material(board, q):
        REJECTS["material"] += 1
        return False

    # 3) pezzo pesante obbligatorio
    if q.require_heavy_piece and not mover_has_heavy_piece(board):
        REJECTS["heavy_piece"] += 1
        return False

    # 4) finali banali
    if q.skip_trivial_endgame and is_trivially_drawn_endgame(board):
        REJECTS["trivial_endgame"] += 1
        return False

    # 5) potenziale di matto
    if q.require_mate_potential and not has_mate_potential(
        board,
        q.mate_potential_min_attackers,
        q.mate_potential_max_escapes,
    ):
        REJECTS["mate_potential"] += 1
        return False

    return True


# ---------------------------------------------------------------------------
# Header-level (solo Games)
# ---------------------------------------------------------------------------
def headers_are_eligible(headers: dict, h: HeaderFilterConfig) -> bool:
    """Filtra le partite per esito, terminazione e rating."""
    if h.only_decisive_games and headers.get("Result", "") not in ("1-0", "0-1"):
        return False

    termination = headers.get("Termination", "") or ""
    if h.skip_time_forfeit and "Time forfeit" in termination:
        return False

    white = parse_rating(headers.get("WhiteElo"))
    black = parse_rating(headers.get("BlackElo"))
    if h.require_both_ratings and (white is None or black is None):
        return False

    ratings = [r for r in (white, black) if r is not None]
    if ratings:
        if h.min_rating is not None and max(ratings) < h.min_rating:
            return False
        if h.max_rating is not None and min(ratings) > h.max_rating:
            return False

    return True