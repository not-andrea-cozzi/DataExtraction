from __future__ import annotations

from typing import Dict

from core.constants import EDGE_ATTACK, EDGE_LEGAL_MOVE, EDGE_PIN

# Passato a build_position_data(edge_time_factors=...): nessun clone a valle.
DEFAULT_EDGE_TIME_FACTORS: Dict[int, float] = {
    EDGE_LEGAL_MOVE: 1.0,
    EDGE_ATTACK: 0.5,
    EDGE_PIN: 0.5,
}
