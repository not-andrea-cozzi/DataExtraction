from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from train.config import VARIANTS, ConfigError, _build

LLM_REQUIRED = ("llm_base_url", "llm_model", "llm_api_key_env")


@dataclass
class PlotsSection:
    mate_min: int = 1
    mate_max: int = 10


@dataclass
class PathsSection:
    holdout_dir: str = ""          # .../HeldOut/heldout_clean (output di heldout.py)
    runs_dir: str = ""             # output.dir di run_train: <runs_dir>/<variant>/model.pt
    out_dir: str = "../Dataset/runs/evaluate"


@dataclass
class GnnSection:
    variants: List[str] = field(default_factory=lambda: ["gat_time_decay", "gat_no_time"])
    batch_size: int = 32
    device: Optional[str] = None
    use_amp: bool = False


@dataclass
class StatsSection:
    alpha: float = 0.05
    correction: Optional[str] = "holm"


@dataclass
class EvalConfig:
    paths: PathsSection = field(default_factory=PathsSection)
    gnn: GnnSection = field(default_factory=GnnSection)
    stats: StatsSection = field(default_factory=StatsSection)
    llm: Dict[str, Any] = field(default_factory=dict)
    
    plots: PlotsSection = field(default_factory=PlotsSection)

    @property
    def mate_order(self) -> List[int]:
        return list(range(self.plots.mate_min, self.plots.mate_max + 1))

    def gnn_csv(self, variant: str) -> str:
        return os.path.join(self.paths.out_dir, "gnn", f"{variant}_predictions.csv")

    @property
    def llm_csv(self) -> str:
        return os.path.join(self.paths.out_dir, "llm", "llm_predictions.csv")

    def report_dir(self, name: str) -> str:
        return os.path.join(self.paths.out_dir, name)

    def validate(self) -> None:
        p, g, s = self.paths, self.gnn, self.stats
        if not os.path.exists(os.path.join(p.holdout_dir, "manifest.json")):
            raise ConfigError(f"[paths] manifest.json non trovato in holdout_dir: {p.holdout_dir!r}")
        if not os.path.isdir(p.runs_dir):
            raise ConfigError(f"[paths] runs_dir non e' una directory: {p.runs_dir!r}")
        if not g.variants or len(set(g.variants)) != len(g.variants):
            raise ConfigError("[gnn] variants vuoto o con duplicati.")
        unknown = [v for v in g.variants if v not in VARIANTS]
        if unknown:
            raise ConfigError(f"[gnn] varianti sconosciute: {unknown}. Valide: {list(VARIANTS)}")
        if g.batch_size < 1:
            raise ConfigError("[gnn] batch_size deve essere >= 1.")
        if not 0.0 < s.alpha < 1.0:
            raise ConfigError("[stats] alpha deve essere in (0, 1).")
        if s.correction not in (None, "holm", "bonferroni"):
            raise ConfigError("[stats] correction deve essere null, 'holm' o 'bonferroni'.")
        if self.llm.get("enabled", True):
            missing = [k for k in LLM_REQUIRED if not self.llm.get(k)]
            if missing:
                raise ConfigError(f"[llm] chiavi mancanti: {missing} (oppure enabled: false).")


_SECTIONS = {"paths": PathsSection, "gnn": GnnSection, "stats": StatsSection, "plots": PlotsSection}


def load_config(path: str) -> EvalConfig:
    if not os.path.exists(path):
        raise ConfigError(f"Config non trovata: {path}")
    if yaml is None:
        raise ConfigError("pyyaml non installato.")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError("Il YAML deve essere un dizionario.")
    unknown = set(raw) - set(_SECTIONS) - {"llm"}
    if unknown:
        raise ConfigError(f"Sezioni sconosciute: {sorted(unknown)}. Valide: {sorted(_SECTIONS) + ['llm']}")
    cfg = EvalConfig(
        **{name: _build(cls, raw.get(name), name) for name, cls in _SECTIONS.items()},
        llm=dict(raw.get("llm") or {}),
    )
    cfg.validate()
    return cfg