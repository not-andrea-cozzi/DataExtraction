from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints

try:
    import yaml
except ImportError:
    yaml = None

T = TypeVar("T")

VARIANTS = ("gat_basic", "gat_time_decay", "gat_no_time")

class ConfigError(Exception):
    pass


def _matches(value: Any, tp: Any) -> bool:
    if tp is Any:
        return True
    origin = get_origin(tp)
    if origin is Union:
        return any(_matches(value, arg) for arg in get_args(tp))
    if origin is list:
        (inner,) = get_args(tp)
        return isinstance(value, list) and all(_matches(v, inner) for v in value)
    if tp is type(None):
        return value is None
    if tp is bool:
        return isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, tp)


def _check_types(obj: Any, section: str) -> None:
    hints = get_type_hints(type(obj))
    for f in fields(obj):
        value = getattr(obj, f.name)
        if not _matches(value, hints[f.name]):
            raise ConfigError(
                f"[{section}] {f.name}: valore {value!r} di tipo {type(value).__name__}, atteso {hints[f.name]}"
            )
        if hints[f.name] is float and isinstance(value, int):
            setattr(obj, f.name, float(value))


def _build(cls: Type[T], raw: Optional[Dict[str, Any]], section: str) -> T:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"[{section}] deve essere una mappa chiave: valore.")
    valid = {f.name for f in fields(cls)}
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(f"[{section}] chiavi sconosciute: {sorted(unknown)}. Valide: {sorted(valid)}")
    obj = cls(**raw)
    _check_types(obj, section)
    return obj


@dataclass
class DataSection:
    train_dir: str = ""
    val_dir: str = ""
    test_dir: Optional[str] = None
    max_shards: Optional[int] = None
    mate_range_min: int = 1
    mate_range_max: int = 5


@dataclass
class ModelSection:
    variants: List[str] = field(default_factory=lambda: list(VARIANTS))
    embedding_dims: int = 32
    gat_hidden_dim_event: int = 32
    gat_hidden_dim_embed: int = 32
    gat_hidden_dim_concat: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    use_batch_norm: bool = True
    activation: str = "elu"
    lambda_decay: float = 0.05
    policy_hidden_dim: int = 128


@dataclass
class TrainingSection:
    batch_size: int = 32
    lr: float = 1e-3
    num_epochs: int = 30
    patience: int = 5
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    num_workers: int = 2
    use_amp: bool = True
    seed: int = 42
    device: Optional[str] = None
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    class_weighted_value: bool = False
    class_weight_shards: int = 4
    resume: bool = True
    init_from: Optional[str] = None
    # ------------------------------------------------------------------
    # NUOVO: opzioni di ottimizzazione usate da run_train.py
    # (lette con getattr(..., default), quindi i default qui sotto sono
    #  gli stessi usati dallo script se il campo non fosse presente)
    # ------------------------------------------------------------------
    lr_schedule: str = "plateau"          # "plateau" (comportamento storico) | "cosine"
    warmup_epochs: int = 0                # warmup lineare iniziale (epoche), consigliato 1-3 con cosine
    min_lr_frac: float = 0.01             # con cosine: lr finale come frazione di quello iniziale
    use_ema: bool = True                  # EMA dei pesi: validazione e best.pt usano i pesi EMA
    ema_decay: float = 0.999
    value_label_smoothing: float = 0.05   # label smoothing sulla value head (mate-in-n)
    value_soft_label_sigma: float = 0.0   # >0: target soft ordinali (sostituisce lo smoothing)
    amp_dtype: str = "float16"            # "float16" | "bfloat16" (fallback automatico se non supportato)
    grad_accum_steps: int = 1             # micro-batch accumulati per step efficace piu' grande
    compile_model: bool = False           # torch.compile (provare solo a training stabile)
    adam_beta2: float = 0.999             # 0.95-0.98 se la loss e' rumorosa o instabile


@dataclass
class OutputSection:
    dir: str = "runs/default"
    save_checkpoints: bool = True
    plots: bool = True


@dataclass
class RunConfig:
    data: DataSection = field(default_factory=DataSection)
    model: ModelSection = field(default_factory=ModelSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    output: OutputSection = field(default_factory=OutputSection)

    @property
    def mate_range(self) -> Tuple[int, int]:
        return (self.data.mate_range_min, self.data.mate_range_max)

    def validate(self) -> None:
        d, m, t = self.data, self.model, self.training
        for name in ("train_dir", "val_dir"):
            path = getattr(d, name)
            if not path or not os.path.isdir(path):
                raise ConfigError(f"[data] {name} non e' una directory esistente: {path!r}")
        if d.test_dir is not None and not os.path.isdir(d.test_dir):
            raise ConfigError(f"[data] test_dir non e' una directory esistente: {d.test_dir!r}")
        if d.max_shards is not None and d.max_shards < 1:
            raise ConfigError("[data] max_shards deve essere >= 1 oppure null.")
        if d.mate_range_min < 1 or d.mate_range_max < d.mate_range_min:
            raise ConfigError(f"[data] mate_range non valido: {self.mate_range}")
        if not m.variants:
            raise ConfigError("[model] variants vuoto.")
        unknown = [v for v in m.variants if v not in VARIANTS]
        if unknown:
            raise ConfigError(f"[model] varianti sconosciute: {unknown}. Valide: {list(VARIANTS)}")
        if len(set(m.variants)) != len(m.variants):
            raise ConfigError("[model] variants contiene duplicati.")
        if t.batch_size < 1 or t.num_epochs < 1 or t.patience < 1:
            raise ConfigError("[training] batch_size, num_epochs e patience devono essere >= 1.")
        if t.num_workers < 0:
            raise ConfigError("[training] num_workers deve essere >= 0.")
        if t.lr <= 0:
            raise ConfigError("[training] lr deve essere > 0.")
        if t.init_from is not None and not os.path.isdir(t.init_from):
            raise ConfigError(f"[training] init_from non e' una directory esistente: {t.init_from!r}")
        if t.policy_loss_weight < 0 or t.value_loss_weight < 0:
            raise ConfigError("[training] i pesi delle loss devono essere >= 0.")
        if t.policy_loss_weight == 0 and t.value_loss_weight == 0:
            raise ConfigError("[training] almeno un peso di loss deve essere > 0.")
        # ------------------------------------------------------------------
        # NUOVO: validazione delle opzioni di ottimizzazione
        # ------------------------------------------------------------------
        if t.lr_schedule not in ("plateau", "cosine"):
            raise ConfigError(f"[training] lr_schedule non valido: {t.lr_schedule!r}. Valide: ('plateau', 'cosine')")
        if t.warmup_epochs < 0:
            raise ConfigError("[training] warmup_epochs deve essere >= 0.")
        if t.lr_schedule == "cosine" and t.warmup_epochs >= t.num_epochs:
            raise ConfigError("[training] warmup_epochs deve essere < num_epochs con lr_schedule: cosine.")
        if not (0.0 < t.ema_decay < 1.0):
            raise ConfigError("[training] ema_decay deve essere in (0, 1).")
        if not (0.0 <= t.min_lr_frac < 1.0):
            raise ConfigError("[training] min_lr_frac deve essere in [0, 1).")
        if not (0.0 <= t.value_label_smoothing < 1.0):
            raise ConfigError("[training] value_label_smoothing deve essere in [0, 1).")
        if t.value_soft_label_sigma < 0:
            raise ConfigError("[training] value_soft_label_sigma deve essere >= 0.")
        if t.amp_dtype not in ("float16", "bfloat16"):
            raise ConfigError(f"[training] amp_dtype non valido: {t.amp_dtype!r}. Valide: ('float16', 'bfloat16')")
        if t.grad_accum_steps < 1:
            raise ConfigError("[training] grad_accum_steps deve essere >= 1.")
        if not (0.0 < t.adam_beta2 < 1.0):
            raise ConfigError("[training] adam_beta2 deve essere in (0, 1).")


_SECTIONS = {
    "data": DataSection,
    "model": ModelSection,
    "training": TrainingSection,
    "output": OutputSection,
}


def load_config(path: str) -> RunConfig:
    if not os.path.exists(path):
        raise ConfigError(f"Config non trovata: {path}")
    if yaml is None:
        raise ConfigError("pyyaml non installato.")
    with open(path, "r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"YAML non valido ({path}): {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError("Il YAML deve essere un dizionario.")
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"Sezioni sconosciute: {sorted(unknown)}. Valide: {sorted(_SECTIONS)}")
    try:
        cfg = RunConfig(**{name: _build(cls, raw.get(name), name) for name, cls in _SECTIONS.items()})
    except TypeError as e:
        raise ConfigError(f"Valore non valido in {path}: {e}") from e
    if cfg.data.test_dir == "":
        cfg.data.test_dir = None
    if cfg.training.init_from == "":
        cfg.training.init_from = None
    cfg.validate()
    return cfg