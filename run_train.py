from __future__ import annotations

import argparse
from functools import partial
import gc
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from core.constants import NUM_EVENT_ID_CATEGORIES, NUM_NODE_FEATURES
from core.labels import mate_in_n_num_classes
from timegnn.train.early_stopping import EarlyStopping
from train import (
    ChessDualHeadModel,
    ChessGATConfig,
    ConfigError,
    RunConfig,
    build_basic_backbone,
    build_time_decay_backbone,
    chess_collate,
    collect_predictions,
    load_config,
    make_plots,
    summarize,
)
from train.dataset import _load_shard, _shard_sizes
from train.heads import build_timed_backbone
from utils.compression import decompress_position_data
from utils.ipc import harden_process_for_ipc

logger = logging.getLogger("timegnn_chess.train")

CONFIG_ENV = "TRAIN_CONFIG"
DEFAULT_CONFIG = "train.yaml"
BACKBONES = {
    "gat_basic": build_basic_backbone,
    "gat_time_decay": partial(build_timed_backbone, use_time=True, zero_clock=False),
    "gat_no_time": partial(build_timed_backbone, use_time=False, zero_clock=True),
}


# --------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model_config(cfg: RunConfig) -> ChessGATConfig:
    m = cfg.model
    return ChessGATConfig(
        num_embedding_features=NUM_EVENT_ID_CATEGORIES,
        embedding_dims=m.embedding_dims,
        gat_hidden_dim_event=m.gat_hidden_dim_event,
        gat_hidden_dim_embed=m.gat_hidden_dim_embed,
        gat_hidden_dim_concat=m.gat_hidden_dim_concat,
        num_heads=m.num_heads,
        num_layers=m.num_layers,
        dropout=m.dropout,
        use_batch_norm=m.use_batch_norm,
        activation=m.activation,
        lambda_decay=m.lambda_decay,
        policy_hidden_dim=m.policy_hidden_dim,
        num_mate_classes=mate_in_n_num_classes(cfg.mate_range),
    )


def _atomic_save(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------ shard streaming
def _identity(x):
    return x


def _prepare(d, path: str):
    if "x_binary" in d:
        d = decompress_position_data(d)
    if not hasattr(d, "outcome"):
        raise ValueError(f"Data senza 'outcome' in {path}.")
    if d.legal_move_indices.dtype != torch.long:
        d.legal_move_indices = d.legal_move_indices.to(torch.long)
    return d


class StreamingShardDataset(IterableDataset):
    """Legge il dataset shard per shard e restituisce batch gia' collati.

    RAM: al massimo UNO shard per worker. Gli shard sono spartiti tra i worker
    (shard i -> worker i % num_workers), quindi ogni shard viene letto una sola volta
    per epoca. Lo shuffle e' shard-order + intra-shard: e' un shuffle globale solo se
    il dataset e' gia' stato mescolato in fase di finalize.
    """

    def __init__(self, split_dir: str, batch_size: int, shuffle: bool, seed: int,
                 max_shards: Optional[int] = None) -> None:
        paths = sorted(glob(os.path.join(split_dir, "shard_*.pt")))
        if not paths:
            raise FileNotFoundError(f"Nessuno shard_*.pt in {split_dir}")
        sizes = _shard_sizes(split_dir, paths)
        if max_shards is not None:
            paths, sizes = paths[:max_shards], sizes[:max_shards]
        self.paths = paths
        self.sizes = sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    @property
    def num_positions(self) -> int:
        return sum(self.sizes)

    @property
    def num_shards(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)

        order = list(range(len(self.paths)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)

        buf: List[Any] = []
        for si in order[wid::nw]:
            recs = _load_shard(self.paths[si])
            if self.shuffle:
                random.Random(self.seed * 7919 + self.epoch * 100003 + si).shuffle(recs)
            for rec in recs:
                buf.append(_prepare(rec, self.paths[si]))
                if len(buf) == self.batch_size:
                    yield chess_collate(buf)
                    buf = []
            del recs
        if buf:
            yield chess_collate(buf)


def make_loader(ds: StreamingShardDataset, cfg: RunConfig, device: str) -> DataLoader:
    workers = min(cfg.training.num_workers, ds.num_shards)
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=workers,
        collate_fn=_identity,
        pin_memory=device.startswith("cuda"),
        prefetch_factor=4 if workers > 0 else None,
        persistent_workers=False,
    )


def check_labels(ds: StreamingShardDataset, num_classes: int, name: str) -> None:
    outcomes = torch.tensor([int(d.outcome) for d in _load_shard(ds.paths[0])])
    lo, hi = int(outcomes.min()), int(outcomes.max())
    if lo < 0 or hi >= num_classes:
        raise RuntimeError(
            f"[{name}] outcome in [{lo},{hi}] fuori da [0,{num_classes - 1}]: "
            f"data.mate_range_min/max non coerente con il dataset."
        )
    logger.info("[%s] outcome osservati nel primo shard: [%d,%d], classi attese: %d", name, lo, hi, num_classes)


def estimate_value_weights(ds: StreamingShardDataset, num_classes: int, device: str, n_shards: int):
    counts = torch.zeros(num_classes)
    for p in ds.paths[:n_shards]:
        for d in _load_shard(p):
            counts[int(d.outcome)] += 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    logger.info("class counts (primi %d shard): %s -> pesi %s",
                min(n_shards, ds.num_shards), counts.long().tolist(), [round(w, 3) for w in weights.tolist()])
    return weights.to(device)


# ---------------------------------------------------------------------- loop
def batch_terms(policy_logits, policy_targets: List[torch.Tensor], value_logits, mate_targets, device: str,
                value_weight: Optional[torch.Tensor] = None):
    valid = [i for i, t in enumerate(policy_targets) if t.numel() > 0]
    if valid:
        padded = pad_sequence(
            [policy_logits[i].float() for i in valid], batch_first=True, padding_value=float("-inf")
        )
        mask = torch.zeros_like(padded, dtype=torch.bool)
        for r, i in enumerate(valid):
            mask[r, policy_targets[i].to(device)] = True
        lse_all = torch.logsumexp(padded, dim=1)
        lse_pos = torch.logsumexp(padded.masked_fill(~mask, float("-inf")), dim=1)
        policy_loss_sum = (lse_all - lse_pos).sum()
        rows = torch.arange(len(valid), device=device)
        policy_correct = mask[rows, padded.argmax(dim=1)].sum()
    else:
        policy_loss_sum = value_logits.new_zeros(())
        policy_correct = value_logits.new_zeros((), dtype=torch.long)

    per_sample = F.cross_entropy(value_logits.float(), mate_targets, reduction="none")
    value_loss_sum = per_sample.sum()
    if value_weight is not None:
        w = value_weight[mate_targets]
        value_loss_opt = (per_sample * w).sum() / w.sum()
    else:
        value_loss_opt = value_loss_sum / mate_targets.numel()
    value_correct = (value_logits.argmax(dim=1) == mate_targets).sum()
    skipped = len(policy_targets) - len(valid)
    return policy_loss_sum, policy_correct, len(valid), skipped, value_loss_sum, value_loss_opt, value_correct


def run_epoch(
    model,
    loader: DataLoader,
    cfg: RunConfig,
    device: str,
    optimizer=None,
    scaler=None,
    value_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    amp = cfg.training.use_amp and device.startswith("cuda")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    pw = cfg.training.policy_loss_weight
    vw = cfg.training.value_loss_weight
    clip = cfg.training.grad_clip

    totals = torch.zeros(4, device=device)
    n_policy_total = n_value_total = skipped_total = 0

    with torch.set_grad_enabled(training):
        for batch_data, node_offsets, legal_move_indices, policy_targets, mate_targets in loader:
            batch_data = batch_data.to(device, non_blocking=True)
            mate_targets = mate_targets.to(device, non_blocking=True)
            legal = [m.to(device, non_blocking=True) for m in legal_move_indices]

            with torch.autocast(device_type=device_type, enabled=amp):
                policy_logits, value_logits = model(batch_data, node_offsets, legal)

            p_loss, p_correct, n_policy, skipped, v_loss, v_loss_opt, v_correct = batch_terms(
                policy_logits, policy_targets, value_logits, mate_targets, device,
                value_weight if training else None,
            )
            n_value = int(mate_targets.numel())
            loss = pw * p_loss / max(n_policy, 1) + vw * v_loss_opt

            if training and loss.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()

            totals += torch.stack(
                [p_loss.detach().float(), p_correct.float(), v_loss.detach().float(), v_correct.float()]
            )
            n_policy_total += n_policy
            n_value_total += n_value
            skipped_total += skipped

    p_loss_sum, p_correct_sum, v_loss_sum, v_correct_sum = totals.tolist()
    return {
        "policy_loss": p_loss_sum / max(n_policy_total, 1),
        "value_loss": v_loss_sum / max(n_value_total, 1),
        "policy_acc": p_correct_sum / max(n_policy_total, 1),
        "value_acc": v_correct_sum / max(n_value_total, 1),
        "policy_skipped": skipped_total,
    }


def train_variant(
    variant: str,
    cfg: RunConfig,
    model_cfg: ChessGATConfig,
    train_ds: StreamingShardDataset,
    val_ds: StreamingShardDataset,
    device: str,
    variant_dir: str,
) -> Tuple[ChessDualHeadModel, List[Dict[str, Any]], int, float]:
    t = cfg.training
    last_path = os.path.join(variant_dir, "last.pt")
    best_path = os.path.join(variant_dir, "best.pt")

    set_seed(t.seed)
    backbone = BACKBONES[variant](model_cfg, NUM_NODE_FEATURES)
    model = ChessDualHeadModel(backbone, model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=t.use_amp and device.startswith("cuda"))
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(1, t.patience // 2))
        if t.use_scheduler else None
    )
    stopper = EarlyStopping(patience=t.patience)

    value_weight = None
    if t.class_weighted_value:
        value_weight = estimate_value_weights(train_ds, model_cfg.num_mate_classes, device, t.class_weight_shards)

    history: List[Dict[str, Any]] = []
    best_epoch = 0
    start_epoch = 0
    finished = False

    if t.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        if scheduler is not None and ck.get("scheduler"):
            scheduler.load_state_dict(ck["scheduler"])
        stopper.__dict__.update(ck["stopper"])
        history, best_epoch = ck["history"], ck["best_epoch"]
        start_epoch, finished = ck["epoch"], ck["finished"]
        logger.info("[%s] resume da epoch %d (finished=%s)", variant, start_epoch, finished)

    def save_last(epoch: int, done: bool) -> None:
        _atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "stopper": dict(stopper.__dict__),
                "history": history,
                "best_epoch": best_epoch,
                "epoch": epoch,
                "finished": done,
            },
            last_path,
        )

    if not finished:
        for epoch in range(start_epoch, t.num_epochs):
            t0 = time.time()

            train_ds.set_epoch(epoch)
            train_loader = make_loader(train_ds, cfg, device)
            train_m = run_epoch(model, train_loader, cfg, device, optimizer, scaler, value_weight)
            del train_loader

            val_loader = make_loader(val_ds, cfg, device)
            val_m = run_epoch(model, val_loader, cfg, device)
            del val_loader
            gc.collect()

            val_total = t.policy_loss_weight * val_m["policy_loss"] + t.value_loss_weight * val_m["value_loss"]
            if not math.isfinite(train_m["policy_loss"] + train_m["value_loss"]) or not math.isfinite(val_total):
                raise RuntimeError(
                    f"[{variant}] loss non finita a epoch {epoch + 1}: abbassa lr o controlla i dati."
                )

            stop = stopper(val_total)
            if stopper.best_loss_updated:
                best_epoch = epoch + 1
                _atomic_save(model.state_dict(), best_path)
            if scheduler is not None:
                scheduler.step(val_total)

            history.append({
                "epoch": epoch + 1,
                "train": train_m,
                "val": val_m,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - t0,
            })
            save_last(epoch + 1, done=stop or epoch + 1 >= t.num_epochs)

            logger.info(
                "[%s] epoch %d (%.0fs, lr %.2e) | train policy %.4f/%.3f value %.4f/%.3f | "
                "val policy %.4f/%.3f value %.4f/%.3f | skipped train=%d val=%d",
                variant, epoch + 1, time.time() - t0, optimizer.param_groups[0]["lr"],
                train_m["policy_loss"], train_m["policy_acc"], train_m["value_loss"], train_m["value_acc"],
                val_m["policy_loss"], val_m["policy_acc"], val_m["value_loss"], val_m["value_acc"],
                train_m["policy_skipped"], val_m["policy_skipped"],
            )
            if stop:
                logger.info("[%s] early stopping a epoch %d (best epoch %d)", variant, epoch + 1, best_epoch)
                break

    if not os.path.exists(best_path):
        raise RuntimeError(f"[{variant}] nessuna epoca con validation loss finita")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, history, best_epoch, float(stopper.best_loss)


def run(cfg: RunConfig) -> Dict[str, Any]:
    harden_process_for_ipc()
    device = resolve_device(cfg.training.device)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)

    t = cfg.training
    train_ds = StreamingShardDataset(cfg.data.train_dir, t.batch_size, True, t.seed, cfg.data.max_shards)
    val_ds = StreamingShardDataset(cfg.data.val_dir, t.batch_size, False, t.seed, cfg.data.max_shards)
    if cfg.data.test_dir:
        eval_name = "test"
        eval_ds = StreamingShardDataset(cfg.data.test_dir, t.batch_size, False, t.seed, cfg.data.max_shards)
    else:
        eval_name, eval_ds = "val", val_ds
    logger.info(
        "device=%s | train=%d (%d shard) val=%d %s=%d posizioni | workers=%d",
        device, train_ds.num_positions, train_ds.num_shards, val_ds.num_positions,
        eval_name, eval_ds.num_positions, t.num_workers,
    )

    model_cfg = build_model_config(cfg)
    check_labels(train_ds, model_cfg.num_mate_classes, "train")
    check_labels(val_ds, model_cfg.num_mate_classes, "val")

    histories: Dict[str, List[Dict[str, Any]]] = {}
    frames = {}
    variants_summary: Dict[str, Any] = {}

    for variant in cfg.model.variants:
        variant_dir = os.path.join(out_dir, variant)
        os.makedirs(variant_dir, exist_ok=True)

        model, history, best_epoch, best_loss = train_variant(
            variant, cfg, model_cfg, train_ds, val_ds, device, variant_dir
        )

        eval_ds.set_epoch(0)
        loader = make_loader(eval_ds, cfg, device)
        frame = collect_predictions(model, loader, device, cfg.data.mate_range_min, t.use_amp)
        del loader

        histories[variant] = history
        frames[variant] = frame
        stats = summarize(frame)
        variants_summary[variant] = {"best_epoch": best_epoch, "best_val_loss": best_loss, **stats}

        with open(os.path.join(variant_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        frame.to_csv(os.path.join(variant_dir, "predictions.csv"), index=False)
        if cfg.output.save_checkpoints:
            _atomic_save(
                {
                    "variant": variant,
                    "state_dict": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "mate_range": cfg.mate_range,
                    "best_epoch": best_epoch,
                },
                os.path.join(variant_dir, "model.pt"),
            )
        logger.info(
            "[%s] %s: policy_acc=%.4f value_acc=%.4f policy_skipped=%d/%d",
            variant, eval_name, stats["policy_acc"], stats["value_acc"], stats["policy_skipped"], stats["n"],
        )

        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary = {"eval_split": eval_name, "variants": variants_summary}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if cfg.output.plots:
        try:
            make_plots(histories, frames, os.path.join(out_dir, "plots"), cfg.mate_range, eval_name)
        except Exception:
            logger.exception("generazione grafici fallita: training e predizioni sono comunque salvati")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Training GAT chess (gat_basic / gat_time_decay)")
    ap.add_argument("--config", default="train.yaml")
    args = ap.parse_args(argv)

    path = args.config or os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)
    try:
        cfg = load_config(path)
    except ConfigError as e:
        logger.error("Errore di configurazione: %s", e)
        return 2
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())