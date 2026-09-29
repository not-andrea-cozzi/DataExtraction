from __future__ import annotations

import gc
import json
import logging
import os
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from timegnn.train.early_stopping import EarlyStopping

from core.constants import NUM_EVENT_ID_CATEGORIES, NUM_NODE_FEATURES
from core.labels import mate_in_n_num_classes


logger = logging.getLogger("timegnn_chess.train")

CONFIG_ENV = "TRAIN_CONFIG"
DEFAULT_CONFIG = "train.yaml"
BACKBONES = {
    "gat_basic": build_basic_backbone,
    "gat_time_decay": build_time_decay_backbone,
}


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


def build_loader(dataset: ChessShardDataset, sampler: ShardAwareSampler, cfg: RunConfig, device: str) -> DataLoader:
    workers = cfg.training.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        collate_fn=chess_collate,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device.startswith("cuda"),
    )


def batch_terms(policy_logits, policy_targets: List[int], value_logits, mate_targets, device: str):
    losses = []
    preds = []
    targets = []
    for logits, target in zip(policy_logits, policy_targets):
        if target < 0:
            continue
        losses.append(-F.log_softmax(logits.float(), dim=0)[target])
        preds.append(logits.argmax())
        targets.append(target)

    if losses:
        policy_loss_sum = torch.stack(losses).sum()
        policy_correct = (torch.stack(preds) == torch.tensor(targets, device=device)).sum()
    else:
        policy_loss_sum = value_logits.new_zeros(())
        policy_correct = value_logits.new_zeros((), dtype=torch.long)

    value_loss_sum = F.cross_entropy(value_logits.float(), mate_targets, reduction="sum")
    value_correct = (value_logits.argmax(dim=1) == mate_targets).sum()
    skipped = len(policy_targets) - len(targets)
    return policy_loss_sum, policy_correct, len(targets), skipped, value_loss_sum, value_correct


def run_epoch(
    model,
    loader: DataLoader,
    sampler: ShardAwareSampler,
    epoch: int,
    cfg: RunConfig,
    device: str,
    optimizer=None,
    scaler=None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sampler.set_epoch(epoch)

    amp = cfg.training.use_amp and device.startswith("cuda")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    pw = cfg.training.policy_loss_weight
    vw = cfg.training.value_loss_weight

    totals = torch.zeros(4, device=device)
    n_policy_total = n_value_total = skipped_total = 0

    with torch.set_grad_enabled(training):
        for batch_data, node_offsets, legal_move_indices, policy_targets, mate_targets in loader:
            batch_data = batch_data.to(device, non_blocking=True)
            mate_targets = mate_targets.to(device, non_blocking=True)
            legal = [m.to(device, non_blocking=True) for m in legal_move_indices]

            with torch.autocast(device_type=device_type, enabled=amp):
                policy_logits, value_logits = model(batch_data, node_offsets, legal)

            p_loss, p_correct, n_policy, skipped, v_loss, v_correct = batch_terms(
                policy_logits, policy_targets, value_logits, mate_targets, device
            )
            n_value = int(mate_targets.numel())
            loss = pw * p_loss / max(n_policy, 1) + vw * v_loss / n_value

            if training and loss.requires_grad:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
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
    train_ds: ChessShardDataset,
    val_ds: ChessShardDataset,
    device: str,
) -> Tuple[ChessDualHeadModel, List[Dict[str, Any]], int, float]:
    t = cfg.training
    train_sampler = ShardAwareSampler(train_ds, shuffle=True, seed=t.seed)
    val_sampler = ShardAwareSampler(val_ds, shuffle=False, seed=t.seed)
    train_loader = build_loader(train_ds, train_sampler, cfg, device)
    val_loader = build_loader(val_ds, val_sampler, cfg, device)

    set_seed(t.seed)
    backbone = BACKBONES[variant](model_cfg, NUM_NODE_FEATURES)
    model = ChessDualHeadModel(backbone, model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=t.use_amp and device.startswith("cuda"))
    stopper = EarlyStopping(patience=t.patience)

    history: List[Dict[str, Any]] = []
    best_state = None
    best_epoch = 0

    for epoch in range(t.num_epochs):
        train_m = run_epoch(model, train_loader, train_sampler, epoch, cfg, device, optimizer, scaler)
        val_m = run_epoch(model, val_loader, val_sampler, epoch, cfg, device)
        val_total = t.policy_loss_weight * val_m["policy_loss"] + t.value_loss_weight * val_m["value_loss"]

        stop = stopper(val_total)
        if stopper.best_loss_updated:
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train": train_m, "val": val_m})

        logger.info(
            "[%s] epoch %d | train policy %.4f/%.3f value %.4f/%.3f | val policy %.4f/%.3f value %.4f/%.3f | skipped train=%d val=%d",
            variant,
            epoch + 1,
            train_m["policy_loss"], train_m["policy_acc"], train_m["value_loss"], train_m["value_acc"],
            val_m["policy_loss"], val_m["policy_acc"], val_m["value_loss"], val_m["value_acc"],
            train_m["policy_skipped"], val_m["policy_skipped"],
        )
        if stop:
            logger.info("[%s] early stopping a epoch %d (best epoch %d)", variant, epoch + 1, best_epoch)
            break

    del train_loader, val_loader
    if best_state is None:
        raise RuntimeError(f"[{variant}] nessuna epoca con validation loss finita")
    model.load_state_dict(best_state)
    return model, history, best_epoch, float(stopper.best_loss)


def run(cfg: RunConfig) -> Dict[str, Any]:
    device = resolve_device(cfg.training.device)
    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)

    train_ds = ChessShardDataset(cfg.data.train_dir, cfg.data.max_shards)
    val_ds = ChessShardDataset(cfg.data.val_dir, cfg.data.max_shards)
    if cfg.data.test_dir:
        eval_name, eval_ds = "test", ChessShardDataset(cfg.data.test_dir, cfg.data.max_shards)
    else:
        eval_name, eval_ds = "val", val_ds
    logger.info(
        "device=%s | train=%d val=%d %s=%d posizioni",
        device, len(train_ds), len(val_ds), eval_name, len(eval_ds),
    )

    model_cfg = build_model_config(cfg)
    histories: Dict[str, List[Dict[str, Any]]] = {}
    frames = {}
    variants_summary: Dict[str, Any] = {}

    for variant in cfg.model.variants:
        variant_dir = os.path.join(out_dir, variant)
        os.makedirs(variant_dir, exist_ok=True)

        model, history, best_epoch, best_loss = train_variant(variant, cfg, model_cfg, train_ds, val_ds, device)

        sampler = ShardAwareSampler(eval_ds, shuffle=False, seed=cfg.training.seed)
        loader = build_loader(eval_ds, sampler, cfg, device)
        frame = collect_predictions(model, loader, device, cfg.data.mate_range_min, cfg.training.use_amp)
        del loader

        histories[variant] = history
        frames[variant] = frame
        stats = summarize(frame)
        variants_summary[variant] = {"best_epoch": best_epoch, "best_val_loss": best_loss, **stats}

        with open(os.path.join(variant_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        frame.to_csv(os.path.join(variant_dir, "predictions.csv"), index=False)
        if cfg.output.save_checkpoints:
            torch.save(
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


def main(config_path: Optional[str] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    path = config_path or os.environ.get(CONFIG_ENV, DEFAULT_CONFIG)
    try:
        cfg = load_config(path)
    except ConfigError as e:
        logger.error("Errore di configurazione: %s", e)
        return 2
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())