"""Smoke test: 3 epoche + valutazione di gat_time_decay e gat_no_time su sottoinsiemi ridotti.

Limiti (in POSIZIONI, non shard):
    train = 10000, val = 100, test = 1000

Uso:
    python train_test.py --config train.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
from glob import glob
from typing import List, Optional

import random

import run_train
from run_train import StreamingShardDataset, _prepare
from train import ConfigError, chess_collate, load_config
from train.dataset import _load_shard, _shard_sizes

logger = logging.getLogger("timegnn_chess.train_test")

MAX_TRAIN = 10_000
MAX_VAL = 100
MAX_TEST = 1_000
NUM_EPOCHS = 3
VARIANTS = ["gat_time_decay", "gat_no_time"]
OUT_DIR = "../Dataset/runs/train_test"


class LimitedStreamingDataset(StreamingShardDataset):
    """StreamingShardDataset che usa solo le prime `max_positions` posizioni.

    Gli shard sono gia' mescolati in fase di finalize, quindi le prime N posizioni
    sono un campione casuale; il sottoinsieme e' fisso tra le epoche.
    """

    def __init__(self, split_dir: str, batch_size: int, shuffle: bool, seed: int,
                 max_shards: Optional[int] = None, max_positions: Optional[int] = None) -> None:
        super().__init__(split_dir, batch_size, shuffle, seed, max_shards)
        if max_positions is None:
            self.takes = list(self.sizes)
            return
        paths, takes, remaining = [], [], max_positions
        for p, size in zip(self.paths, self.sizes):
            if remaining <= 0:
                break
            take = min(size, remaining)
            paths.append(p)
            takes.append(take)
            remaining -= take
        self.paths, self.sizes, self.takes = paths, takes, takes

    def __iter__(self):
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)

        order = list(range(len(self.paths)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)

        buf: List = []
        for si in order[wid::nw]:
            recs = _load_shard(self.paths[si])[: self.takes[si]]
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


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Smoke test training (3 epoche, gat_time_decay vs gat_no_time)")
    ap.add_argument("--config", default="train.yaml")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        logger.error("Errore di configurazione: %s", e)
        return 2

    # override per il test rapido
    cfg.model.variants = list(VARIANTS)
    cfg.training.num_epochs = NUM_EPOCHS
    cfg.training.resume = False
    cfg.data.max_shards = None
    cfg.output.dir = args.out_dir
    if not cfg.data.test_dir:
        logger.warning("test_dir non impostato: la valutazione usera' il val set (max %d).", MAX_VAL)

    limits = {
        os.path.normpath(cfg.data.train_dir): MAX_TRAIN,
        os.path.normpath(cfg.data.val_dir): MAX_VAL,
    }
    if cfg.data.test_dir:
        limits[os.path.normpath(cfg.data.test_dir)] = MAX_TEST

    def limited_factory(split_dir, batch_size, shuffle, seed, max_shards=None):
        return LimitedStreamingDataset(
            split_dir, batch_size, shuffle, seed, max_shards,
            max_positions=limits.get(os.path.normpath(split_dir)),
        )

    # run() istanzia StreamingShardDataset dal proprio namespace: lo sostituiamo
    run_train.StreamingShardDataset = limited_factory

    # pulizia vecchi checkpoint del test (niente resume)
    for v in VARIANTS:
        for name in glob(os.path.join(cfg.output.dir, v, "last.pt")):
            os.remove(name)

    summary = run_train.run(cfg)
    for variant, s in summary["variants"].items():
        logger.info("[%s] %s: policy_acc=%.4f value_acc=%.4f (n=%d)",
                    variant, summary["eval_split"], s["policy_acc"], s["value_acc"], s["n"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())