from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from common import plotter

logger = logging.getLogger("timegnn_chess.reporting")

BASIC = "gat_basic"
TIME_DECAY = "gat_time_decay"


def summarize(frame: pd.DataFrame) -> Dict[str, Any]:
    valid = frame[frame["policy_valid"]]
    by_mate = valid.groupby("mate_n")["policy_correct"].mean()
    return {
        "n": int(len(frame)),
        "policy_skipped": int((~frame["policy_valid"]).sum()),
        "policy_acc": float(valid["policy_correct"].mean()) if len(valid) else float("nan"),
        "value_acc": float((frame["value_true"] == frame["value_pred"]).mean()) if len(frame) else float("nan"),
        "policy_acc_by_mate": {int(k): float(v) for k, v in by_mate.items()},
    }


def paired_frame(a: pd.DataFrame, b: pd.DataFrame) -> Optional[pd.DataFrame]:
    aligned = (
        len(a) == len(b)
        and (a["mate_n"].to_numpy() == b["mate_n"].to_numpy()).all()
        and (a["n_legal"].to_numpy() == b["n_legal"].to_numpy()).all()
    )
    if not aligned:
        return None
    mask = (a["policy_valid"] & b["policy_valid"]).to_numpy()
    return pd.DataFrame(
        {
            "mate_n": a["mate_n"].to_numpy()[mask],
            f"{TIME_DECAY}_correct": a["policy_correct"].to_numpy()[mask],
            f"{BASIC}_correct": b["policy_correct"].to_numpy()[mask],
        }
    )


def _safe(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:
        logger.exception("grafico '%s' non generato", name)
        return None


def make_plots(
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    frames: Mapping[str, pd.DataFrame],
    plots_dir: str,
    mate_range: Tuple[int, int],
    eval_name: str,
) -> None:
    os.makedirs(plots_dir, exist_ok=True)

    def out(name: str) -> str:
        return os.path.join(plots_dir, name)

    _safe("history", lambda: plotter.plot_training_history(histories, save_path=out("history.png")))

    long = pd.concat([f.assign(model=name) for name, f in frames.items()], ignore_index=True)
    policy = long[long["policy_valid"]].assign(correct=lambda d: d["policy_correct"].astype(bool))
    value = long.assign(correct=lambda d: d["value_true"] == d["value_pred"])

    _safe(
        "policy_accuracy_by_mate",
        lambda: plotter.plot_accuracy_by_group(
            policy,
            "mate_n",
            xlabel="Mate in n",
            ylabel="Policy accuracy",
            title=f"Policy accuracy by mate depth ({eval_name})",
            save_path=out("policy_accuracy_by_mate.png"),
        ),
    )
    _safe(
        "value_accuracy_by_mate",
        lambda: plotter.plot_accuracy_by_group(
            value,
            "mate_n",
            xlabel="Mate in n",
            ylabel="Value accuracy",
            title=f"Value accuracy by mate depth ({eval_name})",
            save_path=out("value_accuracy_by_mate.png"),
        ),
    )
    _safe(
        "policy_accuracy_by_legal_moves",
        lambda: plotter.plot_accuracy_by_bins(
            policy,
            "n_legal",
            4,
            xlabel="Legal moves",
            ylabel="Policy accuracy",
            title=f"Policy accuracy by number of legal moves ({eval_name})",
            save_path=out("policy_accuracy_by_legal_moves.png"),
        ),
    )

    class_names = [f"mate {n}" for n in range(mate_range[0], mate_range[1] + 1)]
    _safe(
        "confusion",
        lambda: plotter.plot_confusion_matrices(
            {name: (f["value_true"], f["value_pred"]) for name, f in frames.items()},
            class_names,
            save_path=out("value_confusion.png"),
        ),
    )

    if BASIC in frames and TIME_DECAY in frames:
        wide = paired_frame(frames[TIME_DECAY], frames[BASIC])
        if wide is None:
            logger.warning("confronto pairwise saltato: le predizioni dei due modelli non sono allineate")
        else:
            colors = plotter.model_colors([TIME_DECAY, BASIC])

            def paired() -> None:
                _, table = plotter.plot_paired_comparison(
                    wide,
                    "mate_n",
                    f"{TIME_DECAY}_correct",
                    f"{BASIC}_correct",
                    label_a="time_decay",
                    label_b="basic",
                    color_a=colors[TIME_DECAY],
                    color_b=colors[BASIC],
                    xlabel="Mate in n",
                    title=f"Policy: time_decay vs basic ({eval_name})",
                    save_path=out("paired_time_decay_vs_basic.png"),
                )
                table.to_csv(out("paired_time_decay_vs_basic.csv"), index=False)

            _safe("paired", paired)

    plt.close("all")