from __future__ import annotations

import math
import os
from functools import wraps
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

__all__ = [
    "STYLE",
    "METRICS",
    "model_colors",
    "save_figure",
    "load_debug_frames",
    "wilson_interval",
    "accuracy_table",
    "mcnemar_exact",
    "mcnemar_by_group",
    "confusion_counts",
    "plot_training_history",
    "plot_accuracy_by_group",
    "plot_accuracy_by_bins",
    "plot_paired_comparison",
    "plot_confusion_matrices",
    "plot_board_heatmap",
    "plot_dataset_overview",
]

STYLE: Dict[str, Any] = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}

PALETTE: Tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000",
)
FIXED_COLORS: Dict[str, str] = {
    "gat_basic": "#0072B2",
    "gat_time_decay": "#D55E00",
    "llm": "#009E73",
}
METRICS: Tuple[Tuple[str, str], ...] = (
    ("policy_loss", "Policy loss"),
    ("value_loss", "Value loss"),
    ("policy_acc", "Policy accuracy"),
    ("value_acc", "Value accuracy"),
)
SQUARE_FILES = "abcdefgh"


def model_colors(names: Iterable[Any]) -> Dict[Any, str]:
    fixed = set(FIXED_COLORS.values())
    free = [c for c in PALETTE if c not in fixed]
    colors: Dict[Any, str] = {}
    i = 0
    for name in names:
        if name in colors:
            continue
        if name in FIXED_COLORS:
            colors[name] = FIXED_COLORS[name]
        else:
            colors[name] = free[i % len(free)]
            i += 1
    return colors


def styled(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return fn(*args, **kwargs)

    return wrapper


def save_figure(fig: Figure, path: str, formats: Sequence[str] = ("png",)) -> List[str]:
    _, ext = os.path.splitext(path)
    if ext.lower() in {".png", ".pdf", ".svg"}:
        targets = [path]
    else:
        targets = [f"{path}.{fmt}" for fmt in formats]
    os.makedirs(os.path.dirname(os.path.abspath(targets[0])), exist_ok=True)
    for target in targets:
        fig.savefig(target)
    return targets


def _finish(fig: Figure, save_path: Optional[str], show: bool) -> Figure:
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    return fig


def _get_ax(ax: Optional[Axes], figsize: Tuple[float, float]) -> Tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _group_order(series: pd.Series) -> List[Any]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.dropna().unique())
        return [c for c in series.cat.categories if c in present]
    return sorted(series.dropna().unique())


def load_debug_frames(paths: Iterable[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not frames:
        raise FileNotFoundError("nessun CSV di debug trovato")
    df = pd.concat(frames, ignore_index=True)
    if "clock_is_real" in df.columns:
        df["clock_is_real"] = (
            df["clock_is_real"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        )
    return df


def wilson_interval(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def accuracy_table(
    df: pd.DataFrame,
    group_col: str,
    model_col: str = "model",
    correct_col: str = "correct",
) -> pd.DataFrame:
    work = df.assign(**{correct_col: df[correct_col].astype(int)})
    table = (
        work.groupby([model_col, group_col], observed=True)[correct_col]
        .agg(k="sum", n="count")
        .reset_index()
    )
    intervals = [wilson_interval(k, n) for k, n in zip(table["k"], table["n"])]
    table["acc"] = table["k"] / table["n"]
    table["lo"] = [i[0] for i in intervals]
    table["hi"] = [i[1] for i in intervals]
    return table


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(only_a, only_b), n, 0.5, alternative="two-sided").pvalue)


def _adjust_pvalues(pvalues: Sequence[float], method: Optional[str]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if method is None or len(p) <= 1:
        return p
    m = len(p)
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    if method == "holm":
        adjusted = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(np.argsort(p)):
            running = max(running, (m - rank) * p[idx])
            adjusted[idx] = min(running, 1.0)
        return adjusted
    raise ValueError(f"correction sconosciuta: {method}")


def mcnemar_by_group(
    df: pd.DataFrame,
    group_col: str,
    col_a: str,
    col_b: str,
    correction: Optional[str] = "holm",
) -> pd.DataFrame:
    rows = []
    for g in _group_order(df[group_col]):
        sub = df[df[group_col] == g]
        a = sub[col_a].astype(bool).to_numpy()
        b = sub[col_b].astype(bool).to_numpy()
        only_a = int((a & ~b).sum())
        only_b = int((~a & b).sum())
        rows.append(
            {
                group_col: g,
                "n": len(sub),
                "k_a": int(a.sum()),
                "k_b": int(b.sum()),
                "acc_a": float(a.mean()),
                "acc_b": float(b.mean()),
                "only_a": only_a,
                "only_b": only_b,
                "p_value": mcnemar_exact(only_a, only_b),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out["p_adjusted"] = []
        return out
    out["p_adjusted"] = _adjust_pvalues(out["p_value"], correction)
    return out


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    if t.size and (t.min() < 0 or p.min() < 0 or t.max() >= num_classes or p.max() >= num_classes):
        raise ValueError(f"etichette fuori da [0, {num_classes - 1}]")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


def _bin_series(series: pd.Series, bins: Any) -> pd.Series:
    if isinstance(bins, int):
        binned = pd.qcut(series, q=bins, duplicates="drop")
    else:
        binned = pd.cut(series, bins=list(bins), include_lowest=True)
    labels = [f"{iv.left:.4g}-{iv.right:.4g}" for iv in binned.cat.categories]
    if len(set(labels)) != len(labels):
        labels = [f"{i}: {label}" for i, label in enumerate(labels)]
    return binned.cat.rename_categories(labels)


@styled
def plot_training_history(
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    metrics: Sequence[Tuple[str, str]] = METRICS,
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    colors = model_colors(histories.keys())
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize or (6.0 * ncols, 4.0 * nrows), squeeze=False
    )
    flat = axes.ravel()
    for ax, (key, label) in zip(flat, metrics):
        for name, history in histories.items():
            epochs = [h["epoch"] for h in history]
            for split, linestyle in (("train", "--"), ("val", "-")):
                values = [h[split][key] for h in history]
                ax.plot(
                    epochs,
                    values,
                    linestyle=linestyle,
                    color=colors[name],
                    marker="o",
                    markersize=3,
                    label=f"{name} {split}",
                )
        ax.set_title(label)
        ax.set_xlabel("Epoch")
    for ax in flat[len(metrics):]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _finish(fig, save_path, show)


@styled
def plot_accuracy_by_group(
    df: pd.DataFrame,
    group_col: str,
    model_col: str = "model",
    correct_col: str = "correct",
    order: Optional[Sequence[Any]] = None,
    models: Optional[Sequence[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Accuracy",
    title: Optional[str] = None,
    ylim: Tuple[float, float] = (0.0, 1.0),
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (9.0, 5.0),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    table = accuracy_table(df, group_col, model_col, correct_col)
    model_list = list(models) if models else list(pd.unique(df[model_col]))
    groups = list(order) if order is not None else _group_order(df[group_col])
    counts = table.groupby(group_col, observed=True)["n"].max().reindex(groups).fillna(0)
    colors = model_colors(model_list)
    fig, ax = _get_ax(ax, figsize)

    width = 0.8 / len(model_list)
    x = np.arange(len(groups))
    for i, name in enumerate(model_list):
        sub = table[table[model_col] == name].set_index(group_col).reindex(groups)
        acc = sub["acc"].to_numpy(dtype=float)
        lo = sub["lo"].to_numpy(dtype=float)
        hi = sub["hi"].to_numpy(dtype=float)
        ax.bar(
            x + (i - (len(model_list) - 1) / 2.0) * width,
            acc,
            width,
            yerr=np.vstack([acc - lo, hi - acc]),
            color=colors[name],
            label=name,
            capsize=3,
            edgecolor="white",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={int(counts[g])})" for g in groups])
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel or group_col)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=max(len(model_list), 1))
    return _finish(fig, save_path, show)


def plot_accuracy_by_bins(
    df: pd.DataFrame,
    value_col: str,
    bins: Any,
    model_col: str = "model",
    correct_col: str = "correct",
    **kwargs: Any,
) -> Figure:
    bin_col = f"{value_col}_bin"
    work = df.assign(**{bin_col: _bin_series(df[value_col], bins)})
    kwargs.setdefault("xlabel", value_col)
    return plot_accuracy_by_group(work, bin_col, model_col, correct_col, **kwargs)


@styled
def plot_paired_comparison(
    df: pd.DataFrame,
    group_col: str,
    col_a: str,
    col_b: str,
    label_a: str = "GNN",
    label_b: str = "LLM",
    color_a: str = "#0072B2",
    color_b: str = "#009E73",
    correction: Optional[str] = "holm",
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (9.0, 5.0),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[Figure, pd.DataFrame]:
    table = mcnemar_by_group(df, group_col, col_a, col_b, correction)
    fig, ax = _get_ax(ax, figsize)
    x = np.arange(len(table))
    width = 0.38
    tops = np.zeros(len(table))
    for offset, k_col, acc_col, color, label in (
        (-width / 2.0, "k_a", "acc_a", color_a, label_a),
        (width / 2.0, "k_b", "acc_b", color_b, label_b),
    ):
        acc = table[acc_col].to_numpy(dtype=float)
        intervals = [wilson_interval(k, n) for k, n in zip(table[k_col], table["n"])]
        lo = np.array([i[0] for i in intervals])
        hi = np.array([i[1] for i in intervals])
        ax.bar(
            x + offset,
            acc,
            width,
            yerr=np.vstack([acc - lo, hi - acc]),
            color=color,
            label=label,
            capsize=3,
            edgecolor="white",
            linewidth=0.6,
            error_kw={"elinewidth": 1.0},
        )
        tops = np.maximum(tops, hi)
    for xi, top, p in zip(x, tops, table["p_adjusted"]):
        ax.text(xi, min(top + 0.02, 1.02), _stars(p), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(table[group_col], table["n"])])
    ax.set_ylim(0.0, 1.12)
    ax.set_xlabel(xlabel or group_col)
    ax.set_ylabel("Accuracy")
    if title:
        ax.set_title(title, pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return _finish(fig, save_path, show), table


def _draw_confusion(
    ax: Axes,
    cm: np.ndarray,
    class_names: Sequence[str],
    normalize: Optional[str],
    cmap: str,
) -> None:
    if normalize == "true":
        matrix = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    elif normalize == "pred":
        matrix = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)
    elif normalize == "all":
        matrix = cm / max(cm.sum(), 1)
    elif normalize is None:
        matrix = cm.astype(float)
    else:
        raise ValueError(f"normalize sconosciuto: {normalize}")

    ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0 if normalize else None)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = f"{matrix[i, j]:.2f}\n{cm[i, j]}" if normalize else f"{cm[i, j]}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[i, j] > threshold else "#222222",
            )


@styled
def plot_confusion_matrices(
    results: Mapping[str, Tuple[Sequence[int], Sequence[int]]],
    class_names: Sequence[str],
    normalize: Optional[str] = "true",
    cmap: str = "Blues",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    num_classes = len(class_names)
    fig, axes = plt.subplots(
        1, len(results), figsize=figsize or (4.6 * len(results), 4.4), squeeze=False
    )
    for ax, (name, (y_true, y_pred)) in zip(axes[0], results.items()):
        cm = confusion_counts(y_true, y_pred, num_classes)
        accuracy = cm.trace() / max(cm.sum(), 1)
        _draw_confusion(ax, cm, class_names, normalize, cmap)
        ax.set_title(f"{name} (acc={accuracy:.3f})")
    fig.tight_layout()
    return _finish(fig, save_path, show)


def _draw_move(ax: Axes, move: Tuple[int, int], color: str) -> None:
    src, dst = move
    ax.annotate(
        "",
        xy=(dst % 8, dst // 8),
        xytext=(src % 8, src // 8),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2, "shrinkA": 4, "shrinkB": 4},
    )


@styled
def plot_board_heatmap(
    values: Any,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    best_move: Optional[Tuple[int, int]] = None,
    predicted_move: Optional[Tuple[int, int]] = None,
    colorbar: bool = True,
    figsize: Tuple[float, float] = (5.6, 5.0),
    save_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    grid = np.asarray(values, dtype=float).reshape(8, 8)
    fig, ax = _get_ax(ax, figsize)
    image = ax.imshow(grid, origin="lower", cmap=cmap)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.set_xticks(range(8))
    ax.set_xticklabels(list(SQUARE_FILES))
    ax.set_yticks(range(8))
    ax.set_yticklabels([str(i) for i in range(1, 9)])
    handles = []
    if best_move is not None:
        _draw_move(ax, best_move, "#D55E00")
        handles.append(Line2D([0], [0], color="#D55E00", lw=2.2, label="best move"))
    if predicted_move is not None:
        _draw_move(ax, predicted_move, "#CC79A7")
        handles.append(Line2D([0], [0], color="#CC79A7", lw=2.2, label="predicted move"))
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(handles))
    if colorbar:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    return _finish(fig, save_path, show)


def _panel_mate_by_source(ax: Axes, df: pd.DataFrame) -> None:
    counts = df.groupby(["mate_n", "source"]).size().unstack(fill_value=0).sort_index()
    colors = model_colors(counts.columns)
    labels = [str(i) for i in counts.index]
    bottom = np.zeros(len(counts))
    for source in counts.columns:
        values = counts[source].to_numpy()
        ax.bar(labels, values, bottom=bottom, color=colors[source], label=source, edgecolor="white", linewidth=0.5)
        bottom += values
    ax.set_title("Positions per mate depth")
    ax.set_xlabel("Mate in n")
    ax.set_ylabel("Positions")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))


def _panel_rating(ax: Axes, df: pd.DataFrame) -> None:
    work = df.assign(_rating=pd.to_numeric(df["rating"], errors="coerce")).dropna(subset=["_rating"])
    edges = np.histogram_bin_edges(work["_rating"], bins=30)
    colors = model_colors(pd.unique(work["source"]))
    for source, sub in work.groupby("source"):
        ax.hist(sub["_rating"], bins=edges, density=True, histtype="step", linewidth=1.6, color=colors[source], label=source)
    ax.set_title("Rating distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)


def _panel_clock(ax: Axes, df: pd.DataFrame) -> None:
    work = df.assign(_clock=pd.to_numeric(df["clock_seconds"], errors="coerce"))
    work = work[work["_clock"] > 0]
    lo, hi = float(work["_clock"].min()), float(work["_clock"].max())
    if lo == hi:
        lo, hi = lo * 0.9, hi * 1.1
    edges = np.logspace(np.log10(lo), np.log10(hi), 40)
    for flag, label, color in ((True, "real", "#0072B2"), (False, "synthetic", "#D55E00")):
        sub = work[work["clock_is_real"] == flag]
        if len(sub):
            ax.hist(sub["_clock"], bins=edges, density=True, histtype="step", linewidth=1.6, color=color, label=f"{label} (n={len(sub)})")
    ax.set_xscale("log")
    ax.set_title("Move time")
    ax.set_xlabel("Seconds (log scale)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)


def _panel_split(ax: Axes, df: pd.DataFrame) -> None:
    counts = df.groupby(["split", "mate_n"]).size().unstack(fill_value=0)
    known = ("train", "val", "test")
    order = [s for s in known if s in counts.index] + [s for s in counts.index if s not in known]
    fractions = counts.loc[order].div(counts.loc[order].sum(axis=1), axis=0)
    colors = model_colors(fractions.columns)
    bottom = np.zeros(len(fractions))
    for mate in fractions.columns:
        values = fractions[mate].to_numpy()
        ax.bar(order, values, bottom=bottom, color=colors[mate], label=f"mate {mate}", edgecolor="white", linewidth=0.5)
        bottom += values
    ax.set_title("Mate depth composition per split")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))


@styled
def plot_dataset_overview(
    df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Figure:
    work = df.copy()
    if "clock_is_real" in work.columns:
        work["clock_is_real"] = (
            work["clock_is_real"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        )
    available = set(work.columns)
    candidates = (
        ({"mate_n", "source"}, _panel_mate_by_source),
        ({"rating", "source"}, _panel_rating),
        ({"clock_seconds", "clock_is_real"}, _panel_clock),
        ({"split", "mate_n"}, _panel_split),
    )
    panels = [fn for required, fn in candidates if required <= available]
    if not panels:
        raise ValueError("nessuna colonna utilizzabile per l'overview")
    ncols = 2 if len(panels) > 1 else 1
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (6.5 * ncols, 4.2 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax, fn in zip(flat, panels):
        fn(ax, work)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    return _finish(fig, save_path, show)