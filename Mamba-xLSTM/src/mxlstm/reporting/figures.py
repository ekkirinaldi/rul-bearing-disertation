"""Centralized figure generation for the dissertation report.

Design rule (one-insight-per-file)
----------------------------------
Every PNG produced by this module shows **exactly one chart**: a single
plot, a single bar group, or a single matrix. We never compose subplot
grids — when several views are needed (e.g. train/val/test curves),
each becomes its own file. This keeps the dissertation figures and the
HTML/PDF report easy to caption, embed at full width, and discuss
individually.

Functions that previously returned a single ``Path`` for multi-panel
figures now return ``list[Path]`` so callers can register every artefact.

All helpers:
  * accept numpy / Python primitives (no Lightning dependence),
  * write into the caller-chosen folder (creating it if missing),
  * use ``dpi=120`` for crisp embedding in the PDF report.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # headless safety for batch scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_DPI = 120

# Visual constants used across charts.
_TRAIN_COLOR = "#4c78a8"
_VAL_COLOR = "#f58518"
_TEST_COLOR = "#54a24b"
_PRED_COLOR = "#d62728"
_TRUE_COLOR = "#1f77b4"
_GRID_ALPHA = 0.25


def _slug(s: str) -> str:
    """Filesystem-safe slug for metric keys like ``val/rmse`` -> ``val_rmse``."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


# ---------------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------------


def plot_dataset_overview(
    rows: list[dict],
    save_path: str | Path,
    *,
    title: str = "Dataset overview",
) -> Path:
    """Bar chart of bearing lifetimes (acquisition counts) coloured by split.

    rows: list of ``{"bearing_id", "split", "n_acquisitions", "condition"}``.
    Single insight: how much data each bearing contributes per split.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("plot_dataset_overview: empty rows")

    split_order = {"train": 0, "val": 1, "test": 2}
    df = (
        pd.DataFrame(rows)
        .assign(_split_rank=lambda d: d["split"].map(split_order).fillna(99))
        .sort_values(["_split_rank", "condition", "bearing_id"])
        .drop(columns=["_split_rank"])
        .reset_index(drop=True)
    )
    colors = {"train": _TRAIN_COLOR, "val": _VAL_COLOR, "test": _TEST_COLOR}
    bar_colors = [colors.get(s, "#999") for s in df["split"]]

    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(df)), 4), dpi=_DPI)
    bars = ax.bar(np.arange(len(df)), df["n_acquisitions"], color=bar_colors)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["bearing_id"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("# acquisitions")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors.keys()), loc="upper right", fontsize=9, title="split")
    ax.grid(axis="y", alpha=_GRID_ALPHA)
    for bar, n in zip(bars, df["n_acquisitions"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(n)}", ha="center", va="bottom", fontsize=7, color="#333")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# HI feature visualisations
# ---------------------------------------------------------------------------


def plot_hi_heatmap(
    bearing_id: str,
    hi: np.ndarray,
    feature_names: list[str],
    save_path: str | Path,
    *,
    smoothed: bool = True,
) -> Path:
    """Plot the (T, F) HI matrix as a heatmap.

    Single insight: how each feature evolves through life, side by side.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.18 * len(feature_names))), dpi=_DPI)
    im = ax.imshow(hi.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=6)
    ax.set_xlabel("acquisition index")
    ax.set_title(f"HI matrix — {bearing_id} ({'smoothed' if smoothed else 'raw'})")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="scaled HI value")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_hi_trace(
    bearing_id: str,
    hi: np.ndarray,
    feature_names: list[str],
    save_path: str | Path,
    *,
    feature_index: int | None = None,
) -> Path:
    """Plot a single HI feature trace versus acquisition index.

    If ``hi`` has more than one column, picks the most informative feature
    (highest variance) by default. Single insight: how this one HI signal
    degrades through life.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if hi.ndim != 2:
        raise ValueError(f"plot_hi_trace expects (T, F) matrix, got shape {hi.shape}")
    if feature_index is None:
        feature_index = int(np.argmax(hi.var(axis=0))) if hi.shape[1] > 1 else 0
    name = feature_names[feature_index] if feature_index < len(feature_names) else f"f{feature_index}"

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=_DPI)
    ax.plot(hi[:, feature_index], color=_TRAIN_COLOR, lw=1.2)
    ax.set_xlabel("acquisition index")
    ax.set_ylabel(f"HI value (scaled) — {name}")
    ax.set_title(f"Health indicator trace — {bearing_id} ({name})")
    ax.grid(alpha=_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_rul_label_per_bearing(
    bearing_id: str,
    rul: np.ndarray,
    save_path: str | Path,
    *,
    onset_index: int | None = None,
    eol_index: int | None = None,
) -> Path:
    """Plot the RUL target curve for a single bearing.

    Single insight: how this bearing's training label was constructed
    (healthy plateau → degradation onset → linear decay to 0). Onset and
    EOL are annotated as vertical reference lines when supplied.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rul = np.asarray(rul, dtype=float)
    if onset_index is None and rul.size:
        below = np.where(rul < 1.0 - 1e-6)[0]
        onset_index = int(below[0]) if below.size else None
    if eol_index is None and rul.size:
        eol_index = int(rul.size - 1)

    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=_DPI)
    ax.plot(rul, color=_TRUE_COLOR, lw=1.5, label="RUL target")
    if onset_index is not None:
        ax.axvline(onset_index, color=_VAL_COLOR, lw=1.0, ls="--",
                   label=f"degradation onset (t={onset_index})")
    if eol_index is not None:
        ax.axvline(eol_index, color=_PRED_COLOR, lw=1.0, ls=":",
                   label=f"end-of-life (t={eol_index})")
    ax.set_xlabel("acquisition index")
    ax.set_ylabel("RUL (normalised)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"RUL label — {bearing_id}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_rul_labels_for_bearings(
    items: list[tuple[str, np.ndarray]],
    save_dir: str | Path,
) -> list[Path]:
    """Convenience wrapper: emit one ``rul_label_<bid>.png`` per bearing.

    Each file shows the RUL target curve of a single bearing (single
    insight per file). Use this instead of overlaying many bearings on
    one axis — bearings have different lifetimes and the overlay was
    impossible to read.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for bid, rul in items:
        p = save_dir / f"rul_label_{bid}.png"
        plot_rul_label_per_bearing(bid, np.asarray(rul), p)
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Training curves — one PNG per metric
# ---------------------------------------------------------------------------


def plot_metric_curve(
    metrics_csv: str | Path,
    key: str,
    save_path: str | Path,
) -> Path | None:
    """Plot a single metric column from Lightning's ``metrics.csv``.

    Returns the saved path, or ``None`` if the metric is missing/empty.
    Single insight: how this one metric evolves over epochs.
    """
    metrics_csv = Path(metrics_csv)
    save_path = Path(save_path)
    if not metrics_csv.exists():
        return None
    df = pd.read_csv(metrics_csv)
    if key not in df.columns:
        return None
    sub = df[["epoch", key]].dropna()
    if sub.empty:
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)

    agg = sub.groupby("epoch", as_index=False)[key].mean()
    is_loss = "loss" in key.lower() or "rmse" in key.lower() or "mae" in key.lower()
    color = _TRAIN_COLOR if key.startswith("train") else _VAL_COLOR
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=_DPI)
    ax.plot(agg["epoch"], agg[key], marker="o", lw=1.5, ms=3, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel(key)
    direction = "lower is better" if is_loss else "higher is better"
    ax.set_title(f"{key} — {direction}")
    ax.grid(alpha=_GRID_ALPHA)
    best_idx = int(agg[key].idxmin() if is_loss else agg[key].idxmax())
    best_epoch = float(agg.loc[best_idx, "epoch"])
    best_val = float(agg.loc[best_idx, key])
    ax.axvline(best_epoch, color="#888", lw=0.8, ls=":")
    ax.scatter([best_epoch], [best_val], color=color, s=40, zorder=5,
               edgecolor="black", linewidth=0.5)
    ax.annotate(
        f"best: {best_val:.4f}\nepoch {int(best_epoch)}",
        xy=(best_epoch, best_val),
        xytext=(8, -10), textcoords="offset points",
        fontsize=8, color="#222",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", lw=0.5),
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_training_curves(
    metrics_csv: str | Path,
    save_dir: str | Path,
    *,
    keys: tuple[str, ...] = (
        "train/loss",
        "train/loss_epoch",
        "val/loss",
        "val/rmse",
        "val/mae",
        "val/phm_score",
    ),
    prefix: str = "training_",
) -> list[Path]:
    """Emit one PNG per available metric from Lightning's ``metrics.csv``.

    Each file holds a single metric line chart with the best-epoch marker
    annotated. Returns the list of saved paths (empty if the CSV is
    missing or no requested keys are present).
    """
    metrics_csv = Path(metrics_csv)
    save_dir = Path(save_dir)
    if not metrics_csv.exists():
        return []
    save_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for k in keys:
        path = save_dir / f"{prefix}{_slug(k)}.png"
        result = plot_metric_curve(metrics_csv, k, path)
        if result is not None:
            out.append(result)
    return out


# ---------------------------------------------------------------------------
# Prediction curves
# ---------------------------------------------------------------------------


def _per_bearing_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    diff = pred - y
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
    mae = float(np.mean(np.abs(diff))) if diff.size else float("nan")
    return {"rmse": rmse, "mae": mae}


def plot_per_bearing_predictions(
    per_bearing: dict[str, dict[str, np.ndarray]],
    save_dir: str | Path,
) -> list[Path]:
    """One PNG per bearing comparing true vs predicted RUL.

    Single insight per file: tracking quality on this specific bearing.
    Per-bearing RMSE/MAE are annotated in a corner box for context.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for bid, d in sorted(per_bearing.items()):
        path = save_dir / f"pred_{bid}.png"
        y = np.asarray(d["y"])
        pred = np.asarray(d["pred"])
        t = np.asarray(d["t"])
        m = _per_bearing_metrics(y, pred)

        fig, ax = plt.subplots(figsize=(8, 3.4), dpi=_DPI)
        ax.plot(t, y, label="true RUL", color=_TRUE_COLOR, lw=1.6)
        ax.plot(t, pred, label="predicted RUL", color=_PRED_COLOR, lw=1.0, alpha=0.85)
        below = np.where(y < 1.0 - 1e-6)[0]
        if below.size:
            ax.axvline(t[int(below[0])], color="#888", lw=0.8, ls="--",
                       label="degradation onset")
        ax.set_xlabel("acquisition index")
        ax.set_ylabel("RUL (normalised)")
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"Test predictions — {bid}")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=_GRID_ALPHA)
        ax.text(
            0.99, 0.95,
            f"RMSE={m['rmse']:.4f}\nMAE ={m['mae']:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", lw=0.5),
        )
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        out.append(path)
    return out


def plot_per_bearing_residuals(
    per_bearing: dict[str, dict[str, np.ndarray]],
    save_dir: str | Path,
) -> list[Path]:
    """One PNG per bearing: residual (pred − true) over time.

    Single insight per file: where on the life curve this bearing is
    over-/under-predicted.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for bid, d in sorted(per_bearing.items()):
        path = save_dir / f"residual_{bid}.png"
        y = np.asarray(d["y"])
        pred = np.asarray(d["pred"])
        t = np.asarray(d["t"])
        residual = pred - y
        fig, ax = plt.subplots(figsize=(8, 3.0), dpi=_DPI)
        ax.plot(t, residual, color="#555", lw=1.0)
        ax.axhline(0, color="black", lw=0.6)
        ax.fill_between(t, residual, 0, where=residual >= 0,
                        color=_PRED_COLOR, alpha=0.15, label="over-predict")
        ax.fill_between(t, residual, 0, where=residual < 0,
                        color=_TRAIN_COLOR, alpha=0.15, label="under-predict")
        ax.set_xlabel("acquisition index")
        ax.set_ylabel("residual (pred − true)")
        ax.set_title(f"Residual over time — {bid}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        out.append(path)
    return out


def plot_residuals_overall(
    per_bearing: dict[str, dict[str, np.ndarray]],
    save_path: str | Path,
) -> Path:
    """Aggregate residual scatter: residual vs true RUL across all test bearings.

    Single insight: bias direction as a function of remaining life.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ys = np.concatenate([d["y"] for d in per_bearing.values()])
    ps = np.concatenate([d["pred"] for d in per_bearing.values()])
    res = ps - ys
    fig, ax = plt.subplots(figsize=(6, 4), dpi=_DPI)
    ax.scatter(ys, res, s=6, alpha=0.35, color="#444")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("true RUL")
    ax.set_ylabel("residual (pred − true)")
    mean_bias = float(res.mean())
    ax.set_title(f"Residuals across all test bearings (mean bias = {mean_bias:+.4f})")
    ax.grid(alpha=_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_residual_histogram(
    per_bearing: dict[str, dict[str, np.ndarray]],
    save_path: str | Path,
) -> Path:
    """Histogram of residuals across all test bearings.

    Single insight: the distribution shape of prediction error (bias /
    spread / tails). Complements the scatter view above.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ys = np.concatenate([d["y"] for d in per_bearing.values()])
    ps = np.concatenate([d["pred"] for d in per_bearing.values()])
    res = ps - ys
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=_DPI)
    ax.hist(res, bins=40, color=_TRAIN_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axvline(float(res.mean()), color=_PRED_COLOR, lw=1.0, ls=":",
               label=f"mean = {float(res.mean()):+.4f}")
    ax.set_xlabel("residual (pred − true)")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution (all test bearings)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Ablation comparison — one PNG per metric
# ---------------------------------------------------------------------------


def _metric_direction(metric_key: str) -> str:
    k = metric_key.lower()
    if "rmse" in k or "mae" in k or "loss" in k or k.endswith("phm_score_paper"):
        return "lower is better"
    return "higher is better"


def plot_ablation_metric(
    agg: dict[str, dict[str, dict[str, float]]],
    metric_key: str,
    save_path: str | Path,
) -> Path | None:
    """Bar chart for a single metric across runs.

    Single insight: how each configuration scores on this one metric.
    Bars are sorted from best → worst given the metric direction so the
    reader's eye naturally lands on the winner.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, metrics in agg.items():
        entry = metrics.get(metric_key)
        if not entry or entry.get("mean") is None:
            continue
        rows.append({
            "run": run,
            "mean": float(entry["mean"]),
            "std": float(entry.get("std", 0.0) or 0.0),
            "n": int(entry.get("n", 0) or 0),
        })
    if not rows:
        return None
    direction = _metric_direction(metric_key)
    df = pd.DataFrame(rows).sort_values(
        "mean", ascending=(direction == "lower is better")
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(df) + 2), 4), dpi=_DPI)
    bars = ax.bar(np.arange(len(df)), df["mean"], yerr=df["std"], capsize=3,
                  color=_TRAIN_COLOR, edgecolor="#1c4571")
    if len(bars):
        bars[0].set_color(_TEST_COLOR)
        bars[0].set_edgecolor("#2f6e35")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["run"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key} across runs — {direction} (best run highlighted)")
    ax.grid(axis="y", alpha=_GRID_ALPHA)
    for bar, mean, n in zip(bars, df["mean"], df["n"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{mean:.4f}\n(n={n})",
                ha="center", va="bottom", fontsize=7, color="#222")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_ablation_per_metric(
    agg: dict[str, dict[str, dict[str, float]]],
    save_dir: str | Path,
    *,
    metric_keys: Iterable[str] = ("test/rmse", "test/mae", "test/r2", "test/phm_score"),
    prefix: str = "ablation_",
) -> list[Path]:
    """Emit one bar chart PNG per metric (no grouped bars).

    Returns the list of saved paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for k in metric_keys:
        path = save_dir / f"{prefix}{_slug(k)}.png"
        result = plot_ablation_metric(agg, k, path)
        if result is not None:
            out.append(result)
    return out


# ---------------------------------------------------------------------------
# Step timing chart (from RunLogger summary.json)
# ---------------------------------------------------------------------------


def plot_step_timings(timings: list[dict], save_path: str | Path) -> Path | None:
    """Horizontal bar chart of step elapsed seconds (excludes phase totals).

    Single insight: which pipeline step dominated wall-time.
    """
    save_path = Path(save_path)
    rows = [t for t in timings if t.get("kind") == "step"]
    if not rows:
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: float(r["elapsed_s"]), reverse=True)
    labels = [f"{r.get('phase') or '-'} :: {r['name']}" for r in rows]
    times = [float(r["elapsed_s"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(rows))), dpi=_DPI)
    bars = ax.barh(np.arange(len(rows)), times, color=_TRAIN_COLOR)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("seconds")
    ax.set_title("Pipeline step timings (sorted by duration)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=_GRID_ALPHA)
    for bar, t in zip(bars, times, strict=False):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {t:.2f}s", va="center", fontsize=7, color="#333")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
