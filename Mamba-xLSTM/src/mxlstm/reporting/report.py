"""Build a single HTML+PDF dissertation report from one or more run directories.

A "run directory" is anything written by ``scripts/train.py`` (with the new
logging upgrade): it contains ``summary.json``, optional ``config.yaml``,
``logs/events.jsonl`` + ``logs/summary.json``, ``figures/*.png``, and
optionally ``interp/*.png``.

Usage:

    from mxlstm.reporting.report import build_report
    build_report(
        run_dirs=[Path("results/runs/mamba_xlstm_phm2012_s42")],
        out_html=Path("results/report.html"),
        out_pdf=Path("results/report.pdf"),
        title="Mamba-xLSTM RUL prediction — Mac M-series run",
    )

Or via the CLI ``scripts/build_report.py``.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from mxlstm.reporting.figures import (
    plot_ablation_per_metric,
    plot_per_bearing_predictions,
    plot_per_bearing_residuals,
    plot_residual_histogram,
    plot_residuals_overall,
    plot_step_timings,
)
from mxlstm.reporting.tables import aggregate_by_run_id_prefix, to_markdown


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


# ---------------------------------------------------------------------------
# Helpers to build per-run context
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _embed_image(path: Path) -> str:
    """Inline an image as a data URI so the HTML/PDF is self-contained."""
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{data}"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_") or "run"


def _format_ts(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]


def _select_events(events_path: Path, *, max_events: int = 80) -> list[dict[str, Any]]:
    if not events_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with events_path.open() as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") not in {"phase_start", "phase_end", "step_start", "step_end",
                                      "metric", "artefact", "log"}:
                continue
            if ev.get("kind") == "log" and ev.get("level") not in {"WARNING", "ERROR", "INFO"}:
                continue
            ev["ts_str"] = _format_ts(float(ev["ts"]))
            ev.setdefault("elapsed_s", None)
            rows.append(ev)
    # Keep all phase/step events but cap noisy logs.
    structural = [e for e in rows if e["kind"] in {"phase_start", "phase_end", "step_start", "step_end"}]
    other = [e for e in rows if e["kind"] not in {"phase_start", "phase_end", "step_start", "step_end"}]
    other = other[: max_events - len(structural)] if max_events > len(structural) else []
    merged = sorted(structural + other, key=lambda e: e["ts"])
    return merged


def _list_named_figures(
    paths: list[Path],
    *,
    prefix: str,
    id_key: str,
    limit: int | None,
) -> tuple[list[dict[str, str]], int]:
    selected = paths if limit is None else paths[:limit]
    out = []
    for p in selected:
        out.append({id_key: p.stem.replace(prefix, ""), "path": _embed_image(p)})
    omitted = max(0, len(paths) - len(selected))
    return out, omitted


def _list_pred_figures(figures_dir: Path, *, limit: int | None) -> tuple[list[dict[str, str]], int]:
    return _list_named_figures(
        sorted(figures_dir.glob("pred_*.png")),
        prefix="pred_",
        id_key="bearing_id",
        limit=limit,
    )


def _list_residual_figures(
    figures_dir: Path,
    *,
    limit: int | None,
) -> tuple[list[dict[str, str]], int]:
    return _list_named_figures(
        sorted(figures_dir.glob("residual_*.png")),
        prefix="residual_",
        id_key="bearing_id",
        limit=limit,
    )


def _list_rul_label_figures(
    figures_dir: Path,
    *,
    limit: int | None,
) -> tuple[list[dict[str, str]], int]:
    return _list_named_figures(
        sorted(figures_dir.glob("rul_label_*.png")),
        prefix="rul_label_",
        id_key="bearing_id",
        limit=limit,
    )


def _list_training_curve_figures(figures_dir: Path) -> list[dict[str, str]]:
    out = []
    for p in sorted(figures_dir.glob("training_*.png")):
        key = p.stem.replace("training_", "").replace("_", "/")
        out.append({"key": key, "path": _embed_image(p)})
    return out


def _list_ablation_figures(figures_dir: Path) -> list[dict[str, str]]:
    out = []
    for p in sorted(figures_dir.glob("ablation_*.png")):
        key = p.stem.replace("ablation_", "").replace("_", "/")
        out.append({"key": key, "path": _embed_image(p)})
    return out


def _list_interp_figures(interp_dir: Path, *, key_prefix: str = "") -> dict[str, str]:
    if not interp_dir.exists():
        return {}
    captions = {
        "shap_global.png": "SHAP global feature importance.",
        "sae_umap_clusters.png": "SAE latent space (UMAP, HDBSCAN clusters).",
    }
    out: dict[str, str] = {}
    for p in sorted(interp_dir.glob("*.png")):
        cap = captions.get(p.name) or p.stem.replace("_", " ")
        if key_prefix:
            cap = key_prefix + cap
        out[cap] = _embed_image(p)
    return out


# ---------------------------------------------------------------------------
# Per-run context
# ---------------------------------------------------------------------------


@dataclass
class _Figure:
    key: str
    path: Path
    caption: str


@dataclass
class RunContext:
    run_id: str
    run_dir: str
    seed: int | None
    n_params: int | None
    fit_seconds: float | None
    dataset: str | None
    model_name: str | None
    test_metrics: dict[str, Any]
    figures: dict[str, Any]
    events: list[dict[str, Any]]
    config_text: str | None
    n_generated_figures: int = 0


# Suffix segment in ``algorithm_comparison_<dataset>_<model_key>_s<seed>`` run ids.
_AGGREGATE_MODEL_KEYS: tuple[str, ...] = (
    "sparse_gate_tcn_rul",
    "phase_moe_xlstm_rul",
    "mamba_xlstm_net",
    "mamba_rul",
    "patch_tst_rul",
    "vanilla_xlstm_rul",
    "xlstm_transformer",
    "liquid_wave_rul",
    "nbeats_xlstm_rul",
    "physics_nbeats_rul",
    "nbeats_rul",
    "diffusion_rul",
)


def _display_dataset_label(dataset: str | None) -> str:
    if not dataset:
        return "?"
    d = dataset.lower().replace("-", "_")
    if d in {"phm2012"}:
        return "PHM2012"
    if d in {"xjtusy", "xjtu_sy"}:
        return "XJTU-SY"
    # run_algorithm_comparison dataset key → see xjtu_sy_available_full.yaml
    if d in {"xjtu_available", "xjtu_avail", "xjtusy_available_full"}:
        return "XJTU-SY (cond. 1–2)"
    return dataset


def _friendly_ablation_run_key(run_id_aggregate_key: str) -> str:
    """Pretty row label for aggregated metrics charts/tables."""
    prefix = "algorithm_comparison_"
    if not run_id_aggregate_key.startswith(prefix):
        return run_id_aggregate_key
    rest = run_id_aggregate_key[len(prefix) :]
    for mk in sorted(_AGGREGATE_MODEL_KEYS, key=len, reverse=True):
        suf = "_" + mk
        if rest.endswith(suf):
            ds = rest[: -len(suf)]
            lbl = _display_model_label(mk, run_id_aggregate_key)
            return f"{_display_dataset_label(ds)} \u00b7 {lbl}"
    return run_id_aggregate_key


def _remap_ablation_aggregate_for_display(
    agg: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Map raw aggregate keys (run_id prefixes) to human-readable comparison labels."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for raw_key, payload in agg.items():
        fk = _friendly_ablation_run_key(raw_key)
        if fk in out:
            fk = raw_key  # collision (unexpected): fall back to raw id
        out[fk] = payload
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _side_by_side_comparison_labels(left: RunContext, right: RunContext) -> tuple[str, str]:
    ll = _display_model_label(left.model_name, left.run_id)
    rl = _display_model_label(right.model_name, right.run_id)
    dsl = (left.dataset or "").strip().lower().replace("-", "_")
    dsr = (right.dataset or "").strip().lower().replace("-", "_")
    show_ds = dsl and dsr and (dsl != dsr or ll == rl)
    if show_ds:
        return (
            f"{_display_dataset_label(left.dataset)} — {ll}",
            f"{_display_dataset_label(right.dataset)} — {rl}",
        )
    return ll, rl


def _display_model_label(model_name: str | None, run_id: str) -> str:
    if not model_name:
        return run_id
    m = model_name.lower()
    if "liquid_wave_rul" in m or "liquidwave_rul" in m:
        return "LiquidWave-RUL"
    if "nbeats_xlstm_rul" in m:
        return "N-BEATS-xLSTM-RUL"
    if "physics_nbeats_rul" in m:
        return "Physics-N-BEATS-RUL"
    if "nbeats_rul" in m or "n_beats_rul" in m:
        return "N-BEATS-RUL"
    if "diffusion_rul" in m:
        return "Diffusion-RUL"
    if "sparse_gate_tcn" in m or "sparsegatetcn" in m:
        return "SparseGate-TCN-RUL"
    if "phase_moe" in m or "phasemoexlstm" in m:
        return "PhaseMoE-xLSTM-RUL"
    if "patch_tst_rul" in m or "patchtst" in m:
        return "PatchTST-RUL"
    if "mamba_xlstm_net" in m:
        return "Mamba-xLSTM-Net"
    if "mamba_rul" in m:
        return "Mamba-RUL (SSM-only)"
    if "mamba" in m:
        return "Mamba-xLSTM-Net"
    if "vanilla_xlstm_rul" in m:
        return "Vanilla xLSTM-RUL"
    if "xlstm_transformer" in m or "baseline" in m:
        return "Baseline (xLSTM–Transformer)"
    return model_name


def _model_sort_rank(model_name: str | None) -> int:
    m = (model_name or "").lower()
    if "baseline" in m or "xlstm_transformer" in m:
        return 0
    if "vanilla_xlstm_rul" in m:
        return 1
    if "mamba_xlstm_net" in m:
        return 2
    if "mamba_rul" in m:
        return 3
    if "patch_tst" in m or "patchtst" in m:
        return 4
    if "liquid_wave_rul" in m or "liquidwave_rul" in m:
        return 5
    if "nbeats_xlstm_rul" in m:
        return 6
    if "physics_nbeats_rul" in m:
        return 7
    if "nbeats_rul" in m or "n_beats_rul" in m:
        return 8
    if "diffusion_rul" in m:
        return 9
    if "phase_moe" in m or "phasemoexlstm" in m:
        return 10
    if "sparse_gate_tcn" in m or "sparsegatetcn" in m:
        return 11
    if "mamba" in m:
        return 2
    return 99


def _metric_lower_is_better(key: str) -> bool:
    k = key.lower()
    if "phm_score_paper" in k:
        return True  # Liu et al. convention: lower is better
    if "phm_score" in k:
        return False  # project convention: higher is better
    if k.endswith("r2") or "/r2" in k:
        return False
    if "loss" in k or "rmse" in k or "mae" in k:
        return True
    return True


def _pick_winner(left: Any, right: Any, key: str) -> str:
    """Return 'left', 'right', 'tie', or '—' for non-numeric."""
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return "—"
    if left != left or right != right:  # NaN
        return "—"
    if abs(left - right) < 1e-12:
        return "tie"
    lower_better = _metric_lower_is_better(key)
    if lower_better:
        return "left" if left < right else "right"
    return "left" if left > right else "right"


def _order_pair_for_comparison(a: RunContext, b: RunContext) -> tuple[RunContext, RunContext]:
    """Put recognised model families in a stable dissertation-comparison order."""
    # Fallback: stable sort by model_name then run_id
    key_a = (_model_sort_rank(a.model_name), a.model_name or "", a.run_id)
    key_b = (_model_sort_rank(b.model_name), b.model_name or "", b.run_id)
    if key_a <= key_b:
        return a, b
    return b, a


def _merge_by_field(
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    """Align two lists of dicts that share ``field`` (e.g. bearing_id, key)."""
    lm = {x[field]: x for x in left_items if field in x}
    rm = {x[field]: x for x in right_items if field in x}
    keys = sorted(set(lm) | set(rm), key=lambda k: str(k))
    out: list[dict[str, Any]] = []
    for k in keys:
        out.append({
            field: k,
            "left": lm.get(k, {}).get("path", ""),
            "right": rm.get(k, {}).get("path", ""),
        })
    return out


def _build_pair_comparison(left: RunContext, right: RunContext) -> dict[str, Any]:
    """Side-by-side comparison payload for exactly two runs (figures + metrics)."""
    ll, rl = _side_by_side_comparison_labels(left, right)

    def _fmt_metric(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, (int, float)):
            if v != v:  # NaN
                return "—"
            return f"{float(v):.6f}"
        return str(v)

    metrics_keys = sorted(set(left.test_metrics) | set(right.test_metrics))
    metrics_rows: list[dict[str, Any]] = []
    for mk in metrics_keys:
        lv = left.test_metrics.get(mk)
        rv = right.test_metrics.get(mk)
        if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
            winner = _pick_winner(lv, rv, mk)
        else:
            winner = "—"
        metrics_rows.append({
            "key": mk,
            "left": lv,
            "right": rv,
            "winner": winner,
            "left_str": _fmt_metric(lv),
            "right_str": _fmt_metric(rv),
        })

    def _pair_img(key: str) -> dict[str, str]:
        la = left.figures.get(key) or ""
        ra = right.figures.get(key) or ""
        return {"left": la, "right": ra} if (la or ra) else {}

    rul_l = left.figures.get("rul_labels") or []
    rul_r = right.figures.get("rul_labels") or []
    rul_pairs = _merge_by_field(
        list(rul_l) if isinstance(rul_l, list) else [],
        list(rul_r) if isinstance(rul_r, list) else [],
        "bearing_id",
    )

    train_l = left.figures.get("training_curves") or []
    train_r = right.figures.get("training_curves") or []
    train_pairs = _merge_by_field(
        list(train_l) if isinstance(train_l, list) else [],
        list(train_r) if isinstance(train_r, list) else [],
        "key",
    )

    pred_l = left.figures.get("predictions") or []
    pred_r = right.figures.get("predictions") or []
    pred_pairs = _merge_by_field(
        list(pred_l) if isinstance(pred_l, list) else [],
        list(pred_r) if isinstance(pred_r, list) else [],
        "bearing_id",
    )

    res_l = left.figures.get("residuals_per_bearing") or []
    res_r = right.figures.get("residuals_per_bearing") or []
    res_pairs = _merge_by_field(
        list(res_l) if isinstance(res_l, list) else [],
        list(res_r) if isinstance(res_r, list) else [],
        "bearing_id",
    )

    interp_l = left.figures.get("interp") or {}
    interp_r = right.figures.get("interp") or {}
    interp_keys = sorted(set(interp_l) | set(interp_r))

    same_ds = (
        left.dataset is not None
        and right.dataset is not None
        and left.dataset.strip().lower() == right.dataset.strip().lower()
    )
    return {
        "same_dataset": same_ds,
        "left_label": ll,
        "right_label": rl,
        "left_run_id": left.run_id,
        "right_run_id": right.run_id,
        "left_run_dir": left.run_dir,
        "right_run_dir": right.run_dir,
        "left_model": left.model_name,
        "right_model": right.model_name,
        "interp_keys": interp_keys,
        "interp_left": interp_l,
        "interp_right": interp_r,
        "meta": {
            "seed": (left.seed, right.seed),
            "n_params": (left.n_params, right.n_params),
            "fit_seconds": (left.fit_seconds, right.fit_seconds),
            "dataset": (left.dataset, right.dataset),
        },
        "metrics_rows": metrics_rows,
        "dataset": _pair_img("dataset"),
        "hi_trace": _pair_img("hi_trace"),
        "hi_heatmap": _pair_img("hi_heatmap"),
        "rul_labels": rul_pairs,
        "training_curves": train_pairs,
        "predictions": pred_pairs,
        "residuals_per_bearing": res_pairs,
        "residuals_overall": _pair_img("residuals_overall"),
        "residuals_histogram": _pair_img("residuals_histogram"),
        "timings": _pair_img("timings"),
        "left_config": left.config_text,
        "right_config": right.config_text,
    }


def _load_test_predictions(path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not path.exists():
        return {}

    per_bearing: dict[str, dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as payload:
        for key in payload.files:
            if not key.endswith("_t"):
                continue
            bid = key[:-2]
            pred_key = f"{bid}_pred"
            y_key = f"{bid}_y"
            if pred_key not in payload.files or y_key not in payload.files:
                continue
            per_bearing[bid] = {
                "t": np.asarray(payload[key]),
                "pred": np.asarray(payload[pred_key]),
                "y": np.asarray(payload[y_key]),
            }
    return per_bearing


def _build_prediction_figures(
    per_bearing: dict[str, dict[str, np.ndarray]],
    figures_dir: Path,
    *,
    gallery_limit: int | None,
) -> tuple[dict[str, Any], int]:
    if not per_bearing:
        return {}, 0

    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    try:
        prediction_paths = plot_per_bearing_predictions(per_bearing, figures_dir)
        residual_paths = plot_per_bearing_residuals(per_bearing, figures_dir)
        overall_path = plot_residuals_overall(per_bearing, figures_dir / "residuals_overall.png")
        histogram_path = plot_residual_histogram(
            per_bearing,
            figures_dir / "residuals_histogram.png",
        )
    except Exception:
        return {}, generated

    generated += len(prediction_paths) + len(residual_paths) + 2
    predictions, predictions_omitted = _list_pred_figures(figures_dir, limit=gallery_limit)
    residuals, residuals_omitted = _list_residual_figures(figures_dir, limit=gallery_limit)

    return {
        "predictions": predictions,
        "predictions_omitted": predictions_omitted,
        "residuals_per_bearing": residuals,
        "residuals_omitted": residuals_omitted,
        "residuals_overall": _embed_image(overall_path),
        "residuals_histogram": _embed_image(histogram_path),
    }, generated


def _build_run_context(
    run_dir: Path,
    report_figures_dir: Path,
    *,
    gallery_limit: int | None,
) -> RunContext | None:
    summary = _read_json(run_dir / "summary.json") or {}
    if not summary:
        return None

    source_figures_dir = run_dir / "figures"
    generated_figures_dir = report_figures_dir / _slug(summary.get("run_id", run_dir.name))
    interp_dir = run_dir / "interp"
    explain_dir = run_dir / "explain"
    log_summary = _read_json(run_dir / "logs" / "summary.json") or {}
    n_generated_figures = 0

    timings_fig = None
    if log_summary.get("timings"):
        timings_path = generated_figures_dir / "step_timings.png"
        try:
            plot_step_timings(log_summary["timings"], timings_path)
            timings_fig = _embed_image(timings_path)
            n_generated_figures += 1
        except Exception:
            timings_fig = None

    prediction_figures, prediction_count = _build_prediction_figures(
        _load_test_predictions(run_dir / "test_predictions.npz"),
        generated_figures_dir,
        gallery_limit=gallery_limit,
    )
    n_generated_figures += prediction_count
    if not prediction_figures:
        predictions, predictions_omitted = _list_pred_figures(
            source_figures_dir,
            limit=gallery_limit,
        )
        residuals, residuals_omitted = _list_residual_figures(
            source_figures_dir,
            limit=gallery_limit,
        )
        prediction_figures = {
            "predictions": predictions,
            "predictions_omitted": predictions_omitted,
            "residuals_per_bearing": residuals,
            "residuals_omitted": residuals_omitted,
            "residuals_overall": _embed_image(source_figures_dir / "residuals_overall.png"),
            "residuals_histogram": _embed_image(
                source_figures_dir / "residuals_histogram.png",
            ),
        }

    rul_labels, rul_labels_omitted = _list_rul_label_figures(
        source_figures_dir,
        limit=gallery_limit,
    )

    figures: dict[str, Any] = {
        "dataset": _embed_image(source_figures_dir / "dataset_overview.png"),
        "hi_trace": _embed_image(source_figures_dir / "hi_trace.png"),
        "hi_heatmap": _embed_image(source_figures_dir / "hi_heatmap.png"),
        "rul_labels": rul_labels,
        "rul_labels_omitted": rul_labels_omitted,
        "training_curves": _list_training_curve_figures(source_figures_dir),
        "interp": (
            {**_list_interp_figures(interp_dir), **_list_interp_figures(explain_dir, key_prefix="[explainability] ")}
        ),
        "timings": timings_fig,
        **prediction_figures,
    }
    figures = {k: v for k, v in figures.items() if v or str(k).endswith("_omitted")}

    events = _select_events(run_dir / "logs" / "events.jsonl")
    config_path = run_dir / "config.yaml"
    config_text = config_path.read_text() if config_path.exists() else None

    dataset = summary.get("dataset")
    model_name = summary.get("model_name")
    if dataset is None and config_text:
        m = re.search(r"^\s*dataset:\s*(\S+)", config_text, re.MULTILINE)
        if m:
            dataset = m.group(1)
    if model_name is None and config_text:
        m = re.search(r"^\s*name:\s*(\S+)", config_text, re.MULTILINE)
        if m:
            model_name = m.group(1)

    return RunContext(
        run_id=summary.get("run_id", run_dir.name),
        run_dir=str(run_dir),
        seed=summary.get("seed"),
        n_params=summary.get("n_params"),
        fit_seconds=summary.get("fit_seconds"),
        dataset=dataset,
        model_name=model_name,
        test_metrics=summary.get("test_metrics", {}),
        figures=figures,
        events=events,
        config_text=config_text,
        n_generated_figures=n_generated_figures,
    )


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------


def _abbr_ablation_header(label: str) -> str:
    """Short column titles so wide scoreboards fit A4 without header overlap."""
    key = label.strip()
    mapping = {
        "run": "Run",
        "test/rmse": "RMSE",
        "test/mae": "MAE",
        "test/r2": "R2",
        "test/loss": "Loss",
        "test/phm_score": "PHM",
        "test/phm_score_paper": "PHMpap",
        "test/rmse_per_bearing": "RMSE/b",
        "test/phm_per_bearing": "PHM/b",
    }
    if key in mapping:
        return mapping[key]
    if key.startswith("test/"):
        tail = key[5:].replace("_", " ")
        return tail[:10] + "…" if len(tail) > 10 else tail
    return key[:12] + "…" if len(key) > 12 else key


def _format_ablation_table_cell(cell: str, *, is_first_col: bool) -> str:
    """Format ablation metric cells: mean only (± std omitted in report for readability)."""
    if is_first_col:
        return html.escape(cell)
    if cell == "—" or cell == "-":
        return html.escape(cell)
    if " ± " in cell:
        mean, _, rest = cell.partition(" ± ")
        std = rest.strip()
        if mean and std:
            return f'<span class="cell-mean">{html.escape(mean)}</span>'
    return html.escape(cell)


def _markdown_to_html_table(md: str, *, ablation_style: bool = False) -> str:
    """Lightweight pipe-table → HTML converter (good enough for our summary tables)."""
    lines = [l.strip() for l in md.strip().splitlines() if l.strip()]
    rows = [l.strip("|").split("|") for l in lines if l.startswith("|")]
    rows = [[c.strip() for c in r] for r in rows]
    if len(rows) < 2:
        return f"<pre>{html.escape(md)}</pre>"
    header, sep, *body = rows
    tbl_cls = "metrics ablation-scoreboard" if ablation_style else "metrics"
    if ablation_style:
        out = [
            f"<table class='{tbl_cls}'><colgroup>",
            '<col class="run-col"/>',
            *(['<col class="metric-col"/>'] * max(0, len(header) - 1)),
            "</colgroup><thead><tr>",
        ]
    else:
        out = [f"<table class='{tbl_cls}'><thead><tr>"]
    if ablation_style:
        out += [
            f'<th scope="col" title="{html.escape(c)}">{html.escape(_abbr_ablation_header(c))}</th>'
            for c in header
        ]
    else:
        out += [f"<th>{html.escape(c)}</th>" for c in header]
    out.append("</tr></thead><tbody>")
    for r in body:
        cols = len(header)
        r = list(r[:cols]) + [""] * max(0, cols - len(r))
        out.append("<tr>")
        for i, raw in enumerate(r):
            if i == 0 and ablation_style:
                td_cls = ' class="run-cell"'
            elif ablation_style:
                td_cls = ' class="ablation-metric-cell"'
            else:
                td_cls = ""
            inner = (
                _format_ablation_table_cell(raw, is_first_col=(i == 0))
                if ablation_style
                else html.escape(raw)
            )
            out.append(f"<td{td_cls}>{inner}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _render_to_pdf(html_text: str, html_path: Path, pdf_path: Path) -> str:
    """Try WeasyPrint first, then xhtml2pdf. Return the backend name used."""
    try:
        from weasyprint import HTML  # type: ignore

        HTML(string=html_text, base_url=str(html_path.parent)).write_pdf(str(pdf_path))
        return "weasyprint"
    except Exception as e_weasy:  # pragma: no cover - depends on env
        try:
            from xhtml2pdf import pisa  # type: ignore

            with pdf_path.open("wb") as f:
                result = pisa.CreatePDF(html_text, dest=f)
            if getattr(result, "err", 1):
                raise RuntimeError("xhtml2pdf reported errors")
            return "xhtml2pdf"
        except Exception as e_x:
            raise RuntimeError(
                "PDF export failed. Install WeasyPrint (recommended: `brew install pango "
                "&& pip install weasyprint`) or `pip install xhtml2pdf`. "
                f"WeasyPrint: {e_weasy}. xhtml2pdf: {e_x}"
            ) from e_x


def _report_figures_dir(out_html: Path) -> Path:
    if out_html.name == "report.html":
        return out_html.parent / "figures"
    return out_html.parent / out_html.stem / "figures"


def _metric_keys_for_report(summaries: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "test/rmse",
        "test/mae",
        "test/r2",
        "test/phm_score",
        "test/phm_score_paper",
        "test/rmse_per_bearing",
        "test/phm_per_bearing",
    ]
    seen = {
        key
        for summary in summaries
        for key in (summary.get("test_metrics") or {})
    }
    ordered = [key for key in preferred if key in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_report(
    run_dirs: list[str | Path],
    out_html: str | Path,
    out_pdf: str | Path | None = None,
    *,
    title: str = "Mamba-xLSTM dissertation report",
    subtitle: str = "Bearing remaining-useful-life prediction",
    subtitle_lines: list[str] | None = None,
    include_ablation: bool = True,
    side_by_side: bool | None = None,
    gallery_limit: int | None = 12,
) -> dict[str, Any]:
    """Render an HTML report (and optionally a PDF) summarising the given runs.

    When exactly two runs are passed and ``side_by_side`` is not ``False``, the
    report uses a **side-by-side comparison layout** (metrics + paired figures)
    instead of repeating full per-run sections.

    ``side_by_side`` defaults to ``True`` for two runs and ``False`` otherwise.

    ``subtitle_lines`` (optional) renders the cover subtitle as a bulleted list
    below the one-line ``subtitle`` tagline—use for algorithm names in
    multi-model comparison reports.

    Returns a dict with the produced paths (and the PDF backend if used).
    """
    run_dirs = [Path(p) for p in run_dirs]
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = _report_figures_dir(out_html)
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        c
        for c in (
            _build_run_context(d, figures_dir, gallery_limit=gallery_limit)
            for d in run_dirs
        )
        if c
    ]
    datasets = sorted({c.dataset for c in runs if c.dataset})
    n_generated_figures = sum(c.n_generated_figures for c in runs)

    use_sbs = side_by_side
    if use_sbs is None:
        use_sbs = len(runs) == 2
    comparison: dict[str, Any] | None = None
    if use_sbs and len(runs) == 2:
        left, right = _order_pair_for_comparison(runs[0], runs[1])
        comparison = _build_pair_comparison(left, right)
        runs = [left, right]

    ablation_section = None
    if include_ablation and len(run_dirs) > 1:
        all_summaries = []
        for d in run_dirs:
            s = _read_json(d / "summary.json") or {}
            if s:
                s["run_dir"] = str(d)
                all_summaries.append(s)
        if all_summaries:
            agg_raw = aggregate_by_run_id_prefix(all_summaries)
            agg_disp = _remap_ablation_aggregate_for_display(agg_raw)
            metrics = _metric_keys_for_report(all_summaries)
            ablation_section = {"n_total_runs": len(all_summaries)}
            ablation_dir = figures_dir / "ablation"
            ablation_dir.mkdir(parents=True, exist_ok=True)
            try:
                paths = plot_ablation_per_metric(agg_disp, ablation_dir, metric_keys=metrics)
                n_generated_figures += len(paths)
                ablation_section["figures"] = [
                    {"key": p.stem.replace("ablation_", "").replace("_", "/"),
                     "path": _embed_image(p)}
                    for p in paths
                ]
            except Exception:
                ablation_section["figures"] = []
            ablation_section["table_html"] = _markdown_to_html_table(
                to_markdown(agg_disp, metrics),
                ablation_style=True,
            )

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)),
                      autoescape=select_autoescape(["html"]))
    template = env.get_template("report.html")
    html_text = template.render(
        title=title,
        subtitle=subtitle,
        subtitle_lines=subtitle_lines,
        generated_at=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        runs=runs,
        datasets=datasets,
        generated_figures=n_generated_figures,
        ablation_section=ablation_section,
        comparison=comparison,
    )
    out_html.write_text(html_text)

    result = {
        "html": str(out_html),
        "figures_dir": str(figures_dir),
        "n_figures": n_generated_figures,
    }
    if out_pdf is not None:
        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        try:
            backend = _render_to_pdf(html_text, out_html, out_pdf)
            result["pdf"] = str(out_pdf)
            result["pdf_backend"] = backend
        except RuntimeError as exc:
            result["pdf"] = ""
            result["pdf_error"] = str(exc)
    return result
