"""Aggregate per-run summary.json files into markdown + LaTeX tables."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect_runs(runs_root: str | Path) -> list[dict]:
    out = []
    for d in sorted(Path(runs_root).glob("*")):
        s = d / "summary.json"
        if s.exists():
            try:
                payload = json.loads(s.read_text())
                payload["run_dir"] = str(d)
                out.append(payload)
            except json.JSONDecodeError:
                pass
    return out


def aggregate_by_run_id_prefix(runs: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Group by stripping ``_s\\d+`` from the run_id; return mean ± std per metric."""
    import re

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        key = re.sub(r"_s\d+$", "", r["run_id"])
        groups[key].append(r)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for key, items in groups.items():
        metric_keys = set()
        for it in items:
            metric_keys.update(it.get("test_metrics", {}).keys())
        agg: dict[str, dict[str, float]] = {}
        for mk in sorted(metric_keys):
            vals = np.asarray([it["test_metrics"].get(mk, np.nan) for it in items], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            agg[mk] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1) if vals.size > 1 else 0.0),
                "n": int(vals.size),
            }
        out[key] = agg
    return out


def to_markdown(agg: dict[str, dict[str, dict[str, float]]], metrics: list[str]) -> str:
    headers = ["run"] + metrics
    rows = []
    for key in sorted(agg):
        row = [key]
        for m in metrics:
            cell = agg[key].get(m)
            row.append(f"{cell['mean']:.4f} ± {cell['std']:.4f}" if cell else "—")
        rows.append(row)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def to_latex(agg: dict[str, dict[str, dict[str, float]]], metrics: list[str]) -> str:
    headers = ["run"] + metrics
    lines = [
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for key in sorted(agg):
        cells = [key.replace("_", r"\_")]
        for m in metrics:
            cell = agg[key].get(m)
            cells.append(f"{cell['mean']:.4f} $\\pm$ {cell['std']:.4f}" if cell else "—")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def write_summary(
    runs_root: str | Path,
    out_dir: str | Path,
    metrics: list[str] | None = None,
) -> None:
    metrics = metrics or ["test/rmse", "test/mae", "test/r2", "test/phm_score", "test/rmse_per_bearing", "test/phm_per_bearing"]
    runs = collect_runs(runs_root)
    agg = aggregate_by_run_id_prefix(runs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(to_markdown(agg, metrics))
    (out_dir / "summary.tex").write_text(to_latex(agg, metrics))
    (out_dir / "summary.json").write_text(json.dumps(agg, indent=2))
