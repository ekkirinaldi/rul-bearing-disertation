"""Ad-hoc XJTU-SY vs PHM2012 diagnostic for the v2 comparison run.

Loads ``test_predictions.npz`` per run dir and prints:
- per-bearing RMSE / count
- micro RMSE and R²
- per-dataset label variance (so we can explain why XJTU R² is low)
- val/train window counts (from summary.json + run.log)
- best/last val/rmse trajectory
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/root/disertation-rul-prediction/Mamba-xLSTM/results/runs")
ORDER = [
    ("phm2012", "xlstm_transformer"),
    ("phm2012", "mamba_xlstm_net"),
    ("phm2012", "liquid_wave_rul"),
    ("phm2012", "nbeats_rul"),
    ("phm2012", "diffusion_rul"),
    ("xjtusy", "xlstm_transformer"),
    ("xjtusy", "mamba_xlstm_net"),
    ("xjtusy", "liquid_wave_rul"),
    ("xjtusy", "nbeats_rul"),
    ("xjtusy", "diffusion_rul"),
]


def find_run(ds: str, mk: str) -> Path | None:
    pattern = f"*algorithm_comparison_{ds}_{mk}_s42"
    cands = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def per_bearing(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())
    pairs = {}
    for k in keys:
        if k.endswith("_pred"):
            base = k[:-5]
            ykey = f"{base}_y"
            if ykey in d:
                pairs[base] = (d[ykey], d[k])
    if not pairs and "y" in d and "pred" in d:
        pairs["all"] = (d["y"], d["pred"])
    return pairs


def main() -> None:
    print("=== Per-bearing test RMSE ===")
    print(f"{'dataset':10s} {'model':22s} {'per-bearing':50s}  micro_rmse   r2     std(err)")
    rows = []
    for ds, mk in ORDER:
        p = find_run(ds, mk)
        if p is None:
            print(f"{ds:10s} {mk:22s} MISSING run dir")
            continue
        npz = p / "test_predictions.npz"
        if not npz.exists():
            print(f"{ds:10s} {mk:22s} MISSING test_predictions.npz")
            continue
        pairs = per_bearing(npz)
        parts, all_y, all_p = [], [], []
        for b, (y, pr) in sorted(pairs.items()):
            rmse = float(np.sqrt(np.mean((y - pr) ** 2)))
            parts.append(f"{b}={rmse:.3f}(n={len(y)})")
            all_y.append(y)
            all_p.append(pr)
        y = np.concatenate(all_y)
        pr = np.concatenate(all_p)
        err = y - pr
        micro = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append((ds, mk, parts, micro, r2, float(err.std())))
        print(f"{ds:10s} {mk:22s} {', '.join(parts):50s}  {micro:.4f}     {r2:+.3f}   {err.std():.4f}")

    print("\n=== Test label statistics (using xlstm_transformer run as the canonical (y) for each dataset) ===")
    for ds in ("phm2012", "xjtusy"):
        p = find_run(ds, "xlstm_transformer")
        pairs = per_bearing(p / "test_predictions.npz")
        ys = np.concatenate([y for y, _ in pairs.values()])
        per_b_var = []
        for b, (y, _) in pairs.items():
            per_b_var.append((b, float(y.var()), float(y.std()), len(y)))
        print(f"{ds:10s} n_total={len(ys):5d}  mean(y)={ys.mean():.3f}  std(y)={ys.std():.4f}  var(y)={ys.var():.5f}")
        for b, v, s, n in per_b_var:
            print(f"  bearing {b:6s}  n={n:5d}  std(y)={s:.4f}  var(y)={v:.5f}  range=[{min(0.0, ys.min()):.2f}, {1.0:.2f}]")

    print("\n=== Train / val / test window counts (per run) ===")
    for ds, mk in ORDER:
        p = find_run(ds, mk)
        if p is None:
            continue
        log = p / "logs" / "run.log"
        n_train = n_val = n_test = None
        if log.exists():
            txt = log.read_text(errors="ignore")
            m = re.search(r"train_windows=(\d+)\s+val_windows=(\d+)\s+test_windows=(\d+)", txt)
            if m:
                n_train, n_val, n_test = (int(x) for x in m.groups())
        print(f"{ds:10s} {mk:22s}  n_train={n_train}  n_val={n_val}  n_test={n_test}")

    print("\n=== val/rmse curve stability (from metrics.csv when present) ===")
    import csv
    import statistics

    print(f"{'dataset':10s} {'model':22s} {'n_val_pts':>9s} {'min':>10s} {'max':>10s} {'std':>10s} {'argmin':>7s}")
    for ds, mk in ORDER:
        p = find_run(ds, mk)
        if p is None:
            continue
        candidates = list(p.glob("logs/csv/**/metrics.csv"))
        if not candidates:
            candidates = list(p.glob("**/metrics.csv"))
        if not candidates:
            print(f"{ds:10s} {mk:22s} no metrics.csv")
            continue
        rows = list(csv.DictReader(open(candidates[0])))
        vals = []
        for r in rows:
            v = r.get("val/rmse")
            if v in (None, "", "NaN"):
                continue
            try:
                vals.append((int(r.get("epoch", 0) or 0), float(v)))
            except ValueError:
                continue
        if not vals:
            print(f"{ds:10s} {mk:22s} no val/rmse rows")
            continue
        ys = [v for _, v in vals]
        argmin_epoch = vals[ys.index(min(ys))][0]
        std = statistics.stdev(ys) if len(ys) > 1 else 0.0
        print(
            f"{ds:10s} {mk:22s} {len(ys):>9d} {min(ys):>10.4f} {max(ys):>10.4f} {std:>10.4f} {argmin_epoch:>7d}"
        )


if __name__ == "__main__":
    main()
