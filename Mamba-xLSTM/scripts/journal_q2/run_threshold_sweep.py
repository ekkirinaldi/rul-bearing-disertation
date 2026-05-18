"""Threshold sensitivity sweep: |r| ∈ {0.25, 0.30, 0.35} for PHM2012 and XJTU-SY.

Demonstrates that the dominant BPFx ordering is stable across threshold choices,
addressing reviewer concern about the arbitrary |r| ≥ 0.30 threshold.

Outputs:
  results/journal_q2/threshold_sweep/threshold_sweep_phm2012.json
  results/journal_q2/threshold_sweep/threshold_sweep_xjtusy.json
  results/journal_q2/threshold_sweep/threshold_sweep_phm2012.png
  results/journal_q2/threshold_sweep/threshold_sweep_xjtusy.png

Usage::

    cd Mamba-xLSTM && source .venv/bin/activate
    python -m scripts.journal_q2.run_threshold_sweep
    python -m scripts.journal_q2.run_threshold_sweep --datasets phm2012
    python -m scripts.journal_q2.run_threshold_sweep --thresholds 0.20 0.25 0.30 0.35 0.40
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.journal_q2._helpers import (
    DATASET_INFO,
    OUT_ROOT,
    align_and_correlate,
    collect_hidden_for_dataset,
    encode_with_sae,
    gather_bpfx_amplitudes,
    hit_rate_from_corrs,
    load_default_sae,
    write_json,
)

_DEFAULT_THRESHOLDS = [0.25, 0.30, 0.35]
_DEFAULT_DATASETS   = ["phm2012", "xjtusy"]


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--thresholds", nargs="+", type=float, default=_DEFAULT_THRESHOLDS)
    p.add_argument("--datasets",   nargs="+", default=_DEFAULT_DATASETS,
                   choices=list(DATASET_INFO.keys()))
    p.add_argument("--max-recordings", type=int, default=300)
    p.add_argument("--n-hidden",   type=int, default=5_000)
    return p.parse_args()


def run_sweep(
    dataset_key: str,
    thresholds: list[float],
    *,
    max_recordings: int,
    n_hidden: int,
) -> dict:
    """Run hit-rate at each threshold for one dataset.

    Returns a dict with structure::

        {
          "dataset": "phm2012",
          "bpfx_names": ["BPFO", "BPFI", "BSF", "FTF"],
          "thresholds": [0.25, 0.30, 0.35],
          "hit_rates": [[r_bpfo, r_bpfi, r_bsf, r_ftf], ...],  # one row per threshold
          "hit_rates_pct": ...,
        }
    """
    print(f"\n[{dataset_key}] threshold sweep: {thresholds}")

    sae    = load_default_sae(dataset_key)
    hidden = collect_hidden_for_dataset(dataset_key, n_max=n_hidden)
    z      = encode_with_sae(sae, hidden)

    amps, _, _, freqs_hz = gather_bpfx_amplitudes(
        dataset_key, max_recordings=max_recordings
    )
    bpfx_names = list(freqs_hz.keys())
    corrs = align_and_correlate(z, amps)

    rows:     list[list[float]] = []
    rows_pct: list[list[float]] = []
    for thr in thresholds:
        hr = hit_rate_from_corrs(corrs, thr)
        rows.append(hr.tolist())
        rows_pct.append([round(v * 100, 2) for v in hr.tolist()])
        pct_str = "  ".join(f"{n}={v*100:.2f}%" for n, v in zip(bpfx_names, hr))
        print(f"  thr={thr:.2f}: {pct_str}")

    return {
        "dataset":      dataset_key,
        "bpfx_names":   bpfx_names,
        "freqs_hz":     freqs_hz,
        "thresholds":   thresholds,
        "hit_rates":    rows,
        "hit_rates_pct": rows_pct,
    }


def _plot_sweep(result: dict, out_path) -> None:
    """Bar-cluster plot: one group per threshold, bars for BPFO/BPFI/BSF/FTF."""
    bpfx   = result["bpfx_names"]
    thrs   = result["thresholds"]
    rows   = np.array(result["hit_rates_pct"])   # (n_thr, 4)
    n_thr  = len(thrs)
    n_bpfx = len(bpfx)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    x = np.arange(n_bpfx)
    width = 0.8 / n_thr
    offsets = np.linspace(-(n_thr - 1) / 2, (n_thr - 1) / 2, n_thr) * width

    fig, ax = plt.subplots(figsize=(7, 4))
    for ti, (thr, offset) in enumerate(zip(thrs, offsets)):
        ax.bar(x + offset, rows[ti], width,
               label=f"|r|≥{thr:.2f}",
               color=[c + "BB" for c in colors],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bpfx)
    ax.set_ylabel("Hit-rate (%)")
    ax.set_title(f"{result['dataset'].upper()} — Threshold sensitivity sweep")
    ax.legend(title="threshold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse()
    out_dir = OUT_ROOT / "threshold_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        if not DATASET_INFO[ds]["sae_pt"].exists():
            print(f"SKIP {ds}: SAE not found at {DATASET_INFO[ds]['sae_pt']}")
            continue

        result = run_sweep(
            ds,
            args.thresholds,
            max_recordings=args.max_recordings,
            n_hidden=args.n_hidden,
        )

        out_json = out_dir / f"threshold_sweep_{ds}.json"
        write_json(out_json, result)
        print(f"  → {out_json.name}")

        out_png = out_dir / f"threshold_sweep_{ds}.png"
        _plot_sweep(result, out_png)
        print(f"  → {out_png.name}")

    print(f"\nThreshold sweep complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
