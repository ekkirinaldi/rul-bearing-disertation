"""Bootstrap CI + permutation test + per-bearing breakdown for the existing
trained Mamba-xLSTM SAE on PHM2012 and XJTU-SY.

This is the statistical-inference companion of ``scripts/run_bpfx_mapping.py``.
It does NOT retrain anything; it only loads the existing trained SAE and
hidden states, and produces:

  - bootstrap 95% CI on the per-BPFx hit-rate
  - one-sided (greater) permutation p-value for the same hit-rate
  - per-individual-bearing hit-rate breakdown
  - bar plot with error bars per BPFx (one figure per dataset)

Outputs:
  Mamba-xLSTM/results/journal_q2/stats/<dataset>_stats.json
  Mamba-xLSTM/results/journal_q2/stats/<dataset>_hitrate_ci.png
  Mamba-xLSTM/results/journal_q2/stats/<dataset>_perbearing.png

Usage::

    cd Mamba-xLSTM && source .venv/bin/activate
    python -m scripts.journal_q2.run_stats           # both datasets, default settings
    python -m scripts.journal_q2.run_stats --datasets phm2012
    python -m scripts.journal_q2.run_stats --n-boot 2000 --n-perm 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.journal_q2._helpers import (
    DATASET_INFO,
    OUT_ROOT,
    bootstrap_hit_rate_ci,
    collect_hidden_for_dataset,
    encode_with_sae,
    gather_bpfx_amplitudes,
    hit_rate_from_corrs,
    align_and_correlate,
    load_default_sae,
    per_bearing_hit_rate,
    permutation_pvalues,
    write_json,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_INFO.keys()),
                   choices=list(DATASET_INFO.keys()))
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--n-boot", type=int, default=1_000)
    p.add_argument("--n-perm", type=int, default=1_000)
    p.add_argument("--max-recordings", type=int, default=300)
    p.add_argument("--n-hidden", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_one(
    dataset_key: str,
    *,
    threshold: float,
    n_boot: int,
    n_perm: int,
    max_recordings: int,
    n_hidden: int,
    rng: np.random.Generator,
) -> dict:
    print(f"\n{'='*70}\n  STATS — {dataset_key}\n{'='*70}")

    # 1. Load SAE + hidden states + BPFx amplitudes.
    sae = load_default_sae(dataset_key)
    print(f"  SAE: d_model={sae.cfg.d_model}  d_latent={sae.d_latent}  k={sae.k}")

    hidden = collect_hidden_for_dataset(dataset_key, n_max=n_hidden)
    print(f"  Hidden states: {hidden.shape}")

    z_sparse = encode_with_sae(sae, hidden)
    print(f"  Sparse activations: {z_sparse.shape}")

    amps, bearing_idx, bearing_names, freqs_hz = gather_bpfx_amplitudes(
        dataset_key, max_recordings=max_recordings
    )
    print(f"  BPFx amplitudes: {amps.shape}  ({len(bearing_names)} bearings)")
    bpfx_names = list(freqs_hz.keys())

    # 2. Point estimate
    corrs = align_and_correlate(z_sparse, amps)
    point_hr = hit_rate_from_corrs(corrs, threshold)

    # 3. Bootstrap CI
    print(f"  Running bootstrap with B={n_boot} …")
    boot = bootstrap_hit_rate_ci(
        z_sparse, amps, threshold=threshold, n_boot=n_boot, rng=rng
    )

    # 4. Permutation test
    print(f"  Running permutation with B={n_perm} …")
    perm = permutation_pvalues(
        z_sparse, amps, threshold=threshold, n_perm=n_perm, rng=rng
    )

    # 5. Per-bearing breakdown
    perb = per_bearing_hit_rate(
        z_sparse, amps, bearing_idx, bearing_names, threshold=threshold
    )

    # 6. Pretty print summary
    print(f"\n  {'BPFx':6s}  {'point':>6s}  {'95% CI':>18s}  {'p (perm)':>10s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*18}  {'-'*10}")
    for ci, name in enumerate(bpfx_names):
        print(
            f"  {name:6s}  {point_hr[ci]*100:5.2f}%  "
            f"[{boot['low'][ci]*100:5.2f}, {boot['high'][ci]*100:5.2f}]%  "
            f"{perm['p_value'][ci]:10.4f}"
        )

    # 7. Per-bearing
    print(f"\n  Per-bearing hit-rate (%):")
    print(f"    {'bearing':16s}  " + "  ".join(f"{n:>6s}" for n in bpfx_names))
    for bname, hr in perb.items():
        cells = "  ".join(f"{x*100:5.2f}%" if not np.isnan(x) else "   ---" for x in hr)
        print(f"    {bname:16s}  {cells}")

    # 8. Save JSON
    payload = {
        "dataset": dataset_key,
        "threshold": threshold,
        "n_boot": n_boot,
        "n_perm": n_perm,
        "n_recordings": int(amps.shape[0]),
        "n_bearings": len(bearing_names),
        "bearing_names": bearing_names,
        "bpfx_names": bpfx_names,
        "freqs_hz": freqs_hz,
        "point_hit_rate": point_hr,
        "bootstrap_low": boot["low"],
        "bootstrap_high": boot["high"],
        "permutation_pvalue": perm["p_value"],
        "per_bearing_hit_rate": {b: hr.tolist() for b, hr in perb.items()},
    }
    out_dir = OUT_ROOT / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{dataset_key}_stats.json", payload)
    print(f"\n  JSON saved → {out_dir / f'{dataset_key}_stats.json'}")

    # 9. CI bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(bpfx_names))
    yerr_low = (point_hr - boot["low"]) * 100
    yerr_high = (boot["high"] - point_hr) * 100
    ax.bar(x, point_hr * 100, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
           yerr=[yerr_low, yerr_high], capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bpfx_names)
    ax.set_ylabel("Hit-rate (%)")
    ax.set_title(
        f"{dataset_key.upper()} — Hit-rate with bootstrap 95% CI\n"
        f"(|r|≥{threshold}, B={n_boot}, N={amps.shape[0]} recordings)"
    )
    for ci, p in enumerate(perm["p_value"]):
        ax.text(x[ci], (point_hr[ci] + (boot["high"][ci] - point_hr[ci])) * 100 + 0.3,
                f"p={p:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset_key}_hitrate_ci.png", dpi=150)
    plt.close(fig)

    # 10. Per-bearing heatmap
    perb_arr = np.stack([perb[b] for b in bearing_names], axis=0) * 100
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(bearing_names))))
    im = ax.imshow(perb_arr, aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(np.arange(len(bpfx_names)))
    ax.set_xticklabels(bpfx_names)
    ax.set_yticks(np.arange(len(bearing_names)))
    ax.set_yticklabels(bearing_names)
    ax.set_title(f"{dataset_key.upper()} — Per-bearing hit-rate (%)")
    fig.colorbar(im, ax=ax, label="hit-rate %")
    for i in range(perb_arr.shape[0]):
        for j in range(perb_arr.shape[1]):
            v = perb_arr[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.1f}",
                    ha="center", va="center",
                    color="white" if (np.isnan(v) or v < perb_arr[~np.isnan(perb_arr)].max() / 2) else "black",
                    fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset_key}_perbearing.png", dpi=150)
    plt.close(fig)

    return payload


def main() -> None:
    args = _parse()
    rng = np.random.default_rng(args.seed)
    summary = {}
    for ds in args.datasets:
        if not DATASET_INFO[ds]["sae_pt"].exists():
            print(f"SKIP {ds}: SAE not found at {DATASET_INFO[ds]['sae_pt']}")
            continue
        summary[ds] = run_one(
            ds,
            threshold=args.threshold,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            max_recordings=args.max_recordings,
            n_hidden=args.n_hidden,
            rng=rng,
        )
    write_json(OUT_ROOT / "stats" / "summary.json", summary)
    print(f"\nSummary saved → {OUT_ROOT / 'stats' / 'summary.json'}")


if __name__ == "__main__":
    main()
