"""Aggregate negative-control results across seeds 42/43/44.

Reads per-seed JSON files produced by run_negative_controls.py and computes
mean ± std (population std) for each hit-rate value.  Output is written to
  results/journal_q2/negative_controls/<dataset>_<control>_agg_results.json

Usage::

    cd Mamba-xLSTM && source .venv/bin/activate
    python -m scripts.journal_q2.aggregate_negctrl_seeds
    python -m scripts.journal_q2.aggregate_negctrl_seeds --seeds 42 43 44
    python -m scripts.journal_q2.aggregate_negctrl_seeds --datasets phm2012 xjtusy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.journal_q2._helpers import OUT_ROOT

NEG_DIR = OUT_ROOT / "negative_controls"
DATASETS  = ["phm2012", "xjtusy", "ims", "cwru"]
CONTROLS  = ["untrained_backbone", "gaussian_noise"]
SEEDS     = [42, 43, 44]


def _per_seed_json_path(dataset: str, control: str, backbone_seed: int) -> Path:
    """Path to per-seed JSON for a *backbone* RNG seed (42/43/44).

    ``run_negative_controls.py`` tags untrained files with ``_s{seed}`` but passes
    ``rng_seed=seed+1`` into the Gaussian-noise control so filenames use ``_s{seed+1}``.
    """
    if control == "gaussian_noise":
        file_seed = backbone_seed + 1
    else:
        file_seed = backbone_seed
    tag = "" if file_seed == 0 else f"_s{file_seed}"
    return NEG_DIR / f"{dataset}_{control}{tag}_results.json"


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",    nargs="+", type=int, default=SEEDS)
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--controls", nargs="+", default=CONTROLS, choices=CONTROLS)
    return p.parse_args()


def aggregate(dataset: str, control: str, seeds: list[int]) -> dict | None:
    """Load per-seed JSONs and return aggregated payload, or None if no seeds found."""
    hit_rates: list[list[float]] = []
    mses: list[float] = []
    meta: dict = {}
    backbone_seeds_used: list[int] = []

    for seed in seeds:
        fpath = _per_seed_json_path(dataset, control, seed)
        if not fpath.exists():
            print(f"  [SKIP] not found: {fpath.name}")
            continue
        d = json.loads(fpath.read_text())
        hit_rates.append(d["hit_rate"])
        mses.append(d.get("sae_recon_mse", float("nan")))
        meta = d   # keep last for metadata fields
        backbone_seeds_used.append(seed)

    if not hit_rates:
        return None

    arr = np.array(hit_rates, dtype=float)  # (n_seeds, 4)
    mean_hr = arr.mean(axis=0).tolist()
    std_hr  = arr.std(axis=0).tolist()

    return {
        "dataset":      dataset,
        "control":      control,
        "seeds":        backbone_seeds_used,
        "n_seeds":      len(hit_rates),
        "threshold":    meta.get("threshold", 0.30),
        "epochs":       meta.get("epochs"),
        "n_hidden":     meta.get("n_hidden"),
        "d_model":      meta.get("d_model"),
        "bpfx_names":   meta.get("bpfx_names", ["BPFO", "BPFI", "BSF", "FTF"]),
        "freqs_hz":     meta.get("freqs_hz", {}),
        "hit_rate_mean":  mean_hr,
        "hit_rate_std":   std_hr,
        "hit_rate_mean_pct": [round(v * 100, 2) for v in mean_hr],
        "hit_rate_std_pct":  [round(v * 100, 2) for v in std_hr],
        "sae_recon_mse_mean": float(np.nanmean(mses)),
    }


def main() -> None:
    args = _parse()
    NEG_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for ds in args.datasets:
        summary[ds] = {}
        for ctrl in args.controls:
            print(f"\n[{ds} / {ctrl}]")
            result = aggregate(ds, ctrl, args.seeds)
            if result is None:
                print("  no seed files found — skip")
                continue

            bpfx = result["bpfx_names"]
            for i, name in enumerate(bpfx):
                mean_pct = result["hit_rate_mean_pct"][i]
                std_pct  = result["hit_rate_std_pct"][i]
                print(f"  {name}: {mean_pct:.2f} ± {std_pct:.2f} %")

            out = NEG_DIR / f"{ds}_{ctrl}_agg_results.json"
            out.write_text(json.dumps(result, indent=2))
            print(f"  → {out.name}")
            summary[ds][ctrl] = result

    agg_summary = NEG_DIR / "agg_summary.json"
    agg_summary.write_text(json.dumps(summary, indent=2))
    print(f"\nAggregated summary: {agg_summary}")


if __name__ == "__main__":
    main()
