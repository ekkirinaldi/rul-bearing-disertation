"""Two negative controls for the SAE--BPFx correspondence claim:

  Control 1 (untrained backbone):
      Re-instantiate the Mamba-xLSTM-Net with random Xavier init,
      collect hidden states, train a fresh SAE on those, and run the
      BPFx mapping. A strong reduction relative to the trained baseline
      falsifies the hypothesis that the correspondence is an artefact of
      the architecture itself.

  Control 2 (Gaussian noise):
      Replace the hidden-state pool with same-shape Gaussian noise with
      matched per-dim mean and variance. Train a fresh SAE on this noise,
      and run the BPFx mapping. A strong reduction relative to both the
      trained backbone and Control 1 falsifies the hypothesis that the
      SAE construction itself induces the correspondence.

Outputs:
  Mamba-xLSTM/results/journal_q2/negative_controls/<dataset>_<control>_results.json
  Mamba-xLSTM/results/journal_q2/negative_controls/comparison.json
  Mamba-xLSTM/results/journal_q2/negative_controls/comparison_bar.png

Usage::

    cd Mamba-xLSTM && source .venv/bin/activate
    python -m scripts.journal_q2.run_negative_controls
    python -m scripts.journal_q2.run_negative_controls --datasets phm2012 --epochs 30
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mxlstm.compute import get_device
from mxlstm.interp.sae import SAEConfig, TopKSparseAutoencoder, train_sae

from scripts.journal_q2._helpers import (
    DATASET_INFO,
    OUT_ROOT,
    align_and_correlate,
    collect_hidden_for_dataset,
    encode_with_sae,
    gather_bpfx_amplitudes,
    hit_rate_from_corrs,
    write_json,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_INFO.keys()),
                   choices=list(DATASET_INFO.keys()))
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-recordings", type=int, default=300)
    p.add_argument("--n-hidden", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _train_fresh_sae(
    hidden: np.ndarray, *, epochs: int, device, rng_seed: int
) -> tuple[TopKSparseAutoencoder, float]:
    torch.manual_seed(rng_seed)
    cfg = SAEConfig(d_model=hidden.shape[1], expansion=8)
    sae = TopKSparseAutoencoder(cfg)
    history = train_sae(sae, hidden, epochs=epochs, batch_size=256, lr=1e-3, device=device)
    final_mse = float(history[-1]["recon"]) if history else float("nan")
    return sae, final_mse


def _run_one_control(
    dataset_key: str,
    control_name: str,
    hidden: np.ndarray,
    *,
    epochs: int,
    threshold: float,
    max_recordings: int,
    rng_seed: int,
    device,
) -> dict:
    print(f"\n  -- Control: {control_name} on {dataset_key}")
    sae, sae_recon_mse = _train_fresh_sae(hidden, epochs=epochs, device=device, rng_seed=rng_seed)
    print(f"     SAE final recon MSE: {sae_recon_mse:.6f}")
    z = encode_with_sae(sae, hidden, device=device)

    amps, bearing_idx, bearing_names, freqs_hz = gather_bpfx_amplitudes(
        dataset_key, max_recordings=max_recordings
    )
    bpfx_names = list(freqs_hz.keys())
    corrs = align_and_correlate(z, amps)
    hr = hit_rate_from_corrs(corrs, threshold)

    print(f"     Hit-rate (%) — " +
          "  ".join(f"{n}={hr[i]*100:.2f}" for i, n in enumerate(bpfx_names)))

    payload = {
        "dataset": dataset_key,
        "control": control_name,
        "seed": rng_seed,
        "threshold": threshold,
        "epochs": epochs,
        "n_hidden": int(hidden.shape[0]),
        "d_model": int(hidden.shape[1]),
        "sae_recon_mse": sae_recon_mse,
        "freqs_hz": freqs_hz,
        "bpfx_names": bpfx_names,
        "hit_rate": hr,
    }
    # Seed 0 keeps the legacy filename for backward-compatibility.
    # Seeds 42/43/44 use a seed-suffixed filename so runs don't overwrite each other.
    seed_tag = "" if rng_seed == 0 else f"_s{rng_seed}"
    out = OUT_ROOT / "negative_controls" / f"{dataset_key}_{control_name}{seed_tag}_results.json"
    write_json(out, payload)
    return payload


def main() -> None:
    args = _parse()
    device = get_device()
    rng = np.random.default_rng(args.seed)
    all_results: dict[str, dict] = {}

    for ds in args.datasets:
        info = DATASET_INFO[ds]
        if not info["run_dir"].exists():
            print(f"SKIP {ds}: training run dir not found")
            continue

        print(f"\n{'='*70}\n  NEGATIVE CONTROLS — {ds}\n{'='*70}")

        # Control 1: untrained backbone hidden states
        print(f"\n  Collecting hidden states from UNTRAINED backbone …")
        hidden_random = collect_hidden_for_dataset(
            ds, n_max=args.n_hidden, use_random_init=True
        )
        ctrl1 = _run_one_control(
            ds, "untrained_backbone", hidden_random,
            epochs=args.epochs, threshold=args.threshold,
            max_recordings=args.max_recordings, rng_seed=args.seed,
            device=device,
        )

        # Control 2: Gaussian noise with matched per-dim stats
        print(f"\n  Building Gaussian-noise pool with matched mean/std …")
        # Use the trained-backbone hidden statistics so the noise is on the
        # same scale as the real activations.
        hidden_real = collect_hidden_for_dataset(
            ds, n_max=args.n_hidden, use_random_init=False
        )
        mu = hidden_real.mean(axis=0)
        sd = hidden_real.std(axis=0) + 1e-6
        hidden_noise = (rng.standard_normal(hidden_real.shape).astype(np.float32) * sd + mu)
        ctrl2 = _run_one_control(
            ds, "gaussian_noise", hidden_noise,
            epochs=args.epochs, threshold=args.threshold,
            max_recordings=args.max_recordings, rng_seed=args.seed + 1,
            device=device,
        )

        all_results[ds] = {"untrained_backbone": ctrl1, "gaussian_noise": ctrl2}

    # ------------------------------------------------------------------
    # Comparison vs. trained baseline (read from existing summary if present)
    # ------------------------------------------------------------------
    baseline_path = (
        OUT_ROOT.parent / "bpfx_mapping" / "summary_hitrate_table.json"
    )
    if baseline_path.exists():
        import json

        baseline = json.loads(baseline_path.read_text())
    else:
        baseline = {}
    comparison = {"baseline_trained": baseline, "controls": all_results}
    write_json(OUT_ROOT / "negative_controls" / "comparison.json", comparison)

    # ------------------------------------------------------------------
    # Bar chart: trained vs untrained vs noise per dataset, BPFO leading
    # ------------------------------------------------------------------
    if all_results:
        bpfx = ["BPFO", "BPFI", "BSF", "FTF"]
        ds_keys = list(all_results.keys())
        x = np.arange(len(bpfx))
        width = 0.27
        fig, axes = plt.subplots(1, len(ds_keys), figsize=(5 * len(ds_keys), 4))
        if len(ds_keys) == 1:
            axes = [axes]
        for ax, ds in zip(axes, ds_keys):
            base_hr = baseline.get(ds, {}).get("hit_rate", {})
            base_vals = [base_hr.get(b, 0) * 100 for b in bpfx]
            untr = [all_results[ds]["untrained_backbone"]["hit_rate"][i] * 100
                    for i in range(len(bpfx))]
            noise = [all_results[ds]["gaussian_noise"]["hit_rate"][i] * 100
                     for i in range(len(bpfx))]
            ax.bar(x - width, base_vals, width, label="trained", color="#2E7D32")
            ax.bar(x, untr, width, label="untrained backbone", color="#FFA000")
            ax.bar(x + width, noise, width, label="Gaussian noise", color="#9E9E9E")
            ax.set_xticks(x)
            ax.set_xticklabels(bpfx)
            ax.set_ylabel("Hit-rate (%)")
            ax.set_title(f"{ds.upper()}")
            ax.legend(fontsize=9)
        fig.suptitle("Negative Controls — trained vs untrained vs Gaussian noise",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_ROOT / "negative_controls" / "comparison_bar.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nDone. Results in {OUT_ROOT / 'negative_controls'}")


if __name__ == "__main__":
    main()
