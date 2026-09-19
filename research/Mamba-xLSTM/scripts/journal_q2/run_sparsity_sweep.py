"""Sparsity sweep — re-train the SAE for k in {10, 51, 102, 205} on the
trained Mamba-xLSTM hidden states and re-run BPFx mapping for each.

Default expansion = 8 with d_model = 128 yields d_latent = 1024, so the
four k values correspond to roughly 1, 5, 10, and 20 percent activations
per sample.

Outputs:
  Mamba-xLSTM/results/journal_q2/sparsity_sweep/<dataset>_k=<k>.json
  Mamba-xLSTM/results/journal_q2/sparsity_sweep/<dataset>_sweep.png
  Mamba-xLSTM/results/journal_q2/sparsity_sweep/summary.json

Usage::

    cd Mamba-xLSTM && source .venv/bin/activate
    python -m scripts.journal_q2.run_sparsity_sweep
    python -m scripts.journal_q2.run_sparsity_sweep --ks 10 51 102 205 --epochs 30
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
    p.add_argument("--ks", nargs="+", type=int, default=[10, 51, 102, 205])
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-recordings", type=int, default=300)
    p.add_argument("--n-hidden", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _train_sae_with_k(hidden: np.ndarray, *, k: int,
                       epochs: int, device, seed: int) -> TopKSparseAutoencoder:
    torch.manual_seed(seed)
    cfg = SAEConfig(d_model=hidden.shape[1], expansion=8, k=k)
    sae = TopKSparseAutoencoder(cfg)
    train_sae(sae, hidden, epochs=epochs, batch_size=256, lr=1e-3, device=device)
    return sae


def main() -> None:
    args = _parse()
    device = get_device()
    summary: dict[str, dict[str, dict]] = {}

    for ds in args.datasets:
        if not DATASET_INFO[ds]["run_dir"].exists():
            print(f"SKIP {ds}: run dir not found")
            continue
        print(f"\n{'='*70}\n  SPARSITY SWEEP — {ds}\n{'='*70}")
        hidden = collect_hidden_for_dataset(ds, n_max=args.n_hidden)
        amps, _, _, freqs_hz = gather_bpfx_amplitudes(
            ds, max_recordings=args.max_recordings
        )
        bpfx_names = list(freqs_hz.keys())
        per_k: dict[str, dict] = {}

        for k in args.ks:
            print(f"\n  -- k = {k}")
            sae = _train_sae_with_k(
                hidden, k=k, epochs=args.epochs, device=device, seed=args.seed
            )
            z = encode_with_sae(sae, hidden, device=device)
            corrs = align_and_correlate(z, amps)
            hr = hit_rate_from_corrs(corrs, args.threshold)
            print(f"     Hit-rate — " +
                  "  ".join(f"{n}={hr[i]*100:.2f}%" for i, n in enumerate(bpfx_names)))
            payload = {
                "dataset": ds,
                "k": int(k),
                "k_pct_of_d_latent": float(k) / sae.d_latent,
                "d_latent": sae.d_latent,
                "threshold": args.threshold,
                "freqs_hz": freqs_hz,
                "bpfx_names": bpfx_names,
                "hit_rate": hr,
            }
            write_json(
                OUT_ROOT / "sparsity_sweep" / f"{ds}_k={k}.json", payload
            )
            per_k[str(k)] = payload

        summary[ds] = per_k

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        for bi, bname in enumerate(bpfx_names):
            ys = [per_k[str(k)]["hit_rate"][bi] * 100 for k in args.ks]
            ax.plot(args.ks, ys, marker="o", label=bname)
        ax.set_xlabel("k (active SAE features per sample)")
        ax.set_ylabel("Hit-rate (%)")
        ax.set_xscale("log")
        ax.set_title(f"{ds.upper()} — sparsity sweep")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_ROOT / "sparsity_sweep" / f"{ds}_sweep.png", dpi=150)
        plt.close(fig)

    write_json(OUT_ROOT / "sparsity_sweep" / "summary.json", summary)
    print(f"\nSummary saved → {OUT_ROOT / 'sparsity_sweep' / 'summary.json'}")


if __name__ == "__main__":
    main()
