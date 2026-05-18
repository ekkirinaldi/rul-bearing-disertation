"""Cross-architecture BPFx mapping — train an SAE on the hidden states of
N-BEATS-xLSTM-RUL and SparseGate-TCN-RUL checkpoints, then run the same
BPFx mapping pipeline as the Mamba-xLSTM baseline.

Strategy
--------
1. Locate one canonical checkpoint per (architecture, dataset) combination
   under ``Mamba-xLSTM/results/runs/``. Default heuristic: pick the most
   recent run with seed 42.
2. For each checkpoint:
     a. Build the model from the run's ``config.yaml``.
     b. Load weights.
     c. Collect ``--n-hidden`` hidden states from the train dataloader.
     d. Train a fresh Top-k SAE.
     e. Run BPFx mapping (Pearson r vs envelope-spectrum amplitudes).
3. Aggregate per-architecture hit-rate into a comparison JSON + bar plot.

Outputs:
  Mamba-xLSTM/results/journal_q2/cross_arch/<arch>_<dataset>.json
  Mamba-xLSTM/results/journal_q2/cross_arch/comparison.json
  Mamba-xLSTM/results/journal_q2/cross_arch/comparison_bar.png

Usage::

    cd Mamba-xLSTM && source .venv/bin/activate
    python -m scripts.journal_q2.run_cross_arch
    python -m scripts.journal_q2.run_cross_arch \\
        --architectures nbeats_xlstm_rul sparse_gate_tcn_rul \\
        --datasets phm2012 xjtusy
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from mxlstm.compute import get_device
from mxlstm.data.datamodule import RULDataModule
from mxlstm.interp.explain_extras import resolve_checkpoint
from mxlstm.interp.sae import (
    SAEConfig,
    TopKSparseAutoencoder,
    collect_hidden_states,
    train_sae,
)
from mxlstm.training.lit_module import RULLitModule

from scripts.journal_q2._helpers import (
    DATASET_INFO,
    OUT_ROOT,
    align_and_correlate,
    encode_with_sae,
    gather_bpfx_amplitudes,
    hit_rate_from_corrs,
    write_json,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNS = _REPO_ROOT / "Mamba-xLSTM" / "results" / "runs"


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--architectures",
        nargs="+",
        default=["mamba_xlstm_net", "nbeats_xlstm_rul", "sparse_gate_tcn_rul"],
    )
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_INFO.keys()),
                   choices=list(DATASET_INFO.keys()))
    p.add_argument("--seed", type=int, default=42,
                   help="Pick the latest run with this seed suffix.")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-recordings", type=int, default=300)
    p.add_argument("--n-hidden", type=int, default=5_000)
    return p.parse_args()


def _find_run_dir(arch: str, dataset: str, seed: int) -> Path | None:
    """Pick the latest run directory matching ``arch`` and ``dataset``."""
    pattern = f"*_algorithm_comparison_{dataset}_{arch}_s{seed}"
    matches = sorted(_RUNS.glob(pattern))
    return matches[-1] if matches else None


def _build_model_from_run(run_dir: Path, dataset: str):
    """Build the Lightning model from a run's config.yaml; load weights."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    device = get_device()
    dm = RULDataModule(
        dataset=cfg.data.dataset,
        root=cfg.data.root,
        train_bearings=list(cfg.data.train_bearings),
        val_bearings=list(cfg.data.val_bearings),
        test_bearings=list(cfg.data.test_bearings),
        window_length=int(cfg.data.window_length),
        stride_train=int(cfg.data.stride_train),
        stride_eval=int(cfg.data.stride_eval),
        label_scheme=str(cfg.data.label_scheme),
        smoothing_alpha=float(cfg.data.smoothing_alpha),
        n_bands=int(cfg.data.n_bands),
        hi_pipeline=str(cfg.data.get("hi_pipeline", "default")),
        phm_horizontal_only=bool(cfg.data.get("phm_horizontal_only", False)),
        allow_train_val_overlap=bool(cfg.data.get("allow_train_val_overlap", False)),
        batch_size=int(cfg.data.batch_size),
        num_workers=0,
    )
    dm.setup()

    train_script = _REPO_ROOT / "Mamba-xLSTM" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_train_build", train_script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod._build_model(
        cfg, n_features=dm.n_features, context_length=int(cfg.data.window_length)
    )
    ckpt = resolve_checkpoint(run_dir)
    lit = RULLitModule.load_from_checkpoint(
        str(ckpt), model=model, map_location=device
    )
    lit = lit.to(device).eval()
    return lit, dm, cfg, device


def _layer_for(arch: str) -> str:
    """Pick the hidden-state layer to extract for each architecture.

    All three architectures expose a ``return_hidden=True`` interface
    in this repo; the canonical pre-head representation is keyed
    'fused' for Mamba-xLSTM-Net and matches the same convention for
    the N-BEATS / SparseGate variants.
    """
    return "fused"


def _train_sae_for_arch(
    hidden: np.ndarray, *, epochs: int, device
) -> tuple[TopKSparseAutoencoder, float]:
    """Train an SAE and return (sae, final_recon_mse)."""
    cfg = SAEConfig(d_model=hidden.shape[1], expansion=8)
    sae = TopKSparseAutoencoder(cfg)
    history = train_sae(sae, hidden, epochs=epochs, batch_size=256, lr=1e-3, device=device)
    final_mse = float(history[-1]["recon"]) if history else float("nan")
    return sae, final_mse


def main() -> None:
    args = _parse()
    summary: dict = {}

    for ds in args.datasets:
        amps, _, _, freqs_hz = gather_bpfx_amplitudes(
            ds, max_recordings=args.max_recordings
        )
        bpfx_names = list(freqs_hz.keys())
        summary.setdefault(ds, {})

        for arch in args.architectures:
            run_dir = _find_run_dir(arch, ds, args.seed)
            if run_dir is None:
                print(f"SKIP {arch} / {ds}: no run dir found (seed={args.seed})")
                continue
            print(f"\n{'='*70}\n  CROSS-ARCH — {arch} / {ds}\n  run = {run_dir.name}\n{'='*70}")

            lit, dm, cfg, device = _build_model_from_run(run_dir, ds)
            try:
                hidden = collect_hidden_states(
                    lit.model, dm.train_dataloader(), device=device,
                    layer=_layer_for(arch), max_samples=args.n_hidden,
                )
            except Exception as exc:
                print(f"  WARN: hidden-state collection failed for {arch}: {exc}")
                continue
            print(f"  hidden states: {hidden.shape}")

            sae, sae_recon_mse = _train_sae_for_arch(hidden, epochs=args.epochs, device=device)
            print(f"  SAE final recon MSE: {sae_recon_mse:.6f}")

            # Save the mamba_xlstm_net SAE so that run_stats.py can load it later.
            if arch == "mamba_xlstm_net":
                explain_dir = run_dir / "explain"
                explain_dir.mkdir(parents=True, exist_ok=True)
                sae_save_path = explain_dir / "sae.pt"
                torch.save({"state_dict": sae.state_dict(), "cfg": sae.cfg}, str(sae_save_path))
                print(f"  SAE saved → {sae_save_path}")

            z = encode_with_sae(sae, hidden, device=device)
            corrs = align_and_correlate(z, amps)
            hr = hit_rate_from_corrs(corrs, args.threshold)
            print(f"  hit-rate (%) — " +
                  "  ".join(f"{n}={hr[i]*100:.2f}" for i, n in enumerate(bpfx_names)))

            payload = {
                "dataset": ds,
                "architecture": arch,
                "run_dir": str(run_dir),
                "d_model": int(hidden.shape[1]),
                "d_latent": sae.d_latent,
                "k": sae.k,
                "sae_recon_mse": sae_recon_mse,
                "threshold": args.threshold,
                "freqs_hz": freqs_hz,
                "bpfx_names": bpfx_names,
                "hit_rate": hr,
            }
            write_json(
                OUT_ROOT / "cross_arch" / f"{arch}_{ds}.json", payload
            )
            summary[ds][arch] = payload

    write_json(OUT_ROOT / "cross_arch" / "comparison.json", summary)

    # ---- comparison bar plot ----
    if summary:
        bpfx = ["BPFO", "BPFI", "BSF", "FTF"]
        ds_keys = [k for k, v in summary.items() if v]
        fig, axes = plt.subplots(1, max(len(ds_keys), 1),
                                 figsize=(5 * max(len(ds_keys), 1), 4))
        if len(ds_keys) == 1:
            axes = [axes]
        for ax, ds in zip(axes, ds_keys):
            archs = list(summary[ds].keys())
            x = np.arange(len(bpfx))
            width = 0.8 / max(len(archs), 1)
            for ai, arch in enumerate(archs):
                hr = summary[ds][arch]["hit_rate"]
                ax.bar(x + (ai - (len(archs) - 1) / 2) * width,
                       [v * 100 for v in hr], width, label=arch)
            ax.set_xticks(x)
            ax.set_xticklabels(bpfx)
            ax.set_ylabel("Hit-rate (%)")
            ax.set_title(ds.upper())
            ax.legend(fontsize=8)
        fig.suptitle("Cross-architecture BPFx hit-rate", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_ROOT / "cross_arch" / "comparison_bar.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nDone. Results in {OUT_ROOT / 'cross_arch'}")


if __name__ == "__main__":
    main()
