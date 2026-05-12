"""Stage 4 — BPFx frequency mapping.

For each dataset (PHM2012, XJTU-SY) this script:

1. Computes theoretical bearing characteristic frequencies:
   BPFO, BPFI, BSF, FTF from geometry of NSK 6804 / LDK UER204.

2. Loads the trained SAE (sae.pt) from Stage 3 and the best RUL model
   checkpoint to collect hidden states over ALL training bearings.

3. For every recording in the training set:
   a. Computes the Hilbert envelope spectrum of the raw vibration signal.
   b. Reads band-energy amplitudes at BPFO, BPFI, BSF, FTF
      (±bandwidth Hz around each characteristic frequency).
   c. Runs the window through the SAE encoder → sparse activation vector.

4. Computes Pearson correlation between each SAE feature's activation
   and each BPFx band amplitude across all recordings.

5. A SAE feature is counted as a "hit" for frequency F if |r| >= threshold
   (default 0.3, i.e. moderate correlation).

6. Reports:
   - hit-rate table: % of SAE features that hit each BPFx (per dataset)
   - top-5 SAE features per BPFx (by mean |r|)
   - saves results/bpfx_mapping/<dataset>_bpfx_results.json + .png

Usage::

    cd Mamba-xLSTM
    source .venv/bin/activate        # or: conda activate <env>
    python scripts/run_bpfx_mapping.py

Outputs written to:
    results/bpfx_mapping/phm2012_bpfx_results.json
    results/bpfx_mapping/phm2012_hitrate_bar.png
    results/bpfx_mapping/xjtusy_bpfx_results.json
    results/bpfx_mapping/xjtusy_hitrate_bar.png
    results/bpfx_mapping/summary_hitrate_table.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch

_PKG = Path(__file__).resolve().parents[1] / "src"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mxlstm.interp.sae import SAEConfig, TopKSparseAutoencoder, load_sae  # noqa: E402

# ---------------------------------------------------------------------------
# Bearing geometry & characteristic frequencies
# ---------------------------------------------------------------------------

def bpfx_frequencies(
    n: int, d: float, D: float, fr: float, theta_deg: float = 0.0
) -> dict[str, float]:
    """Return BPFO, BPFI, BSF, FTF in Hz.

    Args:
        n:         number of rolling elements
        d:         rolling element diameter (mm)
        D:         pitch diameter (mm)  — diameter of rolling-element centre circle
        fr:        shaft rotation frequency (Hz)
        theta_deg: contact angle in degrees (0 for deep groove ball bearings)
    """
    theta = np.deg2rad(theta_deg)
    ratio = (d / D) * np.cos(theta)
    bpfo = (n / 2) * fr * (1 - ratio)
    bpfi = (n / 2) * fr * (1 + ratio)
    bsf = (D / (2 * d)) * fr * (1 - ratio**2)
    ftf = (fr / 2) * (1 - ratio)
    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}


# Bearing geometries from 15-domain-rul-bearings.mdc and datasheet references.
# NSK 6804 (PHM2012 / PRONOSTIA): deep groove ball bearing
#   n=13, d=3.50 mm, D=25.50 mm, theta=0°
#   Condition 1: 1800 rpm → fr = 30 Hz
NSK_6804 = dict(n=13, d=3.50, D=25.50, theta_deg=0.0)

# LDK UER204 (XJTU-SY): deep groove ball bearing
#   n=8, d=7.92 mm, D=34.55 mm, theta=0°
#   Condition 1: 2100 rpm → fr = 35 Hz
#   Condition 2: 2250 rpm → fr = 37.5 Hz
LDK_UER204 = dict(n=8, d=7.92, D=34.55, theta_deg=0.0)

DATASET_INFO = {
    "phm2012": {
        "bearing_geom": NSK_6804,
        # PHM2012 has 3 conditions; training bearings are mostly condition 1 (1800 rpm)
        # and condition 2 (1650 rpm). Use 30 Hz (condition 1) as primary reference.
        "fr_hz": 30.0,
        "fs_hz": 25_600.0,
        "raw_data_root": Path(__file__).resolve().parents[2]
            / "data-bearing" / "ieee-phm-2012" / "Learning_set",
        # bearing dirs that belong to training set (7 bearings)
        "train_bearing_dirs": [
            "Bearing1_1", "Bearing1_2", "Bearing1_4",
            "Bearing2_1", "Bearing2_3", "Bearing2_5",
            "Bearing3_1",
        ],
        "sae_pt": Path(__file__).resolve().parents[1]
            / "results/runs"
            / "20260512_151550_algorithm_comparison_phm2012_mamba_xlstm_net_s42"
            / "explain/sae.pt",
        "run_dir": Path(__file__).resolve().parents[1]
            / "results/runs"
            / "20260512_151550_algorithm_comparison_phm2012_mamba_xlstm_net_s42",
    },
    "xjtusy": {
        "bearing_geom": LDK_UER204,
        # Use 35 Hz (condition 1, 2100 rpm) as primary.
        "fr_hz": 35.0,
        "fs_hz": 25_600.0,
        "raw_data_root": Path(__file__).resolve().parents[2]
            / "data-bearing" / "xtju-sy" / "35Hz12kN",
        "train_bearing_dirs": [
            "Bearing1_1", "Bearing1_2", "Bearing1_3",
            "Bearing2_1", "Bearing2_2",
        ],
        "sae_pt": Path(__file__).resolve().parents[1]
            / "results/runs"
            / "20260512_193202_algorithm_comparison_xjtusy_mamba_xlstm_net_s44"
            / "explain/sae.pt",
        "run_dir": Path(__file__).resolve().parents[1]
            / "results/runs"
            / "20260512_193202_algorithm_comparison_xjtusy_mamba_xlstm_net_s44",
    },
}

# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def hilbert_envelope_spectrum(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided envelope spectrum (FFT of the Hilbert envelope)."""
    analytic = signal.hilbert(sig)
    envelope = np.abs(analytic)
    # Detrend envelope before FFT
    envelope -= envelope.mean()
    n = len(envelope)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(envelope)) / n
    return freqs, spectrum


def band_amplitude(
    freqs: np.ndarray, spectrum: np.ndarray, center: float, bw: float = 2.0
) -> float:
    """Mean spectral amplitude in [center-bw, center+bw] Hz band."""
    mask = (freqs >= center - bw) & (freqs <= center + bw)
    if not mask.any():
        return 0.0
    return float(spectrum[mask].mean())


# ---------------------------------------------------------------------------
# Raw data loaders
# ---------------------------------------------------------------------------

def load_phm2012_recordings(bearing_dir: Path) -> list[np.ndarray]:
    """Load all acc_*.csv files for one PHM2012 bearing (horizontal channel)."""
    csvs = sorted(bearing_dir.glob("acc_*.csv"))
    recordings = []
    for f in csvs:
        try:
            data = np.loadtxt(f, delimiter=",", usecols=(4,))  # col 4 = horiz accel
            recordings.append(data.astype(np.float32))
        except Exception:
            continue
    return recordings


def load_xjtusy_recordings(bearing_dir: Path) -> list[np.ndarray]:
    """Load all *.csv files for one XJTU-SY bearing (horizontal channel)."""
    csvs = sorted(bearing_dir.glob("*.csv"), key=lambda p: int(p.stem))
    recordings = []
    for f in csvs:
        try:
            data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0,))  # horizontal
            recordings.append(data.astype(np.float32))
        except Exception:
            continue
    return recordings


# ---------------------------------------------------------------------------
# SAE activation from hidden states via stored model
# ---------------------------------------------------------------------------

def collect_sae_activations_from_features(
    sae: TopKSparseAutoencoder,
    run_dir: Path,
    n_max: int = 5000,
) -> np.ndarray:
    """Load hidden states collected during Stage 3 or re-collect from the model.

    Falls back to collecting from the config.yaml checkpoint if no cached hidden
    states are found.  Returns (N, d_latent) sparse activation array.
    """
    # Try loading hidden states directly — Stage 3 saved sae_history but not hidden.
    # We regenerate them from the model + data.
    import importlib.util
    import sys as _sys

    train_script = Path(__file__).resolve().parents[0] / "train.py"
    spec = importlib.util.spec_from_file_location("_train_build", train_script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from omegaconf import OmegaConf
    from mxlstm.data.datamodule import RULDataModule
    from mxlstm.compute import get_device
    from mxlstm.training.lit_module import RULLitModule
    from mxlstm.interp.sae import collect_hidden_states
    from mxlstm.interp.explain_extras import resolve_checkpoint

    cfg_path = run_dir / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
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

    ckpt = resolve_checkpoint(run_dir)
    model = mod._build_model(cfg, n_features=dm.n_features,
                             context_length=int(cfg.data.window_length))
    lit = RULLitModule.load_from_checkpoint(str(ckpt), model=model, map_location=device)
    lit = lit.to(device).eval()

    hidden = collect_hidden_states(
        lit.model, dm.train_dataloader(), device=device,
        layer="fused", max_samples=n_max,
    )

    sae_dev = next(sae.parameters()).device
    with torch.no_grad():
        z = sae.encode(torch.from_numpy(hidden.astype(np.float32)).to(sae_dev))
        z_sparse = sae.topk(z).cpu().numpy()
    return z_sparse  # (N, d_latent)


# ---------------------------------------------------------------------------
# Main per-dataset analysis
# ---------------------------------------------------------------------------

def analyse_dataset(
    dataset_key: str,
    info: dict,
    out_dir: Path,
    corr_threshold: float = 0.3,
    bpfx_bw_hz: float = 2.0,
    max_recordings: int = 300,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_key}")
    print(f"{'='*60}")

    # 1. Characteristic frequencies
    freqs_hz = bpfx_frequencies(
        **info["bearing_geom"], fr=info["fr_hz"]
    )
    print(f"  Characteristic frequencies (fr={info['fr_hz']} Hz):")
    for k, v in freqs_hz.items():
        print(f"    {k:5s} = {v:.2f} Hz")

    # 2. Load SAE
    print(f"\n  Loading SAE from {info['sae_pt']} …")
    sae = load_sae(info["sae_pt"])
    sae.eval()
    d_latent = sae.d_latent
    print(f"  SAE: d_model={sae.cfg.d_model}  expansion={sae.cfg.expansion}"
          f"  d_latent={d_latent}  k={sae.k}")

    # 3. Collect BPFx amplitudes from raw recordings
    raw_root = info["raw_data_root"]
    loader_fn = load_phm2012_recordings if dataset_key == "phm2012" else load_xjtusy_recordings
    fs = info["fs_hz"]
    bpfx_names = list(freqs_hz.keys())

    bpfx_amps: list[dict[str, float]] = []
    rec_count = 0

    for b_dir_name in info["train_bearing_dirs"]:
        b_dir = raw_root / b_dir_name
        if not b_dir.exists():
            print(f"  WARN: {b_dir} not found, skipping")
            continue
        recs = loader_fn(b_dir)
        print(f"  {b_dir_name}: {len(recs)} recordings")
        for rec in recs:
            if rec_count >= max_recordings:
                break
            env_freqs, env_spec = hilbert_envelope_spectrum(rec, fs)
            amp_row = {
                name: band_amplitude(env_freqs, env_spec, center, bw=bpfx_bw_hz)
                for name, center in freqs_hz.items()
            }
            bpfx_amps.append(amp_row)
            rec_count += 1
        if rec_count >= max_recordings:
            break

    print(f"\n  Total recordings with BPFx amplitudes: {rec_count}")

    # 4. Collect SAE activations for the same N windows from the model
    print("  Collecting SAE activations from model hidden states …")
    z_sparse = collect_sae_activations_from_features(
        sae, info["run_dir"], n_max=rec_count
    )
    # Align lengths (model windows may differ from raw recordings due to stride)
    n_align = min(rec_count, z_sparse.shape[0])
    bpfx_amps_arr = np.array([
        [row[k] for k in bpfx_names] for row in bpfx_amps[:n_align]
    ])  # (n_align, 4)
    z_align = z_sparse[:n_align]  # (n_align, d_latent)

    print(f"  Aligned samples: {n_align}  |  SAE latents: {d_latent}")

    # 5. Pearson correlation: (d_latent, 4)
    print("  Computing Pearson correlations …")
    corrs = np.zeros((d_latent, len(bpfx_names)), dtype=np.float32)
    for fi in range(d_latent):
        feat_act = z_align[:, fi]
        if feat_act.std() < 1e-8:
            continue  # dead feature
        for bi, bname in enumerate(bpfx_names):
            amp_col = bpfx_amps_arr[:, bi]
            if amp_col.std() < 1e-8:
                continue
            r = float(np.corrcoef(feat_act, amp_col)[0, 1])
            corrs[fi, bi] = r if not np.isnan(r) else 0.0

    # 6. Hit-rate: fraction of SAE features with |r| >= threshold
    hits = (np.abs(corrs) >= corr_threshold)  # (d_latent, 4)
    hit_rate = hits.mean(axis=0)             # (4,)

    print(f"\n  Hit-rate (|r| >= {corr_threshold}):")
    hit_rate_dict: dict[str, float] = {}
    for bi, bname in enumerate(bpfx_names):
        hr = float(hit_rate[bi])
        hit_rate_dict[bname] = hr
        n_hits = int(hits[:, bi].sum())
        print(f"    {bname:5s}: {hr*100:5.1f}%  ({n_hits}/{d_latent} features)")

    # 7. Top-5 features per BPFx
    top5: dict[str, list[dict]] = {}
    for bi, bname in enumerate(bpfx_names):
        abs_r = np.abs(corrs[:, bi])
        top_idx = np.argsort(-abs_r)[:5]
        top5[bname] = [
            {"feature_idx": int(i), "r": float(corrs[i, bi])}
            for i in top_idx
        ]

    # 8. Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": dataset_key,
        "bearing_geometry": info["bearing_geom"],
        "fr_hz": info["fr_hz"],
        "characteristic_frequencies_hz": freqs_hz,
        "n_recordings": n_align,
        "d_latent": d_latent,
        "corr_threshold": corr_threshold,
        "hit_rate": hit_rate_dict,
        "top5_features": top5,
    }
    out_path = out_dir / f"{dataset_key}_bpfx_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Results saved → {out_path}")

    # 9. Plot hit-rate bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    bars = ax.bar(bpfx_names, [hit_rate_dict[k] * 100 for k in bpfx_names], color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    ax.set_ylabel("Hit-rate (%)")
    ax.set_title(
        f"SAE Feature Hit-Rate — {dataset_key.upper()}\n"
        f"(|Pearson r| ≥ {corr_threshold}, d_latent={d_latent})"
    )
    ax.set_ylim(0, max(hit_rate_dict.values()) * 130 + 5)
    ax.axhline(corr_threshold * 100, color="red", linestyle="--", alpha=0.5,
               label=f"threshold {corr_threshold*100:.0f}%")
    fig.tight_layout()
    fig_path = out_dir / f"{dataset_key}_hitrate_bar.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved → {fig_path}")

    # 10. Correlation heatmap (d_latent × 4) — subsample to top 50 active features
    active_mask = (np.abs(corrs).max(axis=1) > 0.05)
    active_idx = np.where(active_mask)[0]
    if len(active_idx) > 50:
        # Pick top 50 by max |r| across all BPFx
        sort_by = np.abs(corrs[active_idx]).max(axis=1)
        active_idx = active_idx[np.argsort(-sort_by)[:50]]
    if len(active_idx) > 1:
        fig2, ax2 = plt.subplots(figsize=(5, max(4, len(active_idx) * 0.18)))
        im = ax2.imshow(
            corrs[active_idx], aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6
        )
        ax2.set_xticks(range(len(bpfx_names)))
        ax2.set_xticklabels(bpfx_names, fontsize=10)
        ax2.set_yticks(range(len(active_idx)))
        ax2.set_yticklabels([f"f{i}" for i in active_idx], fontsize=7)
        ax2.set_title(f"Correlation heatmap — {dataset_key.upper()} (top active features)")
        fig2.colorbar(im, ax=ax2, label="Pearson r")
        fig2.tight_layout()
        hmap_path = out_dir / f"{dataset_key}_corr_heatmap.png"
        fig2.savefig(hmap_path, dpi=150)
        plt.close(fig2)
        print(f"  Heatmap saved → {hmap_path}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "bpfx_mapping"
    all_results: dict[str, dict] = {}

    for ds_key, ds_info in DATASET_INFO.items():
        if not ds_info["sae_pt"].exists():
            print(f"SKIP {ds_key}: SAE weights not found at {ds_info['sae_pt']}")
            continue
        result = analyse_dataset(ds_key, ds_info, out_dir)
        all_results[ds_key] = {
            "characteristic_frequencies_hz": result["characteristic_frequencies_hz"],
            "hit_rate": result["hit_rate"],
        }

    # Cross-dataset summary table
    summary_path = out_dir / "summary_hitrate_table.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'='*60}")
    print("  CROSS-DATASET SUMMARY")
    print(f"{'='*60}")
    bpfx_cols = ["BPFO", "BPFI", "BSF", "FTF"]
    header = f"{'Dataset':10s}" + "".join(f"  {c:>6s}" for c in bpfx_cols)
    print(header)
    print("-" * len(header))
    for ds, res in all_results.items():
        hr = res["hit_rate"]
        row = f"{ds:10s}" + "".join(f"  {hr.get(c, 0)*100:5.1f}%" for c in bpfx_cols)
        print(row)
    print(f"\n  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
