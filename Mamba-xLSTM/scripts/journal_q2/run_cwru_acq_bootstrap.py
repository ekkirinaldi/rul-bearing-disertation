"""CWRU acquisition-level bootstrap for the SAE–BPFx correspondence claim.

The file-level bootstrap used in run_stats.py has only 10 resampling units
(one per CWRU .mat file), making confidence intervals unreliable.  This script
instead resamples at the *acquisition* level: each .mat file is split into
non-overlapping windows (default: 4 800 samples ≈ 0.1 s at 48 kHz), giving
~900–1 000 acquisitions whose BPFx amplitudes are computed independently.

The trained Mamba-xLSTM-Net SAE hidden states are loaded at the same
windowed granularity and aligned with the amplitudes before running bootstrap.

Outputs:
  results/journal_q2/stats/cwru_stats_acq.json
  results/journal_q2/stats/cwru_hitrate_ci_acq.png

Usage::

    cd Mamba-xLSTM && source .venv/bin/activate
    python -m scripts.journal_q2.run_cwru_acq_bootstrap
    python -m scripts.journal_q2.run_cwru_acq_bootstrap --window 9600 --n-boot 2000
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
    band_amplitude,
    bpfx_frequencies,
    bootstrap_hit_rate_ci,
    collect_hidden_for_dataset,
    encode_with_sae,
    hilbert_envelope_spectrum,
    hit_rate_from_corrs,
    load_default_sae,
    permutation_pvalues,
    write_json,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--window",  type=int, default=4_800,
                   help="Samples per acquisition window (default 4800 = 0.1 s @ 48 kHz)")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--n-boot",  type=int, default=1_000)
    p.add_argument("--n-perm",  type=int, default=1_000)
    p.add_argument("--n-hidden", type=int, default=5_000)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def _load_cwru_windows(window_size: int) -> tuple[np.ndarray, list[str]]:
    """Load all CWRU .mat files and split into non-overlapping windows.

    Returns
    -------
    windows : list of (window_size,) float32 arrays — one per acquisition window
    labels  : corresponding file stem label for each window
    """
    try:
        import scipy.io
    except ImportError:
        raise RuntimeError("scipy is required: pip install scipy")

    info    = DATASET_INFO["cwru"]
    root    = info["raw_data_root"]
    stems   = info.get("train_bearing_dirs", [])
    fs_hz   = info["fs_hz"]

    all_windows: list[np.ndarray] = []
    all_labels:  list[str]        = []

    for stem in stems:
        mat_path = root / f"{stem}.mat"
        if not mat_path.exists():
            print(f"  [SKIP] {mat_path.name}")
            continue
        try:
            mat = scipy.io.loadmat(str(mat_path))
        except Exception as exc:
            print(f"  [WARN] could not load {mat_path.name}: {exc}")
            continue

        de_key = next(
            (k for k in mat if "DE_time" in k or
             (k.endswith("DE") and not k.startswith("_"))),
            None,
        )
        if de_key is None:
            de_key = max(
                (k for k in mat if not k.startswith("_")),
                key=lambda k: mat[k].size if isinstance(mat[k], np.ndarray) else 0,
                default=None,
            )
        if de_key is None:
            print(f"  [WARN] no DE channel in {mat_path.name}")
            continue

        signal = mat[de_key].squeeze().astype(np.float32)
        n_windows = len(signal) // window_size
        for i in range(n_windows):
            w = signal[i * window_size : (i + 1) * window_size]
            all_windows.append(w)
            all_labels.append(stem)

    n_per_stem = len(all_windows) // max(1, len(stems))
    print(f"  Loaded {len(stems)} files → {len(all_windows)} windows "
          f"({n_per_stem} per file on average, window_size={window_size})")
    return all_windows, all_labels


def _compute_amps_per_window(
    windows: list[np.ndarray],
    freqs_hz: dict[str, float],
    fs_hz: float,
    bw_hz: float = 2.0,
) -> np.ndarray:
    """Compute BPFx band amplitudes for each window.

    Returns
    -------
    amps : (n_windows, 4) float32
    """
    bpfx_names = list(freqs_hz.keys())
    out = np.zeros((len(windows), len(bpfx_names)), dtype=np.float32)
    for i, w in enumerate(windows):
        f_env, A_env = hilbert_envelope_spectrum(w, fs_hz=fs_hz)
        for j, freq in enumerate(freqs_hz.values()):
            out[i, j] = band_amplitude(f_env, A_env, freq, bw_hz=bw_hz)
    return out


def main() -> None:
    args = _parse()
    rng  = np.random.default_rng(args.seed)
    out_dir = OUT_ROOT / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    info     = DATASET_INFO["cwru"]
    fs_hz    = info["fs_hz"]
    freqs_hz = bpfx_frequencies(**info["bearing_geom"], fr=info["fr_hz"])
    bpfx_names = list(freqs_hz.keys())

    # 1. Load and window raw CWRU signals
    print("\n[CWRU acquisition-level bootstrap]")
    print(f"  window_size={args.window}, fs={fs_hz} Hz  "
          f"→ {args.window / fs_hz * 1000:.0f} ms per window")
    windows, labels = _load_cwru_windows(args.window)
    if not windows:
        print("  No CWRU windows found — check data-bearing/cwru/")
        return

    # 2. Compute BPFx amplitude per window
    print(f"  Computing BPFx amplitudes for {len(windows)} windows …")
    amps = _compute_amps_per_window(windows, freqs_hz, fs_hz)
    print(f"  amps shape: {amps.shape}")

    # 3. Load trained SAE + hidden states
    sae    = load_default_sae("cwru")
    hidden = collect_hidden_for_dataset("cwru", n_max=args.n_hidden)
    print(f"  Hidden states: {hidden.shape}")
    z_sparse = encode_with_sae(sae, hidden)
    print(f"  Sparse activations: {z_sparse.shape}")

    # 4. Align and correlate (uses min(N_z, N_amps) samples)
    corrs    = align_and_correlate(z_sparse, amps)
    point_hr = hit_rate_from_corrs(corrs, args.threshold)

    # 5. Acquisition-level bootstrap
    print(f"  Running acquisition-level bootstrap B={args.n_boot} …")
    boot = bootstrap_hit_rate_ci(
        z_sparse, amps, threshold=args.threshold, n_boot=args.n_boot, rng=rng
    )

    # 6. Permutation test
    print(f"  Running permutation test B={args.n_perm} …")
    perm = permutation_pvalues(
        z_sparse, amps, threshold=args.threshold, n_perm=args.n_perm, rng=rng
    )

    # 7. Summary
    print(f"\n  {'BPFx':6s}  {'point':>6s}  {'95% CI':>18s}  {'p (perm)':>10s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*18}  {'-'*10}")
    for ci, name in enumerate(bpfx_names):
        print(
            f"  {name:6s}  {point_hr[ci]*100:5.2f}%  "
            f"[{boot['low'][ci]*100:5.2f}, {boot['high'][ci]*100:5.2f}]%  "
            f"{perm['p_value'][ci]:10.4f}"
        )

    # 8. Save JSON
    n_align = min(z_sparse.shape[0], amps.shape[0])
    payload = {
        "dataset":          "cwru",
        "bootstrap_level":  "acquisition",
        "window_samples":   args.window,
        "window_ms":        round(args.window / fs_hz * 1000, 1),
        "n_windows":        len(windows),
        "n_aligned":        int(n_align),
        "threshold":        args.threshold,
        "n_boot":           args.n_boot,
        "n_perm":           args.n_perm,
        "bpfx_names":       bpfx_names,
        "freqs_hz":         freqs_hz,
        "point_hit_rate":   point_hr.tolist(),
        "point_hit_rate_pct": [round(v * 100, 2) for v in point_hr.tolist()],
        "bootstrap_low":    boot["low"].tolist(),
        "bootstrap_high":   boot["high"].tolist(),
        "permutation_pvalue": perm["p_value"].tolist(),
    }
    out_json = out_dir / "cwru_stats_acq.json"
    write_json(out_json, payload)
    print(f"\n  JSON → {out_json.name}")

    # 9. CI bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(bpfx_names))
    yerr_low  = (point_hr - boot["low"])  * 100
    yerr_high = (boot["high"] - point_hr) * 100
    ax.bar(x, point_hr * 100, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
           yerr=[yerr_low, yerr_high], capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bpfx_names)
    ax.set_ylabel("Hit-rate (%)")
    ax.set_title(
        f"CWRU acquisition-level bootstrap 95% CI\n"
        f"(|r|≥{args.threshold}, B={args.n_boot}, "
        f"N={len(windows)} windows ×{args.window} samples)"
    )
    for ci, p in enumerate(perm["p_value"]):
        ax.text(x[ci],
                (point_hr[ci] + (boot["high"][ci] - point_hr[ci])) * 100 + 0.3,
                f"p={p:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cwru_hitrate_ci_acq.png", dpi=150)
    plt.close(fig)

    # 10. Interpretation note
    print()
    for ci, name in enumerate(bpfx_names):
        trained_gt_null = point_hr[ci] > boot["low"][ci]
        pval_sig = perm["p_value"][ci] < 0.05
        print(f"  {name}: CI spans [0] → "
              f"{'SUPPORTS falsification' if trained_gt_null and pval_sig else 'NOT significant at acq level'}")


if __name__ == "__main__":
    main()
