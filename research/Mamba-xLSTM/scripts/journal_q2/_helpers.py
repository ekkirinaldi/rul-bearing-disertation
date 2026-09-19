"""Shared helpers for the journal Q2 extra experiments.

Re-uses the existing ``scripts/run_bpfx_mapping.py`` utilities for
bearing geometry, envelope analysis, and raw-data loading, and adds
new statistical-inference and per-bearing helpers needed by the
JETS paper.

Public surface:

    DATASET_INFO                   — dataset registry (PHM2012, XJTU-SY, IMS, CWRU)
    OUT_ROOT                       — Path("Mamba-xLSTM/results/journal_q2")

    bpfx_frequencies               — geometry --> {BPFO,BPFI,BSF,FTF}
    hilbert_envelope_spectrum      — Hilbert envelope spectrum
    band_amplitude                 — band-integrated amplitude

    gather_bpfx_amplitudes         — raw recordings --> bpfx amps + bearing labels
    encode_with_sae                — hidden states --> sparse activations
    align_and_correlate            — Pearson r between activations and amplitudes
    hit_rate_from_corrs            — corr matrix --> hit-rate per BPFx
    bootstrap_hit_rate_ci          — recording-resampling 95% CI
    permutation_pvalues            — permutation p-values for hit-rate
    per_bearing_hit_rate           — breakdown of hit-rate by bearing
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths & package import
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PKG_SRC = _REPO_ROOT / "Mamba-xLSTM" / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

OUT_ROOT = _REPO_ROOT / "Mamba-xLSTM" / "results" / "journal_q2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Import existing pipeline utilities from run_bpfx_mapping.py (loaded as a
# stand-alone module to avoid mucking with sys.path).
_RUN_BPFX_PATH = _REPO_ROOT / "Mamba-xLSTM" / "scripts" / "run_bpfx_mapping.py"
_spec = importlib.util.spec_from_file_location("_run_bpfx_mapping", _RUN_BPFX_PATH)
assert _spec is not None and _spec.loader is not None
_RUN_BPFX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RUN_BPFX)

bpfx_frequencies = _RUN_BPFX.bpfx_frequencies
hilbert_envelope_spectrum = _RUN_BPFX.hilbert_envelope_spectrum
band_amplitude = _RUN_BPFX.band_amplitude

# Start with the PHM2012 and XJTU-SY entries from run_bpfx_mapping.py,
# then extend with IMS and CWRU.
DATASET_INFO: dict = dict(_RUN_BPFX.DATASET_INFO)

# ---------------------------------------------------------------------------
# Auto-discovery of the best Mamba-xLSTM-Net run for a dataset after training
# ---------------------------------------------------------------------------

_RESULTS_RUNS = _REPO_ROOT / "Mamba-xLSTM" / "results" / "runs"


def _auto_run_dir(dataset_key: str) -> Path:
    """Return the most-recent Mamba-xLSTM-Net run directory for ``dataset_key``.

    After running ``vps_ims_cwru_journal_q2.sh``, training produces dirs like:
      results/runs/<timestamp>_algorithm_comparison_<dataset>_mamba_xlstm_net_s42/

    Falls back to a placeholder path if no matching directory exists yet.
    """
    pattern = f"*_algorithm_comparison_{dataset_key}_mamba_xlstm_net_s42"
    candidates = sorted(_RESULTS_RUNS.glob(pattern)) if _RESULTS_RUNS.exists() else []
    if candidates:
        return candidates[-1]   # most recent
    # Fallback: placeholder that causes run_stats.py to skip gracefully
    return _RESULTS_RUNS / f"_PLACEHOLDER_{dataset_key}_mamba_xlstm_net_s42"


def _auto_sae_pt(dataset_key: str) -> Path:
    """Return the SAE checkpoint path inside the best Mamba-xLSTM-Net run dir."""
    run_dir = _auto_run_dir(dataset_key)
    return run_dir / "explain" / "sae.pt"


# ---------------------------------------------------------------------------
# IMS bearing geometry (Rexnord ZA-2115, running at 2000 rpm = 33.33 Hz)
# ---------------------------------------------------------------------------
_IMS_ROOT = _REPO_ROOT / "data-bearing" / "IMS"
_IMS_CHANNEL_MAP = {
    "ims_b1": (0, 1),   # bearing 1: columns 0,1
    "ims_b2": (2, 3),   # bearing 2: columns 2,3
    "ims_b3": (4, 5),   # bearing 3: columns 4,5
    "ims_b4": (6, 7),   # bearing 4: columns 6,7
}
_IMS_GEOM = dict(n=16, d=8.41, D=71.50, theta_deg=0.0)
_IMS_FS = 20_480

DATASET_INFO["ims"] = {
    "bearing_geom": _IMS_GEOM,
    "fr_hz": 33.33,
    "fs_hz": float(_IMS_FS),
    "raw_data_root": _IMS_ROOT,
    "train_bearing_dirs": list(_IMS_CHANNEL_MAP.keys()),
    "sae_pt": _auto_sae_pt("ims"),
    "run_dir": _auto_run_dir("ims"),
}

# ---------------------------------------------------------------------------
# CWRU bearing geometry (SKF 6205-2RS, running at ~1797 rpm = 29.95 Hz)
# ---------------------------------------------------------------------------
_CWRU_ROOT = _REPO_ROOT / "data-bearing" / "cwru"
_CWRU_GEOM = dict(n=9, d=7.94, D=39.04, theta_deg=0.0)
_CWRU_FS = 48_000

# Preferred (48 kHz, load=0) subset of files for BPFx amplitude scan.
_CWRU_SUBSET = [
    "IR007_0_109", "IR014_0_174", "IR021_0_213",
    "B007_0_122", "B014_0_189", "B021_0_226",
    "OR007_6_3_138", "OR014_6_3_204", "OR021_6_3_241",
    "Time_Normal_0_097",
]

DATASET_INFO["cwru"] = {
    "bearing_geom": _CWRU_GEOM,
    "fr_hz": 29.95,
    "fs_hz": float(_CWRU_FS),
    "raw_data_root": _CWRU_ROOT,
    "train_bearing_dirs": [
        stem for stem in _CWRU_SUBSET
        if (_CWRU_ROOT / f"{stem}.mat").exists()
    ],
    "sae_pt": _auto_sae_pt("cwru"),
    "run_dir": _auto_run_dir("cwru"),
}


# ---------------------------------------------------------------------------
# IMS raw recording loader (one float32 array per timestamp file)
# ---------------------------------------------------------------------------

def _load_ims_recordings(root: Path, bearing_key: str) -> list[np.ndarray]:
    """Return a list of 1-D arrays (one per acquisition file) for one IMS bearing.

    ``bearing_key`` must be one of ``_IMS_CHANNEL_MAP`` keys (``"ims_b1"`` etc.).
    """
    ch_idx = _IMS_CHANNEL_MAP[bearing_key]
    skip_suffixes = {".zip", ".pdf", ".txt", ".md", ".json", ".parquet"}
    files = sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() not in skip_suffixes],
        key=lambda p: p.name,
    )
    recordings: list[np.ndarray] = []
    for f in files:
        try:
            data = np.loadtxt(f, dtype=np.float32)
            if data.ndim == 1 or data.shape[1] < max(ch_idx) + 1:
                continue
            recordings.append(data[:, ch_idx[0]].astype(np.float32))
        except Exception:
            continue
    return recordings


# ---------------------------------------------------------------------------
# CWRU raw recording loader (one array per .mat file)
# ---------------------------------------------------------------------------

def _load_cwru_recordings(root: Path, bearing_dirs: list[str]) -> list[np.ndarray]:
    """Load each listed .mat file and return its DE_time channel as a 1-D float32 array."""
    try:
        import scipy.io
    except ImportError:
        return []

    recordings: list[np.ndarray] = []
    for stem in bearing_dirs:
        mat_path = root / f"{stem}.mat"
        if not mat_path.exists():
            continue
        try:
            mat = scipy.io.loadmat(str(mat_path))
        except Exception:
            continue
        de_key = next(
            (k for k in mat if "DE_time" in k or (k.endswith("DE") and not k.startswith("_"))),
            None,
        )
        if de_key is None:
            de_key = max(
                (k for k in mat if not k.startswith("_")),
                key=lambda k: mat[k].size if isinstance(mat[k], np.ndarray) else 0,
                default=None,
            )
        if de_key is None:
            continue
        recordings.append(mat[de_key].squeeze().astype(np.float32))
    return recordings


# ---------------------------------------------------------------------------
# Raw recording --> BPFx band amplitudes (per recording, with bearing labels)
# ---------------------------------------------------------------------------

def gather_bpfx_amplitudes(
    dataset_key: str,
    *,
    bpfx_bw_hz: float = 2.0,
    max_recordings: int = 300,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    """Walk the raw-data root for ``dataset_key`` and compute per-recording
    BPFx band amplitudes.

    Returns
    -------
    bpfx_amps : (R, K) float32 — band-integrated envelope amplitude per
                BPFx for each recording.
    bearing_idx : (R,) int32 — index into ``bearing_names`` for each recording.
    bearing_names : list of str — train_bearing_dirs encountered.
    freqs_hz : dict[str, float] — characteristic frequencies in Hz.
    """
    info = DATASET_INFO[dataset_key]
    freqs_hz = bpfx_frequencies(**info["bearing_geom"], fr=info["fr_hz"])
    bpfx_names = list(freqs_hz.keys())

    raw_root = info["raw_data_root"]
    fs = float(info["fs_hz"])

    # Dispatch to the appropriate raw-signal loader.
    if dataset_key == "ims":
        return _gather_ims_amplitudes(
            raw_root, info["train_bearing_dirs"], freqs_hz, bpfx_names,
            fs=fs, bpfx_bw_hz=bpfx_bw_hz, max_recordings=max_recordings,
        )
    if dataset_key == "cwru":
        return _gather_cwru_amplitudes(
            raw_root, info["train_bearing_dirs"], freqs_hz, bpfx_names,
            fs=fs, bpfx_bw_hz=bpfx_bw_hz,
        )

    # PHM2012 / XJTU-SY — original code path.
    loader_fn = (
        _RUN_BPFX.load_phm2012_recordings if dataset_key == "phm2012"
        else _RUN_BPFX.load_xjtusy_recordings
    )

    rows: list[list[float]] = []
    bearings: list[int] = []
    bearing_names: list[str] = []
    rec_count = 0

    for b_idx, b_name in enumerate(info["train_bearing_dirs"]):
        b_dir = raw_root / b_name
        if not b_dir.exists():
            continue
        bearing_names.append(b_name)
        for rec in loader_fn(b_dir):
            if rec_count >= max_recordings:
                break
            env_freqs, env_spec = hilbert_envelope_spectrum(rec, fs)
            row = [
                band_amplitude(env_freqs, env_spec, freqs_hz[name], bw=bpfx_bw_hz)
                for name in bpfx_names
            ]
            rows.append(row)
            bearings.append(len(bearing_names) - 1)
            rec_count += 1
        if rec_count >= max_recordings:
            break

    bpfx_amps = np.asarray(rows, dtype=np.float32)
    bearing_idx = np.asarray(bearings, dtype=np.int32)
    return bpfx_amps, bearing_idx, bearing_names, freqs_hz


def _gather_ims_amplitudes(
    root: Path,
    bearing_keys: list[str],
    freqs_hz: dict[str, float],
    bpfx_names: list[str],
    *,
    fs: float,
    bpfx_bw_hz: float,
    max_recordings: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    rows: list[list[float]] = []
    bearings: list[int] = []
    bearing_names: list[str] = []
    rec_count = 0

    for b_idx, b_key in enumerate(bearing_keys):
        recs = _load_ims_recordings(root, b_key)
        bearing_names.append(b_key)
        for rec in recs:
            if rec_count >= max_recordings:
                break
            env_freqs, env_spec = hilbert_envelope_spectrum(rec, fs)
            row = [
                band_amplitude(env_freqs, env_spec, freqs_hz[name], bw=bpfx_bw_hz)
                for name in bpfx_names
            ]
            rows.append(row)
            bearings.append(b_idx)
            rec_count += 1
        if rec_count >= max_recordings:
            break

    bpfx_amps = np.asarray(rows, dtype=np.float32)
    bearing_idx = np.asarray(bearings, dtype=np.int32)
    return bpfx_amps, bearing_idx, bearing_names, freqs_hz


def _gather_cwru_amplitudes(
    root: Path,
    bearing_dirs: list[str],
    freqs_hz: dict[str, float],
    bpfx_names: list[str],
    *,
    fs: float,
    bpfx_bw_hz: float,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    recs = _load_cwru_recordings(root, bearing_dirs)
    rows: list[list[float]] = []
    bearings: list[int] = []

    for b_idx, rec in enumerate(recs):
        env_freqs, env_spec = hilbert_envelope_spectrum(rec, fs)
        row = [
            band_amplitude(env_freqs, env_spec, freqs_hz[name], bw=bpfx_bw_hz)
            for name in bpfx_names
        ]
        rows.append(row)
        bearings.append(b_idx)

    bpfx_amps = np.asarray(rows, dtype=np.float32)
    bearing_idx = np.asarray(bearings, dtype=np.int32)
    return bpfx_amps, bearing_idx, bearing_dirs, freqs_hz


# ---------------------------------------------------------------------------
# Hidden-state encoding through a (possibly externally-supplied) SAE
# ---------------------------------------------------------------------------

def encode_with_sae(
    sae,
    hidden: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Encode (N, d_model) hidden states through a Top-k SAE; return (N, d_latent)."""
    sae = sae.to(device).eval()
    with torch.no_grad():
        z = sae.encode(torch.from_numpy(hidden.astype(np.float32)).to(device))
        z_sparse = sae.topk(z).cpu().numpy()
    return z_sparse


# ---------------------------------------------------------------------------
# Correlation, hit-rate, and statistical inference
# ---------------------------------------------------------------------------

def align_and_correlate(
    z: np.ndarray, amps: np.ndarray
) -> np.ndarray:
    """Pearson r between every SAE feature and every BPFx column.

    z    : (N, d_latent)
    amps : (M, K)

    Output: (d_latent, K) correlation matrix; aligned to min(N, M) samples.
    """
    n = min(z.shape[0], amps.shape[0])
    z = z[:n]
    amps = amps[:n]
    d_lat = z.shape[1]
    k = amps.shape[1]
    corrs = np.zeros((d_lat, k), dtype=np.float32)
    z_std = z.std(axis=0)
    a_std = amps.std(axis=0)
    for fi in range(d_lat):
        if z_std[fi] < 1e-8:
            continue
        for ai in range(k):
            if a_std[ai] < 1e-8:
                continue
            r = float(np.corrcoef(z[:, fi], amps[:, ai])[0, 1])
            corrs[fi, ai] = r if not np.isnan(r) else 0.0
    return corrs


def hit_rate_from_corrs(
    corrs: np.ndarray, threshold: float = 0.30
) -> np.ndarray:
    """Per-BPFx hit-rate (fraction of features with |r| >= threshold)."""
    return (np.abs(corrs) >= threshold).mean(axis=0)


def bootstrap_hit_rate_ci(
    z: np.ndarray,
    amps: np.ndarray,
    *,
    threshold: float = 0.30,
    n_boot: int = 1_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Bootstrap (1 - alpha) CI for the per-BPFx hit-rate.

    Resamples *recordings* (not features) with replacement; recomputes
    correlations and hit-rate ``n_boot`` times.

    Returns a dict with keys:
      'point'  : (K,) observed hit-rate
      'low'    : (K,) lower quantile
      'high'   : (K,) upper quantile
      'samples': (n_boot, K) full bootstrap distribution
    """
    rng = rng or np.random.default_rng(0)
    n = min(z.shape[0], amps.shape[0])
    z = z[:n]
    amps = amps[:n]
    point = hit_rate_from_corrs(align_and_correlate(z, amps), threshold)

    boot = np.zeros((n_boot, amps.shape[1]), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = hit_rate_from_corrs(
            align_and_correlate(z[idx], amps[idx]), threshold
        )
    low = np.quantile(boot, alpha / 2, axis=0)
    high = np.quantile(boot, 1 - alpha / 2, axis=0)
    return {
        "point": point,
        "low": low,
        "high": high,
        "samples": boot,
    }


def permutation_pvalues(
    z: np.ndarray,
    amps: np.ndarray,
    *,
    threshold: float = 0.30,
    n_perm: int = 1_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Permutation test: shuffle recording order of ``amps`` and recompute
    hit-rate ``n_perm`` times to build a null distribution; report a
    one-sided (greater) p-value.

    Returns
    -------
    dict with keys:
      'observed' : (K,) observed hit-rate
      'null'     : (n_perm, K) null hit-rate distribution
      'p_value'  : (K,) fraction of null samples >= observed
    """
    rng = rng or np.random.default_rng(0)
    n = min(z.shape[0], amps.shape[0])
    z = z[:n]
    amps = amps[:n]
    observed = hit_rate_from_corrs(align_and_correlate(z, amps), threshold)
    null = np.zeros((n_perm, amps.shape[1]), dtype=np.float32)
    for p in range(n_perm):
        perm = rng.permutation(n)
        null[p] = hit_rate_from_corrs(
            align_and_correlate(z, amps[perm]), threshold
        )
    p_value = ((null >= observed[None, :]).sum(axis=0) + 1) / (n_perm + 1)
    return {"observed": observed, "null": null, "p_value": p_value}


def per_bearing_hit_rate(
    z: np.ndarray,
    amps: np.ndarray,
    bearing_idx: np.ndarray,
    bearing_names: list[str],
    *,
    threshold: float = 0.30,
) -> dict[str, np.ndarray]:
    """Hit-rate computed independently for each bearing.

    Returns dict mapping bearing name -> (K,) hit-rate.
    """
    n = min(z.shape[0], amps.shape[0], bearing_idx.shape[0])
    z = z[:n]
    amps = amps[:n]
    bearing_idx = bearing_idx[:n]
    out: dict[str, np.ndarray] = {}
    for bi, bname in enumerate(bearing_names):
        mask = bearing_idx == bi
        if mask.sum() < 5:
            out[bname] = np.full(amps.shape[1], np.nan, dtype=np.float32)
            continue
        out[bname] = hit_rate_from_corrs(
            align_and_correlate(z[mask], amps[mask]), threshold
        )
    return out


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def to_jsonable(obj):
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2))


# ---------------------------------------------------------------------------
# Convenience: load default trained Mamba-xLSTM SAE + hidden states for a dataset
# ---------------------------------------------------------------------------

def load_default_sae(dataset_key: str):
    """Load the default trained Mamba-xLSTM SAE for ``dataset_key``."""
    from mxlstm.interp.sae import load_sae

    sae_pt = DATASET_INFO[dataset_key]["sae_pt"]
    if not sae_pt.exists():
        raise FileNotFoundError(f"Default SAE not found at {sae_pt}")
    return load_sae(sae_pt)


def collect_hidden_for_dataset(
    dataset_key: str,
    *,
    n_max: int = 5_000,
    use_random_init: bool = False,
) -> np.ndarray:
    """Collect ``n_max`` hidden states from the model checkpoint of
    ``dataset_key``. If ``use_random_init`` is True, the model weights are
    re-randomised before extraction (used for the untrained-backbone
    negative control).
    """
    from omegaconf import OmegaConf
    from mxlstm.compute import get_device
    from mxlstm.data.datamodule import RULDataModule
    from mxlstm.training.lit_module import RULLitModule
    from mxlstm.interp.sae import collect_hidden_states
    from mxlstm.interp.explain_extras import resolve_checkpoint

    info = DATASET_INFO[dataset_key]
    cfg_path = info["run_dir"] / "config.yaml"
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

    # Build the model (this gets random init by default).
    train_script = _REPO_ROOT / "Mamba-xLSTM" / "scripts" / "train.py"
    train_spec = importlib.util.spec_from_file_location("_train_build", train_script)
    assert train_spec is not None and train_spec.loader is not None
    train_mod = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train_mod)
    model = train_mod._build_model(
        cfg, n_features=dm.n_features, context_length=int(cfg.data.window_length)
    )

    if use_random_init:
        # Skip checkpoint loading; the model already has random weights.
        lit = RULLitModule(model=model)
    else:
        ckpt = resolve_checkpoint(info["run_dir"])
        lit = RULLitModule.load_from_checkpoint(
            str(ckpt), model=model, map_location=device
        )
    lit = lit.to(device).eval()

    return collect_hidden_states(
        lit.model, dm.train_dataloader(), device=device, layer="fused", max_samples=n_max
    )
