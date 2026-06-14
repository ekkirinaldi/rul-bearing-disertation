"""Dataset → trained run directory mapping and config loading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from app.skf_loader import SKF_STREAMS, list_skf_streams

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MXLSTM = _REPO_ROOT / "Mamba-xLSTM"
_DATA_BEARING = _REPO_ROOT / "data-bearing"

_RUN_DIRS: dict[str, Path] = {
    "phm2012": _MXLSTM / "results/runs/20260515_174110_algorithm_comparison_phm2012_mamba_xlstm_net_s42",
    "xjtusy": _MXLSTM / "results/runs/20260611_104213_algorithm_comparison_xjtusy_mamba_xlstm_net_s42",
}

_DATASET_LABELS: dict[str, str] = {
    "phm2012": "PHM2012 (FEMTO-PRONOSTIA)",
    "xjtusy": "XJTU-SY",
    "skf_ch15_or1": "PT SKF Indonesia — CH-15 OR-1 (real plant)",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    run_dir: Path
    data_root: Path
    cache_dir: Path
    window_length: int
    stride_eval: int
    n_bands: int
    smoothing_alpha: float
    test_bearings: list[str]
    acquisition_interval_s: float
    fs: int
    checkpoint: Path
    hi_scaler_path: Path
    model_name: str
    inference_model_key: str
    has_gt_rul: bool
    stream_mode: str
    transfer_note: str | None = None


def resolve_checkpoint(run_dir: Path) -> Path:
    """Resolve best checkpoint; summary.json may contain a VPS absolute path."""
    ckpt_dir = run_dir / "checkpoints"
    summ_path = run_dir / "summary.json"
    if summ_path.exists():
        try:
            best = json.loads(summ_path.read_text()).get("best_checkpoint") or ""
            if best:
                name = Path(str(best)).name
                local = ckpt_dir / name
                if local.is_file():
                    return local
        except json.JSONDecodeError:
            pass
    cks = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cks:
        raise FileNotFoundError(f"No checkpoints under {ckpt_dir}")
    numbered = [c for c in cks if c.stem != "last"]
    if numbered:
        return numbered[0]
    return cks[0]


def _resolve_data_path(cfg_path: str | Path) -> Path:
    p = Path(cfg_path)
    if p.is_absolute():
        return p
    return (_MXLSTM / p).resolve()


def _load_benchmark_spec(key: str) -> DatasetSpec:
    run_dir = _RUN_DIRS[key]
    cfg = OmegaConf.load(run_dir / "config.yaml")
    data = cfg.data
    data_root = _resolve_data_path(data.root)
    cache_dir = _resolve_data_path(data.cache_dir)
    interval = 10.0 if key == "phm2012" else 60.0
    return DatasetSpec(
        key=key,
        label=_DATASET_LABELS[key],
        run_dir=run_dir,
        data_root=data_root,
        cache_dir=cache_dir,
        window_length=int(data.window_length),
        stride_eval=int(data.stride_eval),
        n_bands=int(data.n_bands),
        smoothing_alpha=float(data.smoothing_alpha),
        test_bearings=[str(b) for b in data.test_bearings],
        acquisition_interval_s=interval,
        fs=25600,
        checkpoint=resolve_checkpoint(run_dir),
        hi_scaler_path=run_dir / "hi_scaler.json",
        model_name=str(cfg.model.name),
        inference_model_key=key,
        has_gt_rul=True,
        stream_mode="bearing",
        transfer_note=None,
    )


def _load_skf_spec() -> DatasetSpec:
    """Industrial SKF stream uses PHM2012 model + scaler for transfer demo."""
    transfer_key = "phm2012"
    run_dir = _RUN_DIRS[transfer_key]
    cfg = OmegaConf.load(run_dir / "config.yaml")
    data = cfg.data
    streams = list_skf_streams()
    return DatasetSpec(
        key="skf_ch15_or1",
        label=_DATASET_LABELS["skf_ch15_or1"],
        run_dir=run_dir,
        data_root=_DATA_BEARING / "skf-ch15-or1",
        cache_dir=_DATA_BEARING / "processed" / "skf-ch15-or1",
        window_length=int(data.window_length),
        stride_eval=int(data.stride_eval),
        n_bands=int(data.n_bands),
        smoothing_alpha=float(data.smoothing_alpha),
        test_bearings=streams,
        acquisition_interval_s=3600.0,
        fs=25600,
        checkpoint=resolve_checkpoint(run_dir),
        hi_scaler_path=run_dir / "hi_scaler.json",
        model_name=str(cfg.model.name),
        inference_model_key=transfer_key,
        has_gt_rul=True,
        stream_mode="skf_trending",
        transfer_note=(
            "RUL from PHM2012-trained Mamba-xLSTM-Net (transfer demo); "
            "Observer trending → synthetic waveform → HI pipeline. "
            "EOL anchored at the field failure event (~31 Aug 2023, envelope "
            "spike collapse); post-repair September baseline is segmented out."
        ),
    )


def load_dataset_spec(key: str) -> DatasetSpec:
    if key == "skf_ch15_or1":
        return _load_skf_spec()
    if key not in _RUN_DIRS:
        raise KeyError(f"Unknown dataset {key!r}; choose from {list_datasets_keys()}")
    return _load_benchmark_spec(key)


def list_datasets_keys() -> list[str]:
    return list(_RUN_DIRS) + ["skf_ch15_or1"]


def list_datasets() -> list[DatasetSpec]:
    return [load_dataset_spec(k) for k in list_datasets_keys()]


def stream_label(dataset_key: str, stream_id: str) -> str:
    if dataset_key == "skf_ch15_or1":
        return SKF_STREAMS.get(stream_id, {}).get("label", stream_id)
    return f"Bearing {stream_id}"


_TIME_NAMES = ["rms", "peak", "kurtosis", "skewness", "crest", "shape", "impulse", "margin", "variance"]
_FREQ_BASE = ["centroid", "entropy", "mean_freq", "rms_freq"]


def feature_names(spec: DatasetSpec) -> list[str]:
    """Reconstruct HI feature names from the scaler size and band count.

    Layout per channel: 9 time-domain + (4 + n_bands) frequency-domain features.
    """
    try:
        n_features = int(len(json.loads(spec.hi_scaler_path.read_text())["min"]))
    except Exception:
        n_features = 2 * (len(_TIME_NAMES) + len(_FREQ_BASE) + spec.n_bands)
    per_channel = len(_TIME_NAMES) + len(_FREQ_BASE) + spec.n_bands
    n_channels = max(1, n_features // per_channel)
    freq_names = _FREQ_BASE + [f"band{b}" for b in range(spec.n_bands)]
    names: list[str] = []
    for c in range(n_channels):
        names.extend(f"td_c{c}_{n}" for n in _TIME_NAMES)
        names.extend(f"fd_c{c}_{n}" for n in freq_names)
    return names[:n_features] if n_features else names
