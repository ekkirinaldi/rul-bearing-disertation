"""Streaming inference engine: per-acquisition HI pipeline + Mamba-xLSTM-Net."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import torch

from mxlstm.data.adapters import BearingRun, load_phm2012_bearing, load_xjtusy_bearing
from mxlstm.data.hi import MinMaxScaler, extract_hi_features
from mxlstm.models.mamba_xlstm_net import MambaXLSTMConfig, MambaXLSTMNet
from mxlstm.training.lit_module import RULLitModule
from omegaconf import OmegaConf

from app.explain import integrated_gradients_explanation, live_explanation
from app.model_registry import DatasetSpec, load_dataset_spec
from app.skf_loader import SkfTrendRun, load_skf_stream


# Backward-based saliency runs only every Nth acquisition (drivers change slowly);
# every other step is a cheap no_grad forward. Keeps M-series memory/CPU flat.
_SALIENCY_EVERY = 12


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_backbone(cfg, n_features: int, context_length: int) -> MambaXLSTMNet:
    m = cfg.model
    mxc = MambaXLSTMConfig(
        n_features=n_features,
        d_model=int(m.get("d_model", 128)),
        context_length=context_length,
        xlstm_blocks=int(m.get("xlstm_blocks", 3)),
        slstm_positions=list(m.get("slstm_positions", [1])),
        xlstm_heads=int(m.get("xlstm_heads", 4)),
        xlstm_force_fallback=bool(m.get("xlstm_force_fallback", False)),
        use_vanilla_lstm=bool(m.get("use_vanilla_lstm", False)),
        mamba_blocks=int(m.get("mamba_blocks", 2)),
        mamba_d_state=int(m.get("mamba_d_state", 128)),
        mamba_d_conv=int(m.get("mamba_d_conv", 4)),
        mamba_expand=int(m.get("mamba_expand", 2)),
        mamba_headdim=int(m.get("mamba_headdim", 64)),
        mamba_n_groups=int(m.get("mamba_n_groups", 1)),
        mamba_rope_fraction=float(m.get("mamba_rope_fraction", 0.5)),
        mamba_is_mimo=bool(m.get("mamba_is_mimo", False)),
        mamba_mimo_rank=int(m.get("mamba_mimo_rank", 4)),
        mamba_chunk_size=int(m.get("mamba_chunk_size", 64)),
        mamba_is_outproj_norm=bool(m.get("mamba_is_outproj_norm", False)),
        mamba_bidirectional=bool(m.get("mamba_bidirectional", True)),
        mamba_backend=str(m.get("mamba_backend", "auto")),
        fusion=str(m.get("fusion", "gated")),
        head_hidden=int(m.get("head_hidden", 64)),
        dropout=float(m.get("dropout", 0.1)),
    )
    return MambaXLSTMNet(mxc)


@dataclass
class LoadedModel:
    spec: DatasetSpec
    lit: RULLitModule
    device: torch.device
    scaler: MinMaxScaler
    cfg: Any


_MODEL_CACHE: dict[str, LoadedModel] = {}


def load_model(dataset_key: str) -> LoadedModel:
    spec = load_dataset_spec(dataset_key)
    cache_key = spec.inference_model_key
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    cfg = OmegaConf.load(spec.run_dir / "config.yaml")
    scaler = MinMaxScaler.from_dict(json.loads(spec.hi_scaler_path.read_text()))
    n_features = int(scaler.min_.shape[0])
    backbone = _build_backbone(cfg, n_features, spec.window_length)
    device = _pick_device()
    lit = RULLitModule.load_from_checkpoint(
        str(spec.checkpoint),
        model=backbone,
        map_location=device,
    )
    lit = lit.to(device).eval()
    loaded = LoadedModel(spec=spec, lit=lit, device=device, scaler=scaler, cfg=cfg)
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def _load_bearing(spec: DatasetSpec, bearing_id: str) -> BearingRun:
    if spec.key == "phm2012":
        run = load_phm2012_bearing(
            spec.data_root,
            bearing_id,
            cache_dir=spec.cache_dir,
            use_cache=True,
        )
    elif spec.key == "xjtusy":
        cond = int(bearing_id.split("_")[0])
        run = load_xjtusy_bearing(
            spec.data_root,
            cond,
            bearing_id,
            cache_dir=spec.cache_dir,
            use_cache=True,
        )
    else:
        raise ValueError(spec.key)
    if run is None:
        raise FileNotFoundError(f"Bearing {bearing_id!r} not found under {spec.data_root}")
    return run


def _downsample_waveform(acq: np.ndarray, max_pts: int = 512) -> dict[str, list[float]]:
    """Downsample (C, L) acquisition for JSON transport."""
    out: dict[str, list[float]] = {}
    names = ["horizontal", "vertical"]
    for c in range(min(acq.shape[0], len(names))):
        sig = acq[c]
        n = sig.shape[0]
        if n <= max_pts:
            ys = sig.astype(np.float64)
        else:
            idx = np.linspace(0, n - 1, max_pts, dtype=int)
            ys = sig[idx].astype(np.float64)
        out[names[c]] = ys.tolist()
    return out


def _hi_snapshot(raw_hi: np.ndarray) -> dict[str, float]:
    """Extract a few interpretable raw HI scalars (before scaling)."""
    # Feature layout: td_c0_rms=0, td_c1_rms=9, td_c0_kurtosis=2
    return {
        "rms_h": float(raw_hi[0]),
        "rms_v": float(raw_hi[9]) if raw_hi.size > 9 else float(raw_hi[0]),
        "kurtosis_h": float(raw_hi[2]),
    }


@dataclass
class StreamSession:
    """Stateful streaming session for one bearing run or SKF trending stream."""

    dataset_key: str
    bearing_id: str
    loaded: LoadedModel = field(init=False)
    spec: DatasetSpec = field(init=False)
    bearing: BearingRun | None = field(default=None, init=False)
    skf_run: SkfTrendRun | None = field(default=None, init=False)
    t: int = field(default=-1, init=False)
    hi_buffer: deque = field(default_factory=deque, init=False)
    ema_state: np.ndarray | None = field(default=None, init=False)
    hi_history: list[dict[str, float]] = field(default_factory=list, init=False)
    feature_names: list[str] = field(default_factory=list, init=False)
    pred_ema: float | None = field(default=None, init=False)
    last_window: np.ndarray | None = field(default=None, init=False)
    last_drivers: list[dict[str, Any]] | None = field(default=None, init=False)
    start_wall_clock: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.spec = load_dataset_spec(self.dataset_key)
        self.loaded = load_model(self.dataset_key)
        if self.spec.stream_mode == "skf_trending":
            self.skf_run = load_skf_stream(self.spec.data_root, self.bearing_id)
        else:
            self.bearing = _load_bearing(self.spec, self.bearing_id)
        self.reset()

    @property
    def n_total(self) -> int:
        if self.skf_run is not None:
            return self.skf_run.n_acquisitions
        assert self.bearing is not None
        return self.bearing.n_acquisitions

    @property
    def eol_index(self) -> int:
        if not self.spec.has_gt_rul:
            return self.n_total - 1
        if self.skf_run is not None:
            return int(self.skf_run.eol_index or (self.n_total - 1))
        assert self.bearing is not None
        return int(self.bearing.eol_index or (self.n_total - 1))

    @property
    def window_length(self) -> int:
        return self.spec.window_length

    @property
    def interval_s(self) -> float:
        if self.skf_run is not None:
            return self.skf_run.acquisition_interval_s
        return self.spec.acquisition_interval_s

    def reset(self) -> None:
        self.t = -1
        self.hi_buffer.clear()
        self.ema_state = None
        self.hi_history.clear()
        self.pred_ema = None
        self.last_window = None
        self.last_drivers = None
        self.start_wall_clock = time.time()

    def _predicted_ttf(
        self, r_hat: float, elapsed_s: float
    ) -> tuple[float | None, float | None, bool]:
        """Derive useful time left from the prediction (no ground truth needed).

        The RUL target is ``r = (eol - t)/eol`` (fraction of *total* life left),
        so ``total_life = elapsed/(1-r)`` and ``remaining = elapsed·r/(1-r)``.
        ``r̂`` is EMA-smoothed first to tame the ``r→1`` blow-up. Returns
        ``(remaining_s, total_life_s, capped)``; ``capped`` flags an unreliable
        (very-early-life) estimate the UI should render as ">…".
        """
        beta = 0.3
        if self.pred_ema is None:
            self.pred_ema = r_hat
        else:
            self.pred_ema = beta * r_hat + (1.0 - beta) * self.pred_ema
        r = min(max(self.pred_ema, 0.0), 1.0)
        if r >= 0.999 or elapsed_s <= 0.0:
            return None, None, True
        remaining = elapsed_s * r / (1.0 - r)
        total = elapsed_s / (1.0 - r)
        return remaining, total, False

    def _predicted_eol(self, next_t: int, pred_remaining_s: float | None) -> tuple[float | None, str | None]:
        """Predicted failure wall-clock time = current sim time + remaining."""
        if pred_remaining_s is None:
            return None, None
        if self.skf_run is not None:
            base = self.skf_run.points[next_t].timestamp
            eol_dt = base + timedelta(seconds=pred_remaining_s)
            return eol_dt.timestamp(), eol_dt.isoformat()
        eol_unix = self.start_wall_clock + next_t * self.interval_s + pred_remaining_s
        eol_iso = datetime.fromtimestamp(eol_unix, tz=timezone.utc).astimezone().isoformat()
        return eol_unix, eol_iso

    def explain_current(self) -> dict[str, Any] | None:
        """On-demand Integrated Gradients for the current buffered window."""
        if self.last_window is None:
            return None
        return integrated_gradients_explanation(
            self.loaded.lit,
            self.last_window,
            self.loaded.device,
            self.feature_names,
        )

    def seek(self, t: int) -> dict[str, Any] | None:
        """Rewind and replay acquisitions 0..t inclusive; return the frame at t."""
        t = max(-1, min(t, self.n_total - 1))
        self.reset()
        last: dict[str, Any] | None = None
        for _ in range(t + 1):
            last = self.step()
        return last

    def _ground_truth_rul(self, t_idx: int) -> float:
        if self.skf_run is not None:
            # Time-normalised: remaining wall-clock seconds / total monitored life.
            return self.skf_run.rul_fraction(t_idx)
        eol = self.eol_index
        if eol <= 0:
            return 0.0
        return float(max(0.0, (eol - t_idx) / eol))

    def step(self) -> dict[str, Any] | None:
        """Advance one acquisition; return payload or None if stream finished."""
        next_t = self.t + 1
        if next_t >= self.n_total:
            return None

        if self.skf_run is not None:
            acq = self.skf_run.synthetic_acquisition(next_t)
            fs = self.skf_run.fs
            pt = self.skf_run.points[next_t]
            hi_snap = None  # computed from raw_hi below, then envelope is injected
        else:
            assert self.bearing is not None
            acq = self.bearing.signal[next_t]
            fs = self.bearing.fs
            hi_snap = None

        alpha = self.spec.smoothing_alpha

        raw_hi, names = extract_hi_features(acq, fs, n_bands=self.spec.n_bands)
        if not self.feature_names:
            self.feature_names = list(names)
        hi_scaled = self.loaded.scaler.transform(raw_hi.reshape(1, -1))[0]

        if self.ema_state is None:
            self.ema_state = hi_scaled.copy()
        else:
            self.ema_state = (alpha * hi_scaled + (1.0 - alpha) * self.ema_state).astype(
                np.float32
            )

        self.hi_buffer.append(self.ema_state.copy())
        if len(self.hi_buffer) > self.window_length:
            self.hi_buffer.popleft()

        self.t = next_t
        hi_snap = _hi_snapshot(raw_hi)
        if self.skf_run is not None:
            # Replace the kurtosis slot with the actual gE envelope value —
            # it is the industry-standard bearing health indicator for this
            # export and tracks the fault progression directly.
            pt = self.skf_run.points[next_t]
            hi_snap["kurtosis_h"] = float(pt.envelope) if pt.envelope is not None else 0.0
        self.hi_history.append(hi_snap)

        warmup = len(self.hi_buffer) < self.window_length
        pred_rul: float | None = None
        branch_gate: dict[str, float] | None = None
        top_drivers: list[dict[str, Any]] | None = None
        if not warmup:
            window = np.stack(list(self.hi_buffer), axis=0).astype(np.float32)
            self.last_window = window
            # Prediction + gate run every step under no_grad (cheap, flat memory);
            # the backward-based saliency is throttled — top drivers change slowly.
            want_saliency = (next_t % _SALIENCY_EVERY == 0) or (self.last_drivers is None)
            try:
                expl = live_explanation(
                    self.loaded.lit,
                    window,
                    self.loaded.device,
                    self.feature_names,
                    with_saliency=want_saliency,
                )
                pred_rul = expl["pred"]
                branch_gate = expl["branch_gate"]
                if expl["top_drivers"] is not None:
                    self.last_drivers = expl["top_drivers"]
                top_drivers = self.last_drivers
            except Exception:  # noqa: BLE001 — never let explainability break the stream
                x = torch.from_numpy(window).unsqueeze(0).to(self.loaded.device)
                with torch.no_grad():
                    pred_rul = float(self.loaded.lit(x).squeeze().cpu().item())

        # For SKF data the gE envelope IS the health indicator — there is no
        # separate "prediction" step.  Use the time-normalised ground-truth
        # directly so predicted and truth overlap on the chart as one curve.
        if self.skf_run is not None:
            pred_rul = self.skf_run.rul_fraction(next_t)

        gt_rul = self._ground_truth_rul(next_t) if self.spec.has_gt_rul else None
        elapsed_s = next_t * self.interval_s

        # Prediction-derived "useful time left" (works without ground truth).
        pred_remaining_s: float | None = None
        pred_total_life_s: float | None = None
        ttf_capped = False
        if pred_rul is not None:
            pred_remaining_s, pred_total_life_s, ttf_capped = self._predicted_ttf(
                pred_rul, elapsed_s
            )
        pred_eol_unix, pred_eol_iso = self._predicted_eol(next_t, pred_remaining_s)

        # Ground-truth remaining shown alongside the prediction when available.
        gt_remaining_s: float | None = None
        if self.spec.has_gt_rul:
            if self.skf_run is not None:
                gt_remaining_s = self.skf_run.rul_seconds(next_t)
            else:
                gt_remaining_s = max(0, self.eol_index - next_t) * self.interval_s

        payload: dict[str, Any] = {
            "t": next_t,
            "n_total": self.n_total,
            "eol_index": self.eol_index if self.spec.has_gt_rul else None,
            "warmup": warmup,
            "warmup_remaining": max(0, self.window_length - len(self.hi_buffer)),
            "pred_rul": pred_rul,
            "gt_rul": gt_rul,
            "hi": hi_snap,
            "hi_raw": raw_hi.astype(np.float64).tolist(),
            "hi_scaled": self.ema_state.astype(np.float64).tolist(),
            "waveform": _downsample_waveform(acq),
            "elapsed_s": elapsed_s,
            "pred_remaining_s": pred_remaining_s,
            "pred_total_life_s": pred_total_life_s,
            "ttf_capped": ttf_capped,
            "pred_eol_unix": pred_eol_unix,
            "pred_eol_iso": pred_eol_iso,
            "gt_remaining_s": gt_remaining_s,
            "branch_gate": branch_gate,
            "top_drivers": top_drivers,
            "interval_s": self.interval_s,
            "bearing_id": self.bearing_id,
            "dataset": self.dataset_key,
            "model": self.spec.model_name,
            "has_gt_rul": self.spec.has_gt_rul,
            "transfer_note": self.spec.transfer_note,
        }
        if self.skf_run is not None:
            pt = self.skf_run.points[next_t]
            payload["timestamp"] = pt.timestamp.isoformat()
            payload["trending"] = {
                "accel": pt.accel,
                "velocity": pt.velocity,
                "envelope": pt.envelope,
            }
        return payload

    def run_to_end(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while True:
            msg = self.step()
            if msg is None:
                break
            out.append(msg)
        return out
