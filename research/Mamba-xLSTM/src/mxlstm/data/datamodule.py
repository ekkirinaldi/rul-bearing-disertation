"""LightningDataModule for HI-based RUL training.

Pipeline:
  1. Load ``BearingRun`` objects via ``mxlstm.data.adapters.load_dataset``.
  2. For each bearing: compute HI sequence (T, F) and RUL labels (T,).
  3. Fit the HI MinMax scaler on **train bearings only** (no leakage).
  4. Smooth HI per bearing.
  5. Slide windows of length L (stride=1 train / L/2 eval) → ``(L, F)`` tensors
     paired with the RUL of the *last* timestep.

The split is **strictly by bearing_id** per §11 #1 of PROJECT_PLAN.md.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from mxlstm.data.adapters import BearingRun, load_dataset
from mxlstm.data.hi import HIPipeline
from mxlstm.data.labels import (
    LabelScheme,
    detect_degradation_onset,
    detect_degradation_onset_liu2026,
    make_rul_labels,
)
from mxlstm.data.liu2026_phm import Liu2026PhmHiPipeline
from mxlstm.data.liu2026_xjtu import Liu2026XjtuHiPipeline
from mxlstm.utils.logging import logger


# ---------------------------------------------------------------------------
# Per-bearing HI cache (kept in memory; persisted to npz for re-runs)
# ---------------------------------------------------------------------------


@dataclass
class BearingHI:
    bearing_id: str
    condition: int
    dataset: str
    hi_raw: np.ndarray            # (T, F) before scaling/smoothing
    hi: np.ndarray                # (T, F) after scaler+smoothing
    rul: np.ndarray               # (T,)
    fs: int
    acquisition_interval_s: float
    eol_index: int
    feature_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Window dataset
# ---------------------------------------------------------------------------


class HIWindowDataset(Dataset):
    """Sliding-window dataset over a list of pre-computed BearingHI objects.

    Each item: ``(x, y, rul_window, meta)`` where
      * ``x`` : float32 tensor (L, F) — last L timesteps' HI
      * ``y`` : float32 tensor scalar — RUL at the last timestep (same as ``rul_window[-1]``)
      * ``rul_window`` : float32 tensor (L,) — RUL labels aligned with each row of ``x``
      * ``meta`` : dict with ``bearing_id`` and ``t_index`` (for plots/debug)
    """

    def __init__(
        self,
        bearings: list[BearingHI],
        *,
        window_length: int,
        stride: int = 1,
    ) -> None:
        self.window_length = int(window_length)
        self.stride = int(stride)
        self._index: list[tuple[int, int]] = []  # (bearing_idx, t_end)
        self.bearings = bearings
        for bi, b in enumerate(bearings):
            T = b.hi.shape[0]
            if T < self.window_length:
                continue
            for t_end in range(self.window_length - 1, T, self.stride):
                self._index.append((bi, t_end))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        bi, t_end = self._index[idx]
        b = self.bearings[bi]
        t_start = t_end - self.window_length + 1
        x = b.hi[t_start : t_end + 1]  # (L, F)
        rul_window = b.rul[t_start : t_end + 1].astype(np.float32)
        y = b.rul[t_end]
        meta = {"bearing_id": b.bearing_id, "t_index": int(t_end), "condition": b.condition}
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.tensor(float(y), dtype=torch.float32),
            torch.from_numpy(rul_window),
            meta,
        )


def _collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    rul_windows = torch.stack([b[2] for b in batch], dim=0)
    metas = [b[3] for b in batch]
    return xs, ys, rul_windows, metas


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class RULDataModule(pl.LightningDataModule):
    """Bearing-RUL DataModule.

    Args:
        dataset: ``"phm2012"`` or ``"xjtusy"``.
        root: Path to raw dataset directory.
        train_bearings, val_bearings, test_bearings: Lists of bearing IDs (e.g. ``"1_1"``).
        window_length: Window length L.
        stride_train, stride_eval: Window strides.
        label_scheme: ``"linear"``, ``"piecewise"``, or ``"piecewise_liu2026"`` (Liu et al. 2026).
        smoothing_alpha: EMA coefficient α for HI smoothing (0 disables).
        n_bands: Frequency bands in HI extraction (default pipeline only).
        hi_pipeline: ``"default"`` (plan §6.1), ``"liu2026_phm"`` (Sensors 2026 PHM2012 track),
        or ``"liu2026_xjtu"`` (Sensors 2026 XJTU-SY track: same §3.2 HI + ISOMAP, plus time index §3.3.1).
        phm_horizontal_only: Use horizontal accelerometer channel only. For PHM2012 this
            matches Liu §3.1.2; for XJTU-SY with ``liu2026_xjtu``, set true for Liu §3.1.1.
        allow_train_val_overlap: Allow val bearing IDs to repeat train IDs (paper has no val).
        batch_size, num_workers: DataLoader settings.
        cache_dir: Optional path to override the BearingRun parquet cache.
    """

    def __init__(
        self,
        dataset: str,
        root: str | Path,
        train_bearings: list[str],
        val_bearings: list[str],
        test_bearings: list[str],
        *,
        window_length: int = 64,
        stride_train: int = 1,
        stride_eval: int = 32,
        label_scheme: LabelScheme = "linear",
        smoothing_alpha: float = 0.1,
        n_bands: int = 5,
        hi_pipeline: str = "default",
        phm_horizontal_only: bool = False,
        allow_train_val_overlap: bool = False,
        batch_size: int = 128,
        num_workers: int = 0,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["root", "cache_dir"])
        self.dataset_name = dataset
        self.root = Path(root)
        self.train_bearings = list(train_bearings)
        self.val_bearings = list(val_bearings)
        self.test_bearings = list(test_bearings)
        self.window_length = int(window_length)
        self.stride_train = int(stride_train)
        self.stride_eval = int(stride_eval)
        self.label_scheme: LabelScheme = label_scheme
        self.smoothing_alpha = float(smoothing_alpha)
        self.n_bands = int(n_bands)
        self.hi_pipeline = str(hi_pipeline)
        self.phm_horizontal_only = bool(phm_horizontal_only)
        self.allow_train_val_overlap = bool(allow_train_val_overlap)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.cache_dir = cache_dir

        if self.hi_pipeline not in ("default", "liu2026_phm", "liu2026_xjtu"):
            raise ValueError(f"Unknown hi_pipeline: {self.hi_pipeline}")
        if self.hi_pipeline == "liu2026_phm" and self.dataset_name != "phm2012":
            raise ValueError("hi_pipeline='liu2026_phm' is only defined for dataset phm2012.")
        if self.hi_pipeline == "liu2026_xjtu" and self.dataset_name not in ("xjtusy", "xjtu_sy", "xjtu-sy"):
            raise ValueError("hi_pipeline='liu2026_xjtu' is only defined for dataset xjtusy.")
        if self.label_scheme == "piecewise_liu2026" and self.hi_pipeline not in ("liu2026_phm", "liu2026_xjtu"):
            raise ValueError(
                "label_scheme='piecewise_liu2026' requires hi_pipeline='liu2026_phm' or 'liu2026_xjtu'."
            )
        if self.hi_pipeline in ("liu2026_phm", "liu2026_xjtu") and self.label_scheme != "piecewise_liu2026":
            raise ValueError(
                f"hi_pipeline='{self.hi_pipeline}' requires label_scheme='piecewise_liu2026'."
            )
        train_set = set(self.train_bearings)
        val_set = set(self.val_bearings)
        test_set = set(self.test_bearings)
        if train_set & test_set:
            raise ValueError("Train and test bearing splits overlap.")
        if val_set & test_set:
            raise ValueError("Val and test bearing splits overlap.")
        if not self.allow_train_val_overlap and (train_set & val_set):
            raise ValueError(
                "Train and val bearing splits overlap. Set allow_train_val_overlap=true "
                "if val mirrors train for Lightning-only monitoring (Liu et al. has no val set)."
            )

        self.pipeline: HIPipeline | None = None
        self.liu_pipeline: Liu2026PhmHiPipeline | Liu2026XjtuHiPipeline | None = None
        self._train_ds: HIWindowDataset | None = None
        self._val_ds: HIWindowDataset | None = None
        self._test_ds: HIWindowDataset | None = None
        self._n_features: int | None = None
        self._cached_scaler_payload: dict[str, Any] | None = None

    # ----- properties -------------------------------------------------------

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("DataModule.setup() must be called first.")
        return self._n_features

    # ----- core pipeline ----------------------------------------------------

    @staticmethod
    def _horizontal_only_run(run: BearingRun) -> BearingRun:
        sig = run.signal[:, 0:1, :].copy()
        return BearingRun(
            dataset=run.dataset,
            condition=run.condition,
            bearing_id=run.bearing_id,
            signal=sig,
            channel_names=["horizontal"],
            fs=run.fs,
            acquisition_interval_s=run.acquisition_interval_s,
            rpm=run.rpm,
            load_N=run.load_N,
            full_life=run.full_life,
            eol_index=run.eol_index,
            metadata=dict(run.metadata),
        )

    def _build_bearings(self, ids: list[str], runs_by_id: dict[str, Any]) -> list[BearingHI]:
        out: list[BearingHI] = []
        for bid in ids:
            run = runs_by_id.get(bid)
            if run is None:
                logger.warning(f"Skipping missing bearing {bid} in {self.dataset_name}.")
                continue
            if self.phm_horizontal_only:
                run = self._horizontal_only_run(run)

            if self.hi_pipeline in ("liu2026_phm", "liu2026_xjtu"):
                assert self.liu_pipeline is not None
                hi_raw = self.liu_pipeline.extract(run.signal)
                hi_pre = self.liu_pipeline.transform_hi_raw(hi_raw)
                T = run.n_acquisitions
                onset = detect_degradation_onset_liu2026(hi_pre[:, 0])
                rul = make_rul_labels(T, scheme="piecewise", degradation_onset=onset)
                hi = self.liu_pipeline.transform_signal(run.signal)
                names = (
                    ["liu_isomap_hi", "time_idx"]
                    if self.hi_pipeline == "liu2026_xjtu"
                    else ["liu_isomap_hi"]
                )
            else:
                assert self.pipeline is not None
                hi_raw = self.pipeline.extract(run.signal)
                hi = self.pipeline.transform_hi(hi_raw)
                T = run.n_acquisitions
                if self.label_scheme == "piecewise":
                    rms_idx = (
                        self.pipeline.feature_names.index("td_c0_rms")
                        if "td_c0_rms" in self.pipeline.feature_names
                        else 0
                    )
                    onset = detect_degradation_onset(hi_raw[:, rms_idx])
                    rul = make_rul_labels(T, scheme="piecewise", degradation_onset=onset)
                elif self.label_scheme == "fault_severity":
                    severity_raw = run.metadata.get("fault_severity", 0)
                    # Normalise: severity in {0,1,2,3} → [0, 1]
                    constant_val = float(severity_raw) / 3.0
                    rul = make_rul_labels(T, scheme="fault_severity", constant_value=constant_val)
                else:
                    rul = make_rul_labels(T, scheme=self.label_scheme, degradation_onset=None)
                names = list(self.pipeline.feature_names)

            out.append(
                BearingHI(
                    bearing_id=run.bearing_id,
                    condition=run.condition,
                    dataset=run.dataset,
                    hi_raw=hi_raw,
                    hi=hi,
                    rul=rul,
                    fs=run.fs,
                    acquisition_interval_s=run.acquisition_interval_s,
                    eol_index=int(run.eol_index),
                    feature_names=names,
                )
            )
        return out

    def _prepared_cache_dir(self) -> Path:
        if self.cache_dir is not None:
            base = Path(self.cache_dir)
        else:
            base = self.root.parent / "processed" / self.dataset_name
        return base / "prepared_windows"

    def _prepared_cache_key(self) -> str:
        payload = {
            "version": 1,
            "dataset": self.dataset_name,
            "root": str(self.root.expanduser().resolve()),
            "cache_dir": str(Path(self.cache_dir).expanduser().resolve()) if self.cache_dir is not None else None,
            "train_bearings": self.train_bearings,
            "val_bearings": self.val_bearings,
            "test_bearings": self.test_bearings,
            "window_length": self.window_length,
            "stride_train": self.stride_train,
            "stride_eval": self.stride_eval,
            "label_scheme": self.label_scheme,
            "smoothing_alpha": self.smoothing_alpha,
            "n_bands": self.n_bands,
            "hi_pipeline": self.hi_pipeline,
            "phm_horizontal_only": self.phm_horizontal_only,
            "allow_train_val_overlap": self.allow_train_val_overlap,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _prepared_cache_path(self) -> Path:
        return self._prepared_cache_dir() / f"{self.dataset_name}_{self._prepared_cache_key()}.pkl"

    def _scaler_payload(self) -> dict[str, Any]:
        if self.hi_pipeline in ("liu2026_phm", "liu2026_xjtu"):
            if self.liu_pipeline is None:
                raise RuntimeError("Liu2026 pipeline not fit yet.")
            return {"kind": self.hi_pipeline, "payload": self.liu_pipeline.to_dict()}
        if self.pipeline is None or self.pipeline.scaler is None:
            raise RuntimeError("Scaler not fit yet.")
        return {"kind": "default", "payload": self.pipeline.scaler.to_dict()}

    def _restore_prepared_cache(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError) as exc:
            logger.warning(f"Ignoring unreadable prepared data cache {path}: {exc}")
            return False

        train_bearings = payload["train_bearings"]
        val_bearings = payload["val_bearings"]
        test_bearings = payload["test_bearings"]
        self._n_features = int(payload["n_features"])
        self._cached_scaler_payload = payload.get("scaler_payload")
        self._train_ds = HIWindowDataset(
            train_bearings, window_length=self.window_length, stride=self.stride_train
        )
        self._val_ds = HIWindowDataset(
            val_bearings, window_length=self.window_length, stride=self.stride_eval
        )
        self._test_ds = HIWindowDataset(
            test_bearings, window_length=self.window_length, stride=self.stride_eval
        )
        logger.info(
            f"DataModule[{self.dataset_name}] restored prepared cache {path.name} "
            f"features={self._n_features} train_windows={len(self._train_ds)} "
            f"val_windows={len(self._val_ds)} test_windows={len(self._test_ds)}"
        )
        return True

    def _save_prepared_cache(
        self,
        path: Path,
        train_bearings: list[BearingHI],
        val_bearings: list[BearingHI],
        test_bearings: list[BearingHI],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        payload = {
            "n_features": self._n_features,
            "scaler_payload": self._scaler_payload(),
            "train_bearings": train_bearings,
            "val_bearings": val_bearings,
            "test_bearings": test_bearings,
        }
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)
        logger.info(f"DataModule[{self.dataset_name}] wrote prepared cache {path}")

    def setup(self, stage: str | None = None) -> None:  # noqa: D401
        """Load runs, fit scaler on train bearings, build window datasets."""
        if self._train_ds is not None and self._val_ds is not None and self._test_ds is not None:
            return  # idempotent

        prepared_cache = self._prepared_cache_path()
        if self._restore_prepared_cache(prepared_cache):
            return

        # Determine which conditions we need; load only what's required.
        needed_ids = set(self.train_bearings + self.val_bearings + self.test_bearings)
        # Some datasets (ims, cwru) use non-numeric bearing IDs; fall back to
        # loading all available bearings when condition extraction fails.
        try:
            needed_conditions: list[int] | None = sorted(
                {int(b.split("_")[0]) for b in needed_ids}
            )
        except ValueError:
            needed_conditions = None
        runs = load_dataset(
            self.dataset_name,
            root=self.root,
            conditions=needed_conditions,
            cache_dir=self.cache_dir,
        )
        runs_by_id = {r.bearing_id: r for r in runs}
        missing = needed_ids - set(runs_by_id)
        if missing:
            logger.warning(f"Bearings requested but not found on disk: {sorted(missing)}")

        # Fit HIPipeline on train bearings only.
        fs = next(iter(runs_by_id.values())).fs

        def _prep_run(bid: str) -> BearingRun | None:
            r = runs_by_id.get(bid)
            if r is None:
                return None
            return self._horizontal_only_run(r) if self.phm_horizontal_only else r

        if self.hi_pipeline == "liu2026_phm":
            self.liu_pipeline = Liu2026PhmHiPipeline(fs=fs, smoothing_alpha=self.smoothing_alpha)
            train_signals = []
            for b in self.train_bearings:
                pr = _prep_run(b)
                if pr is not None:
                    train_signals.append(pr.signal)
            if not train_signals:
                raise RuntimeError("No training bearings found on disk; cannot fit Liu2026 HI pipeline.")
            self.liu_pipeline.fit(train_signals)
            self.pipeline = None
        elif self.hi_pipeline == "liu2026_xjtu":
            self.liu_pipeline = Liu2026XjtuHiPipeline(fs=fs, smoothing_alpha=self.smoothing_alpha)
            train_signals = []
            for b in self.train_bearings:
                pr = _prep_run(b)
                if pr is not None:
                    train_signals.append(pr.signal)
            if not train_signals:
                raise RuntimeError(
                    "No training bearings found on disk; cannot fit Liu2026 XJTU HI pipeline."
                )
            self.liu_pipeline.fit(train_signals)
            self.pipeline = None
        else:
            self.pipeline = HIPipeline(fs=fs, n_bands=self.n_bands, smoothing_alpha=self.smoothing_alpha)
            train_signals = []
            for b in self.train_bearings:
                pr = _prep_run(b)
                if pr is not None:
                    train_signals.append(pr.signal)
            if not train_signals:
                raise RuntimeError("No training bearings found on disk; cannot fit HI scaler.")
            self.pipeline.fit(train_signals)
            self.liu_pipeline = None

        train_bearings = self._build_bearings(self.train_bearings, runs_by_id)
        val_bearings = self._build_bearings(self.val_bearings, runs_by_id)
        test_bearings = self._build_bearings(self.test_bearings, runs_by_id)

        self._n_features = train_bearings[0].hi.shape[1]
        self._train_ds = HIWindowDataset(train_bearings, window_length=self.window_length, stride=self.stride_train)
        self._val_ds = HIWindowDataset(val_bearings, window_length=self.window_length, stride=self.stride_eval)
        self._test_ds = HIWindowDataset(test_bearings, window_length=self.window_length, stride=self.stride_eval)
        self._cached_scaler_payload = self._scaler_payload()
        self._save_prepared_cache(prepared_cache, train_bearings, val_bearings, test_bearings)

        logger.info(
            f"DataModule[{self.dataset_name}] features={self._n_features} "
            f"train_windows={len(self._train_ds)} val_windows={len(self._val_ds)} "
            f"test_windows={len(self._test_ds)}"
        )

    # ----- dataloaders ------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "setup() not called"
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "setup() not called"
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None, "setup() not called"
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    # ----- accessors --------------------------------------------------------

    @property
    def train_bearings_data(self) -> list[BearingHI]:
        assert self._train_ds is not None, "setup() not called"
        return self._train_ds.bearings

    @property
    def val_bearings_data(self) -> list[BearingHI]:
        assert self._val_ds is not None, "setup() not called"
        return self._val_ds.bearings

    @property
    def test_bearings_data(self) -> list[BearingHI]:
        assert self._test_ds is not None, "setup() not called"
        return self._test_ds.bearings

    def save_scaler(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._cached_scaler_payload is not None:
            Path(path).write_text(json.dumps(self._cached_scaler_payload["payload"], indent=2))
            return
        if self.hi_pipeline in ("liu2026_phm", "liu2026_xjtu"):
            if self.liu_pipeline is None:
                raise RuntimeError("Liu2026 pipeline not fit yet.")
            Path(path).write_text(json.dumps(self.liu_pipeline.to_dict(), indent=2))
            return
        if self.pipeline is None or self.pipeline.scaler is None:
            raise RuntimeError("Scaler not fit yet.")
        Path(path).write_text(json.dumps(self.pipeline.scaler.to_dict(), indent=2))
