"""Self-contained PHM2012 and XJTU-SY loaders.

Returns lightweight ``BearingRun`` objects (a local dataclass — *not*
``brul.data.base.BearingRun``) so the package has zero external runtime
dependencies beyond the standard scientific stack.

Both loaders honor the parquet cache layout produced by the historical
``brul`` project (``<cache>/<bid>_<fingerprint>.parquet`` with columns
``t_idx, sample_idx, h, v`` and a sibling ``.meta.json``). When the cache
is missing the loaders read the raw CSV folders directly and persist a
parquet for next time.

PHM2012:
  Raw layout    : ``<root>/{Learning_set, Test_set, Full_Test_Set}/Bearing<C>_<I>/acc_*.csv``
  CSV columns   : hour, minute, second, microsecond, horizontal_g, vertical_g
  Sampling      : 25_600 Hz, 2560 samples per acquisition (= 0.1 s)
  Interval      : 10 s between acquisitions

XJTU-SY:
  Raw layout    : ``<root>/{35Hz12kN, 37.5Hz11kN, 40Hz10kN}/Bearing<C>_<I>/<int>.csv``
  CSV columns   : Horizontal_vibration_signals, Vertical_vibration_signals
  Sampling      : 25_600 Hz, 32_768 samples per acquisition (= 1.28 s)
  Interval      : 60 s between acquisitions
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Local lightweight BearingRun
# ---------------------------------------------------------------------------


@dataclass
class BearingRun:
    """Minimal bearing run-to-failure container used by ``mxlstm.data``."""

    dataset: str
    condition: int
    bearing_id: str
    signal: np.ndarray  # (T, C, L)
    channel_names: list[str]
    fs: int
    acquisition_interval_s: float
    rpm: float = 0.0
    load_N: float = 0.0
    full_life: bool = True
    eol_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signal.ndim != 3:
            raise ValueError(f"signal must be (T, C, L); got {self.signal.shape}")
        if self.signal.shape[1] != len(self.channel_names):
            raise ValueError("channel name count != channel axis size")
        if self.eol_index is None:
            self.eol_index = self.signal.shape[0] - 1
        if self.signal.dtype != np.float32:
            self.signal = self.signal.astype(np.float32)

    @property
    def n_acquisitions(self) -> int:
        return self.signal.shape[0]


# ---------------------------------------------------------------------------
# PHM2012 specifics
# ---------------------------------------------------------------------------


_PHM_FS = 25_600
_PHM_SAMPLES = 2560
_PHM_INTERVAL_S = 10.0
_PHM_CONDITIONS = {
    1: ("Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"),
    2: ("Bearing2_1", "Bearing2_2", "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7"),
    3: ("Bearing3_1", "Bearing3_2", "Bearing3_3"),
}
_PHM_OPERATING = {
    1: {"rpm": 1800.0, "load_N": 4000.0},
    2: {"rpm": 1650.0, "load_N": 4200.0},
    3: {"rpm": 1500.0, "load_N": 5000.0},
}


def _phm_bearing_folder(root: Path, bearing_id: str) -> Path | None:
    """Locate Bearing<C>_<I> across Learning_set / Test_set / Full_Test_Set."""
    folder_name = f"Bearing{bearing_id.replace('_', '_')}"
    for subset in ("Learning_set", "Full_Test_Set", "Test_set"):
        p = root / subset / folder_name
        if p.is_dir():
            return p
    return None


def _phm_fingerprint(folder: Path) -> str:
    files = sorted(folder.glob("acc_*.csv"))
    h = hashlib.sha1()
    for f in files:
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:8]


def _phm_load_from_cache(cache_dir: Path, bearing_id: str, fingerprint: str | None = None) -> tuple[np.ndarray, int] | None:
    """Locate a cached parquet for ``bearing_id``.

    Tries the exact ``<bid>_<fingerprint>.parquet`` name first; if that's
    missing (e.g. fingerprint scheme drift) falls back to globbing
    ``<bid>_*.parquet`` and using the first match. Returns None if no
    cache is present.
    """
    candidates: list[Path] = []
    if fingerprint:
        exact = cache_dir / f"{bearing_id}_{fingerprint}.parquet"
        if exact.exists():
            candidates.append(exact)
    if not candidates:
        candidates = sorted(cache_dir.glob(f"{bearing_id}_*.parquet"))
    if not candidates:
        return None
    pq_path = candidates[0]
    meta_path = pq_path.with_suffix("").with_name(pq_path.stem + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    meta = json.loads(meta_path.read_text())
    t = pq.read_table(pq_path)
    df = t.to_pandas()
    n_acq = int(meta["n_acquisitions"])
    spa = int(meta["samples_per_acquisition"])
    h = df["h"].to_numpy(dtype=np.float32).reshape(n_acq, spa)
    v = df["v"].to_numpy(dtype=np.float32).reshape(n_acq, spa)
    sig = np.stack([h, v], axis=1)  # (T, C, L)
    return sig.astype(np.float32), spa


def _read_csv_robust(path: Path, *, usecols: tuple[int, ...]) -> np.ndarray:
    """Read a 6-column PHM2012 CSV that may use ``,`` or ``;`` as delimiter."""
    with open(path, "rb") as f:
        head = f.read(2048)
    delim = b";" if head.count(b";") > head.count(b",") else b","
    return np.loadtxt(path, delimiter=delim.decode(), dtype=np.float32, usecols=usecols)


def _phm_load_from_raw(folder: Path) -> np.ndarray:
    files = sorted(folder.glob("acc_*.csv"))
    if not files:
        raise FileNotFoundError(f"No acc_*.csv in {folder}")
    out = np.empty((len(files), 2, _PHM_SAMPLES), dtype=np.float32)
    for i, f in enumerate(files):
        try:
            arr = _read_csv_robust(f, usecols=(4, 5))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse {f}: {exc}") from exc
        if arr.shape[0] < _PHM_SAMPLES:
            pad = np.zeros((_PHM_SAMPLES - arr.shape[0], 2), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        out[i, 0] = arr[:_PHM_SAMPLES, 0]
        out[i, 1] = arr[:_PHM_SAMPLES, 1]
    return out


def _phm_write_cache(cache_dir: Path, bearing_id: str, fingerprint: str, sig: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return
    T, _, L = sig.shape
    t_idx = np.repeat(np.arange(T, dtype=np.int32), L)
    sample_idx = np.tile(np.arange(L, dtype=np.int32), T)
    table = pa.table({
        "t_idx": t_idx,
        "sample_idx": sample_idx,
        "h": sig[:, 0].reshape(-1),
        "v": sig[:, 1].reshape(-1),
    })
    pq.write_table(table, cache_dir / f"{bearing_id}_{fingerprint}.parquet")
    (cache_dir / f"{bearing_id}_{fingerprint}.meta.json").write_text(
        json.dumps({"n_acquisitions": T, "samples_per_acquisition": L}, indent=2)
    )


def load_phm2012_bearing(
    root: str | Path,
    bearing_id: str,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> BearingRun | None:
    """Load a single PHM2012 bearing as a ``BearingRun``."""
    root = Path(root)
    folder = _phm_bearing_folder(root, bearing_id)
    if folder is None:
        return None
    cache_dir = Path(cache_dir) if cache_dir else (root.parent / "processed" / "phm2012")
    fp = _phm_fingerprint(folder)

    sig = None
    if use_cache:
        loaded = _phm_load_from_cache(cache_dir, bearing_id, fp)
        if loaded is not None:
            sig, _ = loaded
    if sig is None:
        sig = _phm_load_from_raw(folder)
        if use_cache:
            try:
                _phm_write_cache(cache_dir, bearing_id, fp, sig)
            except Exception:
                pass

    cond = int(bearing_id.split("_")[0])
    op = _PHM_OPERATING.get(cond, {"rpm": 0.0, "load_N": 0.0})
    return BearingRun(
        dataset="phm2012",
        condition=cond,
        bearing_id=bearing_id,
        signal=sig,
        channel_names=["horizontal", "vertical"],
        fs=_PHM_FS,
        acquisition_interval_s=_PHM_INTERVAL_S,
        rpm=op["rpm"],
        load_N=op["load_N"],
        full_life=True,
        metadata={"source_folder": str(folder)},
    )


def load_phm2012(
    root: str | Path,
    conditions: Iterable[int] | None = None,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> list[BearingRun]:
    conditions = sorted(conditions or _PHM_CONDITIONS.keys())
    runs: list[BearingRun] = []
    for cond in conditions:
        for folder_name in _PHM_CONDITIONS[cond]:
            bid = folder_name.replace("Bearing", "")
            run = load_phm2012_bearing(root, bid, cache_dir=cache_dir, use_cache=use_cache)
            if run is not None:
                runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# XJTU-SY specifics
# ---------------------------------------------------------------------------


_XJTU_FS = 25_600
_XJTU_SAMPLES = 32_768
_XJTU_INTERVAL_S = 60.0
_XJTU_FOLDERS = {
    1: "35Hz12kN",
    2: "37.5Hz11kN",
    3: "40Hz10kN",
}
_XJTU_OPERATING = {
    1: {"rpm": 35.0 * 60.0, "load_N": 12_000.0},      # 35 Hz × 60 s = 2100 rpm
    2: {"rpm": 37.5 * 60.0, "load_N": 11_000.0},      # 2250 rpm
    3: {"rpm": 40.0 * 60.0, "load_N": 10_000.0},      # 2400 rpm
}


def _xjtu_fingerprint(folder: Path) -> str:
    """Fingerprint numbered ``*.csv`` files (XJTU layout), not PHM ``acc_*.csv``."""
    def _key(p: Path) -> int:
        m = re.match(r"(\d+)\.csv", p.name)
        return int(m.group(1)) if m else 1 << 30
    files = sorted(folder.glob("*.csv"), key=_key)
    h = hashlib.sha1()
    for f in files:
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:8]


def _xjtu_load_from_raw(folder: Path) -> np.ndarray:
    """Load all numbered csvs in ``folder`` sorted by integer file name."""
    def _key(p: Path) -> int:
        m = re.match(r"(\d+)\.csv", p.name)
        return int(m.group(1)) if m else 1 << 30
    files = sorted(folder.glob("*.csv"), key=_key)
    if not files:
        raise FileNotFoundError(f"No numeric *.csv in {folder}")
    out = np.empty((len(files), 2, _XJTU_SAMPLES), dtype=np.float32)
    for i, f in enumerate(files):
        arr = np.loadtxt(f, delimiter=",", dtype=np.float32, skiprows=1)
        if arr.shape[0] < _XJTU_SAMPLES:
            pad = np.zeros((_XJTU_SAMPLES - arr.shape[0], 2), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        out[i, 0] = arr[:_XJTU_SAMPLES, 0]
        out[i, 1] = arr[:_XJTU_SAMPLES, 1]
    return out


def load_xjtusy_bearing(
    root: str | Path,
    condition: int,
    bearing_id: str,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> BearingRun | None:
    """Load a single XJTU-SY bearing. ``bearing_id`` may be ``"1_1"`` or ``"Bearing1_1"``."""
    root = Path(root)
    cond_folder = root / _XJTU_FOLDERS[int(condition)]
    if not cond_folder.is_dir():
        return None
    bid = bearing_id if bearing_id.startswith("Bearing") else f"Bearing{bearing_id}"
    folder = cond_folder / bid
    if not folder.is_dir():
        return None

    cache_dir = Path(cache_dir) if cache_dir else (root.parent / "processed" / "xjtusy")
    fp = _xjtu_fingerprint(folder)
    raw_n = len(list(folder.glob("*.csv")))

    sig: np.ndarray | None = None
    if use_cache:
        candidates: list[Path] = []
        exact = cache_dir / f"{bid}_{fp}.parquet"
        if exact.exists():
            candidates.append(exact)
        if not candidates:
            candidates = sorted(cache_dir.glob(f"{bid}_*.parquet"))
        for cache_pq in candidates:
            cache_meta = cache_pq.with_suffix("").with_name(cache_pq.stem + ".meta.json")
            if not cache_meta.exists():
                continue
            try:
                import pyarrow.parquet as pq

                meta = json.loads(cache_meta.read_text())
                if int(meta.get("n_acquisitions", -1)) != raw_n:
                    continue
                t = pq.read_table(cache_pq)
                df = t.to_pandas()
                T = int(meta["n_acquisitions"])
                L = int(meta["samples_per_acquisition"])
                h = df["h"].to_numpy(dtype=np.float32).reshape(T, L)
                v = df["v"].to_numpy(dtype=np.float32).reshape(T, L)
                sig = np.stack([h, v], axis=1).astype(np.float32)
                break
            except Exception:
                sig = None
    if sig is None:
        sig = _xjtu_load_from_raw(folder)
        if use_cache:
            _phm_write_cache(cache_dir, bid, fp, sig)
            (cache_dir / f"{bid}_{fp}.meta.json").write_text(
                json.dumps({"n_acquisitions": int(sig.shape[0]),
                            "samples_per_acquisition": int(sig.shape[2])}, indent=2)
            )

    short_id = bid.replace("Bearing", "")
    op = _XJTU_OPERATING[int(condition)]
    return BearingRun(
        dataset="xjtusy",
        condition=int(condition),
        bearing_id=short_id,
        signal=sig,
        channel_names=["horizontal", "vertical"],
        fs=_XJTU_FS,
        acquisition_interval_s=_XJTU_INTERVAL_S,
        rpm=op["rpm"],
        load_N=op["load_N"],
        full_life=True,
        metadata={"source_folder": str(folder)},
    )


def load_xjtusy(
    root: str | Path,
    conditions: Iterable[int] | None = None,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> list[BearingRun]:
    conditions = sorted(conditions or _XJTU_FOLDERS.keys())
    runs: list[BearingRun] = []
    root = Path(root)
    for cond in conditions:
        cond_folder = root / _XJTU_FOLDERS.get(int(cond), "")
        if not cond_folder.is_dir():
            continue
        for sub in sorted(cond_folder.iterdir()):
            if sub.is_dir() and sub.name.startswith("Bearing"):
                run = load_xjtusy_bearing(root, cond, sub.name, cache_dir=cache_dir, use_cache=use_cache)
                if run is not None:
                    runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Public dispatcher (matches the original adapters.py contract)
# ---------------------------------------------------------------------------


def load_dataset(
    dataset: str,
    root: str | Path,
    conditions: list[int] | None = None,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> list[BearingRun]:
    if dataset == "phm2012":
        return load_phm2012(root, conditions=conditions, cache_dir=cache_dir, use_cache=use_cache)
    if dataset in ("xjtusy", "xjtu_sy", "xjtu-sy"):
        return load_xjtusy(root, conditions=conditions, cache_dir=cache_dir, use_cache=use_cache)
    raise ValueError(f"Unknown dataset: {dataset}")


def load_one_bearing(dataset: str, root: str | Path, bearing_id: str, **kwargs: Any) -> BearingRun | None:
    if dataset == "phm2012":
        return load_phm2012_bearing(root, bearing_id, **kwargs)
    if dataset in ("xjtusy", "xjtu_sy", "xjtu-sy"):
        condition = int(bearing_id.split("_")[0])
        return load_xjtusy_bearing(root, condition=condition, bearing_id=bearing_id, **kwargs)
    raise ValueError(f"Unknown dataset: {dataset}")
