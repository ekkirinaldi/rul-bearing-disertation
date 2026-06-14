"""Self-contained PHM2012, XJTU-SY, IMS, and CWRU loaders.

Returns lightweight ``BearingRun`` objects (a local dataclass — *not*
``brul.data.base.BearingRun``) so the package has zero external runtime
dependencies beyond the standard scientific stack.

Both original loaders honor the parquet cache layout produced by the
historical ``brul`` project (``<cache>/<bid>_<fingerprint>.parquet``
with columns ``t_idx, sample_idx, h, v`` and a sibling ``.meta.json``).
When the cache is missing the loaders read the raw folders directly and
persist a parquet for next time.

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

IMS (NASA PrognCenter):
  Raw layout    : ``<root>/<timestamp_filename>`` (flat directory, no subdirs)
  File format   : ASCII tab-separated, 8 columns; each file = 1 acquisition.
                  Columns 0–7 map to channels 1–8 (pairs per bearing:
                  Bearing 1 = cols 0,1; Bearing 2 = cols 2,3;
                  Bearing 3 = cols 4,5; Bearing 4 = cols 6,7).
  Sampling      : 20_480 Hz, 20_480 samples per acquisition (= 1.0 s)
  Interval      : 10 min (600 s) between acquisitions

CWRU (Case Western Reserve University):
  Raw layout    : ``<root>/*.mat``
  Naming        : ``{fault_type}{size_mils}_{load_hp}_{rpm_id}.mat``
                  fault_type in {B, IR, OR, Normal}; e.g. ``IR007_0_109.mat``
  MAT variables : ``DE_time`` (drive-end accelerometer, primary channel used)
  Sampling      : 12_000 or 48_000 Hz (depends on rpm_id); 48k files preferred
  Task          : Fault diagnosis (no run-to-failure; no RUL labels).
                  Fault class is encoded in bearing_id and metadata.
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
    if dataset == "ims":
        return load_ims(root, cache_dir=cache_dir, use_cache=use_cache)
    if dataset == "cwru":
        return load_cwru(root, cache_dir=cache_dir)
    raise ValueError(f"Unknown dataset: {dataset}")


def load_one_bearing(dataset: str, root: str | Path, bearing_id: str, **kwargs: Any) -> BearingRun | None:
    if dataset == "phm2012":
        return load_phm2012_bearing(root, bearing_id, **kwargs)
    if dataset in ("xjtusy", "xjtu_sy", "xjtu-sy"):
        condition = int(bearing_id.split("_")[0])
        return load_xjtusy_bearing(root, condition=condition, bearing_id=bearing_id, **kwargs)
    if dataset == "ims":
        return load_ims_bearing(root, bearing_id, **kwargs)
    if dataset == "cwru":
        return load_cwru_bearing(root, bearing_id, **kwargs)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# IMS (NASA PrognCenter) specifics
# ---------------------------------------------------------------------------

_IMS_FS = 20_480
_IMS_SAMPLES = 20_480   # one second per acquisition file
_IMS_INTERVAL_S = 600.0  # files are ~10 min apart
_IMS_RPM = 2000.0
_IMS_BEARINGS = {
    "1": (0, 1),   # channel indices (0-based) in the 8-column file
    "2": (2, 3),
    "3": (4, 5),
    "4": (6, 7),
}


def _ims_list_files(root: Path) -> list[Path]:
    """Return sorted list of ASCII data files in ``root``.

    The IMS raw directory contains one file per acquisition named as a
    timestamp string (e.g. ``2003.10.22.12.06.24``). Files with common
    non-data extensions are skipped.
    """
    skip = {".zip", ".pdf", ".txt", ".md", ".json", ".parquet"}
    files = sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() not in skip],
        key=lambda p: p.name,
    )
    return files


def _ims_load_file(path: Path, ch_indices: tuple[int, int]) -> np.ndarray:
    """Load one IMS acquisition file; return (L,) array for the two channels stacked."""
    data = np.loadtxt(path, dtype=np.float32)  # (20480, 8) or similar
    n = min(data.shape[0], _IMS_SAMPLES)
    out = np.zeros((_IMS_SAMPLES, 2), dtype=np.float32)
    out[:n, 0] = data[:n, ch_indices[0]]
    out[:n, 1] = data[:n, ch_indices[1]]
    return out  # (L, 2)


def load_ims_bearing(
    root: str | Path,
    bearing_id: str,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> BearingRun | None:
    """Load one IMS bearing as a ``BearingRun``.

    ``bearing_id`` should be one of ``"1"``, ``"2"``, ``"3"``, ``"4"``.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    if bearing_id not in _IMS_BEARINGS:
        raise ValueError(f"IMS bearing_id must be one of {list(_IMS_BEARINGS)}; got {bearing_id!r}")
    ch_idx = _IMS_BEARINGS[bearing_id]

    cache_dir = Path(cache_dir) if cache_dir else (root.parent / "processed" / "ims")
    cache_key = f"ims_bearing{bearing_id}"
    cache_pq = cache_dir / f"{cache_key}.parquet"
    cache_meta = cache_dir / f"{cache_key}.meta.json"

    sig: np.ndarray | None = None
    if use_cache and cache_pq.exists() and cache_meta.exists():
        try:
            import pyarrow.parquet as _pq

            meta = json.loads(cache_meta.read_text())
            t = _pq.read_table(cache_pq)
            df = t.to_pandas()
            T = int(meta["n_acquisitions"])
            L = int(meta["samples_per_acquisition"])
            h_arr = df["h"].to_numpy(dtype=np.float32).reshape(T, L)
            v_arr = df["v"].to_numpy(dtype=np.float32).reshape(T, L)
            sig = np.stack([h_arr, v_arr], axis=1).astype(np.float32)
        except Exception:
            sig = None

    if sig is None:
        files = _ims_list_files(root)
        if not files:
            return None
        acqs: list[np.ndarray] = []
        for f in files:
            try:
                acq = _ims_load_file(f, ch_idx)  # (L, 2)
                acqs.append(acq)
            except Exception:
                continue
        if not acqs:
            return None
        sig = np.stack(acqs, axis=0)          # (T, L, 2)
        sig = sig.transpose(0, 2, 1)          # (T, 2, L)
        if use_cache:
            try:
                _phm_write_cache(cache_dir, cache_key, "nofingerprint", sig)
                (cache_dir / f"{cache_key}_nofingerprint.meta.json").rename(cache_meta)
            except Exception:
                pass

    return BearingRun(
        dataset="ims",
        condition=1,
        bearing_id=bearing_id,
        signal=sig,
        channel_names=["ch_a", "ch_b"],
        fs=_IMS_FS,
        acquisition_interval_s=_IMS_INTERVAL_S,
        rpm=_IMS_RPM,
        load_N=0.0,
        full_life=True,
        metadata={"channel_indices": list(ch_idx), "source_root": str(root)},
    )


def load_ims(
    root: str | Path,
    *,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> list[BearingRun]:
    """Load all four IMS bearings from ``root``."""
    root = Path(root)
    runs: list[BearingRun] = []
    for bid in _IMS_BEARINGS:
        run = load_ims_bearing(root, bid, cache_dir=cache_dir, use_cache=use_cache)
        if run is not None:
            runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# CWRU (Case Western Reserve University) specifics
# ---------------------------------------------------------------------------

_CWRU_FS = 48_000   # 48 kHz preferred; 12 kHz files also present
_CWRU_SAMPLES_48K = 48_000   # 1 second at 48 kHz
_CWRU_RPM_MAP = {
    "0": 1797.0,
    "1": 1772.0,
    "2": 1750.0,
    "3": 1730.0,
}
_CWRU_FAULT_PATTERN = re.compile(
    r"^(?P<type>B|IR|OR|Normal)(?P<size>\d{3})?_(?P<load>[0-3])_(?P<rpm_id>\d+)\.mat$",
    re.IGNORECASE,
)


def _cwru_parse_filename(fname: str) -> dict[str, str] | None:
    """Parse a CWRU .mat filename into a metadata dict."""
    m = re.match(
        r"^(?P<type>B|IR|OR|Time_Normal|Normal)"
        r"(?P<size>\d{3})?_(?P<pos>[0-9]+)?_?(?P<load>[0-3])_(?P<rpm_id>\d+)\.mat$",
        fname, re.IGNORECASE,
    )
    if m is None:
        # Try simpler Normal format: Time_Normal_0_097.mat
        m2 = re.match(r"Time_Normal_(?P<load>[0-3])_(?P<rpm_id>\d+)\.mat$", fname, re.IGNORECASE)
        if m2 is None:
            return None
        return {
            "fault_type": "normal",
            "fault_size_mils": "0",
            "load_hp": m2.group("load"),
            "rpm_id": m2.group("rpm_id"),
        }
    fault_type = m.group("type").upper()
    if fault_type in ("NORMAL", "TIME_NORMAL"):
        fault_type = "normal"
    return {
        "fault_type": fault_type.lower(),
        "fault_size_mils": m.group("size") or "0",
        "load_hp": m.group("load"),
        "rpm_id": m.group("rpm_id"),
    }


_CWRU_ACQ_SAMPLES = 480  # ~10 ms per acquisition at 48 kHz; gives T≈100 per file


def load_cwru_bearing(
    root: str | Path,
    bearing_id: str,
    *,
    acq_samples: int = _CWRU_ACQ_SAMPLES,
    **_kwargs: Any,
) -> BearingRun | None:
    """Load one CWRU .mat file as a ``BearingRun``.

    ``bearing_id`` should be the stem of the .mat filename (without extension),
    e.g. ``"IR007_0_109"`` or ``"B014_2_124"``.

    The raw signal (48 kHz, ~48 000 samples) is segmented into T acquisitions
    of ``acq_samples`` samples each, yielding ``signal.shape = (T, 1, acq_samples)``.
    This makes the BearingRun compatible with the HI pipeline and the sequence
    modelling datamodule.  For each file all acquisitions share the same fault
    condition, so ``fault_severity`` metadata is constant within the run.
    """
    root = Path(root)
    mat_path = root / f"{bearing_id}.mat"
    if not mat_path.exists():
        return None

    try:
        import scipy.io
    except ImportError as exc:
        raise ImportError("scipy is required to load CWRU .mat files; install it with 'pip install scipy'") from exc

    mat = scipy.io.loadmat(str(mat_path))

    # Find the DE_time key (naming varies slightly between files)
    de_key = next(
        (k for k in mat if "DE_time" in k or (k.endswith("DE") and not k.startswith("_"))),
        None,
    )
    if de_key is None:
        # Fallback: pick the largest numeric array
        de_key = max(
            (k for k in mat if not k.startswith("_")),
            key=lambda k: mat[k].size if isinstance(mat[k], np.ndarray) else 0,
            default=None,
        )
    if de_key is None:
        return None

    raw = mat[de_key].squeeze().astype(np.float32)
    # Determine FS from array length (if 48k samples ≈ 1 s → 48 kHz; else 12 kHz)
    fs = 48_000 if raw.size >= 40_000 else 12_000

    # Segment into T acquisitions of acq_samples each.
    T = max(1, raw.size // acq_samples)
    used = T * acq_samples
    raw = raw[:used]
    # shape: (T, acq_samples)
    chunks = raw.reshape(T, acq_samples)
    # Expand channel dim: (T, 1, acq_samples)
    sig = chunks[:, np.newaxis, :]

    meta = _cwru_parse_filename(mat_path.name)
    if meta is None:
        meta = {"fault_type": "unknown", "fault_size_mils": "0",
                "load_hp": "0", "rpm_id": "0"}

    rpm = _CWRU_RPM_MAP.get(meta["load_hp"], 1797.0)

    # Fault severity proxy: 0 = normal, 1 = 0.007", 2 = 0.014", 3 = 0.021" (mils).
    # Filenames use three-digit mils ("007", "021"); normalize before lookup.
    fault_size_mils = meta.get("fault_size_mils", "0")
    try:
        mils_int = int(str(fault_size_mils).lstrip("0") or "0")
    except ValueError:
        mils_int = 0
    if mils_int == 7:
        severity = 1
    elif mils_int == 14:
        severity = 2
    elif mils_int == 21:
        severity = 3
    else:
        severity = 0

    return BearingRun(
        dataset="cwru",
        condition=int(meta.get("load_hp", 0)),
        bearing_id=bearing_id,
        signal=sig,
        channel_names=["DE_time"],
        fs=fs,
        acquisition_interval_s=float(acq_samples) / fs,
        rpm=rpm,
        load_N=0.0,
        full_life=True,
        eol_index=T - 1,
        metadata={
            "fault_type": meta["fault_type"],
            "fault_size_mils": fault_size_mils,
            "fault_severity": severity,
            "load_hp": meta.get("load_hp", "0"),
            "source_file": str(mat_path),
        },
    )


def load_cwru(
    root: str | Path,
    *,
    cache_dir: str | Path | None = None,  # unused; accepted for API parity
) -> list[BearingRun]:
    """Load all CWRU .mat files from ``root``."""
    root = Path(root)
    runs: list[BearingRun] = []
    for mat_file in sorted(root.glob("*.mat")):
        run = load_cwru_bearing(root, mat_file.stem)
        if run is not None:
            runs.append(run)
    return runs
