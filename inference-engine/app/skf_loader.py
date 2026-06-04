"""Load PT SKF Observer HTML-XLS trending exports (CH-15 OR-1)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

_DATE_FMT = "%d/%m/%y %I:%M:%S %p"

# Measurement point → (accel file, velocity file, envelope file)
SKF_STREAMS: dict[str, dict[str, str]] = {
    "ch1_01_nde": {
        "label": "Channel 1-01 OR-1 NDE (grinding)",
        "accel": "OA CH 1-01-R-A-NDE.xls",
        "velocity": "OA CH1-01-R-V-NDE.xls",
        "envelope": "OA CH 1-01-R-ENV-NDE.xls",
    },
    "ch3_02_de": {
        "label": "Channel 3-02 OR-1 DE (grinding)",
        "accel": "OA Ch3-02-R-A-DE.xls",
        "velocity": "OA CH3-02-R-V-DE.xls",
        "envelope": "OA Ch3-02-R-ENV-DE.xls",
    },
}

_PHM_SAMPLES = 2560
_PHM_FS = 25_600


@dataclass
class SkfTrendPoint:
    timestamp: datetime
    accel: float
    velocity: float
    envelope: float | None = None


@dataclass
class SkfTrendRun:
    """Aligned SKF Observer trending series for one measurement point."""

    stream_id: str
    label: str
    points: list[SkfTrendPoint] = field(default_factory=list)
    fs: int = _PHM_FS
    samples_per_acquisition: int = _PHM_SAMPLES

    @property
    def n_acquisitions(self) -> int:
        return len(self.points)

    def synthetic_acquisition(self, idx: int) -> np.ndarray:
        """Build a 2-channel pseudo-waveform from trending scalars for HI extraction."""
        pt = self.points[idx]
        rng = np.random.default_rng(idx)
        noise = rng.normal(0.0, 1e-4, self.samples_per_acquisition).astype(np.float32)
        h = np.full(self.samples_per_acquisition, pt.accel, dtype=np.float32) + noise
        v = np.full(self.samples_per_acquisition, pt.velocity, dtype=np.float32) + noise
        return np.stack([h, v], axis=0)

    @property
    def acquisition_interval_s(self) -> float:
        if len(self.points) < 2:
            return 3600.0
        deltas = [
            (self.points[i + 1].timestamp - self.points[i].timestamp).total_seconds()
            for i in range(min(20, len(self.points) - 1))
        ]
        med = float(np.median(deltas))
        return med if med > 0 else 3600.0


def _parse_observer_html_xls(path: Path) -> list[tuple[datetime, float]]:
    """Parse SKF Observer export (HTML table saved as .xls). Uses first X/Y pair."""
    text = path.read_text(encoding="utf-8-sig")
    rows = re.findall(r"<tr>(.*?)</tr>", text, re.S)
    out: list[tuple[datetime, float]] = []
    for row in rows[1:]:
        cells = [c.strip() for c in re.findall(r"<td>(.*?)</td>", row)]
        if len(cells) < 2 or not cells[0] or not cells[1]:
            continue
        try:
            ts = datetime.strptime(cells[0], _DATE_FMT)
            val = float(cells[1])
        except ValueError:
            continue
        out.append((ts, val))
    out.sort(key=lambda x: x[0])
    return out


def _align_series(
    accel: list[tuple[datetime, float]],
    velocity: list[tuple[datetime, float]],
    envelope: list[tuple[datetime, float]] | None,
) -> list[SkfTrendPoint]:
    """Align A/V/ENV series (Observer exports use slightly different timestamps)."""
    if not accel:
        return []

    # Same row count → pair by index (typical Observer multi-file export).
    if len(accel) == len(velocity):
        env_vals = [v for _, v in envelope] if envelope else [None] * len(accel)
        if envelope and len(envelope) != len(accel):
            env_vals = [None] * len(accel)
        return [
            SkfTrendPoint(
                timestamp=ts,
                accel=a,
                velocity=velocity[i][1],
                envelope=env_vals[i] if i < len(env_vals) else None,
            )
            for i, (ts, a) in enumerate(accel)
        ]

    # Fallback: nearest velocity sample within 30 minutes.
    points: list[SkfTrendPoint] = []
    for ts, a in accel:
        best: tuple[datetime, float] | None = None
        best_dt = 1e18
        for vts, vval in velocity:
            dt = abs((vts - ts).total_seconds())
            if dt < best_dt:
                best_dt = dt
                best = (vts, vval)
        if best is None or best_dt > 1800:
            continue
        env_val = None
        if envelope:
            for ets, eval_ in envelope:
                if abs((ets - ts).total_seconds()) <= 1800:
                    env_val = eval_
                    break
        points.append(SkfTrendPoint(timestamp=ts, accel=a, velocity=best[1], envelope=env_val))
    return points


def load_skf_stream(root: str | Path, stream_id: str) -> SkfTrendRun:
    if stream_id not in SKF_STREAMS:
        raise KeyError(f"Unknown SKF stream {stream_id!r}; choose from {list(SKF_STREAMS)}")
    meta = SKF_STREAMS[stream_id]
    root = Path(root)
    accel = _parse_observer_html_xls(root / meta["accel"])
    velocity = _parse_observer_html_xls(root / meta["velocity"])
    env_path = root / meta["envelope"]
    envelope = _parse_observer_html_xls(env_path) if env_path.is_file() else None
    points = _align_series(accel, velocity, envelope)
    if not points:
        raise FileNotFoundError(f"No aligned trending points for {stream_id} under {root}")
    return SkfTrendRun(stream_id=stream_id, label=meta["label"], points=points)


def list_skf_streams() -> list[str]:
    return list(SKF_STREAMS)
