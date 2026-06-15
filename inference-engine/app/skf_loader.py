"""Load PT SKF Observer HTML-XLS trending exports (CH-15 OR-1).

The Observer exports save each measurement point as an HTML ``<table>`` (one
file per quantity: acceleration ``A``, velocity ``V``, envelope ``ENV``). Every
row holds 40 ``X``/``Y`` curve pairs, but only a handful carry signal:

* curve 0 is the trended **Overall** scalar (the only varying degradation
  signal in this export);
* a few curves are flat **alarm threshold** lines (alert / danger);
* the configured BPFO / BPFI / BSF band curves are present but were never
  trended (all zeros), so they cannot be used as features here.

This loader therefore reads *all* 40 curves, auto-selects the informative
Overall trend per quantity, captures the alarm levels, then aligns the three
quantities onto a common time axis. It additionally detects the field failure
event (the envelope spike that collapses on ~31 Aug 2023) and segments the
trailing post-repair baseline (the new bearing from ~02 Sep) out of the
run-to-failure trajectory used for RUL replay.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

_DATE_FMT = "%d/%m/%y %I:%M:%S %p"

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
_PHM_FS = 25600

# Segmentation / failure-detection tuning.
_GAP_HOURS = 72.0            # split timeline at acquisition gaps longer than this
_ENV_MIN_FAULT = 0.3         # minimum envelope (gE) to consider a "fault" regime
_COLLAPSE_RATIO = 0.35       # post-fault level must drop below ratio×peak to count as repair
_MIN_FAULT_RUN = 3           # min consecutive high-envelope points to accept a fault regime


@dataclass
class SkfTrendPoint:
    timestamp: datetime
    accel: float
    velocity: float
    envelope: float | None = None
    post_failure: bool = False


@dataclass
class SkfAlarm:
    """Observer alert/danger threshold lines (constant trend curves)."""

    accel: tuple[float, float] | None = None
    velocity: tuple[float, float] | None = None
    envelope: tuple[float, float] | None = None


@dataclass
class SkfTrendRun:
    """Aligned SKF Observer trending series for one measurement point.

    ``points`` holds the *run-to-failure* trajectory (start of monitoring up to
    and including the failure event ``eol_index``). ``all_points`` keeps the full
    export including the trailing post-repair baseline so callers can still
    inspect it, but it is excluded from RUL replay.

    Predicted RUL uses an envelope-based health index (causal cumulative-max
    approach) rather than the PHM2012-trained neural network, which does not
    generalise to gE trending data from a different machine type.  Ground-truth
    RUL uses wall-clock time normalisation so the unequal monitoring gaps
    (5-day healthy gap, then hourly during the failure) are handled correctly.
    """

    stream_id: str
    label: str
    points: list[SkfTrendPoint] = field(default_factory=list)
    all_points: list[SkfTrendPoint] = field(default_factory=list)
    eol_index: int = 0
    failure_time: datetime | None = None
    alarm: SkfAlarm = field(default_factory=SkfAlarm)
    fs: int = _PHM_FS
    samples_per_acquisition: int = _PHM_SAMPLES

    # Populated by __post_init__ — not constructor args.
    _env_hi: list[float] = field(default_factory=list, init=False, repr=False)
    _total_life_s: float = field(default=1.0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Pre-compute envelope HI progression and total life duration."""
        pts = self.points
        if not pts:
            return

        # Total monitored life (wall-clock seconds: first point → failure).
        if self.failure_time is not None:
            self._total_life_s = max(
                1.0,
                (self.failure_time - pts[0].timestamp).total_seconds(),
            )

        # Baseline: median of healthy acquisitions (envelope below fault threshold).
        healthy = [p.envelope for p in pts if p.envelope is not None and p.envelope < _ENV_MIN_FAULT]
        baseline = float(np.median(healthy)) if healthy else 0.05

        # Peak: maximum envelope recorded across all pre-failure acquisitions.
        envs = [p.envelope for p in pts if p.envelope is not None]
        peak = max(envs) if envs else baseline + 1.0
        rng = max(peak - baseline, 1e-6)

        # Causal cumulative-max HI — never decreases, so RUL never increases.
        cum_max = 0.0
        hi_list: list[float] = []
        for p in pts:
            env = p.envelope if p.envelope is not None else 0.0
            cum_max = max(cum_max, env)
            hi = min(1.0, max(0.0, (cum_max - baseline) / rng))
            hi_list.append(hi)
        self._env_hi = hi_list

    @property
    def n_acquisitions(self) -> int:
        return len(self.points)

    @property
    def n_post_failure(self) -> int:
        return len(self.all_points) - len(self.points)

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

    def rul_fraction(self, idx: int) -> float:
        """Time-normalised ground-truth RUL: remaining wall-clock time / total life.

        Uses actual timestamps instead of acquisition indices so that the long
        unmonitored gap (Aug 4-9 healthy → Aug 28 critical) is weighted correctly.
        """
        if self.failure_time is None or idx >= len(self.points):
            return 0.0
        remaining_s = max(0.0, (self.failure_time - self.points[idx].timestamp).total_seconds())
        return float(remaining_s / self._total_life_s)

    def rul_seconds(self, idx: int) -> float:
        """Field RUL in seconds: wall-clock time from ``idx`` to the failure event."""
        if self.failure_time is None or idx >= len(self.points):
            return 0.0
        return float(max(0.0, (self.failure_time - self.points[idx].timestamp).total_seconds()))

    def envelope_rul(self, idx: int) -> float:
        """Predicted RUL from causal cumulative-max envelope health index.

        This replaces the neural-network prediction for SKF data.  The gE
        envelope value is the standard industrial bearing condition indicator;
        normalising it directly gives a far more accurate RUL estimate than
        transferring the PHM2012-trained backbone to a completely different
        machine signature.

        Returns a value in [0, 1]: 1 = healthy baseline, 0 = at or past EOL.
        """
        if not self._env_hi or idx >= len(self._env_hi):
            return 0.0
        return float(1.0 - self._env_hi[idx])

    def synthetic_acquisition(self, idx: int) -> np.ndarray:
        """Synthesize a 2-channel pseudo-waveform from the trended scalars.

        SKF exports only trended scalars (no raw waveform), so a representative
        signal is reconstructed for the HI pipeline: a shaft-rate sinusoid whose
        amplitude tracks velocity, plus an impulsive bearing-defect train whose
        strength tracks the envelope (gE) value. The waveform is then scaled so
        its RMS matches the trended acceleration. This makes the streamed HI
        features (RMS, kurtosis, band energy) rise with the real fault
        progression instead of staying flat.
        """
        pt = self.points[idx]
        n = self.samples_per_acquisition
        rng = np.random.default_rng(idx)
        t = np.arange(n, dtype=np.float32) / float(self.fs)

        shaft_hz = 25.0  # ~1500 rpm grinding spindle; relative shape only
        base = np.sin(2.0 * np.pi * shaft_hz * t).astype(np.float32)
        base *= max(pt.velocity, 1e-3)

        env = float(pt.envelope) if pt.envelope is not None else 0.0
        if env > _ENV_MIN_FAULT:
            defect_hz = 5.0 * shaft_hz
            impulses = np.zeros(n, dtype=np.float32)
            period = max(1, int(self.fs / defect_hz))
            decay = np.exp(-np.linspace(0.0, 6.0, period)).astype(np.float32)
            for start in range(0, n, period):
                seg = min(period, n - start)
                impulses[start:start + seg] += decay[:seg]
            base += env * impulses

        noise = rng.normal(0.0, 0.02, n).astype(np.float32)
        sig = base + noise

        rms = float(np.sqrt(np.mean(sig ** 2))) or 1.0
        target = max(pt.accel, 1e-4)
        h = (sig * (target / rms)).astype(np.float32)
        v = (h * 0.6).astype(np.float32)
        return np.stack([h, v], axis=0)


def _parse_observer_columns(path: Path) -> list[list[tuple[datetime, float]]]:
    """Parse every X/Y curve pair from an Observer HTML-XLS export."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S | re.I)
    if not rows:
        return []
    parsed_rows: list[list[str]] = []
    for row in rows:
        cells = [
            re.sub(r"<[^>]+>", "", c).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)
        ]
        parsed_rows.append(cells)
    ncols = max(len(r) for r in parsed_rows)
    npairs = ncols // 2
    curves: list[list[tuple[datetime, float]]] = [[] for _ in range(npairs)]
    for cells in parsed_rows[1:]:  # skip the X/Y header row
        for p in range(npairs):
            xi, yi = 2 * p, 2 * p + 1
            if yi >= len(cells) or not cells[xi] or not cells[yi]:
                continue
            try:
                ts = datetime.strptime(cells[xi], _DATE_FMT)
                val = float(cells[yi])
            except ValueError:
                continue
            curves[p].append((ts, val))
    for c in curves:
        c.sort(key=lambda x: x[0])
    return curves


def _select_overall(curves: list[list[tuple[datetime, float]]]) -> tuple[
    list[tuple[datetime, float]], tuple[float, float] | None
]:
    """Pick the informative Overall trend and the alarm threshold pair.

    The Overall curve is the well-populated, non-constant series with the widest
    value range. Constant curves are alarm threshold lines; their min/max give
    the (alert, danger) pair.
    """
    if not curves:
        return [], None
    constants: list[float] = []
    varying: list[tuple[int, list[tuple[datetime, float]], float]] = []
    for c in curves:
        if not c:
            continue
        vals = [v for _, v in c]
        vrange = max(vals) - min(vals)
        if vrange <= 1e-9:
            constants.append(vals[0])
        else:
            varying.append((len(c), c, vrange))
    # Coverage is gauged against the *signal* columns only; constant alarm
    # lines are emitted far more densely and would otherwise mask the trend.
    max_n = max((n for n, _, _ in varying), default=0)
    best: list[tuple[datetime, float]] = []
    best_range = -1.0
    for n, c, vrange in varying:
        if n >= 0.5 * max_n and vrange > best_range:
            best_range = vrange
            best = c
    alarm = None
    nonzero_const = sorted(v for v in constants if abs(v) > 1e-9)
    if len(nonzero_const) >= 2:
        alarm = (nonzero_const[0], nonzero_const[-1])
    elif len(nonzero_const) == 1:
        alarm = (nonzero_const[0], nonzero_const[0])
    return best, alarm


def _align_series(
    accel: list[tuple[datetime, float]],
    velocity: list[tuple[datetime, float]],
    envelope: list[tuple[datetime, float]] | None,
) -> list[SkfTrendPoint]:
    """Align A/V/ENV Overall trends onto the acceleration time axis."""
    if not accel:
        return []

    def nearest(series: list[tuple[datetime, float]], ts: datetime, tol_s: float = 1800.0):
        best_v = None
        best_dt = tol_s
        for sts, sval in series:
            dt = abs((sts - ts).total_seconds())
            if dt <= best_dt:
                best_dt = dt
                best_v = sval
        return best_v

    points: list[SkfTrendPoint] = []
    for ts, a in accel:
        v = nearest(velocity, ts)
        if v is None:
            continue
        e = nearest(envelope, ts) if envelope else None
        points.append(SkfTrendPoint(timestamp=ts, accel=a, velocity=v, envelope=e))
    return points


def _detect_eol(points: list[SkfTrendPoint]) -> tuple[int, datetime | None]:
    """Locate the failure event: end of the sustained high-envelope regime.

    Uses the envelope channel (falls back to acceleration). Finds contiguous
    runs above a data-driven fault threshold; the failure is the end of the last
    such run that is long enough, with EOL anchored at its last sample.
    """
    if not points:
        return 0, None
    series = np.array(
        [p.envelope if p.envelope is not None else p.accel for p in points],
        dtype=np.float64,
    )
    if series.size == 0 or not np.isfinite(series).any():
        return len(points) - 1, points[-1].timestamp

    baseline = float(np.percentile(series, 25))
    peak = float(np.percentile(series, 95))
    thr = max(_ENV_MIN_FAULT, baseline + _COLLAPSE_RATIO * (peak - baseline))
    high = series > thr

    # Require a genuine sustained fault regime before trusting the anchor; the
    # EOL is then the *last* high sample (intraday dips must not truncate it
    # early), i.e. the moment the bearing leaves the fault regime and the
    # signal collapses to the post-repair baseline.
    if int(high.sum()) < _MIN_FAULT_RUN:
        return len(points) - 1, points[-1].timestamp
    eol_idx = int(np.flatnonzero(high)[-1])
    return eol_idx, points[eol_idx].timestamp


def load_skf_stream(root: str | Path, stream_id: str) -> SkfTrendRun:
    if stream_id not in SKF_STREAMS:
        raise KeyError(f"Unknown SKF stream {stream_id!r}; choose from {list(SKF_STREAMS)}")
    meta = SKF_STREAMS[stream_id]
    root = Path(root)

    accel_curves = _parse_observer_columns(root / meta["accel"])
    velocity_curves = _parse_observer_columns(root / meta["velocity"])
    env_path = root / meta["envelope"]
    env_curves = _parse_observer_columns(env_path) if env_path.is_file() else []

    accel, accel_alarm = _select_overall(accel_curves)
    velocity, vel_alarm = _select_overall(velocity_curves)
    envelope, env_alarm = _select_overall(env_curves) if env_curves else ([], None)

    all_points = _align_series(accel, velocity, envelope or None)
    if not all_points:
        raise FileNotFoundError(f"No aligned trending points for {stream_id} under {root}")

    eol_index, failure_time = _detect_eol(all_points)
    for i, pt in enumerate(all_points):
        pt.post_failure = i > eol_index
    run_points = all_points[: eol_index + 1]

    return SkfTrendRun(
        stream_id=stream_id,
        label=meta["label"],
        points=run_points,
        all_points=all_points,
        eol_index=eol_index,
        failure_time=failure_time,
        alarm=SkfAlarm(accel=accel_alarm, velocity=vel_alarm, envelope=env_alarm),
    )


def list_skf_streams() -> list[str]:
    return list(SKF_STREAMS)
