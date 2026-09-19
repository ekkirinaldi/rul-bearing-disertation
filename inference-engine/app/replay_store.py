"""Saveable / replayable runs + significant-drop failure analysis.

The live streaming engine (:mod:`app.engine`) computes a fresh prediction per
acquisition and forgets it. This module captures a *complete* run-to-failure for
one bearing or plant stream, persists every per-acquisition value to disk, and
runs an offline analysis pass that locates the **significant RUL drops** — the
moments where the bearing's predicted health falls sharply — and explains each
one with the interpretability tools already built (Integrated Gradients, the
fusion gate, the top input drivers, and the raw HI feature deltas that physically
changed across the drop).

The result is a single JSON artefact per run, replayable in the dashboard with
no model in the loop, and a failure analysis answering *what made this bearing
break*. Works identically for PHM2012, XJTU-SY, and the SKF plant transfer.

Persistence layout (git-ignored, local only):

    inference-engine/runs/
        <dataset>__<bearing>__<YYYYmmdd_HHMMSS>.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.engine import StreamSession
from app.model_registry import feature_names, list_datasets, stream_label

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# Drop-detection defaults. ``pred_rul`` is on a 0..1 scale (1 = healthy); a drop
# of ``min_drop`` over ``lookback`` post-warmup frames counts as significant.
_SMOOTH = 5
_LOOKBACK = 10
_MIN_DROP = 0.08
_MIN_GAP = 15
_MAX_EVENTS = 10
# Waveform points retained per channel in the persisted frame (the live stream
# downsamples to 512; replay does not need that resolution).
_WAVE_PTS = 220


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def _thin(arr: list[float] | None, n: int) -> list[float] | None:
    if not arr:
        return arr
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n, dtype=int)
    return [float(arr[i]) for i in idx]


def _slim_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Strip the persisted frame to a replay-sufficient, size-bounded record."""
    wf = frame.get("waveform") or {}
    out = dict(frame)
    out["waveform"] = {
        "horizontal": _thin(wf.get("horizontal"), _WAVE_PTS),
        "vertical": _thin(wf.get("vertical"), _WAVE_PTS),
    }
    return out


def record_run(
    dataset: str,
    bearing: str,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    drop_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream a full run-to-failure, persist it, and analyse its drops.

    ``progress_cb(t, n_total)`` is invoked once per acquisition so a caller can
    surface a progress bar. Returns the assembled (and saved) run dict.
    """
    session = StreamSession(dataset_key=dataset, bearing_id=bearing)
    feats = feature_names(session.spec)
    if not session.feature_names:
        session.feature_names = list(feats)

    frames: list[dict[str, Any]] = []
    windows: dict[int, np.ndarray] = {}
    n_total = session.n_total

    while True:
        frame = session.step()
        if frame is None:
            break
        if not frame.get("warmup") and session.last_window is not None:
            windows[int(frame["t"])] = session.last_window.copy()
        frames.append(_slim_frame(frame))
        if progress_cb is not None:
            progress_cb(int(frame["t"]) + 1, n_total)

    drops = _analyse_drops(session, frames, windows, feats, drop_params or {})

    run = _finalise(session, frames, drops, feats)
    save_run(run)
    return run


# --------------------------------------------------------------------------- #
# Drop detection + per-drop explanation
# --------------------------------------------------------------------------- #
def _smooth(values: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or values.size < win:
        return values
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(values, kernel, mode="same")


def detect_drops(
    pred: list[float],
    t_index: list[int],
    *,
    smooth: int = _SMOOTH,
    lookback: int = _LOOKBACK,
    min_drop: float = _MIN_DROP,
    min_gap: int = _MIN_GAP,
    max_events: int = _MAX_EVENTS,
) -> list[dict[str, Any]]:
    """Locate significant declines in a 0..1 RUL series.

    Operates on post-warmup samples only (``pred`` aligned to ``t_index``). A
    candidate forms wherever the smoothed RUL falls by ``min_drop`` across
    ``lookback`` samples; candidates are non-maximum-suppressed by magnitude with
    a ``min_gap`` separation, then capped at ``max_events``.
    """
    n = len(pred)
    if n <= lookback + 1:
        return []
    s = _smooth(np.asarray(pred, dtype=np.float64), smooth)

    candidates: list[tuple[int, float, int]] = []  # (i, magnitude, from_i)
    for i in range(lookback, n):
        j = i - lookback
        delta = float(s[j] - s[i])
        if delta >= min_drop:
            candidates.append((i, delta, j))
    if not candidates:
        return []

    candidates.sort(key=lambda c: c[1], reverse=True)
    picked: list[tuple[int, float, int]] = []
    for cand in candidates:
        if all(abs(cand[0] - p[0]) >= min_gap for p in picked):
            picked.append(cand)
        if len(picked) >= max_events:
            break

    picked.sort(key=lambda c: c[0])
    events: list[dict[str, Any]] = []
    for rank, (i, mag, j) in enumerate(picked):
        events.append(
            {
                "order": rank,
                "from_pos": j,
                "to_pos": i,
                "from_t": int(t_index[j]),
                "to_t": int(t_index[i]),
                "from_rul": float(pred[j]),
                "to_rul": float(pred[i]),
                "magnitude": float(mag),
            }
        )
    return events


def _hi_delta(
    frames: list[dict[str, Any]],
    from_t: int,
    to_t: int,
    feats: list[str],
    scale: np.ndarray | None,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    """Raw HI features that changed most across a drop (physical explanation).

    Ranked by the change *relative to how much each feature normally varies*
    across the run (``|after − before| / std``). This surfaces features that
    made a genuinely large move, not near-zero noise features whose ratio change
    happens to be large.
    """
    by_t = {int(f["t"]): f for f in frames}
    fa, fb = by_t.get(from_t), by_t.get(to_t)
    if not fa or not fb:
        return []
    a = np.asarray(fa.get("hi_raw") or [], dtype=np.float64)
    b = np.asarray(fb.get("hi_raw") or [], dtype=np.float64)
    if a.size == 0 or a.shape != b.shape:
        return []
    delta = b - a
    if scale is None or scale.shape != a.shape:
        scale = np.abs(a) + np.abs(b)
    score = np.abs(delta) / (scale + 1e-9)  # change in units of normal variability
    order = np.argsort(score)[::-1][:top_k]
    out: list[dict[str, Any]] = []
    for i in order:
        name = feats[i] if i < len(feats) else f"f{i}"
        out.append(
            {
                "name": name,
                "before": float(a[i]),
                "after": float(b[i]),
                "delta": float(delta[i]),
                "score": float(score[i]),
                "rel_change": float(score[i]),
                "dir": "up" if delta[i] >= 0 else "down",
            }
        )
    return out


def _analyse_drops(
    session: StreamSession,
    frames: list[dict[str, Any]],
    windows: dict[int, np.ndarray],
    feats: list[str],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Detect drops, then attach IG + drivers + gate + HI deltas to each."""
    pred: list[float] = []
    t_index: list[int] = []
    for f in frames:
        if not f.get("warmup") and f.get("pred_rul") is not None:
            pred.append(float(f["pred_rul"]))
            t_index.append(int(f["t"]))

    events = detect_drops(
        pred,
        t_index,
        smooth=int(params.get("smooth", _SMOOTH)),
        lookback=int(params.get("lookback", _LOOKBACK)),
        min_drop=float(params.get("min_drop", _MIN_DROP)),
        min_gap=int(params.get("min_gap", _MIN_GAP)),
        max_events=int(params.get("max_events", _MAX_EVENTS)),
    )

    # Per-feature variability across the whole run — the yardstick for "big move".
    hi_mat = np.asarray(
        [f["hi_raw"] for f in frames if f.get("hi_raw")], dtype=np.float64
    )
    scale = hi_mat.std(axis=0) if hi_mat.size else None

    by_t = {int(f["t"]): f for f in frames}
    for ev in events:
        to_t = ev["to_t"]
        frame = by_t.get(to_t, {})
        ev["elapsed_s"] = frame.get("elapsed_s")
        ev["timestamp"] = frame.get("timestamp")
        ev["branch_gate"] = frame.get("branch_gate")
        ev["drivers"] = frame.get("top_drivers")
        ev["hi_delta"] = _hi_delta(frames, ev["from_t"], to_t, feats, scale)
        ev["ig"] = None
        try:
            ev["ig"] = session.explain_window(windows.get(to_t))
        except Exception:  # noqa: BLE001 — never let an IG failure drop the event
            ev["ig"] = None
    return events


# --------------------------------------------------------------------------- #
# Assembly + persistence
# --------------------------------------------------------------------------- #
def _finalise(
    session: StreamSession,
    frames: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    feats: list[str],
) -> dict[str, Any]:
    spec = session.spec
    now = datetime.now(timezone.utc).astimezone()
    run_id = f"{spec.key}__{session.bearing_id}__{now.strftime('%Y%m%d_%H%M%S')}"

    post = [f for f in frames if not f.get("warmup") and f.get("pred_rul") is not None]
    first_rul = post[0]["pred_rul"] if post else None
    last_rul = post[-1]["pred_rul"] if post else None
    biggest = max((d["magnitude"] for d in drops), default=0.0)

    return {
        "run_id": run_id,
        "meta": {
            "dataset": spec.key,
            "dataset_label": spec.label,
            "bearing": session.bearing_id,
            "bearing_label": stream_label(spec.key, session.bearing_id),
            "model": spec.model_name,
            "checkpoint": spec.checkpoint.name,
            "device": str(session.loaded.device),
            "window_length": spec.window_length,
            "interval_s": session.interval_s,
            "n_total": session.n_total,
            "n_frames": len(frames),
            "eol_index": session.eol_index if spec.has_gt_rul else None,
            "has_gt_rul": spec.has_gt_rul,
            "transfer_note": spec.transfer_note,
            "feature_names": feats,
            "recorded_at": now.isoformat(),
        },
        "summary": {
            "first_rul": first_rul,
            "last_rul": last_rul,
            "n_drops": len(drops),
            "biggest_drop": biggest,
        },
        "drops": drops,
        "frames": frames,
    }


def list_targets() -> list[tuple[str, str]]:
    """Every (dataset, bearing/stream) pair the dashboard can replay."""
    targets: list[tuple[str, str]] = []
    for spec in list_datasets():
        for bearing in spec.test_bearings:
            targets.append((spec.key, bearing))
    return targets


def find_recording(dataset: str, bearing: str) -> str | None:
    """run_id of the newest saved run for a dataset+bearing, or None."""
    for r in list_runs():
        meta = r.get("meta", {})
        if meta.get("dataset") == dataset and meta.get("bearing") == bearing:
            return r["run_id"]
    return None


def record_all(
    *,
    skip_existing: bool = True,
    item_cb: Callable[[int, int, str, str, str], None] | None = None,
    frame_cb: Callable[[int, int, str, str, int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Record every dataset+bearing sequentially.

    ``item_cb(done, total, dataset, bearing, phase)`` fires at item boundaries
    (phase ∈ {"start", "skipped", "done", "error"}); ``frame_cb(idx, total,
    dataset, bearing, t, n)`` forwards per-acquisition progress within a run.
    Already-recorded pairs are skipped when ``skip_existing``.
    """
    targets = list_targets()
    total = len(targets)
    results: list[dict[str, Any]] = []
    for idx, (dataset, bearing) in enumerate(targets):
        if item_cb:
            item_cb(idx, total, dataset, bearing, "start")
        existing = find_recording(dataset, bearing) if skip_existing else None
        if existing:
            results.append({"dataset": dataset, "bearing": bearing, "run_id": existing, "skipped": True})
            if item_cb:
                item_cb(idx + 1, total, dataset, bearing, "skipped")
            continue
        try:
            cb = None
            if frame_cb is not None:
                cb = lambda t, n, i=idx, d=dataset, b=bearing: frame_cb(i, total, d, b, t, n)  # noqa: E731
            run = record_run(dataset, bearing, progress_cb=cb)
            results.append(
                {
                    "dataset": dataset,
                    "bearing": bearing,
                    "run_id": run["run_id"],
                    "n_drops": run["summary"]["n_drops"],
                }
            )
            if item_cb:
                item_cb(idx + 1, total, dataset, bearing, "done")
        except Exception as exc:  # noqa: BLE001 — one bad bearing must not abort the batch
            results.append({"dataset": dataset, "bearing": bearing, "error": str(exc)})
            if item_cb:
                item_cb(idx + 1, total, dataset, bearing, "error")
    return results


def save_run(run: dict[str, Any]) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = run["run_id"]
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(run))
    return run_id


def list_runs() -> list[dict[str, Any]]:
    """Lightweight index of saved runs (meta + summary, no frames)."""
    if not RUNS_DIR.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        items.append(
            {
                "run_id": data.get("run_id", path.stem),
                "meta": data.get("meta", {}),
                "summary": data.get("summary", {}),
                "size_bytes": path.stat().st_size,
            }
        )
    return items


def load_run(run_id: str) -> dict[str, Any] | None:
    path = RUNS_DIR / f"{_safe_id(run_id)}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def delete_run(run_id: str) -> bool:
    path = RUNS_DIR / f"{_safe_id(run_id)}.json"
    if path.is_file():
        path.unlink()
        return True
    return False


def _safe_id(run_id: str) -> str:
    """Guard the filename against path traversal — keep the run_id charset only."""
    return Path(str(run_id)).name
