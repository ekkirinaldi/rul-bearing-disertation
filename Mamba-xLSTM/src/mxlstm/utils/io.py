"""IO helpers: HI cache (npz) read/write, run directory bootstrap."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def make_run_dir(root: str | Path, run_id: str | None = None) -> Path:
    """Create ``<root>/runs/<timestamp>_<run_id>/`` and return its path.

    Only ``checkpoints/`` is created eagerly because Lightning's
    ``ModelCheckpoint`` needs the directory to exist when callbacks are
    constructed. ``figures/``, ``logs/``, ``csv_logs/`` and ``interp/``
    are created lazily by the components that write into them, so a run
    that crashes early or runs without figures will not leave behind
    empty folders. Cross-run outputs (``reports/``, ``tables/``) live at
    ``<root>/`` not here.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{run_id}" if run_id else ts
    run_dir = Path(root) / "runs" / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_hi_npz(
    path: str | Path,
    hi: np.ndarray,
    feature_names: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist HI ``(T, F)`` + feature names + optional extras."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"hi": hi.astype(np.float32)}
    if extra:
        for k, v in extra.items():
            payload[k] = np.asarray(v)
    np.savez_compressed(p, **payload)
    p.with_suffix(".names.json").write_text(json.dumps(feature_names, indent=2))


def load_hi_npz(path: str | Path) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    p = Path(path)
    with np.load(p, allow_pickle=False) as f:
        hi = f["hi"].astype(np.float32)
        extras = {k: f[k] for k in f.files if k != "hi"}
    names_path = p.with_suffix(".names.json")
    names = json.loads(names_path.read_text()) if names_path.exists() else []
    return hi, names, extras
