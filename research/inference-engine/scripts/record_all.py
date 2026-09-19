#!/usr/bin/env python3
"""Record every dataset+bearing run sequentially into the run library.

Usage:
    ../Mamba-xLSTM/.venv/bin/python scripts/record_all.py [--force]

By default already-recorded runs are skipped; --force re-records everything.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Mamba-xLSTM" / "src"))

from app import replay_store as rs  # noqa: E402


def main() -> int:
    force = "--force" in sys.argv[1:]
    targets = rs.list_targets()
    print(f"Recording {len(targets)} runs (skip_existing={not force})")
    t0 = time.time()
    last = {"line": ""}

    def item_cb(done: int, total: int, dataset: str, bearing: str, phase: str) -> None:
        if phase == "start":
            print(f"[{done + 1}/{total}] {dataset}/{bearing} … ", end="", flush=True)
        elif phase == "skipped":
            print("skip (already recorded)")
        elif phase == "done":
            print("done")
        elif phase == "error":
            print("ERROR")

    def frame_cb(idx, total, dataset, bearing, t, n) -> None:
        line = f"{t}/{n}"
        if line != last["line"]:
            last["line"] = line
            print(f"\r[{idx + 1}/{total}] {dataset}/{bearing} … {t}/{n}", end="", flush=True)

    results = rs.record_all(skip_existing=not force, item_cb=item_cb, frame_cb=frame_cb)
    dt = time.time() - t0

    ok = [r for r in results if r.get("run_id") and not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    errored = [r for r in results if r.get("error")]
    print(f"\nDone in {dt/60:.1f} min — recorded {len(ok)}, skipped {len(skipped)}, errors {len(errored)}")
    for r in ok:
        print(f"  + {r['dataset']}/{r['bearing']}: {r['n_drops']} drops ({r['run_id']})")
    for r in errored:
        print(f"  ! {r['dataset']}/{r['bearing']}: {r['error']}")
    return 1 if errored else 0


if __name__ == "__main__":
    sys.exit(main())
