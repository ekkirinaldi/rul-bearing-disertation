"""Idempotent cleaner for ``results/``.

Enforces the canonical layout::

    results/
    ├── runs/<timestamp>_<run_id>/   per-run outputs (figures/, logs/, ...)
    ├── reports/                     cross-run HTML/PDF reports
    └── tables/                      cross-run aggregated tables

What it does (safe by default — pass ``--apply`` to actually mutate):

  1. Move any ``report*.{html,pdf}`` files at ``results/`` root into
     ``results/reports/``.
  2. Delete top-level empty stub folders ``results/{checkpoints,figures,tables}``
     (they're recreated lazily by writers when actually needed; the
     persistent cross-run ``reports/`` and ``tables/`` are kept once they
     have content).
  3. Recursively remove empty subfolders inside ``results/runs/*/``
     (stale ``figures/``, ``tables/`` left over from earlier scaffolding).
  4. Optionally delete entire run directories that contain no
     ``summary.json`` (they're failed/empty runs); pass ``--drop-empty-runs``.

Usage::

    python scripts/clean_results.py                 # dry run
    python scripts/clean_results.py --apply         # do it
    python scripts/clean_results.py --apply --drop-empty-runs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# Folders we never delete from results/ root once they have content.
_TOP_LEVEL_KEEP = {"runs", "reports", "tables"}


def _print(msg: str, *, apply: bool) -> None:
    prefix = "[do]  " if apply else "[dry] "
    print(prefix + msg)


def _move_root_reports(root: Path, apply: bool) -> int:
    moved = 0
    reports_dir = root / "reports"
    for ext in ("html", "pdf"):
        for f in root.glob(f"report*.{ext}"):
            dest = reports_dir / f.name
            _print(f"move {f}  →  {dest}", apply=apply)
            if apply:
                reports_dir.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest.unlink()
                shutil.move(str(f), str(dest))
            moved += 1
    return moved


def _is_empty(p: Path) -> bool:
    if not p.is_dir():
        return False
    return next(p.iterdir(), None) is None


def _prune_empty_dirs(root: Path, apply: bool, *, keep_root: bool = True) -> int:
    """Bottom-up walk; remove empty directories. ``root`` itself is kept by default."""
    removed = 0
    if not root.exists():
        return 0
    # Walk bottom-up so we remove leaves before their parents.
    for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if not p.is_dir():
            continue
        if keep_root and p == root:
            continue
        if _is_empty(p):
            _print(f"rmdir {p}", apply=apply)
            if apply:
                p.rmdir()
            removed += 1
    return removed


def _drop_root_stubs(root: Path, apply: bool) -> int:
    """Delete top-level empty stub folders that are *not* in the keep-list."""
    removed = 0
    if not root.exists():
        return 0
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in _TOP_LEVEL_KEEP:
            continue
        if _is_empty(p):
            _print(f"rmdir {p}  (top-level stub)", apply=apply)
            if apply:
                p.rmdir()
            removed += 1

    # Also remove canonical top-level dirs if they exist *and* are empty.
    for name in _TOP_LEVEL_KEEP:
        p = root / name
        if p.is_dir() and _is_empty(p):
            _print(f"rmdir {p}  (empty canonical dir)", apply=apply)
            if apply:
                p.rmdir()
            removed += 1
    return removed


def _drop_failed_runs(runs_root: Path, apply: bool) -> int:
    removed = 0
    if not runs_root.exists():
        return 0
    for run in sorted(runs_root.iterdir()):
        if not run.is_dir():
            continue
        if not (run / "summary.json").exists():
            _print(f"rm -r {run}  (no summary.json)", apply=apply)
            if apply:
                shutil.rmtree(run)
            removed += 1
    return removed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results"),
                   help="Results root (default: results/).")
    p.add_argument("--apply", action="store_true",
                   help="Actually mutate the filesystem (default: dry run).")
    p.add_argument("--drop-empty-runs", action="store_true",
                   help="Also delete entire run directories that have no summary.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root.resolve()
    if not root.exists():
        print(f"results root '{root}' does not exist; nothing to do.", file=sys.stderr)
        return

    print(f"Cleaning {root}  ({'APPLY' if args.apply else 'DRY RUN'})")
    moved = _move_root_reports(root, args.apply)

    # Prune inside per-run directories first so root cleanup sees them empty if applicable.
    runs_dir = root / "runs"
    pruned_runs = _prune_empty_dirs(runs_dir, args.apply, keep_root=True)

    dropped = 0
    if args.drop_empty_runs:
        dropped = _drop_failed_runs(runs_dir, args.apply)

    pruned_root = _drop_root_stubs(root, args.apply)

    print(f"\nSummary  (mode={'apply' if args.apply else 'dry-run'}):")
    print(f"  reports moved into results/reports/   : {moved}")
    print(f"  empty subdirs pruned under runs/      : {pruned_runs}")
    print(f"  failed run directories removed        : {dropped}")
    print(f"  top-level empty stubs removed         : {pruned_root}")
    if not args.apply:
        print("\nRe-run with --apply to perform these changes.")


if __name__ == "__main__":
    main()
