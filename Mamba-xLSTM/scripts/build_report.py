"""Build an HTML + PDF dissertation report from one or more run directories.

Examples
--------

Single run::

    python scripts/build_report.py \
        --runs results/runs/20260421_*_mamba_xlstm_phm2012_s42 \
        --out-html results/report.html \
        --out-pdf  results/report.pdf

All runs (ablation comparison)::

    python scripts/build_report.py \
        --runs results/runs/* \
        --title "Mamba-xLSTM ablations on PHM2012" \
        --out-html results/ablation_report.html \
        --out-pdf  results/ablation_report.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "src"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mxlstm.reporting.report import build_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True,
                   help="One or more run directories (the parent of summary.json).")
    p.add_argument("--name", type=str, default=None,
                   help="Report basename (default: 'report' for one run, "
                        "'<run_id>' if a single run is given).")
    p.add_argument("--out-html", type=Path, default=None,
                   help="Override HTML path. Defaults to results/reports/<name>.html.")
    p.add_argument("--out-pdf", type=Path, default=None,
                   help="Override PDF path. Defaults to results/reports/<name>.pdf. "
                        "Pass --no-pdf to skip the PDF.")
    p.add_argument("--no-pdf", action="store_true",
                   help="Produce HTML only.")
    p.add_argument("--reports-dir", type=Path, default=Path("results/reports"),
                   help="Default cross-run reports directory (used when "
                        "--out-html / --out-pdf are not supplied).")
    p.add_argument("--title", type=str, default="Mamba-xLSTM dissertation report")
    p.add_argument("--subtitle", type=str,
                   default="Bearing remaining-useful-life prediction")
    p.add_argument(
        "--subtitle-line",
        action="append",
        default=[],
        metavar="TEXT",
        help="Cover subtitle as a bullet (repeat once per algorithm). "
             "Omits comma-joined subtitle block; the --subtitle line is shown as tagline.",
    )
    p.add_argument("--no-ablation", action="store_true",
                   help="Skip the cross-run aggregation section.")
    p.add_argument("--sequential", action="store_true",
                   help="When two runs are given, emit the classic per-run sections "
                        "instead of the side-by-side comparison layout.")
    p.add_argument("--gallery-limit", type=int, default=12,
                   help="Maximum per-bearing figures to include per gallery. "
                        "Use 0 to include every available bearing.")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace, valid_runs: list[Path]) -> tuple[Path, Path | None]:
    if args.name:
        base = args.name
    elif len(valid_runs) == 1:
        base = valid_runs[0].name
    else:
        base = "report"
    out_html = args.out_html or (args.reports_dir / f"{base}.html")
    if args.no_pdf:
        out_pdf = None
    else:
        out_pdf = args.out_pdf or (args.reports_dir / f"{base}.pdf")
    return out_html, out_pdf


def main() -> None:
    args = parse_args()

    valid_runs = []
    for r in args.runs:
        if not r.exists():
            print(f"[warn] skipping missing path {r}", file=sys.stderr)
            continue
        if not (r / "summary.json").exists():
            print(f"[warn] {r} has no summary.json — skipping", file=sys.stderr)
            continue
        valid_runs.append(r)

    if not valid_runs:
        print("No runnable summaries found. Did you train anything yet?", file=sys.stderr)
        sys.exit(1)

    out_html, out_pdf = _resolve_paths(args, valid_runs)

    print(f"Building report from {len(valid_runs)} run(s):")
    for r in valid_runs:
        print(f"  - {r}")
    print(f"  → HTML: {out_html}")
    if out_pdf is not None:
        print(f"  → PDF:  {out_pdf}")

    result = build_report(
        run_dirs=valid_runs,
        out_html=out_html,
        out_pdf=out_pdf,
        title=args.title,
        subtitle=args.subtitle,
        subtitle_lines=args.subtitle_line or None,
        include_ablation=not args.no_ablation,
        side_by_side=False if args.sequential else None,
        gallery_limit=None if args.gallery_limit == 0 else args.gallery_limit,
    )
    print(f"HTML report: {result['html']}")
    if "pdf" in result:
        if result["pdf"]:
            print(f"PDF report:  {result['pdf']}  (backend: {result['pdf_backend']})")
        else:
            print(f"PDF skipped:  {result.get('pdf_error', 'PDF backend unavailable')}")
    print(f"Figures:      {result['figures_dir']} ({result['n_figures']} generated)")


if __name__ == "__main__":
    main()
