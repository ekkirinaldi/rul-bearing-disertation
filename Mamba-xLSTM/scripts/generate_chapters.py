"""Orchestrator: generate Bab III, IV, dan V disertasi sebagai file DOCX terpisah.

Usage (from repo root):
    Mamba-xLSTM/.venv/bin/python Mamba-xLSTM/scripts/generate_chapters.py

Output (di root workspace):
    Bab3_Pengembangan_Model.docx
    Bab4_Hasil_dan_Pembahasan.docx
    Bab5_Kesimpulan_dan_Saran.docx
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts dir to sys.path so that "from chapters import ..." works.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from chapters._docx_utils import write_chapter  # noqa: E402
from chapters import bab3, bab4, bab5  # noqa: E402

REPO_ROOT = _HERE.parents[1]


def main() -> None:
    targets = [
        ("Bab3_Pengembangan_Model.docx", bab3.build),
        ("Bab4_Hasil_dan_Pembahasan.docx", bab4.build),
        ("Bab5_Kesimpulan_dan_Saran.docx", bab5.build),
    ]

    print("=" * 72)
    print("Generating dissertation chapters as DOCX")
    print("=" * 72)

    summaries = []
    for filename, builder in targets:
        out_path = REPO_ROOT / filename
        info = write_chapter(builder, out_path)
        summaries.append((filename, info))
        print(
            f"  Wrote {filename}  "
            f"(paragraphs={info['paragraphs']}, tables={info['tables']}, "
            f"images={info['inline_images']})"
        )

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    for filename, info in summaries:
        print(f"  {filename}")
        print(f"    path:        {info['path']}")
        print(f"    paragraphs:  {info['paragraphs']}")
        print(f"    tables:      {info['tables']}")
        print(f"    images:      {info['inline_images']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
