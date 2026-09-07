"""Draft-derived artwork must exist, stay legible, and stay credited.

`make draft-assets` (tools/extract_draft_media.py) crops the SKF training
material out of Pak Toto's annotated draft into assets/skf/. The draft itself
is gitignored, so the PNGs are committed; these tests keep the set complete
and make sure every slide that shows one credits its source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "content" / "sidang-terbuka.yaml"


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def _referenced() -> list[str]:
    return sorted(set(re.findall(r"assets/skf/([\w-]+\.png)", _spec())))


@pytest.mark.parametrize("name", _referenced())
def test_referenced_skf_asset_exists_and_is_legible(name):
    target = ROOT / "assets" / "skf" / name
    assert target.exists(), f"run `make draft-assets` — missing {target.name}"
    with Image.open(target) as image:
        width, height = image.size
    # Small draft pictures are shown as insets; anything below this is a
    # thumbnail that will blur on the projector.
    assert min(width, height) >= 140, f"{name}: {width}x{height}px is too small"


def test_every_committed_skf_asset_is_used():
    committed = sorted(p.name for p in (ROOT / "assets" / "skf").glob("*.png"))
    unused = [n for n in committed if n not in _referenced()]
    assert unused == [], f"assets/skf/ files no slide uses: {unused}"


def test_every_slide_with_skf_artwork_credits_skf():
    slides = re.split(r"\n  - layout:", _spec())
    for slide in slides:
        if "assets/skf/" not in slide:
            continue
        title = re.search(r'title: "?([^"\n]+)', slide)
        assert "SKF" in slide.split("type: source", 1)[-1] or "PT SKF Indonesia" in slide, (
            f"slide {title.group(1) if title else '?'!r} shows SKF artwork without a credit"
        )
        assert "type: source" in slide, (
            f"slide {title.group(1) if title else '?'!r} shows SKF artwork without a `source` block"
        )
