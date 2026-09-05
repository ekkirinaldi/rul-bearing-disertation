"""The paper-style algorithm diagrams must exist and stay slide-ready.

`make diagrams` (tools/render_diagrams.py) renders one PNG per method into
assets/diagrams/. These tests keep the committed set complete and catch a
diagram that regressed to an unusable size; the visual QA itself is manual.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.render_diagrams import DIAGRAMS, KIND, OUT  # noqa: E402


@pytest.mark.parametrize("name", sorted(DIAGRAMS))
def test_diagram_rendered_and_sized(name):
    target = OUT / f"{name}.png"
    assert target.exists(), f"run `make diagrams` — missing {target.name}"
    with Image.open(target) as image:
        width, height = image.size
    assert width >= 1800, f"{name}: {width}px wide — rendered below 200 dpi?"
    assert height < width, f"{name}: portrait aspect will not fit the slide band"


def test_every_diagram_is_referenced_by_the_deck():
    spec = (ROOT / "content" / "sidang-terbuka.yaml").read_text(encoding="utf-8")
    unused = [n for n in DIAGRAMS if f"diagrams/{n}.png" not in spec]
    assert unused == [], f"diagrams rendered but not on any slide: {unused}"


def test_palette_kinds_are_wellformed():
    for kind, colors in KIND.items():
        assert set(colors) == {"fc", "ec"}, kind
        for value in colors.values():
            assert value.startswith("#") and len(value) == 7, (kind, value)


def test_labels_keep_component_nouns_english():
    """Adjusted-translation register: named components stay English
    (input, output, gate, attention, hidden state, matrix memory); only
    verbs, connectors, and KBBI cognates for generic operations (proyeksi,
    konvolusi, normalisasi) are Indonesian. The calques below were flagged
    by the user on the rendered diagrams and must not come back."""
    import re

    source = (ROOT / "tools" / "render_diagrams.py").read_text(encoding="utf-8")
    calques = re.findall(
        r"\b(masukan|keluaran|gerbang|atensi|jangkauan reseptif|terdilatasi|"
        r"pohon keputusan|hutan acak|regresi logistik)\b",
        source, flags=re.IGNORECASE,
    )
    assert calques == [], f"translated component nouns in the diagram labels: {calques}"
    assert re.search(r"\bvia\b", source) is None, "`via` is not an Indonesian connector"
