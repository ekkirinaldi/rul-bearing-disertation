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

from tools.render_diagrams import CHARTS, DIAGRAMS, KIND, OUT  # noqa: E402


@pytest.mark.parametrize("name", sorted({**DIAGRAMS, **CHARTS}))
def test_diagram_rendered_and_sized(name):
    target = OUT / f"{name}.png"
    assert target.exists(), f"run `make diagrams` — missing {target.name}"
    with Image.open(target) as image:
        width, height = image.size
    assert width >= 1800, f"{name}: {width}px wide — rendered below 200 dpi?"
    assert height < width, f"{name}: portrait aspect will not fit the slide band"


def test_every_diagram_is_referenced_by_the_deck():
    spec = (ROOT / "content" / "sidang-terbuka.yaml").read_text(encoding="utf-8")
    unused = [n for n in {**DIAGRAMS, **CHARTS} if f"diagrams/{n}.png" not in spec]
    assert unused == [], f"diagrams rendered but not on any slide: {unused}"


def test_charts_and_diagrams_do_not_share_names():
    """A domain chart has no input contract, so it must not shadow a diagram."""
    assert set(CHARTS).isdisjoint(DIAGRAMS)


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

    pattern = (
        r"\b(masukan|keluaran|gerbang|atensi|jangkauan reseptif|terdilatasi|"
        r"pohon keputusan|hutan acak|regresi logistik|bobot|derau|galat|"
        r"jendela|fitur|sampel|proyeksi|konvolusi|normalisasi|rekonstruksi|"
        r"atribusi|kesenjangan|cincin (luar|dalam)|bantalan|sangkar|"
        r"lini produksi|lini otomasi|amplop|dasbor|laju cuplik)\b"
    )
    for path in (ROOT / "tools" / "render_diagrams.py",
                 ROOT / "content" / "sidang-terbuka.yaml"):
        source = path.read_text(encoding="utf-8")
        calques = re.findall(pattern, source, flags=re.IGNORECASE)
        assert calques == [], f"{path.name}: translated technical nouns {calques}"
        assert re.search(r"\bvia\b", source) is None, (
            f"{path.name}: `via` is not an Indonesian connector"
        )


def test_foreign_spans_render_as_italics():
    """Research writing italicises foreign terms, figure labels included."""
    from tools.render_diagrams import fr

    assert fr("[[receptive field]] menutup jendela") == (
        "$\\mathit{receptive\\ field}$ menutup jendela"
    )
    assert fr("[[gate]]", bold=True) == "$\\boldsymbol{gate}$"
    assert fr("[[cross-feature attention]]").count("$") == 4   # hyphen stays upright
    assert fr("$h \\in \\mathbb{R}^{128}$") == "$h \\in \\mathbb{R}^{128}$"


def test_every_diagram_declares_its_input():
    """The strip is the only place a viewer learns what the model eats, so a
    new diagram must not be able to ship without one."""
    from tools.render_diagrams import INPUT_SPEC

    assert set(INPUT_SPEC) == set(DIAGRAMS), (
        f"INPUT_SPEC and DIAGRAMS disagree: "
        f"{set(INPUT_SPEC) ^ set(DIAGRAMS)}"
    )
    for name, fields in INPUT_SPEC.items():
        assert len(fields) == 3, f"{name}: expected (besaran, sensor, tensor)"
        for field in fields:
            assert field.strip(), f"{name}: an empty contract field"


def test_input_spec_states_a_concrete_shape():
    """Every contract has to end in something countable — a tensor shape, a
    window length, a sample count. Prose alone does not distinguish models."""
    import re

    from tools.render_diagrams import INPUT_SPEC

    for name, fields in INPUT_SPEC.items():
        assert re.search(r"\d", " ".join(fields)), (
            f"{name}: the contract states no countable quantity"
        )


def test_strip_fits_the_canvas_width():
    """The bar is drawn at a fixed 11,2 in; an over-long line would spill out
    of the rounded rectangle instead of wrapping."""
    from tools.render_diagrams import _CHAR_IN, _SPAN, INPUT_SPEC, _strip_layout

    for name in INPUT_SPEC:
        label, fs, _ = _strip_layout(name)
        for line in label.split("\n"):
            width = len(_SPAN.sub(r"\1", line)) * fs * _CHAR_IN
            assert width <= 11.2, f"{name}: strip line is {width:.2f} in wide"
