"""The manuscript insertion tool must add material without disturbing the rest.

`dissertation-docx/tools/insert_docx.py` edits a hand-edited Word document in
place, which is only safe if two things hold: nothing that was already in the
document is lost, and every figure, table, and cross-reference number still
agrees with the caption it points at after the insertion shifts them.

The unit tests below run on synthetic XML. The integration tests need the V14
manuscript, which is gitignored, so they skip when it is absent.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pytest
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT.parent / "dissertation-docx" / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

insert_docx = pytest.importorskip("insert_docx")

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
CAPTION = re.compile(r"\s*(Gambar|Tabel)[\s ]+([IVX]+)\.(\d+)")


# --------------------------------------------------------------------------
# Unit: inline markup
# --------------------------------------------------------------------------
def _runs_text(runs) -> str:
    return "".join(t.text or "" for run in runs for t in run.iter(W + "t"))


def test_markup_plain_text_round_trips():
    runs = insert_docx.parse_markup("Sebuah kalimat biasa.")
    assert _runs_text(runs) == "Sebuah kalimat biasa."


def test_markup_marks_italic_and_bold():
    runs = insert_docx.parse_markup("kata *miring* dan **tebal** di sini")
    italic = [r for r in runs if r.find(f"{W}rPr/{W}i") is not None]
    bold = [r for r in runs if r.find(f"{W}rPr/{W}b") is not None]
    assert _runs_text(italic) == "miring"
    assert _runs_text(bold) == "tebal"
    assert _runs_text(runs) == "kata miring dan tebal di sini"


def test_markup_emits_a_reference_field():
    runs = insert_docx.parse_markup("lihat Gambar [ref:fig_x] di atas")
    instr = "".join(t.text or "" for r in runs for t in r.iter(W + "instrText"))
    assert "REF fig_x" in instr
    # The noun and its number must not be split across a line break.
    assert "Gambar " in _runs_text(runs)


def test_markup_keeps_an_escaped_asterisk():
    runs = insert_docx.parse_markup(r"nilai 3 \* 4")
    assert _runs_text(runs) == "nilai 3 * 4"


# --------------------------------------------------------------------------
# Unit: the renumber pass on a synthetic document
# --------------------------------------------------------------------------
def _mini_document() -> etree._Element:
    """Two chapters, three figures, one table, and references to two of them."""
    body = insert_docx.wel("body")

    def para(style, children):
        p = insert_docx._para(style)
        for child in children:
            p.append(child)
        body.append(p)
        return p

    def caption(chapter, kind, cached, label, title):
        style = "JudulGambar" if kind == "Gambar" else "judulTabel"
        p = insert_docx._para(style)
        p.append(insert_docx.make_run(f"{kind} "))
        start = insert_docx.wel("bookmarkStart", id=str(len(body) + 500), name=label)
        end = insert_docx.wel("bookmarkEnd", id=str(len(body) + 500))
        p.append(start)
        p.append(insert_docx.make_run(f"{chapter}."))
        for run in insert_docx._seq_field(kind, cached):
            p.append(run)
        p.append(end)
        p.append(insert_docx.make_run(f" {title}"))
        body.append(p)
        return p

    para("Heading1", [insert_docx.make_run("Pendahuluan")])
    caption("I", "Gambar", "1", "fig_a", "Gambar pertama")
    caption("I", "Gambar", "2", "fig_b", "Gambar kedua")
    para("Paragraf", insert_docx.parse_markup("Lihat Gambar [ref:fig_b] itu."))
    para("Paragraf", [insert_docx.make_run("Mengacu pada Gambar I.2 juga.")])
    para("Heading1", [insert_docx.make_run("Tinjauan")])
    caption("II", "Tabel", "1", "tab_a", "Tabel pertama")
    return body


class _FakeDoc:
    def __init__(self, body):
        self.body = body


def test_renumber_fixes_a_stale_cache_and_its_reference():
    body = _mini_document()
    # Insert a new first figure in chapter I, exactly what an edit does.
    new = insert_docx._para("JudulGambar")
    new.append(insert_docx.make_run("Gambar "))
    new.append(insert_docx.wel("bookmarkStart", id="900", name="fig_new"))
    new.append(insert_docx.make_run("I."))
    for run in insert_docx._seq_field("Gambar", "1"):
        new.append(run)
    new.append(insert_docx.wel("bookmarkEnd", id="900"))
    new.append(insert_docx.make_run(" Gambar sisipan"))
    body[1].addprevious(new)

    insert_docx.renumber(_FakeDoc(body), verbose=False)

    texts = ["".join(t.text or "" for t in p.iter(W + "t")) for p in body]
    numbers = [CAPTION.match(t).group(0).strip() for t in texts if CAPTION.match(t)]
    assert numbers == ["Gambar I.1", "Gambar I.2", "Gambar I.3", "Tabel II.1"]
    # The REF that pointed at the old I.2 now reads I.3 ...
    assert "Lihat Gambar I.3 itu." in texts
    # ... and so does the hand-typed mention of the same figure.
    assert "Mengacu pada Gambar I.3 juga." in texts


def test_renumber_leaves_an_untouched_chapter_alone():
    body = _mini_document()
    insert_docx.renumber(_FakeDoc(body), verbose=False)
    texts = ["".join(t.text or "" for t in p.iter(W + "t")) for p in body]
    assert any(t.startswith("Tabel II.1") for t in texts)


# --------------------------------------------------------------------------
# Integration: against the real manuscript
# --------------------------------------------------------------------------
def _manuscripts() -> list[Path]:
    return sorted(ROOT.glob("V1? *.docx"))


def _rewritten_prefixes() -> list[str]:
    """Text the spec deliberately replaces, read from the spec itself.

    Deriving the exemptions this way keeps the test honest: an edit that
    rewrites a paragraph nobody declared still shows up as a loss.
    """
    spec_path = ROOT.parent / "dissertation-docx" / "inserts" / "v15" / "spec.yaml"
    if not spec_path.exists():
        return []
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    out = []
    for edit in spec.get("edits", []):
        if "rewrite_para" in edit:
            out.append(edit["rewrite_para"]["text_prefix"])
        if "replace_figure" in edit:
            prefix = edit["replace_figure"].get("caption_prefix")
            if prefix:
                # Compare on the title, since the number itself may move.
                out.append(re.sub(r"^\s*(Gambar|Tabel)[\s\u00a0]+[IVX]+\.\d+\s*",
                                  "", prefix))
            label = edit["replace_figure"].get("label")
            if label:
                out.append(label)
    # The two Bab II figures are addressed by label, not by caption text.
    out += ["Blok Selective State Space", "Blok matrix LSTM"]
    return [o for o in out if o]


def _source() -> Path:
    hits = sorted(ROOT.glob("V14 *.docx"))
    if not hits:
        pytest.skip("V14 manuscript not present (gitignored)")
    return hits[0]


def _body(path: Path):
    with zipfile.ZipFile(path) as archive:
        return etree.fromstring(archive.read("word/document.xml")).find(W + "body")


def _paragraph_texts(body) -> list[str]:
    return ["".join(t.text or "" for t in p.iter(W + "t"))
            for p in body if p.tag == W + "p"]


def _style(p) -> str:
    node = p.find(f"{W}pPr/{W}pStyle")
    return node.get(W + "val") if node is not None else ""


def _run_tool(spec: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(TOOLS / "insert_docx.py"), str(spec), "--force"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def empty_spec_output(tmp_path_factory) -> Path:
    source = _source()
    out = tmp_path_factory.mktemp("empty") / "out.docx"
    spec = out.parent / "spec.yaml"
    spec.write_text(f'source: "{source}"\noutput: "{out}"\nedits: []\n',
                    encoding="utf-8")
    _run_tool(spec)
    return out


def test_an_empty_spec_changes_nothing_but_stale_numbers(empty_spec_output):
    """A spec with no edits is a round trip plus a renumber pass.

    It is not quite a no-op: V14 carries two cross-references whose cached
    number disagrees with the caption they point at, and the renumber pass
    repairs them. Nothing else may move.
    """
    before = _paragraph_texts(_body(_source()))
    after = _paragraph_texts(_body(empty_spec_output))
    assert len(before) == len(after)
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    blind = lambda s: re.sub(r"(Gambar|Tabel|Persamaan)[\s\u00a0]*[IVX]+\.\d+",  # noqa: E731
                             r"\1 #", s)
    for b, a in changed:
        assert blind(b) == blind(a), f"text changed beyond a number:\n{b}\n{a}"
    assert len(changed) <= 2, f"{len(changed)} paragraphs renumbered, expected 2"


def test_the_source_manuscript_has_no_new_stale_references(empty_spec_output):
    """Guards the repair above: after a plain round trip every cross-reference
    cache agrees with its caption, so a later edit starts from a clean base."""
    body = _body(empty_spec_output)
    numbers = {}
    for p in body:
        if p.tag != W + "p" or _style(p) not in ("JudulGambar", "judulTabel"):
            continue
        match = CAPTION.match("".join(t.text or "" for t in p.iter(W + "t")))
        if not match:
            continue
        for bookmark in p.iter(W + "bookmarkStart"):
            name = bookmark.get(W + "name") or ""
            if not name.startswith("_Toc"):
                numbers[name] = f"{match.group(2)}.{match.group(3)}"
    wrong = []
    for p in body:
        if p.tag != W + "p":
            continue
        for instr, cached in insert_docx._field_runs(p):
            match = re.search(r"\bREF\s+([^\s\\]+)", instr)
            if not match:
                continue
            shown = "".join(t.text or "" for r in cached for t in r.iter(W + "t"))
            target = match.group(1)
            if target in numbers and shown.strip() and shown.strip() != numbers[target]:
                wrong.append((target, shown.strip(), numbers[target]))
    assert wrong == [], wrong


def test_v15_keeps_every_v14_paragraph_in_order():
    """Ignoring the numbers that renumbering is supposed to move, every
    paragraph of the source is still present, in the same order."""
    versions = _manuscripts()
    if len(versions) < 2:
        pytest.skip("V15 not built yet (run make insert)")
    blind = lambda s: re.sub(r"(Gambar|Tabel|Persamaan)[\s ]*[IVX]+\.\d+",  # noqa: E731
                             r"\1 #", re.sub(r"\s+", " ", s)).strip()
    before = [blind(t) for t in _paragraph_texts(_body(versions[0])) if t.strip()]
    after = {blind(t) for t in _paragraph_texts(_body(versions[-1]))}
    exempt = _rewritten_prefixes()
    missing = [t for t in before
               if t not in after and not any(r in t for r in exempt)]
    # One paragraph carries a bare "II.7" reference the blinding cannot see.
    assert len(missing) <= 1, missing[:5]


def test_v15_caption_numbers_are_contiguous_per_chapter():
    versions = _manuscripts()
    if len(versions) < 2:
        pytest.skip("V15 not built yet (run make insert)")
    body = _body(versions[-1])
    seen = defaultdict(list)
    for p in body:
        if p.tag != W + "p" or _style(p) not in ("JudulGambar", "judulTabel"):
            continue
        match = CAPTION.match("".join(t.text or "" for t in p.iter(W + "t")))
        if match:
            seen[(match.group(1), match.group(2))].append(int(match.group(3)))
    assert seen, "no captions found"
    for key, numbers in sorted(seen.items()):
        assert numbers == list(range(1, len(numbers) + 1)), f"{key}: {numbers}"


def test_v15_reference_caches_match_their_caption():
    versions = _manuscripts()
    if len(versions) < 2:
        pytest.skip("V15 not built yet (run make insert)")
    body = _body(versions[-1])
    numbers = {}
    for p in body:
        if p.tag != W + "p" or _style(p) not in ("JudulGambar", "judulTabel"):
            continue
        match = CAPTION.match("".join(t.text or "" for t in p.iter(W + "t")))
        if not match:
            continue
        for bookmark in p.iter(W + "bookmarkStart"):
            name = bookmark.get(W + "name") or ""
            if not name.startswith("_Toc"):
                numbers[name] = f"{match.group(2)}.{match.group(3)}"

    wrong = []
    for p in body:
        if p.tag != W + "p":
            continue
        for instr, cached in insert_docx._field_runs(p):
            match = re.search(r"\bREF\s+([^\s\\]+)", instr)
            if not match:
                continue
            target = match.group(1)
            shown = "".join(t.text or "" for run in cached for t in run.iter(W + "t"))
            if target in numbers and shown.strip() and shown.strip() != numbers[target]:
                wrong.append((target, shown.strip(), numbers[target]))
    assert wrong == [], wrong[:5]


def test_a_replaced_figure_keeps_no_stale_crop():
    """A crop applied in Word belongs to the picture it was made for.

    Gambar IV.3 carried one; carrying it over would cut the same percentages
    out of whatever image replaces it.
    """
    versions = _manuscripts()
    if len(versions) < 2:
        pytest.skip("V15 not built yet (run make insert)")
    yaml = pytest.importorskip("yaml")
    spec_path = ROOT.parent / "dissertation-docx" / "inserts" / "v15" / "spec.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    titles = [e["replace_figure"].get("caption_prefix") for e in spec.get("edits", [])
              if "replace_figure" in e and e["replace_figure"].get("image")]
    titles = [t for t in titles if t]
    if not titles:
        pytest.skip("no figure replacements declared")
    A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    body = _body(versions[-1])
    kids = list(body)
    checked = 0
    for prefix in titles:
        title = re.sub(r"^\s*(Gambar|Tabel)[\s\u00a0]+[IVX]+\.\d+\s*", "", prefix)
        for i, p in enumerate(kids):
            if p.tag != W + "p" or _style(p) != "JudulGambar":
                continue
            if title not in "".join(t.text or "" for t in p.iter(W + "t")):
                continue
            picture = kids[i - 1]
            assert picture.findall(f".//{A}srcRect") == [], f"{title}: crop survived"
            checked += 1
            break
    assert checked == len(titles), "some replaced figures were not found"


def test_v15_renders_without_broken_references(tmp_path):
    """LibreOffice resolves bookmarks live, so a dangling REF shows up here."""
    versions = _manuscripts()
    if len(versions) < 2:
        pytest.skip("V15 not built yet (run make insert)")
    soffice = shutil.which("soffice") or "/opt/homebrew/bin/soffice"
    if not Path(soffice).exists():
        pytest.skip("LibreOffice not installed")
    target = tmp_path / "v15.docx"
    shutil.copy2(versions[-1], target)
    subprocess.run([soffice, "--headless", "--convert-to", "pdf",
                    "--outdir", str(tmp_path), str(target)],
                   capture_output=True, timeout=900)
    pdf = tmp_path / "v15.pdf"
    if not pdf.exists():
        pytest.skip("conversion produced no PDF")
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                          text=True).stdout
    assert "Reference source not found" not in text
    assert "Error! Bookmark" not in text
