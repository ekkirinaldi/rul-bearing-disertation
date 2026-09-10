#!/usr/bin/env python3
"""Insert new material into a hand-edited dissertation DOCX, in place.

The manuscript of record is a Word file that has been edited by hand, so the
LaTeX-to-DOCX pipeline (`make babN`) cannot be used to add a section: it would
regenerate the chapter and discard those edits. This tool takes the document as
it stands, applies a declarative list of insertions, and writes a new file.

    python3 tools/insert_docx.py inserts/v15/spec.yaml [--dry-run] [--force]

The spec (YAML, paths relative to it) names the source and output documents and
a list of edits. Every edit anchors on an existing paragraph and adds styled
blocks after or before it:

    edits:
      - anchor: {text_prefix: "Di antara banyaknya komponen", position: before}
        blocks:
          - para: "Teks dengan *miring*, **tebal**, dan Gambar [ref:fig_x]."
          - figure: {image: a.png, width_fraction: 0.9, label: fig_x, caption: "..."}
          - table:  {label: tab_x, caption: "...", columns: [...], widths: [...],
                     rows: [[...]], header_repeat: true}
      - replace_figure: {label: fig_y, image: b.png, caption: "..."}
      - bib_entry: "Nama, I. (2022): Judul, ..."

Structure follows RULES.md: template styles only (Heading3, Paragraf, Gambar,
JudulGambar, judulTabel, IsiTabel, Daftarpustaka), one empty `Paragraf` between
blocks, captions built as a literal Roman prefix plus a `SEQ` field wrapped in a
bookmark, cross-references as `REF` fields, tables with a fixed layout and the
ITB grid border.

Because a figure inserted mid-chapter shifts every later number, the tool ends
with a renumber pass that rewrites the cached result of every SEQ field and of
every REF that points at one, so the document reads correctly before Word ever
recalculates fields.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import shutil
import struct
import sys
import tempfile
import unicodedata
from pathlib import Path

import yaml
from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from pack import pack  # noqa: E402
from restyle import (  # noqa: E402
    _apply_itb_tbl_borders,
    make_run,
    para_text,
    sanitize_bookmark,
    wel,
)
from unpack import unpack  # noqa: E402

WNS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ANS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PICNS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
WPNS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
CTNS = "http://schemas.openxmlformats.org/package/2006/content-types"
PRNS = "http://schemas.openxmlformats.org/package/2006/relationships"
W = "{%s}" % WNS
R = "{%s}" % RNS
A = "{%s}" % ANS
WP = "{%s}" % WPNS
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
EMU_PER_TWIP = 635
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
# Styles this tool is allowed to emit; all must exist in the template.
STYLES = {"Heading2", "Heading3", "Paragraf", "Gambar", "JudulGambar",
          "judulTabel", "IsiTabel", "Daftarpustaka"}
CAPTION_STYLE = {"figure": ("Gambar", "JudulGambar"), "table": ("Tabel", "judulTabel")}


class SpecError(SystemExit):
    """A problem in the spec or the document that the user must fix."""


# --------------------------------------------------------------------------
# Small XML helpers
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    """Collapse whitespace (NBSP included) so anchors match what a reader sees."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def _style(node) -> str:
    if node.tag != W + "p":
        return node.tag.split("}")[-1]
    s = node.find(f"{W}pPr/{W}pStyle")
    return s.get(W + "val") if s is not None else ""


def _para(style: str, runs=()) -> etree._Element:
    p = wel("p")
    ppr = wel("pPr")
    ppr.append(wel("pStyle", val=style))
    p.append(ppr)
    for run in runs:
        p.append(run)
    return p


def _italic_rpr() -> etree._Element:
    rpr = wel("rPr")
    rpr.append(wel("i"))
    rpr.append(wel("iCs"))
    return rpr


def _bold_rpr() -> etree._Element:
    rpr = wel("rPr")
    rpr.append(wel("b"))
    rpr.append(wel("bCs"))
    return rpr


def _noproof_rpr() -> etree._Element:
    rpr = wel("rPr")
    rpr.append(wel("noProof"))
    return rpr


def _ref_field(label: str, cached: str = "?") -> list:
    """A complex REF field, the form every cross-reference in V14 uses."""
    out = []
    begin = wel("r")
    begin.append(wel("fldChar", fldCharType="begin"))
    out.append(begin)
    instr_run = wel("r")
    instr = etree.SubElement(instr_run, W + "instrText")
    instr.set(XML_SPACE, "preserve")
    instr.text = f" REF {label} \\h "
    out.append(instr_run)
    sep = wel("r")
    sep.append(wel("fldChar", fldCharType="separate"))
    out.append(sep)
    out.append(make_run(cached, _noproof_rpr()))
    end = wel("r")
    end.append(_noproof_rpr())
    end.append(wel("fldChar", fldCharType="end"))
    out.append(end)
    return out


def _seq_field(name: str, cached: str) -> list:
    out = []
    begin = wel("r")
    begin.append(wel("fldChar", fldCharType="begin"))
    out.append(begin)
    instr_run = wel("r")
    instr = etree.SubElement(instr_run, W + "instrText")
    instr.set(XML_SPACE, "preserve")
    instr.text = f" SEQ {name} \\* ARABIC \\s 1 "
    out.append(instr_run)
    sep = wel("r")
    sep.append(wel("fldChar", fldCharType="separate"))
    out.append(sep)
    out.append(make_run(cached, _noproof_rpr()))
    end = wel("r")
    end.append(_noproof_rpr())
    end.append(wel("fldChar", fldCharType="end"))
    out.append(end)
    return out


# --------------------------------------------------------------------------
# Inline markup -> runs
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*]+?\*|\[ref:[A-Za-z0-9_]+\])", re.S)


def parse_markup(text: str) -> list:
    """`**bold**`, `*italic*`, and `[ref:label]` into a run list.

    A reference is preceded by a non-breaking space when the text before it
    ends in "Gambar", "Tabel", or "Persamaan": Word's own cross-reference
    insertion does the same, and it keeps the number from wrapping away from
    its noun.
    """
    text = _norm(text).replace(r"\*", "\x00")
    runs: list = []
    pos = 0
    plain_tail = ""

    def emit_plain(chunk: str) -> None:
        nonlocal plain_tail
        if not chunk:
            return
        runs.append(make_run(chunk.replace("\x00", "*")))
        plain_tail = chunk

    for match in _TOKEN.finditer(text):
        emit_plain(text[pos:match.start()])
        token = match.group(0)
        if token.startswith("[ref:"):
            label = token[5:-1]
            if plain_tail.endswith(("Gambar ", "Tabel ", "Persamaan ")) and runs:
                last = runs[-1].find(W + "t")
                last.text = last.text[:-1] + " "
            runs.extend(_ref_field(label))
            plain_tail = ""
        elif token.startswith("**"):
            runs.append(make_run(token[2:-2].replace("\x00", "*"), _bold_rpr()))
            plain_tail = ""
        else:
            runs.append(make_run(token[1:-1].replace("\x00", "*"), _italic_rpr()))
            plain_tail = ""
        pos = match.end()
    emit_plain(text[pos:])
    return runs


def markup_refs(text: str) -> list[str]:
    return re.findall(r"\[ref:([A-Za-z0-9_]+)\]", text)


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------
class Document:
    def __init__(self, work: Path):
        self.work = work
        self.doc_path = work / "word" / "document.xml"
        self.rels_path = work / "word" / "_rels" / "document.xml.rels"
        self.ct_path = work / "[Content_Types].xml"
        self.tree = etree.parse(str(self.doc_path))
        self.body = self.tree.getroot().find(W + "body")
        self.rels = etree.parse(str(self.rels_path))
        self.content_types = etree.parse(str(self.ct_path))
        self.styles = {
            s.get(W + "styleId")
            for s in etree.parse(str(work / "word" / "styles.xml")).iter(W + "style")
        }
        self._scan_ids()
        self._media_by_hash: dict[str, str] = {}

    # -- ids --------------------------------------------------------------
    def _scan_ids(self) -> None:
        """Bookmark and shape ids are document-wide: headers and footnotes too."""
        bookmark_ids, doc_pr_ids, rel_ids = [0], [1], [0]
        for part in self.work.rglob("*.xml"):
            try:
                root = etree.parse(str(part)).getroot()
            except etree.XMLSyntaxError:
                continue
            bookmark_ids += [int(b.get(W + "id")) for b in root.iter(W + "bookmarkStart")
                             if (b.get(W + "id") or "").isdigit()]
            doc_pr_ids += [int(d.get("id")) for d in root.iter(WP + "docPr")
                           if (d.get("id") or "").isdigit()]
        for rel in self.rels.getroot():
            match = re.fullmatch(r"rId(\d+)", rel.get("Id") or "")
            if match:
                rel_ids.append(int(match.group(1)))
        self.next_bookmark = max(bookmark_ids) + 1
        self.next_doc_pr = max(d for d in doc_pr_ids if d < 10**6) + 1
        self.next_rel = max(rel_ids) + 1

    def bookmark_pair(self, name: str):
        ident = self.next_bookmark
        self.next_bookmark += 1
        start = wel("bookmarkStart", id=str(ident), name=name)
        end = wel("bookmarkEnd", id=str(ident))
        return start, end

    # -- geometry ---------------------------------------------------------
    def content_width_twips(self, node) -> int:
        """Text width of the section that governs `node`, in twips."""
        sect = None
        for candidate in self.body.iter(W + "sectPr"):
            if candidate.getparent() is self.body:
                sect = sect or candidate
        following = node.itersiblings()
        for sibling in following:
            found = sibling.find(f"{W}pPr/{W}sectPr")
            if found is not None:
                sect = found
                break
        if sect is None:
            return 7933
        pg = sect.find(W + "pgSz")
        mar = sect.find(W + "pgMar")
        if pg is None or mar is None:
            return 7933
        return (int(pg.get(W + "w")) - int(mar.get(W + "left"))
                - int(mar.get(W + "right")))

    # -- media ------------------------------------------------------------
    def add_image(self, path: Path) -> str:
        """Copy an image in (deduplicated) and return its relationship id."""
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in self._media_by_hash:
            return self._media_by_hash[digest]
        media = self.work / "word" / "media"
        media.mkdir(parents=True, exist_ok=True)
        target = media / f"insert_{digest[:12]}{path.suffix.lower()}"
        target.write_bytes(data)
        rel_id = f"rId{self.next_rel}"
        self.next_rel += 1
        rel = etree.SubElement(self.rels.getroot(), f"{{{PRNS}}}Relationship")
        rel.set("Id", rel_id)
        rel.set("Type", IMAGE_REL)
        rel.set("Target", f"media/{target.name}")
        self._register_extension(path.suffix.lower().lstrip("."))
        self._media_by_hash[digest] = rel_id
        return rel_id

    def _register_extension(self, ext: str) -> None:
        root = self.content_types.getroot()
        known = {d.get("Extension") for d in root.iter(f"{{{CTNS}}}Default")}
        if ext in known:
            return
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "emf": "image/x-emf"}.get(ext)
        if mime is None:
            raise SpecError(f"unknown image type: .{ext}")
        default = etree.SubElement(root, f"{{{CTNS}}}Default")
        default.set("Extension", ext)
        default.set("ContentType", mime)

    def save(self, out: Path, update_fields: bool) -> None:
        self.tree.write(str(self.doc_path), xml_declaration=True,
                        encoding="UTF-8", standalone=True)
        self.rels.write(str(self.rels_path), xml_declaration=True,
                        encoding="UTF-8", standalone=True)
        self.content_types.write(str(self.ct_path), xml_declaration=True,
                                 encoding="UTF-8", standalone=True)
        if update_fields:
            settings_path = self.work / "word" / "settings.xml"
            settings = etree.parse(str(settings_path))
            root = settings.getroot()
            if root.find(W + "updateFields") is None:
                root.insert(0, wel("updateFields", val="true"))
            settings.write(str(settings_path), xml_declaration=True,
                           encoding="UTF-8", standalone=True)
        pack(str(self.work), str(out))


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise SpecError(f"{path.name}: only PNG images are supported")
    return struct.unpack(">II", data[16:24])


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def build_picture(doc: Document, image: Path, rel_id: str, width_twips: int,
                  fraction: float) -> etree._Element:
    px_w, px_h = png_size(image)
    cx = int(round(width_twips * fraction)) * EMU_PER_TWIP
    cy = int(round(cx * px_h / px_w))
    doc_pr_id = doc.next_doc_pr
    doc.next_doc_pr += 2
    xml = f"""<w:p xmlns:w="{WNS}" xmlns:r="{RNS}" xmlns:a="{ANS}"
      xmlns:pic="{PICNS}" xmlns:wp="{WPNS}">
      <w:pPr><w:pStyle w:val="Gambar"/></w:pPr>
      <w:r>
        <w:rPr><w:noProof/></w:rPr>
        <w:drawing>
          <wp:inline distT="0" distB="0" distL="0" distR="0">
            <wp:extent cx="{cx}" cy="{cy}"/>
            <wp:effectExtent l="0" t="0" r="0" b="0"/>
            <wp:docPr id="{doc_pr_id}" name="Picture"/>
            <wp:cNvGraphicFramePr/>
            <a:graphic>
              <a:graphicData uri="{PICNS}">
                <pic:pic>
                  <pic:nvPicPr>
                    <pic:cNvPr id="{doc_pr_id + 1}" name="Picture"
                               descr="{image.name}"/>
                    <pic:cNvPicPr>
                      <a:picLocks noChangeAspect="1" noChangeArrowheads="1"/>
                    </pic:cNvPicPr>
                  </pic:nvPicPr>
                  <pic:blipFill>
                    <a:blip r:embed="{rel_id}"/>
                    <a:stretch><a:fillRect/></a:stretch>
                  </pic:blipFill>
                  <pic:spPr bwMode="auto">
                    <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                    <a:noFill/>
                    <a:ln w="9525"><a:noFill/><a:headEnd/><a:tailEnd/></a:ln>
                  </pic:spPr>
                </pic:pic>
              </a:graphicData>
            </a:graphic>
          </wp:inline>
        </w:drawing>
      </w:r>
    </w:p>"""
    return etree.fromstring(xml)


def build_caption(doc: Document, kind: str, label: str, caption: str,
                  chapter: str) -> etree._Element:
    """`Gambar <chapter>.<SEQ>` bookmarked, then the caption text."""
    noun, style = CAPTION_STYLE[kind]
    seq_name = "Gambar" if kind == "figure" else "Tabel"
    para = _para(style)
    para.append(make_run(f"{noun} "))
    start, end = doc.bookmark_pair(sanitize_bookmark(label))
    para.append(start)
    para.append(make_run(f"{chapter}."))
    for run in _seq_field(seq_name, "1"):
        para.append(run)
    para.append(end)
    body = parse_markup(caption)
    if body:
        first = body[0].find(W + "t")
        if first is not None and first.text:
            first.text = " " + first.text
        else:
            para.append(make_run(" "))
    for run in body:
        para.append(run)
    return para


def build_table(spec: dict, width_twips: int) -> etree._Element:
    columns, rows = spec["columns"], spec["rows"]
    weights = spec.get("widths") or [1] * len(columns)
    if len(weights) != len(columns):
        raise SpecError(f"table {spec.get('label')}: widths and columns disagree")
    total = float(sum(weights))
    grid = [int(round(width_twips * w / total)) for w in weights]
    grid[-1] += width_twips - sum(grid)          # absorb the rounding

    tbl = wel("tbl")
    tblpr = wel("tblPr")
    tblpr.append(wel("tblW", w=str(width_twips), type="dxa"))
    tblpr.append(wel("tblLayout", type="fixed"))
    _apply_itb_tbl_borders(tblpr)
    look = wel("tblLook", val="04A0")
    look.set(W + "firstRow", "1")
    look.set(W + "lastRow", "0")
    look.set(W + "firstColumn", "0")
    look.set(W + "lastColumn", "0")
    look.set(W + "noHBand", "0")
    look.set(W + "noVBand", "1")
    tblpr.append(look)
    tbl.append(tblpr)
    tbl_grid = wel("tblGrid")
    for width in grid:
        tbl_grid.append(wel("gridCol", w=str(width)))
    tbl.append(tbl_grid)

    def add_row(cells: list[str], header: bool) -> None:
        tr = wel("tr")
        if header and spec.get("header_repeat", True):
            trpr = wel("trPr")
            trpr.append(wel("tblHeader"))
            tr.append(trpr)
        for text, width in zip(cells, grid):
            tc = wel("tc")
            tcpr = wel("tcPr")
            tcpr.append(wel("tcW", w=str(width), type="dxa"))
            tc.append(tcpr)
            para = _para("IsiTabel")
            for run in parse_markup(str(text)):
                if header:
                    rpr = run.find(W + "rPr")
                    if rpr is None:
                        rpr = wel("rPr")
                        run.insert(0, rpr)
                    rpr.append(wel("b"))
                    rpr.append(wel("bCs"))
                para.append(run)
            tc.append(para)
            tr.append(tc)
        tbl.append(tr)

    add_row(list(columns), header=True)
    for row in rows:
        if len(row) != len(columns):
            raise SpecError(f"table {spec.get('label')}: row has {len(row)} cells, "
                            f"expected {len(columns)}")
        add_row([str(c) for c in row], header=False)
    return tbl


# --------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------
class Anchors:
    """Resolves spec anchors against the body, ignoring front matter.

    The table of contents repeats every heading and caption verbatim, so the
    search starts at the first chapter heading and runs to the end of the
    document, which keeps the bibliography and the appendices reachable while
    leaving the front matter out.
    """

    def __init__(self, doc: Document):
        self.body = doc.body
        self.kids = list(self.body)
        first = 0
        for i, node in enumerate(self.kids):
            if _style(node) == "Heading1":
                first = i
                break
        self.lo, self.hi = first, len(self.kids)

    def _candidates(self):
        return [(i, self.kids[i]) for i in range(self.lo, self.hi)
                if self.kids[i].tag == W + "p"]

    def find_heading(self, text: str) -> int:
        want = _norm(text)
        hits = [i for i, p in self._candidates()
                if _style(p) in ("Heading2", "Heading3") and _norm(para_text(p)) == want]
        return self._one(hits, f"heading {text!r}",
                         [_norm(para_text(p)) for i, p in self._candidates()
                          if _style(p) in ("Heading2", "Heading3")])

    def find_prefix(self, prefix: str, style: str | None) -> int:
        want = _norm(prefix)
        hits = [i for i, p in self._candidates()
                if (style is None or _style(p) == style)
                and _norm(para_text(p)).startswith(want)]
        return self._one(hits, f"text_prefix {prefix!r}",
                         [_norm(para_text(p))[:80] for i, p in self._candidates()])

    def find_label(self, label: str) -> int:
        name = sanitize_bookmark(label)
        hits = [i for i, p in self._candidates()
                if any(b.get(W + "name") in (name, label)
                       for b in p.iter(W + "bookmarkStart"))]
        return self._one(hits, f"label {label!r}", [])

    def find_caption_prefix(self, prefix: str) -> int:
        want = _norm(prefix)
        hits = [i for i, p in self._candidates()
                if _style(p) in ("JudulGambar", "judulTabel")
                and _norm(para_text(p)).startswith(want)]
        return self._one(hits, f"caption_prefix {prefix!r}",
                         [_norm(para_text(p))[:80] for i, p in self._candidates()
                          if _style(p) in ("JudulGambar", "judulTabel")])

    @staticmethod
    def _one(hits: list[int], what: str, pool: list[str]) -> int:
        if len(hits) == 1:
            return hits[0]
        if not hits:
            close = difflib.get_close_matches(what, pool, n=3, cutoff=0.4)
            hint = ("\n  did you mean:\n    " + "\n    ".join(close)) if close else ""
            raise SpecError(f"anchor not found: {what}{hint}")
        raise SpecError(f"anchor is ambiguous ({len(hits)} matches): {what}")

    def chapter_of(self, index: int) -> str:
        """The Roman numeral of the chapter that contains body child `index`."""
        seen = 0
        for i in range(0, index + 1):
            if _style(self.kids[i]) == "Heading1":
                seen += 1
        return ROMAN[max(seen, 1) - 1]

    def end_of_section(self, index: int) -> int:
        """Index of the last node of the subbab whose heading is at `index`."""
        level = _style(self.kids[index])
        stop = {"Heading2": ("Heading1", "Heading2"),
                "Heading3": ("Heading1", "Heading2", "Heading3")}[level]
        for i in range(index + 1, self.hi):
            if _style(self.kids[i]) in stop:
                end = i - 1
                while end > index and self.kids[end].tag == W + "p" \
                        and not para_text(self.kids[end]).strip():
                    end -= 1
                return end
        return self.hi - 1


# --------------------------------------------------------------------------
# Renumbering
# --------------------------------------------------------------------------
def _field_runs(para) -> list[tuple[str, list]]:
    """(instruction, cached runs) for every field in a paragraph.

    Both encodings appear in the manuscript: pandoc wrote complex fields, and
    the captions added by hand in Word are `fldSimple`.
    """
    out: list[tuple[str, list]] = []
    for simple in para.findall(W + "fldSimple"):
        out.append((simple.get(W + "instr") or "", list(simple.findall(W + "r"))))
    depth, instr, cached, collecting = 0, "", [], False
    for run in para.findall(W + "r"):
        char = run.find(W + "fldChar")
        if char is not None:
            kind = char.get(W + "fldCharType")
            if kind == "begin":
                depth += 1
                if depth == 1:
                    instr, cached, collecting = "", [], False
                continue
            if kind == "separate" and depth == 1:
                collecting = True
                continue
            if kind == "end":
                if depth == 1:
                    out.append((instr, cached))
                depth = max(depth - 1, 0)
                collecting = False
                continue
        if depth != 1:
            continue
        if collecting:
            cached.append(run)
        else:
            text = run.find(W + "instrText")
            if text is not None and text.text:
                instr += text.text
    return out


def _set_cached(runs: list, value: str) -> None:
    if not runs:
        return
    keep = runs[0]
    for extra in runs[1:]:
        parent = extra.getparent()
        if parent is not None:
            parent.remove(extra)
    for child in list(keep):
        if child.tag != W + "rPr":
            keep.remove(child)
    text = etree.SubElement(keep, W + "t")
    text.set(XML_SPACE, "preserve")
    text.text = value


# Not every number in the manuscript is a field. A few captions were typed by
# hand in Word, and 41 in-text mentions spell the number out instead of using a
# REF field. Renumbering has to reach those too, or the document ends up with
# two figures called "Gambar II.11".
CAPTION_NUMBER = re.compile(r"^(\s*(?:Gambar|Tabel)[\s\u00a0]+)([IVX]+)\.(\d+)")
INTEXT_NUMBER = re.compile(r"(Gambar|Tabel|Persamaan)([\s\u00a0]+)([IVX]+)\.(\d+)")


def _plain_runs(para) -> list:
    """Runs holding ordinary text: field instructions and results excluded."""
    out, depth = [], 0
    for child in para:
        tag = child.tag.split("}")[-1]
        if tag == "fldSimple":
            continue
        if tag != "r":
            continue
        char = child.find(W + "fldChar")
        if char is not None:
            kind = char.get(W + "fldCharType")
            if kind == "begin":
                depth += 1
            elif kind == "end":
                depth = max(depth - 1, 0)
            continue
        if depth or child.find(W + "instrText") is not None:
            continue
        if child.find(W + "t") is not None:
            out.append(child)
    return out


def _plain_text(para) -> tuple[str, list]:
    runs = _plain_runs(para)
    joined, spans, pos = "", [], 0
    for run in runs:
        text = "".join(t.text or "" for t in run.findall(W + "t"))
        spans.append((run, pos, pos + len(text)))
        joined += text
        pos += len(text)
    return joined, spans


def _rewrite_span(spans: list, start: int, end: int, value: str) -> bool:
    """Replace joined[start:end] with `value` across the runs it covers."""
    touched = [(run, s, e) for run, s, e in spans if e > start and s < end]
    if not touched:
        return False
    for i, (run, s, e) in enumerate(touched):
        texts = run.findall(W + "t")
        if not texts:
            continue
        current = "".join(t.text or "" for t in texts)
        lo, hi = max(start - s, 0), min(end - s, len(current))
        new = current[:lo] + (value if i == 0 else "") + current[hi:]
        for extra in texts[1:]:
            run.remove(extra)
        texts[0].set(XML_SPACE, "preserve")
        texts[0].text = new
    return True


def _retarget_literal_mentions(kids: list, remap: dict) -> list[str]:
    """Move hand-typed "Gambar II.4" mentions onto the new numbering.

    Only text outside a field is touched, and only a number the renumber pass
    actually moved; a mention whose figure did not move is left alone. A
    caption's own number was already handled, so the match at position 0 of a
    caption is skipped here. Every mention is collected before anything is
    written, and the edits are applied right to left: rewriting in place would
    otherwise let a number that has just moved match the map a second time.
    """
    done: list[str] = []
    for node in kids:
        if node.tag != W + "p":
            continue
        is_caption = _style(node) in ("JudulGambar", "judulTabel")
        joined, spans = _plain_text(node)
        edits = []
        for match in INTEXT_NUMBER.finditer(joined):
            if is_caption and match.start() == 0:
                continue
            kind, _, chapter, number = match.groups()
            new_number = remap.get((kind, chapter, number))
            if new_number is None or new_number == number:
                continue
            edits.append((match.start(4), match.end(4), new_number,
                          f"{kind} {chapter}.{number} -> {kind} {chapter}.{new_number}"))
        for lo, hi, value, label in reversed(edits):
            if _rewrite_span(spans, lo, hi, value):
                done.append(label)
    return done


def renumber(doc: Document, verbose: bool = True) -> dict[str, str]:
    """Recompute every SEQ cache, then every REF cache that points at one."""
    kids = list(doc.body)
    counters: dict[tuple[str, str], int] = {}
    numbers: dict[str, str] = {}
    remap: dict[tuple[str, str, str], str] = {}
    changes: list[str] = []
    chapter, in_body = "", False

    for node in kids:
        style = _style(node)
        if style == "Heading1":
            in_body = True
            index = sum(1 for k in kids[:kids.index(node) + 1]
                        if _style(k) == "Heading1")
            chapter = ROMAN[index - 1]
        elif style == "Lampiran":
            match = re.match(r"\s*Lampiran\s+([A-G])", para_text(node))
            if match:
                chapter = match.group(1)
        if node.tag != W + "p" or not in_body:
            continue
        fields = _field_runs(node)
        if not fields:
            # A caption whose number was typed by hand still occupies a slot in
            # the sequence, and its literal number has to move with the rest.
            if style in ("JudulGambar", "judulTabel"):
                joined, spans = _plain_text(node)
                match = CAPTION_NUMBER.match(joined)
                if match:
                    name = "Gambar" if style == "JudulGambar" else "Tabel"
                    key = (chapter, name)
                    counters[key] = counters.get(key, 0) + 1
                    new_number = str(counters[key])
                    old_number = match.group(3)
                    if old_number != new_number:
                        changes.append(f"  {chapter}.{old_number} -> "
                                       f"{chapter}.{new_number}  ({name}, literal)")
                    _rewrite_span(spans, match.start(3), match.end(3), new_number)
                    remap[(name, chapter, old_number)] = new_number
            continue
        # The prefix run ("II.") sits immediately before the field, inside the
        # bookmark that a REF points at.
        for instr, cached in fields:
            match = re.search(r"\bSEQ\s+(\w+)", instr)
            if not match:
                continue
            name = match.group(1)
            key = (chapter, name)
            counters[key] = counters.get(key, 0) + 1
            new = str(counters[key])
            old = "".join(t.text or "" for run in cached for t in run.iter(W + "t"))
            if old != new:
                changes.append(f"  {chapter}.{old or '?'} -> {chapter}.{new}"
                               f"  ({name})")
            if old:
                remap[(name, chapter, old)] = new
            _set_cached(cached, new)
            for bookmark in node.iter(W + "bookmarkStart"):
                bookmark_name = bookmark.get(W + "name") or ""
                if bookmark_name.startswith("_Toc"):
                    continue
                numbers[bookmark_name] = f"{chapter}.{new}"

    unresolved: list[str] = []
    empty_caches: list[str] = []
    for node in kids:
        if node.tag != W + "p":
            continue
        for instr, cached in _field_runs(node):
            match = re.search(r"\bREF\s+([^\s\\]+)", instr)
            if not match:
                continue
            target = match.group(1)
            shown = "".join(t.text or "" for run in cached for t in run.iter(W + "t"))
            if target in numbers:
                if not shown.strip():
                    # V14 carries one REF whose cached result is empty, so Word
                    # renders nothing there. Filling it would put a number into
                    # a sentence that never showed one; leave it for the author.
                    empty_caches.append(target)
                    continue
                _set_cached(cached, numbers[target])
            elif not target.startswith("_Toc"):
                unresolved.append(target)
    if unresolved:
        raise SpecError("REF targets that no caption defines: "
                        + ", ".join(sorted(set(unresolved))))

    if verbose and empty_caches:
        print(f"left {len(empty_caches)} cross-reference(s) with an empty cached "
              f"result untouched: {', '.join(sorted(set(empty_caches)))}")

    retyped = _retarget_literal_mentions(kids, remap)
    if verbose and retyped:
        print(f"rewrote {len(retyped)} hand-typed cross-reference(s):")
        for line in retyped[:20]:
            print(f"  {line}")
        if len(retyped) > 20:
            print(f"  ... and {len(retyped) - 20} more")
    if verbose and changes:
        print(f"renumbered {len(changes)} caption(s):")
        for line in changes[:40]:
            print(line)
        if len(changes) > 40:
            print(f"  ... and {len(changes) - 40} more")
    return numbers


# --------------------------------------------------------------------------
# Applying the spec
# --------------------------------------------------------------------------
def _blank(count: int = 1) -> list:
    return [_para("Paragraf") for _ in range(count)]


def _is_blank(node) -> bool:
    return (node is not None and node.tag == W + "p"
            and not para_text(node).strip()
            and node.find(f".//{W}drawing") is None)


def apply_edit(doc: Document, anchors: Anchors, edit: dict, base: Path,
               defined: dict, referenced: set, dry_run: bool) -> None:
    if "bib_entry" in edit:
        if not dry_run:
            insert_bib(doc, edit["bib_entry"])
        else:
            print(f"  bib_entry: {_norm(edit['bib_entry'])[:70]}...")
        return

    if "rewrite_para" in edit:
        spec = edit["rewrite_para"]
        index = anchors.find_prefix(spec["text_prefix"], spec.get("style", "Paragraf"))
        print(f"  rewrite [{index}] "
              f"{_norm(para_text(anchors.kids[index]))[:58]}")
        if not dry_run:
            rewrite_para(anchors, index, spec["text"], referenced)
        return

    if "replace_figure" in edit:
        spec = edit["replace_figure"]
        if "label" in spec and "caption_prefix" not in spec:
            index = anchors.find_label(spec["label"])
        else:
            index = anchors.find_caption_prefix(spec["caption_prefix"])
        caption_para = anchors.kids[index]
        print(f"  replace_figure -> [{index}] {_norm(para_text(caption_para))[:64]}")
        if not dry_run:
            replace_figure(doc, anchors, index, spec, base, defined)
        return

    anchor = edit["anchor"]
    position = anchor.get("position", "after")
    if "heading" in anchor:
        index = anchors.find_heading(anchor["heading"])
        if position == "after_section":
            index = anchors.end_of_section(index)
            position = "after"
        elif position == "after":
            # Keep the blank separator that follows every heading.
            if _is_blank(anchors.kids[index + 1]):
                index += 1
    elif "text_prefix" in anchor:
        index = anchors.find_prefix(anchor["text_prefix"], anchor.get("style", "Paragraf"))
    else:
        index = anchors.find_label(anchor["label"])

    target = anchors.kids[index]
    chapter = anchors.chapter_of(index)
    width = doc.content_width_twips(target)
    print(f"  {position:5s} [{index}] Bab {chapter} | "
          f"{_style(target)}: {_norm(para_text(target))[:56]}")

    nodes: list = []
    for block in edit["blocks"]:
        kind, spec = next(iter(block.items()))
        if nodes:
            nodes.extend(_blank())
        if kind == "heading3":
            nodes.append(_para("Heading3", parse_markup(spec)))
        elif kind == "para":
            nodes.append(_para("Paragraf", parse_markup(spec)))
            referenced.update(markup_refs(spec))
        elif kind == "figure":
            image = (base / spec["image"]).resolve()
            if not image.exists():
                raise SpecError(f"image not found: {image}")
            rel_id = doc.add_image(image) if not dry_run else "rIdX"
            if not dry_run:
                nodes.append(build_picture(doc, image, rel_id, width,
                                           float(spec.get("width_fraction", 0.9))))
                nodes.append(build_caption(doc, "figure", spec["label"],
                                           spec["caption"], chapter))
            defined[spec["label"]] = f"Gambar (Bab {chapter})"
            referenced.update(markup_refs(spec["caption"]))
        elif kind == "table":
            if not dry_run:
                nodes.append(build_caption(doc, "table", spec["label"],
                                           spec["caption"], chapter))
                nodes.extend(_blank())
                nodes.append(build_table(spec, width))
            defined[spec["label"]] = f"Tabel (Bab {chapter})"
            referenced.update(markup_refs(spec["caption"]))
            for row in [spec["columns"], *spec["rows"]]:
                for cell in row:
                    referenced.update(markup_refs(str(cell)))
        else:
            raise SpecError(f"unknown block type: {kind}")
    if dry_run:
        return

    # One blank paragraph separates the inserted span from its neighbours.
    if position == "before":
        nodes.extend(_blank())
        for node in nodes:
            target.addprevious(node)
        previous = nodes[0].getprevious()
        if _is_blank(previous) and _is_blank(nodes[0]):
            pass
        elif not _is_blank(previous) and previous is not None:
            nodes[0].addprevious(_para("Paragraf"))
    else:
        nodes = _blank() + nodes
        anchor_node = target
        for node in nodes:
            anchor_node.addnext(node)
            anchor_node = node
        following = nodes[-1].getnext()
        if following is not None and not _is_blank(following):
            nodes[-1].addnext(_para("Paragraf"))
    anchors.kids = list(doc.body)


def rewrite_para(anchors: Anchors, index: int, text: str, referenced: set) -> None:
    """Replace the text of an existing paragraph, keeping its style.

    Used to reword prose in place; the paragraph's own style, and its position
    among its neighbours, stay exactly as they were.
    """
    para = anchors.kids[index]
    for child in list(para):
        if child.tag != W + "pPr":
            para.remove(child)
    for run in parse_markup(text):
        para.append(run)
    referenced.update(markup_refs(text))


def replace_figure(doc: Document, anchors: Anchors, index: int, spec: dict,
                   base: Path, defined: dict) -> None:
    """Point an existing figure at a new image and rewrite its caption text."""
    caption = anchors.kids[index]
    picture = caption.getprevious()
    while picture is not None and picture.find(f".//{W}drawing") is None:
        picture = picture.getprevious()
    if picture is None:
        raise SpecError(f"no picture paragraph before caption at index {index}")
    if spec.get("image"):
        image = (base / spec["image"]).resolve()
        if not image.exists():
            raise SpecError(f"image not found: {image}")
        rel_id = doc.add_image(image)
        blip = picture.find(f".//{A}blip")
        blip.set(R + "embed", rel_id)
        # A crop set in Word belongs to the picture being replaced: leaving it
        # in place would cut the same percentages out of the new image.
        for fill in picture.iter(f"{{{PICNS}}}blipFill"):
            for crop in fill.findall(A + "srcRect"):
                fill.remove(crop)
        # Cached size hints refer to the old bitmap; Word rewrites them itself.
        for ext_list in blip.findall(A + "extLst"):
            blip.remove(ext_list)
        px_w, px_h = png_size(image)
        width = doc.content_width_twips(picture)
        cx = int(round(width * float(spec.get("width_fraction", 0.9)))) * EMU_PER_TWIP
        cy = int(round(cx * px_h / px_w))
        for tag, xname, yname in ((WP + "extent", "cx", "cy"), (A + "ext", "cx", "cy")):
            for node in picture.iter(tag):
                node.set(xname, str(cx))
                node.set(yname, str(cy))
        for node in picture.iter(f"{{{PICNS}}}cNvPr"):
            node.set("descr", image.name)

    # Rewrite the caption title: keep "Gambar ", the bookmark, and the field;
    # replace everything after the bookmarkEnd.
    end = caption.find(W + "bookmarkEnd")
    if end is None:
        raise SpecError("caption has no bookmarkEnd; cannot rewrite safely")
    label = spec.get("label")
    if label:
        # A hand-inserted caption may carry only a _Toc bookmark; give it a
        # stable name so the new prose can reference it.
        names = {b.get(W + "name") for b in caption.iter(W + "bookmarkStart")}
        if sanitize_bookmark(label) not in names:
            start, new_end = doc.bookmark_pair(sanitize_bookmark(label))
            first_field = caption.find(W + "fldSimple")
            if first_field is None:
                for run in caption.findall(W + "r"):
                    if run.find(W + "fldChar") is not None:
                        first_field = run
                        break
            prefix_run = first_field.getprevious() if first_field is not None else None
            (prefix_run if prefix_run is not None else caption[0]).addprevious(start)
            end.addnext(new_end)
            end = new_end
            # Only a bookmark this tool creates has to earn a reference in the
            # text; a caption that already carried one is V14's own float.
            defined[label] = "Gambar (label baru)"
    seen_end = False
    for child in list(caption):
        if child is end:
            seen_end = True
            continue
        if seen_end:
            caption.remove(child)
    runs = parse_markup(spec["caption"])
    if runs:
        first = runs[0].find(W + "t")
        if first is not None and first.text:
            first.text = " " + first.text
    for run in runs:
        caption.append(run)


def insert_bib(doc: Document, text: str) -> None:
    entries = [p for p in doc.body if _style(p) == "Daftarpustaka"]
    if not entries:
        raise SpecError("no Daftarpustaka paragraphs found")
    new = _para("Daftarpustaka", parse_markup(text))
    key = _norm(text).casefold()
    prefix = key[:30]
    for entry in entries:
        if _norm(para_text(entry)).casefold().startswith(prefix):
            raise SpecError(f"bibliography already has an entry starting {prefix!r}")
    for entry in entries:
        if _norm(para_text(entry)).casefold() > key:
            entry.addprevious(new)
            return
    entries[-1].addnext(new)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate(doc: Document, defined: dict, referenced: set) -> None:
    used = {_style(n) for n in doc.body.iter(W + "p")} - {""}
    missing = sorted((used & STYLES) - doc.styles)
    if missing:
        raise SpecError(f"styles used but absent from styles.xml: {missing}")

    names: dict[str, int] = {}
    ids: dict[str, int] = {}
    for bookmark in doc.body.iter(W + "bookmarkStart"):
        name = bookmark.get(W + "name")
        names[name] = names.get(name, 0) + 1
        ident = bookmark.get(W + "id")
        ids[ident] = ids.get(ident, 0) + 1
    duplicates = sorted(n for n, c in names.items() if c > 1 and not n.startswith("_"))
    if duplicates:
        raise SpecError(f"duplicate bookmark names: {duplicates}")
    duplicate_ids = sorted(i for i, c in ids.items() if c > 1)
    if duplicate_ids:
        raise SpecError(f"duplicate bookmark ids: {duplicate_ids}")
    for name in names:
        if len(name) > 40:
            raise SpecError(f"bookmark name longer than 40 chars: {name}")

    # RULES section 7: every float must be referenced in the text.
    orphans = sorted(label for label in defined
                     if sanitize_bookmark(label) not in
                     {sanitize_bookmark(r) for r in referenced})
    if orphans:
        raise SpecError("new floats never referenced in the text (RULES 7): "
                        + ", ".join(orphans))

    targets = {rel.get("Id") for rel in doc.rels.getroot()}
    for blip in doc.body.iter(f"{A}blip"):
        embed = blip.get(R + "embed")
        if embed not in targets:
            raise SpecError(f"picture points at a missing relationship: {embed}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def resolve_source(spec_path: Path, pattern: str) -> Path:
    matches = sorted((spec_path.parent / pattern).parent.glob(
        Path(pattern).name))
    if len(matches) != 1:
        raise SpecError(f"source pattern {pattern!r} matched {len(matches)} files")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve anchors and report, write nothing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output document")
    parser.add_argument("--update-fields-on-open", action="store_true",
                        help="ask Word to refresh every field when it opens the file")
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    base = spec_path.parent
    source = resolve_source(spec_path, spec["source"])
    out = (base / spec["output"]).resolve()
    if out == source.resolve():
        raise SpecError("the spec would overwrite its own source document")
    if out.exists() and not args.force and not args.dry_run:
        raise SpecError(f"output exists (use --force): {out.name}")

    print(f"source: {source.name}")
    work = Path(tempfile.mkdtemp(prefix="insert-docx-"))
    try:
        unpack(str(source), str(work))
        doc = Document(work)
        anchors = Anchors(doc)
        defined: dict[str, str] = {}
        referenced: set[str] = set()
        print(f"{len(spec['edits'])} edits:")
        for edit in spec["edits"]:
            apply_edit(doc, anchors, edit, base, defined, referenced, args.dry_run)
        if args.dry_run:
            print("dry run: nothing written")
            renumber(doc, verbose=True)
            return 0
        renumber(doc)
        validate(doc, defined, referenced)
        doc.save(out, args.update_fields_on_open)
        print(f"wrote {out}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
