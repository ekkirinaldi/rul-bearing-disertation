#!/usr/bin/env python3
"""Post-process a pandoc-produced chapter docx into ITB dissertation format.

Usage:
    python3 restyle.py RAW.docx OUT.docx \
        --aux ../writings/disertation/build/disertasi.aux \
        --figmap assets/figure-map.tsv --bab 6 --chapter-label bab:kesimpulan

Steps (all verified against the ITB template internals):
  1. Re-tag pandoc paragraph styles to template styles
     (FirstParagraph/BodyText -> Paragraf, ImageCaption -> JudulGambar,
      TableCaption -> judulTabel).
  2. Captions: replace the @@CAP:label@@ token with
     "Gambar|Tabel {STYLEREF 1 \\s}.{SEQ Gambar|Tabel \\* ARABIC \\s 1} "
     fields carrying pre-computed cached results, bookmarked with the
     sanitized label so REF fields can point at the number.
  3. @@REF:label@@ tokens -> REF fields with cached number from the .aux.
  4. @@EQNUM:label@@ paragraphs -> numbered display equation:
     [tab] oMath [tab] (STYLEREF.SEQ Persamaan), bookmarked.
  5. Image extents set to width_fraction x text width (from figure-map),
     preserving aspect ratio.
  6. numbering.xml: startOverride so the standalone chapter numbers as Bab N.
  7. sectPr pgNumType -> arabic, starting at the chapter's real page number.
  8. Assertions: no tokens left, every cached number equals the .aux value.
"""
import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_tex import parse_aux, parse_figmap  # noqa: E402
from unpack import unpack  # noqa: E402
from pack import pack  # noqa: E402

WNS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MNS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
ANS = "http://schemas.openxmlformats.org/drawingml/2006/main"
WPNS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
W = "{%s}" % WNS
M = "{%s}" % MNS
R = "{%s}" % RNS
A = "{%s}" % ANS
WP = "{%s}" % WPNS

STYLE_MAP = {
    "FirstParagraph": "Paragraf",
    "BodyText": "Paragraf",
    "ImageCaption": "JudulGambar",
    "TableCaption": "judulTabel",
    "Compact": "IsiTabel",
    "CaptionedFigure": "Gambar",
}

# Custom styles the ITB template lacks; added to styles.xml when used.
# IsiTabel: table-cell text (TNR 12pt, single-spaced per Pedoman).
# Gambar: figure-container paragraph (centered), same name V5 used.
EXTRA_STYLES = """
<w:style w:type="paragraph" w:customStyle="1" w:styleId="IsiTabel"
    xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:name w:val="Isi Tabel"/><w:qFormat/>
  <w:pPr><w:spacing w:before="20" w:after="20" w:line="240" w:lineRule="auto"/></w:pPr>
  <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
</w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Gambar"
    xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:name w:val="Gambar"/><w:qFormat/>
  <w:pPr><w:keepNext/><w:spacing w:before="120" w:after="120" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>
  <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
</w:style>
"""

CAPTION_KIND = {"fig": ("Gambar", "JudulGambar"), "tab": ("Tabel", "judulTabel")}


def wel(tag, **attrs):
    e = etree.Element(W + tag)
    for k, v in attrs.items():
        e.set(W + k, str(v))
    return e


def make_run(text, rpr=None):
    r = wel("r")
    if rpr is not None:
        r.append(rpr)
    t = etree.SubElement(r, W + "t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return r


def make_field(instr, cached):
    fld = wel("fldSimple", instr=instr)
    fld.append(make_run(cached))
    return fld


def sanitize_bookmark(label):
    name = re.sub(r"[^A-Za-z0-9_]", "_", label)
    if not name[0].isalpha():
        name = "b" + name
    return name[:40]


def para_text(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


class Restyler:
    def __init__(self, tree, labels, bab):
        self.doc = tree
        self.body = tree.find(W + "body")
        self.labels = labels
        self.bab = bab
        ids = [int(b.get(W + "id")) for b in tree.iter(W + "bookmarkStart")]
        self.next_bm = max(ids, default=100) + 1
        self.seq = {"Gambar": 0, "Tabel": 0, "Persamaan": 0}
        self.bookmarks = {}

    def bookmark_pair(self, name):
        i = self.next_bm
        self.next_bm += 1
        return (wel("bookmarkStart", id=i, name=name), wel("bookmarkEnd", id=i))

    # -- step 1 ---------------------------------------------------------
    def retag_styles(self):
        for ps in self.doc.iter(W + "pStyle"):
            val = ps.get(W + "val")
            if val in STYLE_MAP:
                ps.set(W + "val", STYLE_MAP[val])

    # -- step 2 ---------------------------------------------------------
    def number_for(self, label):
        if label not in self.labels:
            sys.exit(f"FATAL: label {label} not in aux")
        return self.labels[label]

    def transform_captions(self):
        for p in list(self.doc.iter(W + "p")):
            first_t = next(p.iter(W + "t"), None)
            if first_t is None or not (first_t.text or "").startswith("@@CAP:"):
                continue
            m = re.match(r"@@CAP:([^@]+)@@", first_t.text)
            label = m.group(1)
            kind, style = CAPTION_KIND[label.split(":")[0]]
            self.seq[kind] += 1
            aux_num = self.number_for(label)
            roman, seqno = aux_num.rsplit(".", 1)
            expect = f"{'IVXLCDM'[0:0]}{self.int_to_roman(self.bab)}.{self.seq[kind]}"
            assert aux_num == expect, f"caption number mismatch {label}: aux={aux_num} computed={expect}"
            first_t.text = first_t.text[m.end():]
            # ensure caption style
            ps = p.find(f"{W}pPr/{W}pStyle")
            ps.set(W + "val", style)
            run = first_t.getparent()
            bm_s, bm_e = self.bookmark_pair(sanitize_bookmark(label))
            self.bookmarks[label] = sanitize_bookmark(label)
            # literal Roman chapter prefix + SEQ field: LibreOffice mangles
            # STYLEREF, and the chapter a float lives in is structurally
            # stable; SEQ still renumbers within the chapter on F9 in Word.
            prefix = [
                make_run(f"{kind} "),
                bm_s,
                make_run(f"{roman}."),
                make_field(f" SEQ {kind} \\* ARABIC \\s 1 ", seqno),
                bm_e,
                make_run(" "),
            ]
            # addprevious inserts directly before `run`, so iterate in
            # document order to keep the sequence intact
            for el in prefix:
                run.addprevious(el)

    @staticmethod
    def int_to_roman(n):
        vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
                (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
                (5, "V"), (4, "IV"), (1, "I")]
        out = ""
        for v, s in vals:
            while n >= v:
                out += s
                n -= v
        return out

    # -- step 3 ---------------------------------------------------------
    def transform_refs(self):
        token = re.compile(r"@@REF:([^@]+)@@")
        for t in list(self.doc.iter(W + "t")):
            if "@@REF:" not in (t.text or ""):
                continue
            run = t.getparent()
            rpr = run.find(W + "rPr")
            parts = token.split(t.text)
            # parts = [text, label, text, label, ..., text]
            new_els = []
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    if part:
                        new_els.append(make_run(
                            part, rpr.__copy__() if rpr is not None else None))
                else:
                    cached = self.number_for(part)
                    new_els.append(make_field(
                        f" REF {sanitize_bookmark(part)} \\h ", cached))
            for el in reversed(new_els):
                run.addnext(el)
            run.getparent().remove(run)

    # -- step 4 ---------------------------------------------------------
    def transform_equations(self, content_twips):
        for p in list(self.doc.iter(W + "p")):
            m = re.fullmatch(r"@@EQNUM:([^@]+)@@", para_text(p).strip())
            if not m:
                continue
            label = m.group(1)
            prev = p.getprevious()
            omath_para = prev.find(f".//{M}oMathPara") if prev is not None else None
            if omath_para is None:
                sys.exit(f"FATAL: @@EQNUM:{label}@@ without preceding math paragraph")
            omath = omath_para.find(M + "oMath")
            # rebuild prev as: [tab] math [tab] (STYLEREF.SEQ)
            for child in list(prev):
                if child.tag != W + "pPr":
                    prev.remove(child)
            ppr = prev.find(W + "pPr")
            if ppr is None:
                ppr = wel("pPr")
                prev.insert(0, ppr)
            for old in ppr.findall(W + "tabs"):
                ppr.remove(old)
            tabs = wel("tabs")
            tabs.append(wel("tab", val="center", pos=content_twips // 2))
            tabs.append(wel("tab", val="right", pos=content_twips))
            ppr.append(tabs)
            tab_run = wel("r")
            tab_run.append(wel("tab"))
            prev.append(tab_run)
            prev.append(omath)
            tab_run2 = wel("r")
            tab_run2.append(wel("tab"))
            prev.append(tab_run2)
            self.seq["Persamaan"] += 1
            if label.startswith("eq:auto_"):
                roman = self.int_to_roman(self.bab)
                seqno = str(self.seq["Persamaan"])
            else:
                aux_num = self.number_for(label)
                roman, seqno = aux_num.rsplit(".", 1)
                assert int(seqno) == self.seq["Persamaan"], \
                    f"equation number mismatch {label}: aux={aux_num} computed seq={self.seq['Persamaan']}"
            bm_s, bm_e = self.bookmark_pair(sanitize_bookmark(label))
            prev.append(make_run("("))
            prev.append(bm_s)
            prev.append(make_run(f"{roman}."))
            prev.append(make_field(f" SEQ Persamaan \\* ARABIC \\s 1 ", seqno))
            prev.append(bm_e)
            prev.append(make_run(")"))
            p.getparent().remove(p)

    # -- step 5 ---------------------------------------------------------
    def resize_images(self, word_dir, figmap, content_emu):
        rels = etree.parse(str(word_dir / "_rels" / "document.xml.rels"))
        rid_to_target = {
            rel.get("Id"): rel.get("Target")
            for rel in rels.getroot()
            if rel.get("Type", "").endswith("/image")
        }
        # pandoc renames media parts, so match by content hash
        import hashlib

        def sha(p):
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()

        base = Path(__file__).resolve().parent.parent
        frac_by_hash = {sha(base / asset): frac for asset, frac in figmap.values()
                        if (base / asset).exists()}
        for drawing in self.doc.iter(W + "drawing"):
            blip = drawing.find(f".//{A}blip")
            if blip is None:
                continue
            target = rid_to_target.get(blip.get(R + "embed"), "")
            frac = frac_by_hash.get(sha(word_dir / target))
            if frac is None:
                sys.exit(f"FATAL: embedded image {target} not in figure map")
            extent = drawing.find(f".//{WP}extent")
            cx, cy = int(extent.get("cx")), int(extent.get("cy"))
            new_cx = int(content_emu * frac)
            new_cy = int(cy * new_cx / cx)
            extent.set("cx", str(new_cx))
            extent.set("cy", str(new_cy))
            aext = drawing.find(f".//{A}xfrm/{A}ext")
            if aext is not None:
                aext.set("cx", str(new_cx))
                aext.set("cy", str(new_cy))

    # -- step 5b --------------------------------------------------------
    def fix_tables(self, hints, content_twips):
        tables = list(self.doc.iter(W + "tbl"))
        if hints and len(hints) != len(tables):
            sys.exit(f"FATAL: {len(hints)} table hints but {len(tables)} tables")
        for idx, tbl in enumerate(tables):
            grid = tbl.find(W + "tblGrid")
            ncols = len(grid)
            weights = hints[idx] if hints else [1.0] * ncols
            if len(weights) != ncols:
                sys.exit(f"FATAL: table {idx}: hint has {len(weights)} cols, "
                         f"docx table has {ncols}")
            total = sum(weights)
            widths = [int(content_twips * w / total) for w in weights]
            widths[-1] = content_twips - sum(widths[:-1])
            for col, wd in zip(grid, widths):
                col.set(W + "w", str(wd))
            tblpr = tbl.find(W + "tblPr")
            for tag, attrs in (("tblW", {"w": content_twips, "type": "dxa"}),
                               ("tblLayout", {"type": "fixed"})):
                el = tblpr.find(W + tag)
                if el is None:
                    el = wel(tag)
                    tblpr.append(el)
                for k, v in attrs.items():
                    el.set(W + k, str(v))
            for tr in tbl.iter(W + "tr"):
                ci = 0
                for tc in tr.findall(W + "tc"):
                    tcpr = tc.find(W + "tcPr")
                    if tcpr is None:
                        tcpr = wel("tcPr")
                        tc.insert(0, tcpr)
                    span_el = tcpr.find(W + "gridSpan")
                    span = int(span_el.get(W + "val")) if span_el is not None else 1
                    tcw = tcpr.find(W + "tcW")
                    if tcw is None:
                        tcw = wel("tcW")
                        tcpr.append(tcw)
                    tcw.set(W + "w", str(sum(widths[ci:ci + span])))
                    tcw.set(W + "type", "dxa")
                    ci += span

    # -- step 5a2 -------------------------------------------------------
    def apply_spans(self):
        """@@SPANn@@ tokens (from \\multicolumn) -> gridSpan=n, drop the
        n-1 empty padding cells that follow in the same row."""
        for tc in list(self.doc.iter(W + "tc")):
            t = tc.find(f".//{W}t")
            if t is None or not (t.text or "").startswith("@@SPAN"):
                continue
            m = re.match(r"@@SPAN(\d+)@@", t.text)
            n = int(m.group(1))
            t.text = t.text[m.end():]
            tcpr = tc.find(W + "tcPr")
            if tcpr is None:
                tcpr = wel("tcPr")
                tc.insert(0, tcpr)
            span = tcpr.find(W + "gridSpan")
            if span is None:
                span = wel("gridSpan")
                tcpr.append(span)
            span.set(W + "val", str(n))
            tr = tc.getparent()
            cells = tr.findall(W + "tc")
            idx = cells.index(tc)
            extras = cells[idx + 1:idx + n]
            assert len(extras) == n - 1, \
                f"@@SPAN{n}@@ row has only {len(cells) - idx} cells"
            for extra in extras:
                assert not para_text(extra).strip(), \
                    f"padding cell after @@SPAN{n}@@ is not empty"
                tr.remove(extra)

    # -- step 5b2 -------------------------------------------------------
    def fix_layout_tables(self):
        """Tables holding images (from side-by-side minipage figures): keep
        the row on one page, bottom-align cells, and keep the block attached
        to the caption paragraph that follows."""
        for tbl in self.doc.iter(W + "tbl"):
            if tbl.find(f".//{W}drawing") is None:
                continue
            for tr in tbl.findall(W + "tr"):
                trpr = tr.find(W + "trPr")
                if trpr is None:
                    trpr = wel("trPr")
                    tr.insert(0, trpr)
                if trpr.find(W + "cantSplit") is None:
                    trpr.append(wel("cantSplit"))
            for tc in tbl.iter(W + "tc"):
                tcpr = tc.find(W + "tcPr")
                valign = tcpr.find(W + "vAlign")
                if valign is None:
                    valign = wel("vAlign")
                    tcpr.append(valign)
                valign.set(W + "val", "bottom")
                for p in tc.findall(W + "p"):
                    ppr = p.find(W + "pPr")
                    if ppr is None:
                        ppr = wel("pPr")
                        p.insert(0, ppr)
                    if ppr.find(W + "keepNext") is None:
                        kn = wel("keepNext")
                        ps = ppr.find(W + "pStyle")
                        ppr.insert(1 if ps is not None else 0, kn)
                    jc = ppr.find(W + "jc")
                    if jc is None:
                        jc = wel("jc")
                        ppr.append(jc)
                    jc.set(W + "val", "center")

    # -- step 5c --------------------------------------------------------
    def insert_separators(self):
        """One blank line between blocks, per Pedoman SIII.2 (same convention
        as the V5 draft: literal empty paragraphs)."""

        def kind(el):
            if el.tag == W + "tbl":
                return "tbl"
            if el.tag != W + "p":
                return None
            if not para_text(el).strip() and el.find(f".//{M}oMath") is None \
                    and el.find(f".//{W}drawing") is None:
                return None  # already an empty separator
            ps = el.find(f"{W}pPr/{W}pStyle")
            return ps.get(W + "val") if ps is not None else "Normal"

        NO_SEP_PAIRS = {("Gambar", "JudulGambar"), ("judulTabel", "tbl"),
                        ("tbl", "JudulGambar")}
        # body-level bookmarkStart/End (pandoc section anchors) are
        # transparent: pair only real content blocks across them
        blocks = [el for el in self.body
                  if el.tag not in (W + "bookmarkStart", W + "bookmarkEnd")]
        for cur, nxt in zip(blocks, blocks[1:]):
            k1, k2 = kind(cur), kind(nxt)
            if k1 is None or k2 is None:
                continue
            if nxt.tag == W + "sectPr" or (k1, k2) in NO_SEP_PAIRS:
                continue
            sep = wel("p")
            ppr = wel("pPr")
            st = wel("pStyle", val="Paragraf")
            ppr.append(st)
            sep.append(ppr)
            cur.addnext(sep)

    # -- steps 6+7 ------------------------------------------------------
    def fix_section(self, start_page):
        sect = self.body.find(W + "sectPr")
        pgnum = sect.find(W + "pgNumType")
        if pgnum is None:
            pgnum = wel("pgNumType")
            sect.append(pgnum)
        pgnum.set(W + "fmt", "decimal")
        pgnum.set(W + "start", str(start_page))

    def content_width_twips(self):
        sect = self.body.find(W + "sectPr")
        sz = sect.find(W + "pgSz")
        mar = sect.find(W + "pgMar")
        return (int(sz.get(W + "w")) - int(mar.get(W + "left"))
                - int(mar.get(W + "right")))

    # -- step 7b --------------------------------------------------------
    def clean_tbl_captions(self):
        """pandoc copies the caption text (with our token) into the table
        accessibility attribute w:tblCaption - strip the token there."""
        for cap in self.doc.iter(W + "tblCaption"):
            cap.set(W + "val",
                    re.sub(r"@@CAP:[^@]+@@", "", cap.get(W + "val", "")))

    # -- step 8 ---------------------------------------------------------
    def assert_clean(self):
        xml = etree.tostring(self.doc).decode()
        assert "@@" not in xml, \
            f"leftover tokens: {re.findall(r'.{{30}}@@.{{30}}', xml)[:3]}"


def add_extra_styles(styles_path):
    tree = etree.parse(str(styles_path))
    root = tree.getroot()
    existing = {s.get(W + "styleId") for s in root.iter(W + "style")}
    frag = etree.fromstring(f"<root>{EXTRA_STYLES}</root>")
    for style in frag:
        if style.get(W + "styleId") not in existing:
            root.append(style)
    tree.write(str(styles_path), xml_declaration=True, encoding="UTF-8",
               standalone=True)


def fix_numbering(numbering_path, bab):
    """startOverride so a standalone chapter numbers as Bab N, and flush-left
    subbab headings: the template's abstractNum 2 levels 1-2 carry a stray
    3818-twip indent + tab that centers them mid-page, contradicting the
    juknis (subbab: 12pt bold, left-aligned)."""
    tree = etree.parse(str(numbering_path))
    for an in tree.iter(W + "abstractNum"):
        if an.get(W + "abstractNumId") != "2":
            continue
        for lvl in an.iter(W + "lvl"):
            if lvl.get(W + "ilvl") not in ("1", "2"):
                continue
            ppr = lvl.find(W + "pPr")
            if ppr is not None:
                for tag in ("tabs", "ind"):
                    el = ppr.find(W + tag)
                    if el is not None:
                        ppr.remove(el)
                ind = wel("ind", left=0, firstLine=0)
                ppr.append(ind)
            if lvl.find(W + "suff") is None:
                suff = wel("suff", val="space")
                # schema order: suff goes right before lvlText
                lvl.find(W + "lvlText").addprevious(suff)
    for num in tree.iter(W + "num"):
        if num.get(W + "numId") == "2":
            for old in num.findall(W + "lvlOverride"):
                num.remove(old)
            ov = wel("lvlOverride", ilvl=0)
            ov.append(wel("startOverride", val=bab))
            num.append(ov)
            break
    else:
        sys.exit("FATAL: numId 2 not found in numbering.xml")
    tree.write(str(numbering_path), xml_declaration=True,
               encoding="UTF-8", standalone=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--aux", required=True)
    ap.add_argument("--figmap", required=True)
    ap.add_argument("--bab", type=int, required=True)
    ap.add_argument("--chapter-label", required=True)
    ap.add_argument("--tblhints", help="JSON column-weight hints from preprocess")
    args = ap.parse_args()

    labels = parse_aux(args.aux)
    figmap = parse_figmap(args.figmap)
    pages = {}  # chapter start pages
    pat = re.compile(r"\\newlabel\{([^}]+)\}\{\{.*?\}\{([0-9]+)\}")
    for line in Path(args.aux).read_text().splitlines():
        m = pat.match(line)
        if m and "@cref" not in m.group(1):
            pages[m.group(1)] = int(m.group(2))

    workdir = tempfile.mkdtemp(prefix="restyle_")
    unpack(args.raw, workdir)
    docpath = Path(workdir) / "word" / "document.xml"
    tree = etree.parse(str(docpath))

    import json
    hints = []
    if args.tblhints and Path(args.tblhints).exists():
        hints = json.loads(Path(args.tblhints).read_text())

    rs = Restyler(tree.getroot(), labels, args.bab)
    content_twips = rs.content_width_twips()
    rs.retag_styles()
    rs.transform_captions()
    rs.transform_refs()
    rs.transform_equations(content_twips)
    rs.apply_spans()
    rs.fix_tables(hints, content_twips)
    rs.fix_layout_tables()
    rs.insert_separators()
    rs.resize_images(Path(workdir) / "word", figmap, content_twips * 635)
    rs.fix_section(pages.get(args.chapter_label, 1))
    rs.clean_tbl_captions()
    rs.assert_clean()
    tree.write(str(docpath), xml_declaration=True, encoding="UTF-8",
               standalone=True)

    add_extra_styles(Path(workdir) / "word" / "styles.xml")
    fix_numbering(Path(workdir) / "word" / "numbering.xml", args.bab)
    pack(workdir, args.out)
    shutil.rmtree(workdir)
    print(f"restyled {args.raw} -> {args.out} "
          f"(Gambar={rs.seq['Gambar']}, Tabel={rs.seq['Tabel']}, "
          f"Persamaan={rs.seq['Persamaan']})")


if __name__ == "__main__":
    main()
