#!/usr/bin/env python3
"""Assemble the final disertasi.docx.

Order: frontmatter -> Bab I..VI -> DAFTAR PUSTAKA (fresh citeproc run over
every cited key) -> LAMPIRAN divider page -> Lampiran A..G.

Strategy: the front-matter package (template-derived) is the base; chapter
and lampiran bodies are transplanted into it. All packages share the same
template footers/styles, so footer relationship IDs line up; images are
copied across with fresh relationship IDs. The merged main body is a single
section: Arabic page numbers from 1, Heading 1 numbering runs Bab I..VI
naturally (per-chapter startOverrides are not copied), and SEQ caption
fields restart per chapter via \\s 1.

Usage: python3 tools/merge_master.py
"""
import copy
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unpack import unpack          # noqa: E402
from pack import pack              # noqa: E402
import restyle                     # noqa: E402

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

BASE = Path(__file__).resolve().parent.parent
TEXDIR = BASE.parent / "writings" / "disertation"

CHAPTERS = [BASE / "chapters" / f"bab{n}.docx" for n in range(1, 7)]
LAMPIRAN = [BASE / "lampiran" / f"lamp{l}.docx" for l in "ABCDEFG"]
TEX_SOURCES = (sorted((TEXDIR / "chapters").glob("0[1-6]-*.tex"))
               + sorted((TEXDIR / "lampiran").glob("[A-G]-*.tex")))


def para_text(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def page_break_para():
    p = etree.Element(W + "p")
    r = etree.SubElement(p, W + "r")
    br = etree.SubElement(r, W + "br")
    br.set(W + "type", "page")
    return p


def make_heading(style, text, jc=None):
    p = etree.Element(W + "p")
    ppr = etree.SubElement(p, W + "pPr")
    ps = etree.SubElement(ppr, W + "pStyle")
    ps.set(W + "val", style)
    if jc:
        j = etree.SubElement(ppr, W + "jc")
        j.set(W + "val", jc)
    r = etree.SubElement(p, W + "r")
    t = etree.SubElement(r, W + "t")
    t.set(XML_SPACE, "preserve")
    t.text = text
    return p


# -- bibliography ------------------------------------------------------------

def cited_keys():
    keys = set()
    pat = re.compile(r"\\cite(?:titb|nameitb)(?:\[[^]]*\])?\{([^}]+)\}")
    for tex in TEX_SOURCES:
        for m in pat.finditer(tex.read_text()):
            keys.update(k.strip() for k in m.group(1).split(","))
    return sorted(keys)


def build_bibliography():
    """citeproc run over every cited key -> list of Daftarpustaka paras."""
    keys = cited_keys()
    frag = ("\\documentclass{article}\\begin{document}\n"
            "\\nocite{" + ",".join(keys) + "}\n\\end{document}\n")
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / "bib.tex"
        tex.write_text(frag)
        out = Path(td) / "bib.docx"
        subprocess.run(
            ["pandoc", str(tex), "-f", "latex", "-t", "docx", "--citeproc",
             "--csl", str(BASE / "assets" / "itb-sps.csl"),
             "--bibliography", str(TEXDIR / "references.bib"),
             "--metadata", "lang=id-ID", "-o", str(out)],
            check=True)
        with zipfile.ZipFile(out) as z:
            root = etree.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root[0].findall(W + "p"):
        ps = p.find(f"{W}pPr/{W}pStyle")
        if ps is not None and ps.get(W + "val") == "Bibliography" \
                and para_text(p).strip():
            ps.set(W + "val", "Daftarpustaka")
            paras.append(p)
    assert len(paras) == len(keys), \
        f"bibliography count mismatch: {len(paras)} paras vs {len(keys)} keys"
    return paras


# -- package transplantation -------------------------------------------------

class Merger:
    def __init__(self, base_dir):
        self.dir = Path(base_dir)
        self.doc = etree.parse(str(self.dir / "word" / "document.xml"))
        self.body = self.doc.getroot()[0]
        self.rels = etree.parse(str(self.dir / "word" / "_rels"
                                    / "document.xml.rels"))
        self.next_rid = 1 + max(
            int(rel.get("Id")[3:]) for rel in self.rels.getroot()
            if rel.get("Id", "").startswith("rId"))
        self.next_img = 1000
        self.styles = etree.parse(str(self.dir / "word" / "styles.xml"))
        self.style_ids = {s.get(W + "styleId")
                          for s in self.styles.getroot().iter(W + "style")}

    def detach_final_sectpr(self):
        """Turn the body-level sectPr into an explicit section break so the
        appended main matter forms a new section."""
        sectpr = self.body.find(W + "sectPr")
        self.body.remove(sectpr)
        p = etree.Element(W + "p")
        ppr = etree.SubElement(p, W + "pPr")
        ppr.append(sectpr)
        self.body.append(p)
        return sectpr

    def append_package(self, docx_path):
        """Append a chapter/lampiran body (minus its sectPr); returns the
        package's sectPr for possible reuse as the final body sectPr."""
        src = Path(tempfile.mkdtemp(prefix="merge_src_"))
        unpack(str(docx_path), str(src))
        sdoc = etree.parse(str(src / "word" / "document.xml"))
        sbody = sdoc.getroot()[0]
        srels = {rel.get("Id"): rel for rel in etree.parse(
            str(src / "word" / "_rels" / "document.xml.rels")).getroot()}
        sectpr = sbody.find(W + "sectPr")
        sbody.remove(sectpr)

        self._merge_styles(src)
        for el in list(sbody):
            self._remap_rels(el, srels, src)
            self.body.append(el)
        shutil.rmtree(src)
        return sectpr

    def _merge_styles(self, src):
        """Copy styles the base lacks (e.g. pandoc's SourceCode) across."""
        ssty = etree.parse(str(src / "word" / "styles.xml"))
        for s in ssty.getroot().iter(W + "style"):
            sid = s.get(W + "styleId")
            if sid not in self.style_ids:
                self.styles.getroot().append(copy.deepcopy(s))
                self.style_ids.add(sid)

    def _remap_rels(self, el, srels, src):
        """Images referenced from a transplanted element: copy the media part
        and register a fresh relationship in the base package."""
        for blip in el.iter("{http://schemas.openxmlformats.org/drawingml/"
                            "2006/main}blip"):
            rid = blip.get(R + "embed")
            target = srels[rid].get("Target")
            data = (src / "word" / target).read_bytes()
            ext = Path(target).suffix
            name = f"media/merged{self.next_img}{ext}"
            self.next_img += 1
            dest = self.dir / "word" / name
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(data)
            new_rid = f"rId{self.next_rid}"
            self.next_rid += 1
            rel = etree.SubElement(self.rels.getroot(), REL + "Relationship")
            rel.set("Id", new_rid)
            rel.set("Type", srels[rid].get("Type"))
            rel.set("Target", name)
            blip.set(R + "embed", new_rid)

    def finish(self, final_sectpr):
        # single main-matter section: starts on a fresh page, Arabic from 1
        t = final_sectpr.find(W + "type")
        if t is None:
            t = etree.Element(W + "type")
            pgsz = final_sectpr.find(W + "pgSz")
            (pgsz.addprevious(t) if pgsz is not None
             else final_sectpr.insert(0, t))
        t.set(W + "val", "nextPage")
        pg = final_sectpr.find(W + "pgNumType")
        pg.set(W + "fmt", "decimal")
        pg.set(W + "start", "1")
        # footers: pandoc kept only the cover footers (footer1 = unnumbered)
        # in the chapter packages; the template's main body uses footer4
        # (decimal PAGE field). Point the section there.
        footer4_rid = next(rel.get("Id") for rel in self.rels.getroot()
                           if rel.get("Target") == "footer4.xml")
        for ref in list(final_sectpr):
            if "footerReference" in ref.tag or "headerReference" in ref.tag \
                    or ref.tag == W + "titlePg":
                final_sectpr.remove(ref)
        fr = etree.Element(W + "footerReference")
        fr.set(W + "type", "default")
        fr.set(R + "id", footer4_rid)
        final_sectpr.insert(0, fr)
        self.body.append(final_sectpr)
        self.doc.write(str(self.dir / "word" / "document.xml"),
                       xml_declaration=True, encoding="UTF-8",
                       standalone=True)
        self.styles.write(str(self.dir / "word" / "styles.xml"),
                          xml_declaration=True, encoding="UTF-8",
                          standalone=True)
        self.rels.write(str(self.dir / "word" / "_rels"
                            / "document.xml.rels"),
                        xml_declaration=True, encoding="UTF-8",
                        standalone=True)


def main():
    workdir = Path(tempfile.mkdtemp(prefix="merge_"))
    unpack(str(BASE / "frontmatter" / "frontmatter.docx"), str(workdir))
    m = Merger(workdir)
    m.detach_final_sectpr()

    final_sectpr = None
    for i, ch in enumerate(CHAPTERS):
        if i:
            m.body.append(page_break_para())
        sectpr = m.append_package(ch)
        if final_sectpr is None:
            final_sectpr = sectpr
        print(f"appended {ch.name}")

    # daftar pustaka
    m.body.append(page_break_para())
    m.body.append(make_heading("JudulBab", "DAFTAR PUSTAKA"))
    bib = build_bibliography()
    for p in bib:
        m.body.append(p)
    print(f"bibliography: {len(bib)} entries")

    # lampiran divider page (Pedoman: a page containing only LAMPIRAN)
    m.body.append(page_break_para())
    m.body.append(make_heading("JudulBab", "LAMPIRAN"))

    for lamp in LAMPIRAN:
        m.body.append(page_break_para())
        m.append_package(lamp)
        print(f"appended {lamp.name}")

    m.finish(final_sectpr)

    # numbering: subbab flush-left fix; chapter Heading1s number naturally
    restyle.fix_numbering(workdir / "word" / "numbering.xml", 1)
    restyle.add_extra_styles(workdir / "word" / "styles.xml")
    restyle.enable_mirror_margins(workdir / "word" / "settings.xml")

    # post-merge assertions
    doc = etree.parse(str(workdir / "word" / "document.xml"))
    xml = etree.tostring(doc).decode()
    assert "@@" not in xml, "leftover tokens in merged document"
    body = doc.getroot()[0]
    h1 = [p for p in body.iter(W + "p")
          if (ps := p.find(f"{W}pPr/{W}pStyle")) is not None
          and ps.get(W + "val") == "Heading1"]
    assert len(h1) == 6, f"expected 6 Heading1 (Bab I..VI), got {len(h1)}"
    lamps = [p for p in body.iter(W + "p")
             if (ps := p.find(f"{W}pPr/{W}pStyle")) is not None
             and ps.get(W + "val") == "Lampiran"]
    assert len(lamps) == 7, f"expected 7 Lampiran headings, got {len(lamps)}"
    used = {ps.get(W + "val") for ps in body.iter(W + "pStyle")}
    missing = used - m.style_ids
    assert not missing, f"styles referenced but undefined: {missing}"

    out = BASE / "disertasi.docx"
    pack(str(workdir), str(out))
    shutil.rmtree(workdir)
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
