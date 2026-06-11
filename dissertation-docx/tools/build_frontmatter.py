#!/usr/bin/env python3
"""Build frontmatter/frontmatter.docx from the ITB template package.

Keeps the template's cover, pengesahan, pedoman penggunaan pages and the
Roman-numeral footers / section breaks; fills in the dissertation metadata
and transplants abstrak ID/EN + kata pengantar from the LaTeX sources;
swaps the placeholder daftar isi/gambar/tabel for live TOC fields; rebuilds
the daftar singkatan table from 00-daftar-singkatan.tex.

Usage: python3 tools/build_frontmatter.py
"""
import copy
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unpack import unpack          # noqa: E402
from pack import pack              # noqa: E402

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

BASE = Path(__file__).resolve().parent.parent
TEXDIR = BASE.parent / "writings" / "disertation"

TITLE_ID = ("PERAWATAN PREDIKTIF UNTUK SISTEM PRODUKSI DENGAN PENDEKATAN "
            "ANALISIS BIG DATA DAN KECERDASAN BUATAN MENGGUNAKAN DATA "
            "KONDISI MESIN DAN INFORMASI KUALITAS YANG REAL TIME")
# TODO(verify-with-pembimbing): judul EN belum ada di sumber LaTeX
TITLE_EN = ("PREDICTIVE MAINTENANCE FOR PRODUCTION SYSTEMS WITH BIG DATA "
            "ANALYTICS AND ARTIFICIAL INTELLIGENCE APPROACH USING REAL-TIME "
            "MACHINE CONDITION DATA AND QUALITY INFORMATION")
AUTHOR_UP, AUTHOR = "TOTO SUHARTO", "Toto Suharto"
NIM = "NIM: 33420002"
PRODI_ID = "(Program Studi Doktor Teknik dan Manajemen Industri)"
PRODI_EN = "(Doctoral Program in Industrial Engineering and Management)"
PROMOTOR = "Prof. Dr. Ir. Kadarsah Suryadi, DEA"
PEMBIMBING2 = "Prof. Ir. Bermawi Priyatna Iskandar, M.Sc., Ph.D."
PEMBIMBING3 = "Prof. Dr. Ir. Bambang Riyanto Trilaksono"
BULAN_TAHUN = "Mei 2026"

FIELD_HINT = "[Perbarui daftar ini di Word: Ctrl+A lalu F9]"


def para_text(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def para_style(p):
    ps = p.find(f"{W}pPr/{W}pStyle")
    return ps.get(W + "val") if ps is not None else ""


def set_para_text(p, text):
    """Replace the paragraph content with a single run, keeping pPr and the
    first run's rPr."""
    first_rpr = None
    for r in p.findall(W + "r"):
        if first_rpr is None:
            first_rpr = r.find(W + "rPr")
            if first_rpr is not None:
                first_rpr = copy.deepcopy(first_rpr)
        p.remove(r)
    run = etree.SubElement(p, W + "r")
    if first_rpr is not None:
        run.append(first_rpr)
    t = etree.SubElement(run, W + "t")
    t.set(XML_SPACE, "preserve")
    t.text = text


def make_para(style, runs):
    """runs = list of (text, bold, italic, jc) -- jc applied at para level via
    the style only; use make_para_jc for explicit alignment."""
    p = etree.Element(W + "p")
    ppr = etree.SubElement(p, W + "pPr")
    ps = etree.SubElement(ppr, W + "pStyle")
    ps.set(W + "val", style)
    for text, bold, italic in runs:
        r = etree.SubElement(p, W + "r")
        rpr = etree.SubElement(r, W + "rPr")
        if bold:
            etree.SubElement(rpr, W + "b")
        if italic:
            etree.SubElement(rpr, W + "i")
        t = etree.SubElement(r, W + "t")
        t.set(XML_SPACE, "preserve")
        t.text = text
    return p


def tex_runs(s):
    """Tiny LaTeX -> run list: handles \\emph{}/\\textbf{}, \\textsuperscript
    is flattened, ~ -> nbsp, -- -> endash."""
    s = (s.replace("~", " ").replace("---", "—")
         .replace("--", "–").replace(r"\%", "%").replace(r"\&", "&"))
    s = re.sub(r"\\textsuperscript\{([^{}]*)\}", r"\1", s)
    runs = []
    pos = 0
    for m in re.finditer(r"\\(emph|textbf)\{([^{}]*)\}", s):
        if m.start() > pos:
            runs.append((s[pos:m.start()], False, False))
        runs.append((m.group(2), m.group(1) == "textbf", m.group(1) == "emph"))
        pos = m.end()
    if pos < len(s):
        runs.append((s[pos:], False, False))
    return [(t, b, i) for t, b, i in runs if t]


def make_toc_field(instr):
    p = etree.Element(W + "p")
    fld = etree.SubElement(p, W + "fldSimple")
    fld.set(W + "instr", instr)
    r = etree.SubElement(fld, W + "r")
    t = etree.SubElement(r, W + "t")
    t.set(XML_SPACE, "preserve")
    t.text = FIELD_HINT
    return p


# -- LaTeX front-matter sources --------------------------------------------

def convert_fragment(tex_path, env):
    """pandoc a front-matter fragment -> list of paragraph elements."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocess_tex import demote_simple_math, modernize_math
    src = Path(tex_path).read_text()
    m = re.search(r"\\begin\{%s\}(.*)\\end\{%s\}" % (env, env), src, re.S)
    body = m.group(1)
    # signature block (kata pengantar) handled separately
    body = re.sub(r"\\begin\{flushright\}.*?\\end\{flushright\}", "", body,
                  flags=re.S)
    # keywords paragraph handled separately
    body = re.sub(r"\\bigskip\\noindent\s*\\textbf\{(Kata kunci|Keywords):\}.*",
                  "", body, flags=re.S)
    # flatten enumerate to literal numbering
    def flat_enum(m):
        items = re.split(r"\\item\s*", m.group(1))
        return "\n\n".join(f"{i}. {it.strip()}"
                           for i, it in enumerate(items[1:], 1))
    body = re.sub(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", flat_enum,
                  body, flags=re.S)
    body = re.sub(r"\\vspace\{[^}]*\}|\\bigskip|\\noindent", "", body)
    body = modernize_math(body)
    body = demote_simple_math(body)
    wrapped = ("\\documentclass[12pt]{report}\n\\begin{document}\n"
               + body + "\n\\end{document}\n")
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / "frag.tex"
        tex.write_text(wrapped)
        out = Path(td) / "frag.docx"
        subprocess.run(["pandoc", str(tex), "-f", "latex", "-t", "docx",
                        "-o", str(out)], check=True)
        import zipfile
        with zipfile.ZipFile(out) as z:
            root = etree.fromstring(z.read("word/document.xml"))
    paras = []
    for el in root[0]:
        if el.tag == W + "p" and para_text(el).strip():
            paras.append(el)
    return paras


def keywords_text(tex_path, tag):
    src = Path(tex_path).read_text()
    m = re.search(r"\\textbf\{%s:\}(.*?)\\end\{" % tag, src, re.S)
    kw = re.sub(r"\\emph\{([^{}]*)\}", r"\1", m.group(1))
    return re.sub(r"\s+", " ", kw).strip()


def parse_singkatan(tex_path):
    rows = []
    for line in Path(tex_path).read_text().splitlines():
        m = re.match(r"^\s*(\S[^&]*?)\s*&\s*(.*?)\s*\\\\\s*$", line)
        if m and not m.group(1).startswith(("\\textbf", "\\hline")):
            rows.append((m.group(1), m.group(2)))
    return rows


def restyle_paras(paras, style, separators=False):
    for p in paras:
        ppr = p.find(W + "pPr")
        if ppr is None:
            ppr = etree.Element(W + "pPr")
            p.insert(0, ppr)
        ps = ppr.find(W + "pStyle")
        if ps is None:
            ps = etree.SubElement(ppr, W + "pStyle")
        ps.set(W + "val", style)
    if separators:
        out = []
        for i, p in enumerate(paras):
            if i:
                out.append(make_para(style, []))
            out.append(p)
        return out
    return paras


# -- main -------------------------------------------------------------------

def main():
    workdir = Path(tempfile.mkdtemp(prefix="frontmatter_"))
    unpack(str(BASE / "assets" / "template.docx"), str(workdir))
    docpath = workdir / "word" / "document.xml"
    tree = etree.parse(str(docpath))
    body = tree.getroot()[0]

    # 1. keep only the front-matter sections (cut at the 2nd section break)
    sect_paras = [p for p in body.findall(W + "p")
                  if p.find(f"{W}pPr/{W}sectPr") is not None]
    cut = sect_paras[1]
    sectpr = copy.deepcopy(cut.find(f"{W}pPr/{W}sectPr"))
    idx = list(body).index(cut)
    for el in list(body)[idx:]:
        body.remove(el)
    body.append(sectpr)

    # 2. plain text substitutions at run level
    RUN_SUBS = [
        ("NIM: 35000001", NIM), ("NIM: 3500001", NIM),
        ("NAMA MAHASISWA", AUTHOR_UP), ("Nama Mahasiswa", AUTHOR),
        ("(Program Studi Doktor Teknik Sipil)", PRODI_ID),
        ("(Doctoral Program in Civil Engineering)", PRODI_EN),
        ("Nama Pembimbing 1", PROMOTOR),
        ("Nama Pembimbing 2", PEMBIMBING2),
        ("Nama Pembimbing 3", PEMBIMBING3),
        ("Bulan 2016", BULAN_TAHUN),
    ]
    for t in body.iter(W + "t"):
        if t.text:
            for old, new in RUN_SUBS:
                t.text = t.text.replace(old, new)

    # 2b. placeholders split across runs: rewrite the whole paragraph
    for p in body.findall(W + "p"):
        txt = para_text(p)
        if re.fullmatch(r"NIM:\s*3500+1", txt.strip()):
            set_para_text(p, NIM)
        elif "Teknik Sipil" in txt and "Doctoral" not in txt:
            set_para_text(p, PRODI_ID)
        elif "Civil Engineering" in txt:
            set_para_text(p, PRODI_EN)
        elif txt.strip() in ("NAMA MAHASISWA", "Nama Mahasiswa"):
            set_para_text(p, AUTHOR_UP if txt.strip().isupper() else AUTHOR)

    # 3. title placeholders (whole paragraph rewrite)
    for p in body.findall(W + "p"):
        txt = para_text(p)
        if txt.startswith("TULIS JUDUL DISERTASI PADA BAGIAN INI") or \
           txt.startswith("TULIS JUDUL DISERTASI BAHASA INDONESIA"):
            set_para_text(p, TITLE_ID)
        elif txt.startswith("JUDUL BAGIAN DISERTASI DAN JUDUL DISERTASI"):
            set_para_text(p, TITLE_EN)
        elif txt.startswith("JUDUL DISERTASI DITULIS DENGAN HURUF KAPITAL"):
            set_para_text(p, TITLE_ID)
            # pengesahan must start its own page (template relied on the
            # catatan filler pages that are deleted below)
            ppr = p.find(W + "pPr")
            if ppr.find(W + "pageBreakBefore") is None:
                ps = ppr.find(W + "pStyle")
                pbb = etree.Element(W + "pageBreakBefore")
                ppr.insert(1 if ps is not None else 0, pbb)

    # 4. abstrak ID
    abstrak = restyle_paras(
        convert_fragment(TEXDIR / "chapters" / "00-abstrak-id.tex",
                         "abstractid"), "Abstrak", separators=True)
    kw_id = keywords_text(TEXDIR / "chapters" / "00-abstrak-id.tex",
                          "Kata kunci")
    for p in list(body.findall(W + "p")):
        txt = para_text(p)
        if para_style(p) == "Abstrak" and (
                txt.startswith("Abstrak ditulis dalam bahasa")
                or txt.startswith("Abstrak disertasi memuat")):
            body.remove(p)
        elif txt.startswith("Kata kunci: kata kunci 1"):
            for el in abstrak:
                p.addprevious(el)
            new = make_para("Abstrak", [("Kata kunci: ", True, False),
                                        (kw_id, False, False)])
            p.addprevious(new)
            body.remove(p)

    # 5. abstract EN
    abstract = restyle_paras(
        convert_fragment(TEXDIR / "chapters" / "00-abstract-en.tex",
                         "abstracten"), "Abstrak", separators=True)
    kw_en = keywords_text(TEXDIR / "chapters" / "00-abstract-en.tex",
                          "Keywords")
    for p in list(body.findall(W + "p")):
        txt = para_text(p)
        if txt.startswith("Pada bagian ini, abstrak ditulis dalam bahasa"):
            body.remove(p)
        elif txt.startswith("Keywords: kata kunci 1"):
            for el in abstract:
                p.addprevious(el)
            new = make_para("Abstrak", [("Keywords: ", True, False),
                                        (kw_en, False, False)])
            p.addprevious(new)
            body.remove(p)

    # 6. kata pengantar
    kp = restyle_paras(
        convert_fragment(TEXDIR / "chapters" / "00-kata-pengantar.tex",
                         "katapengantar"), "Paragraf", separators=True)
    for p in list(body.findall(W + "p")):
        txt = para_text(p)
        if txt.startswith("Halaman kata pengantar dicetak"):
            for el in kp:
                p.addprevious(el)
            # signature block, right-aligned
            for line in ("Bandung, " + BULAN_TAHUN, "", "", AUTHOR):
                sig = make_para("Paragraf", [(line, False, False)] if line else [])
                jc = etree.SubElement(sig.find(W + "pPr"), W + "jc")
                jc.set(W + "val", "right")
                p.addprevious(sig)
            body.remove(p)
        elif txt.startswith("Cara menulis kata pengantar"):
            body.remove(p)

    # 7. daftar isi / lampiran / gambar / tabel -> TOC fields
    TOC_FIELDS = {
        "DAFTAR ISI": ' TOC \\o "1-3" \\h \\z ',
        "DAFTAR LAMPIRAN": ' TOC \\h \\z \\t "Lampiran;1;Lampiransub1;2" ',
        "DAFTAR GAMBAR DAN ILUSTRASI": ' TOC \\h \\z \\c "Gambar" ',
        "DAFTAR TABEL": ' TOC \\h \\z \\c "Tabel" ',
    }
    for p in list(body.findall(W + "p")):
        if para_style(p) in ("TOC1", "TOC2", "TOC3"):
            body.remove(p)
        elif para_style(p) in ("judulBab15", "JudulBab") and \
                para_text(p).strip() in TOC_FIELDS:
            p.addnext(make_toc_field(TOC_FIELDS[para_text(p).strip()]))

    # 8. halaman peruntukan: optional page, not used in the manuscript
    for p in list(body.findall(W + "p")):
        if para_style(p) == "halperuntukan" or \
                para_text(p).strip() == "HALAMAN PERUNTUKAN":
            body.remove(p)

    # 9. template usage notes
    for p in list(body.findall(W + "p")):
        txt = para_text(p)
        if para_style(p) == "keterangan" or txt.startswith("Catatan :") or \
                txt.startswith("Penomoran halaman abstrak"):
            body.remove(p)

    # 9b. cover: the real title is 4 lines vs the 2-line placeholder; drop
    # two spacer paragraphs before the ITB logo so the cover fits one page
    logo_p = next(p for p in body.findall(W + "p")
                  if p.find(f".//{W}drawing") is not None)
    removed = 0
    sib = logo_p.getprevious()
    while removed < 2 and sib is not None and sib.tag == W + "p" \
            and not para_text(sib).strip():
        prev = sib.getprevious()
        body.remove(sib)
        sib = prev
        removed += 1

    # 9c. orphaned page breaks: deleting template blocks can leave two break
    # paragraphs with nothing between them -> blank pages. Collapse any
    # break/empty run that contains a break into a single break paragraph.
    def is_empty_p(el):
        return el.tag == W + "p" and not para_text(el).strip() \
            and el.find(f".//{W}drawing") is None
    def has_break(el):
        return el.tag == W + "p" and \
            el.find(f'.//{W}br[@{W}type="page"]') is not None
    run = []
    for el in list(body) + [None]:
        if el is not None and is_empty_p(el):
            run.append(el)
            continue
        if any(has_break(x) for x in run) and len(run) > 1:
            keep = next(x for x in run if has_break(x))
            for x in run:
                if x is not keep:
                    body.remove(x)
        run = []

    # 9d. DAFTAR LAMPIRAN gets its own page (template flowed it after the
    # placeholder daftar isi)
    for p in body.findall(W + "p"):
        if para_style(p) in ("judulBab15", "JudulBab") and \
                para_text(p).strip() == "DAFTAR LAMPIRAN":
            br = etree.Element(W + "p")
            r = etree.SubElement(br, W + "r")
            b = etree.SubElement(r, W + "br")
            b.set(W + "type", "page")
            p.addprevious(br)

    # 10. daftar singkatan table
    rows = parse_singkatan(TEXDIR / "chapters" / "00-daftar-singkatan.tex")
    tbl = body.find(W + "tbl")
    assert tbl is not None, "template singkatan table not found"
    trs = tbl.findall(W + "tr")
    sample = trs[1]
    # header row kept from template (3-column; halaman column left blank)
    for tr in trs[1:]:
        tbl.remove(tr)
    for sing, longform in rows:
        tr = copy.deepcopy(sample)
        tcs = tr.findall(W + "tc")
        cells = [tex_runs(sing), tex_runs(longform), []]
        for tc, runs in zip(tcs, cells):
            for old in tc.findall(W + "p")[1:]:
                tc.remove(old)
            p = tc.find(W + "p")
            for r in p.findall(W + "r"):
                p.remove(r)
            for text, bold, italic in runs:
                r = etree.SubElement(p, W + "r")
                rpr = etree.SubElement(r, W + "rPr")
                if bold:
                    etree.SubElement(rpr, W + "b")
                if italic:
                    etree.SubElement(rpr, W + "i")
                t = etree.SubElement(r, W + "t")
                t.set(XML_SPACE, "preserve")
                t.text = text
        tbl.append(tr)

    tree.write(str(docpath), xml_declaration=True, encoding="UTF-8",
               standalone=True)
    outdir = BASE / "frontmatter"
    outdir.mkdir(exist_ok=True)
    out = outdir / "frontmatter.docx"
    pack(str(workdir), str(out))
    print(f"built {out} ({len(rows)} singkatan, "
          f"{len(abstrak)}+{len(abstract)} abstract paras, {len(kp)} KP paras)")


if __name__ == "__main__":
    main()
