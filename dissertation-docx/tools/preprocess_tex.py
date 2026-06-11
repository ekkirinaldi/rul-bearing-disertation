#!/usr/bin/env python3
"""Preprocess a dissertation chapter .tex fragment for pandoc -> docx conversion.

Usage:
    python3 preprocess_tex.py CHAPTER.tex OUT.tex \
        --aux ../writings/disertation/build/disertasi.aux \
        --figmap assets/figure-map.tsv --bab 6

Transformations
  1. tikzpicture blocks (with optional \\resizebox wrapper) are replaced by
     \\includegraphics of the pre-rendered PNG (figure-map key "tikz:babN").
  2. \\includegraphics paths are rewritten to dissertation-docx asset paths.
  3. Cross-references:
       \\ref{fig:|tab:}  -> @@REF:label@@        (becomes a Word REF field)
       \\ref{eq:}        -> @@REF:label@@
       \\eqref{eq:}      -> (@@REF:label@@)
       \\ref{bab:|sec:|subsec:|lamp:...} -> literal number from the .aux
  4. \\caption{...} inside figure/table envs gets a leading @@CAP:label@@ token.
  5. equation environments: \\label removed, @@EQNUM:label@@ paragraph emitted
     after the environment (restyle.py turns it into a numbered equation).
     align environments are rejected (handle per-chapter when encountered).
  6. The fragment is wrapped in a standalone document with shim macros for
     \\citetitb/\\citenameitb (-> biblatex commands pandoc understands).

Fails loudly on unresolved labels or unsupported constructs.
"""
import argparse
import re
import sys
from pathlib import Path

SHIM = r"""\documentclass[12pt]{report}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{tabularx}
\usepackage{subcaption}
\newcommand{\citetitb}[2][]{\autocite[#1]{#2}}
\newcommand{\citenameitb}[2][]{\textcite[#1]{#2}}
\begin{document}
"""

REF_LITERAL_PREFIXES = ("bab:", "sec:", "subsec:", "lamp:", "app:", "alg:", "chap:")
REF_FIELD_PREFIXES = ("fig:", "tab:", "eq:")


def parse_aux(aux_path: str) -> dict:
    labels = {}
    pat = re.compile(r"\\newlabel\{([^}]+)\}\{\{(.*?)\}\{")
    for line in Path(aux_path).read_text().splitlines():
        m = pat.match(line)
        if m and "@cref" not in m.group(1):
            # strip formatting macros that can appear in the number field
            num = re.sub(r"\\[a-zA-Z]+\s*", "", m.group(2)).replace("{", "").replace("}", "")
            labels[m.group(1)] = num
    return labels


def parse_figmap(figmap_path: str) -> dict:
    fmap = {}
    for line in Path(figmap_path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tex_arg, asset, width = line.split("\t")
        fmap[tex_arg] = (asset, float(width))
    return fmap


def find_env(src: str, env: str, start: int = 0):
    """Yield (begin_idx, end_idx) spans of \\begin{env}...\\end{env}."""
    open_tag, close_tag = f"\\begin{{{env}}}", f"\\end{{{env}}}"
    i = start
    while True:
        b = src.find(open_tag, i)
        if b == -1:
            return
        e = src.find(close_tag, b)
        if e == -1:
            sys.exit(f"FATAL: unbalanced {env} environment")
        yield b, e + len(close_tag)
        i = e + len(close_tag)


def match_braces(src: str, open_idx: int) -> int:
    """Given index of '{', return index after the matching '}'."""
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "{" and (i == 0 or src[i - 1] != "\\"):
            depth += 1
        elif src[i] == "}" and src[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return i + 1
    sys.exit("FATAL: unbalanced braces")


def replace_tikz(src: str, fmap: dict, bab: int) -> str:
    key = f"tikz:bab{bab}"
    out, pos = [], 0
    spans = list(find_env(src, "tikzpicture"))
    if spans and key not in fmap:
        sys.exit(f"FATAL: tikzpicture found but no figure-map entry for {key}")
    for b, e in spans:
        # extend to a wrapping \resizebox{..}{..}{% ... } if present
        rb = src.rfind("\\resizebox", pos, b)
        if rb != -1 and not src[rb:b].strip().startswith("\\resizebox") is False:
            seg = src[rb:b]
            # only treat as wrapper if nothing but the resizebox args between
            if re.fullmatch(r"\\resizebox\{[^}]*\}\{[^}]*\}\{%?\s*", seg):
                b = rb
                e2 = src.find("}", e)  # closing brace of resizebox arg
                if e2 != -1 and src[e:e2].strip() in ("", "%"):
                    e = e2 + 1
        asset, _ = fmap[key]
        out.append(src[pos:b])
        out.append(f"\\includegraphics[width=\\textwidth]{{{asset}}}")
        pos = e
    out.append(src[pos:])
    return "".join(out)


def replace_includegraphics(src: str, fmap: dict) -> str:
    def sub(m):
        opts, arg = m.group(1) or "", m.group(2)
        if arg in fmap:
            return f"\\includegraphics{opts}{{{fmap[arg][0]}}}"
        if arg.startswith("assets/"):  # already rewritten (tikz replacement)
            return m.group(0)
        sys.exit(f"FATAL: no figure-map entry for includegraphics arg '{arg}'")

    return re.sub(r"\\includegraphics(\[[^]]*\])?\{([^}]+)\}", sub, src)


def replace_refs(src: str, labels: dict) -> str:
    unresolved = []

    def lookup(label):
        if label not in labels:
            unresolved.append(label)
            return "??"
        return labels[label]

    def sub_ref(m):
        label = m.group(1)
        if label.startswith(REF_FIELD_PREFIXES):
            return f"@@REF:{label}@@"
        if label.startswith(REF_LITERAL_PREFIXES):
            return lookup(label)
        unresolved.append(label)
        return "??"

    def sub_eqref(m):
        return f"(@@REF:{m.group(1)}@@)"

    src = re.sub(r"\\eqref\{([^}]+)\}", sub_eqref, src)
    src = re.sub(r"\\ref\{([^}]+)\}", sub_ref, src)
    if unresolved:
        sys.exit(f"FATAL: unresolved/unknown ref labels: {sorted(set(unresolved))}")
    return src


def simplify_longtables(src: str) -> str:
    """Keep only the first header of a longtable.

    Pandoc renders both the \\endfirsthead and \\endhead header definitions
    as table rows. Word repeats the (single) header row itself via tblHeader,
    so the continuation header and the foot blocks are dropped.
    """
    def fix(block: str) -> str:
        for end in (r"\endlastfoot", r"\endfoot", r"\endhead"):
            if r"\endfirsthead" in block and end in block:
                pre, _, rest = block.partition("\\endfirsthead")
                return pre + rest.split(end, 1)[1]
        return block

    out, pos = [], 0
    for b, e in find_env(src, "longtable"):
        out.append(src[pos:b])
        out.append(fix(src[b:e]))
        pos = e
    out.append(src[pos:])
    return "".join(out)


def replace_multicolumn(src: str) -> str:
    """\\multicolumn{n}{spec}{X} -> @@SPANn@@X plus n-1 empty cells.

    Pandoc's LaTeX reader drops \\multicolumn cells entirely; the token tells
    restyle.py to set gridSpan=n on the cell and drop the padding cells.
    """
    out, pos = [], 0
    while True:
        i = src.find(r"\multicolumn{", pos)
        if i == -1:
            break
        j = match_braces(src, i + len(r"\multicolumn"))   # {n}
        n = int(src[i + len(r"\multicolumn{"):j - 1])
        k = match_braces(src, j)                           # {spec}
        e = match_braces(src, k)                           # {content}
        content = src[k + 1:e - 1]
        out.append(src[pos:i])
        out.append(f"@@SPAN{n}@@" + content + " & " * (n - 1))
        pos = e
    out.append(src[pos:])
    return "".join(out)


def replace_subcaptions(src: str) -> str:
    """\\caption*{X} (minipage subcaptions) -> plain paragraph X.

    Pandoc drops \\caption* inside minipages entirely; as a paragraph the
    text survives in the layout-table cell below its image.
    """
    out = []
    pos = 0
    while True:
        i = src.find(r"\caption*{", pos)
        if i == -1:
            break
        end = match_braces(src, i + len(r"\caption*"))
        content = src[i + len(r"\caption*{"):end - 1]
        out.append(src[pos:i])
        out.append(f"\\par {content}\\par")
        pos = end
    out.append(src[pos:])
    return "".join(out)


def tag_captions(src: str) -> str:
    """Inside figure/table/longtable envs, prefix the caption with @@CAP:label@@."""
    for env in ("figure", "table", "longtable"):
        while True:
            replaced = False
            for b, e in find_env(src, env):
                block = src[b:e]
                if "@@CAP:" in block:
                    continue
                lab = re.search(r"\\label\{((?:fig|tab):[^}]+)\}", block)
                cap = block.find("\\caption{")
                if not lab or cap == -1:
                    sys.exit(f"FATAL: {env} without caption+label: {block[:120]}...")
                insert_at = b + cap + len("\\caption{")
                src = src[:insert_at] + f"@@CAP:{lab.group(1)}@@" + src[insert_at:]
                replaced = True
                break
            if not replaced:
                break
    return src


SYMBOL_MAP = {
    r"\rightarrow": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
    r"\times": "×", r"\approx": "≈", r"\leq": "≤", r"\geq": "≥",
    r"\pm": "±", r"\sim": "~", r"\cdot": "·", r"\infty": "∞",
}

SIMPLE_MATH_CHARS = re.compile(r"^[A-Za-z0-9=+\-<>(),.%\s]*$")


def demote_simple_math(src: str) -> str:
    """Turn trivial inline math into plain text.

    OMML gives punctuation spacing to commas, so $r = 0{,}507$ renders as
    "0 , 507" in Word/LO. Scalar expressions (letters, digits, decimal commas,
    basic operators, lone symbols) do not need a math zone at all: emit text
    with single-letter variables italicized. Anything with \\frac, sub/sup-
    scripts etc. stays math.
    """
    def sub(m):
        body = m.group(1).strip()
        for tex, uni in SYMBOL_MAP.items():
            if body == tex:
                return uni
            body = body.replace(tex + " ", uni + " ").replace(" " + tex, " " + uni)
        body = body.replace("{,}", ",").replace("{.}", ".")
        if not SIMPLE_MATH_CHARS.match(body.replace("→", "").replace("×", "")
                                       .replace("≈", "").replace("≤", "")
                                       .replace("≥", "").replace("±", "")):
            return m.group(0)  # keep as math
        body = re.sub(r"[A-Za-z]+", lambda w: rf"\emph{{{w.group(0)}}}", body)
        return body.replace("%", r"\%")

    return re.sub(r"\$([^$]+)\$", sub, src)


def tag_equations(src: str) -> str:
    if "\\begin{align}" in src or "\\begin{align*}" in src:
        sys.exit("FATAL: align environment present - extend tag_equations first")
    auto = [0]

    def process(m):
        body = m.group(1)
        lab = re.search(r"\\label\{([^}]+)\}", body)
        if lab:
            label = lab.group(1)
            body = body.replace(lab.group(0), "")
        else:
            auto[0] += 1
            label = f"eq:auto_{auto[0]}"
        return (f"\\begin{{equation}}{body}\\end{{equation}}\n\n"
                f"@@EQNUM:{label}@@\n")

    return re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", process, src, flags=re.S)


def extract_table_hints(src: str) -> list:
    """Column-width weights for each tabular/longtable, in document order.

    p{Xcm} columns weigh X; X (tabularx) weighs 3; l/c/r weigh 2.
    Side-by-side minipages inside a figure become a pandoc layout table:
    one hint per group, weighted by the minipage widths.
    restyle.py turns the weights into fixed DXA column widths.
    """
    positioned = []
    for m in re.finditer(r"\\begin\{(tabular|longtable|tabularx)\}", src):
        i = m.end()
        if m.group(1) == "tabularx":  # first arg is the width - skip it
            i = match_braces(src, i)
        spec = src[i:match_braces(src, i)]
        # strip >{...} <{...} @{...} !{...} column decorators before tokenizing
        while True:
            stripped = re.sub(r"[><@!]\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", "", spec)
            # expand *{N}{cols} repetition syntax
            stripped = re.sub(r"\*\{(\d+)\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                              lambda mm: int(mm.group(1)) * mm.group(2), stripped)
            if stripped == spec:
                break
            spec = stripped
        weights = []
        for cm in re.finditer(r"[pmb]\{([0-9.]+)cm\}|([lcrX])", spec):
            if cm.group(1):
                weights.append(float(cm.group(1)))
            elif cm.group(2) == "X":
                weights.append(3.0)
            else:
                weights.append(2.0)
        if weights:
            positioned.append((m.start(), weights))
    for b, e in find_env(src, "figure"):
        mp = re.findall(r"\\begin\{minipage\}(?:\[[^]]*\])?\{([0-9.]+)\\textwidth\}",
                        src[b:e])
        if len(mp) >= 2:
            positioned.append((b, [float(w) for w in mp]))
    return [w for _, w in sorted(positioned)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--aux", required=True)
    ap.add_argument("--figmap", required=True)
    ap.add_argument("--bab", type=int, required=True)
    args = ap.parse_args()

    src = Path(args.src).read_text()
    labels = parse_aux(args.aux)
    fmap = parse_figmap(args.figmap)

    # guard: constructs the pipeline does not handle
    for bad in (r"\num{", r"\SI{", r"\cref{", r"\autoref{", r"\pageref{",
                r"\begin{algorithm}", r"\footnote{"):
        if bad in src:
            sys.exit(f"FATAL: unsupported construct in source: {bad}")

    # table hints must see the original minipage/figure structure
    import json
    hints = extract_table_hints(src)

    src = replace_tikz(src, fmap, args.bab)
    src = replace_includegraphics(src, fmap)
    src = demote_simple_math(src)
    src = simplify_longtables(src)
    src = replace_multicolumn(src)
    src = replace_subcaptions(src)
    src = tag_captions(src)
    src = tag_equations(src)
    src = replace_refs(src, labels)

    Path(args.out + ".tblhints.json").write_text(json.dumps(hints))
    Path(args.out).write_text(SHIM + src + "\n\\end{document}\n")
    print(f"preprocessed {args.src} -> {args.out} ({len(hints)} table hints)")


if __name__ == "__main__":
    main()
