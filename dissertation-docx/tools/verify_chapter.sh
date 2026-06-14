#!/bin/bash
# Verify a converted chapter docx against its LaTeX source.
# Usage: tools/verify_chapter.sh chapters/bab6.docx ../writings/disertation/chapters/06-kesimpulan.tex
set -u
DOCX="$1"
TEX="$2"
SOFFICE="/Applications/LibreOffice.app/Contents/MacOS/soffice"
OUT="build/verify"
mkdir -p "$OUT"
fail=0

# 1. XML well-formed + no leftover tokens
python3 - "$DOCX" <<'EOF' || fail=1
import sys, zipfile
from lxml import etree
with zipfile.ZipFile(sys.argv[1]) as z:
    for n in z.namelist():
        if n.endswith(('.xml', '.rels')):
            etree.fromstring(z.read(n))
    doc = z.read('word/document.xml').decode()
assert '@@' not in doc, 'leftover @@ tokens'
print('OK xml well-formed, no tokens')
EOF

# 2. render to PDF
"$SOFFICE" --headless --convert-to pdf --outdir "$OUT" "$DOCX" >/dev/null 2>&1
PDF="$OUT/$(basename "${DOCX%.docx}").pdf"
[ -f "$PDF" ] && echo "OK rendered $PDF" || { echo "FAIL: PDF render"; fail=1; }

# 3. rendered text sanity: no unresolved references
pdftotext "$PDF" "$OUT/docx_text.txt" 2>/dev/null
if grep -q "Error: Reference source not found" "$OUT/docx_text.txt"; then
  echo "FAIL: unresolved REF fields in render"; fail=1
else
  echo "OK no unresolved REF fields"
fi
if grep -q '??' "$OUT/docx_text.txt"; then
  echo "WARN: '??' found in rendered text"; fi

# 4. content volume: docx text within 10% of detexed LaTeX
python3 - "$OUT/docx_text.txt" "$TEX" <<'EOF' || fail=1
import re, sys
docx_words = len(open(sys.argv[1]).read().split())
tex = open(sys.argv[2]).read()
tex = re.sub(r'(?<!\\)%.*', '', tex)               # comments
tex = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', tex, flags=re.S)
tex = re.sub(r'\\[a-zA-Z]+(\[[^]]*\])?', ' ', tex)  # commands
tex = re.sub(r'[{}~$&]', ' ', tex)
tex_words = len(tex.split())
ratio = docx_words / tex_words
print(f'docx={docx_words}w tex~={tex_words}w ratio={ratio:.2f}')
assert 0.85 < ratio < 1.30, 'content volume mismatch'
print('OK content volume')
EOF

# 5. citation sanity: rendered text has no raw cite keys or et al.
if grep -qE '\\cite|citetitb|et al\.' "$OUT/docx_text.txt"; then
  echo "FAIL: raw citation artifacts in render"; fail=1
else
  echo "OK citations clean"
fi

exit $fail
