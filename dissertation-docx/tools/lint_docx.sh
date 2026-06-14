#!/bin/bash
# ITB prose lint for DOCX chapters - adapted from writings/disertation/scripts/lint-itb.sh.
# Text-level rule families run on pandoc-extracted text; plus docx-specific checks.
# Usage: tools/lint_docx.sh chapters/*.docx
set -u
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'; NC='\033[0m'
FATAL=0
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fatal() { echo -e "${RED}[FATAL]${NC} $1"; FATAL=1; }

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

for DOCX in "$@"; do
  base=$(basename "$DOCX" .docx)
  TXT="$TMPDIR/$base.txt"
  pandoc "$DOCX" -t plain --wrap=none -o "$TXT" 2>/dev/null || { fatal "$DOCX: pandoc extraction failed"; continue; }
  echo "=== $DOCX ==="

  # English abstract is exempt from Indonesian prose rules
  is_english=0
  case "$base" in *abstract-en*|*abstrak-en*) is_english=1 ;; esac

  # --- docx-specific -----------------------------------------------------
  hits=$(grep -nE '@@(REF|CAP|EQNUM):' "$TXT" || true)
  [ -n "$hits" ] && fatal "token @@ tersisa:
$(echo "$hits" | head -3)"

  hits=$(grep -nE 'Error: Reference source not found' "$TXT" || true)
  [ -n "$hits" ] && fatal "REF field tidak ter-resolve"

  hits=$(grep -nE '\\(cite|citetitb|citenameitb|ref|label)\{' "$TXT" || true)
  [ -n "$hits" ] && fatal "artefak perintah LaTeX mentah:
$(echo "$hits" | head -3)"

  # caption <-> mention cross-check (orphan float equivalent)
  python3 - "$TXT" <<'EOF'
import re, sys
txt = open(sys.argv[1]).read()
caps = set(re.findall(r'^(?:Gambar|Tabel) ([IVXLCDM]+\.\d+)', txt, re.M))
mentions = set(re.findall(r'(?:Gambar|Tabel)[  ]([IVXLCDM]+\.\d+)', txt))
orphan_caps = caps - mentions
if orphan_caps:
    print(f"\033[0;33m[WARN]\033[0m float tanpa rujukan dalam teks: {sorted(orphan_caps)}")
EOF

  # --- prosa Indonesia ----------------------------------------------------
  [ "$is_english" -eq 1 ] && { ok "abstrak EN: dilewati untuk aturan prosa ID"; continue; }

  hits=$(grep -inE '\b(saya|kami|kita)\b' "$TXT" || true)
  [ -n "$hits" ] && fatal "kata ganti orang pertama:
$(echo "$hits" | head -3)"

  hits=$(grep -inE '\bet[[:space:]]*al\.?' "$TXT" || true)
  [ -n "$hits" ] && fatal "'et al.' (harus 'dkk.'):
$(echo "$hits" | head -3)"

  hits=$(grep -inE '[A-Za-z][[:space:]]+&[[:space:]]+[A-Za-z]' "$TXT" || true)
  [ -n "$hits" ] && warn "'&' sebagai pengganti 'dan':
$(echo "$hits" | head -3)"

  while IFS=: read -r wrong right; do
    case "$wrong" in ''|\#*) continue ;; esac
    hits=$(grep -inE "(^|[^[:alnum:]])${wrong}([^[:alnum:]]|$)" "$TXT" || true)
    [ -n "$hits" ] && warn "ejaan tidak baku '${wrong}' (gunakan '${right}'):
$(echo "$hits" | head -2)"
  done <<'SPELLPAIRS'
obyek:objek
obyektif:objektif
analisa:analisis
sintesa:sintesis
aktifitas:aktivitas
praktek:praktik
nasehat:nasihat
resiko:risiko
frekwensi:frekuensi
sistim:sistem
jadual:jadwal
managemen:manajemen
technologi:teknologi
effektif:efektif
effisien:efisien
assesment:asesmen
azas:asas
hipotesa:hipotesis
SPELLPAIRS

  # thousands separator [0-9].[0-9]{3} is legal ITB format - exclude exact-3-digit groups
  hits=$(grep -nE '[^A-Za-z0-9.][0-9]+\.[0-9]+' "$TXT" | grep -vE '([0-9]+\.){2,}|http|arXiv|10\.[0-9]{4}|[0-9]\.[0-9]{3}([^0-9]|$)' || true)
  [ -n "$hits" ] && warn "kemungkinan desimal dengan titik (harus koma):
$(echo "$hits" | head -3)"

  hits=$(grep -inE '(,[[:space:]]*di[[:space:]]+mana\b|[.;][[:space:]]+[Dd]i[[:space:]]+mana\b)' "$TXT" || true)
  [ -n "$hits" ] && warn "'di mana' sebagai kata ganti relatif:
$(echo "$hits" | head -3)"

  hits=$(grep -nE '(^|[.!?][[:space:]]+)(Maka|Sedangkan|Sehingga)\b' "$TXT" || true)
  [ -n "$hits" ] && warn "kalimat dimulai konjungsi terlarang:
$(echo "$hits" | head -3)"

  ai_patterns=(
    'tidak dapat dipungkiri' 'memainkan peran (yang )?penting'
    '\bpenting untuk dicatat\b' '\bperlu dicatat bahwa\b'
    '\b(dalam|di) era (modern|digital|industri)\b' '\btentunya\b'
    '\bmerupakan hal yang (sangat )?(penting|krusial|vital|mendasar)\b'
    '\bsecara (komprehensif|holistik)\b' '\bperlu dipahami bahwa\b'
    '\btidak dapat dipisahkan\b' '\bkeluarga\b' '\bpada (dasarnya|intinya)\b'
    '\bdapat dikatakan bahwa\b' '\bberbagai macam\b'
    '\bperlu diperhatikan bahwa\b' '\b(sangat|amat) penting\b' '—'
  )
  for pat in "${ai_patterns[@]}"; do
    hits=$(grep -inE "$pat" "$TXT" || true)
    [ -n "$hits" ] && warn "frasa mesin /$pat/:
$(echo "$hits" | head -2)"
  done

  hits=$(grep -inE '\b(bantalan|gelinding)\b' "$TXT" || true)
  [ -n "$hits" ] && warn "istilah domain: gunakan 'bearing'/'rolling':
$(echo "$hits" | head -2)"

  ok "$base selesai diperiksa"
done

echo "============================================================"
if [ "$FATAL" -eq 0 ]; then
  echo -e "${GREEN}Lint passed (no fatal). Perhatikan WARN di atas.${NC}"; exit 0
else
  echo -e "${RED}Lint FAILED - perbaiki [FATAL] di atas.${NC}"; exit 1
fi
