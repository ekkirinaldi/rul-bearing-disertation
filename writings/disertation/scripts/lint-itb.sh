#!/usr/bin/env bash
# scripts/lint-itb.sh
# Pemeriksaan otomatis kepatuhan format ITB.
# Exit code != 0 jika ada pelanggaran fatal.
#
# Aturan dirinci di .cursor/rules/14-build-workflow.mdc  bagian D

set -uo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

CHAPTERS="chapters/*.tex"
LAMPIRAN="lampiran/*.tex"
ALL_BODY="$CHAPTERS $LAMPIRAN"
TEX_MAIN="disertasi.tex"
FATAL=0

ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fatal() { echo -e "${RED}[FATAL]${NC} $1"; FATAL=1; }

echo "============================================================"
echo "  ITB Disertasi — Format Lint"
echo "============================================================"

# --- 1. First-person pronouns (DILARANG di body Indonesia) -------------
echo ""
echo "--- 1. First-person pronouns (saya/kami/kita) ---"
hits=$(grep -inE '\b(saya|kami|kita)\b' $ALL_BODY 2>/dev/null \
  | grep -vE ':[[:space:]]*%' \
  | grep -vE 'kata-pengantar' || true)
if [ -n "$hits" ]; then
  fatal "Ditemukan kata ganti orang pertama di body (gunakan bentuk pasif):"
  echo "$hits"
else
  ok "Tidak ada kata ganti orang pertama di body."
fi

# --- 2. "et al." (harus "dkk." pada body Indonesia) --------------------
echo ""
echo "--- 2. Penggunaan 'et al.' ---"
hits=$(grep -inE '\bet[[:space:]]*al\.?' $ALL_BODY 2>/dev/null || true)
if [ -n "$hits" ]; then
  fatal "Ditemukan 'et al.' di body. Gunakan 'dkk.' (auto via biblatex localization):"
  echo "$hits"
else
  ok "Tidak ada 'et al.' di body."
fi

# --- 3. Ampersand "&" sebagai pengganti "dan" --------------------------
echo ""
echo "--- 3. '&' sebagai konjungsi ---"
hits=$(grep -inE '[A-Za-z][[:space:]]*&[[:space:]]*[A-Za-z]' $ALL_BODY 2>/dev/null \
  | grep -vE 'tabular|align|matrix|cases|begin\{|end\{|\\&|multicolumn|hline' \
  | grep -vE '%.*&' || true)
if [ -n "$hits" ]; then
  warn "Ditemukan '&' di luar konteks tabular/math. Cek apakah seharusnya 'dan':"
  echo "$hits"
else
  ok "Tidak ada '&' yang mencurigakan."
fi

# --- 4. Spelling baku ---------------------------------------------------
echo ""
echo "--- 4. Spelling baku KBBI ---"
# Portabel (tanpa associative array / bash 4+): cocok untuk bash 3.2 macOS.
spell_warn=0
while IFS=: read -r wrong right; do
  case "$wrong" in ''|\#*) continue ;; esac
  hits=$(grep -inE "(^|[^[:alnum:]])${wrong}([^[:alnum:]]|$)" $ALL_BODY 2>/dev/null || true)
  if [ -n "$hits" ]; then
    warn "Ejaan tidak baku '${wrong}' (gunakan '${right}'):"
    echo "$hits"
    spell_warn=1
  fi
done <<'SPELLPAIRS'
obyek:objek
obyektif:objektif
analisa:analisis
sintesa:sintesis
aktifitas:aktivitas
aktip:aktif
praktek:praktik
nasehat:nasihat
resiko:risiko
atlit:atlet
frekwensi:frekuensi
sistim:sistem
jadual:jadwal
managemen:manajemen
managament:manajemen
technologi:teknologi
effektif:efektif
effisien:efisien
assesment:asesmen
azas:asas
hipotesa:hipotesis
SPELLPAIRS
[ "$spell_warn" -eq 0 ] && ok "Tidak ada ejaan tidak baku yang dideteksi."

# --- 5. Decimal separator -----------------------------------------------
echo ""
echo "--- 5. Pemisah desimal (Indonesia: koma, bukan titik) ---"
# Heuristik: angka.angka di luar konteks ref/label/cite/numbered list.
hits=$(grep -inE '[^A-Za-z][0-9]+\.[0-9]+' $ALL_BODY 2>/dev/null \
  | grep -vE '\\(ref|cite|citetitb|citenameitb|parencite|textcite|label|input|include|include|usepackage|documentclass)' \
  | grep -vE '\\(eq|fig|tab|sec|subsec|bab|lamp|alg|chap):' \
  | grep -vE 'chapter|section|figure|table' \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*%' \
  | grep -vE 'v[0-9]+\.[0-9]+|version|dataset|XJTU|PHM' \
  | head -30 || true)
if [ -n "$hits" ]; then
  count=$(echo "$hits" | wc -l)
  if [ "$count" -gt 0 ]; then
    warn "Mungkin ada angka desimal dengan titik (Indonesia gunakan koma — atau pakai \\num{...}):"
    echo "$hits"
  fi
else
  ok "Tidak ada angka desimal yang mencurigakan."
fi

# --- 6. Float yatim (figure/table/algorithm tanpa \ref) ----------------
echo ""
echo "--- 6. Float yatim (label fig:/tab:/alg: tanpa \\ref) ---"
orphan=0
for f in $ALL_BODY; do
  [ -f "$f" ] || continue
  labels=$(grep -oE '\\label\{(fig|tab|alg):[^}]+\}' "$f" 2>/dev/null \
    | sed -E 's/\\label\{(.*)\}/\1/' || true)
  for lbl in $labels; do
    if ! grep -qE "\\\\(ref|autoref|cref|Cref)\{$lbl\}" $ALL_BODY $TEX_MAIN 2>/dev/null; then
      warn "Float yatim (tidak dirujuk \\ref): $lbl di $f"
      orphan=1
    fi
  done
done
[ "$orphan" -eq 0 ] && ok "Semua float dirujuk dengan \\ref/\\cref."

# --- 7. Sitasi yang belum ada di .bib ----------------------------------
echo ""
echo "--- 7. Citation keys vs references.bib ---"
if [ -f references.bib ]; then
  cited=$(grep -ohE '\\(cite|citetitb|citenameitb|parencite|textcite|citet|citep)\{[^}]+\}' \
            $ALL_BODY disertasi.tex 2>/dev/null \
          | sed -E 's/.*\{([^}]+)\}/\1/' \
          | tr ',' '\n' \
          | tr -d ' ' \
          | sort -u)
  defined=$(grep -E '^@[a-zA-Z]+\{' references.bib \
            | sed -E 's/^@[a-zA-Z]+\{([^,]+),.*/\1/' \
            | sort -u)
  if [ -n "$cited" ]; then
    missing=$(comm -23 <(echo "$cited") <(echo "$defined"))
    if [ -n "$missing" ]; then
      fatal "Sitasi merujuk key yang tidak ada di references.bib:"
      echo "$missing"
    else
      ok "Semua sitasi terdefinisi di references.bib."
    fi
  else
    warn "Belum ada sitasi di body. Tambahkan referensi ke literatur."
  fi
else
  warn "references.bib tidak ditemukan."
fi

# --- 8. Cross-reference yang rusak (\ref dengan key tidak ada) ---------
echo ""
echo "--- 8. Cross-reference rusak ---"
ref_keys=$(grep -ohE '\\(ref|autoref|cref|Cref)\{[^}]+\}' $ALL_BODY disertasi.tex 2>/dev/null \
           | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u)
label_keys=$(grep -ohE '\\label\{[^}]+\}' $ALL_BODY disertasi.tex 2>/dev/null \
             | sed -E 's/\\label\{([^}]+)\}/\1/' | sort -u)
broken=$(comm -23 <(echo "$ref_keys") <(echo "$label_keys"))
if [ -n "$broken" ]; then
  fatal "Cross-reference merujuk label yang tidak ada:"
  echo "$broken"
else
  ok "Semua \\ref merujuk label yang ada."
fi

# --- 9. TODO / FIXME / placeholder \dots di body -----------------------
echo ""
echo "--- 9. TODO / FIXME / placeholder ---"
hits=$(grep -inE '(TODO|FIXME|XXX|\\dots\b)' $ALL_BODY 2>/dev/null \
  | grep -vE ':[[:space:]]*%' || true)
if [ -n "$hits" ]; then
  count=$(echo "$hits" | wc -l)
  warn "Ditemukan $count TODO/FIXME/\\dots — tidak fatal, tetapi pastikan dihapus sebelum sidang:"
  echo "$hits" | head -10
else
  ok "Tidak ada TODO/FIXME/\\dots di body."
fi

# --- 10. Caption gambar / tabel kosong ---------------------------------
echo ""
echo "--- 10. \\caption{} kosong ---"
hits=$(grep -inE '\\caption\{[[:space:]]*\}' $ALL_BODY 2>/dev/null || true)
if [ -n "$hits" ]; then
  fatal "Ditemukan \\caption kosong:"
  echo "$hits"
else
  ok "Semua \\caption terisi."
fi

# --- 11. "di mana" sebagai kata ganti relatif --------------------------
echo ""
echo "--- 11. 'di mana' sebagai kata ganti relatif ---"
hits=$(grep -inE '(,\s*di\s+mana\b|[.;]\s+[Dd]i\s+mana\b)' $ALL_BODY 2>/dev/null \
  | grep -vE ':[[:space:]]*%' || true)
if [ -n "$hits" ]; then
  warn "Ditemukan pola 'di mana' — kemungkinan kata ganti relatif (calque 'where'). Ganti dengan 'yang', 'tempat', 'pada saat', atau restrukturisasi:"
  echo "$hits"
else
  ok "Tidak ada pola 'di mana' yang mencurigakan."
fi

# --- 12. Kalimat dimulai dengan konjungsi terlarang --------------------
echo ""
echo "--- 12. Kalimat dimulai dengan Maka/Sedangkan/Sehingga ---"
hits=$(grep -inE '^\s*(Maka|Sedangkan|Sehingga)\b' $ALL_BODY 2>/dev/null \
  | grep -vE ':[[:space:]]*%' || true)
if [ -n "$hits" ]; then
  warn "Kalimat dimulai dengan konjungsi terlarang (Maka/Sedangkan/Sehingga). Restrukturisasi kalimat:"
  echo "$hits"
else
  ok "Tidak ada kalimat yang dimulai dengan Maka/Sedangkan/Sehingga."
fi

# --- 13. Heading berakhir dengan titik ---------------------------------
echo ""
echo "--- 13. Heading \\section/\\subsection berakhir dengan titik ---"
hits=$(grep -inE '\\(sub)*section\*?\{[^}]*\.\s*\}' $ALL_BODY 2>/dev/null \
  | grep -vE ':[[:space:]]*%' || true)
if [ -n "$hits" ]; then
  warn "Judul subbab berakhir dengan titik (pedoman: judul bukan kalimat — hapus titiknya):"
  echo "$hits"
else
  ok "Tidak ada heading yang berakhir dengan titik."
fi

# --- 14. \footnote di bab body -----------------------------------------
echo ""
echo "--- 14. \\footnote di bab body ---"
hits=$(grep -inE '\\footnote\{' $ALL_BODY 2>/dev/null \
  | grep -vE ':[[:space:]]*%' \
  | grep -vE 'kata-pengantar' || true)
if [ -n "$hits" ]; then
  warn "Ditemukan \\footnote di body. Referensi harus inline via \\citetitb{}, bukan catatan kaki:"
  echo "$hits"
else
  ok "Tidak ada \\footnote di body."
fi

# --- 15. Sitasi (\cite) di dalam teks abstrak --------------------------
echo ""
echo "--- 15. \\cite di teks abstrak ---"
ABSTRACTS=""
[ -f "chapters/00-abstrak-id.tex" ]  && ABSTRACTS="$ABSTRACTS chapters/00-abstrak-id.tex"
[ -f "chapters/00-abstract-en.tex" ] && ABSTRACTS="$ABSTRACTS chapters/00-abstract-en.tex"
if [ -n "$ABSTRACTS" ]; then
  hits=$(grep -inE '\\(cite|citetitb|citenameitb|parencite|textcite)\{' $ABSTRACTS 2>/dev/null \
    | grep -vE ':[[:space:]]*%' || true)
  if [ -n "$hits" ]; then
    fatal "Ditemukan sitasi di abstrak. Pedoman ITB  bagian II.2: abstrak tidak boleh memuat rujukan literatur:"
    echo "$hits"
  else
    ok "Tidak ada sitasi di teks abstrak."
  fi
else
  warn "File abstrak (00-abstrak-id.tex / 00-abstract-en.tex) tidak ditemukan — lewati cek #15."
fi

# --- 16. Penanda frasa mesin (AI-marker) -----------------------------------
echo ""
echo "--- 16. Penanda frasa mesin (AI-marker) ---"
ai_warn=0
_check_ai() {
  local pattern="$1" label="$2"
  hits=$(grep -inE "$pattern" $ALL_BODY 2>/dev/null | grep -vE ':[[:space:]]*%' || true)
  if [ -n "$hits" ]; then
    warn "Frasa mesin '$label' — restrukturisasi agar lebih organik:"
    echo "$hits" | head -3
    ai_warn=1
  fi
}
_check_ai 'tidak dapat dipungkiri'                          'tidak dapat dipungkiri'
_check_ai 'memainkan peran (yang )?penting'                 'memainkan peran penting'
_check_ai '\bpenting untuk dicatat\b'                       'penting untuk dicatat'
_check_ai '\bperlu dicatat bahwa\b'                         'perlu dicatat bahwa'
_check_ai '\b(dalam|di) era (modern|digital|industri)\b'   'dalam/di era modern/digital/industri'
_check_ai '\btentunya\b'                                    'tentunya (informal/klise mesin)'
_check_ai '\bmerupakan hal yang (sangat )?(penting|krusial|vital|mendasar)\b' \
                                                            'merupakan hal yang penting/krusial'
_check_ai '\bsecara (komprehensif|holistik)\b'              'secara komprehensif/holistik'
_check_ai '\bperlu dipahami bahwa\b'                        'perlu dipahami bahwa'
_check_ai '\btidak dapat dipisahkan\b'                      'tidak dapat dipisahkan (klise)'
_check_ai '\bkeluarga\b'  'keluarga (untuk algoritma/model — gunakan: jenis/varian/kelompok/kelas/model)'
_check_ai '\bpada (dasarnya|intinya)\b'                     'pada dasarnya/intinya (filler mesin)'
_check_ai '\bdapat dikatakan bahwa\b'                       'dapat dikatakan bahwa (hedging klise)'
_check_ai '\bberbagai macam\b'                              'berbagai macam (kabur — sebutkan jumlah konkret)'
_check_ai '\bperlu diperhatikan bahwa\b'                    'perlu diperhatikan bahwa'
_check_ai '\b(sangat|amat) penting\b'                       'sangat/amat penting (vague intensifier)'
# Pembuka kalimat "Lebih lanjut,"
hits=$(grep -inE '^\s*(Lebih lanjut)[,.]' $ALL_BODY 2>/dev/null | grep -vE ':[[:space:]]*%' || true)
if [ -n "$hits" ]; then
  warn "Frasa mesin 'Lebih lanjut,' sebagai pembuka — gunakan 'Selain itu,' atau integrasikan:"
  echo "$hits" | head -3
  ai_warn=1
fi
# Em-dash: Unicode — atau LaTeX --- di luar konteks tabel/komentar
em_hits=$(grep -inE '(—|---[^-])' $ALL_BODY 2>/dev/null \
  | grep -vE ':[[:space:]]*%' \
  | grep -vE '(\\hline|\\toprule|\\midrule|\\bottomrule|\\cline|\\rule\b)' \
  || true)
if [ -n "$em_hits" ]; then
  warn "Em dash (— atau ---) di prosa — ganti dengan koma, titik koma, titik dua, atau pisah kalimat:"
  echo "$em_hits" | head -3
  ai_warn=1
fi
[ "$ai_warn" -eq 0 ] && ok "Tidak ada frasa mesin yang terdeteksi."

# --- 17. Istilah teknis domain bearing ------------------------------------
echo ""
echo "--- 17. Istilah teknis domain bearing ---"
term_warn=0
_check_term() {
  local pattern="$1" wrong="$2" right="$3"
  hits=$(grep -inE "$pattern" $ALL_BODY 2>/dev/null | grep -vE ':[[:space:]]*%' || true)
  if [ -n "$hits" ]; then
    warn "Istilah tidak baku '$wrong' — gunakan '$right' (terminologi teknis baku):"
    echo "$hits" | head -3
    term_warn=1
  fi
}
_check_term '\bbantalan\b'  'bantalan'  'bearing'
_check_term '\bgelinding\b' 'gelinding' 'rolling'
[ "$term_warn" -eq 0 ] && ok "Istilah teknis domain sudah konsisten."

# --- Summary -----------------------------------------------------------
echo ""
echo "============================================================"
if [ "$FATAL" -eq 0 ]; then
  echo -e "${GREEN}  Lint passed (no fatal). Perhatikan WARN di atas.${NC}"
  echo "============================================================"
  exit 0
else
  echo -e "${RED}  Lint FAILED — perbaiki [FATAL] di atas sebelum lanjut.${NC}"
  echo "============================================================"
  exit 1
fi
