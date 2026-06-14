# .latexmkrc — Konfigurasi latexmk untuk Disertasi Doktor ITB
# Aturan lengkap: lihat .cursor/rules/14-build-workflow.mdc  bagian B

# Engine: 4 = LuaLaTeX (preferred); ganti ke 1 untuk pdfLaTeX
$pdf_mode = 4;

# LuaLaTeX command (default engine)
$lualatex = 'lualatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# pdfLaTeX command (fallback — gunakan dengan: latexmk -pdf disertasi.tex)
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Backend bibliografi: biber (untuk biblatex)
$biber = 'biber --validate-datamodel %O %S';
$bibtex_use = 2;

# Maksimum repeat untuk konvergensi cross-reference
$max_repeat = 5;

# Ekstensi file menengah yang dihapus oleh `latexmk -c`
$clean_ext = 'synctex.gz acn acr alg aux bbl bcf blg fdb_latexmk fls glg glo gls idx ilg ind ist lof log lot out run.xml toc xdy';

# File default jika `latexmk` dijalankan tanpa argumen
@default_files = ('disertasi.tex');

# PDF viewer (optional, untuk -pvc mode)
# $pdf_previewer = 'evince';      # Linux GNOME
# $pdf_previewer = 'open -a Skim'; # macOS
# $pdf_previewer = 'sumatrapdf';   # Windows
