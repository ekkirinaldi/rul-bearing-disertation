# Disertasi Doktor ITB — Cursor Workspace

LaTeX scaffold + Cursor rules untuk menulis Disertasi Doktor di Institut Teknologi Bandung (ITB), mengikuti **Pedoman Penulisan Disertasi Doktor ITB** (SPs-ITB, 2016) dan **Juknis Disertasi (Mei 2019)**.

Rules disusun spesifik untuk topik dissertation: *"Bearing RUL Prediction with Mamba-xLSTM Hybrid and Sparse Autoencoder Interpretability"* (lihat `PROJECT_PLAN.md`), namun struktur format-nya berlaku untuk semua disertasi ITB.

---

## Mengapa LaTeX, bukan Word?

ITB menyediakan template `.docx` resmi, tetapi:
- **Reproducibility** — LaTeX bersifat plain-text, mudah di-version-control dengan Git, dan diff-nya jelas.
- **Cursor / AI-assisted editing** — model bahasa dapat membaca, mengedit, dan men-generate kode LaTeX jauh lebih reliable daripada XML internal `.docx`.
- **Cross-reference otomatis** — `\ref{}`, `\cite{}`, `\autoref{}` dan BibTeX/Biber menangani penomoran Bab/Gambar/Tabel/Persamaan/Pustaka secara konsisten.
- **Persamaan matematika** — kualitas typesetting LaTeX jauh di atas Word.
- **Kompatibilitas format ITB** — semua aturan ITB (margin asimetris bolak-balik, font Times, spasi 1.5, dst.) diatur di `itbdisertasi.cls`.

Setelah disertasi selesai, file PDF hasil LaTeX dapat **langsung diserahkan untuk Sidang Tertutup/Terbuka** karena format visualnya identik dengan template Word resmi ITB.

---

## Struktur Folder

```
disertasi-itb/
├── .cursor/
│   └── rules/                       # Cursor rules — auto-loaded saat coding
│       ├── 00-project-meta.mdc
│       ├── 01-itb-format-essentials.mdc      (alwaysApply)
│       ├── 02-latex-conventions.mdc           (*.tex, *.cls)
│       ├── 03-bahasa-indonesia-baku.mdc       (*.tex)
│       ├── 04-citations-bibliography.mdc      (*.tex, *.bib)
│       ├── 05-figures-tables-equations.mdc    (*.tex)
│       ├── 06-bab1-pendahuluan.mdc            (chapters/01*.tex)
│       ├── 07-bab2-tinjauan-pustaka.mdc
│       ├── 08-bab3-dasar-teori.mdc
│       ├── 09-bab4-metodologi.mdc
│       ├── 10-bab5-hasil-pembahasan.mdc
│       ├── 11-bab6-kesimpulan.mdc
│       ├── 12-abstrak-front-matter.mdc
│       ├── 13-novelty-and-quality.mdc
│       ├── 14-build-workflow.mdc
│       └── 15-domain-rul-bearings.mdc
├── chapters/                        # Per-bab content
│   ├── 00-bagian-persiapan.tex      # halaman pengesahan, pedoman, kata pengantar
│   ├── 00-abstrak-id.tex
│   ├── 00-abstract-en.tex
│   ├── 01-pendahuluan.tex           # Bab I
│   ├── 02-tinjauan-pustaka.tex      # Bab II
│   ├── 03-dasar-teori.tex           # Bab III
│   ├── 04-metodologi.tex            # Bab IV
│   ├── 05-hasil-pembahasan.tex      # Bab V
│   └── 06-kesimpulan.tex            # Bab VI
├── lampiran/                        # Lampiran A, B, C, ...
├── figures/                         # Semua .pdf/.png gambar
├── itbdisertasi.cls                 # Custom class — enforces ITB format
├── disertasi.tex                    # Master file — \input semua bab
├── references.bib                   # BibTeX, format ITB-friendly
├── Makefile                         # latexmk + utilities
└── PROJECT_PLAN.md                  # Plan riset (bearings/RUL)
```

---

## Cara Pakai dengan Cursor

1. **Buka folder `disertasi-itb/` sebagai workspace** di Cursor.
2. Cursor akan otomatis memuat semua `.mdc` di `.cursor/rules/`.
3. Beberapa rule ber-`alwaysApply: true` (selalu aktif), sebagian lain ber-`globs` (aktif saat file pattern cocok), sisanya bertipe `Agent Requested` (Cursor memutuskan kapan memanggil).
4. Saat meminta Cursor menulis Bab IV, misalnya, dia akan secara otomatis menerapkan rule `09-bab4-metodologi.mdc` plus rule global format ITB.

### Contoh prompt yang bagus untuk Cursor

> "Tulis subbab IV.2 tentang arsitektur Mamba-xLSTM-Net. Gunakan persamaan untuk forward pass mLSTM dan Mamba selective scan. Sertakan diagram blok sebagai `figure`."

Cursor akan otomatis:
- Menulis dalam Bahasa Indonesia Baku tanpa kata ganti orang pertama (rule `03`)
- Memberi nomor persamaan `(IV.x)` (rule `05`)
- Menyisipkan `\label{eq:...}` dan `\label{fig:...}` untuk cross-reference
- Mematuhi struktur "minimal 1 paragraf antara judul anak-bab dan sub-anak-bab" (rule `01`)

### Contoh prompt yang bagus untuk Bibliography

> "Tambahkan referensi: Beck et al. 2024, xLSTM, NeurIPS, ke `references.bib` dan sitasi di paragraf terakhir III.2."

Cursor akan otomatis:
- Menambah entri BibTeX dengan kunci `beck2024xlstm`
- Memakai `\citetitb{beck2024xlstm}` yang menghasilkan format ITB: `(Beck dkk., 2024)`

---

## Build

```bash
make            # compile penuh (latexmk -pdf)
make watch      # auto-recompile saat file berubah
make clean      # hapus aux/log/bbl/dst.
make wordcount  # hitung kata di body (tidak termasuk preamble/bib)
make check      # validator: cari kesalahan format ITB umum
```

---

## Kepatuhan terhadap Pedoman ITB

`itbdisertasi.cls` mengimplementasikan poin-poin wajib berikut (lihat `01-itb-format-essentials.mdc` untuk daftar lengkap):

| Aspek | Aturan ITB | Implementasi |
|---|---|---|
| Font tubuh utama | Times New Roman 12 pt | `\usepackage{newtxtext,newtxmath}` |
| Spasi tubuh utama | 1,5 | `\linespread{1.5}` |
| Spasi caption/footnote/pustaka | 1 | `\setstretch{1}` lokal |
| Margin halaman ganjil | kiri 4 cm, lainnya 3 cm | `geometry` + `twoside` |
| Margin halaman genap | kanan 4 cm, lainnya 3 cm | mirror margins |
| Penomoran bab | Romawi (Bab I, II, ...) | `\renewcommand{\thechapter}{\Roman{chapter}}` |
| Penomoran subbab | I.1, I.2, ... | `titlesec` config |
| Penomoran persamaan | (I.1), (I.2), ... | `\numberwithin{equation}{chapter}` |
| Caption gambar/tabel | "Gambar I.1", "Tabel I.1" | `caption` package + ID labels |
| Penomoran halaman persiapan | i, ii, iii, ... | `\pagenumbering{roman}` |
| Penomoran halaman tubuh | 1, 2, 3, ... | `\pagenumbering{arabic}` |
| Bibliografi | abjad, hanging 1.27 cm, 1 spasi | custom bibstyle |
| Sitasi `dkk.` (>2 penulis) | wajib bukan `et al.` | bibstyle Bahasa Indonesia |

---

## Catatan untuk Promotor / Tim Pembimbing

PDF output telah diverifikasi visual-match terhadap template `.docx` resmi ITB. Apabila pembimbing meminta versi `.docx`, gunakan:

```bash
make docx       # konversi via Pandoc — best-effort, periksa manual
```

Namun untuk submission akhir, **kirim PDF yang dihasilkan LaTeX**.
