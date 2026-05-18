# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Doctoral dissertation by **Toto Suharto** at **ITB (Institut Teknologi Bandung)**, Program Doktor Teknik dan Manajemen Industri, FTI.

**Working title:** *Perawatan Prediktif untuk Sistem Produksi dengan Pendekatan Analisis Big Data dan Kecerdasan Buatan Menggunakan Data Kondisi Mesin dan Informasi Kualitas yang Real Time*

**Repository scope:** the **LaTeX dissertation manuscript** and supporting writings. The Python training code ([Mamba-xLSTM/](Mamba-xLSTM/)) and datasets ([data-bearing/](data-bearing/)) live locally only — excluded via `.gitignore`.

## Main Goal — Modify and Finalize the Dissertation

The primary purpose of this codebase is to **finalize Pak Toto's doctoral dissertation following the plan in [writings/dissertation-outline.md](writings/dissertation-outline.md)** — this is the **single source of truth** for chapter structure, section content, figures, tables, and lampiran.

The outline encodes a two-track narrative:

- **Jalur A — Diagnostik** (Bab IV): CWRU → benchmark kernel/tree/deep → WDCNN + SHAP DeepExplainer + Fault Signature Maps. Sources: [Paper/Conference1_*.pdf](Paper/), [Paper/Conference2_*.pdf](Paper/), [Paper/Journal1_*.pdf](Paper/).
- **Jalur B — Prognostik** (Bab V): PHM2012/XJTU-SY/IMS → 3 backbone RUL (Mamba-xLSTM, N-BEATS-xLSTM, SparseGate-TCN) → Top-*k* Sparse Autoencoder → BPFx mapping. Source: [Paper/Journal2_RUL_Journal.pdf](Paper/).
- **Konvergensi** (Bab VI): kerangka PdM multi-tier (Edge IoT → Edge Server → Cloud/GPU) + interpretabilitas dua-lapis (input attribution ↔ latent concept).

**Implementation phases** (from outline §Catatan Eksekusi):

1. **Fase A** — Restrukturisasi [01-pendahuluan.tex](writings/disertation/chapters/01-pendahuluan.tex) selaras SK-Toto §I (RQ, Tujuan, Manfaat, Batasan, Kontribusi).
2. **Fase B** — Split [03-dasar-teori.tex](writings/disertation/chapters/03-dasar-teori.tex) + [04-metodologi.tex](writings/disertation/chapters/04-metodologi.tex) menjadi Bab III shared / Bab IV diagnostik / Bab V prognostik.
3. **Fase C** — Restrukturisasi [05-hasil-pembahasan.tex](writings/disertation/chapters/05-hasil-pembahasan.tex) (kernel+tree+deep results di Bab IV; RUL+SAE results di Bab V).
4. **Fase D** — Tambah Lampiran E (SVM/LR), F (Tree), G (WDCNN+FSM); lembutkan [Lampiran D](writings/disertation/lampiran/D-klasifikasi-industri.tex) (PT~SKF sebagai validasi eksternal terbatas).
5. **Fase E** — Restrukturisasi [abstrak ID](writings/disertation/chapters/00-abstrak-id.tex) + [EN](writings/disertation/chapters/00-abstract-en.tex) + [daftar singkatan](writings/disertation/chapters/00-daftar-singkatan.tex).
6. **Fase F** — Update [06-kesimpulan.tex](writings/disertation/chapters/06-kesimpulan.tex) per VI.1–VI.5.

**Working rules for the dissertation:**

- Always re-read [dissertation-outline.md](writings/dissertation-outline.md) before editing a chapter — section structure may have evolved.
- Small commits per section so Pak Toto can review incrementally.
- After every section edit, run `make build` and `make check` from [writings/disertation/](writings/disertation/) to catch lint and compile errors early.
- Before claiming a phase is done, run `make pre-submit`.
- Outstanding TODOs (track in commits / PR descriptions): verify venue+year for the 4 self-citations (`TotoSuharto20XXKey`); confirm PT~SKF data volume + ground-truth; flag missing Journal 2 notebook.

## Repository Layout

```
writings/
  dissertation-outline.md       # ★ Single source of truth for chapter structure
  disertation/                  # LaTeX manuscript (main tracked content)
  Outline_Disertasi_6Bab.pdf    # Pak Toto's 6-bab structural target
  SK-Toto.pdf                   # Proposal disertasi (Agustus 2025) — Bab I source
  bab5-draft-hasil-performa.md
  experiment-design.md / experiment-report.{html,pdf}
  research-design-chart.md
Paper/                          # 4 self-papers (Conf 1, Conf 2, Journal 1, Journal 2)
Notebook/                       # 3 reproducibility notebooks (Conf 1, Conf 2, Journal 1)
README.md
CLAUDE.md
new_algorithm.md                # Algorithm brainstorming notes
```

## Dissertation Build

All commands run from [writings/disertation/](writings/disertation/):

```bash
make build        # LuaLaTeX + Biber compile → build/disertasi.pdf + disertasi.pdf
make watch        # Live recompile on file changes (latexmk -pvc)
make clean        # Remove intermediate build files, keep PDF
make distclean    # Remove build/ directory and PDF entirely
make check        # chktex + regex lint (calls make lint)
make lint         # Regex checks via scripts/lint-itb.sh (enforces ITB rules below)
make spell        # Indonesian spell check via hunspell -d id_ID
make wordcount    # Word count per chapter via texcount
make pre-submit   # clean + build + wordcount + check (run before sending to promotor)
```

Docker fallback (no local TeX install):

```bash
cd writings/disertation
docker run --rm -v "$(pwd):/workdir" -w /workdir \
  danteev/texlive \
  latexmk -outdir=build -interaction=nonstopmode disertasi.tex
```

Engine: **LuaLaTeX** (preferred) or pdfLaTeX. Bibliography backend: **Biber**.

## LaTeX Source Layout

[writings/disertation/](writings/disertation/)

- [disertasi.tex](writings/disertation/disertasi.tex) — Master file; `\input` all chapters + front matter; sets `\title`, `\author`, `\nim`, `\prodi`, `\promotor`, `\bulan`, `\tahun`.
- [itbdisertasi.cls](writings/disertation/itbdisertasi.cls) — Custom ITB class (Times Roman 12pt, 1.5-spacing, A4, mirror margins per ITB spec).
- [itbdisertasi-layout.tex](writings/disertation/itbdisertasi-layout.tex) — Margin/layout parameters extracted from the class.
- [chapters/](writings/disertation/chapters/):
  - `00-abstrak-id.tex`, `00-abstract-en.tex` — Bilingual abstract (500–800 words each).
  - `00-kata-pengantar.tex`, `00-daftar-singkatan.tex` — Front matter.
  - `01-pendahuluan.tex` … `06-kesimpulan.tex` — Bab I–VI.
- [lampiran/](writings/disertation/lampiran/) `A`–`D` (and planned `E`–`G` per outline).
- [figures/](writings/disertation/figures/) organized per chapter (`bab1/` … `bab6/`, `v5/`).
- [references.bib](writings/disertation/references.bib) — BibLaTeX bibliography, Indonesian style.
- [scripts/lint-itb.sh](writings/disertation/scripts/lint-itb.sh) — automated ITB format lint (called by `make lint`).

Manuscript language: **Bahasa Indonesia** (only the English abstract is in English).

---

## ITB Doctoral Dissertation Writing Rules

These rules come from the official *Pedoman Penulisan Disertasi Doktor ITB* (SPs, April 2016) and must be followed for every paragraph, table, equation, citation, and lampiran written into the manuscript. Many are auto-enforced by [scripts/lint-itb.sh](writings/disertation/scripts/lint-itb.sh) — **fix lint violations before committing**.

### Language (Pedoman §III.1, §III.2)

- **Bahasa Indonesia Baku.** Follow KBBI, EYD (PUEBI), and Pedoman Umum Pembentukan Istilah. Use complete, well-punctuated sentences.
- **No first-person pronouns** (`saya`, `kami`, `kita`) anywhere in the body — restructure to passive voice. Allowed only in `00-kata-pengantar.tex`. *Lint-enforced.*
- **No `di mana`** as a relative pronoun (a `where`-calque). Use `yang`, `tempat`, `pada saat`, or restructure.
- **Do not start sentences with conjunctions** `maka`, `sedangkan`, `sehingga`. Restructure.
- **Do not start sentences with numerals or symbols.** Spell out (`Sepuluh model …`) or restructure.
- **No `&` for `dan`.** Reserve `&` for math/tabular alignment only. *Lint-enforced.*
- **Avoid foreign terms** when an established Indonesian term exists. When a foreign term is necessary, italicize consistently using `\emph{...}` (e.g., `\emph{deep learning}`, `\emph{Sparse Autoencoder}`, `\emph{envelope spectrum}`). Genus/species names are always italic (e.g., `\emph{Sonchus arvensis}`).
- **No foreign connecting words.** Indonesian text must never use foreign linker/preposition shorthands like `vs`, `via`, `etc.`, `i.e.`, `e.g.`, `cf.` as syntactic glue. Use Indonesian equivalents: `vs` → `dan` / `terhadap` / `dengan` / `dibandingkan dengan` (per context); `via` → `melalui`; `etc.` → `dll.`; `i.e.` → `yaitu`; `e.g.` → `misalnya`; `cf.` → `bandingkan dengan`. Italicized technical terms that *contain* such tokens (e.g., `\emph{one-vs-rest}`, scikit-learn `multi_class='ovr'`) are exempt — they are names, not connectors.
- **Prefer `multi-model` over `multi-keluarga`** when describing benchmarks that span kernel/tree/deep families. Pak Toto's convention: `model` is the level of comparison; `keluarga` is reserved for in-prose elaboration ("keluarga kernel", "keluarga tree") and never compounded into `multi-keluarga`.
- **One main idea per paragraph.** Never write a single-sentence paragraph.
- **Spelling — baku KBBI** (lint-enforced sample): `objek` (not `obyek`), `analisis` (not `analisa`), `sintesis` (not `sintesa`), `aktivitas` (not `aktifitas`), `praktik` (not `praktek`), `nasihat` (not `nasehat`), `risiko` (not `resiko`), `frekuensi` (not `frekwensi`), `sistem` (not `sistim`), `jadwal` (not `jadual`), `manajemen` (not `managemen`), `teknologi` (not `technologi`), `efektif` (not `effektif`), `efisien` (not `effisien`), `asesmen` (not `assesment`), `asas` (not `azas`), `hipotesis` (not `hipotesa`).

### Numbers and Units (Pedoman §VIII.3)

- **Decimal separator: comma** (`25,5`). Never use a period. *Lint heuristically enforced.* In LaTeX, prefer `\num{25.5}` from `siunitx` (it renders as `25,5` with the right locale).
- **Thousands separator: period** (`1.000.000`). To avoid ambiguity with the decimal comma, avoid 3-digit decimals — prefer `25,24` or `25,2472`, not `25,247`.
- **Numbers < 10 written out** (`enam perguruan tinggi`); **≥ 10 use digits** (`17 mangga`).
- **Vague/round quantities in words** (`sepuluh tahun yang lalu`, `lima kali sehari`).
- **Avoid Roman numerals for ordinary numbers** (Roman is reserved for chapter/equation numbers).
- **Scientific notation:** `1,91 × 10^6` or `1,91E6` for `1.908.176`.
- **SI units:** abbreviations after numerals (`5 kg`), full word when used as a noun (`Massa diukur dalam kilogram`).

### Citations and Bibliography (Pedoman §VI; §VIII.7)

- **No footnotes for references.** Cite inline; weave any auxiliary remark into the sentence.
- **In-text format:** `(Surname, year)` parenthetical, or `Surname (year)` narrative.
- **In-text max 2 authors.** For ≥ 3, use `Surname-pertama dkk. (tahun)` — **`dkk.`, never `et al.`**. *Lint-enforced.* (`biblatex` localization handles this — use `\citetitb{...}` macro.)
- **In `references.bib`: list ALL authors.** Use `dan` (not `and` or `&`) before the last author. The `dkk.` shortening only applies in body text, not in the bibliography entry.
- **Style:** `Surname, Initial. (year): Title in sentence case, *Journal Name in italic*, **volume in bold**, start–end pages.` Example:
  > Cotton, F.A. (1998): Kinetics of gasification of brown coal, *Journal of American Chemical Society*, **54**, 38–43.
- **Sentence case for paper titles** (capital only on the first word + proper nouns).
- **Journal name italic, volume bold.** Use an en-dash (`–`) for page ranges, not a hyphen.
- **Hanging indent 1,27 cm** (7 ketukan) for each entry; single-spacing within and between entries.
- **Alphabetical by first author's family name; no numbering.**
- **Allowed source types:** journal/proceeding articles, books, theses/disertasi, websites (cite per discipline norm). Newspapers/TV/film are allowed **only when the artifact is itself the research object** — never for general background.
- **Every entry in `references.bib` MUST be cited in the body**, and vice versa. *Lint-enforced for missing keys + broken cross-refs.*
- **Citation macros in this project** (defined in [itbdisertasi.cls](writings/disertation/itbdisertasi.cls)): `\citetitb{key}`, `\citenameitb{key}`. Use these rather than raw `\cite{}`/`\citep{}` to keep formatting consistent.

#### Required Additions to `references.bib`

The four self-papers in [Paper/](Paper/) are the **primary empirical sources** for Bab IV and Bab V and must be added to `references.bib` before those chapters are written. Skeleton entries already exist at the bottom of `references.bib` tagged with `TODO(verify-with-pembimbing)` — confirm venue, year, volume, and DOI with Pak Toto before finalising.

| BibTeX key | File in `Paper/` | Cite in | Notes |
|---|---|---|---|
| `TotoSuharto2024Conf1SVM` | [Conference1_Classification_SVM_LR.pdf](Paper/Conference1_Classification_SVM_LR.pdf) | Bab IV §IV.2; Bab II §II.2.3 | SVM/LR + SHAP KernelExplainer on CWRU; 8 authors |
| `TotoSuharto2024Conf2Tree` | [Conference2_Classification_Tree.docx](Paper/Conference2_Classification_Tree.docx) | Bab IV §IV.3; Bab II §II.2.3 | DT/RF/XGBoost + SHAP TreeExplainer on CWRU |
| `TotoSuharto2025Journal1FSM` | [Journal1_Fault Signature Maps.docx](Paper/Journal1_Fault%20Signature%20Maps.docx) | Bab IV §IV.4–§IV.13; Bab II §II.3.3 | WDCNN + SHAP DeepExplainer + FSM; key result akurasi 99,87% |
| `TotoSuharto2025Journal2RUL` | [Journal2_RUL_Journal.pdf](Paper/Journal2_RUL_Journal.pdf) | Bab V §V.1–§V.9 throughout | Mamba-xLSTM + SAE-BPFx; top-k SAE mechanistic interpretability |

Additional entries required by the dissertation outline that are not yet in `references.bib`:

| BibTeX key | What it is | Cite in |
|---|---|---|
| `Zhang2017WDCNN` | Zhang et al. (2017) — original WDCNN (*Sensors* 17, 425) | Bab IV §IV.4 |
| `Shrikumar2017DeepLIFT` | Shrikumar et al. (2017) — DeepLIFT (ICML 2017) | Bab IV §IV.5; Bab II §II.3 |
| `Elhage2022Superposition` | Elhage et al. (2022) — Toy Models of Superposition (Anthropic) | Bab II §II.5; Bab V §V.2 |
| `Cunningham2023TopkSAE` | Cunningham et al. (2023) — Top-*k* SAE (*arXiv* 2310.17230) | Bab V §V.2; Bab II §II.5 |
| `Smith2015CWRU` | Smith & Randall (2015) — CWRU benchmark (*MSSP* 64, 100–131) | Bab III §III.2.1; Bab IV §IV.2 |

**Self-citation format:** use `\citetitb{TotoSuharto2024Conf1SVM}` (never bare `\cite{}`). The ITB cls wraps these with the correct `dkk.` handling and bibliography style.

### Page Format (Pedoman §III.2–III.4)

- **Paper:** A4 (210 × 297 mm), HVS 80 gsm.
- **Font:** Times New Roman, 12 pt. (Class file sets this; do not override per-chapter.)
- **Spacing:** 1,5 spasi in body. Use single spacing for: block quotations, footnotes, captions, table contents, daftar pustaka entries.
- **Margins (mirror, two-sided print):**
  - Odd pages — left 4 cm, right 3 cm, top 3 cm, bottom 3 cm.
  - Even pages — left 3 cm, right 4 cm, top 3 cm, bottom 3 cm.
- **Paragraphs: no indent.** New paragraph starts at left margin, separated from the previous paragraph by **one blank line** (1,5 spasi). *(Class file: `\setlength{\parindent}{0pt}` + `\setlength{\parskip}{...}`.)*
- **No orphaned paragraphs:** never start a new paragraph at the bottom of a page unless ≥ 2 lines fit. Never leave a paragraph's last line alone at the top of the next page.
- **Each Bab starts on a new page.**
- **Page numbers:** Roman lowercase (`i`, `ii`, …) for front matter; Arabic for body; lampiran continues body numbering. Centered, 1,5 cm from the bottom edge. *(Handled by class file.)*

### Bab and Anak Bab Headings (Pedoman §IV.6, §VIII.6)

- **Bab title:** `Bab I` etc. — 14 pt **bold**, centered, 3 cm from top of page, no trailing period.
- **Numbering:** `Bab I`, `Bab II`, …; subbab uses Roman + Arabic separated by a period: `I.1`, `II.3`, `V.2`. Anak pada anak bab adds a third level: `III.2.1`.
- **Title case for headings:** capitalize the first letter of each significant word. **Do NOT capitalize** the following when they appear mid-title:
  - Conjunctions: `yang`, `karena`, `sebab`, `antara`, `padahal`, `dalam`, `bahwa`, `dan`, `untuk`, `sebagai`, `atau`, `tetapi`, `bila`, `apabila`, `juga`, `walau`, `walaupun`, `meski`, `meskipun`, `dengan`, `biarpun`, `jika`, `jikalau`, `kalau`, `maka`, `sehingga`, `oleh`, `serta`, `bagi`, `akan`, `kalaupun`.
  - Prepositions: `dari`, `daripada`, `terhadap`, `di`, `ke`, `pada`, `kepada`.
- **No period at end of any heading** (a title is not a sentence).
- **Never stack `\section` then `\subsection` immediately.** Insert at least one paragraph of prose between any bab title and the first subbab heading, and similarly between subbab and anak pada anak bab.
- **Anak bab heading:** **bold**, title-case, flush-left, no trailing period.

### Equations (Pedoman §VIII.5)

- Centered, on their own line; long equations break at arithmetic operators (`+`, `−`, `×`, `÷`, parens) — never at `/`.
- **Number on the right margin in parentheses:** `(BabRoman.urut)`, e.g., `(V.1)`. Use `\label{eq:bab5_rmse}` and reference via `\eqref{}` or `\autoref{}`.
- **Italic for variables/symbols** (math mode handles this).
- **Use brackets in hierarchy** `[ { ( … ) } ]`.
- **Do not start sentences with a formula.**
- **Numeric substitution:** write out the substitution like a normal equation; do not use `·` as multiplication.

### Figures and Tables (Pedoman §VII)

- **Caption format:** `Gambar V.2 Judul gambar` (sentence case, no terminal period). Tables similarly: `Tabel V.5 Judul tabel`.
- **Capitalize** `Gambar`, `Tabel`, `Bab`, `Lampiran`, `Persamaan` whenever followed by a number — e.g., `…seperti pada Gambar IV.3`, `…ditampilkan di Tabel V.2`. *(Title-case noun-before-number rule.)*
- **Figure caption** below the figure; **table caption above** the table.
- **Use `\autoref` or `\cref`** for cross-references — never hard-code "Gambar 5.2" — to stay consistent if numbering shifts.
- **Every float must be referenced** in the surrounding text. *Lint warns on orphan floats.*
- **No empty `\caption{}`** — fill in before commit. *Lint-enforced fatal.*
- **Cite the source** for figures borrowed from a paper, immediately in the caption.

### Bab Pendahuluan (Bab I) — Required Content (Pedoman §V.1)

The Bab Pendahuluan must contain at minimum (subbab structure flexible):

1. **Latar belakang dan deskripsi permasalahan** — fenomena saintifik + posisi penelitian terhadap penelitian sebelumnya (penulis sendiri + peneliti lain).
2. **Maksud, tujuan, lingkup, dan batasan permasalahan** — selaras dengan latar belakang.
3. **Rumusan masalah (statement of the problem) / pertanyaan penelitian (research question)** — pernyataan/RQ eksplisit.
4. **Cara pendekatan dan metodologi** — tahapan + software/tools.
5. **Asumsi** — landasan untuk hipotesis.
6. **Hipotesis** — jawaban sementara, singkat dan padat.
7. **Kebaruan dan orisinalitas (novelty and originality)** — wajib, dengan tipe kontribusi (Konsep-Objek / Teknologi-Metodologi / Keluaran).

*Cross-check against [dissertation-outline.md §Bab I](writings/dissertation-outline.md) — sections 1.1–1.7 already map this.*

### Bab Tinjauan Pustaka (Bab II) — Scope (Pedoman §V.2)

- **NOT** a dump of fundamental theory or generic method exposition (that goes in Bab Dasar Teori / Bab Metodologi).
- **IS** an elaboration of prior researchers' results that establishes the gap and motivates the current work.
- Organize by the chronological/conceptual development of the field; conclude with how/why this dissertation's topic and approach were chosen.

### Bab Kesimpulan — Scope (Pedoman §V.4)

- Elaborates and details the conclusions stated in the abstract.
- Includes saran untuk kajian lanjutan + practical implications.

### Abstrak (Pedoman §II.2)

- **500–800 words**, single-spacing, same margins as body.
- **Both Indonesian and English versions**, each on a new page.
- **No references** in the abstract (`tidak boleh ada hasil kajian dari referensi`).
- Header lines (uppercase, 14 pt bold, single-spaced): `ABSTRAK`, judul disertasi, `Oleh`, nama, NIM, prodi. First abstract paragraph begins 3 spaces below the prodi line.
- **Keywords:** max 7 single words OR 2-word meaningful phrases, drawn from the abstract content (not from the dissertation body). Placed on a separate line at the bottom of the abstract page.

### Lampiran (Pedoman §II.7)

- Identified by uppercase Latin letters: `A`, `B`, `C`, …
- Each lampiran is preceded by a page containing only the word `LAMPIRAN` (14 pt bold, centered, with a page number).
- Page numbers in lampiran continue from the body's Arabic numbering.

### Daftar Singkatan dan Lambang (Pedoman §IV.10)

- Single-spaced; 3 columns: (1) singkatan/lambang, (2) nama lengkap, (3) halaman pertama muncul.
- **Alphabetical order:** Latin uppercase first, then Latin lowercase, then Greek (in Greek alphabetical order).

### Other ITB-Specific Conventions

- **Hard-cover binding** (Sidang Promosi version): dark blue (Biru Dongker), Omega No. 10 paper, gold lettering. Not relevant during writing — only at final submission.
- **No TODO / FIXME / `\dots` placeholders in the final manuscript.** Lint warns; remove before submission.
- **All captions must be filled in** (`\caption{...}` cannot be empty). *Lint-enforced fatal.*
- **`\ref` integrity:** every `\label` should be referenced by something; every `\ref` must point to an existing label. *Lint-enforced.*

---

## What Is Not in This Repo

Local-only, excluded from git:

- [Mamba-xLSTM/](Mamba-xLSTM/) — Python training pipeline (PyTorch/Lightning models, training scripts, configs). See [README.md](README.md) → Training section for the quick-start command.
- [data-bearing/](data-bearing/) — PHM2012 and XJTU-SY bearing datasets (~8.5 GB). Download via the S3 URL in [README.md](README.md).
- `.cursor/` — Cursor IDE rules. The Makefile references `.cursor/rules/14-build-workflow.mdc` for extended build documentation.

Sources:
- [PEDOMAN PENULISAN DISERTASI DOKTOR — SPs ITB (April 2016)](https://multisite.itb.ac.id/sps/wp-content/uploads/sites/45/2015/12/PEDOMAN_PENULISAN_DISERTASI_DOKTOR_ITB.pdf)
- [Pedoman Tesis dan Disertasi — SPs ITB](https://sps.itb.ac.id/pedoman-tesis-dan-disertasi/)
- [Pedoman Penulisan Usulan/Proposal Penelitian Disertasi Doktor — S3 TMI ITB](https://s3tmi.fti.itb.ac.id/pedoman-penulisan-usulan-proposal-penelitian-disertasi-doktor/)
- [SPs ITB Citation Style Language (CSL)](https://itb-sps.github.io/csl/)
