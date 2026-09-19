# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Doctoral dissertation by **Toto Suharto** at **ITB (Institut Teknologi Bandung)**, Program Doktor Teknik dan Manajemen Industri, FTI.

**Working title:** *Perawatan Prediktif untuk Sistem Produksi dengan Pendekatan Analisis Big Data dan Kecerdasan Buatan Menggunakan Data Kondisi Mesin dan Informasi Kualitas yang Real Time*

**Repository scope:** the **dissertation manuscript** ([dissertation-docx/](dissertation-docx/)), the sidang deck ([presentation/](presentation/)) and supporting writings. The Python training code ([Mamba-xLSTM/](Mamba-xLSTM/)) and datasets ([data-bearing/](data-bearing/)) live locally only — excluded via `.gitignore`.

## ★ DOCX Is the Manuscript

The manuscript of record is **[dissertation-docx/disertasi.docx](dissertation-docx/)**, on the official ITB template (`template-disertasi_Mei2019.docx`). It is edited directly in Word. **There is no source format to regenerate it from:** the LaTeX tree it was ported from was deleted in September 2026 once the port was finished, and is retrievable from git history if ever needed.

Drafts circulated for review live alongside the deck as `presentation/V<N> …docx`. **V16 is current.** Only the current version is tracked; `.gitignore` carries one negation line per version, added deliberately because each file is ~18 MB and a DOCX is a compressed zip that git cannot delta.

- **Writing rules for DOCX work:** [dissertation-docx/RULES.md](dissertation-docx/RULES.md) — style map (Heading1–3, `Paragraf`, `JudulGambar`/`judulTabel`, `Daftarpustaka`), caption/REF/SEQ field recipes, citation convention (baked text via ITB-SPs CSL), and the ITB prose rules below.
- **Lint:** `bash tools/lint_docx.sh chapters/*.docx disertasi.docx` from `dissertation-docx/` (or `make lint`). Fix `[FATAL]` before commit.
- **Adding material:** `make insert-dry`, then `make insert`, from `dissertation-docx/`. This inserts into the hand-edited manuscript in place ([tools/insert_docx.py](dissertation-docx/tools/insert_docx.py)). Never regenerate a chapter wholesale — the Word edits *are* the content.
- **`make master`** merges the existing chapter and lampiran files; it does not rebuild them.
- **After manual edits in Word:** run the lint; in the final master, update fields once (Ctrl+A, F9) to populate Daftar Isi/Gambar/Tabel.

**Editing a DOCX by script.** Text replacement over `word/document.xml` has three traps, each of which fails silently: a self-closing `<w:t xml:space="preserve"/>` matches a naive opening-tag regex and swallows the markup after it; the text uses non-breaking spaces (`Bab\u00a0III`, `HI\u00a036-D`); and several matches inside one run overwrite each other unless edits are recorded in original coordinates and applied back to front. Apply replacement pairs longest-first, and always diff the full `<m:oMath>` list before and after, since `<m:t>` math runs are a separate namespace that `<w:t>` edits never touch. Verify with: part count unchanged, every `.xml`/`.rels` still parses, no markup leaked into the extracted text.

## Main Goal — Modify and Finalize the Dissertation

The primary purpose of this codebase is to **finalize Pak Toto's doctoral dissertation following the plan in [writings/dissertation-outline.md](writings/dissertation-outline.md)** — this is the **single source of truth** for chapter structure, section content, figures, tables, and lampiran.

The outline encodes a two-track narrative:

- **Jalur A — Diagnostik** (Bab IV): CWRU → benchmark kernel/tree/deep → WDCNN + SHAP DeepExplainer + Fault Signature Maps. Sources: [Paper/Conference1_*.pdf](Paper/), [Paper/Conference2_*.pdf](Paper/), [Paper/Journal1_*.pdf](Paper/).
- **Jalur B — Prognostik** (Bab V): PHM2012/XJTU-SY/IMS → 3 backbone RUL (Mamba-xLSTM, N-BEATS-xLSTM, SparseGate-TCN) → Top-*k* Sparse Autoencoder → BPFx mapping. Source: [Paper/Journal2_RUL_Journal.pdf](Paper/).
- **Konvergensi** (Bab VI): kerangka PdM multi-tier (Edge IoT → Edge Server → Cloud/GPU) + interpretabilitas dua-lapis (input attribution ↔ latent concept).

**Where the work stands.** The six-bab restructuring the outline calls for is done and ported into the DOCX. What remains is revision: responding to Pak Toto's comments, keeping the deck and the manuscript in step, and closing the TODOs below.

**Working rules for the dissertation:**

- Always re-read [dissertation-outline.md](writings/dissertation-outline.md) before editing a chapter — section structure may have evolved.
- Small commits per section so Pak Toto can review incrementally.
- After every section edit, run `make lint` from [dissertation-docx/](dissertation-docx/) and fix every `[FATAL]`.
- **Code is the ground truth.** When the manuscript and the run artifacts under `Mamba-xLSTM/results/` disagree, fix the writing, not the code — and identify which run produced a number before changing either.
- Outstanding TODOs (track in commits / PR descriptions): verify venue+year for the 4 self-citations; confirm PT~SKF data volume + ground-truth; flag missing Journal 2 notebook.

## Repository Layout

```
dissertation-docx/              # ★ The manuscript: disertasi.docx + chapters/ + lampiran/
  RULES.md                      # Writing and formatting rules for DOCX work
  tools/                        # insert_docx.py, lint_docx.sh, merge_master.py
  assets/                       # template.docx, itb-sps.csl, figures/, figure-map.tsv
presentation/                   # Sidang deck (content/*.yaml -> build.py) + V<N> draft DOCX
writings/
  dissertation-outline.md       # ★ Single source of truth for chapter structure
  journal-q2/                   # JETS Q2 paper — its own LaTeX project, unrelated to the
                                #   dissertation build; jets-mechanistic-interp/ + jets-docs/
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

## Build and Checks

All commands run from [dissertation-docx/](dissertation-docx/):

```bash
make insert-dry   # preview an insertion spec without touching the manuscript
make insert       # apply it (tools/insert_docx.py)
make lint         # ITB format + prose lint over the chapter DOCX files
make master       # merge frontmatter + chapters + lampiran into disertasi.docx
```

The sidang deck is separate, from [presentation/](presentation/):

```bash
python3 build.py lint --spec content/sidang-terbuka.yaml     # structure + overflow
python3 build.py build --spec content/sidang-terbuka.yaml --strict
python3 build.py preview --spec content/sidang-terbuka.yaml --png   # PDF + slide-NN.png
python3 tools/render_diagrams.py [--only name1,name2]        # matplotlib figures
```

Manuscript language: **Bahasa Indonesia** (only the English abstract is in English).

---

## ITB Doctoral Dissertation Writing Rules

These rules come from the official *Pedoman Penulisan Disertasi Doktor ITB* (SPs, April 2016) and must be followed for every paragraph, table, equation, citation, and lampiran written into the manuscript. Many are auto-enforced by [dissertation-docx/tools/lint_docx.sh](dissertation-docx/tools/lint_docx.sh) — **fix lint violations before committing**. [RULES.md](dissertation-docx/RULES.md) carries the Word-mechanics counterpart: styles, fields, caption recipes.

### Language (Pedoman §III.1, §III.2)

- **Bahasa Indonesia Baku.** Follow KBBI, EYD (PUEBI), and Pedoman Umum Pembentukan Istilah. Use complete, well-punctuated sentences.
- **No first-person pronouns** (`saya`, `kami`, `kita`) anywhere in the body — restructure to passive voice. Allowed only in Kata Pengantar. *Lint-enforced.*
- **No `di mana`** as a relative pronoun (a `where`-calque). Use `yang`, `tempat`, `pada saat`, or restructure.
- **Do not start sentences with conjunctions** `maka`, `sedangkan`, `sehingga`. Restructure.
- **Do not start sentences with numerals or symbols.** Spell out (`Sepuluh model …`) or restructure.
- **No `&` for `dan`.** Reserve `&` for equations and table markup only. *Lint-enforced.*
- **Avoid foreign terms** when an established Indonesian term exists. When a foreign term is necessary, italicize it consistently (*deep learning*, *Sparse Autoencoder*, *envelope spectrum*). Genus/species names are always italic (*Sonchus arvensis*).
- **Domain terminology — bearing field: use English.** In rotating-machinery condition-monitoring literature the accepted international terms are `bearing` (not `bantalan`) and `rolling` (not `gelinding`). Always write `bearing` and `rolling element` in body text; do not substitute the Indonesian translations. *Lint-enforced (cek #17).*
- **No foreign connecting words.** Indonesian text must never use foreign linker/preposition shorthands like `vs`, `via`, `etc.`, `i.e.`, `e.g.`, `cf.` as syntactic glue. Use Indonesian equivalents: `vs` → `dan` / `terhadap` / `dengan` / `dibandingkan dengan` (per context); `via` → `melalui`; `etc.` → `dll.`; `i.e.` → `yaitu`; `e.g.` → `misalnya`; `cf.` → `bandingkan dengan`. Italicized technical terms that *contain* such tokens (*one-vs-rest*, scikit-learn `multi_class='ovr'`) are exempt — they are names, not connectors.
- **Do not use `keluarga` to label a group of algorithms or models.** "Keluarga" carries a biological/genealogical connotation and reads unnaturally in Indonesian technical prose. Use instead: `jenis`, `varian`, `golongan`, `kelompok`, `kelas`, `algoritma`, or `model` — whichever fits the context. Examples: ~~"keluarga kernel"~~ → `model berbasis kernel`; ~~"keluarga tree"~~ → `algoritma berbasis pohon keputusan`; ~~"multi-keluarga"~~ → `multi-model`. *Lint-enforced (cek #16).*
- **No em dashes (`—`) in prose.** Em dashes are a strong AI-writing marker absent from standard Indonesian academic style. Replace with: a comma (parenthetical insert), a semicolon (related clause), a colon (elaboration), or restructure into two sentences. Watch for Word's AutoFormat turning `--` into `—` as you type. *Lint-enforced (cek #16).*
- **One main idea per paragraph.** Never write a single-sentence paragraph.
- **Spelling — baku KBBI** (lint-enforced sample): `objek` (not `obyek`), `analisis` (not `analisa`), `sintesis` (not `sintesa`), `aktivitas` (not `aktifitas`), `praktik` (not `praktek`), `nasihat` (not `nasehat`), `risiko` (not `resiko`), `frekuensi` (not `frekwensi`), `sistem` (not `sistim`), `jadwal` (not `jadual`), `manajemen` (not `managemen`), `teknologi` (not `technologi`), `efektif` (not `effektif`), `efisien` (not `effisien`), `asesmen` (not `assesment`), `asas` (not `azas`), `hipotesis` (not `hipotesa`).

### Gaya Prosa Organik — Menghindari Ciri Tulisan Mesin

Every paragraph must read as written by a researcher who thought carefully about their subject — not assembled from templates. Patterns drawn from Dito Eka Cahya, *Perancangan dan Implementasi Robot Pengikut Garis dengan Sensor Kamera Pihak Ketiga*, ITB 2009.

**Variasi pembuka paragraf.** Never open two consecutive paragraphs with the same word or phrase. Rotate through these natural openers:
- `Pada [frasa nominal],` — local context: `Pada sistem ini,` / `Pada percobaan berikut,`
- `Untuk [frasa kerja],` — states purpose: `Untuk dapat mengontrol robot,`
- `Dengan [frasa nominal/verbal],` — instrument/method: `Dengan menggunakan metode Harris,`
- `Dalam [frasa nominal],` — scope framing: `Dalam penelitian ini,` / `Dalam hal ini,`
- `Setelah [klausa],` — sequential: `Setelah proses pemetaan dilakukan,`
- `Berdasarkan [nominal],` — grounds a claim: `Berdasarkan hasil percobaan,`
- `Salah satu [nominal] adalah` — specific instance from a class
- `[Subjek] merupakan/adalah [klaim].` — direct definitional opening
- `[Fakta historis/empiris].` — concrete fact, not meta-statement

**Penghubung yang diizinkan di awal kalimat:** `Oleh karena itu,` · `Dengan demikian,` · `Selain itu,` · `Namun,` · `Akan tetapi,` · `Adapun` · `Di sisi lain,` · `Sebaliknya,` · `Berdasarkan hal tersebut,`

**Penghubung yang hanya boleh di TENGAH kalimat** (bukan awal): `sehingga` · `sedangkan` · `maka` (kecuali pola `jika…, maka…` atau `berdasarkan…, maka…`).

**Pola kausal organik.** Use these natural causal frames; never use "memainkan peran penting":
- `Hal ini disebabkan oleh [sebab].`
- `Hal ini terjadi karena [alasan].`
- `Hal ini mengakibatkan [akibat].`
- `Karena [X], [Y pun terjadi/dilakukan].`

**Frasa mesin yang dilarang** (*lint-enforced* — cek #16):

| Terlarang | Alternatif |
|---|---|
| `tidak dapat dipungkiri` | nyatakan fakta langsung, atau `terbukti bahwa` |
| `memainkan peran penting` | `berfungsi sebagai X` / `menentukan nilai Y` |
| `penting untuk dicatat bahwa` / `perlu dicatat bahwa` | integrasikan langsung ke kalimat utama |
| `dalam era modern` / `di era digital` | `sejak tahun 20XX` / `dengan berkembangnya X` |
| `tentunya` | hapus; jika genuinely implied, tulis `dengan sendirinya` |
| `secara komprehensif` / `secara holistik` | jelaskan cakupan secara konkret |
| `merupakan hal yang (sangat) penting/krusial` | jelaskan secara konkret mengapa penting |
| `perlu dipahami bahwa` / `perlu diperhatikan bahwa` | tulis pernyataan langsung tanpa preambel |
| Em dash `—` / `---` dalam prosa | koma, titik koma, titik dua, atau pisah kalimat |
| `pada dasarnya` / `pada intinya` | hapus atau nyatakan klaim secara langsung |
| `dapat dikatakan bahwa` | hapus — tulis pernyataan tanpa hedging |
| `berbagai macam` | sebutkan jumlah konkret atau gunakan `sejumlah` / `beberapa` |
| `Lebih lanjut,` (pembuka kalimat) | `Selain itu,` / `Di samping itu,` / integrasikan ke kalimat sebelumnya |
| `sangat penting` / `amat penting` | jelaskan secara konkret mengapa penting |

**Tanda tulisan manusiawi yang wajib dijaga:**
- **Angka spesifik:** selalu sebutkan hasil numerik aktual (`12,55 piksel`, `1706 milidetik`), bukan "nilai yang cukup besar".
- **Akui kegagalan/keterbatasan** — jika pendekatan pertama tidak berhasil, dokumentasikan dan jelaskan solusi iteratifnya.
- **Rationale eksplisit** — setiap pilihan desain disertai `Alasan dipilihnya X adalah karena…` atau setara.
- **Panjang paragraf 3–5 kalimat** — hindari paragraf satu kalimat dan paragraf lebih dari 7 kalimat.
- **Variasi panjang kalimat** — selingi kalimat pendek (10–15 kata) di antara kalimat panjang (25–35 kata).
- **Tidak ada dua paragraf berurutan dengan penghubung pembuka yang sama** — jika paragraf sebelumnya dibuka `Selain itu,`, paragraf berikutnya tidak boleh sama.

### Numbers and Units (Pedoman §VIII.3)

- **Decimal separator: comma** (`25,5`). Never use a period. *Lint heuristically enforced.* Watch Word's locale: an English keyboard layout will produce `25.5`.
- **Thousands separator: period** (`1.000.000`). To avoid ambiguity with the decimal comma, avoid 3-digit decimals — prefer `25,24` or `25,2472`, not `25,247`.
- **Numbers < 10 written out** (`enam perguruan tinggi`); **≥ 10 use digits** (`17 mangga`).
- **Vague/round quantities in words** (`sepuluh tahun yang lalu`, `lima kali sehari`).
- **Avoid Roman numerals for ordinary numbers** (Roman is reserved for chapter/equation numbers).
- **Scientific notation:** `1,91 × 10^6` or `1,91E6` for `1.908.176`.
- **SI units:** abbreviations after numerals (`5 kg`), full word when used as a noun (`Massa diukur dalam kilogram`).

### Citations and Bibliography (Pedoman §VI; §VIII.7)

- **No footnotes for references.** Cite inline; weave any auxiliary remark into the sentence.
- **In-text format:** `(Surname, year)` parenthetical, or `Surname (year)` narrative.
- **In-text max 2 authors.** For ≥ 3, use `Surname-pertama dkk. (tahun)` — **`dkk.`, never `et al.`**. *Lint-enforced.* Citation text is baked into the document, not a live field, so it is typed and must be got right by hand.
- **In the Daftar Pustaka: list ALL authors.** Use `dan` (not `and` or `&`) before the last author. The `dkk.` shortening only applies in body text, never in the bibliography entry.
- **Style:** `Surname, Initial. (year): Title in sentence case, *Journal Name in italic*, **volume in bold**, start–end pages.` Example:
  > Cotton, F.A. (1998): Kinetics of gasification of brown coal, *Journal of American Chemical Society*, **54**, 38–43.
- **Sentence case for paper titles** (capital only on the first word + proper nouns).
- **Journal name italic, volume bold.** Use an en-dash (`–`) for page ranges, not a hyphen.
- **Hanging indent 1,27 cm** (7 ketukan) for each entry; single-spacing within and between entries.
- **Alphabetical by first author's family name; no numbering.**
- **Allowed source types:** journal/proceeding articles, books, theses/disertasi, websites (cite per discipline norm). Newspapers/TV/film are allowed **only when the artifact is itself the research object** — never for general background.
- **Every entry in the Daftar Pustaka MUST be cited in the body**, and vice versa. Entries are sorted alphabetically by the first author's family name; insert a new one in position rather than appending it.
- **Style reference:** the ITB-SPs CSL (`dissertation-docx/assets/itb-sps.csl`) is what produced the existing entries. Match an existing entry's run structure when adding one: plain run for authors and title, italic run for journal name and volume.

#### The Four Self-Papers

The four self-papers in [Paper/](Paper/) are the **primary empirical sources** for Bab IV and Bab V. Cite them by the baked text in the Daftar Pustaka; confirm venue, year, volume and DOI with Pak Toto before finalising.

| File in `Paper/` | Cite in | Notes |
|---|---|---|
| [Conference1_Classification_SVM_LR.pdf](Paper/Conference1_Classification_SVM_LR.pdf) | Bab IV §IV.2; Bab II §II.2.3 | SVM/LR + SHAP KernelExplainer on CWRU |
| [Conference2_Classification_Tree.pdf](Paper/Conference2_Classification_Tree.pdf) | Bab IV §IV.3; Bab II §II.2.3 | DT/RF/XGBoost + SHAP TreeExplainer on CWRU |
| [Journal1_Fault Signature Maps.pdf](Paper/Journal1_Fault%20Signature%20Maps.pdf) | Bab IV §IV.4–§IV.13; Bab II §II.3.3 | WDCNN + SHAP DeepExplainer + FSM; key result akurasi 99,87% |
| [Journal2_RUL_Journal.pdf](Paper/Journal2_RUL_Journal.pdf) | Bab V §V.1–§V.9 throughout | Mamba-xLSTM + SAE-BPFx; top-k SAE mechanistic interpretability |

A fifth paper is in progress and is **not** part of the dissertation build: the JETS Q2 manuscript in [writings/journal-q2/](writings/journal-q2/), on mechanistic interpretability of RUL models via top-*k* sparse autoencoders. It is its own LaTeX project with its own `references.bib` and class file — leave it alone when working on the dissertation.

### Page Format (Pedoman §III.2–III.4)

- **Paper:** A4 (210 × 297 mm), HVS 80 gsm.
- **Font:** Times New Roman, 12 pt. (Class file sets this; do not override per-chapter.)
- **Spacing:** 1,5 spasi in body. Use single spacing for: block quotations, footnotes, captions, table contents, daftar pustaka entries.
- **Margins (mirror, two-sided print):**
  - Odd pages — left 4 cm, right 3 cm, top 3 cm, bottom 3 cm.
  - Even pages — left 3 cm, right 4 cm, top 3 cm, bottom 3 cm.
- **Paragraphs: no indent.** New paragraph starts at left margin, separated from the previous paragraph by **one blank line** (1,5 spasi). *(Word style `Paragraf` carries this; do not indent by hand.)*
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
- **Number on the right margin in parentheses:** `(BabRoman.urut)`, e.g., `(V.1)`. Built with SEQ fields and referenced with REF fields — recipe in [RULES.md §5](dissertation-docx/RULES.md).
- **Italic for variables/symbols** (math mode handles this).
- **Use brackets in hierarchy** `[ { ( … ) } ]`.
- **Do not start sentences with a formula.**
- **Numeric substitution:** write out the substitution like a normal equation; do not use `·` as multiplication.

### Figures and Tables (Pedoman §VII)

- **Caption format:** `Gambar V.2 Judul gambar` (sentence case, no terminal period). Tables similarly: `Tabel V.5 Judul tabel`.
- **Capitalize** `Gambar`, `Tabel`, `Bab`, `Lampiran`, `Persamaan` whenever followed by a number — e.g., `…seperti pada Gambar IV.3`, `…ditampilkan di Tabel V.2`. *(Title-case noun-before-number rule.)*
- **Figure caption** below the figure; **table caption above** the table.
- **Use REF fields** for cross-references — never hard-code "Gambar 5.2" — so numbering survives insertions. Recipe in [RULES.md §4](dissertation-docx/RULES.md).
- **Every float must be referenced** in the surrounding text. *Lint warns on orphan floats.*
- **No empty caption** — fill it in before commit. *Lint-enforced fatal.*
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
- **No TODO / FIXME / `…` placeholders in the final manuscript.** Lint warns; remove before submission.
- **All captions must be filled in.** *Lint-enforced fatal.*
- **Field integrity:** every REF field must resolve. After a round of edits, select all and press F9 so Daftar Isi, Daftar Gambar and Daftar Tabel repopulate; a field showing `Error! Reference source not found.` is a fatal.

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
