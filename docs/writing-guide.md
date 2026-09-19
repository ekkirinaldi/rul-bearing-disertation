# Writing Guide

The rules below come from the official *Pedoman Penulisan Disertasi Doktor ITB* (SPs, April 2016) and must be followed for every paragraph, table, equation, citation, and lampiran written into the manuscript. Many are auto-enforced by [manuscript/tools/lint_docx.sh](../manuscript/tools/lint_docx.sh) — **fix lint violations before committing**. [manuscript/RULES.md](../manuscript/RULES.md) carries the Word-mechanics counterpart: styles, fields, caption recipes.

## Language (Pedoman §III.1, §III.2)

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
- **No em dashes (`—`) in prose.** Standard Indonesian academic style does not use them. Replace with: a comma (parenthetical insert), a semicolon (related clause), a colon (elaboration), or restructure into two sentences. Watch for Word's AutoFormat turning `--` into `—` as you type. *Lint-enforced (cek #16).*
- **One main idea per paragraph.** Never write a single-sentence paragraph.
- **Spelling — baku KBBI** (lint-enforced sample): `objek` (not `obyek`), `analisis` (not `analisa`), `sintesis` (not `sintesa`), `aktivitas` (not `aktifitas`), `praktik` (not `praktek`), `nasihat` (not `nasehat`), `risiko` (not `resiko`), `frekuensi` (not `frekwensi`), `sistem` (not `sistim`), `jadwal` (not `jadual`), `manajemen` (not `managemen`), `teknologi` (not `technologi`), `efektif` (not `effektif`), `efisien` (not `effisien`), `asesmen` (not `assesment`), `asas` (not `azas`), `hipotesis` (not `hipotesa`).

## Gaya Prosa

Each paragraph should read as the work of a researcher who has thought about the subject, not as a filled-in template. The patterns below are drawn from Dito Eka Cahya, *Perancangan dan Implementasi Robot Pengikut Garis dengan Sensor Kamera Pihak Ketiga*, ITB 2009.

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

**Frasa klise yang dihindari** (*lint-enforced* — cek #16):

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

**Kebiasaan yang dijaga:**
- **Angka spesifik:** selalu sebutkan hasil numerik aktual (`12,55 piksel`, `1706 milidetik`), bukan "nilai yang cukup besar".
- **Akui kegagalan/keterbatasan** — jika pendekatan pertama tidak berhasil, dokumentasikan dan jelaskan solusi iteratifnya.
- **Rationale eksplisit** — setiap pilihan desain disertai `Alasan dipilihnya X adalah karena…` atau setara.
- **Panjang paragraf 3–5 kalimat** — hindari paragraf satu kalimat dan paragraf lebih dari 7 kalimat.
- **Variasi panjang kalimat** — selingi kalimat pendek (10–15 kata) di antara kalimat panjang (25–35 kata).
- **Tidak ada dua paragraf berurutan dengan penghubung pembuka yang sama** — jika paragraf sebelumnya dibuka `Selain itu,`, paragraf berikutnya tidak boleh sama.

## Numbers and Units (Pedoman §VIII.3)

- **Decimal separator: comma** (`25,5`). Never use a period. *Lint heuristically enforced.* Watch Word's locale: an English keyboard layout will produce `25.5`.
- **Thousands separator: period** (`1.000.000`). To avoid ambiguity with the decimal comma, avoid 3-digit decimals — prefer `25,24` or `25,2472`, not `25,247`.
- **Numbers < 10 written out** (`enam perguruan tinggi`); **≥ 10 use digits** (`17 mangga`).
- **Vague/round quantities in words** (`sepuluh tahun yang lalu`, `lima kali sehari`).
- **Avoid Roman numerals for ordinary numbers** (Roman is reserved for chapter/equation numbers).
- **Scientific notation:** `1,91 × 10^6` or `1,91E6` for `1.908.176`.
- **SI units:** abbreviations after numerals (`5 kg`), full word when used as a noun (`Massa diukur dalam kilogram`).

## Citations and Bibliography (Pedoman §VI; §VIII.7)

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
- **Style reference:** the ITB-SPs CSL (`manuscript/assets/itb-sps.csl`) is what produced the existing entries. Match an existing entry's run structure when adding one: plain run for authors and title, italic run for journal name and volume.

### The Four Self-Papers

The four self-papers in [research/papers/](../research/papers/) are the **primary empirical sources** for Bab IV and Bab V. Cite them by the baked text in the Daftar Pustaka; confirm venue, year, volume and DOI with Pak Toto before finalising.

| File in `research/papers/` | Cite in | Notes |
|---|---|---|
| [Conference1_Classification_SVM_LR.pdf](../research/papers/Conference1_Classification_SVM_LR.pdf) | Bab IV §IV.2; Bab II §II.2.3 | SVM/LR + SHAP KernelExplainer on CWRU |
| [Conference2_Classification_Tree.pdf](../research/papers/Conference2_Classification_Tree.pdf) | Bab IV §IV.3; Bab II §II.2.3 | DT/RF/XGBoost + SHAP TreeExplainer on CWRU |
| [Journal1_Fault Signature Maps.pdf](../research/papers/Journal1_Fault%20Signature%20Maps.pdf) | Bab IV §IV.4–§IV.13; Bab II §II.3.3 | WDCNN + SHAP DeepExplainer + FSM; key result akurasi 99,87% |
| [Journal2_RUL_Journal.pdf](../research/papers/Journal2_RUL_Journal.pdf) | Bab V §V.1–§V.9 throughout | Mamba-xLSTM + SAE-BPFx; top-k SAE mechanistic interpretability |

A fifth paper is in progress and is **not** part of the dissertation build: the JETS Q2 manuscript in [writings/journal-q2/](../writings/journal-q2/), on mechanistic interpretability of RUL models via top-*k* sparse autoencoders. It is its own LaTeX project with its own `references.bib` and class file — leave it alone when working on the dissertation.

## Page Format (Pedoman §III.2–III.4)

- **Paper:** A4 (210 × 297 mm), HVS 80 gsm.
- **Font:** Times New Roman, 12 pt. (The Word template sets this; do not override it per chapter.)
- **Spacing:** 1,5 spasi in body. Use single spacing for: block quotations, footnotes, captions, table contents, daftar pustaka entries.
- **Margins (mirror, two-sided print):**
  - Odd pages — left 4 cm, right 3 cm, top 3 cm, bottom 3 cm.
  - Even pages — left 3 cm, right 4 cm, top 3 cm, bottom 3 cm.
- **Paragraphs: no indent.** New paragraph starts at left margin, separated from the previous paragraph by **one blank line** (1,5 spasi). *(Word style `Paragraf` carries this; do not indent by hand.)*
- **No orphaned paragraphs:** never start a new paragraph at the bottom of a page unless ≥ 2 lines fit. Never leave a paragraph's last line alone at the top of the next page.
- **Each Bab starts on a new page.**
- **Page numbers:** Roman lowercase (`i`, `ii`, …) for front matter; Arabic for body; lampiran continues body numbering. Centered, 1,5 cm from the bottom edge. *(Handled by the template's section setup.)*

## Bab and Anak Bab Headings (Pedoman §IV.6, §VIII.6)

- **Bab title:** `Bab I` etc. — 14 pt **bold**, centered, 3 cm from top of page, no trailing period.
- **Numbering:** `Bab I`, `Bab II`, …; subbab uses Roman + Arabic separated by a period: `I.1`, `II.3`, `V.2`. Anak pada anak bab adds a third level: `III.2.1`.
- **Title case for headings:** capitalize the first letter of each significant word. **Do NOT capitalize** the following when they appear mid-title:
  - Conjunctions: `yang`, `karena`, `sebab`, `antara`, `padahal`, `dalam`, `bahwa`, `dan`, `untuk`, `sebagai`, `atau`, `tetapi`, `bila`, `apabila`, `juga`, `walau`, `walaupun`, `meski`, `meskipun`, `dengan`, `biarpun`, `jika`, `jikalau`, `kalau`, `maka`, `sehingga`, `oleh`, `serta`, `bagi`, `akan`, `kalaupun`.
  - Prepositions: `dari`, `daripada`, `terhadap`, `di`, `ke`, `pada`, `kepada`.
- **No period at end of any heading** (a title is not a sentence).
- **Never place a subheading directly under a heading.** Insert at least one paragraph of prose between any bab title and the first subbab heading, and similarly between subbab and anak pada anak bab.
- **Anak bab heading:** **bold**, title-case, flush-left, no trailing period.

## Equations (Pedoman §VIII.5)

- Centered, on their own line; long equations break at arithmetic operators (`+`, `−`, `×`, `÷`, parens) — never at `/`.
- **Number on the right margin in parentheses:** `(BabRoman.urut)`, e.g., `(V.1)`. Built with SEQ fields and referenced with REF fields — recipe in [RULES.md §5](../manuscript/RULES.md).
- **Italic for variables/symbols** (the equation editor does this).
- **Use brackets in hierarchy** `[ { ( … ) } ]`.
- **Do not start sentences with a formula.**
- **Numeric substitution:** write out the substitution like a normal equation; do not use `·` as multiplication.

## Figures and Tables (Pedoman §VII)

- **Caption format:** `Gambar V.2 Judul gambar` (sentence case, no terminal period). Tables similarly: `Tabel V.5 Judul tabel`.
- **Capitalize** `Gambar`, `Tabel`, `Bab`, `Lampiran`, `Persamaan` whenever followed by a number — e.g., `…seperti pada Gambar IV.3`, `…ditampilkan di Tabel V.2`. *(Title-case noun-before-number rule.)*
- **Figure caption** below the figure; **table caption above** the table.
- **Use REF fields** for cross-references — never hard-code "Gambar 5.2" — so numbering survives insertions. Recipe in [RULES.md §4](../manuscript/RULES.md).
- **Every float must be referenced** in the surrounding text. *Lint warns on orphan floats.*
- **No empty caption** — fill it in before commit. *Lint-enforced fatal.*
- **Cite the source** for figures borrowed from a paper, immediately in the caption.

## Bab Pendahuluan (Bab I) — Required Content (Pedoman §V.1)

The Bab Pendahuluan must contain at minimum (subbab structure flexible):

1. **Latar belakang dan deskripsi permasalahan** — fenomena saintifik + posisi penelitian terhadap penelitian sebelumnya (penulis sendiri + peneliti lain).
2. **Maksud, tujuan, lingkup, dan batasan permasalahan** — selaras dengan latar belakang.
3. **Rumusan masalah (statement of the problem) / pertanyaan penelitian (research question)** — pernyataan/RQ eksplisit.
4. **Cara pendekatan dan metodologi** — tahapan + software/tools.
5. **Asumsi** — landasan untuk hipotesis.
6. **Hipotesis** — jawaban sementara, singkat dan padat.
7. **Kebaruan dan orisinalitas (novelty and originality)** — wajib, dengan tipe kontribusi (Konsep-Objek / Teknologi-Metodologi / Keluaran).

*Cross-check against [dissertation-outline.md §Bab I](dissertation-outline.md) — sections 1.1–1.7 already map this.*

## Bab Tinjauan Pustaka (Bab II) — Scope (Pedoman §V.2)

- **NOT** a dump of fundamental theory or generic method exposition (that goes in Bab Dasar Teori / Bab Metodologi).
- **IS** an elaboration of prior researchers' results that establishes the gap and motivates the current work.
- Organize by the chronological/conceptual development of the field; conclude with how/why this dissertation's topic and approach were chosen.

## Bab Kesimpulan — Scope (Pedoman §V.4)

- Elaborates and details the conclusions stated in the abstract.
- Includes saran untuk kajian lanjutan + practical implications.

## Abstrak (Pedoman §II.2)

- **500–800 words**, single-spacing, same margins as body.
- **Both Indonesian and English versions**, each on a new page.
- **No references** in the abstract (`tidak boleh ada hasil kajian dari referensi`).
- Header lines (uppercase, 14 pt bold, single-spaced): `ABSTRAK`, judul disertasi, `Oleh`, nama, NIM, prodi. First abstract paragraph begins 3 spaces below the prodi line.
- **Keywords:** max 7 single words OR 2-word meaningful phrases, drawn from the abstract content (not from the dissertation body). Placed on a separate line at the bottom of the abstract page.

## Lampiran (Pedoman §II.7)

- Identified by uppercase Latin letters: `A`, `B`, `C`, …
- Each lampiran is preceded by a page containing only the word `LAMPIRAN` (14 pt bold, centered, with a page number).
- Page numbers in lampiran continue from the body's Arabic numbering.

## Daftar Singkatan dan Lambang (Pedoman §IV.10)

- Single-spaced; 3 columns: (1) singkatan/lambang, (2) nama lengkap, (3) halaman pertama muncul.
- **Alphabetical order:** Latin uppercase first, then Latin lowercase, then Greek (in Greek alphabetical order).

## Other ITB-Specific Conventions

- **Hard-cover binding** (Sidang Promosi version): dark blue (Biru Dongker), Omega No. 10 paper, gold lettering. Not relevant during writing — only at final submission.
- **No TODO / FIXME / `…` placeholders in the final manuscript.** Lint warns; remove before submission.
- **All captions must be filled in.** *Lint-enforced fatal.*
- **Field integrity:** every REF field must resolve. After a round of edits, select all and press F9 so Daftar Isi, Daftar Gambar and Daftar Tabel repopulate; a field showing `Error! Reference source not found.` is a fatal.

## Editing the DOCX by Script

Text replacement over `word/document.xml` has three traps, each of which fails silently: a self-closing `<w:t xml:space="preserve"/>` matches a naive opening-tag regex and swallows the markup after it; the text uses non-breaking spaces (`Bab\u00a0III`, `HI\u00a036-D`); and several matches inside one run overwrite each other unless edits are recorded in original coordinates and applied back to front. Apply replacement pairs longest-first, and always diff the full `<m:oMath>` list before and after, since `<m:t>` math runs are a separate namespace that `<w:t>` edits never touch. Verify with: part count unchanged, every `.xml`/`.rels` still parses, no markup leaked into the extracted text.

## Sources

- [PEDOMAN PENULISAN DISERTASI DOKTOR — SPs ITB (April 2016)](https://multisite.itb.ac.id/sps/wp-content/uploads/sites/45/2015/12/PEDOMAN_PENULISAN_DISERTASI_DOKTOR_ITB.pdf)
- [Pedoman Tesis dan Disertasi — SPs ITB](https://sps.itb.ac.id/pedoman-tesis-dan-disertasi/)
- [Pedoman Penulisan Usulan/Proposal Penelitian Disertasi Doktor — S3 TMI ITB](https://s3tmi.fti.itb.ac.id/pedoman-penulisan-usulan-proposal-penelitian-disertasi-doktor/)
- [SPs ITB Citation Style Language (CSL)](https://itb-sps.github.io/csl/)
