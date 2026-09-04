# presentation/ — Dissertation Deck Builder

A build service that renders the doctoral defence presentation from a YAML
content file onto the approved ITB template. The `.pptx` is a build artifact:
**edit the YAML, never the deck**.

The design system (palette, type scale, grid, component geometry) was measured
from the approved reference deck in [`reference/`](reference/) and
frozen in [`deck/theme.py`](deck/theme.py), so regenerated decks stay visually
identical to the version Pak Toto approved.

**Manuscript of record: the V14 DOCX** at the top of this directory
(`V14 PERAWATAN PREDIKTIF … .docx`, gitignored binary). Every number, figure
citation, and claim in the deck traces to it — not to the older
`dissertation-docx/` tree, whose numbers conflict with V14 in places.

## Deck structure (52 slides)

The main deck (40 slides) is organised around V14's **four rumusan masalah**:
an intro arc (latar belakang, istilah inti untuk penguji non-informatika,
kesenjangan, the RM spine slide reprising Tabel I.3, kerangka, dataset), then
four acts — one per RM — each opened by a `section` divider restating the
question and closed by a "Jawaban RM-N" slide with its epistemic status
(empiris / empiris-kualitatif / konseptual). Bridging slides for the
industrial-engineering panel sit just-in-time: ML vs DL, cara membaca RUL,
analogi stabilo for SHAP, kamus konsep for the SAE. Twelve backup slides
("Cadangan") follow the closing slide for the Q&A session.

Every algorithm carries a **paper-style block diagram** rendered by
`make diagrams` (see below): the classic-ML trio, one architecture slide per
RUL backbone (Mamba-xLSTM-Net, N-BEATS-xLSTM-RUL, SparseGate-TCN-RUL), the
sinyal→FSM pipeline, the Top-k SAE, the SAE→BPFx procedure, WDCNN (backup),
and the streaming inference engine (backup).

## Quick start

```bash
cd presentation
make install          # python-pptx, PyYAML, lxml, Pillow
make v14-assets       # extract deck figures from the V14 DOCX media
make diagrams         # render the paper-style algorithm diagrams (matplotlib)
make derived-assets   # crop IMS/CWRU-free versions of the result figures
make build            # content/sidang-terbuka.yaml -> out/…​.pptx
make preview          # build + PDF via LibreOffice
make png              # build + one PNG per slide (needs poppler)
make test             # renderer, layout and manuscript-provenance tests
```

Build a different deck without touching the defaults:

```bash
make build SPEC=content/seminar-kemajuan.yaml OUT=out/Seminar.pptx
```

## Layout of the service

```
presentation/
├── V14 PERAWATAN PREDIKTIF ….docx   ★ manuscript of record (gitignored binary)
├── build.py                    CLI: build · lint · preview · inspect
├── Makefile
├── assets/
│   ├── logo-itb.png            ITB seal used on the cover
│   ├── diagrams/               algorithm diagrams rendered by make diagrams
│   ├── derived/                IMS/CWRU-free crops of manuscript result figures
│   └── v14/                    figures extracted from the V14 DOCX media
├── content/
│   └── sidang-terbuka.yaml     ★ the deck content — this is what you edit
├── reference/
│   └── Sidang_Disertasi_…_v2.pptx   approved template the design was derived from
├── tools/
│   ├── extract_v14_media.py    pulls figures out of the V14 DOCX by media index
│   ├── render_diagrams.py      draws the 9 algorithm diagrams (matplotlib)
│   └── derive_assets.py        crops IMS/CWRU panels out of the result figures
├── deck/
│   ├── theme.py                design tokens: palette, type scale, grid
│   ├── shapes.py               primitives + inline markup + text measurement
│   ├── blocks.py               the block vocabulary (see below)
│   ├── layout.py               vertical flow: measure, distribute, place
│   ├── charts.py               native PowerPoint charts, styled
│   ├── slides.py               slide layouts: cover · content · section · closing
│   └── builder.py              spec loading, linting, rendering
├── tests/
│   ├── test_deck.py            renderer + layout regressions
│   ├── test_diagrams.py        diagram set complete, sized, and referenced
│   └── test_provenance.py      every slide figure must exist in the V14 manuscript
└── out/                        build artifacts (git-ignored)
```

## Authoring a slide

Every content slide is an eyebrow, a title, and a list of blocks that flow
top-down inside the body band (y = 1.60 … 6.86 in):

```yaml
- layout: content
  eyebrow: Hasil Prognostik · Bab V
  title: Tiga Backbone RUL dan Pemenang Lintas Dataset
  blocks:
    - type: cards
      h: 1.50
      items:
        - title: Mamba-xLSTM-Net
          body: "**898K parameter**\nSSM selektif Mamba dan memori matrix-LSTM"
    - type: table
      flex: true
      columns: [Dataset, Backbone Pemenang, RMSE]
      rows:
        - ["PHM2012", "SparseGate-TCN-RUL", "0,226 ± 0,030"]
```

### Inline markup

| Markup | Renders as |
|---|---|
| `**tebal**` | bold, in the accent colour (navy on light, white on dark) |
| `*miring*` | italic |
| `***tebal miring***` | both |
| `~~redup~~` | the muted secondary colour |

### Layout controls (any block)

| Key | Effect |
|---|---|
| `h` | fixed height in inches; skips measurement |
| `gap` | space *before* this block (default 0.16 in) |
| `flex` | absorb the leftover vertical space; a number sets the share |
| `pin: bottom` | stick to the bottom of the region |
| `tone` | surface: `light` · `alt` · `white` · `dark` · `accent` · `cream` · `plain` |

### Block vocabulary

| `type` | What it draws |
|---|---|
| `lead` / `text` | a paragraph (or list of paragraphs) of marked-up prose |
| `heading` | small section label, optional `sub:` and `rule: true` |
| `cards` | a row (or grid, via `columns:`) of panels — `badge`, `key`, `tag`, `title`, `body` |
| `stats` | big-number cards — `value`, `label`, `note`; `widths:` for unequal columns |
| `metrics` | badge + title + `hint` + explanation, laid out in columns |
| `banner` | full-width callout strip, optional left `label:` column |
| `bullets` | square-marker list |
| `numbered` | circled-number list |
| `panel` | titled container holding nested `blocks:`; `band: true` for a navy header |
| `columns` | side-by-side regions, each with its own nested `blocks:` |
| `table` | styled table: navy header, zebra body, hairline borders |
| `chart` | native bar chart; `labels:` sets literal text labels |
| `tiers` | the three-tier architecture blueprint with arrows |
| `agenda` | numbered agenda grid |
| `glossary` | columns of term / definition pairs |
| `refs` | columns of grouped references |
| `figure` | image via `image:`, or a captioned placeholder |
| `maprows` | full-width `key → claim → evidence` rows |
| `kvrows` | `LABEL | value` rows |
| `source` | the small "Sumber dan metode:" credit line |
| `spacer` / `rule` | vertical space, horizontal rule |

To drop a real figure in place of a placeholder, point `image:` at a file
(paths resolve relative to the YAML):

```yaml
- type: figure
  image: ../../dissertation-docx/assets/figures/bab5/hitrate_panel.png
```

## Linting

`make lint` runs before every build and fails `--strict` builds on:

- unknown layout or block types, missing titles;
- **content that overflows the body band**, reported in inches so you know
  exactly how much to trim.

`make test` additionally asserts that no shape escapes the canvas, that every
content slide carries a footer and page number, that only theme colours appear
in the deck, that tables keep the navy header, and that the ITB language rules
hold (no `&` standing in for `dan`, decimal commas on percentages).

### Provenance guard

`tests/test_provenance.py` is the important one. It reads the **V14 DOCX** and
asserts that **every headline figure on a slide actually appears in the
manuscript** — accuracies, RMSE values, hit-rates, thresholds, acquisition
counts, citations (90 claims). Two further tests assert that every
`Gambar X.N` / `Tabel X.N` cited on a slide exists in V14, and that no
pre-V14 reference (Lampiran D.5, Gambar D.2/D.4) survives. A claim that
drifts from the dissertation fails the build rather than reaching the defence.

The older `dissertation-docx/` tree is deliberately **not** in the corpus:
its numbers conflict with V14 (XJTU-SY 15 rekaman vs 10 bearing, IMS dropped,
SKF moved from Lampiran D into Subbab IV.15 / V.5.3, figures renumbered), and
a claim passing via the stale tree would defeat the guard.

It skips cleanly when the V14 DOCX is absent. When a number legitimately
changes in the manuscript, update the YAML and the matching entry in `CLAIMS`.

## Algorithm diagrams

`tools/render_diagrams.py` (`make diagrams`) draws one PNG per method into
`assets/diagrams/`, in the block-diagram idiom of Jiang dkk. (2026),
*Sensors* **26**:1578: rounded colour-coded blocks, dashed "n×" repeat
containers, zoom-in panels for cell internals, and a legend row per figure.
One master palette maps component function to colour across all nine figures
(kuning masukan · ungu proyeksi/konvolusi · biru pemodelan sekuens · hijau
memori/state · merah muda gating · hijau tua keluaran · emas fitur aktif),
with the deck navy as ink. Every number drawn is asserted against the V14
text by `tests/test_provenance.py`, and the slides that carry a redrawn
figure cite it as "Digambar ulang berdasarkan Gambar X.N". Iterate with
`python3 tools/render_diagrams.py --only <name>`.

## Re-deriving the template

If the approved template ever changes, dump its geometry and type and update
[`deck/theme.py`](deck/theme.py) to match:

```bash
make inspect FILE=reference/Sidang_Disertasi_Toto_Suharto_v2.pptx
```

## Content provenance

Figures and claims in `content/sidang-terbuka.yaml` are sourced from the
**V14 manuscript**, not re-derived here:

| Slide content | Source in V14 |
|---|---|
| Empat RM verbatim, Tabel I.3 lattice (Tujuan–RM–Kesenjangan–Novelti–Bab) | Subbab I.2, I.3, I.5 |
| Bearing 40–50% (Nandi dkk., 2005), Making Indonesia 4.0, PT SKF Cikarang | Bab I |
| Benchmark CWRU (DT 92,40 → WDCNN 99,87%), Ball_014→IR_014, HI 36-D | Bab IV |
| FSM 0,940 / 0,216 / 17,6%, ablasi BatchNorm +133% / −3,60 pp, 42,67 ms | Subbab IV.6–IV.13 |
| RUL winners (SparseGate 0,226 PHM2012; Mamba 0,213 XJTU-SY), PHM Score | Tabel V.2/V.4/V.5 |
| SAE (128→1.024, k=51), hit-rate, kontrol negatif, Bonferroni 0,004, sweep k=205 | Subbab V.2–V.4, V.6–V.9 |
| SKF: IV.15 transfer diagnostik; V.5.3 streaming (158/78 akuisisi, EoL ±1 hari) | Subbab IV.15, V.5.3, V.5.4 |
| Keterbatasan (6) dan rekomendasi (7) | Subbab VI.4, VI.5 |

When those numbers change in the manuscript, update the YAML **and** the
matching entry in `tests/test_provenance.py`, then rebuild.

### V14-internal inconsistencies (flagged for the manuscript, not fixable here)

The deck follows V14's Bab-body values wherever V14 disagrees with itself:

| Inconsistency in V14 | Deck's choice |
|---|---|
| "tiga permasalahan" leftovers (I.1, I.6.1) vs four RMs in I.2 | four RMs |
| LR accuracy 94,3% (Tabel IV.1) vs 94,4% (Gambar D.2 caption) | 94,3% |
| DT/RF/XGB 92,40/95,8/97,1 (Bab IV) vs 94,0/94,9/95,3 (appendix captions) | Bab IV values |
| Var-C 96,13% (Tabel IV.5) vs 92,0% (Gambar F.2 caption) | 96,13% |
| WDCNN best epoch 54 (body) vs 47 (Gambar IV.6 caption) | epoch not cited |
| "empat dataset publik" (III.2) vs "tiga dataset benchmark dan satu sumber industri" (I.8) | tiga + SKF |
| Gambar III.1 / V.15 / V.17 artwork still shows IMS and CWRU panels; Gambar IV.8 mismatches its caption | deck uses IMS/CWRU-free crops from `make derived-assets` |
| Subbab V still keeps CWRU in the SAE narrative (Tabel II.3 note, Gambar V.15/V.17 captions, "12 uji primer atas 4 dataset", "empat dataset publik") | deck drops CWRU from the hit-rate section per the review decision; the 12-uji / p < 0,004 arithmetic stays as V14 states it |

IMS and CWRU are out of the deck's SAE/hit-rate section (guarded by
`test_ims_and_cwru_out_of_the_sae_section`). The source artwork in
`dissertation-docx/assets` still carries their panels, so `make derived-assets`
crops presentation-local copies into `assets/derived/`; regenerating the
figures in the manuscript pipeline makes those crops obsolete. The
negative-controls chart has no PHM2012/XJTU-SY panels at all, so the deck now
states the control results as text.
