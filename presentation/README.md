# presentation/ — Dissertation Deck Builder

A build service that renders the doctoral defence presentation from a YAML
content file onto the approved ITB template. The `.pptx` is a build artifact:
**edit the YAML, never the deck**.

The design system (palette, type scale, grid, component geometry) was measured
from the approved reference deck in [`reference/`](reference/) and
frozen in [`deck/theme.py`](deck/theme.py), so regenerated decks stay visually
identical to the version Pak Toto approved.

## Quick start

```bash
cd presentation
make install          # python-pptx, PyYAML, lxml
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
├── build.py                    CLI: build · lint · preview · inspect
├── Makefile
├── assets/
│   └── logo-itb.png            ITB seal used on the cover
├── content/
│   └── sidang-terbuka.yaml     ★ the deck content — this is what you edit
├── reference/
│   └── Sidang_Disertasi_…_v2.pptx   approved template the design was derived from
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
│   └── test_provenance.py      every slide figure must exist in the manuscript
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

`tests/test_provenance.py` is the important one. It reads the DOCX chapters,
lampiran and frontmatter under `dissertation-docx/` and asserts that **every
headline figure on a slide actually appears in the manuscript** — accuracies,
RMSE values, hit-rates, thresholds, acquisition counts, citations. A claim that
drifts from the dissertation fails the build rather than reaching the defence.

It skips cleanly when the manuscript tree is absent. When a number legitimately
changes in the manuscript, update the YAML and the matching entry in `CLAIMS`.

## Re-deriving the template

If the approved template ever changes, dump its geometry and type and update
[`deck/theme.py`](deck/theme.py) to match:

```bash
make inspect FILE=reference/Sidang_Disertasi_Toto_Suharto_v2.pptx
```

## Content provenance

Figures and claims in `content/sidang-terbuka.yaml` are sourced from the
manuscript, not re-derived here:

| Slide content | Source |
|---|---|
| Bearing 40–50%, WDCNN 99,87%, FSM 0,940 / 0,216 / 0,735 | `writings/disertation/chapters/00-abstrak-id.tex` |
| RUL winners per dataset (PHM2012 / XJTU-SY / IMS) | `chapters/05-prognostik.tex`, Tabel V.9 |
| BPFx hit-rate across four datasets, IMS partial mismatch, CWRU underpowered | `chapters/05-prognostik.tex`, Tabel V.12 + §V.9 |
| Chapter and novelty structure (N1–N5, RM-1…3) | `writings/dissertation-outline.md` |
| SKF streaming validation (158/78 acquisitions, EoL, fusion gate, attributions) | `lampiran/lampD.docx`, Subbab D.5 |

When those numbers change in the manuscript, update the YAML **and** the
matching entry in `tests/test_provenance.py`, then rebuild.

### Corrections found during the audit

The first draft inherited several figures from the reference deck that the
manuscript does not support. These are fixed and now guarded by tests:

| Was | Is | Source |
|---|---|---|
| 40–50% cited to Nectoux dkk. (2012) | Lei (2018) | Bab I |
| XJTU-SY "15 bearing" | 10 bearing | Bab III |
| IMS cited to Lee dkk. (2007) | Qiu dkk. (2006), Rexnord ZA-2115, empat bearing | Bab III |
| Critical threshold "< 15%" | peringatan 40%, kritis 20% | Lampiran D.5.1 |
| "Ball-14 terbaca sebagai IR-14" · "27 kesalahan SVM → 1" | not in the manuscript; replaced with the Bab IV findings (3,4 pp bagging, 1,3 pp boosting, 2,1 pp non-linearity) | Bab IV |
| Receptive field "42,7 ms" | 42,67 ms | Bab IV |
| "EoL terpaut ~1 hari dari kerusakan aktual" | EoL is *anchored* to the observed failure; residual 34 min (NDE) / 40 min (DE) vs zero | Lampiran D.5 |
| SKF panels "Gambar V.3–V.6" | Gambar D.2–D.5 | Lampiran D.5 |
| Ablation accuracies presented without context | flagged as the 50-epoch ablation protocol, subset 30/50 | Bab IV |
