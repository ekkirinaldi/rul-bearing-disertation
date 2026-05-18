# JETS Word Track — `jets-docs/`

**Target venue:** Journal of Engineering and Technological Sciences (JETS),
ITB Press, Q2 in Scopus.

**This folder contains the Word-native submission version of the paper.**

---

## What lives here

| File / Folder | Purpose |
|---|---|
| `paper.docx` | **The submission file.** Open and edit in Microsoft Word. |
| `references.bib` | BibTeX source (for Zotero / Mendeley citation insertion if needed). |
| `figures/` | All figures as PNG (self-contained, no LaTeX dependency). |

---

## Editing workflow

1. Open `paper.docx` in **Microsoft Word**.
2. Make edits, apply JETS styles from the Style picker (`Text`, `Heading 1`,
   `Heading 2`, `Reference`, `Figure`, `Table`).
3. Save in Word — the `.docx` is the canonical file.
4. Commit the updated `paper.docx` binary.

This file is the **only** source of truth for the Word track. Do **not**
regenerate it from LaTeX; edits made in Word will be lost if the seeding
script is re-run.

---

## Relationship to the LaTeX track

The LaTeX source lives in `../jets-mechanistic-interp/`.
The two tracks are **intentionally independent**:

- `../jets-mechanistic-interp/` → PDF via LuaLaTeX (for arXiv / dissertation).
- `jets-docs/` → `.docx` via Microsoft Word (for JETS submission portal).

Changes made in one track are **not** automatically reflected in the other.
When content changes occur (new experiment results, revised sections), update
both tracks manually.

---

## Re-seeding from LaTeX (emergency only)

If the Word file is accidentally lost, `_seed/seed_from_latex.py` can regenerate
it. Requirements: Docker running, `python-docx` installed.

```bash
# From repo root
python3 writings/journal-q2/jets-docs/_seed/seed_from_latex.py
```

The `_seed/` folder is in `.gitignore` and will not be committed; keep a local
copy if re-seeding is needed.

---

## JETS submission checklist

- [ ] Title, authors, affiliations correct on page 1.
- [ ] Abstract ≤ 350 words (current: ~235 words).
- [ ] Keywords 5–10 items, alphabetical, italic.
- [ ] All figures numbered Figure 1, 2, 3 … consecutively.
- [ ] All tables numbered Table 1, 2, 3 … consecutively.
- [ ] Equations numbered (1), (2), (3) … consecutively (not by section).
- [ ] References in APA author-year format, alphabetically sorted.
- [ ] Page count within JETS limit (check journal guidelines).
- [ ] Corresponding author email matches submission portal.
