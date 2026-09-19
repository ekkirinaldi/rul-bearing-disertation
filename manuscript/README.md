# Manuscript

`V16-disertasi.docx` is the current manuscript. It is edited in Word on the ITB template (`assets/template.docx`), and each version sent to the promotor is saved as the next `V<N>`. Earlier versions, backups and the circulated sidang deck are in `versions/`, which is kept on disk and not in git. Each DOCX is about 18 MB and does not delta-compress, so only the current version is tracked. Add the next one to `.gitignore` deliberately when it replaces V16.

| Path | |
|---|---|
| `RULES.md` | Word mechanics: styles, captions, REF and SEQ fields, citation format |
| `assets/` | ITB template, citation style, the figures placed in the manuscript, `figure-map.tsv` |
| `inserts/` | Insertion specs; `v15/spec.yaml` ported the deck's domain material into V14 to make V15 |
| `tools/` | `insert_docx.py` (inserts styled paragraphs, figures and tables in place), `lint_docx.sh` (ITB checks) |

The prose rules are in [../docs/writing-guide.md](../docs/writing-guide.md).

```bash
make lint          # fix every [FATAL]; review each [WARN]
make insert-dry    # preview SPEC (default inserts/v15/spec.yaml)
make insert        # apply it
```

A spec anchors each edit on a unique paragraph of the source version and writes a new file, so the Word edits in the source are never regenerated or lost. `presentation/tests/test_insert_docx.py` checks that every source paragraph survives, in order, and that caption numbers and cross-references still agree.

After editing in Word, select all and press F9 once so the lists of contents, figures and tables pick up new captions.
