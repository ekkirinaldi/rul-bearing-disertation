# DOCX dissertation pipeline.
#
# The manuscript is edited directly in Word. The LaTeX tree that chapters were
# once regenerated from was removed in September 2026, and the regeneration
# targets went with it: running them against a hand-revised document would have
# discarded the Word edits anyway, which is why material has been inserted in
# place since June. What remains operates on the DOCX itself.
#
# Usage: make insert-dry / make insert / make lint / make master

TEMPLATE := assets/template.docx
FIGMAP   := assets/figure-map.tsv

CHAPTERS := $(foreach n,1 2 3 4 5 6,chapters/bab$(n).docx)
LAMPIRAN := $(foreach l,A B C D E F G,lampiran/lamp$(l).docx)

# In-place insertion into the hand-edited manuscript (see tools/insert_docx.py).
SPEC ?= inserts/v15/spec.yaml

insert-dry:
	python3 tools/insert_docx.py $(SPEC) --dry-run

insert:
	python3 tools/insert_docx.py $(SPEC) --force

lint:
	bash tools/lint_docx.sh chapters/*.docx

frontmatter: frontmatter/frontmatter.docx
frontmatter/frontmatter.docx: tools/build_frontmatter.py $(TEMPLATE)
	python3 tools/build_frontmatter.py

# Merges the existing chapter and lampiran files; it does not rebuild them.
master: disertasi.docx
disertasi.docx: frontmatter/frontmatter.docx tools/merge_master.py $(CHAPTERS) $(LAMPIRAN)
	python3 tools/merge_master.py

clean:
	rm -rf build/pre build/raw build/verify

.PHONY: insert insert-dry lint frontmatter master clean
