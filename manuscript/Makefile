# Manuscript tooling. The manuscript is edited in Word; these targets insert
# prepared material into it and check it against the ITB rules.
#
#   make insert-dry   preview an insertion spec without writing anything
#   make insert       apply it, writing the next version
#   make lint         ITB format and prose lint over the current manuscript

MANUSCRIPT := V16-disertasi.docx
SPEC       ?= inserts/v15/spec.yaml

insert-dry:
	python3 tools/insert_docx.py $(SPEC) --dry-run

insert:
	python3 tools/insert_docx.py $(SPEC) --force

lint:
	bash tools/lint_docx.sh $(MANUSCRIPT)

.PHONY: insert-dry insert lint
