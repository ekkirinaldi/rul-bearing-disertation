# ============================================================
#  Disertasi Doktor ITB — Makefile
#  Aturan lengkap: lihat .cursor/rules/14-build-workflow.mdc
# ============================================================

MAIN      := disertasi
TEX       := $(MAIN).tex
PDF       := $(MAIN).pdf
BUILDDIR  := build
LATEXMK   := latexmk
LATEXMK_FLAGS := -outdir=$(BUILDDIR) -interaction=nonstopmode -halt-on-error -file-line-error

CHAPTERS  := $(wildcard chapters/*.tex)
LAMPIRAN  := $(wildcard lampiran/*.tex)
ALL_TEX   := $(TEX) $(CHAPTERS) $(LAMPIRAN)

.PHONY: all build watch clean distclean wordcount check spell lint pre-submit help

all: build

build:    ## Compile sekali (lualatex + biber + lualatex x2)
	@mkdir -p $(BUILDDIR)
	$(LATEXMK) $(LATEXMK_FLAGS) $(TEX)
	@cp $(BUILDDIR)/$(PDF) ./$(PDF)
	@echo "PDF tersedia di ./$(PDF)"

watch:    ## Compile otomatis saat file berubah (latexmk -pvc)
	@mkdir -p $(BUILDDIR)
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(TEX)

clean:    ## Hapus file build menengah, simpan PDF
	$(LATEXMK) -outdir=$(BUILDDIR) -c $(TEX)

distclean:## Hapus seluruh direktori build dan PDF
	$(LATEXMK) -outdir=$(BUILDDIR) -C $(TEX)
	rm -rf $(BUILDDIR)
	rm -f $(PDF)

wordcount:## Hitung jumlah kata per bab
	@echo "=== Word count per chapter ==="
	@for f in $(CHAPTERS); do \
		echo "--- $$f ---"; \
		texcount -inc -1 -sum "$$f" 2>/dev/null || true; \
	done
	@echo "=== Total (estimasi) ==="
	@texcount -inc -1 -sum -merge $(TEX) 2>/dev/null || true

check:    ## Jalankan semua quality check (chktex + regex lint)
	@echo "=== chktex ==="
	@chktex -q -n8 -n24 $(ALL_TEX) || true
	@$(MAKE) lint

lint:     ## Regex check: forbidden phrases, format issues
	@bash scripts/lint-itb.sh

spell:    ## Spell check Indonesian
	@for f in $(filter-out chapters/00-abstract-en.tex,$(CHAPTERS)); do \
		echo "--- spell: $$f ---"; \
		hunspell -d id_ID -t -l "$$f" 2>/dev/null | sort -u | head -20 || true; \
	done

pre-submit: clean build wordcount check ## Jalankan SEBELUM kirim ke promotor / sidang
	@echo ""
	@echo "============================================================"
	@echo "  Pre-submit checks selesai."
	@echo "  Periksa output di atas, lalu inspeksi visual PDF (margin,"
	@echo "  daftar isi, bibliografi, sampul)."
	@echo "  Lihat checklist lengkap di .cursor/rules/14-build-workflow.mdc"
	@echo "============================================================"

help:     ## Tampilkan daftar target ini
	@echo "Disertasi Doktor ITB — Makefile targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
