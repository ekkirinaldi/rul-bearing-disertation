# DOCX dissertation build pipeline
# Per-chapter: preprocess (LaTeX -> pandoc-ready) -> pandoc -> restyle (ITB format)
# Usage: make bab6 / make verify-bab6 / make all-chapters

TEXDIR   := ../writings/disertation
AUX      := $(TEXDIR)/build/disertasi.aux
BIB      := $(TEXDIR)/references.bib
CSL      := assets/itb-sps.csl
TEMPLATE := assets/template.docx
FIGMAP   := assets/figure-map.tsv
SOFFICE  := /Applications/LibreOffice.app/Contents/MacOS/soffice

PANDOC_FLAGS := -f latex -t docx --citeproc --csl $(CSL) --bibliography $(BIB) \
  --metadata lang=id-ID --metadata suppress-bibliography=true \
  --resource-path .:assets/figures --reference-doc $(TEMPLATE) \
  --lua-filter assets/color-spans.lua

# chapter metadata: number, source tex, chapter label
BAB1_SRC := $(TEXDIR)/chapters/01-pendahuluan.tex
BAB2_SRC := $(TEXDIR)/chapters/02-tinjauan-pustaka.tex
BAB3_SRC := $(TEXDIR)/chapters/03-metodologi-umum.tex
BAB4_SRC := $(TEXDIR)/chapters/04-diagnostik.tex
BAB5_SRC := $(TEXDIR)/chapters/05-prognostik.tex
BAB6_SRC := $(TEXDIR)/chapters/06-kesimpulan.tex
BAB1_LBL := bab:pendahuluan
BAB2_LBL := bab:tinjauan-pustaka
BAB3_LBL := bab:metodologi-umum
BAB4_LBL := bab:diagnostik
BAB5_LBL := bab:prognostik
BAB6_LBL := bab:kesimpulan

define CHAPTER_RULE
bab$(1): chapters/bab$(1).docx
chapters/bab$(1).docx: $$(BAB$(1)_SRC) tools/preprocess_tex.py tools/restyle.py $(FIGMAP)
	@mkdir -p build/pre build/raw chapters
	python3 tools/preprocess_tex.py $$(BAB$(1)_SRC) build/pre/bab$(1).tex \
	  --aux $(AUX) --figmap $(FIGMAP) --bab $(1)
	pandoc build/pre/bab$(1).tex $(PANDOC_FLAGS) -o build/raw/bab$(1).docx
	python3 tools/restyle.py build/raw/bab$(1).docx chapters/bab$(1).docx \
	  --aux $(AUX) --figmap $(FIGMAP) --bab $(1) --chapter-label $$(BAB$(1)_LBL) \
	  --tblhints build/pre/bab$(1).tex.tblhints.json

verify-bab$(1): chapters/bab$(1).docx
	bash tools/verify_chapter.sh chapters/bab$(1).docx $$(BAB$(1)_SRC)
endef

$(foreach n,1 2 3 4 5 6,$(eval $(call CHAPTER_RULE,$(n))))

all-chapters: bab1 bab2 bab3 bab4 bab5 bab6

# lampiran: letter, source tex, chapter label (page start from aux)
LAMPA_SRC := $(TEXDIR)/lampiran/A-detail-fitur.tex
LAMPB_SRC := $(TEXDIR)/lampiran/B-hyperparameter-baseline.tex
LAMPC_SRC := $(TEXDIR)/lampiran/C-kode-implementasi.tex
LAMPD_SRC := $(TEXDIR)/lampiran/D-klasifikasi-industri.tex
LAMPE_SRC := $(TEXDIR)/lampiran/E-svm-lr.tex
LAMPF_SRC := $(TEXDIR)/lampiran/F-tree.tex
LAMPG_SRC := $(TEXDIR)/lampiran/G-wdcnn-fsm.tex
LAMPA_LBL := lamp:fitur
LAMPB_LBL := lamp:hparam
LAMPC_LBL := lamp:kode
LAMPD_LBL := lamp:klasifikasi_industri
LAMPE_LBL := lamp:svm_lr
LAMPF_LBL := lamp:tree
LAMPG_LBL := lamp:wdcnn_fsm

define LAMP_RULE
lamp$(1): lampiran/lamp$(1).docx
lampiran/lamp$(1).docx: $$(LAMP$(1)_SRC) tools/preprocess_tex.py tools/restyle.py $(FIGMAP)
	@mkdir -p build/pre build/raw lampiran
	python3 tools/preprocess_tex.py $$(LAMP$(1)_SRC) build/pre/lamp$(1).tex \
	  --aux $(AUX) --figmap $(FIGMAP) --bab 0
	pandoc build/pre/lamp$(1).tex $(PANDOC_FLAGS) -o build/raw/lamp$(1).docx
	python3 tools/restyle.py build/raw/lamp$(1).docx lampiran/lamp$(1).docx \
	  --aux $(AUX) --figmap $(FIGMAP) --bab 0 --lampiran $(1) \
	  --chapter-label $$(LAMP$(1)_LBL) --tblhints build/pre/lamp$(1).tex.tblhints.json

verify-lamp$(1): lampiran/lamp$(1).docx
	bash tools/verify_chapter.sh lampiran/lamp$(1).docx $$(LAMP$(1)_SRC)
endef

$(foreach l,A B C D E F G,$(eval $(call LAMP_RULE,$(l))))

all-lampiran: lampA lampB lampC lampD lampE lampF lampG

frontmatter: frontmatter/frontmatter.docx
frontmatter/frontmatter.docx: tools/build_frontmatter.py $(TEMPLATE)
	python3 tools/build_frontmatter.py

master: disertasi.docx
disertasi.docx: frontmatter/frontmatter.docx tools/merge_master.py \
  $(foreach n,1 2 3 4 5 6,chapters/bab$(n).docx) \
  $(foreach l,A B C D E F G,lampiran/lamp$(l).docx)
	python3 tools/merge_master.py

lint:
	bash tools/lint_docx.sh chapters/*.docx

clean:
	rm -rf build/pre build/raw build/verify

.PHONY: all-chapters all-lampiran frontmatter master lint clean \
  $(foreach n,1 2 3 4 5 6,bab$(n) verify-bab$(n)) \
  $(foreach l,A B C D E F G,lamp$(l) verify-lamp$(l))
