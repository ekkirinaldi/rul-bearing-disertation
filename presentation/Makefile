SPEC ?= content/sidang-terbuka.yaml
OUT  ?= out/Sidang_Disertasi_Toto_Suharto.pptx
PY   ?= python3

.PHONY: help build lint preview png test inspect clean install

help:
	@echo "make build              render $(SPEC) -> $(OUT)"
	@echo "make lint               structural + overflow checks, no render"
	@echo "make preview            build, then convert to PDF (LibreOffice)"
	@echo "make png                build, then convert to per-slide PNG"
	@echo "make test               run the renderer test suite"
	@echo "make inspect FILE=x.ppt dump an existing deck's geometry and type"
	@echo "make clean              remove out/"
	@echo ""
	@echo "Override SPEC= and OUT= to build a different deck."

install:
	$(PY) -m pip install -r requirements.txt

build:
	$(PY) build.py build --spec $(SPEC) --out $(OUT) --strict

lint:
	$(PY) build.py lint --spec $(SPEC)

preview:
	$(PY) build.py preview --spec $(SPEC) --out $(OUT)

png: preview
	@command -v pdftoppm >/dev/null || { echo "pdftoppm not found (brew install poppler)"; exit 1; }
	pdftoppm -r 110 -png out/preview/$(notdir $(basename $(OUT))).pdf out/preview/slide
	@echo "PNG → out/preview/slide-NN.png"

test:
	$(PY) -m pytest tests -q

inspect:
	@test -n "$(FILE)" || { echo "usage: make inspect FILE=deck.pptx"; exit 1; }
	$(PY) build.py inspect "$(FILE)"

clean:
	rm -rf out __pycache__ deck/__pycache__ tests/__pycache__ .pytest_cache
