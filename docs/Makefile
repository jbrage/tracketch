# Minimal Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Build PDF and copy it to docs/ for easy access.
.PHONY: latexpdf
latexpdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@if [ -f "$(BUILDDIR)/latex/tracketch.pdf" ]; then \
		cp "$(BUILDDIR)/latex/tracketch.pdf" "$(SOURCEDIR)/tracketch.pdf"; \
		echo "Copied $(BUILDDIR)/latex/tracketch.pdf -> $(SOURCEDIR)/tracketch.pdf"; \
	fi

# Compatibility alias: `pdflatex` is not a Sphinx builder name.
.PHONY: pdflatex
pdflatex:
	@$(MAKE) latexpdf

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
