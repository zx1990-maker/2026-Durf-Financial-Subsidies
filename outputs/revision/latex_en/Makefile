# =============================================================================
#  English paper — LaTeX build script
#  Requires: xelatex (with booktabs, threeparttable, tabularx, makecell, etc.)
#  Usage: make            —— build main_en.pdf (runs xelatex twice for refs)
#         make clean      —— remove intermediate files
# =============================================================================

MAIN = main_en
PDF  = $(MAIN).pdf

all: $(PDF)

$(PDF): $(MAIN).tex $(wildcard figures/*.pdf)
	xelatex -interaction=nonstopmode $(MAIN).tex
	xelatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.fls *.fdb_latexmk

.PHONY: all clean
