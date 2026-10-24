#!/bin/sh
# Build the SIDE24 V3 draft (TeX Live 2023 or later: amsart, amsplain, hyperref, enumitem, booktabs, tabularx).
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
