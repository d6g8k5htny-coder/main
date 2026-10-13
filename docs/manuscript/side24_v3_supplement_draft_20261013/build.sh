#!/bin/sh
# Build the SIDE24 V3 supplement draft (TeX Live 2023 or later: amsart, amsplain, hyperref, enumitem, booktabs, tabularx).
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode supp_main.tex
bibtex supp_main
pdflatex -interaction=nonstopmode supp_main.tex
pdflatex -interaction=nonstopmode supp_main.tex
