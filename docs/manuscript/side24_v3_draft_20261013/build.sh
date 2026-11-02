#!/bin/sh
# Build the SIDE24 V3 draft (TeX Live 2023 or later: amsart, amsplain, hyperref, enumitem, booktabs, tabularx).
#
#   sh build.sh                 build, copy the result to the distributed file name, verify SHA256SUMS
#   sh build.sh --update-sums   build, copy, and rewrite SHA256SUMS (for the author, when the sources changed)
#
# The build is byte-reproducible: SOURCE_DATE_EPOCH fixes the PDF's creation and modification dates and its
# trailer /ID (pdfTeX honours it with FORCE_SOURCE_DATE=1), so a rebuild from the same sources gives the same
# bytes and the SHA256SUMS line of the PDF is a real check. The epoch is 13 October 2026 00:00Z, the draft's date.
# TeX writes main.pdf; the distributed artifact named by README and SHA256SUMS is SIDE24_V3_DRAFT_20261013.pdf,
# so the copy below is part of the build (a revision that skipped it could ship a stale PDF with a passing manifest).
set -e
cd "$(dirname "$0")"
export SOURCE_DATE_EPOCH=1791849600 FORCE_SOURCE_DATE=1
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cp main.pdf SIDE24_V3_DRAFT_20261013.pdf
if [ "$1" = "--update-sums" ]; then
  sha256sum SIDE24_V3_DRAFT_20261013.pdf V3_CHANGE_LIST.md appA_records.tex build.sh main.tex refs.bib \
    s0_status.tex s1_field.tex s2_results.tex s3_proof.tex s4_scope.tex s5_discussion.tex s6_literature.tex \
    s7_disclosure.tex > SHA256SUMS
  echo "SHA256SUMS rewritten"
else
  sha256sum -c SHA256SUMS
fi
