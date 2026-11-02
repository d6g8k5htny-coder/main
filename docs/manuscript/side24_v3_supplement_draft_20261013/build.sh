#!/bin/sh
# Build the SIDE24 V3 supplement draft (TeX Live 2023 or later: amsart, amsplain, hyperref, enumitem, booktabs, tabularx).
#
#   sh build.sh                 build, copy the result to the distributed file name, verify SHA256SUMS
#   sh build.sh --update-sums   build, copy, and rewrite SHA256SUMS (for the author, when the sources changed)
#
# The build is byte-reproducible: SOURCE_DATE_EPOCH fixes the PDF's creation and modification dates and its
# trailer /ID (pdfTeX honours it with FORCE_SOURCE_DATE=1), so a rebuild from the same sources gives the same
# bytes and the SHA256SUMS line of the PDF is a real check. The epoch is 13 October 2026 00:00Z, the draft's date.
# TeX writes supp_main.pdf; the distributed artifact named by README and SHA256SUMS is
# SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf, so the copy below is part of the build.
set -e
cd "$(dirname "$0")"
export SOURCE_DATE_EPOCH=1791849600 FORCE_SOURCE_DATE=1
pdflatex -interaction=nonstopmode supp_main.tex
bibtex supp_main
pdflatex -interaction=nonstopmode supp_main.tex
pdflatex -interaction=nonstopmode supp_main.tex
cp supp_main.pdf SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf
if [ "$1" = "--update-sums" ]; then
  sha256sum CONCORDANCE.md SIDE24_V3_SUPPLEMENT_DRAFT_20261013.pdf build.sh checks/dep_graph_check.out.txt \
    checks/dep_graph_check.py checks/s10_s11_fold_check.out.txt checks/s10_s11_fold_check.py \
    checks/s12_derivations_check.out.txt checks/s12_derivations_check.py checks/s12_parent_check.out.txt \
    checks/s12_parent_check.py checks/s1s2_verify_check.out.txt checks/s1s2_verify_check.py \
    checks/s5_envelope_check.out.txt checks/s5_envelope_check.py checks/s5_factor_check.out.txt \
    checks/s5_factor_check.py checks/s5_finite_scale_check.out.txt checks/s5_finite_scale_check.py \
    checks/s6_limit_check.out.txt checks/s6_limit_check.py checks/s8_capK_check.out.txt \
    checks/s8_capK_check.py checks/s9_c2_d2_check.out.txt checks/s9_c2_d2_check.py \
    checks/test_dep_graph_check.py refs.bib supp_main.tex supp_refs_s12.bib supp_refs_s8s9.bib \
    supp_s0_status.tex supp_s10_third.tex supp_s11_ctwo.tex supp_s12_parent.tex supp_s13_cap.tex \
    supp_s1_setting.tex supp_s2_contact.tex supp_s3_coefficient.tex supp_s4_correction.tex \
    supp_s5_selection.tex supp_s6_limit.tex supp_s7_rate.tex supp_s8_remainder.tex supp_s9_cusp.tex \
    writer_originals/supp_s12_parent.rev2orig.tex writer_originals/supp_s5_selection.w4orig.tex > SHA256SUMS
  echo "SHA256SUMS rewritten"
else
  sha256sum -c SHA256SUMS
fi
