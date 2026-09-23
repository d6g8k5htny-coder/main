# W2 Addendum (post-freeze errata; frozen artifacts untouched)

- **A1 (W13 finding F4 — check-count reconciliation).** The correct count is **113 ck() checks** (113 PASS lines, 0 FAIL, exit 0) in `run1.log`; the figure "114" printed in W2_DERIVATION.md (Appendix A) and in my report to the lead was a grep artifact: `grep -c PASS run1.log` also matches the final line "ALL CHECKS PASSED". The frozen derivation's phrase "114 checks, all PASS" should read "113 checks, all PASS". No mathematical content is affected.
- **A2 (W13 finding F5 — scipy dependency).** `W2_sanity.py` depends on **scipy 1.16.2** (`scipy.stats.norm` for the Gaussian pdf/cdf, `scipy.integrate.quad` for the 1-D quadrature cross-check in Part 2), in addition to numpy 2.2.5, mpmath 1.3.0, CPython 3.12 (environment already recorded in run1.log header context / Appendix A).

*Filed by W2 in response to lead message [19fb9293-5c52-85f2-8000-095a601387af]; W2_sanity.py, run1.log, and W2_DERIVATION.md (body-sha256 26e2d1dd3b36f1f7d3cc71f519a156dd5dee241854c2ba9bcd0bce36626ffc9f) are unchanged.*
