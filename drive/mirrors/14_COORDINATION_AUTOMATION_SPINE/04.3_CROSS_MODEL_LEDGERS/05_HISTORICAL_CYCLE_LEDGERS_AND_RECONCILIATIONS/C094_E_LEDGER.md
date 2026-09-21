# C094 E-LEDGER — PASS 1

**Policy:** append-only. A failure, near-loss, or blocked import is preserved
with evidence, cause, downstream effect, and terminal disposition.

---

## E-C094-1 — Prescribed C093 verification order mutates a manifested file

**Evidence:** `C094_FRESH_STATE_VERIFICATION.json`.

**Observed sequence:**

```text
verify before audit: PASS
audit:               PASS, but rewrites Q0_C093_AUDIT_REPORT.json
verify after audit:  FAIL, hash and size mismatch
```

**Cause:** `audit_q0_c093_release.py` writes its current shallow audit result
over the deep-execution audit report included in the immutable manifest.

**Downstream impact:** release-pipeline reproducibility only; no mathematical
claim changes.

**Disposition:** `ERRATUM`. In every C094+ release, immutable-manifest
verification precedes writers, or the audit writes to a new versioned output
path.

---

## E-C094-2 — UB-G upper coefficient rounded in the wrong direction

**Evidence:** `C094_UBG_ADJUDICATION.json`.

**Finding:**

\[
(0.66+2.82)/0.80=4.35,
\]

while the frozen display was \(4.3\). The unrounded near/exterior assembly is
approximately \(4.3444138731\).

**Cause:** downward decimal rounding of an upper bound, with no displayed
offset certificate.

**Downstream impact:** `UB_G`, the two-sided rate display, endpoint arithmetic,
claim-language files, deprecation aliases, and both theorem hashes.

**Disposition:** `CORRECTION`. Replacement upper coefficient: \(4.35\).

---

## E-C094-3 — The \(0.8411\) lower coefficient lost its dimensions

**Evidence:** `C094_LOWER_COEFFICIENT_ADJUDICATION.json`.

**Finding:** the source writes

\[
1-q\ge C^*(r)r^3AO(r)(1-O(r^3)),
\qquad C^*(r)\ge0.8411.
\]

It does not license

\[
1-q\ge0.8411r^3
\]

without propagating the subunit AO and Bonferroni factors.

**Cause:** a first-moment coefficient was promoted to a final probability
coefficient.

**Downstream impact:** `LOWER_RATE_PG`, `RATE_PROGRAM_GRADE`, every “two-sided
closed theorem” wording.

**Disposition:** `CORRECTION`. Preserve \(C^*(r)\ge0.8411\); move finite lower
probability claims to Q0-SHARP.

---

## E-C094-4 — C091 same-name files lacked per-file provenance rows

**Evidence:** `C094_C091_C093_PROVENANCE.json`.

**Finding:** ten same-name artifacts had changed or release-specific bytes:
four portability edits, one control-character repair, one diagnostic
re-execution, and four identities; seven complete C091 artifacts were
archive-only.

**Cause:** C093 documented a portability pass globally but did not provide a
per-file same-name delta ledger.

**Downstream impact:** provenance clarity only.

**Disposition:** `ERRATUM-CLOSED` by the C094 table.

---

## E-C094-5 — Randomized orthant diagnostics changed at identical byte count

**Evidence:** same-size `gaussian_corridor_certificate_report.json` comparison.

**Finding:** twelve scalar leaves changed, exclusively
`orthant_probability_diagnostic` and `orthant_over_naive_product`. Maximum
relative change was approximately \(4.18\times10^{-7}\). The adjudication,
Chernoff bounds, covariance data, counterexample labels, and kill decision
were byte-equivalent as JSON values.

**Cause:** numerical multivariate-normal CDF diagnostics were re-executed
without a frozen integration state.

**Downstream impact:** none on theorem inputs; reproducibility metadata was
incomplete.

**Disposition:** `DIAGNOSTIC-REEXECUTION`. Future diagnostics record all
algorithmic seeds/state or are explicitly treated as non-hashed displays.

---

## E-C094-6 — “Linear repulsion is known” conflated three different claims

**Evidence:** `C094_CITATION_PROVENANCE.json`.

**Finding:**

- Ladgham–Lachièze-Rey describe typed extrema/saddle hard repulsion with three
  additional powers at the factorial-moment level;
- Azaïs–Delmas give aggregate neutrality in dimension two;
- neither fact alone supplies the pair-Palm two-height marked-window law.

**Cause:** spatial/type correlation, marked height density, and conditioned
six-pin transfer were collapsed into one citation label.

**Downstream impact:** Q0-REFEREE, BR-REP, BR-MARK, Q0-B.

**Disposition:** `CORRECTION-REGISTERED`; no citation-only mark transfer is
permitted.

---

## E-C094-7 — Nicolaescu identifier correction was stated too broadly

**Evidence:** `C094_CITATION_PROVENANCE.json`.

**Finding:** both IDs are real and refer to different papers:

```text
1101.5990  Critical sets of random smooth functions on compact manifolds
1209.0639  Random Morse functions and spectral geometry
```

**Cause:** an operator-line correction treated the IDs as globally
interchangeable.

**Downstream impact:** bibliography only.

**Disposition:** `ADJUDICATED-CONFLICT`. Match identifier to title and claimed
content.

---

## E-C094-8 — Rung minimum was consumed as a domain infimum

**Evidence:** `C094_IMPORTED_PRIORS_RECHECK.json`.

**Finding:** the C091 allowed-product values at registered rungs exceed 80,
but the full-domain limit is

\[
79.9889203915211757\ldots
\]

and the function crosses 80 at

\[
r=0.0018360362779673286\ldots.
\]

**Cause:** sampled-rung validation of a uniformly quantified interval claim.

**Downstream impact:** Q0-SHARP finite lower gate and verifier v5.

**Disposition:** `CORRECTION`. `DOMAIN-INFIMUM` is a first-class active gate.

---

## E-C094-9 — Exact six-pin factorization was nearly lost as an unowned import

**Evidence:** `C094_SIX_PIN_FACTORIZATION.json`.

**Finding:** for the exact periodized field,

\[
m_L(x,y)=k_L(y)m_L(x,0)
\]

follows exactly from product-kernel structure and parity-block decoupling.

**Cause:** the operator-line result was absent from C093 successor inputs.

**Downstream impact:** BR-MARK and Theorem B design.

**Disposition:** `IMPORTED-AFTER-REVERIFICATION`, grade `DERIVED-EXACT`.
It reshapes but does not close BR-MARK.

---

## E-C094-10 — Two operator-line measurements lack immutable source artifacts

**Items:**

- height-universality / flatness of \(q\) in \(b\);
- \(b/4\,H_{xx}(M)\) Richardson candidate.

**Cause:** no raw table, exact claim statement, or source hash exists in the
available C091/C093/Q0-X artifacts.

**Downstream impact:** Q0-IV, Q0-B, Q0-SHARP.

**Disposition:** `BLOCKED-EXTERNAL`. The unblock condition is receipt of the
immutable source plus independent rerun.

---

## E-C094-11 — C093 declared a two-sided core whose own source retained losses

**Evidence:** E-C094-2 and E-C094-3.

**Finding:** the semantic v4 graph could pass because it encoded theorem
statements, not the arithmetic assembly and propagation formulas behind them.

**Cause:** missing `ASSEMBLY`, `DOMAIN-INFIMUM`, and source-formula
cross-check gates.

**Downstream impact:** C094 corrected core; Q0-LLM verifier v5.

**Disposition:** `CORRECTION`. Live C094 root is the upper-rate theorem;
verifier v5 must include the three named gates.

---

## E-C094-12 — C093 release-close terminology overstated Pass-1 reproducibility

**Finding:** the frozen ZIP is immutable and verifies before mutation, but the
documented four-command sequence is not itself idempotent.

**Cause:** a writer was placed before hash verification.

**Downstream impact:** C093 closeout wording only.

**Disposition:** `ERRATUM`. C093 bytes remain immutable; C094 carries the
replacement procedure.

---

**Pass-1 E-ledger entries:** 12  
**Homeless entries:** 0
