# C095 E-LEDGER

**Cycle:** C095  
**Policy:** append-only; supersede-never-overwrite.

---

## E-C095-1 — Gate Framework v1.0 mixed status, assumption role, and warrant

**Finding:** `Primitive`, `Proven`, `Open`, and `Killed` were presented as one
linear epistemic ladder.

**Cause:** unlike concepts were compressed into a convenient ordering.

**Impact:** dependency-grade rules became contradictory and difficult to
implement mechanically.

**Disposition:** `CORRECTION`. Master v1.1 separates:

```text
claim status
assumption role
warrant class
reasoning mode
```

---

## E-C095-2 — Composition witness list omitted UNION_BOUND

**Finding:** v1.0 required a witness for unions while its closed witness list
did not contain `UNION_BOUND`; the worked example also suggested disjointness.

**Cause:** products, unions, and monotone assemblies were treated as one
operation.

**Impact:** an implementation could incorrectly require independence for a
union bound or accept a product under a generic “assembly” label.

**Disposition:** `CORRECTION`. v1.1 uses operation-specific witnesses and adds
`UNION_BOUND`, `MONOTONE-ASSEMBLY`, and `INTERVAL-ARITHMETIC`.

---

## E-C095-3 — The standalone master omitted C094-mandated numerical gates

**Finding:** Master v1.0 contained 13 gates but omitted:

```text
ASSEMBLY
DOMAIN-INFIMUM
BAND-PROVENANCE
```

**Evidence:** E-C094-2, E-C094-3, E-C094-5, E-C094-8, and E-C094-11.

**Disposition:** `CORRECTION`. Master v1.1 contains 16 gates plus the SCHEMA
preflight.

---

## E-C095-4 — COMMON-MODE and HEURISTIC-BRIDGE retrodiction evidence was not self-contained

**Finding:** v1.0 Appendix A described two examples for each new gate but did
not identify frozen failure IDs, artifact hashes, or the earlier catch
mechanism.

**Impact:** the gates' logic is implementable, but the meta-claim that they
passed the framework's own retrodiction protocol could not be independently
verified from the accessible release.

**Disposition:** `PROVISIONAL`. Both gates are implemented. Their
framework-level retrodiction grade remains provisional until the evidence
package is supplied.

---

## E-C095-5 — Initial v5 ASSEMBLY implementation did not consume named residual terms

**Finding:** the first v5 draft stored `residual_terms` but did not require
those terms to appear in the input map and arithmetic expression.

**Catch mechanism:** self-application of Gate 14 to the current UB-G object.

**Impact:** the draft could have repeated the exact failure it was designed to
catch.

**Disposition:** `CORRECTION`. v5 now fails when any required or residual term
is absent or unused.

---

## E-C095-6 — C094 4.35 UB-G correction still lacked a full residual coefficient

**Finding:** the C094 base coefficient

\[
(0.657+2.8185310984\ldots)/0.80
=
4.3444138731\ldots
\]

does not itself include the positive Gamma and collar residuals named by the
source.

The rounded display inputs

\[
(0.66+2.82)/0.80=4.35
\]

leave zero display-input margin.

**Impact:** the uniform decimal \(4.35r^3\) is not promotable until

\[
\sup_{0<r\le0.025}
\frac{\Gamma(r)+\operatorname{Collar}(r)}{r^3}
\]

is explicitly budgeted in the assembly.

**Disposition:** `CORRECTION`. Structural UB-G remains Derived. The decimal
upper and q₀ consequence are `Proven-Modulo UB_G_RESIDUAL_UNIFORM`.

---

## E-C095-7 — The current Q0 contract fails the verifier built to audit it

**Evidence:** `q0_llm_verifier_v5_validation.json`.

**Observed gates:**

```text
ASSEMBLY
UB_G_RESIDUAL_UNIFORM
```

**Disposition:** expected and preserved. No grandfather exception is created.

---

## E-C095-8 — A single regime-of-validity field was insufficient

**Finding:** establishment regime alone does not state where a result is being
deployed.

**Disposition:** `CORRECTION`. F-1 in v1.1 records both establishment and
deployment regimes and requires bridges for mismatches.

---

## E-C095-9 — COMMON-MODE and HEURISTIC-BRIDGE lacked mechanical schema fields

**Disposition:** v1.1 adds F-4 and F-5; v5 implements the corresponding
certificates and grade caps.

---

## E-C095-10 — Exact six-pin covariance structure was hidden inside a generic 6×6 Schur complement

**Finding:** parity and the product kernel split the conditional covariance
exactly into a 4×4 even block and a 2×2 odd block.

**Result:**

\[
C_6(X,Y)
=
k(x-x')k(y-y')
-
k(y)k(y')e(x)^TE^{-1}e(x')
-
k'(y)k'(y')o(x)^TO^{-1}o(x').
\]

**Disposition:** `DERIVED-EXACT`. BR-MARK is substantially simplified but not
closed.

---

## E-C095-11 — Gate count was treated as a presentation invariant

**Finding:** v1.0's “13 gates + 1 field” count conflicted with frozen C094
requirements.

**Disposition:** `CORRECTION`. The count is an output of the failure record.
v1.1 has 16 numbered gates, one preflight, and five field groups.

---

## E-C095-12 — Uniform numerical claims need both certificate type and domain coverage

**Finding:** a certificate value alone cannot distinguish an analytic
infimum from a sampled rung minimum.

**Disposition:** `CORRECTION`. v5 records certificate type, full domain,
extremum direction, proof hash, and the certified extremum value.

---

**Entries:** 12  
**Homeless entries:** 0
