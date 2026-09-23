# C102 Adversarial Referee Objection Ledger

**Verdict:** the qualitative finite-\(C\) rate and \(q_0=1\) limit survive the internal referee conversion. Independent SARD-G specialist acceptance remains the named external dependency.

| ID | topic | severity | disposition |
|---|---|---|---|
| `REF-C102-01` | Defect decomposition | `LOAD-BEARING` | `ANSWERED-BY-NEW-PROOF` |
| `REF-C102-02` | Loop/crater coverage | `LOAD-BEARING` | `ANSWERED-BY-NEW-PROOF` |
| `REF-C102-03` | Finite-jet nondegeneracy | `LOAD-BEARING` | `ANSWERED-BY-NEW-PROOF` |
| `REF-C102-04` | Corrected pair-frame invertibility | `LOAD-BEARING` | `ANSWERED` |
| `REF-C102-05` | Small-r transfer | `LOAD-BEARING` | `ANSWERED-BY-NEW-LEMMA` |
| `REF-C102-06` | Chart completeness | `LOAD-BEARING` | `ANSWERED-BY-ATLAS` |
| `REF-C102-07` | Type-indicator continuity | `LOAD-BEARING` | `ANSWERED-BY-NEW-LEMMA` |
| `REF-C102-08` | Pair-Palm denominator | `LOAD-BEARING` | `ANSWERED-BY-NEW-LEMMA` |
| `REF-C102-09` | Collision integrability | `LOAD-BEARING` | `ANSWERED` |
| `REF-C102-10` | Near maximum-window integrability | `LOAD-BEARING` | `ANSWERED` |
| `REF-C102-11` | Global interceptor integrability | `LOAD-BEARING` | `ANSWERED` |
| `REF-C102-12` | COMMON-MODE diagnostics | `GRADE` | `SUSTAINED-AS-LIMITATION` |
| `REF-C102-13` | SARD-G/Morse-Smale status | `EXTERNAL` | `EXTERNAL-REVIEW-TRACK` |
| `REF-C102-14` | Claim language | `SEMANTIC` | `ANSWERED-BY-FENCE` |

## Full adjudications

### REF-C102-01 — Defect decomposition

**Objection.** C101 displayed a three-class defect decomposition but did not write the deterministic elder-rule proof in the successor package.

**Answer.** C102 proves {D(M)!=S} subset Pi union Gamma under Morse-Smale and distinct critical values, by an exhaustive earlier-death / same-component / distinct-component case split.

**Disposition:** `ANSWERED-BY-NEW-PROOF`

**Remaining boundary.** External use must state or discharge Morse-Smale/R0.

**Evidence:** `C102_DETERMINISTIC_REDUCTION.md`

### REF-C102-02 — Loop/crater coverage

**Objection.** A high-level path could join the two local saddle arms without being represented by a window critical-point count.

**Answer.** The M-component is born at f(M). If the arms are already connected above f(S), their first connection after the birth of M occurs at an index-one saddle value in (f(S),f(M)); this is Pi.

**Disposition:** `ANSWERED-BY-NEW-PROOF`

**Remaining boundary.** Fails if critical values tie or saddle branches terminate at saddles.

**Evidence:** `C102_DETERMINISTIC_REDUCTION.md`

### REF-C102-03 — Finite-jet nondegeneracy

**Objection.** The compactness arguments assume positive covariance matrices but the exact torus finite-jet theorem was not previously written.

**Answer.** Strictly positive periodized BF Fourier weights reduce zero variance to a polynomial-exponential identity on Z^2. Applying one-dimensional exponential-polynomial independence twice forces every derivative coefficient to vanish.

**Disposition:** `ANSWERED-BY-NEW-PROOF`

**Remaining boundary.** No numerical eigenvalue floor is extracted.

**Evidence:** `C102_FINITE_JET_NONDEGENERACY.md`, `C102_FINITE_JET_NONDEGENERACY.json`

### REF-C102-04 — Corrected pair-frame invertibility

**Objection.** Raw six-pin covariance degenerates as r tends to zero.

**Answer.** The corrected frame transform has exact determinant -r^-5, and its limiting six-jet Gram determinant is 12.

**Disposition:** `ANSWERED`

**Remaining boundary.** None for qualitative invertibility.

**Evidence:** `C102_FINITE_JET_NONDEGENERACY.json`

### REF-C102-05 — Small-r transfer

**Objection.** Positive limiting matrices do not automatically prove a uniform finite-r statement without a transfer argument.

**Answer.** Hermite-Genocchi divided-difference representations extend every scaled covariance entry analytically through its collision face. Positive limiting eigenvalues give a small-r neighborhood; exact finite-r nondegeneracy and compactness cover the remainder.

**Disposition:** `ANSWERED-BY-NEW-LEMMA`

**Remaining boundary.** Transfer radii and numerical floors are existential.

**Evidence:** `C102_DIVIDED_DIFFERENCE_ANALYTICITY.md`, `C102_CHART_ATLAS.json`

### REF-C102-06 — Chart completeness

**Objection.** The generic formulas lose rank on the pair axis, transverse line, and pin-collision faces.

**Answer.** Thirteen named rows map every Gaussian/Kac-Rice rank-loss face to a separate axis, transverse, collision, fixed-annulus, exterior, or type-boundary chart.

**Disposition:** `ANSWERED-BY-ATLAS`

**Remaining boundary.** Coverage means no gap found in the registered Gaussian counting regions; it is not a universal claim outside the theorem decomposition.

**Evidence:** `C102_CHART_ATLAS.csv`, `C102_CHART_ATLAS.md`

### REF-C102-07 — Type-indicator continuity

**Objection.** Maximum/saddle indicators are discontinuous, so compactness of Gaussian means and covariances does not alone imply continuity of typed determinant moments.

**Answer.** The discontinuity boundaries are polynomial zero sets of nondegenerate Gaussian vectors and have measure zero. Uniform Gaussian polynomial moments supply dominated convergence.

**Disposition:** `ANSWERED-BY-NEW-LEMMA`

**Remaining boundary.** A separate chart is still required at covariance rank-loss faces.

**Evidence:** `C102_TYPE_BOUNDARY_CONTINUITY.md`

### REF-C102-08 — Pair-Palm denominator

**Objection.** The Kac-Rice numerator powers are useless if the typed pair-Palm normalizer vanishes faster than r^2.

**Answer.** On a positive-probability event with common transverse curvature Q<=-epsilon and bounded normalized mixed jets, M is a maximum and S a saddle with determinant product >=c epsilon^2 r^2. The coalesced limit is E[Q^2 1_{Q<0}]>0.

**Disposition:** `ANSWERED-BY-NEW-LEMMA`

**Remaining boundary.** No numerical lower coefficient c_Z is claimed.

**Evidence:** `C102_PALM_NORMALIZER_POSITIVITY.md`

### REF-C102-09 — Collision integrability

**Objection.** The third-gradient density diverges near an already pinned critical point.

**Answer.** C100's exact cubic determinant product has a rho^2 collision zero against the rho^-2 density scale, with exponential angular suppression on the axial faces.

**Disposition:** `ANSWERED`

**Remaining boundary.** No sharp collision coefficient is claimed.

**Evidence:** `C100_COLLAR_CUBIC_CLOSURE.md`, `C100_COLLAR_SCALED_FRAME.json`, `C102_CHART_ATLAS.json`

### REF-C102-10 — Near maximum-window integrability

**Objection.** A third maximum at the same cubic scale could cause a nonintegrable three-point maximum count.

**Answer.** The exact cubic third determinant is a nonpositive sum of squares throughout the height window. A finite-t maximum requires quartic boundary-layer corrections, supplying the needed determinant power.

**Disposition:** `ANSWERED`

**Remaining boundary.** No two-sided sharp maximum-count asymptotic is claimed.

**Evidence:** `C099_CUBIC_TYPE_NOGO.md`, `C099_NEAR_CUBIC_CLOSURE.md`

### REF-C102-11 — Global interceptor integrability

**Objection.** The singular near window-critical count may diverge after pair-Palm normalization.

**Answer.** Under the third value/gradient constraints each leading Hessian determinant has eta^2, giving eta^6 in the product. The resulting polar integral is C r^7 integral t^-5 dt=O(r^3).

**Disposition:** `ANSWERED`

**Remaining boundary.** The upper scale is conservative and not asserted sharp.

**Evidence:** `C101_WINDOW_SADDLE_FRAME.json`, `C101_GLOBAL_INTERCEPTOR_CLOSURE.md`, `C101_INDEPENDENT_CHECK.json`

### REF-C102-12 — COMMON-MODE diagnostics

**Objection.** The numerical station scans reuse the same exact-covariance and Monte Carlo stack and cannot independently certify the theorem.

**Answer.** All scans remain MEASURED-DIAGNOSTIC and are excluded from the theorem grade. Exact symbolic proofs and a separate raw affine C101 arithmetic reconstruction carry the load.

**Disposition:** `SUSTAINED-AS-LIMITATION`

**Remaining boundary.** A fully independent software implementation remains desirable.

**Evidence:** `C101_INDEPENDENT_CHECK.json`, `C101_E_LEDGER.md`

### REF-C102-13 — SARD-G/Morse-Smale status

**Objection.** The full Gaussian theorem needs the global no-saddle-connection result, not merely finite-jet nondegeneracy.

**Answer.** The live SARD-G package is internally program-grade closed with a charted mismatch functional, nonzero RKHS representative, and countable Cameron-Martin disintegration. Independent specialist acceptance remains explicitly pending.

**Disposition:** `EXTERNAL-REVIEW-TRACK`

**Remaining boundary.** Named external specialist review.

**Evidence:** `04 PART IV SARD G.md`, `R0_SOURCE_PRECEDENCE_DECISION_C090.md`

### REF-C102-14 — Claim language

**Objection.** The existence proof could be misread as restoring 4.35, 4.3, 0.8411, or a publication-grade explicit coefficient.

**Answer.** The C102 package states only existence of a finite C_Q0 and the selection limit. All previous numerical upper/lower displays remain NOT-CLAIMED.

**Disposition:** `ANSWERED-BY-FENCE`

**Remaining boundary.** None if the mandatory qualifier block travels with the theorem.

**Evidence:** `Q0_C101_QUALITATIVE_RATE_THEOREM.md`, `C102_QUALITATIVE_RATE_PACKAGE.md`

## Final grade boundary

```text
qualitative existence of finite C_Q0:
    SURVIVES INTERNAL REFEREE CONVERSION

q(r,6/5) -> 1:
    SURVIVES INTERNAL REFEREE CONVERSION

explicit numerical coefficient:
    NOT-CLAIMED

independent SARD-G specialist acceptance:
    EXTERNAL-REVIEW-TRACK
```
