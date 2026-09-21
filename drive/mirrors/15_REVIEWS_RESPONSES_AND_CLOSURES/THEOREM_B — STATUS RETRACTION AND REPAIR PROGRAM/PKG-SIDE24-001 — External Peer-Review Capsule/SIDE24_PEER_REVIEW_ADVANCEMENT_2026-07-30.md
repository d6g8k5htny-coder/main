# Side-24 peer-review advancement report

**Author:** Dylan Roy  
**Audit date:** 2026-07-30  
**Scope:** Fixed \(d=3\), side-\(24\), normalized periodized Bargmann–Fock, finite nonessential superlevel \(H_0\) theorem only.

## Outcome

The fixed-side theorem remains the strongest submission candidate in the current corpus. This pass did not find a contradiction in the load-bearing exponent, normalization, selection, or coefficient arguments. It did find and repair two definite manuscript defects and one omitted supporting proposition:

1. The Fourier coefficients of the normalized periodized covariance were written without their positive normalization factor. The genericity proof now uses the exact coefficient
   \[
   \widehat K_{24}(m)
   =\frac{(2\pi)^{3/2}}{24^3Z_{24}}
   e^{-2\pi^2|m|^2/24^2}.
   \]
2. The contact variable was defined as \(\nu_t\) but subsequently used as \(u_t\). The definition is now consistently \(u_t=-\partial_t^3f_{24}/12\).
3. The existence of the first-moment lifetime density was previously implicit. A new proposition now proves absolute continuity of the elder-paired birth–death measure by two-point Kac–Rice away from the spatial diagonal, a countable separation cover, and the linear lifetime pushforward.

The real-analytic sample-path argument was also strengthened: Gaussian Fourier decay gives almost-sure locally uniform convergence on every complex strip, hence an entire periodic extension, rather than merely \(C^\infty\) convergence on the real torus.

## Hostile proof audit

### Passed in this pass

- The ordered midpoint/displacement map has determinant one.
- The full directed \(S^2\) counts each typed ordered maximum–saddle pair once.
- Elder \(H_0\) pairing has unit critical-pair multiplicity.
- The cubic gap change contributes \(r^3/6\).
- The corrected six-pin density contributes \(r^{-6}\).
- The two soft Hessian determinants contribute \(r^2\).
- Pair polar measure contributes \(r^2\,dr\).
- The resulting radial measure is \((1/6)r\,dr\).
- The lifetime map \(\ell=\kappa r^3/6\) gives
  \[
  r\,dr=\frac{6^{2/3}}3\kappa^{-2/3}\ell^{-1/3}\,d\ell.
  \]
- The polynomial correction evaluates exactly to
  \[
  P_3(24)=-\frac{620813376}{35}.
  \]
- Adler–Taylor Theorem 11.3.4 and Lemma 11.2.10 are correctly matched to, respectively, the Morse-genericity and overdetermined zero-set steps once their hypotheses are stated explicitly.

### Still requiring independent human scrutiny

The following are the two genuinely high-risk proof locations:

1. The elder-selection implication that every failure outside the controlled local dynamics produces a collar, singular-near, fixed-distance, or same-maximum witness.
2. The energy-adapted capture/escape lemma for unbounded \(\lambda_{\max}(T_r)\), including transfer from the pin-preserving polynomial model to the exact conditioned field.

The internal derivations are detailed and mutually consistent, but these are topology/dynamics arguments where independent expert reconstruction matters more than further same-method algebra.

## Literature position

The closest located primary sources do not state the theorem proved here:

- Adler–Bobrowski–Borman–Subag–Weinberger and Pranav study persistent homology of Gaussian/random fields, with general and computational emphasis.
- Chazal–Divol prove density existence for broad random persistence settings, not a smooth Gaussian-field short-lifetime law.
- Klein–Agam and Ancona–Gass–Letendre–Stecconi analyze critical-point correlations and Kac–Rice singularities without the elder persistence mark.
- Hirsch–Lachièze-Rey prove limit theorems for topological functionals of Gaussian critical points, not the near-diagonal lifetime coefficient.

The targeted search found no prior \(\ell^{-1/3}\) theorem for the finite nonessential \(H_0\) lifetime density of a smooth Gaussian field, no prior selected maximum–saddle coefficient equal to (2.9), and no prior side-\(24\) correction equal to (2.10). This supports novelty but is not an exhaustive MathSciNet/zbMATH determination.

## Submission judgment

The theorem is mathematically substantial enough to justify preparing a conventional paper. It is not yet prudent to submit the current package unchanged. The minimum remaining work is:

1. independent human verification of the two high-risk elder-selection/dynamics steps;
2. replacement of the provenance-style exact appendix by a continuous journal proof with globally numbered lemmas;
3. a systematic database and citation-network search;
4. ordinary bibliography, notation, and journal-style editing.

Directed-rounding certification of the anisotropic integral would strengthen the numerical story but is not necessary for the analytic theorem or the stated \(10^{-180}\) correction remainder.

## Current revised manuscript identity

- File: `SIDE24_MANUSCRIPT_CORE.md`
- Bytes: `21550`
- SHA-256: `fb576de64f17f64a23423b185e2abc902a436d3a6a1bbb2aee04d756ad371183`
