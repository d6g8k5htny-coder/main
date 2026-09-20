# Reliability, evidence, and known failure record

## Purpose

This file separates four categories that are easy to conflate:

1. **exact derivations written in the package;**
2. **standard external mathematics used without reproof;**
3. **new arguments presented only as proof sketches;**
4. **numerical evidence.**

The intended reviewer should not give the same weight to all four.

No numerical experiment is used as a substitute for a missing theorem. No withdrawn numerical constant is part of the requested review.

---

# 1. Claim-status table

| Claim or ingredient | Current evidentiary status | Main file |
|---|---|---|
| Exact periodized Bargmann–Fock covariance | Definition | 01 |
| Finite-jet nondegeneracy from positive Fourier weights | Full proof written | 01 |
| Elder-rule defect dichotomy | Full deterministic proof written | 01 |
| Pair-Palm Kac–Rice count identity | Standard formula, explicitly instantiated | 01 |
| Corrected pair-coordinate determinant \(-r^{-5}\) | Exact algebra | 01, 03 |
| Pair-Palm normalizer \(Z_r\asymp r^2\) | Leading proof written; uniform remainder by compactness | 01 |
| Type-indicator continuity | Full finite-dimensional proof written | 01 |
| Exterior/fixed-annulus \(O(r^3)\) count | Standard compactness argument, no explicit constants | 01 |
| Collar \(\rho^{-2}\) density versus \(\rho^2\) determinant cancellation | Exact leading algebra; finite-\(r\) envelope sketched | 01 |
| Third-maximum cubic type no-go | Exact sum-of-squares algebra | 01 |
| Window-critical \(\eta^6\) determinant factor | Exact algebra | 01 |
| Full typed-to-adjacent Palm transfer | Proof sketch; high-priority review item | 01, 03 |
| No saddle–saddle heteroclinic connections | Proposed proof; charting and slicing need specialist review | 02 |
| Exact \(\ell^{-1/3}\) Jacobian | Full derivation | 03 |
| Candidate-pair contact intensity before adjacency | Exact power cancellation plus compactness | 03 |
| Positive uniform adjacency factor | Proof sketch | 03 |
| Uniform elder-rule selection on compact marks | Exact local algebra plus compactness sketch | 03 |
| Full normalized-gap tail domination | Gaussian-tail proof sketch | 03 |
| Influence region determines the death partner | Full deterministic proof | 04 |
| Positive-level subcritical crossing decay | Published theorem | 04 |
| Transfer of crossing decay to local pair-Palm law | New proof sketch | 04 |
| Selected-set Campbell identity | Exact cross-fit identity | 04 |
| Selected same-field intensity \(O(\ell)\) | New Kac–Rice/percolation proof sketch | 04 |
| Thermodynamic stabilization | Conditional on the preceding transfer lemmas | 04 |
| Critical localization length diverges | Qualitative argument from published subcritical and critical results | 04 |

---

# 2. Exact constants and fixed conventions

The following are part of the mathematical setup, not fitted values.

## Field and geometry

\[
L=24,
\qquad
b=\frac65,
\qquad
M=(-r/2,0),
\qquad
S=(r/2,0),
\qquad
0<r\le0.025.
\]

The pair value gap in the fixed-torus theorem is

\[
\ell=\frac{r^3}{6}.
\]

The local regional decomposition uses:

\[
B(M,2r)\cup B(S,2r)
\]

for the collars and a fixed physical ball

\[
B_3(0)
\]

for the near region.

These choices are inherited from the local analysis. The qualitative existence theorem does not claim optimality of the radii.

## Palm-normalizer limit

For

\[
Q\sim N(-b,2),
\]

\[
z_0
=
E[Q^2\mathbf 1_{\{Q<0\}}]
=
(b^2+2)\Phi\!\left(\frac b{\sqrt2}\right)
+
\sqrt2b\,\phi\!\left(\frac b{\sqrt2}\right).
\]

At \(b=6/5\),

\[
z_0\approx3.230978535.
\]

The formula is exact. The decimal is a numerical evaluation.

## Near-diagonal Jacobian

\[
r\,dr
=
\frac{6^{2/3}}3
\kappa^{-2/3}
\ell^{-1/3}\,d\ell.
\]

The factor \(6^{2/3}/3\) is exact.

---

# 3. Numerical observations

All quantities in this section are supporting evidence only.

## 3.1 Pair-Palm normalizer

Monte Carlo estimates of \(Z_r\) under the six-pin Gaussian law were:

| \(r\) | \(Z_r\) | \(Z_r/r^2\) |
|---:|---:|---:|
| 0.05 | 0.00808418 | 3.23367 |
| 0.025 | 0.00201529 | 3.22447 |
| 0.0125 | 0.000503621 | 3.22318 |

These are consistent with the exact limiting coefficient \(3.23098\).

The estimates share the same covariance and Monte Carlo implementation; their agreement is not an independent proof.

---

## 3.2 Exterior third-point intensities

At sampled exterior stations and three separations, the pair-Palm maximum-window intensity divided by the stationary maximum-height benchmark lay between

\[
0.98467
\quad\text{and}\quad
1.01030.
\]

The analogous exterior saddle-window ratio was within roughly one percent of the stationary benchmark.

This supports the finite transfer mechanism but does not establish a uniform continuum bound.

---

## 3.3 Collar collision scaling

Nested collision scans down to scaled collision radius \(0.02\) found:

- stable \(r^3\rho^2\) times the third-gradient density;
- stable triple determinant moment divided by \(r^6\rho^2\);
- approximately linear scaling of the third-critical-point intensity in \(r\).

These tests were designed to detect a missing collision power.

---

## 3.4 Transversality testbed

For

\[
f_0(x,y)
=
-\frac12y^2(1-2x)+3x^2-2x^3,
\]

the \(x\)-axis is an exact saddle–saddle connection.

With perturbation

\[
\mu y e^{-8(x-1/2)^2},
\]

the section mismatch was approximately antisymmetric:

\[
D(0.02)\approx0.01488,
\qquad
D(-0.02)\approx-0.01488.
\]

The derivative was approximately

\[
0.7441.
\]

Variational, adjoint, endpoint-atom, and finite-difference comparisons ranged from displayed precision to about two percent discrepancy, depending on the test.

Direct RKHS curve quadrature failed by more than twenty percent because the integrable endpoint singularity converges too slowly for naive truncation. The abstract support-separation proof does not use that quadrature.

---

## 3.5 Near-diagonal persistence experiment

### Failed first experiment

A four-neighbor vertex filtration gave cumulative exponents approximately \(0.15\)–\(0.29\), excluding the predicted value \(2/3\).

A large population of bars occurred at an \(h^2\)-class grid scale.

### Coupled refinement experiment

A second pre-registered experiment used:

- Freudenthal triangulations;
- identical Fourier realizations on nested grids;
- interpolation-error estimates;
- cross-resolution diagram matching;
- stability filtering;
- left-truncated likelihoods.

For the primary \(256\to512\) fit:

\[
\widehat\alpha=-0.4814,
\qquad
95\%\text{ interval }[-0.7115,-0.2437],
\]

from 543 stable bars. The interval contains \(-1/3\).

For a larger upper cutoff:

\[
\widehat\alpha=-0.5005,
\qquad
[-0.6385,-0.3791],
\]

which excludes \(-1/3\).

The numerical evidence is therefore supportive under the primary criterion but not a precise confirmation.

---

## 3.6 Critical-height diagnostics

A finite-size cluster experiment conditioned on a single field value, not on a maximum–saddle pair, estimated

\[
\gamma\approx2.439,
\qquad
\nu\approx1.372,
\qquad
\gamma/\nu\approx1.778.
\]

These are close to standard two-dimensional percolation values. The conditioning law is different from the pair-Palm law, so these estimates are not theorem inputs.

A simpler selected-set experiment found a same-field/cross-fit amplification around \(4.5\)–\(5.5\) on its resolved window range. It did not show divergence, but it did not implement the full pair-Palm law.

---

# 4. Withdrawn numerical claims

The following numbers appeared in earlier attempts but are not claimed in this package.

## Upper constants

\[
4.3,\qquad
4.35,\qquad
0.97,\qquad
0.99,\qquad
1.01.
\]

Reasons for withdrawal included:

- downward rounding of an upper assembly;
- a far-field coefficient based on multiplying dependent checkpoint probabilities;
- failure at a registered endpoint;
- absence of a full-interval supremum proof;
- unrecovered provenance of a stated \(0.02\) uncertainty band.

## Lower constants

\[
0.8501,\qquad
0.84.
\]

Reasons for withdrawal included:

- omission of multiplicative finite-\(r\) losses;
- unproved six-pin marked repulsion and mark-density estimates;
- use of sampled separations instead of the full-domain infimum.

## Other retired language

The measured value

\[
0.946
\]

at one separation was once described as if it were a scale-free “truth constant.” It is only a rung-tagged measurement and is not used here.

---

# 5. Failed approaches preserved for reliability

## 5.1 Finite-jet steering of the orbit

An early transversality idea was to prescribe \(\nabla h\) at one interior point while holding endpoint jets fixed. This was abandoned because an analytic Cameron–Martin perturbation remains uncontrolled along the rest of the orbit and the full derivative may cancel.

The RKHS distribution argument was introduced specifically to avoid that problem.

## 5.2 Naive explicit curve quadrature

Uniform and geometrically refined arclength quadratures failed because the endpoint weight behaved like \(s^{-5/6}\) in the testbed. Refining the cutoff without matched asymptotics made the result worse.

The abstract derivative and support-separation proof are independent of this numerical failure.

## 5.3 Independent-corridor product

A far-field bound multiplied one-step Gaussian crossing probabilities. Joint calculations differed from the product by factors as large as approximately \(3.35\).

The numerical coefficient based on that product was withdrawn. The thermodynamic note proposes a selected-set Campbell replacement.

## 5.4 Raw-grid persistence slope

The first persistence experiment counted a large grid-scale bar population and strongly rejected the predicted exponent. It was not erased. A continuum-consistent replacement was run under a separate pre-registered protocol.

---

# 6. Common-mode limitations

Several numerical instruments share:

- the same covariance implementation;
- the same Gaussian conditioning routines;
- the same determinant-type logic;
- related Monte Carlo sampling code.

Agreement among them can therefore fail to detect a shared sign convention, coordinate, or conditioning error.

The strongest independent checks available were:

- exact symbolic derivations of determinant factorizations;
- a separate direct affine solve for rational test cases;
- finite-difference versus variational comparisons in the transversality testbed;
- a second persistence experiment with a different filtration and estimator.

The package does not include source code, so the external reviewer is being asked to assess the written mathematics, not to certify software reproducibility.

---

# 7. Reliability questions the present arguments do not settle

1. Is the typed-to-adjacent Palm transfer genuinely uniform?
2. Is the Cameron–Martin chart/slicing proof a correct substitute for a published parametric transversality theorem?
3. Are the quartic boundary-layer estimates in the third-maximum count sufficient?
4. Does the local-Palm percolation transfer survive determinant reweighting and a third saddle pin?
5. Is the adjacency contact factor provably positive and continuous?
6. Is the full normalized-gap Gaussian tail bound uniform in every parameter needed for the lifetime coefficient?

Those are the reasons this package is being sent to specialists.

---

# 8. Suggested weight of evidence

A reasonable preliminary weighting is:

- **high confidence:** exact finite-dimensional algebra and deterministic elder-rule arguments;
- **moderate confidence:** corrected-frame compactness and Palm-normalizer scaling;
- **uncertain:** global Gaussian transversality charting;
- **uncertain:** adjacency-mark contact limit;
- **uncertain:** pair-Palm percolation transfer;
- **supporting only:** all numerical evidence.

The correct external outcome may be that the main results are valid only after adding one or more explicit hypotheses. That is an acceptable and useful review conclusion.

---

# 9. Bibliography

## Gaussian fields and Kac–Rice

1. R. J. Adler and J. E. Taylor, *Random Fields and Geometry*, Springer, 2007.
2. J.-M. Azaïs and M. Wschebor, *Level Sets and Extrema of Random Processes and Fields*, Wiley, 2009.
3. V. I. Bogachev, *Gaussian Measures*, American Mathematical Society, 1998.
4. S. Janson, *Gaussian Hilbert Spaces*, Cambridge University Press, 1997.

## Dynamical systems and transversality

5. R. Abraham and J. Robbin, *Transversal Mappings and Flows*, Benjamin, 1967.
6. M. W. Hirsch, C. C. Pugh, and M. Shub, *Invariant Manifolds*, Lecture Notes in Mathematics 583, Springer, 1977.
7. J. Palis, Jr. and W. de Melo, *Geometric Theory of Dynamical Systems*, Springer, 1982.

## Persistence

8. H. Edelsbrunner and J. Harer, *Computational Topology: An Introduction*, American Mathematical Society, 2010.
9. D. Cohen-Steiner, H. Edelsbrunner, and J. Harer, “Stability of persistence diagrams,” *Discrete & Computational Geometry* 37 (2007), 103–120.

## Bargmann–Fock percolation

10. A. Rivera and H. Vanneuville, “The critical threshold for Bargmann–Fock percolation,” arXiv:1711.05012.
11. S. Muirhead and H. Vanneuville, “The sharp phase transition for level set percolation of smooth planar Gaussian fields,” arXiv:1806.11545.

## Adjacent numerical/topological work

12. J. Feldbrugge, M. van Engelen, R. van de Weygaert, P. Pranav, and G. Vegter, “Stochastic Homology of Gaussian vs. non-Gaussian Random Fields: Graphs towards Betti Numbers and Persistence Diagrams,” arXiv:1908.01619.
