# W11 — LITERATURE AND METHODS REPORT
## Primary-source methods support for the WP enclosure campaign (K3 swarm)

Prepared for the K3 lead. Scope: primary sources only (papers/monographs), for *certification-grade* methods —
Kac–Rice validity, rigorous Gaussian integration, interval/verified computing, explicit-constant Gaussian tail
bounds, tube/supremum inequalities, quantitative IFT/continuation, and kriging residual-variance laws.
Each entry gives the citation, then 2–4 sentences on exactly what the result delivers and its hypotheses.
Confidence flags: [VERIFIED] = theorem number/page cross-checked against a quoting primary/secondary source in
this investigation; [STANDARD] = well-known citation, details from memory of the primary source; [DERIVED] =
not a named theorem, assembled from cited pieces.

---

## Q1. Kac–Rice with indicators on Hessian type AND field value; conditioning on ∇f = 0 only

**Target identity.** For a smooth Gaussian field f on D ⊂ R² and Borel sets B (Hessian types) and W (value
window),
E[ #{x ∈ D : ∇f(x)=0, Hf(x) ∈ B, f(x) ∈ W} ]
 = ∫_D p_{∇f(x)}(0) · E[ |det Hf(x)| · 1_{Hf(x)∈B} · 1_{f(x)∈W} | ∇f(x)=0 ] dx.        (KR-I)
The value f(x) is *not* pinned by the conditioning; it appears inside the conditional expectation.

**(a) Weighted Kac–Rice, fields.** J.-M. Azaïs & M. Wschebor, *Level Sets and Extrema of Random Processes
and Fields*, Wiley, 2009 — Theorem 6.2 (Rice formula, first moment, random fields), Theorem 6.3 (k-th
factorial moments), **Theorem 6.4 (weighted/"marked" Rice formula)** [VERIFIED: Thm 6.4 is the weighted
version, confirmed by multiple quoting papers, e.g. Azaïs et al. 2020 and Azaïs–León discretization-error
paper]. Thm 6.4 gives E[ Σ_{t∈E, X(t)=u} g(t, Z(·)) ] = ∫_E E[ |det X'(t)| g(t,Z(·)) | X(t)=u ] p_{X(t)}(u) dt
for a weight g depending on the location and on an auxiliary field Z(·) — here X = ∇f and Z = (f, Hf), and
g(t, Z) = 1_{Hf(t)∈B}·1_{f(t)∈W}. Hypotheses: X a.s. C¹, X(t) has a density p_{X(t)} bounded near u
(Gaussian: nondegenerate covariance suffices), the Bulinskaya condition P{∃t: X(t)=u, det X'(t)=0} = 0,
continuity of the conditional distributions, and g lower semicontinuous (indicators of open sets are the
canonical admissible case; indicators of closed sets follow by approximation). In the jointly nondegenerate
Gaussian case the regular-conditional-distribution continuity hypothesis is automatic.

**(b) Indicator-of-open-set formulation with full field germ.** D. Armentano, J.-M. Azaïs, J. R. León,
"On a general Kac–Rice formula for the measure of a level set", arXiv:2304.07424 (2023), Theorem 6.1
("Expected integral on the level set") [VERIFIED]. States E[∫ g(t,Z(·)) dσ ] = ∫ E[Δ(t) g(t,Z(·)) | X(t)=u]
p_{X(t)}(u) dt with hypotheses (a) g l.s.c. in t, (b) g l.s.c. in Z for the weak C⁰ topology, (c) the law of
Z(·) given X(t)=v is well-defined and weakly continuous in v; **Remark 8: for jointly non-degenerate Gaussian
(X,Z), condition (c) always holds, and admissible g are often indicators of open sets.** Section 7.1 of the
same paper applies it to critical points of Gaussian fields, producing exactly the value-window form
∫_W E[det X'' | X'=0, X=x] p_{(X',X)}(0,x) dx — i.e., the equivalent "integrated over the value" writing of
(KR-I). This is the cleanest primary citation that the f-value indicator may sit inside the expectation with
only ∇f conditioned.

**(c) Gaussian metatheorem.** R. J. Adler & J. E. Taylor, *Random Fields and Geometry*, Springer Monographs
in Mathematics, 2007 — **Theorem 11.2.1 (Kac–Rice metatheorem for Gaussian critical points)** [VERIFIED:
numbering confirmed by Cheng–Schwartzman, "Expected number and height distribution of critical points of
smooth isotropic Gaussian random fields", Bernoulli 24 (2018), which uses Thm 11.2.1 for exactly the joint
indicator E[|det ∇²X| 1_{index=i} 1_{X≥u} | ∇X=0]]. Hypotheses (AT conditions): f Gaussian on a compact
stratified manifold, a.s. C² (their condition (11.3.1), slightly stronger than C², implied by C³), the joint
vector (f(t), ∇f(t), ∇²f(t)) nondegenerate Gaussian for each t, and a nondegeneracy/continuity condition on
pairs (∇f(t), ∇f(s)) for t ≠ s (their (iii)–(v)). AT Theorem 12.1.1 is the non-Gaussian generalization.

**(d) Bulinskaya-type hypotheses (no degenerate critical points a.s.).** E. V. Bulinskaya, "On the mean
number of crossings of a level by a stationary Gaussian process", Theory Probab. Appl. 6 (1961) 435–438
[STANDARD]. Book form: **Azaïs–Wschebor 2009, Proposition 6.5** (Gaussian fields: if X is C² a.s. and the
density of X(t) is bounded near u, then P{∃t: X(t)=u, det X'(t)=0}=0 — the a.s.-Morse statement, quoted as
AW Prop 6.5 in the Weyl–Heisenberg zeros literature) and the standalone **Bulinskaya lemma: a C¹ random
field Y: T→R^d whose pointwise density is bounded in a neighborhood of the origin satisfies P{Y vanishes
somewhere}=0** (stated as Proposition 6.11 in AW09 per the quoting paper "Critical point asymptotics for
Gaussian random waves…", Adv. Math. 2023 — numbering [VERIFIED] only through that quotation; also cited as
"Proposition 1.20" in Beliaev–McAuley–Muirhead, *A covariance formula for topological events of smooth
Gaussian fields*, Ann. Probab. 48 (2020), which references the AW numbering differently; treat Prop 6.5 as
the safe cite). For our periodized Bargmann–Fock field, full spectral support makes every finite jet Gram
matrix positive definite, so both the density-boundedness and the no-degenerate-zero hypotheses hold.

**Bottom line for the manuscript's Preliminary 2.4/§2.3:** (KR-I) with the joint indicator is rigorous as
stated; the correct anchor citations are AW09 Thm 6.2+6.4 (+ Prop 6.5) or AT07 Thm 11.2.1, with the
value-window indicator handled by the "admissible weights include indicators of open sets" clause (AAL
arXiv:2304.07424, Thm 6.1 + Remark 8, or the Cheng–Schwartzman use of AT Thm 11.2.1). One-sided pointwise
bounds built on (KR-I) (Cauchy–Schwarz, Cantelli) are then bounds on a rigorously defined integrand.

---

## Q2. Rigorous Gaussian integration: enclosures for E[g(Z)], Z a 3D Gaussian

**(a) Classical derivative remainder for Gauss–Hermite (Krylov class).** V. I. Krylov, *Approximate
Calculation of Integrals* (Macmillan, 1962; transl. A. H. Stroud) [STANDARD]; also Davis–Rabinowitz,
*Methods of Numerical Integration*. For I[f] = ∫_R e^{−x²} f(x) dx, the n-point Gauss–Hermite rule has exact
Peano remainder R_n[f] = (n!√π)/(2ⁿ (2n)!) · f^{(2n)}(ξ) for some ξ ∈ R, f ∈ C^{2n}. What it gives: an exact
error functional depending on a single high derivative — defensible if one can *bound* f^{(2n)} globally,
e.g. via interval automatic differentiation/Taylor models. For tensor-product rules in d=3 the error splits
coordinatewise, so the bound depends on the pure derivatives ∂_i^{2n_i} g; for "peaked" integrands (window
width ℓ = r³/6) these derivatives scale like ℓ^{−2n} — the Krylov bound is honest but pessimistic unless the
peak is resolved, i.e. n grows like the inverse window scale.

**(b) Gauss–Hermite in Hermite/Sobolev scales (Sloan-class, explicit in the operator norm).** F. Y. Kuo,
R. Scheichl, C. Schwab, I. H. Sloan, E. Ullmann, "Multilevel quasi-Monte Carlo methods for lognormal
diffusion problems", Math. Comp. 86 (2017) 2827–2860 (the 1D GH error theorem appears as Theorem 8.1 in the
arXiv version 2002.00624) [VERIFIED]. For f(x) = e^{−x²/2} g(x) with g in the domain of A^r, A =
(x + d/dx)/√2, 3 ≤ r ≤ 2m: |Σ wᵢ f(xᵢ) − ∫ f| ≤ C m / √(2m(2m−1)…(2m−r+1)) · ‖A^r g‖_{L²}, C independent of
m, r, f. What it gives: the error *decays superalgebraically in m* once ‖A^r g‖ is controlled; the constant is
explicit except for the (computable, small) universal C. Dependence on derivatives is through the
L²-Hermite norm, much better behaved for peaked g than sup-derivative Peano bounds. A different explicit
bound is in Mastroianni–Monegato (1994).

**(c) Tanh-sinh / double-exponential quadrature with rigorous error.** H. Takahasi & M. Mori, "Double
exponential formulas for numerical integration", Publ. RIMS Kyoto Univ. 9 (1974) 721–741; M. Mori &
M. Sugihara, "The double-exponential transformation in numerical analysis", J. Comput. Appl. Math. 127
(2001) 287–296 [STANDARD]. The DE error splits as E = E_D (discretization, bounded via the Hardy-space norm
of the transformed integrand in a strip) + E_T (truncation, bounded by a computable tail); total error
O(exp(−cN/log N)), optimal in the Hardy class (M. Sugihara, "Optimality of the double exponential formula —
functional analysis approach", Numer. Math. 75 (1997) 379–395). **Explicit-constant versions:**
T. Okayama, T. Matsuo, M. Sugihara, "Error estimates with explicit constants for Sinc approximation, Sinc
quadrature and Sinc indefinite integration", Numer. Math. 124 (2013) 361–394 [VERIFIED] — all constants
explicit functions of the strip width and decay parameters. **Verified implementation:**
N. Yamanaka, T. Okayama, S. Oishi, T. Ogita, "A fast verified automatic integration algorithm using double
exponential formula", Nonlinear Theory and Its Applications, IEICE 1 (2010) 119–132 [VERIFIED] — a complete
interval-arithmetic algorithm producing guaranteed enclosures via the explicit DE bounds. What it gives for
us: E[g(Z)] with Z 3D Gaussian, g involving |det H| (piecewise-smooth, globally Lipschitz after the
T²−D²−X² change of variables) and analytically smoothed window factors, fits the DE Hardy class after the
standard x ↦ μ + σ·√2·(x-coordinates) scaling; the only non-smoothness, the |·| kink and the mollified
indicator, are handled by splitting the domain at det H = 0 (a quadric) — on each side g is analytic with
computable strip width, so the Okayama–Matsuo–Sugihara bounds apply with certifiable parameters.

**(d) Adaptive Simpson with rigorous a posteriori bounds.** J. N. Lyness, "Notes on the adaptive Simpson
quadrature routine", J. ACM 16 (1969) 483–495 (heuristic); the rigorous version requires verified local
error functionals — K. Petras, "Self-validating integration and approximation of piecewise analytic
functions", J. Comput. Appl. Math. 145 (2002) 345–359 [STANDARD] gives guaranteed enclosures for piecewise
analytic integrands via verified bounds on the error constants. Verdict: usable as an independent
cross-check; not competitive with DE for the production enclosure.

**(e) Taylor models / interval Gaussian expectation.** K. Makino & M. Berz, "Taylor models and other
validated functional inclusion methods", Int. J. Pure Appl. Math. 4 (2003) 379–456; M. Berz & K. Makino,
"Verified integration of ODEs and flows using differential algebraic methods on high-order Taylor models",
Reliab. Comput. 4 (1998) 361–369 (COSY Infinity) [STANDARD]. Taylor models give polynomial enclosures with
rigorous remainder over boxes; combined with the Krylov Peano remainder (a) this yields certified derivative
bounds ‖∂^α g‖ over each tensor cell, turning (a)/(b) into a true enclosure. This is the recommended way to
certify the derivative norms appearing in (a)–(c) at 80–120 digits.

**Which gives defensible enclosures for our E[g(Z)]?** Recommendation (details §8): split at det H = 0,
mollify the window indicator analytically with a certified mollification error (via Laurent–Massart-type tail
control near the boundary, Q4), then **tensor DE quadrature with Okayama–Matsuo–Sugihara explicit constants
evaluated in Arb ball arithmetic**, cross-checked by high-order tensor Gauss–Hermite with the Krylov/Kuo et
al. remainder certified by Taylor-model derivative bounds. Agreement of the two independent enclosures within
their radii is itself a certificate.

---

## Q3. Interval arithmetic and verified computing for Gaussian probabilities

**(a) Standards and core libraries.** IEEE Std 1788-2015, *IEEE Standard for Interval Arithmetic* (2015)
[STANDARD] — the semantic standard (flavor, decorations) any certificate implementation should cite.
S. M. Rump, "INTLAB — INTerval LABoratory", in *Developments in Reliable Computing* (Kluwer, 1999) 77–104
[STANDARD] — the MATLAB workhorse, includes verified linear systems with rigorous condition control.
F. Johansson, "Arb: efficient arbitrary-precision midpoint-radius interval arithmetic", IEEE Trans.
Computers 66(8) (2017) 1281–1292 [VERIFIED] — arbitrary-precision ball arithmetic with rigorous error
bounds for elementary and special functions; erf/erfc and hence the Gaussian CDF/Mills ratio are available
with certified radii (via Johansson, "Computing hypergeometric functions rigorously", ACM TOMS 42 (2016)
Art. 30 [STANDARD], which covers the hypergeometric evaluations behind erf, Marcum-Q-type functions, etc.).
This is the natural match for the manuscript's 80–120-digit regime.

**(b) mpmath iv limitations.** F. Johansson et al., *mpmath: a Python library for arbitrary-precision
floating-point arithmetic* (v1.3, 2023; mpmath.org) [VERIFIED]. mpmath's `iv` context provides correct
outward rounding for arithmetic and standard transcendental functions, but its interval linear algebra is
minimal: matrix inversion/solves on intervals are performed naively (no rigorous pivoting strategy, no
verified condition estimate), and the dependency problem makes interval Gaussian elimination catastrophically
wide for correlated (near-singular pin) Gram matrices. Conclusion: mpmath `iv` is acceptable for scalar
function evaluation but **must not be used for the Gram-matrix inversions in (2.4)**; instead compute an
approximate inverse in high-precision floating point and *certify* it by interval residual bounds
(‖I − A·B‖ < 1 ⇒ enclosure of A⁻¹ via Neumaier, *Interval Methods for Systems of Equations*, Cambridge
Univ. Press, 1990 [STANDARD]), which for the 9×9 and smaller Gram matrices here is cheap at 120 digits.

**(c) Rigorous multivariate normal probabilities (Genz class).** A. Genz, "Numerical computation of
multivariate normal probabilities", J. Comput. Graph. Statist. 1(2) (1992) 141–149 [VERIFIED]; A. Genz,
"Numerical computation of rectangular bivariate and trivariate normal and t probabilities", Stat. Comput. 14
(2004) 251–260 [VERIFIED]; A. Genz & F. Bretz, *Computation of Multivariate Normal and t Probabilities*,
Lecture Notes in Statistics 195, Springer, 2009 [VERIFIED]. The Genz transform rewrites a rectangle
probability as a (d−1)-dimensional integral of products of 1D Gaussian CDFs over a bounded simplex-like
domain, evaluated by (randomized) quasi-Monte Carlo — accurate but *not* a certificate. For d = 3 the same
transform reduces to a 2D, then iterated 1D, integrals of products of Φ's, which **are** enclosable by the
verified DE machinery of Q2(c) with Arb Φ evaluations; Genz–Bretz QMC should be kept as the independent
measured-grade cross-check, exactly paralleling the manuscript's cross-representation philosophy. State of
the art for high-dimensional MVN (Genton–Keyes–Turkiyyah, JCGS 27 (2018) 268–277; hierarchical/low-rank) is
irrelevant at d = 3.

---

## Q4. Gaussian quadratic-form tails with EXPLICIT constants; P(det H < 0)

**(a) Chi-square (Laurent–Massart).** B. Laurent & P. Massart, "Adaptive estimation of a quadratic
functional by model selection", Ann. Statist. 28(5) (2000) 1302–1338, **Lemma 1** [VERIFIED]: for D² ~ χ²_d,
P(D² − d ≥ 2√(dx) + 2x) ≤ e^{−x} and P(D² − d ≤ −2√(dx)) ≤ e^{−x}. Fully explicit constants, one-sided in
both directions; the standard certificate-grade chi-square bound.

**(b) Hanson–Wright.** D. L. Hanson & E. T. Wright, "A bound on tail probabilities for quadratic forms in
independent random variables", Ann. Math. Statist. 42 (1971) 1079–1083; modern form M. Rudelson &
R. Vershynin, "Hanson–Wright inequality and sub-Gaussian concentration", Electron. Commun. Probab. 18
(2013) no. 82, 1–9 [VERIFIED]: P(|XᵀAX − E XᵀAX| ≥ t) ≤ 2 exp(−c min(t²/(K⁴‖A‖²_F), t/(K²‖A‖))) with a
*universal but unspecified* c. For certificates needing a printed constant, an explicit-constant proof for
the sub-Gaussian case with constants 144 and 16√2 is given in "A Tutorial on the Non-Asymptotic Theory of
System Identification" (arXiv:2309.03873, Theorem 2.1 + Appendix A) [VERIFIED], and improved explicit
constants via an elementary proof in I. Ziemann, "An elementary proof of the Hanson–Wright inequality"
(arXiv:2509.00881, 2025) [VERIFIED]. For *Gaussian* X specifically, sharper two-sided bounds with fully
explicit constants exist: R. Latała, "Estimation of moments and tails of Gaussian chaoses", Ann. Probab.
34(6) (2006) 2315–2331 [STANDARD], giving upper and lower tail estimates for decoupled Gaussian quadratic
chaos in terms of explicit norms of the matrix — the best matched-shape tool for det-type (indefinite)
quadratic forms.

**(c) Exact spectral method for det H (2×2).** Writing T = (H₁₁+H₂₂)/2, D = (H₁₁−H₂₂)/2, X = H₁₂ gives the
exact algebraic identity det H = T² − D² − X², with (T, D, X) jointly Gaussian (a linear image of
(H₁₁, H₁₂, H₂₂), mean ν̃ and covariance Σ̃ computable from (ν, Σ)). Then
P(det H < 0) = P(D² + X² > T²) = E_T[ P(D² + X² > T² | T) ],
and conditional on T = t (Gaussian conditioning, Σ̃ ≻ 0), D²+X² is (after whitening the conditional
2D Gaussian) a noncentral chi-square with 2 d.f.; its tail is the Marcum Q-function Q₁(λ(t), |t|/σ), which
Arb can enclose at any precision. The remaining 1D integral over t has an analytic integrand and is
enclosable by DE quadrature (Q2(c)); crude but fully rigorous two-sided bounds come from Laurent–Massart
applied to the whitened (D,X) plus Gaussian Mills-ratio bounds for T. E[det H] > 0 enters only through the
location of the region T² > D²+X²; no normal approximation is involved anywhere. For the *centered* GOE
special case the law of det is closed-form (product-of-eigenvalues formulas; classical Mehta, *Random
Matrices*), but the general (ν, Σ) case is best handled by the conditioning route above — [DERIVED] from
standard Gaussian conditioning, exact.

---

## Q5. Gaussian tube/supremum inequalities with explicit constants (γ-LOC tube clearance, P-NMZ-γ class)

**(a) Borell–TIS (non-asymptotic, explicit variance proxy).** C. Borell, "The Brunn–Minkowski inequality
in Gauss space", Invent. Math. 30 (1975) 207–216; B. Tsirelson, I. Ibragimov, V. Sudakov, "Norms of Gaussian
sample functions", Proc. 3rd Japan–USSR Symp. Probab. Theory, Lecture Notes in Math. 550 (1976) 20–41
[STANDARD]. Clean modern statement (e.g. Adler–Taylor 2007, Theorem 2.1.1; Boucheron–Lugosi–Massart,
*Concentration Inequalities*, Oxford 2013, Thm 5.8 [VERIFIED as quoting source]): for a centered, a.s.
bounded Gaussian field on T with σ²_T = sup_{t∈T} Var f(t),
P(sup_T f − E sup_T f ≥ u) ≤ exp(−u²/(2σ²_T)) (one-sided; also two-sided with the median).
Constants are exact (1 and 1/2); the only non-explicit input is E sup_T f, handled by (b).

**(b) Dudley metric-entropy bound with explicit constants.** R. M. Dudley, "The sizes of compact subsets of
Hilbert space and continuity of Gaussian processes", J. Funct. Anal. 1 (1967) 290–330; explicit-constant
form: P. Massart, *Concentration Inequalities and Model Selection* (Saint-Flour 2003), Springer LNM 1896,
2007, Theorem 3.18 — E sup ≤ 12 ∫₀^{σ_X} √log N(ε) dε with N the packing number in the canonical metric
d(s,t) = (E|f(s)−f(t)|²)^{1/2} [VERIFIED: constant 12 confirmed via the GP-bandit paper quoting Massart
Thm 3.18]; R. van Handel, *Probability in High Dimension* (2014/2016), Corollary 5.25 (same 12) and
**Proposition 5.35 (local chaining tail)**: P(sup_{B(t₀,r)} (X_t − X_{t₀}) ≥ C∫₀^r √log N(ε) dε + x) ≤
C exp(−x²/(Cr²)), C universal (traceable from the proof) [VERIFIED]. For our C¹ (indeed analytic)
stationary field, d²(s,t) = 2(1−K(s−t)) ≤ λ₂|s−t|² with λ₂ the second spectral moment, so covering numbers
of a tube of radius γ around a curve are bounded by volume arguments with explicit constants; the Dudley
integral is then a computable function of (length, γ, spectral moments). Everything E-sup-side of P-NMZ-γ is
therefore certifiable with printed constants.

**(c) Pickands/Piterbarg exact asymptotics (for the asymptotic-shape premise, not the enclosure).**
J. Pickands III, "Upcrossing probabilities for stationary Gaussian processes", Trans. Amer. Math. Soc. 145
(1969) 51–73, and "Asymptotic properties of the maximum in a stationary Gaussian process", ibid. 75–86;
V. I. Piterbarg, *Asymptotic Methods in the Theory of Gaussian Processes and Fields*, Translations of
Mathematical Monographs 148, AMS, 1996 [VERIFIED]. For a stationary a.s.-smooth field with covariance
r(t) = 1 − C|t|² + o(|t|²) (our α = 2 smooth case), P(sup_{[0,T]} f > u) ~ H₂ · T · C^{1/2} · u² Ψ(u)
(H₂ = 1/√π in the smooth limit; Rice form) — the moving-tube version: T. L. Mikhaleva & V. I. Piterbarg,
"On the distribution of the maximum of a Gaussian field with constant variance on a smooth manifold",
Theory Probab. Appl. 41 (1997) 367–379, and **V. Piterbarg & S. Stamatovich, "On maximum of Gaussian
non-centered fields indexed on smooth manifolds", in *Asymptotic Methods in Probability and Statistics with
Applications*, Birkhäuser, 2001, 189–203** [VERIFIED] — the non-centered (drifted) field on a manifold is
exactly the γ-LOC situation (field conditioned/pinned, supremum over a tube shrinking onto a path with a
unique variance maximum; Piterbarg-type exact asymptotics with the "Piterbarg constant" replacing Pickands'
when the trend and curvature scales coincide).

**(d) Tube method with error control (two-sided, the closest match to a certificate).** J. Sun, "Tail
probabilities of the maxima of Gaussian random fields", Ann. Probab. 21(1) (1993) 34–71 [VERIFIED]:
approximates P(sup_T f ≥ u) by the Weyl tube volume (κ₀Ψ(u) + curvature terms) with an explicit error
bound in terms of the critical radius of the index set — two-sided, and the error term is a computable
geometric quantity. A. Takemura & S. Kuriki, "On the equivalence of the tube and Euler characteristic
methods…", Ann. Appl. Probab. 12 (2002) 768–796; J. Taylor, A. Takemura, R. Adler, "Validity of the
expected Euler characteristic heuristic", Ann. Probab. 33 (2005) 1362–1396 (the EEC approximation error is
o(Ψ(u))-class, i.e. asymptotic, not an enclosure); J. E. Taylor, "A Gaussian kinematic formula", Ann.
Probab. 34 (2006) 122–158 [all VERIFIED]. Verdict: the tube method is the right *asymptotic* justification
for the P-NMZ-γ growth shape; the certified bound itself should be (a)+(b).

---

## Q6. Structural stability / quantitative continuation for critical points (frozen-rung → r-uniform)

**(a) Newton–Kantorovich with explicit radii.** L. V. Kantorovich, "Functional analysis and applied
mathematics", Uspekhi Mat. Nauk 3(6) (1948) 89–185 (English transl. NBS Report 1509, 1952) [STANDARD];
W. B. Gragg & R. A. Tapia, "Optimal error bounds for the Newton–Kantorovich theorem", SIAM J. Numer. Anal.
11(1) (1974) 10–13 [VERIFIED]. Standard quantitative form (e.g. Kelley, *Iterative Methods for Linear and
Nonlinear Equations*, SIAM 1995): if ‖F'(x₀)⁻¹‖ ≤ β, ‖F'(x₀)⁻¹F(x₀)‖ ≤ η, and F' is γ-Lipschitz on a ball
of radius ≥ r₋ around x₀ with βηγ ≤ 1/2, then a unique root exists in B(x₀, r₋),
r₋ = (1 − √(1−2βηγ))/(βγ), with all Newton iterates confined to B(x₀, r₊). Applied with F(x; r) = ∇f_r(x):
β from the certified smallest singular value of the Hessian at the frozen rung, γ from a certified C³ bound,
η from the certified rung-to-rung perturbation; r-uniformity reduces to uniformity of (β, γ, η) over the
finitely many certified rungs plus a covering argument between rungs.

**(b) Certified/interval variants (machine-executable).** R. Krawczyk, "Newton-Algorithmen zur Bestimmung
von Nullstellen mit Fehlerschranken", Computing 4 (1969) 187–201; R. E. Moore, "A test for existence of
solutions to nonlinear systems", SIAM J. Numer. Anal. 14 (1977) 611–615; S. M. Rump's uniqueness extension
(1983); consolidated as the **Krawczyk–Rump theorem in A. Neumaier, *Interval Methods for Systems of
Equations*, Cambridge Univ. Press, 1990, Theorem 5.2** [VERIFIED via quoting sources]: if the Krawczyk
operator K([x]) ⊂ int([x]) then [x] contains a unique zero. Radii-polynomial variant: A. Hungria,
J.-P. Lessard, J. D. Mireles James, "Rigorous numerics for analytic solutions of differential equations:
the radii polynomial approach", Math. Comp. 85 (2016) 1427–1459 [VERIFIED]. These are exactly the tools the
campaign's "Task 4c Kantorovich enclosures" already uses; (b) is the machine-executable form of (a).

**(c) Kupka–Smale / structural stability context.** I. Kupka, "Contribution à la théorie des champs
génériques", Contrib. Differential Equations 2 (1963) 457–484; S. Smale, "Stable manifolds for differential
equations and diffeomorphisms", Ann. Scuola Norm. Sup. Pisa 17 (1963) 97–116; M. Peixoto, "Structural
stability on two-dimensional manifolds", Topology 1 (1962) 101–120; monograph: J. Palis & W. de Melo,
*Geometric Theory of Dynamical Systems*, Springer, 1982 [STANDARD]. Content: nondegenerate critical points
with distinct critical values and no saddle connections are generic (Kupka–Smale) and structurally stable
under C¹-small perturbations on surfaces; for the *quantitative* persistence needed here, the Morse–Smale
stability theorem only asserts existence of a C¹ neighborhood — the explicit radius must come from (a)/(b),
with Sard-type measure control for the critical-value separation. Genericity of the random field itself
(no saddle–saddle connections, no degenerate critical points a.s.) follows from Q1(d) Bulinskaya-type
arguments, not from Kupka–Smale.

---

## Q7. Kriging/kernel prediction: residual variance after conditioning on an order-k jet

**Claim checked:** v(x) := Var(f(x) | k-jet of f at x₀) ∝ r^{2k+2} as r = |x − x₀| → 0, for an analytic
stationary field with nondegenerate jets. **Confirmed in form and constant** [DERIVED, elementary]:
write the Taylor expansion of f(x) about x₀ to order k with remainder of order k+1; conditioning on the
full k-jet kills all terms of order ≤ k exactly (they are pinned), so the residual is the order-(k+1)
Taylor polynomial in the *conditional* (k+1)-jet plus o(r^{k+1}); hence
v(x₀ + h) = (h^{⊗(k+1)})ᵀ Σ_{k+1|k} (h^{⊗(k+1)}) / ((k+1)!)² + o(|h|^{2k+2}),
where Σ_{k+1|k} is the conditional covariance of the (k+1)-jet given the k-jet (positive definite by full
spectral support; computable by the Kronecker/Schur formula, manuscript Preliminary 2.3). In particular
v ∝ r^{2k+2} with a computable angular coefficient, and v(x₀+h) = det G_{≤k+1}/det G_{≤k} in 1D (Schur
complement ratio of jet Gram determinants) — exactly the determinantal structure the manuscript's §3.4
already computes at 80–120 digits. Nondegeneracy hypotheses are the same jet-Gram positive-definiteness
already certified (manuscript Lemma 3.1 / Def. 2.2).

**Framework citations (native space / power function):**
- H. Wendland, *Scattered Data Approximation*, Cambridge Monographs on Applied and Computational
  Mathematics 17, Cambridge Univ. Press, 2005 — Ch. 11: the power-function error identity
  |f(x) − s_{f,X}(x)| ≤ P_{K,X}(x) ‖f‖_{N_K} (Thm 11.4-class), with P²_{K,X}(x) = K(x,x) − k_X(x)ᵀK_X⁻¹k_X(x)
  = exactly the kriging residual variance; convergence orders in fill distance for smooth kernels
  (Thm 11.13-class; for analytic/Gaussian kernels, faster-than-algebraic in h for *scattered* data, but the
  single-point jet conditioning above is a different, purely local regime where the r^{2k+2} law governs)
  [STANDARD].
- F. J. Narcowich, J. D. Ward, H. Wendland, "Sobolev bounds on functions with scattered zeros, with
  applications to radial basis function surface fitting", Math. Comp. 74 (2005) 743–763 [STANDARD] — the
  "zeros lemma"/sampling inequality: functions whose jets vanish to order k on a set with fill distance h
  satisfy ‖u‖ ≤ C h^{k+1} (appropriate norms); this is the rigorous general-position result behind
  "order-k jet pinning ⇒ error O(h^{k+1})", i.e. variance O(h^{2k+2}).
- Z. Wu & R. Schaback, "Local error estimates for radial basis function interpolation of scattered data",
  IMA J. Numer. Anal. 13 (1993) 13–27; R. Schaback, "Improved error bounds for scattered data interpolation
  by radial basis functions", Math. Comp. 68 (1999) 201–216 [STANDARD] — local (near-data) refinements of
  the power-function bounds, the right class for the clustered-pin geometry (three r-clustered stations
  acting as an effective jet).
- M. L. Stein, *Interpolation of Spatial Data: Some Theory for Kriging*, Springer, 1999 — Ch. 3: local
  behavior of kriging variances and equivalence of local predictions; M. Scheuerer, R. Schaback,
  M. Schlather, "Interpolation of spatial data — a stochastic or a deterministic problem?", J. Approx.
  Theory 2013 (preprint 2009) [STANDARD] — the kriging = kernel-interpolation = power-function equivalence
  used to pass between conditional-variance and approximation-error language.
- Applied derivative-observation GP literature (for the engineering formulation): E. Solak et al.,
  "Derivative observations in Gaussian process models of dynamic systems", NeurIPS 2003 [STANDARD].

**Caveat [flag]:** I did not find a single named theorem stating verbatim "v(x) ∝ r^{2k+2} for jet
conditioning"; the statement above is a *derivation* from Gaussian conditioning + Taylor expansion +
positive-definiteness of the (k+1)-jet Gram matrix, with the Wendland/Narcowich–Ward framework providing
the matching general estimates. For the manuscript this is sufficient — the constant is computed exactly
from certified Gram determinants anyway — but the citation should be phrased as "by Gaussian conditioning
and Taylor expansion (cf. Wendland 2005 Ch. 11; Narcowich–Ward–Wendland 2005)" rather than attributed to a
specific theorem number.

---

## Q8. METHODS RECOMMENDATION

### 8.1 WP enclosure (rigorous upper bound for the typed∩window Kac–Rice integrand)

1. **Foundation.** Cite (KR-I) from AW09 Thm 6.2/6.4 (weights include the joint indicator of Hessian type
   and value window; Bulinskaya hypothesis by AW09 Prop 6.5 + full spectral support). All subsequent
   inequalities are pointwise bounds on a rigorously defined integrand — this closes the provenance gap that
   allowed the missing-√ defect.
2. **Quadrature (production enclosure).** Transform the 3D conditional Gaussian expectation by the
   whitening map; split the domain across the quadric det H = 0 (so the integrand — |det|·(mollified
   window) — is analytic on each piece); mollify the window indicator with an analytic bump of width δ and
   bound the mollification error by explicit Gaussian tails (Laurent–Massart Lemma 1 on the relevant
   chi-square pieces, plus Mills-ratio bounds; error O(δ/σ_boundary)). Then apply **tensor-product DE
   (tanh-sinh) quadrature with the explicit-constant error bounds of Okayama–Matsuo–Sugihara (Numer. Math.
   124, 2013), evaluated entirely in Arb ball arithmetic at 80–120 digits**, following the verified-DE
   architecture of Yamanaka–Okayama–Oishi–Ogita (IEICE NOLTA 1, 2010). The strip-width and decay parameters
   of the transformed integrand are certified by Taylor-model bounds (Makino–Berz) on each tensor cell.
3. **Independent cross-check.** High-order tensor Gauss–Hermite with the Krylov Peano remainder
   f^{(2n)}-bounds certified by interval AD, or the Kuo–Scheichl–Schwab–Sloan–Ullmann Hermite-norm bound
   (Math. Comp. 86, 2017, Thm 8.1-class). Require the two enclosures to overlap; the intersection is the
   certified value.
4. **Tail bounds.** For quadratic-form pieces (det H, boundary layers, smoothing errors): Laurent–Massart
   (2000) Lemma 1 for chi-square; Latała (2006) for indefinite Gaussian chaos where shape matters; the
   Rudelson–Vershynin (2013) Hanson–Wright only with an explicit-constant version (arXiv:2309.03873
   Thm 2.1: 144, 16√2; or Ziemann arXiv:2509.00881). For P(det H < 0)-type quantities use the exact
   T²−D²−X² conditioning to 1D Marcum-Q/Φ integrals enclosed in Arb — no concentration inequality needed
   at 2×2.
5. **Interval framework.** Arb (Johansson 2017) as the ball-arithmetic engine at 80–120 digits
   (IEEE 1788-2015 as the cited semantic standard); approximate-then-certify matrix inversion via interval
   residual bounds (Neumaier 1990) for all Gram inversions in the Kronecker conditioning — **do not use
   mpmath `iv` for matrix inversion** (no rigorous pivoting/condition control; dependency blow-up on
   near-singular pin Gram matrices). mpmath high-precision floats remain acceptable for approximate
   quantities that are later certified. Genz (1992/2004) QMC values serve as measured-grade cross-checks
   only; at d=3 all needed MVN rectangle probabilities reduce by the Genz transform to iterated 1D
   integrals of Φ-products, enclosable by item 2.

### 8.2 γ-LOC tube premise (P-NMZ-γ)

- **Certified (non-asymptotic) clearance bound:** Borell–TIS (Borell 1975; TIS 1976; AT07 Thm 2.1.1 form)
  with σ²_T = sup variance over the tube computed from the conditional covariance, and E sup_T f bounded by
  the explicit-constant Dudley integral (Massart LNM 1896 Thm 3.18, constant 12; van Handel Cor. 5.25 /
  Prop. 5.35 for the local chaining tail), with covering numbers of the γ-tube bounded via
  d²(s,t) ≤ λ₂|s−t|² and volume packing. This yields P(sup over tube − trend ≥ x) ≤ exp(−(x−E)₊²/(2σ²_T))
  with every constant printed — the correct rigorous skeleton for P-NMZ-γ at fixed (r, γ).
- **Asymptotic-shape justification (motivation, not certificate):** the tube method (Sun, Ann. Probab. 21,
  1993 — two-sided tube-volume approximation with explicit geometric error) for the γ-dependence, and
  Piterbarg–Stamatovich (2001)/Mikhaleva–Piterbarg (1997) for the exact drifted-field-on-a-manifold
  asymptotics when the tube clearance grows; Pickands (1969)/Piterbarg (1996) for the stationary backbone.
- **Continuation/structural-stability inputs** (frozen-rung → r-uniform): Kantorovich (1948) with
  Gragg–Tapia (1974) optimal radii in analytic form; machine-executed as the Krawczyk–Rump inclusion
  (Neumaier 1990, Thm 5.2) or radii polynomials (Hungria–Lessard–Mireles James 2016), with β, γ, η
  certified per rung and r-uniformity by covering. Kupka–Smale (1963)/Palis–de Melo (1982) cited only for
  genericity context; quantitative radii come from Kantorovich/Krawczyk, not from structural stability.

### 8.3 Jet-rigidity (prediction-horizon) inputs

The §3.4 effective-jet / rigidity statements are supported by Q7: residual variance after conditioning on
an order-k jet is ∝ r^{2k+2} with constant = Schur-complement ratio of certified jet Gram determinants
(derivation above; framework Wendland 2005 Ch. 11, Narcowich–Ward–Wendland 2005, Wu–Schaback 1993,
Stein 1999). This converts the measured "rigidity" observation into a provable two-sided law at the
manuscript's existing certification grade.

---

## Appendix. Confidence and negative results

- [VERIFIED] theorem numbers: AW09 Thm 6.2/6.3/6.4; AT07 Thm 11.2.1; AAL arXiv:2304.07424 Thm 6.1 +
  Remark 8; Laurent–Massart Lemma 1; Rudelson–Vershynin ECP 18 (2013) no. 82; explicit-constant HW
  (arXiv:2309.03873 Thm 2.1); Massart Thm 3.18 (constant 12); van Handel Cor 5.25/Prop 5.35;
  Okayama–Matsuo–Sugihara Numer. Math. 124 (2013); Yamanaka et al. IEICE NOLTA 1 (2010); Johansson Arb
  IEEE TC 66(8) (2017) 1281–1292; Genz JCGS 1(2) (1992) 141–149; Genz & Bretz LNS 195 (2009);
  Gragg–Tapia SIAM J. Numer. Anal. 11(1) (1974) 10–13; Kuo et al. Math. Comp. 86 (2017) GH error theorem;
  Sun Ann. Probab. 21(1) (1993) 34–71; Takemura–Kuriki AAP 12 (2002); Taylor–Takemura–Adler Ann. Probab.
  33 (2005); Piterbarg TMM 148 (1996); Mikhaleva–Piterbarg TPA 41 (1997) 367–379; Piterbarg–Stamatovich
  (2001); Krawczyk–Rump = Neumaier Thm 5.2; Hungria–Lessard–Mireles James Math. Comp. 85 (2016) 1427–1459.
- Numbering uncertainty (minor): Bulinskaya lemma in AW09 cited variously as Prop 6.5 (a.s.-Morse for
  Gaussian fields) and Prop 6.11 (general bounded-density version); recommend citing Prop 6.5 for the
  Gaussian case used here.
- Negative result: no single named theorem was found stating verbatim the r^{2k+2} jet-conditioning law
  (Q7); it is a short derivation (given) sitting inside the Wendland/Narcowich–Ward power-function
  framework. Also negative: no *rigorous* off-the-shelf multivariate normal CDF library exists; at d=3 the
  Genz transform + verified DE integration is the defensible route, and Genz–Bretz randomized QMC is not
  certificate-grade.
