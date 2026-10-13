# Claim-by-claim check of an external note on zeta zeros and the de Bruijn–Newman flow

**Exploratory. Scientific effect: NONE. No status change.** Companion to [`README.md`](README.md).

The note is an external note forwarded by the project owner on 12 October 2026. It is not
reproduced here and its author is not named. Each of its claims is **paraphrased** in the
second column; the first column gives the note's own section number so that a reader holding
the note can find the passage. Nothing in the note, or in this check, is progress toward RH or
a bound on Λ beyond known results.

**Verdicts.** **C** correct · **CC** correct with a caveat · **I** imprecise · **W** wrong.

**Citation tags.** [snippet]: bibliographic details checked only against search-engine
snippets (arXiv, journal and encyclopedia pages could not be fetched: proxy 403 / DNS).
[memory]: recalled from memory, unchecked. [num]: re-checked numerically in this packet
(mpmath 1.4.1, high precision, not interval-certified); the script is named.

Notation: α is the exponent of the pair or gap law near 0 (density ∝ r^α); β = (α+1)/m only in
the sense of Math- `reviews/collision_mechanism_20260925/NOTE.md` equation (D1); k is the fold
coefficient in ℓ = k r³ (κ in the Math- note).

## Table

| # | The note's claim (paraphrased) | Verdict | Corrected statement and evidence |
|---|---|---|---|
| 0a | It announces two links between zeta zeros and the project, then lists three. | I | Three links are listed: folds ↔ heat-flow collisions, lattice ↔ theta ↔ ζ, GUE ↔ contact law. |
| 0b | It presents the links as genuine connections and disclaims that they show the project proves RH. | C | Agreed; nothing below changes this. |
| 1.1 | ζ is defined by the Dirichlet series for Re s > 1 and continued. | CC | ζ continues meromorphically to ℂ; its only singularity is a simple pole at s = 1 with residue 1 [Riemann 1859; Titchmarsh ch. II; memory]. |
| 1.2 | The nontrivial zeros lie in the open strip 0 < Re s < 1. | C | A theorem: ζ(1+it) ≠ 0 (Hadamard; de la Vallée Poussin 1896) plus the functional equation. The trivial zeros are at −2, −4, …. |
| 1.3 | RH asserts that all nontrivial zeros lie on Re s = ½. | C | Agreed. |
| 2.1 | The fold normal form x³/3 − ax has critical points ±√a that merge at a = 0. | C | This is the A₂ fold, i.e. Cerf birth–death of a critical pair. |
| 2.2 | The height gap between the two critical points scales like the cube of their separation. | C | Exactly f(−√a) − f(√a) = (4/3)a^{3/2} = r³/6 with r = 2√a. In the project's convention ℓ = k r³ with k = \|f'''\|/12, so this normal form has k = 1/6 (`test_exact.py`). |
| 2.3 | Combining the cubic gap with the probability of configurations gives the project's exponent. | CC | −1/3 comes from the pair intensity ∝ r (the ledger r A_r dr) pushed through ℓ = k r³, with admissible marks and a positive pairing limit. Geometry alone does not fix it: subsection §D1 of the Math- note shows cubic splitting need not give a density coefficient, and subsection §D2 shows the exponent changes when the mark density blows up near 0. |
| 3.1 | It defines the de Bruijn–Newman family H_t(z) = ∫_0^∞ e^{tu²} Φ(u) cos(zu) du. | CC (incomplete) | Φ(u) = Σ_{n≥1}(2π²n⁴e^{9u} − 3πn²e^{5u}) exp(−πn²e^{4u}) (Rodgers–Tao / Polymath15 conventions [snippet]). H_0(z) = Ξ(z/2)/8 with Ξ(x) = ξ(½+ix) [num, `dbn/check_dbn_basic.py`: relative difference ≤ 7·10⁻⁴¹; three implementations, README §5.1]. Φ is even, positive for u ≥ 0 termwise (2π > 3) and hence everywhere by evenness, and Φ(u) = e^u[2x²ψ''(x) + 3xψ'(x)] at x = e^{4u}, ψ = (Θ−1)/2 [num]. So the kernel is a theta derivative and its evenness is the theta transformation. |
| 3.2 | At t = 0, H_t is the completed zeta function in coordinates where RH means all zeros are real. | C | H_0(z) = ξ(½ + iz/2)/8, so the zero ρ = ½ + iγ corresponds to z = 2γ. |
| 3.3 | H_t solves a backward heat equation. | C | ∂_tH = −∂_z²H, differentiating under the integral (justified by Φ(u) = O(e^{9u−πe^{4u}})). [num] finite-difference residual ≤ 4·10⁻²⁷ against terms of size 10⁻⁴ to 10⁻¹⁰. |
| 3.4 | Zeros move with t; real zeros approach and collide, and with time reversed they leave the axis. | I (direction) | For a simple zero the exact velocity is ż_j = H_zz/H_z, which through the Hadamard product equals 2Σ_{k≠j} 1/(z_j − z_k) [Csordas–Smith–Varga 1994, snippet; used by Rodgers–Tao], summed symmetrically (not absolutely convergent), e.g. ż_j = 2[1/(2z_j) + Σ_{k≠j} 2z_j/(z_j² − z_k²)] over positive zeros. As t increases zeros repel; as t decreases neighbours attract, collide at a real double zero and become a conjugate pair. [num] at t = 0, H_zz/H_z = −0.5796306, −0.7549955, −0.4654455 for zeros #1, #2, #5 (`dbn/zero_velocity.py`), matching finite differences; the 300-zero symmetric sum (−0.579717, −0.755108, −0.465554) is truncation-limited at about 1.5·10⁻⁴ and rules out the opposite sign. |
| 3.5 | (Introduction) The zeros of Riemann's xi function collide under a heat flow. | I | It is the zeros of H_t that collide, and only at t ≤ Λ; for t > Λ all zeros are simple [CSV 1994; snippet]. No multiple zero of Ξ itself (t = 0) is known. Computed example: a double zero at z₀ = 99.328469305575750948, t₀ = −1.877909072972535264767 (zeros 9/10 of H_0). Collisions at negative time say nothing about RH, since Λ ≥ 0 is proved. |
| 3.6 | All zeros of H_t are real exactly when t ≥ Λ. | C | de Bruijn 1950: real for t ≥ 1/2, and real at t implies real at every t' > t. Newman 1976: Λ exists and Λ ≤ 1/2. Ki–Kim–Lee 2009: Λ < 1/2, and for every t > 0 all but finitely many zeros are real and simple [all snippet]. The time variable rescales by 4 between normalizations (in the Ξ(x) variable de Bruijn's 1/2 becomes 1/8). |
| 3.7 | RH is equivalent to Λ ≤ 0. | C | From 3.2 and 3.6. |
| 3.8 | Rodgers–Tao proved Λ ≥ 0, so RH is equivalent to Λ = 0. | C | Rodgers–Tao, Forum Math. Pi 8 (2020) e6, arXiv:1801.05914 [snippet]. Method: assuming Λ < 0, run the zero dynamics with a mollified energy to show H_0's zeros would be in local equilibrium, contradicting known local statistics of zeros such as Montgomery's pair-correlation estimates (per the abstract [snippet]). |
| 3.9 | (Background asked for) Upper bounds on Λ. | C | Polymath15, Res. Math. Sci. 6 (2019) art. 31, arXiv:1904.12438: Λ ≤ 0.22, from de Bruijn's lemma (zeros of H_{t0} in \|Im z\| ≤ y0 ⇒ Λ ≤ t0 + y0²/2) [snippet]. Platt–Trudgian, Bull. LMS 53 (2021): Λ ≤ 0.2 [snippet]. |
| 3.10 | (Background asked for) Lower bounds on Λ from Lehmer pairs. | C | Csordas–Smith–Varga, Constr. Approx. 10 (1994) 107–129; Csordas–Odlyzko–Smith–Varga, ETNA 1 (1993) Λ > −5.895·10⁻⁹; Odlyzko, Numer. Algorithms 25 (2000) Λ > −2.7·10⁻⁹; Saouter–Gourdon–Demichel, Math. Comp. 80 (2011) Λ > −1.14541·10⁻¹¹ [all snippet]; Csordas–Norfolk–Varga 1988 (−50) and te Riele 1991 (−5) [memory]. Mechanism: a very close pair at t = 0 must have collided at some t_c < 0 near 0, so Λ ≥ t_c. Two-body heuristic ṙ ≈ 4/r gives r² ≈ 8(t − t_c), consistent with 3.11. CSV make this rigorous with a condition on the other zeros; their exact inequality was not reproduced. |
| 3.11 | Near a collision, H ≈ A(x² − 2τ), with zeros at ±√(2τ). | C (side unspecified) | At a double zero H = H_z = 0 and A = H_zz/2 ≠ 0; the PDE gives H_t = −2A and H_zt = −H_zzz, so H = A(x² − 2τ) + (H_zzz/6)(x³ − 6xτ) + O(x⁴ + x²\|τ\| + τ²). The zeros are real for τ > 0 and a conjugate pair z₀ + O(\|τ\|) ± i√(2\|τ\|) for τ < 0, whatever the sign of A. The pair's midpoint drifts by 2(B/A)τ, so ±√(2τ) is leading order only (README Proposition 1; κ_r = 4C/A − 2B²/A²). [num] at the computed collision r²/(8τ) = 0.999008, 0.9999007, 0.99999007, 0.9999999 for τ = 10⁻², 10⁻³, 10⁻⁴, 10⁻⁶; Im z/√(2\|τ\|) → 1 below t₀. |
| 3.12 | The antiderivative F_t has the two merging zeros as its new critical points. | C | F ≈ A(x³/3 − 2τx) + c for τ > 0. |
| 3.13 | The new bar has height ≈ \|A\| r³/6. | C | F(x₋) − F(x₊) = (4/3)A(2τ)^{3/2} = A r³/6 with r = 2√(2τ); so ℓ = (8√2/3)\|A\|τ^{3/2}(1 + κ_tτ + …) and ℓ = (\|A\|r³/6)(1 + κ_ℓτ + …), κ_ℓ = 6B²/A² − 48C/(5A) (`test_exact.py`, exact). [num] area/(\|A\|r³/6) = 1.0045, 1.00045, 1.000045, 1.00000045. Superlevel H0 persistence of the sampled F_t at τ = +0.01 has exactly one finite bar in the window, length 1.0445·10⁻¹⁸ (exact area 1.0450·10⁻¹⁸); at τ = −0.01 there is none (`dbn/check_collision_persistence.py`). |
| 3.14 | This follows analytically near a nondegenerate double zero. | C | Nondegenerate means H_zz ≠ 0. |
| 3.15 | Such collisions are studied in the literature; the new ingredient is the persistence description. | CC | The literature is 3.10 plus Rodgers–Tao (dynamics, local equilibrium) and Polymath15. The persistence phrasing is itself standard birth–death in a generic one-parameter family [Cerf 1970; Cohen-Steiner–Edelsbrunner–Morozov 2006 (vineyards); memory]. No new mathematics. |
| 4.1 | The project's Fourier weights are normalized Gaussians e^{−ω²\|n\|²/2}/Σ_j e^{−ω²\|j\|²/2}, ω = 2π/L. | C | Matches `periodized_gaussian_h0/PROOF.md` lines 44–47. |
| 4.2 | This is the same Gaussian-lattice construction as Jacobi's theta function. | CC | Exactly Σ_j e^{−ω²\|j\|²/2} = Θ(2π/L²)^d, and Poisson/Jacobi gives the image normalizer [num, `theta/check_theta.py`, `theta/project_theta_checks.py`]. The covariance is K_L(z) = Π_i ϑ₃(πz_i/L, q)/ϑ₃(0, q), q = e^{−2π²/L²}. **Caveat:** at L = 24 the periodization changes the jets at a point by O(e^{−L²/2}poly(L)) ≈ 10⁻¹²⁵ (K'', K⁗, K⁽⁶⁾ at 0 = −1, 3, −15 to working precision), so the local fold law sees the planar squared-exponential kernel; the project uses Poisson summation only to get positive Fourier weights. |
| 4.3 | Riemann's Mellin formula expresses π^{−s/2}Γ(s/2)ζ(s) through Θ − 1. | C | Valid for Re s > 1 [num at s = 2, 3, 4.5, 2+3i, 18 digits]; for Re s ≤ 1 the integral diverges at u → 0 (integrand ~ u^{(s−3)/2}); continuation needs Riemann's split at u = 1. |
| 4.4 | The theta transformation yields the functional equation, citing Mumford. | CC | Correct, with attribution fixed: the transformation Θ(u) = u^{−1/2}Θ(1/u) is classical (Jacobi; Poisson summation); its use to prove ξ(s) = ξ(1−s) is Riemann 1859 (second proof); Mumford's Tata Lectures I is expository [all memory]. [num] transformation at u = 0.37, 2, 0.05; Riemann's continued form checked at s = 0.3+5i and s = ρ₁. |
| 4.5 | Higher-dimensional lattice sums lead to Epstein zeta functions. | C | ∫_0^∞(Θ^d − 1)u^{w−1}du = π^{−w}Γ(w)Z_d(w) for **Re w > d/2**, with Λ_d(w) = Λ_d(d/2 − w). Z_2 = 4ζ(w)L(w, χ₋₄), Z_4 = 8(1 − 4^{1−w})ζ(w)ζ(w−1) [num]. |
| 4.6 | A functional equation alone does not locate zeros. | C (sharpened) | Z_4 has zeros at ½ + iγ and 3/2 + iγ, off its symmetry line Re w = 1 [num at γ₁]. Z_4 does have an Euler product; it is an imprimitive, shifted product, and shows that a theta functional equation does not force zeros onto the symmetry line for such a product. Epstein zetas of positive-definite integral **binary** quadratic forms with class number > 1 have infinitely many zeros in Re s > 1 [Davenport–Heilbronn 1936; snippet]; this does not apply directly to ℤ^d. The heat flow also destroys the Euler product for t ≠ 0 (Polymath15's coefficients exp((t/4)log²n) are not multiplicative). |
| 4.7 | (Not in the note; added) Mellin transform of the project's normalizer over the side length. | elementary restatement | ∫_0^∞(N_F(L, d) − 1)L^{−s−1}dL = ½(2π²)^{−s/2}Γ(s/2)Z_d(s/2) for **Re s > d**; for d = 1 it is (2π²)^{−s/2}Γ(s/2)ζ(s). This is Riemann's formula after u = 2π/L²; novelty not claimed [num: d = 1 at s = 3, 2+i; d = 2 at s = 3; d = 4 at s = 6]. ζ appears only after integrating over all torus sizes, which the project holds fixed. |
| 4.8 | The theta link is an analytic link, not a proof link. | C | Agreed. |
| 5.1 | The GUE pair correlation 1 − (sin πr/πr)² behaves like (π²/3)r² at small r. | C | This is Montgomery's pair-correlation conjecture (Dyson/GUE); series (π²/3)r² − (2π⁴/45)r⁴ + … [num, `stats/check_gue_transfer.py`]. The GUE nearest-neighbour spacing density has the same leading term; the Wigner surmise gives 32/π². |
| 5.2 | The project's contact intensity is proportional to r. | C | The four-factor ledger in `fold_structural_universality/PROOF.md` lines 390–404. In 1D the zeros of f' for a stationary Gaussian process repel linearly, and the project's 1D theorem gives ν = C₀ℓ^{−1/3} + …. |
| 5.3 | GUE statistics for zeta zeros remain conjectural. | C | Montgomery 1973 (assuming RH): pair correlation for test functions with Fourier support in (−1, 1) [snippet]. Rudnick–Sarnak 1996: n-level correlations with restricted support [snippet]. Odlyzko 1987 and later: numerical support [snippet]. The r → 0 law is not proved. |
| 5.4 | Proposed test: with ℓ ∝ r³ and gaps ∝ r² dr, short bars should follow ℓ⁰ rather than ℓ^{−1/3}. | CC | The computation is right: ν(ℓ) = (c/3)E[k^{−(α+1)/3}]ℓ^{(α−2)/3}, giving ℓ⁰ for α = 2 and ℓ^{−1/3} for α = 1 (`test_exact.py`). **Missing admission condition:** E[k^{−(α+1)/3}] < ∞, i.e. E[k^{−1}] for α = 2. With mark density ∝ k^q near 0, subsection §D2 (α = 2, m = 3) gives ℓ⁰ if q > 0, log(1/ℓ) if q = 0 and ℓ^q if −1 < q < 0 (exact: k ~ U(0,1) gives ν = log(1/ℓ); `test_exact.py`, `stats/check_gue_transfer.py`). **Numerics** (README §5.3, exploratory): zeta short bars are decisive against −1/3 and consistent with ≈ 0 given the finite-cutoff bias seen in controls. |
| 5.5 | The key question is which parts of the law are geometry and which are measure. | C | This is the project's own stated question (`docs/PERSISTENCE_UNIVERSALITY_PROGRAM.md` lines 3–6 and 76–87). Subsections §D1/§D2 already answer it in the product-intensity model: m = 3 is geometry; α and q come from the measure; exponent (α+1)/m − 1 when q + 1 > β. |
| 5.6 | (Added) What the antiderivative's persistence actually measures. | — | A bar is F(max) − max(left valley, right valley): a signed sum of lobe areas, a single lobe only when the death point is adjacent. It depends on the positive normalization and on the sign of H. [num, `stats/check_Z_persistence.py`] Hardy Z on [600, 900]: 228 zeros (exact `nzeros` count 228); 113 bars, 80.5% single-lobe; log(lobe area) on log(gap) slope 2.995. On [5000, 5200]: 214 zeros (exact 214), 106 bars, 82.1% single-lobe, slope 3.054. With the Ξ weight 80/113 and 82/106 pairs are shared with the Hardy version; with Ξ's actual sign 0; with a mild e^{0.01t} weight 112/113 and 104/106. These windows (~110 bars) cannot estimate an exponent. |
| 6.1 | The proposed route ends at proving that the zeta heat flow has no positive-time transition, then Λ = 0, then RH. | CC → I | The logic is valid but circular: the absence of a positive-time transition ⇔ Λ ≤ 0 ⇔ RH. The step is equivalent to RH, not merely related to it. |
| 6.2 | (Implicit) If Λ > 0, the transition at Λ is a real collision. | plausible, unverified | It would follow if Ki–Kim–Lee finiteness held locally uniformly in t; that uniformity was not verified. |
| 6.3 | Real-axis persistence may miss zeros off the line. | I | A purely local analysis misses them, but comparing the number of real zeros (critical points of F_t) with N(T) from the argument principle detects them; this is how RH is verified numerically (Riemann–Siegel plus Turing's method; to 3·10¹², Platt–Trudgian [memory]). Our windows match the exact `nzeros` counts (228/228, 214/214). |
| 6.4 | The fundamental obstacle: the project's laws are random while ζ is deterministic, and a local fold says nothing global. | C | Agreed. |
| 7(a) | Proposed study: persistence of F_t on finite intervals, tied to zero events. | CC | Bar births and deaths correspond to real double zeros, plus zeros crossing the interval ends; pairing switches happen at equal critical values with no zero event; the diagram depends on normalization (5.6). For t > Λ there are no folds. **Under RH, fold events occur only at t ≤ 0 (t = 0 only if Ξ has a multiple zero).** Heuristically (Lehmer-pair mechanism) collision times approach 0 from below as height grows; that is not a theorem. |
| 7(b) | Proposed study: invariants or global theta constraints on H_t. | CC | Such constraints exist already: de Bruijn's strip lemma, Newman's monotonicity, Ki–Kim–Lee, and the Rodgers–Tao argument (survey: Newman–Wu, Bull. AMS 57 (2020) [snippet]). De Bruijn and Newman hold for broad kernel classes; Ki–Kim–Lee and Rodgers–Tao use ζ-specific zero information (arithmetic input, not theta structure). The Rodgers–Tao energy argument is a technique for the lower bound, not a global constraint ruling out configurations for t > 0. |
| 7(c) | Proposed study: compare collision statistics with fold theory. | CC | The project's theorems concern random laws on closed manifolds admitted through stated interfaces and do not apply to the deterministic H_t. A random model is needed (README §6.1, T-CUE). |
| 7-table | Summary table: lattice ↔ theta ↔ ζ exact; folds ↔ de Bruijn–Newman exact locally; persistence ↔ RH not established; the exponent does not explain the critical line. | C | Add: the theta link runs through a Mellin transform over scale (4.7), and the local fold correspondence needs H_zz ≠ 0. |

## Errors and imprecisions in the note

1. It announces two links but lists three (0a).
2. The time direction of real-zero collisions is muddled (3.4).
3. The kernel Φ and the 1/8 normalization are missing (3.1).
4. It says the zeros of Ξ collide; it is the zeros of H_t, at t ≤ Λ (3.5).
5. Its decisive step is equivalent to RH (6.1).
6. Off-line zeros are detectable by counting (6.3).
7. The persistence description of collisions is standard Cerf/vineyard material (3.15).
8. The ℓ⁰ test is missing its admission condition, which subsection §D2 already covers (5.4).
9. The global constraints it asks for already exist (7(b)).
10. The theta-transformation attribution is non-primary: the transformation is classical
    (Jacobi; Poisson summation), its use for ξ(s) = ξ(1−s) is Riemann 1859, and Mumford is
    expository [memory] (4.4).

## What is kept, and how it is labelled

- **Classical, re-checked numerically:** every C or CC row with [num].
- **Elementary restatements (novelty not claimed; no literature search):** 4.7 (ζ as the Mellin
  transform of the project normalizer over L); the L = 24 jet estimate (4.2); Proposition 1 in
  the README (fold of a backward-heat solution, with the second-order coefficients κ_r, κ_ℓ,
  κ_t and the midpoint drift, now proved exactly as power series in `test_exact.py`); the
  normalization dependence of the barcode (5.6); the lobe-area identity
  ∫_a^c H_t = ½∫_a^c (x−a)(c−x)∂_tH_t dx.
- **Conjecture (exploratory, conditional):** Conjecture Z, restated in a well-posed form in
  README §6.2 (empirical bar measure per zero on [T, T+H], T → ∞ first, amplitude-normalized
  bars, exponent-only; any coefficient comparison carries an arithmetic-correction caveat and a
  declared CUE matching rule). A random-matrix version (T-CUE, README §6.1) is a research
  target. Neither is a route to RH.
- **Numerics:** README §5. Exploratory only; the data were inspected before any prediction was
  frozen.

## Citation provenance

| Citation | Used for | Status |
|---|---|---|
| Riemann 1859 | functional equation, Mellin formula, second proof via theta | memory |
| Jacobi (theta transformation); Poisson summation | Θ(u) = u^{−1/2}Θ(1/u) | memory |
| Mumford, *Tata Lectures on Theta I* | expository theta source | memory |
| Titchmarsh, *The Theory of the Riemann Zeta-Function*, ch. II | continuation, zero-free line | memory |
| Hadamard; de la Vallée Poussin 1896 | ζ(1+it) ≠ 0 | memory |
| de Bruijn 1950 (Duke Math. J. 17) | real zeros for t ≥ 1/2, strip lemma | snippet |
| Newman 1976 (PAMS 61) | existence of Λ, Λ ≤ 1/2 | snippet |
| Csordas–Smith–Varga 1994 (Constr. Approx. 10) | zero dynamics, Lehmer pairs | snippet |
| Csordas–Odlyzko–Smith–Varga 1993 (ETNA 1) | Λ > −5.895·10⁻⁹ | snippet |
| Odlyzko 2000 (Numer. Algorithms 25) | Λ > −2.7·10⁻⁹ | snippet |
| Saouter–Gourdon–Demichel 2011 (Math. Comp. 80) | Λ > −1.14541·10⁻¹¹ | snippet |
| Csordas–Norfolk–Varga 1988 (−50); te Riele 1991 (−5); Csordas–Ruttan–Varga 1991 (−0.0991) | historical lower bounds | memory (table seen in snippets) |
| Norfolk–Ruttan–Varga 1992 (−0.385) | historical lower bound matched numerically by the (34/35) collision | snippet; pair used not confirmed |
| Ki–Kim–Lee 2009 (Adv. Math. 222) | Λ < 1/2; all but finitely many zeros real and simple for t > 0 | snippet (statement); ζ-specific inputs: memory |
| Rodgers–Tao 2020 (Forum Math. Pi 8) | Λ ≥ 0; method | snippet |
| Polymath15 2019 (Res. Math. Sci. 6) | Λ ≤ 0.22 | snippet |
| Platt–Trudgian 2021 (Bull. LMS 53) | Λ ≤ 0.2 | snippet |
| Platt–Trudgian, RH verified to 3·10¹² | numerical verification height | memory |
| Newman–Wu 2020 (Bull. AMS 57) | survey | snippet |
| Montgomery 1973 (Proc. Sympos. Pure Math. 24) | pair correlation, restricted support | snippet |
| Rudnick–Sarnak 1996 (Duke 81) | n-level correlations | snippet |
| Odlyzko 1987 (Math. Comp. 48) and later | numerical GUE evidence | snippet |
| Davenport–Heilbronn 1936 (JLMS 11) | Epstein zetas of binary forms with zeros in Re s > 1 | snippet |
| Cerf 1970; Cohen-Steiner–Edelsbrunner–Morozov 2006 | birth–death, vineyards | memory |
| Hughes–Keating–O'Connell 2000; Gonek 1989; Hejhal 1989 | negative discrete moments of ζ'(ρ), a(k) | memory |
| Keating–Snaith 2000 | N ≈ log(T/2π) for moments | memory |
| Bogomolny–Bohigas–Leboeuf–Monastra 2006 | effective N for spacing statistics | memory (formula and constant pending check) |
| Selberg's central limit theorem; Radziwiłł–Soundararajan 2017 | log-normal amplitudes | memory |
| Edwards 1974 | Riemann–Siegel correction terms | memory; validated numerically against `mpmath.siegelz` |
| Mathlib (functional equation of ζ via Jacobi theta and Mellin) | Lean infrastructure remark | memory |

A nonauthor reviewer with working web access should confirm every snippet and memory item
(README §9, slice B).
