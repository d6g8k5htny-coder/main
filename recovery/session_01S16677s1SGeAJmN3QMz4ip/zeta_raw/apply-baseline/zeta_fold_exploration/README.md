# Zeta zeros, heat-flow folds and short-bar statistics

**Exploratory packet. Scientific effect: NONE. No status change. No register change.**
This is not a theorem-acceptance entry, a confirmation run or a held-out experiment.
It makes no claim of progress toward the Riemann Hypothesis (RH), and no bound on the
de Bruijn–Newman constant Λ beyond known results. None should be read into it.

- Author: Anthropic Claude (delegated AI work for the project owner, Dylan Roy).
- Nonauthor review from a different provider: **PENDING**. Dylan's personal reading: **PENDING**.
- Object: `ZETA-FOLD-EXPLORATION-20261012`, draft v1 (v0 plus the two internal review passes, §10).
- Project sources read at main `8977619e` (the cited main files are unchanged since `03acb7be`)
  and Math- `79cb314d`.

## In brief

An external note forwarded by the project owner on 12 October 2026 proposed three links
between the zeros of the Riemann zeta function and this project's short-bar law. This
packet checks each link, says what is classical, what is wrong, and what is useful to the
project, and keeps the scripts and results so that anyone can re-run them.

**What we found, in plain terms.**

1. **Folds.** When two zeros of the de Bruijn–Newman heat flow H_t merge, the antiderivative
   of H_t gains or loses exactly one persistence bar, and that bar follows the same cubic fold
   law as in this project (bar length ∝ gap³). This is ordinary fold geometry, not something
   special to zeta. We computed two actual collisions to about 40 digits. Both are at
   negative time (t ≈ −1.878 and t ≈ −0.385), and the fold law is accurate only close to the
   collision time. None of this says anything about RH.
2. **Theta functions.** The project's periodized Gaussian field is normalized by a Jacobi theta
   value, and zeta can be built from the same theta function. But zeta appears only after
   integrating over all torus sizes, while
   the project holds the torus size fixed. At the project's size the periodization is
   invisible locally (about 10⁻¹²⁵). This link is a footnote.
3. **Statistics.** This is the useful part, and it is about the project's own question, not
   about RH. Zeta zeros (like eigenvalues of random unitary matrices, CUE) repel each other
   quadratically at short range, whereas the zeros in the project's Gaussian models repel
   linearly. With the same cubic fold, that one change moves the short-bar density from
   ℓ^(−1/3) to roughly ℓ⁰. Our numerics are decisive against −1/3 for zeta. They are
   consistent with 0 once the small finite-cutoff bias that the estimator also shows on
   controls with known answers (0.02 to 0.07) is allowed for. The data were inspected before
   any prediction was frozen, so this is exploration, not confirmation.

**What is worth doing next** (§6): a theorem for fixed-size CUE matrices (T-CUE), and a
frozen, pre-registered comparison on fresh zeta windows. **What this packet does not claim or
pursue** is listed in §7.

| Link | Status | What it gives this project |
|---|---|---|
| Fold ↔ de Bruijn–Newman zero collisions | Exact, but only near the collision time. Generic fold geometry; the persistence wording is standard Cerf birth–death and vineyards. | A clean deterministic fold family whose time-transversality is forced by a PDE (Proposition 1). Nothing about RH. |
| Gaussian lattice ↔ theta ↔ ζ | Exact and classical. ζ appears only after a Mellin transform over the torus side L, a parameter the project holds fixed. | A footnote. At L = 24 the periodization changes local jets by about e^(−288). |
| GUE/CUE pair statistics ↔ contact law | Decisive against −1/3 at the heights tested; consistent with ≈ 0 given the finite-cutoff estimator bias of order 0.02–0.07 seen in the controls. Intervals are sampling-only. GUE behaviour of zeta zeros is itself a conjecture. | A natural, non-Gaussian **altered pair-count tilt**: the same cubic fold with r² repulsion gives density exponent ≈ 0, not −1/3. |

**Files.**

| Path | Contents |
|---|---|
| [`CLAIMS.md`](CLAIMS.md) | Claim-by-claim check of the note (paraphrased), with verdicts, corrected statements and citation provenance. |
| `dbn/` | de Bruijn–Newman flow: three independent H_t implementations (`dbn_common.py`, `hflow.py`, `indep_ht.py`), collision search and fold checks, κ checks. |
| `theta/` | Theta, Mellin, Poisson and Epstein checks, and the project-normalizer checks. |
| `stats/` | Riemann–Siegel zeta zeros, Gaussian and CUE samples, barcodes, exponent estimates and review cross-checks. |
| `results/` | Small committed summaries (JSON, CSV, SVG, `.out` logs). `results/raw/` holds regenerable `.npz` samples and is not committed. |
| [`MANIFEST.json`](MANIFEST.json) | SHA-256 and size of every committed file, and SHA-256 of the raw samples from the exploratory run (written and checked by `make_manifest.py`). |
| `test_exact.py` | Exact rational checks (standard library only). |
| `test_numeric.py` | Numeric smoke checks (skipped unless numpy and mpmath are installed). |

## 1. Classical background

Every item below is classical. Items marked **[num]** were re-checked numerically here with
mpmath 1.4.1 (high precision, not interval-certified). Citation tags: **[snippet]** means the
bibliographic details were checked only against search-engine snippets (arXiv, journal and
encyclopedia pages could not be fetched from this environment: proxy 403 / DNS);
**[memory]** means recalled from memory and unchecked. A full list is at the end of this
section.

### 1.1 ξ, Ξ and the critical line

ξ(s) = ½ s(s−1) π^(−s/2) Γ(s/2) ζ(s) is entire and satisfies ξ(s) = ξ(1−s). Define
Ξ(x) = ξ(½ + ix), which is real and even for real x. RH says every zero of Ξ is real. The
nontrivial zeros of ζ lie in 0 < Re s < 1; this follows from ζ(1+it) ≠ 0 (Hadamard;
de la Vallée Poussin, 1896) together with the functional equation [Riemann 1859; Titchmarsh
ch. II; memory].

### 1.2 The de Bruijn–Newman flow

    Φ(u) = Σ_{n≥1} (2π² n⁴ e^{9u} − 3π n² e^{5u}) exp(−π n² e^{4u}),
    H_t(z) = ∫_0^∞ e^{t u²} Φ(u) cos(z u) du.

These are the conventions of Rodgers–Tao and Polymath15 [snippet].

- **Normalization.** H_0(z) = Ξ(z/2)/8, so the zero ρ = ½ + iγ corresponds to z = 2γ.
  **[num]** Three independent implementations agree (§5.1): relative difference ≤ 7·10⁻⁴¹
  at z ∈ {0, 10, 28.27, 40, 75, 150} (Gauss–Legendre), ≤ 10⁻⁴⁴ for z ≤ 250 (trapezoid rule)
  and about 10⁻⁴⁷ (tanh-sinh). The zeros of H_0 match 2·Im ρ_k for k = 1…41 to 30 digits.
- **Kernel.** Φ is even. It is positive for u ≥ 0 termwise, because each term equals
  πn²e^{5u}(2πn²e^{4u} − 3) e^{−πn²e^{4u}} and 2π > 3; hence it is positive everywhere by
  evenness. With ψ = (Θ−1)/2 and x = e^{4u}, Φ(u) = e^u [2x² ψ''(x) + 3x ψ'(x)], so Φ is a
  derivative of the Jacobi theta function, and its evenness is the theta transformation.
  **[num]** Both checked to 20 digits.
- **Backward heat equation.** ∂_t H = −∂_z² H, by differentiating under the integral.
  **[num]** Finite-difference residual ≤ 4·10⁻²⁷ at three points, against terms of size
  10⁻⁴ to 10⁻¹⁰.
- **Zero dynamics.** For a simple zero, H(t, z_j(t)) = 0 and the heat equation give the exact
  velocity ż_j = −H_t/H_z = H_zz/H_z. Through the Hadamard product this becomes
  dz_j/dt = 2 Σ_{k≠j} 1/(z_j − z_k) [Csordas–Smith–Varga 1994, snippet; used by Rodgers–Tao].
  That sum is not absolutely convergent and must be summed symmetrically, for example over
  positive zeros as ż_j = 2[1/(2z_j) + Σ_{k≠j} 2z_j/(z_j² − z_k²)]. Zeros therefore **repel as
  t increases**. As t decreases, neighbours attract, can merge into a real double zero, and
  then become a complex-conjugate pair.
  **[num]** At t = 0 the exact velocities H_zz/H_z are −0.5796306, −0.7549955 and −0.4654455
  for zeros #1, #2 and #5 (`dbn/zero_velocity.py`), matching finite differences. The
  symmetric sum over 300 zeros plus a tail estimate gives −0.579717, −0.755108 and −0.465554,
  which is truncation-limited at about 1.5·10⁻⁴.
- **The constant Λ.** There is a Λ such that H_t has only real zeros exactly when t ≥ Λ
  [de Bruijn 1950; Newman 1976; snippet]. RH ⇔ Λ ≤ 0. Rodgers–Tao (2020) proved Λ ≥ 0, so
  RH ⇔ Λ = 0 [snippet].
  - Upper bounds: Λ ≤ 0.22 [Polymath15 2019]; Λ ≤ 0.2 [Platt–Trudgian 2021]; [snippet].
  - Lower bounds before Rodgers–Tao came from Lehmer pairs [CSV 1994;
    Csordas–Odlyzko–Smith–Varga 1993; Odlyzko 2000; Saouter–Gourdon–Demichel 2011; snippet;
    the 1988–1992 sequence of bounds, memory].
  - For every t > 0, all but finitely many zeros of H_t are real and simple
    [Ki–Kim–Lee 2009; snippet].
  - De Bruijn's strip lemma: if all zeros of H_{t0} satisfy |Im z| ≤ y0, then Λ ≤ t0 + y0²/2.
  - Survey: Newman–Wu 2020 [snippet].
  - The time variable rescales by 4 between the z and Ξ(x) conventions.
  - **Which results are kernel-generic.** De Bruijn's lemma and Newman's monotonicity hold for
    broad kernel classes. Ki–Kim–Lee and Rodgers–Tao use ζ-specific zero information
    (arithmetic input, not theta structure): Rodgers–Tao use pair-correlation estimates and
    zero counting, per their abstract [snippet]; Ki–Kim–Lee use zero-distribution facts
    [memory]. The Rodgers–Tao energy argument is a technique for proving the lower bound; it is
    not a global constraint that rules out configurations for t > 0.

### 1.3 Theta, Mellin and Epstein

Let Θ(u) = Σ_{n∈ℤ} e^{−π u n²}.

- **Theta transformation.** Θ(u) = u^{−1/2} Θ(1/u) is classical (Jacobi; Poisson summation)
  [memory]. Its use to prove ξ(s) = ξ(1−s) is Riemann 1859, the second proof [memory].
  Mumford's *Tata Lectures on Theta I* is an expository source [memory].
- **Mellin formula.** π^{−s/2} Γ(s/2) ζ(s) = ½ ∫_0^∞ (Θ(u) − 1) u^{s/2−1} du for Re s > 1.
  **[num]** Checked at s = 2, 3, 4.5 and 2+3i to 18 digits. As u → 0 the integrand behaves
  like u^{(s−3)/2}, so the integral diverges for Re s ≤ 1; continuation needs Riemann's split
  at u = 1.
- **Higher dimensions.** For d ≥ 2 and Re w > d/2,
  ∫_0^∞ (Θ^d − 1) u^{w−1} du = π^{−w} Γ(w) Z_d(w), where Z_d(w) = Σ'_{n∈ℤ^d} |n|^{−2w} is the
  Epstein zeta function of ℤ^d, with Λ_d(w) = Λ_d(d/2 − w). Two cases:
  Z_2 = 4 ζ(w) L(w, χ_{−4}) and Z_4 = 8 (1 − 4^{1−w}) ζ(w) ζ(w−1).
- **A theta functional equation does not locate zeros.** Z_4 has zeros at ½ + iγ and
  3/2 + iγ, off its symmetry line Re w = 1. Z_4 does have an Euler product: it is an
  imprimitive, shifted product, and it shows that a theta functional equation does not force
  zeros onto the symmetry line for such a product. Separately, Epstein zeta functions of
  positive-definite integral **binary** quadratic forms with class number > 1 have zeros in
  Re s > 1 [Davenport–Heilbronn 1936; snippet]; that result does not apply directly to the
  project's ℤ^d forms. **[num]** The factorizations, the functional equations and the
  off-line zeros of Z_4 at γ_1 were checked.

### 1.4 Pair correlation

- **Montgomery's theorem (assuming RH)** gives pair correlation only for test functions whose
  Fourier support lies in (−1, 1) [Montgomery 1973; snippet].
- **The conjecture.** The full GUE law R_2(r) = 1 − (sin πr / πr)² = (π²/3) r² − (2π⁴/45) r⁴ + …
  is Montgomery's pair-correlation conjecture. Restricted-support extensions are
  Rudnick–Sarnak 1996 [snippet].
- **Evidence.** Odlyzko's computations support it numerically [Odlyzko 1987 and later; snippet].
- **Small r.** The r → 0 law is not proved. The GUE nearest-neighbour spacing density has the
  same leading term (π²/3) s²; the Wigner surmise gives 32/π² instead.

**Citation provenance.**

- *Checked against search snippets only:* de Bruijn 1950, Newman 1976, Csordas–Smith–Varga
  1994, Csordas–Odlyzko–Smith–Varga 1993, Odlyzko 1987 and 2000, Saouter–Gourdon–Demichel 2011,
  Ki–Kim–Lee 2009, Rodgers–Tao 2020, Polymath15 2019, Platt–Trudgian 2021 (Λ ≤ 0.2),
  Montgomery 1973, Rudnick–Sarnak 1996, Davenport–Heilbronn 1936, Newman–Wu 2020, and the
  historical table of Λ lower bounds including the −0.385 entry attributed to
  Norfolk–Ruttan–Varga 1992.
- *From memory, unchecked:* Riemann 1859, Jacobi (theta transformation), Titchmarsh, Mumford,
  Edwards 1974 (Riemann–Siegel coefficients, also validated numerically against
  `mpmath.siegelz`), Cerf 1970, Cohen-Steiner–Edelsbrunner–Morozov 2006,
  Hughes–Keating–O'Connell 2000 (and their arithmetic factor a(k)), Gonek 1989, Hejhal 1989,
  Keating–Snaith 2000, Selberg's central limit theorem and Radziwiłł–Soundararajan 2017,
  Bogomolny–Bohigas–Leboeuf–Monastra 2006 (effective N and its constant), Platt–Trudgian's
  verification of RH to height 3·10¹², Csordas–Ruttan–Varga 1991, the 1988–1992 sequence of
  lower bounds on Λ, and Mathlib's proof of ζ's functional equation through Jacobi theta.

A nonauthor reviewer with working web access should confirm each of these.

## 2. What the project states, and what it does not

Locations are in main `8977619e` unless marked Math- (Math- links are pinned to `79cb314d`).

- **Fold law.**
  - The normal form x³/3 − h²x has gap = separation³/6
    ([`experiments/universality/fold_controls.py`](../universality/fold_controls.py) lines 28–31).
  - In the project's convention ℓ = k r³ with k = |F'''|/12 at contact
    ([`fold_structural_universality/PROOF.md`](../universality/fold_structural_universality/PROOF.md)
    lines 191–193 and 244–248).
- **Contact ledger.**
  - The pair intensity is ∝ r: (d−1) + 3 − (d+3) + 2 = 1, and no exponent can be inferred
    while omitting the contact-density or determinant-product orders (`PROOF.md` lines 390–404).
  - The cubic fold alone supplies the gap order, not the radial power (`fold_controls.py`
    lines 3–4). The exact toy controls give density powers −2/3, −1/3 and 0 for radial
    powers 0, 1 and 2; the note's proposed test is literally the case `cubic_2`
    (`results/theta/fold_controls_rerun.out` is the stdout of
    `python3 -B experiments/universality/fold_controls.py` run from the repository root).
- **Theorems.** These are distinct objects, which the note treats as one theorem.
  - The conditional structural theorem: COUNT, CONTACT, SELECT, ENVELOPE and FAR, d ≥ 1,
    with a positive coefficient required (`PROOF.md` lines 56–70).
  - The periodized-Gaussian adapter: fixed d ≥ 2 and L, existential constants, and no joint
    unbounded-d, variable-L, cutoff or model-family uniformity
    ([`periodized_gaussian_h0/PROOF.md`](../universality/periodized_gaussian_h0/PROOF.md)
    lines 54–66 and 80–83).
  - The 1D theorems: Math- `frontiers/d1_third_order_law_20261001` and
    `frontiers/d1_line_20261004`, ν(ℓ) = C₀ ℓ^{−1/3} + C₁ ℓ^{1/4} + ….
- **Geometry versus measure, already in the repositories.** In
  [Math- `reviews/collision_mechanism_20260925/NOTE.md`](https://github.com/d6g8k5htny-coder/Math-/blob/79cb314d5b9f6b8b410bfd0df200414314f6b8c7/reviews/collision_mechanism_20260925/NOTE.md)
  Section D:
  - **Equation (D1)** is a marked collision-to-lifetime transfer theorem. With candidate-pair
    intensity r^α (α is the pair or gap exponent) and ℓ = h(r, z) ≈ κ r^m, the density exponent
    is β − 1 with β = (α+1)/m, under its hypotheses 1–4 (envelope, integrable negative mark
    moment, ratio and derivative limits, uniform comparison). In this packet β always has this
    Math- meaning, and κ in this sense is the fold coefficient, written k in Proposition 1.
  - **Subsection §D1** is a counterexample: cubic splitting alone need not give a density
    coefficient.
  - **Subsection §D2** gives an exact small-mark phase diagram for a mark density ∝ κ^q:
    exponent β − 1 if q + 1 > β, an extra log(1/ℓ) if q + 1 = β, and exponent q if q + 1 < β.
  - **Equation (D2)**, the cumulative statement as written in NOTE.md, was marked
    AMEND_REQUIRED by the PR25 nonauthor review
    ([`reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md`](https://github.com/d6g8k5htny-coder/Math-/blob/79cb314d5b9f6b8b410bfd0df200414314f6b8c7/reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md)
    lines 60 and 184–197). That review accepted equation (D1), the subsection §D1
    counterexample and the subsection §D2 phase diagram. The reviewed replacement for the
    cumulative statement is
    [`CUMULATIVE_TRANSFER_CORRECTION.md`](https://github.com/d6g8k5htny-coder/Math-/blob/79cb314d5b9f6b8b410bfd0df200414314f6b8c7/reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md),
    which restores the ratio hypothesis h/(κ r^m) → 1. This packet cites that correction,
    not the unamended (D2), whenever it needs the cumulative fallback.
  - The amplitude counterexample at `fold_structural_universality/PROOF.md` lines 658–683
    shows that the small-amplitude boundary is not controlled by covariance or positive
    moments.
- **No prior zeta material.** Neither repository has prior material on ζ, Epstein zetas,
  GUE/CUE, de Bruijn–Newman or Lehmer pairs. The one exception is `history/2025/`, which its
  own README calls obsolete and wrong. **Do not cite it.**

## 3. Proposition 1: the fold of a backward-heat solution (elementary)

**Proposition 1.** Assume:

- H(t, z) is real-valued and real-analytic near (t₀, z₀) ∈ ℝ², and solves ∂_t H = −∂_z² H.
- H(t₀, z₀) = ∂_z H(t₀, z₀) = 0.
- A := ½ ∂_z² H(t₀, z₀) ≠ 0.

Set B := ∂_z³ H/6 and C := ∂_z⁴ H/24, both at (t₀, z₀), and τ := t − t₀. Then there are
δ, ε > 0 such that, on W = (z₀ − δ, z₀ + δ):

**(a) Zeros.**
- For 0 < τ < ε, H(t, ·) has exactly two zeros z₋(τ) < z₊(τ) in W, and both are simple.
- For −ε < τ < 0 it has no zero in W. Its two zeros nearest z₀ are the non-real pair
  z₀ + O(|τ|) ± i √(2|τ|) (1 + O(|τ|)).
- At τ = 0, z₀ is the only zero in W, and it is double.

**(b) Gap and midpoint.** For 0 < τ < ε:
- r := z₊ − z₋ satisfies r² = 8τ g(τ), where g is real-analytic, g(0) = 1 and
  g'(0) = κ_r := 4C/A − 2B²/A².
- The midpoint is z₀ + 2(B/A) τ + O(τ²).

**(c) Persistence.** Let F(t, z) = ∫_{z*}^{z} H(t, u) du for any base point z*, restricted to any
closed interval I ⊇ W̄. Then for 0 < τ < ε' (some ε' ≤ ε):
- The two critical points z± of F(t, ·) form a pair in the superlevel H0 barcode of
  F(t, ·)|_I. The one that is a local maximum is born at its value and dies at the value of
  the other.
- The bar length is

      ℓ(τ) = |∫_{z₋}^{z₊} H(t,u) du|
           = (|A|/6) r³ (1 + κ_ℓ τ + O(τ²))
           = (8√2/3) |A| τ^{3/2} (1 + κ_t τ + O(τ²)),

  with κ_ℓ = 6B²/A² − 48C/(5A) and κ_t = κ_ℓ + (3/2) κ_r = 3B²/A² − 18C/(5A).
- For τ < 0, F(t, ·) has no critical point in W.

**Proof.** Replacing H by −H preserves the equation, so assume A > 0.

*(a), via a minimum argument.*
1. Choose δ and ε₀ so that |∂_z² H − 2A| < A on [t₀ − ε₀, t₀ + ε₀] × W̄. Then each H(t, ·) is
   strictly convex on W, so it has at most two zeros there, counted with multiplicity.
2. By the implicit function theorem, ∂_z H(t, ·) has a unique zero c(t) ∈ W for t near t₀,
   with c analytic and c(t₀) = z₀. The minimum of H(t, ·) on W is m(t) = H(t, c(t)).
3. m'(t) = ∂_t H + ∂_z H · c' = ∂_t H(t, c(t)) = −∂_z² H(t, c(t)). Hence m(t₀) = 0 and
   m'(t₀) = −2A < 0.
4. Convexity gives H(t₀, z₀ ± δ) ≥ (A/2) δ² > 0, and this stays positive for t near t₀.
5. So for small ε: m < 0 when τ > 0 (two simple zeros), m > 0 when τ < 0 (no real zero in W),
   and m = 0 when τ = 0 (a double zero).
6. For τ < 0, Hurwitz's theorem applied to the complexification gives exactly two zeros in a
   small disc around z₀. Neither is real, so they form a conjugate pair. Their asymptotics
   come from the scaling below with τ = −σ².

*(b), via scaling.*
1. On solutions, ∂_t^k ∂_z^j H = (−1)^k ∂_z^{j+2k} H. Regrouping the absolutely convergent
   Taylor series at (t₀, z₀) therefore gives H(t₀+τ, z₀+x) = Σ_{n≥2} c_n v_n(x, τ), where
   c_n = ∂_z^n H/n! and v_n(x, τ) = Σ_{k ≤ n/2} n!/(k!(n−2k)!) (−τ)^k x^{n−2k} are the backward
   heat polynomials. For example v₂ = x² − 2τ, v₃ = x³ − 6xτ and v₄ = x⁴ − 12x²τ + 12τ².
2. Since v_n(σy, σ²) = σⁿ v_n(y, 1), the function
   h(y, σ) := σ^{−2} H(t₀ + σ², z₀ + σy) = Σ c_n σ^{n−2} v_n(y, 1) is real-analytic near
   {σ = 0} × [−K, K].
3. It satisfies h(y, 0) = A(y² − 2) and the exact symmetry h(−y, −σ) = h(y, σ).
4. The implicit function theorem at the simple roots ±√2 gives analytic branches y±(σ), and
   z± = z₀ + σ y±(σ). By (a) these are the only zeros in W. The symmetry gives
   y₋(σ) = −y₊(−σ).
5. So r/σ = y₊(σ) + y₊(−σ) is even in σ. Likewise σ(y₊(σ) + y₋(σ))/2 is even in σ. Both are
   therefore analytic in τ = σ².
6. Substituting y± = ±√2 + y₁σ + … into
   h = A(y²−2) + σB(y³−6y) + σ²C(y⁴−12y²+12) + O(σ³) gives y₁ = 2B/A for both branches, and
   the stated κ_r.

*(c), the elder argument.*
1. Here (A > 0) z₋ is a local maximum of F with value M, and z₊ a local minimum with value
   m₀. F increases on [z₀ − δ, z₋], decreases on [z₋, z₊] and increases on [z₊, z₀ + δ].
2. Since ∂_z² H ≥ A on W, ∫_{z₊}^{z₀+δ} H ≥ (A/6)(z₀ + δ − z₊)³ ≥ Aδ³/48 for small τ.
   Meanwhile M − m₀ = −∫_{z₋}^{z₊} H = O(τ^{3/2}). Hence F(z₀ + δ) > M, and symmetrically
   F(z₀ − δ) < m₀.
3. For λ ∈ (m₀, M], the component of {F ≥ λ} through z₋ is an interval inside W containing no
   other critical point.
4. At λ = m₀ it merges with the component through z₊. That component contains z₀ + δ, where
   F > M, so it was born earlier. By the elder rule the bar of z₋ is [m₀, M].

*The value of ℓ.*
1. Write ℓ = σ³ |J(σ)| with J(σ) = ∫_{y₋(σ)}^{y₊(σ)} h(y, σ) dy. The symmetry gives
   J(−σ) = J(σ), so ℓ/τ^{3/2} is analytic in τ.
2. Also J(0) = ∫_{−√2}^{√2} A(y² − 2) dy = −(8√2/3) A.
3. For the τ-coefficient, use the exact identity for consecutive zeros a < c,
   ∫_a^c H = −½ ∫_a^c (u − a)(c − u) ∂_u² H du
   ([`PERSISTENCE_BRIDGE.md`](../../docs/research-translation/20260930/PERSISTENCE_BRIDGE.md)
   lines 94–99).
4. Then ∂_u² H = 2A + 6Bx + C(12x² − 24τ) + O(τ^{3/2}), with x = u − z₀ = m + s and
   m = 2(B/A)τ + O(τ²). The needed moments are ∫(r²/4 − s²) ds = r³/6, ∫(r²/4 − s²) s ds = 0
   and ∫(r²/4 − s²) s² ds = r⁵/120.
5. Together with r² = 8τ(1 + O(τ)) these give κ_ℓ. Then κ_t follows from
   r³ = (8τ)^{3/2} g^{3/2}. ∎

**Checks.**
- `test_exact.py` derives κ_r, κ_ℓ, κ_t and the midpoint drift exactly, as power series in
  σ = √τ with coefficients in ℚ(√2), for rational A, B, C (A > 0 and A < 0, with extra v₅
  and v₆ terms), and checks the three kernel moments and J(0) exactly.
- `dbn/kappa_check.py` integrates a truncated heat-polynomial solution with
  (A, B, C, D) = (1, 0.3, 0.2, −0.15). At τ = 10⁻⁴:

  | Quantity | Measured | Predicted |
  |---|---|---|
  | κ_r | 0.61974 | 0.62 |
  | κ_ℓ | −1.37928 | −1.38 |
  | κ_t | −0.44979 | −0.45 |
  | midpoint / τ | 0.60024 | 2B/A = 0.6 |

  The differences are O(τ).
- Two independent review checks agree: `dbn/kappa_indep_degree6.py` (degree 6, A < 0,
  nonzero v₅ and v₆; agreement to about 10⁻⁵ at τ = 10⁻⁷, and Im z/√(2|τ|) → 1) and
  `dbn/kappa_indep_random.py` (random heat polynomials v₂–v₇ with Richardson extrapolation;
  agreement to 10 digits).

**Remarks.**
1. *Generic geometry.* ℓ = (|F'''|/12) r³ (1 + O(r²)) holds for every nondegenerate fold.
   Here F''' = 2A, so k = |A|/6. Neither ζ nor the heat equation enters this relation.
2. *What the heat equation adds.* First, transversality is automatic: ∂_t H = −2A ≠ 0 at the
   double zero, so the discriminant is crossed transversally and the pair always opens in
   forward time. Second, the opening law r² = 8τ has no free constant.
3. *Consequence for H_t.* Folds of H_t occur only at t₀ ≤ Λ, since for t > Λ all zeros are
   simple [CSV 1994; snippet]. **Under RH, fold events occur only at t ≤ 0** (t = 0 only if Ξ
   has a multiple zero). Heuristically (the Lehmer-pair mechanism), collision times approach
   0 from below as height grows; this is a heuristic, not a theorem of Rodgers–Tao or anyone
   else. So under RH a vineyard of F_t = ∫ H_t over t > 0 has no births or deaths inside a
   window; the only changes are pairing switches and zeros crossing the window ends.
4. *Persistence language is not new.* Describing this as one birth in a vineyard is standard
   Cerf theory [Cerf 1970; Cohen-Steiner–Edelsbrunner–Morozov 2006; memory].

## 4. Elementary restatements in the project's variables (novelty not claimed; no literature search)

These are one-line consequences of classical facts, written in the project's variables. They
were checked numerically. No literature search was done, and no novelty is claimed.

1. **ζ lives along the scale direction.** Let N_F(L, d) = Σ_{j∈ℤ^d} e^{−2π²|j|²/L²} be the
   project's Fourier normalizer, which equals Θ(2π/L²)^d. For Re s > d,

       ∫_0^∞ (N_F(L, d) − 1) L^{−s−1} dL = ½ (2π²)^{−s/2} Γ(s/2) Z_d(s/2).

   For d = 1 this is (2π²)^{−s/2} Γ(s/2) ζ(s) (Re s > 1). This is Riemann's Mellin formula after
   the substitution u = 2π/L², with nothing added. **[num]** Checked for d = 1 at s = 3 and
   2+i, d = 2 at s = 3, and d = 4 at s = 6. ζ appears only after integrating over **all torus
   sizes**, a direction the project disclaims.
2. **The periodization is invisible locally.** K_L(z) = Π_i ϑ₃(πz_i/L, q)/ϑ₃(0, q) with
   q = e^{−2π²/L²}. At L = 24 it differs from the planar squared exponential by
   O(e^{−L²/2} poly(L)), about 10⁻¹²⁵ in the jets at a point; K'', K⁗ and K⁽⁶⁾ at 0 are −1, 3
   and −15 to working precision. The project uses Poisson/Jacobi only to make every Fourier
   weight positive. The modular symmetry u ↦ 1/u plays no role.
3. **The barcode of ∫H depends on how H is normalized, not only on its zeros.** Multiplying H
   by a positive weight w keeps the critical points but changes bar lengths, and can change
   pairings. On Hardy-Z windows **[num]**:
   - with the Ξ weight instead of Hardy normalization, 80/113 pairs ([600, 900]) and 82/106
     ([5000, 5200]) are shared;
   - with Ξ's actual sign, 0 pairs are shared, because superlevel and sublevel roles swap;
   - with a mild weight e^{0.01t}, 112/113 and 104/106 are shared.

   Short bars from close pairs are robust: for C¹ weights and a close pair with midpoint x₀,
   ℓ_w = w(x₀) ℓ (1 + O(r²)) (the O(r) term vanishes by symmetry about the midpoint). So the
   short-bar **exponent** is unchanged, while the coefficient picks up a w^{−β}-weighted
   average, β = (α+1)/m: w^{−1} for GUE-type repulsion (α = 2) and w^{−2/3} for the Gaussian
   case (α = 1).
4. **The lobe area as a weighted integral of ∂_tH.** For consecutive zeros a < c of H_t,
   ∫_a^c H_t = ½ ∫_a^c (x − a)(c − x) ∂_t H_t dx. This is the integration-by-parts identity
   used in the proof of Proposition 1, combined with the heat equation. **[num]** Both sides
   equal −6.31861166707933·10⁻⁴ between 2γ₁ and 2γ₂ (`dbn/lobe_identity.py`,
   `theta/project_theta_checks.py`).

## 5. Numerical results (exploratory)

Everything here is exploratory. The data were inspected before any prediction was frozen,
so nothing below is a confirmation. Nothing is interval-certified. All confidence intervals
are percentile cluster-bootstrap intervals that cover **sampling error only**; they do not
include the systematic finite-cutoff bias discussed in §5.3.

### 5.1 Three implementations of H_t

- `dbn/dbn_common.py`: composite Gauss–Legendre quadrature in mpmath.
- `dbn/hflow.py`: exact fixed-point trapezoid rule, step 1/160, cut-off u ≤ 1.25, 80–110
  digits. It produced the published collision points.
- `dbn/indep_ht.py`: tanh-sinh quadrature on subintervals (written for the math review).

**Agreement at the collisions.**
- The claims run (`dbn/check_collision.py`, 624 Gauss–Legendre nodes, 40 digits) agrees with
  the trapezoid code on the (9/10) collision to about 27 significant digits in t₀ and 29 in
  z₀; that reflects the claims run's precision, not the accuracy of the published points.
- **Independent quadratures agree to about 10⁻⁴³ for both collisions.** An independent
  Gauss–Legendre evaluation (2880 nodes, 90 digits) at the published (x₀, t*) gives Newton
  corrections of at most 3.3·10⁻⁴⁴ in x and 1.5·10⁻⁴⁵ in t, for both (9/10) and (34/35)
  (`results/dbn/indep_collision.out`).
- Recomputing with the trapezoid code at step 1/256, u ≤ 1.5 and 110 digits moves the points
  by Δt* ≤ 1.2·10⁻⁴⁰ and Δx₀ ≤ 2.5·10⁻³⁹ (`results/dbn/refine_check.out`).
- The tanh-sinh implementation, solving from coarse seeds, confirms both collisions to the 30
  printed digits (`results/dbn/indep_ht_c910.out`, `indep_ht_c3435.out`).

### 5.2 Two computed collisions of H_t zeros

The method follows each adjacent pair's critical point c_k(t) and records the sign change of
H_t(c_k(t)). It bisects to about 10⁻⁴⁵, then solves H = H_z = 0 by 2D Newton (`dbn/track.py`,
`dbn/fold_check.py`).

| Pair (zeros at t = 0) | t* | x₀ = z₀ | A = H_zz/2 | B/A | C/A | κ_r | κ_ℓ |
|---|---|---|---|---|---|---|---|
| 9/10 (γ = 48.0052, 49.7738) | −1.877909072972535264767 | 99.328469305575750948 | 2.76275·10⁻¹⁶ | −0.419058 | 0.0629861 | −0.0992746 | 0.448990 |
| 34/35 (γ = 111.0295, 111.8747) | −0.384561116627981414865 | 223.275780185615519914 | −1.76353·10⁻³⁶ | −0.486922 | 0.0700905 | −0.193823 | 0.749687 |

**Why these are the first collisions in their windows.**
- Direct zero counts (`dbn/count_check.py`): window z ≤ 103 has 10 real zeros at t = 0 and at
  t = −1.87, but 8 at −1.89; window z ≤ 270 has 46 at t = 0 and at −0.38, but 44 at −0.39.
- These counts confirm "first collision" because, as t decreases, real zeros can only merge
  and leave the axis (Proposition 1 gives no births in backward time), so the number of real
  zeros in a fixed window can only drop, except through zeros crossing the window edges.
  Zero 46 sits 0.5 from the z = 270 edge but moves inward (269.513 at t = 0, 269.496 at
  t = −0.38), so there is no boundary flux.

**Later events and tracker completeness.**
- Lowest 46 zeros (z ≤ 270, step 0.01): after (34/35), the tracker found, further back in t,
  27/28 at −0.8811, 45/46 at −0.9600, 24/25 at −1.1439, 40/41 at −1.1519, 19/20 at −1.2755 and
  13/14 at −1.2853. It then stopped at t ≈ −1.355 when its own critical-point ordering check
  failed. That is a limitation of the code, not a mathematical event. **The event list is
  complete only down to t ≈ −1.355.**
- Lowest 10 zeros (z ≤ 103, step 0.02): after (9/10), the tracker found 4/5 at −3.9548, 7/8 at
  −4.0338 and 2/3 at −12.4945. Those first three events are reproduced at steps 0.01, 0.02,
  0.025, 0.04 and 0.05. Near t ≈ −6.38 the tracker's bookkeeping becomes step-size
  dependent (ordering failures at steps 0.01 and 0.04; a spurious event with a zero-count
  warning at 0.025 and 0.05; `results/dbn/track_low10_dt_sensitivity.out`). The event list for
  this window is therefore complete only down to about t ≈ −6.38, and the (2/3) event is
  reported without a completeness claim.
- **Every collision we found (lowest 46 zeros) is at t < 0.** Which collision is "first"
  depends on the window.

**Fold-law relative errors.** τ = t − t*. The rows "t = 0" use the actual zeta zeros.

| τ | pair 34/35: r²/(8τ) − 1 | pair 34/35: ℓ/(\|A\|r³/6) − 1 | pair 9/10: r²/(8τ) − 1 | pair 9/10: ℓ/(\|A\|r³/6) − 1 |
|---|---|---|---|---|
| t = 0 (τ = 0.3846 or 1.878) | −7.14% | +33.1% | −16.7% | +128% |
| 0.1 | −1.92% | +7.77% | −0.99% | +4.6% |
| 10⁻³ | −1.94·10⁻⁴ | +7.50·10⁻⁴ | −9.93·10⁻⁵ | +4.49·10⁻⁴ |
| 10⁻⁶ | −1.94·10⁻⁷ | +7.50·10⁻⁷ | — | — |
| 10⁻¹² | −1.94·10⁻¹³ | +7.50·10⁻¹³ | −9.93·10⁻¹⁴ | +4.49·10⁻¹³ |

**How well the theory fits.**
- Over τ ≤ 10⁻⁴, both errors scale as τ^{1.000000}.
- They agree with κτ to a relative 6·10⁻⁶ – 3.5·10⁻⁵, which is the size of the O(τ²) term.
- The midpoint drift 2(B/A)τ agrees to the same accuracy.
- Below t*, Im z / √(2(t* − t)) = 1.00097, 1.0000097 and 1.0000001 at t* − t = 10⁻², 10⁻⁴
  and 10⁻⁶.
- **What this convergence does and does not show.** The computed H̃ is a finite sum
  Σ W_k e^{t u_k²} cos(z u_k), which is itself an exact analytic solution of
  ∂_t H = −∂_z² H. The small-τ convergence therefore holds for any real-analytic
  backward-heat solution, including the quadrature approximant; it checks the κ formulas and
  the code. Agreement with ζ rests on the quadrature accuracy checks (H_0 = Ξ/8 and the
  independent quadratures of §5.1). The ζ-specific content is (t*, x₀, A, B/A, C/A) and the
  t = 0 rows.

**Persistence check** (pair 9/10, τ = +0.01, sampled antiderivative): there is exactly one
finite bar in the window, with length 1.04454·10⁻¹⁸. The exact lobe area is 1.0450·10⁻¹⁸. At
τ = −0.01 there is no finite bar.

**Reading.**
- The fold law describes H_t only near the collision time. At t = 0 the low zeta zeros are far
  from that regime.
- t* = −0.38456 for the (34/35) pair numerically coincides with the reported historical lower
  bound ≈ −0.385 (attributed in search snippets to Norfolk–Ruttan–Varga 1992). The pair used
  there is not confirmed. From memory, the 1991 Csordas–Ruttan–Varga bound −0.0991 is
  stronger than that 1992 entry, so the conventions or pairs in that table need checking.
- These collisions only give bounds Λ ≥ t* that Rodgers–Tao's Λ ≥ 0 supersedes.

### 5.3 Short-bar statistics: Gaussian process, CUE and zeta

**Notation and estimator.**
- α is the exponent of the pair or gap law near 0 (density ∝ r^α), so the small-gap CDF slope
  is α + 1. β = (α+1)/m is used only in the Math- sense of §2 (m = 3 here).
- s is the short-bar CDF slope (N(ℓ) ~ ℓ^s); the density exponent is s − 1.
- Bar lengths are in units of the median bar. s is the Hill (maximum-likelihood power-law)
  estimate on all bars below a cutoff.
- Intervals are 95% cluster-bootstrap percentile intervals, B = 400 (sampling-only). A cluster
  is one Gaussian torus, one CUE matrix, or a block of 2000 consecutive zeta lobes. Analysis
  RNG seed 424242.
- Barcode: superlevel H0 of the antiderivative on the alternating sequence of critical values,
  by the elder rule. The infinite bar and bars born at path endpoints are dropped
  (`stats/analyze.py`). `stats/barcode_bruteforce.py` checks this against a brute-force
  elder-rule computation on 400 random multi-path cases and an analytic test function: 0
  mismatches, maximum difference 4·10⁻¹⁶. Note that `dbn/pers1d.py`, used on the sampled
  windows of §5.2 and §5.4, keeps endpoint-born bars; the two conventions differ by one or two
  bars per window.

**Samples.**

- **Zeta.** Riemann–Siegel main sum plus the corrections C₀, C₁, C₂ (`stats/rs_zeta.py`).
  - The correction functions are Chebyshev fits to mpmath derivatives (fit residual ≤ 3·10⁻¹⁵
    on the fit nodes; the end-to-end check below is the real validation).
  - Maximum |error| against `mpmath.siegelz` at 60 random points: 3.4·10⁻¹⁰ on [10⁴, 10⁵] and
    6.5·10⁻⁹ on [10⁵, 10⁶] (validation RNG seed 20261012); a fresh check at 14 points,
    including the midpoint of the closest pair, gives ≤ 2.6·10⁻⁹ (`results/stats/rs_spot_check.out`).
  - Zeros are bracketed on a grid of 20 points per mean spacing, plus a near-miss search that
    recovered 4 and 52 close pairs hidden inside one grid cell. The GUE expectation for such
    within-cell pairs is 4.4 and 55.1; gaps below 0.05 number 205 against a GUE estimate of
    220.6.
  - Zero counts are 127,927 and 1,609,077. Both equal `mpmath.nzeros(hi) − nzeros(lo)`
    exactly, as do the sub-windows [500000, 500300] (538/538) and [999700, 10⁶] (573/573).
  - Lobes use 16-point Gauss–Legendre quadrature. Relative error against
    `mpmath.quad(siegelz)` is 3·10⁻¹¹ to 5·10⁻⁹ on typical lobes, and 1.6·10⁻⁵ on the
    worst-conditioned lobe (a pair 0.003 apart at t ≈ 663318.508).
  - Scale: t is unfolded by the smooth counting function θ(t)/π + 1, and the integrand is
    Z(t)/√(log(t/2π)). Raw Z gives the same exponents.
- **Gaussian.** The 1D case of the project's covariance, applied to H = F'
  (`stats/gauss_proc.py`).
  - Fourier weights q_n ∝ exp(−ω² n²/2) with ω = 1/M, M = 1000 (L = 2π·1000), so the
    covariance of H is ≈ e^{−x²/2}.
  - 500 samples, seeds 10000–10499, FFT on 2¹⁹ points. F is the exact spectral antiderivative.
    Lobe-area relative error against direct trigonometric sums is 6·10⁻¹¹.
  - Zeros come from cubic Hermite interpolation. `gauss_proc.py` flagged 17 grid cells as
    possible hidden pairs but did not resolve them. **5 hidden pairs resolved after the fact;
    effect on s ≤ 0.003** (`stats/gauss_hidden.py`: a Hermite subgrid search over 113
    candidate cells found 5 real pairs, against 6.0 expected; adding their bars moves s from
    0.6500 to 0.6486 at cut 0.01 and from 0.6567 to 0.6538 at cut 0.003, and the gap slope
    from 1.982 to 1.980). The Gaussian rows below do not include them; the correction pushes
    the cut-0.003 estimate slightly further below 2/3.
- **CUE.** Haar U(N) matrices, with Z_U(θ) = Π_j 2 sin((θ − θ_j)/2) evaluated as a direct
  product (`stats/cue.py`).
  - N = 12: 86,016 matrices. N = 128: 8,064 matrices. Each gives 1,032,192 lobes. Master
    seed 1. Unfolded to mean gap 1. Quadrature with 16 versus 32 nodes differs by ≤ 2·10⁻¹³.
  - **End/cut convention, which differs from the T-CUE target of §6.** Each circle is cut at
    θ₁, the first eigenangle after angle 0, and F = ∫Z_U is treated as a path from θ₁ to
    θ₁ + 2π. F is not periodic, because Z_U's constant Fourier mode is nonzero. Bars born at
    the path endpoints are dropped, so the two endpoint-adjacent lobes can never give short
    bars. Those positions are atypical: the wrap-around gap contains angle 0 and is
    length-biased (mean 1.178; P(gap < 0.2) = 0.0012 against 0.0089 for interior gaps), and
    the excluded lobes hold 11.25% of lobes below 0.01 × median at N = 12, not 2/12. Exponents
    are unaffected (lobe Hill 0.977 for all lobes against 0.973 for interior lobes; gap Hill
    3.003 against 3.003), but the coefficient at N = 12 is shifted by roughly 10% (a
    numerics-review estimate; the share figures are in `results/stats/cue_endpoint_check.out`).

**Short-bar CDF slope s.** −1/3 density gives s = 2/3; density ℓ⁰ gives s = 1.

| Sample | Lobes / bars | s (cut 0.1) | s (cut 0.01) | s (cut 0.003) | LS fit on [10⁻³, 10⁻²] |
|---|---|---|---|---|---|
| Gaussian | 998,836 / 498,674 | 0.631 [0.627, 0.636] | 0.650 [0.641, 0.659] | 0.657 [0.643, 0.671] | 0.643 [0.631, 0.657] |
| CUE N=128 | 1,032,192 / 501,103 | 0.791 [0.786, 0.796] | 0.935 [0.920, 0.951] | 0.971 [0.942, 1.001] | 0.949 [0.926, 0.972] |
| CUE N=12 | 1,032,192 / 386,701 | 0.901 [0.893, 0.909] | 0.980 [0.954, 1.007] | 1.019 [0.973, 1.070] | 0.982 [0.948, 1.018] |
| ζ, t ∈ [10⁴, 10⁵] | 127,926 / 63,962 | 1.002 [0.977, 1.029] | 1.049 [0.968, 1.145] | 1.001 [0.873, 1.147] | 1.034 [0.921, 1.173] |
| ζ, t ∈ [10⁵, 10⁶] | 1,609,076 / 804,536 | 0.947 [0.941, 0.953] | 0.980 [0.959, 1.001] | 0.972 [0.938, 1.009] | 0.978 [0.950, 1.007] |

**Density exponent at cut 0.01, next to the controls with known answers.**

| Sample | Density exponent s − 1 (sampling CI) | Expected | Interval contains it? | Estimate − expected |
|---|---|---|---|---|
| Gaussian | −0.350 [−0.359, −0.341] | −1/3 | **no** | −0.017 (−0.018 with the 5 hidden pairs) |
| CUE N=12 | −0.020 [−0.046, +0.007] | 0 | yes | −0.020 |
| CUE N=128 | −0.065 [−0.080, −0.049] | 0 | **no** | −0.065 |
| ζ, t ∈ [10⁵, 10⁶] | −0.020 [−0.041, +0.001] | — | excludes −1/3 by ≈ 0.31; the upper end only just reaches 0 | — |
| ζ, t ∈ [10⁴, 10⁵] | +0.049 [−0.032, +0.145] | — | — | — |

**Reading of these numbers.** The ζ data are **decisive against −1/3** (a difference of about
0.33, far larger than any error source). They are **consistent with ≈ 0** given the cut-0.01
bias of about −0.02 seen in the Gaussian and CUE N=12 controls (and −0.065 in CUE N=128). The
value −0.020 [−0.041, +0.001] is not a measurement of 0: the intervals are sampling-only, and
these data cannot separate ℓ⁰ from a mark-driven ℓ^q with |q| ≲ 0.04. At cut 0.003 the
Gaussian interval contains 2/3, and its local slopes on [10⁻⁴, 10^−3.5] and [10^−3.5, 10⁻³]
are 0.678 and 0.681. Finite-ℓ corrections are expected: the 1D theorem has a C₁ ℓ^{1/4} term,
whose sign and size for this F have not been computed, and the hypotheses of the 1D theorem
were not checked verbatim for an antiderivative with a zero mode.

**Block-size sensitivity of the zeta intervals** (`stats/block_sensitivity.py`,
`results/stats/block_sensitivity.json`). log|ζ| is approximately log-correlated, so
2000-lobe blocks need not be independent at the amplitude level. Recomputing for ζ, t ∈ [10⁵, 10⁶]
with blocks of 500, 2000 and 20000 lobes (3219, 805 and 81 clusters):

| Statistic | Value | 500 | 2000 | 20000 |
|---|---|---|---|---|
| s at cut 0.01 | 0.980 | [0.958, 0.999] | [0.960, 1.001] | [0.958, 1.004] |
| s at cut 0.003 | 0.972 | [0.936, 1.009] | [0.937, 1.008] | [0.940, 1.006] |
| flat-density coefficient proxy F(0.01)/0.01 | 1.038 | [1.016, 1.057] | [1.018, 1.062] | [1.013, 1.065] |
| fraction of bars < 0.01 × median in the lowest local-amplitude decile | 0.038 | [0.034, 0.042] | [0.034, 0.042] | [0.033, 0.043] |

The exponent intervals barely move; the amplitude-sensitive intervals widen by up to about a
quarter at 20000-lobe blocks, and those are the ones to quote for coefficients. (F is the
fraction of bars below x × median, so F(x)/x is the coefficient the CDF would have if the
density were exactly flat. For ζ, t ∈ [10⁴, 10⁵] it is 0.844 [0.786, 0.911] with 2000-lobe
blocks; the two windows differ, which is consistent with, but not evidence for, a slowly
drifting coefficient.)

**Small-gap CDF slope (= α + 1, cut r < 0.2).**

| Sample | Slope | 95% CI |
|---|---|---|
| Gaussian | 1.982 | [1.965, 2.001] |
| CUE N=128 | 2.993 | [2.930, 3.053] |
| CUE N=12 | 3.003 | [2.940, 3.067] |
| ζ [10⁴, 10⁵] | 3.108 | [2.914, 3.350] |
| ζ [10⁵, 10⁶] | 2.930 | [2.882, 2.983] |

Two-term GUE expectations at cuts 0.4, 0.3, 0.2 and 0.15 are 2.84, 2.91, 2.96 and 2.98.

**Zeta agrees with CUE in the leading exponent only, not in gap shape.** At moderate r the zeta
gap shape is significantly off GUE and off CUE N=12. Gap Hill slope at cut 0.4: ζ [10⁴, 10⁵]
3.05 [2.98, 3.13]; ζ [10⁵, 10⁶] 2.896 [2.876, 2.916]; GUE two-term 2.836; CUE N=12 2.859
[2.837, 2.880]. An independent CUE simulation at cut 0.4 (`stats/cue_gap_shape.py`) gives
N=2: 2.924, N=3: 2.872, N=4: 2.864, N=6: 2.851, N=12: 2.855, so ζ [10⁵, 10⁶] lies between N=2
and N=3. This matches the finite-height effective dimension for spacing statistics of
Bogomolny–Bohigas–Leboeuf–Monastra 2006, N_eff = log(T/2π)/√(12Λ₀) ≈ 2.2–2.8 here, where Λ₀ ≈ 1.57
is an arithmetic constant unrelated to the de Bruijn–Newman Λ [formula and constant from
memory, pending check]. Keating–Snaith's N ≈ log(T/2π) is the matching for value
distribution and moments [memory]. The gap part and the amplitude part of a short-bar
coefficient therefore match CUE at very different N, so any coefficient-level comparison
needs a matching rule declared in advance. Exponents are unaffected.

**Fold check.** For gaps r < 0.35, the slope of log(lobe area) on log(gap) is 3.000
(Gaussian), 3.032 (CUE 128), 3.025 (CUE 12), 3.010 (ζ [10⁴, 10⁵]) and 3.009 (ζ [10⁵, 10⁶]).

**What makes zeta's short bars short.** Take the 8,352 ζ [10⁵, 10⁶] bars below 0.01 × median:
- 100% are an adjacent maximum–minimum pair, and 100% have a gap in the lowest gap decile;
- the mean log₁₀ shortfall is −2.70: gap −0.87, peak/local amplitude −1.90, local amplitude
  +0.05, shape +0.02; Spearman correlation with gap is 0.82, with local amplitude −0.02;
- **short bars lie in the lowest local-amplitude decile at 3.8%, the same as all close gaps
  (4.0% for gap < 0.1 and for gap < 0.2), so bar shortness adds no amplitude selection.**
  The local amplitude is the median peak of the 12 nearest lobes, and level rigidity makes a
  close pair's neighbours larger; with a far-neighbour proxy (skipping the nearest 6 lobes on
  each side) close gaps sit in the lowest decile at 8.0–8.1%. So roughly half of the deficit
  below 10% is the definition and half is a real correlation
  (`results/stats/localamp_baseline.out`).
- CUE N=128 is the contrast: its short bars sit in the lowest local-amplitude decile at 33%,
  against 9.5% for its close gaps, which supports a mark (small-amplitude) effect there.

**Matched lower tail** (`stats/matched_tail.py`): s(lobe area) − s(gap³) at the same tail
fraction q.
- ζ [10⁵, 10⁶]: q = 0.01, 0.0004 [−0.0088, 0.0097]; q = 0.003, −0.014 [−0.031, +0.0005]
  (borderline).
- CUE N=12: q = 0.01, −0.019 [−0.032, −0.005].
- CUE N=128: q = 0.01, −0.034 [−0.052, −0.016]; q = 0.003, −0.024 [−0.058, +0.006].
- ζ [10⁴, 10⁵]: q = 0.003, +0.058 [0.007, 0.119] on 384 lobes (one of many comparisons).

**No mark effect larger than about 0.02 in s is detected for zeta at q = 0.01. Small effects
(about 0.02) appear in CUE N=12, and larger ones in CUE N=128, consistent with the subsection
§D2 mark mechanism at finite ℓ. The admission condition itself is untested.** This test
detects mark effects only of size |Δs| ≳ 0.02 at q ≥ 0.001. It cannot verify the mark
condition and cannot exclude ℓ^q with small negative q. The slower convergence of CUE N=128
is consistent with small-amplitude marks; that reading was made after seeing the data.

**Reading.** The same cubic fold, with quadratic rather than linear pair repulsion, moves the
density exponent from −1/3 to ≈ 0. This is **consistent with the equation (D1) prediction for
α = 2, m = 3 (β = 1): heuristic for ζ, theorem target for CUE (T-CUE, §6).** (D1) is a
theorem about intensity measures of random models under its hypotheses 1–4; for the
deterministic ζ none of these objects is defined or verified. The analogue of the mark
condition here concerns how small |Z''| can be at close pairs in the chosen normalization;
it is not proved for ζ. Fits over moderate lengths give misleading intermediate exponents:
for example CUE N=128 below 0.1 × median gives density ≈ −0.21. The heights 10⁴–10⁶ are modest.

### 5.4 Small Hardy-Z windows

| Window | Zeros found / exact (`mpmath.nzeros`) | Riemann–von Mangoldt main term | Finite bars | Single-lobe | Slope log(area) on log(gap) | Residual sd (total) |
|---|---|---|---|---|---|---|
| [600, 900], step 0.02 | 228 / 228 | 228.0 | 113 | 80.5% | 2.995 | 0.257 (1.264) |
| [5000, 5200], step 0.015 | 214 / 214 | 213.2 | 106 | 82.1% | 3.054 | 0.351 (1.394) |

The main term is printed for orientation only; the exact counts are the completeness check.
The ten shortest bars in [600, 900] all have ℓ/gap³ between 0.50 and 1.01. These barcodes use
`dbn/pers1d.py`, which keeps endpoint-born bars (§5.3). No exponent is estimated from windows
of about 110 bars. On normalization dependence, see §4 item 3.

## 6. Research targets worth exploring

Ranked by value to the project's own goals (separating geometry from measure, and admission),
not by any relevance to RH. These are labelled targets, not results.

### 6.1 Theorem target T-CUE: the short-bar law for fixed-N CUE (first)

- **Question.** Let U be Haar on U(N), N ≥ 2, Z_U(θ) = Π_j 2 sin((θ − θ_j)/2), and
  F_U = ∫ Z_U on a fixed deterministic arc, with explicit end handling (bars born at the arc
  ends dropped), or on the full circle. Does the expected short-bar density satisfy
  ν_N(ℓ) → c_N ∈ (0, ∞) as ℓ ↓ 0, with

      c_N = (1/3) ∫ b₀ κ^{−1} dλ

  in the notation of equation (D1) with α = 2, m = 3, β = 1? What is c_N explicitly (N = 2 in
  closed form), and what is the next-order term?
- **Why it matters.** It would be the first natural non-Gaussian law in which the project's
  transfer theorem is discharged with α ≠ 1. The fold is the same (m = 3); the contact law is
  different (r² from the Vandermonde |Δ|²); the exponent becomes 0. This is exactly the
  altered pair-count tilt boundary prediction in
  [`experiments/universality/PROTOCOL.md`](../universality/PROTOCOL.md) lines 53–64. It would
  show that CONTACT (pair intensity ∝ r) is a necessary admission input, not a consequence of
  the fold, beyond the toy control `cubic_2` and the product-model phase diagram of subsection
  §D2.
- **Feasibility.** Moderate and finite-dimensional. The Weyl density and the Palm conditioning
  at a coincidence are explicit. Heuristically, a third eigenvalue near the pair has density
  ∝ |θ₃ − θ₀|⁴ and κ ∝ |θ₃ − θ₀|, so P(κ < ε) ≍ ε⁵ (q = 4 in subsection §D2) and E[κ^{−1}] < ∞.
  Elder selection near contact works as in Proposition 1(c). The main work is **all four (D1)
  hypotheses 1–4**, including domination (1–2); hypothesis 4 (fixed-constant comparison) is
  the one most likely to fail near triple coincidences. Do N = 2 exactly first.
- **Fallback.** If uniformity fails, the cumulative theorem of
  `CUMULATIVE_TRANSFER_CORRECTION.md` still applies under its hypotheses: κ > 0; an envelope
  0 ≤ B ≤ G with B → B₀ and ∫ G κ^{−β} dλ < ∞; h continuous and strictly increasing with
  h(0+) = 0 and the retained ratio limit h/(κ r^m) → 1; and a lower comparison h ≥ c₀ κ r^m.
  It gives N(ℓ) ~ ℓ^β/(α+1) · ∫ B₀ κ^{−β} dλ and cannot be differentiated without further
  control.
- **What counts as a result.** A Math- frontiers proof candidate with explicit positive c_N, a
  stdlib checker (exact N = 2), and a coefficient-level comparison with CUE data. **Before any
  coefficient comparison**, recompute the CUE bars under exactly the T-CUE convention, or
  derive c_N for the cut-at-θ₁ convention used in `stats/cue.py` (§5.3: about a 10% shift at
  N = 12).
- **Risk.** Uniformity may fail near triple coincidences. The N → ∞ (sine-kernel) limit needs
  amplitude normalization (Keating–Snaith log-normal spread of order ½ log N; memory). It
  says nothing about ζ without further conjectures.
- **Numerical grounding** (cut 0.01; cut 0.003 in parentheses): CUE N=12 s = 0.980
  [0.954, 1.007] (1.019 [0.973, 1.070]); CUE N=128 0.935 [0.920, 0.951] (0.971 [0.942, 1.001]).
  Gap CDF slopes 2.99–3.00; fold slope 3.03.

### 6.2 A frozen zeta short-bar comparison (second)

**Conjecture Z** (exploratory; conditional on GUE behaviour of small zero gaps and on a
small-mark condition for the amplitude-normalized contact coefficient; stated so that it is
well posed).
- *Object.* For height T and window length H = H(T) (with H log T → ∞ and H/T → 0), let F be
  the antiderivative of Z(t)/√(log(t/2π)) in the unfolded variable θ(t)/π + 1 on [T, T+H].
  Define **amplitude-normalized bars** ℓ̃ = ℓ/σ_loc, where σ_loc is the local amplitude
  (median lobe peak over the 12 nearest lobes), rescaled by the window's median. Let μ_{T,H}
  be the **empirical bar measure normalized per zero**:
  μ_{T,H}(E) = #{bars of F on [T, T+H] with ℓ̃ ∈ E} / #{zeros in [T, T+H]}.
- *Order of limits.* On any fixed window μ_{T,H}((0, ℓ]) = 0 below the shortest bar, so
  ℓ → 0 alone is vacuous. The statement takes **T → ∞ first**, then ℓ in a range
  [ℓ_min(T), ℓ₀] with ℓ_min(T) → 0.
- *Exponent form (the main statement).* log μ_{T,H}((0, ℓ]) / log ℓ → 1 in that order of
  limits, i.e. short-bar density exponent 0, against C₀ ℓ^{−1/3} for smooth stationary Gaussian
  processes (Math- d1 theorems).
- *Coefficient form (only with a caveat).* If μ_{T,H}((0, ℓ]) ~ c(T) ℓ, the comparison of c(T)
  with a CUE reference must expect an arithmetic correction, as in the analogous moment
  conjectures (the Hughes–Keating–O'Connell factor a(k); memory), and must fix in advance
  which N defines the CUE reference: gap statistics match N_eff ≈ 2–3 at these heights,
  moments match N ≈ log(T/2π) (§5.3). Without that, a "fail" on unnormalized bars would be
  uninterpretable. The unnormalized coefficient need not converge: local amplitudes have
  log-normal spread of variance ~ ½ log log T (Selberg; memory), and by analogy with negative
  discrete moments of ζ'(ρ) [Gonek 1989; Hejhal 1989; Hughes–Keating–O'Connell 2000; memory],
  where the first pole at k = −3/2 comes from the same r² repulsion, c(T) may drift slowly.

**Protocol.** Follow `PROTOCOL.md` lines 53–64: before generating new data, freeze the windows
(for example t ∈ [10⁶, 10⁷] with the existing Riemann–Siegel pipeline, about 1.6·10⁷ zeros, or
published zero tables at large height), the cutoffs, the estimator, the block size (20000
lobes or more for coefficient intervals), the control-calibrated bias allowance at each cutoff,
the CUE reference and its matching rule (separate N for gaps and amplitudes, or a
zeta-calibrated reference), and the decision rule. The current windows were inspected, so
they stay exploratory. **What counts as a result:** a frozen prediction file plus a run giving
pass/fail/indeterminate on "exponent 0" at the stated heights. It is not a theorem and not RH
evidence.

### 6.3 Lower priority

- **Proposition 1 and the vineyard birth law** (§3): essentially done; needs only a nonauthor
  read. Novelty is essentially nil, and it must not be presented as new or as related to RH.
  Follow-up question: in the project's own scale or heat-flow families, does a PDE likewise
  force folds to be transversal?
- **Sensitivity of the barcode to normalization and marks** (§4 item 3): a short deterministic lemma
  (ℓ_w = w(x₀)ℓ(1 + O(r²)), coefficient w^{−β}-weighted) and the existing table. It is the
  deterministic counterpart of the amplitude counterexample and of the subsection §D2
  small-mark boundary, and informs how `PROTOCOL.md` observables are normalized. Low value if
  it stays a remark.
- **Collision–bar correspondence at t = 0** (optional, lowest): check ℓ(0) ≈ (8√2/3)|A||t_c|^{3/2}
  against the backward collision time t_c for close pairs at moderate heights, using
  Polymath15's effective approximation of H_t. Illustration only; known in substance, and
  accurate only for very close pairs (the low collisions are off by 33–128% at t = 0).

## 7. What this packet does not claim or pursue

- **Any RH program.** The note's proposed route ends at showing that the zeta heat flow has no
  positive-time transition. That statement is equivalent to Λ ≤ 0, which is RH; it is not
  merely tied to RH. No packet content is progress toward RH, support for RH, or a bound on Λ
  beyond known results.
- **That −1/3 explains or relates to the critical line.** It does not, and the data contradict
  it for ζ: the zeta short-bar density exponent is about 0, not −1/3.
- **New lower bounds on Λ from collisions.** Every collision we found (lowest 46 zeros) is at
  t < 0; for example t* ≈ −0.3846 numerically coincides with the reported −0.385 bound (pair
  used not confirmed). Rodgers–Tao's Λ ≥ 0 supersedes all such bounds.
- **Numerical searches for positive-time folds.** Such a search is an RH counterexample search.
  RH has already been verified numerically to large height (3·10¹², Platt–Trudgian; memory).
- **The theta/modular route.** At L = 24 the periodization is invisible (about 10⁻¹²⁵ in the
  jets); ζ needs a Mellin transform over L, which the project disclaims; Epstein zetas violate
  the RH analogue, so theta structure plus a functional equation selects nothing; and a formal
  theta bridge in Lean would add nothing (from memory, Mathlib already proves ζ's functional
  equation via Jacobi theta and Mellin, and the Math- Side24 Lean scratch already uses
  Mathlib's Poisson/theta lemmas).
- **Hunting for new global theta constraints on H_t.** The constraints the note asks for exist
  already (de Bruijn's strip lemma, Newman, Ki–Kim–Lee, Rodgers–Tao; §1.2), and the question is
  outside the project's competence and mission.
- **Applying the project's Gaussian theorems to ζ or H_t.** They are random-field results with
  admission interfaces. ζ is deterministic and is not admitted.
- **A covariance-matched Gaussian null model with covariance Ξ(z/2)/Ξ(0).** It exists by
  Bochner's theorem (Φ is even and positive), and the Math- line theorem's (R1)–(R3) probably
  apply, but its zero set has constant density and encodes nothing about ζ's zeros.
- **Using real-axis persistence to detect off-line zeros.** Counting real zeros against N(T)
  from the argument principle (Turing's method) already does this, and better; it is how RH
  is verified numerically.
- **Fitting exponents at moderate lengths or on small windows.** Fits over 0.1–1 × median give
  misleading intermediate values (−0.21 for CUE N=128), and windows of about 110 bars cannot
  estimate an exponent.
- **Calling the persistence phrasing of collisions new.** It is Cerf theory and vineyards; the
  local fold algebra is not new either, and §4 claims no novelty.
- **Treating the existing zeta/CUE/Gaussian runs as confirmations.** The data were inspected
  first. Any confirmation needs a frozen protocol on fresh data (§6.2).
- **Citing or reviving `history/2025/`.** That retired chapter contains a tautological
  critical-line result and describes itself as obsolete and wrong.
- **Committing the forwarded note verbatim or naming its author.** Its claims are paraphrased
  in `CLAIMS.md`; the note is referred to only as an external note forwarded by the project
  owner on 12 October 2026.
- **RH-adjacent framing on the public site or reader home before nonauthor review.** Packet and
  PR titles use "zeta zeros" and "de Bruijn–Newman flow".
- **The d ≥ 2 Epstein analogues, or a modular-symmetry version of the persistence picture,**
  without a concrete project question.
  Neither has a link to the admission problem.

## 8. Reproduce

Run from this directory. Requirements are pinned in `requirements.txt`: `numpy==2.5.3` (the
same pin as `experiments/periodic_h0`) and `mpmath==1.4.1`. The runs here used Python 3.13.16.
Pillow is optional and only rasterizes PNG previews (`stats/make_plots.py --png`).

**Tests.** The exact tests need only the standard library; the numeric tests skip unless numpy
and mpmath are importable and take a few seconds.

    python3 -B -S -m unittest -v test_exact          # stdlib only
    python3 -B -m unittest -v test_exact test_numeric
    # from the repository root:
    python3 -B -E -S -m unittest discover -s experiments/zeta_fold_exploration -p 'test_exact*.py'

The repository's own `python3 -B -S -m unittest discover -s tests` does not enter
`experiments/`, and the script directories here are not packages, so test discovery does not
pick up the scripts. `python3 -B make_manifest.py --check` verifies every committed file
against `MANIFEST.json`.

**Scripts.** Each script finds its inputs and outputs relative to its own file, so it can be
run from anywhere. Small outputs go to `results/<dbn|theta|stats>/`; large samples go to
`results/raw/` (set `ZETA_FOLD_RAW` to use another directory). Where a script prints to stdout,
the saved `.out` file named in its docstring is that stdout. Approximate runtimes:

- `dbn/check_dbn_basic.py` (~90 s), `dbn/check_collision.py` (~95 s) then
  `dbn/check_collision_persistence.py`.
- `dbn/validate_H0.py`, `dbn/count_check.py 1 270 -0.39` (and the other five count runs),
  `dbn/refine_check.py` (~30 s), `dbn/indep_collision.py` (seconds), `dbn/kappa_check.py`,
  `dbn/kappa_indep_degree6.py`, `dbn/kappa_indep_random.py`, `dbn/zero_velocity.py` (seconds),
  `dbn/smoke_hflow.py`.
- `dbn/track.py 270` (~3 min) and `dbn/track.py 103 -40 0.02 low10` (minutes); then
  `dbn/fold_check.py -0.3845611166 223.2757801856 pair34_35` and
  `dbn/fold_check.py -1.8779090730 99.3284693056 pair9_10` (seconds each), then
  `dbn/fold_analyze.py pair34_35 pair9_10` (also writes `fold_tau_r_ell.csv`).
- `dbn/indep_ht.py 60 c910` (~2 min), `dbn/indep_ht.py 60 c3435` (~7 min),
  `dbn/lobe_identity.py` (~8 min).
- `theta/check_theta.py` and `stats/check_gue_transfer.py` (seconds),
  `theta/project_theta_checks.py` (~7 min).
- `stats/sample_Z.py 600 900 0.02` and `stats/sample_Z.py 5000 5200 0.015` (several minutes
  each), then `stats/check_Z_persistence.py Z_grid_600_900.npz` and the same for the 5000 window.
- `stats/rs_zeta.py validate`, `stats/rs_zeta.py A 1e4 1e5`, `stats/rs_zeta.py B 1e5 1e6`,
  `stats/gauss_proc.py 500 4`, `stats/cue.py 12 86016 4`, `stats/cue.py 128 8064 4`, then
  `stats/analyze.py`, `stats/matched_tail.py`, `stats/make_plots.py` (about 7 minutes in
  total; `analyze.py` stdout duplicates `summary.json` and is not kept). The sample generators
  use a process pool, so set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` to avoid BLAS thread
  oversubscription (without it `cue.py 128` ran more than ten times slower here).
- Review cross-checks on the same samples: `stats/check_gauss.py`, `stats/check_zeta_quad.py`,
  `stats/barcode_bruteforce.py`, `stats/gauss_hidden.py`, `stats/localamp_baseline.py`,
  `stats/cue_gap_shape.py`, `stats/rs_spot_check.py`, `stats/cue_endpoint_check.py`,
  `stats/block_sensitivity.py` (each under a minute).

**Reproduction status in this packet.** After moving the scripts here, the following were
re-run and gave byte-identical outputs (apart from timing fields): `check_collision`,
`check_collision_persistence`, `check_dbn_basic`, `refine_check`, `indep_collision`,
`track.py 270`, `track.py 103 -40 0.02 low10`, `fold_check` for both pairs (CSV and JSON;
the printed Newton residual differs because the seed differs), `fold_analyze` (including
`fold_tau_r_ell.csv`), `validate_H0`, `count_check` (window [1, 103] at t = −1.89),
`check_gue_transfer`, `analyze`, `matched_tail`, `make_plots`, `check_zeta_quad`,
`barcode_bruteforce`, `gauss_hidden`, `localamp_baseline`, `rs_spot_check`. The re-run of
`theta/check_theta.py` and `theta/project_theta_checks.py` reproduced every scripted line
(the earlier logs had one extra ad hoc line and a shell footer, now gone);
`check_Z_persistence` now also prints the exact `nzeros` counts. Raw `.npz` samples (about
200 MB) are not committed; their SHA-256 values from the exploratory run are listed in
[`MANIFEST.json`](MANIFEST.json). Regenerated in a scratch copy of this packet, `gauss.npz`,
`cue12.npz` and `cue128.npz` were byte-identical to the recorded hashes; `zeta_A.npz` differed
only at relative 10⁻¹⁰ (floating-point summation order), below the Riemann–Siegel error, with
identical counts in `zeta_A_info.json`. Regeneration is therefore not guaranteed to be
byte-identical.

## 9. Review status

A nonauthor read from a non-Anthropic provider is requested, in four slices:

- **A.** Proposition 1 and the κ coefficients.
- **B.** `CLAIMS.md` and the citations, ideally by a lane with working arXiv/journal access,
  confirming every item marked [snippet] or [memory].
- **C.** The numerics, re-run independently, with zero counts checked against `nzeros` and a
  different seed or an independent Riemann–Siegel check.
- **D.** A wording audit of the non-claims.

The two review passes already applied (§10) were Anthropic Claude reading Anthropic Claude
work. They earn no organizational-independence credit.

## 10. Review findings applied in this version

Two same-provider review passes (mathematics; numerics) read draft v0. Every finding is
applied here:

| # | Finding (severity) | Where applied |
|---|---|---|
| M1 | "All collisions at t < 0" was stated as fact (high) | "Every collision we found (lowest 46 zeros) is at t < 0": §5.2, §7 |
| M2 | "Numerically decisive" overstated; CIs sampling-only (high) | Bottom-line table, §5.3 control-comparison table and reading |
| M3 | §4 called "new" (medium) | §4 retitled; novelty not claimed |
| M4 | Cumulative (D2) unamended; (D1) vs §D1/§D2 labels (medium) | §2, §6.1 fallback cites `CUMULATIVE_TRANSFER_CORRECTION.md` |
| M5 | Theta transformation attributed to Riemann (medium) | §1.3 |
| M6 | β used in two senses (medium) | α for the gap/pair exponent throughout; β only in the Math- sense (§2, §5.3) |
| M7 | "Under RH every fold is at t < 0, accumulating at 0⁻" (medium) | §3 Remark 3 |
| M8 | Conjecture Z ill-posed (medium) | §6.2 restated |
| M9 | ζ result called "exactly" the transfer theorem (medium) | §5.3 reading |
| M10 | Kernel-generic vs ζ-specific literature (low) | §1.2 |
| M11 | "Reproduced" collisions; −0.385 match (low) | "computed", §5.1–5.2, §7 |
| M12 | Φ positivity wording (low) | §1.2 |
| M13 | Zero ODE summation; exact velocities (low) | §1.2, `dbn/zero_velocity.py` |
| M14 | Davenport–Heilbronn, Z_4, domains (low) | §1.3, §4 item 1 |
| M15 | Hypotheses 1–4 consistently (low) | §2, §6.1 |
| M16 | w^{−1} weight; O(r) (low) | §4 item 3, §6.3 |
| M17 | "time derivative", "later events", "explained by" (low) | §4 item 4, §5.2, §5.3 |
| N1 | "Decisive ≈ 0"; control biases (medium) | §5.3 |
| N2 | 10% amplitude baseline (medium) | §5.3 short-bar composition |
| N3 | Zeta gap shape vs CUE N; matching rule (medium) | §5.3, §6.2 |
| N4 | CUE cut convention vs T-CUE (medium) | §5.3 Samples, §6.1, `stats/cue.py`, `stats/cue_endpoint_check.py` |
| N5 | τ-convergence is generic (low) | §5.2 |
| N6 | Collision accuracy statements (low) | §5.1; `refine_check.out`, `indep_collision.out` committed |
| N7 | Matched-tail limits (low) | §5.3 |
| N8 | 17 unresolved Gaussian cells (low) | Report wording, §5.3; `stats/gauss_hidden.py` |
| N9 | Block bootstrap (low) | Sensitivity run, §5.3 |
| N10 | Main-term counts; endpoint conventions (low) | §5.4, §5.3; `check_Z_persistence.py` prints `nzeros` |
| N11 | Barcode, RS and completeness confirmed (info) | §5.3; brute-force barcode test in `test_numeric.py` |
| N12 | Tracker completeness; monotonicity (info) | §5.2 |
