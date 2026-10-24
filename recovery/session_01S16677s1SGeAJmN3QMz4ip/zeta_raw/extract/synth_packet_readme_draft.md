# Zeta zeros, heat-flow folds and short-bar statistics

**Exploratory packet. Scientific effect: NONE. No status change.** This packet is not
a theorem-acceptance entry, a confirmation run or a held-out experiment. It makes
**no claim of progress toward the Riemann Hypothesis (RH)** and none should be read
into it. Author: Anthropic Claude (Dylan Roy — delegated AI work). Nonauthor review
from a different provider: **PENDING**. Dylan's personal reading: **PENDING**.

Object: `ZETA-FOLD-EXPLORATION-20261012`, draft v0. Project sources read at main
`03acb7be` and Math- `79cb314d`.

## Purpose

In October 2026 the owner forwarded an external note. It proposed three links
between RH and this project:

1. the cubic fold that drives the short-bar law also appears when zeros of the
   de Bruijn–Newman heat flow collide;
2. the periodized Gaussian covariance is theta-function mathematics, the same
   used to build ζ;
3. random-matrix (GUE) statistics of zeta zeros can be compared with the
   project's contact law.

This packet records what in the note is classical, what is correct but
elementary, what is wrong, and what is worth pursuing for the project's own
question. The program states that question as follows
([`docs/PERSISTENCE_UNIVERSALITY_PROGRAM.md`](../../docs/PERSISTENCE_UNIVERSALITY_PROGRAM.md)
lines 3–6 and 76–87): which parts of the short-bar law come from the fold geometry,
and which from the probability law transverse to the discriminant (admission)?

**Bottom line.**

| Link | Status | What it gives this project |
|---|---|---|
| Fold ↔ de Bruijn–Newman zero collisions | Exact, but only near the collision time. It is generic fold geometry. The persistence wording is standard Cerf birth–death and vineyards. | A clean deterministic fold family whose transversality is forced by a PDE (Proposition 1). Nothing about RH. |
| Gaussian lattice ↔ theta ↔ ζ | Exact and classical. ζ appears only after a Mellin transform over the torus side L, a parameter the project holds fixed. | Nothing beyond a footnote. At L = 24 the periodization changes local jets by about e^(−288). |
| GUE/CUE pair statistics ↔ contact law | Numerically decisive at the heights tested. GUE behaviour of zeta zeros is itself a conjecture. | A natural, non-Gaussian **altered pair-count tilt**: the same cubic fold with r² repulsion gives density exponent ≈ 0, not −1/3. This is the useful part. |

## 1. Classical background

Every item below is classical. Items marked **[num]** were re-verified
numerically here with mpmath 1.4.1 (high precision, not interval-certified).
The citation provenance is given at the end of this section.

### 1.1 ξ, Ξ and RH

ξ(s) = ½ s(s−1) π^(−s/2) Γ(s/2) ζ(s) is entire and satisfies ξ(s) = ξ(1−s).
Define Ξ(x) = ξ(½ + ix), which is real and even for real x. RH says every zero of
Ξ is real. The nontrivial zeros of ζ lie in 0 < Re s < 1. This follows from
ζ(1+it) ≠ 0 (Hadamard; de la Vallée Poussin, 1896) together with the functional
equation [Riemann 1859; Titchmarsh ch. II].

### 1.2 The de Bruijn–Newman flow

Let

    Φ(u) = Σ_{n≥1} (2π² n⁴ e^{9u} − 3π n² e^{5u}) exp(−π n² e^{4u}),
    H_t(z) = ∫_0^∞ e^{t u²} Φ(u) cos(z u) du.

These are the conventions of Rodgers–Tao and Polymath15.

- **Normalization.** H_0(z) = Ξ(z/2)/8, so the zero ρ = ½ + iγ corresponds to z = 2γ.
  **[num]** The relative difference is ≤ 7·10⁻⁴¹ at z ∈ {0, 10, 28.27, 40, 75, 150}.
  A second, independent implementation gives ≤ 10⁻⁴⁴ for z ≤ 250. The zeros of H_0
  match 2·Im ρ_k for k = 1…41 to 30 digits.
- **Kernel.** Φ is even and positive (termwise, since 2π > 3). With ψ = (Θ−1)/2 and
  x = e^{4u}, Φ(u) = e^u [2x² ψ''(x) + 3x ψ'(x)], so Φ is a derivative of the
  Jacobi theta function. Its evenness is the theta transformation.
  **[num]** Both checked to 20 digits.
- **Backward heat equation.** ∂_t H = −∂_z² H, by differentiating under the
  integral. **[num]** The residual is ≤ 4·10⁻²⁷ at three points, against terms
  of size 10⁻⁴ to 10⁻¹⁰.
- **Zero dynamics.** For simple zeros, dz_j/dt = 2 Σ_{k≠j} 1/(z_j − z_k)
  [Csordas–Smith–Varga 1994; used by Rodgers–Tao]. Zeros therefore **repel as t
  increases**. As t decreases, neighbours attract, can merge into a real double
  zero, and then become a complex-conjugate pair. **[num]** At t = 0, finite
  differences give dz/dt = −0.579631, −0.754995 and −0.465445 for zeros #1, #2
  and #5. The ODE with 300 zeros plus a tail estimate gives −0.579717, −0.755108
  and −0.465554.
- **The constant Λ.** There is a Λ such that H_t has only real zeros exactly when
  t ≥ Λ [de Bruijn 1950; Newman 1976]. RH ⇔ Λ ≤ 0. Rodgers–Tao (2020) proved
  Λ ≥ 0, so RH ⇔ Λ = 0.
  - Upper bounds: Λ ≤ 0.22 [Polymath15 2019]; Λ ≤ 0.2 [Platt–Trudgian 2021].
  - Lower bounds before Rodgers–Tao came from Lehmer pairs [CSV 1994;
    Csordas–Odlyzko–Smith–Varga 1993; Odlyzko 2000; Saouter–Gourdon–Demichel 2011].
  - For every t > 0, all but finitely many zeros of H_t are real and simple
    [Ki–Kim–Lee 2009].
  - De Bruijn's strip lemma: if all zeros of H_{t0} satisfy |Im z| ≤ y0, then
    Λ ≤ t0 + y0²/2.
  - Survey: Newman–Wu 2020.
  - The time variable rescales by 4 between the z and Ξ(x) conventions.

### 1.3 Theta, Mellin and Epstein

Let Θ(u) = Σ_{n∈ℤ} e^{−π u n²}.

- **Theta transformation.** Θ(u) = u^{−1/2} Θ(1/u). The primary source is
  Riemann 1859 (his second proof of the functional equation). Mumford's
  *Tata Lectures on Theta I* is an expository source.
- **Mellin formula.** π^{−s/2} Γ(s/2) ζ(s) = ½ ∫_0^∞ (Θ(u) − 1) u^{s/2−1} du for
  Re s > 1. **[num]** Checked at s = 2, 3, 4.5 and 2+3i to 18 digits. At u → 0
  the integrand behaves like u^{(s−3)/2}, so the integral diverges for Re s ≤ 1.
- **Higher dimensions.** For d ≥ 2, ∫_0^∞ (Θ^d − 1) u^{w−1} du = π^{−w} Γ(w) Z_d(w),
  where Z_d(w) = Σ'_{n∈ℤ^d} |n|^{−2w} is the Epstein zeta function. Two cases:
  - Z_2 = 4 ζ(w) L(w, χ_{−4});
  - Z_4 = 8 (1 − 4^{1−w}) ζ(w) ζ(w−1).
- **The functional equation does not locate zeros.** Z_4 has zeros at ½ + iγ and
  3/2 + iγ, off its symmetry line Re w = 1. Epstein zetas of forms with class
  number > 1 have zeros in Re s > 1 [Davenport–Heilbronn 1936]. **[num]** The
  factorizations, the functional equations and the off-line zeros of Z_4 at γ_1
  were checked.

### 1.4 Pair correlation

- **Montgomery's theorem (assuming RH)** gives pair correlation only for test
  functions whose Fourier support lies in (−1, 1) [Montgomery 1973].
- **The conjecture.** The full GUE law
  R_2(r) = 1 − (sin πr / πr)² = (π²/3) r² − (2π⁴/45) r⁴ + … is Montgomery's
  pair-correlation conjecture. Restricted-support extensions are
  Rudnick–Sarnak 1996.
- **Evidence.** Odlyzko's computations support it numerically [Odlyzko 1987 and
  later].
- **Small r.** The r → 0 law is not proved. The GUE nearest-neighbour spacing
  density has the same leading term (π²/3) s². (The Wigner surmise gives 32/π²
  instead.)

**Citation provenance.** arXiv, journal and encyclopedia pages could not be fetched
from this environment (proxy 403 / DNS).

- *Checked against search snippets only:* de Bruijn 1950, Newman 1976, CSV 1994,
  COSV 1993, Odlyzko 1987/2000, SGD 2011, Ki–Kim–Lee 2009, Rodgers–Tao 2020,
  Polymath15 2019, Platt–Trudgian 2021, Montgomery 1973, Rudnick–Sarnak 1996,
  Davenport–Heilbronn 1936, Newman–Wu 2020.
- *From memory, unchecked:* Riemann 1859, Titchmarsh, Mumford, Cerf 1970,
  Cohen-Steiner–Edelsbrunner–Morozov 2006, Hughes–Keating–O'Connell 2000,
  Gonek 1989, Hejhal 1989, Keating–Snaith 2000, Selberg's central limit theorem
  and Radziwiłł–Soundararajan 2017, Bogomolny et al. 2006, Edwards 1974, and the
  1988–1992 sequence of lower bounds on Λ.

A nonauthor reviewer with working web access should confirm each of these.

## 2. What the project states, and what it does not

Locations are in main `03acb7be` unless marked Math-.

- **Fold law.**
  - The normal form x³/3 − h²x has gap = separation³/6
    (`experiments/universality/fold_controls.py` lines 28–31).
  - In the project's convention ℓ = k r³ with k = |F'''|/12 at contact
    (`fold_structural_universality/PROOF.md` lines 191–193 and 244–248).
- **Contact ledger.**
  - The pair intensity is ∝ r: (d−1) + 3 − (d+3) + 2 = 1, and "no exponent can be
    inferred while omitting the transformed contact-density order or the
    determinant-product order" (`PROOF.md` lines 396–404).
  - "The cubic fold alone supplies p, not alpha" (`fold_controls.py` lines 3–4).
    The exact toy controls give density powers −2/3, −1/3 and 0 for radial powers
    0, 1 and 2.
- **Theorems.** These are distinct objects. The note merges them under "your
  central theorem".
  - The conditional structural theorem: COUNT, CONTACT, SELECT, ENVELOPE and FAR,
    d ≥ 1, with a positive coefficient required (`PROOF.md` lines 56–70).
  - The periodized-Gaussian adapter: fixed d ≥ 2 and L, existential constants, and
    "No joint unbounded-d, variable-L, cutoff or model-family uniformity is
    asserted" (`periodized_gaussian_h0/PROOF.md` lines 54–66 and 80–83).
  - The 1D theorems: Math- `frontiers/d1_third_order_law_20261001` and
    `frontiers/d1_line_20261004`, ν(ℓ) = C₀ ℓ^{−1/3} + C₁ ℓ^{1/4} + ….
- **Geometry versus measure, already in the repositories.**
  - Math- `reviews/collision_mechanism_20260925/NOTE.md` §D has a marked
    collision-to-lifetime transfer theorem. With pair intensity r^α and ℓ ~ κ r^m,
    the density exponent is β − 1 with β = (α+1)/m.
  - Its §D1 shows that cubic splitting alone need not give a density coefficient.
  - Its §D2 gives an exact small-κ phase diagram for a mark density ∝ κ^q:
    exponent β − 1 if q + 1 > β, an extra log(1/ℓ) if q + 1 = β, and exponent q
    if q + 1 < β.
  - The amplitude counterexample at `PROOF.md` lines 658–683 shows that the
    small-amplitude boundary is "not controlled by covariance or positive moments".
- **No prior zeta material.** Neither repository has prior material on ζ, Epstein
  zetas, GUE/CUE, de Bruijn–Newman or Lehmer pairs. The one exception is
  `history/2025/`, which its own README calls "obsolete and wrong". **Do not cite it.**

## 3. Proposition 1: the fold of a backward-heat solution (elementary)

**Proposition 1.** Make the following assumptions:

- H(t, z) is real-valued and real-analytic near (t₀, z₀) ∈ ℝ², and solves
  ∂_t H = −∂_z² H.
- H(t₀, z₀) = ∂_z H(t₀, z₀) = 0.
- A := ½ ∂_z² H(t₀, z₀) ≠ 0.

Set B := ∂_z³ H/6 and C := ∂_z⁴ H/24, both at (t₀, z₀), and τ := t − t₀.
Then there are δ, ε > 0 such that, on W = (z₀ − δ, z₀ + δ):

**(a) Zeros.**
- For 0 < τ < ε, H(t, ·) has exactly two zeros z₋(τ) < z₊(τ) in W, and both are simple.
- For −ε < τ < 0 it has no zero in W. Its two zeros nearest z₀ are the non-real
  pair z₀ + O(|τ|) ± i √(2|τ|) (1 + O(|τ|)).
- At τ = 0, z₀ is the only zero in W, and it is double.

**(b) Gap and midpoint.** For 0 < τ < ε:
- r := z₊ − z₋ satisfies r² = 8τ g(τ), where g is real-analytic, g(0) = 1 and
  g'(0) = κ_r := 4C/A − 2B²/A².
- The midpoint is z₀ + 2(B/A) τ + O(τ²).

**(c) Persistence.** Let F(t, z) = ∫_{z*}^{z} H(t, u) du for any base point z*,
restricted to any closed interval I ⊇ W̄. Then for 0 < τ < ε' (some ε' ≤ ε):
- The two critical points z± of F(t, ·) form a pair in the superlevel H0 barcode
  of F(t, ·)|_I. The one that is a local maximum is born at its value and dies at
  the value of the other.
- The bar length is

      ℓ(τ) = |∫_{z₋}^{z₊} H(t,u) du|
           = (|A|/6) r³ (1 + κ_ℓ τ + O(τ²))
           = (8√2/3) |A| τ^{3/2} (1 + κ_t τ + O(τ²)),

  with κ_ℓ = 6B²/A² − 48C/(5A) and κ_t = κ_ℓ + (3/2) κ_r = 3B²/A² − 18C/(5A).
- For τ < 0, F(t, ·) has no critical point in W.

**Proof.** Replacing H by −H preserves the equation, so assume A > 0.

*(a), via a minimum argument.*
1. Choose δ and ε₀ so that |∂_z² H − 2A| < A on [t₀ − ε₀, t₀ + ε₀] × W̄. Then
   each H(t, ·) is strictly convex on W, so it has at most two zeros there,
   counted with multiplicity.
2. By the implicit function theorem, ∂_z H(t, ·) has a unique zero c(t) ∈ W for
   t near t₀, with c analytic and c(t₀) = z₀. The minimum of H(t, ·) on W is
   m(t) = H(t, c(t)).
3. m'(t) = ∂_t H + ∂_z H · c' = ∂_t H(t, c(t)) = −∂_z² H(t, c(t)). Hence m(t₀) = 0
   and m'(t₀) = −2A < 0.
4. Convexity gives H(t₀, z₀ ± δ) ≥ (A/2) δ² > 0, and this stays positive for t
   near t₀.
5. So for small ε: m < 0 when τ > 0 (two simple zeros), m > 0 when τ < 0 (no real
   zero in W), and m = 0 when τ = 0 (a double zero).
6. For τ < 0, Hurwitz's theorem applied to the complexification gives exactly two
   zeros in a small disc around z₀. Neither is real, so they form a conjugate
   pair. Their asymptotics come from the scaling below with τ = −σ².

*(b), via scaling.*
1. On solutions, ∂_t^k ∂_z^j H = (−1)^k ∂_z^{j+2k} H. Regrouping the absolutely
   convergent Taylor series at (t₀, z₀) therefore gives
   H(t₀+τ, z₀+x) = Σ_{n≥2} c_n v_n(x, τ), where c_n = ∂_z^n H/n! and
   v_n(x, τ) = Σ_{k ≤ n/2} n!/(k!(n−2k)!) (−τ)^k x^{n−2k} are the backward heat
   polynomials. For example v₂ = x² − 2τ, v₃ = x³ − 6xτ and
   v₄ = x⁴ − 12x²τ + 12τ².
2. Since v_n(σy, σ²) = σⁿ v_n(y, 1), the function
   h(y, σ) := σ^{−2} H(t₀ + σ², z₀ + σy) = Σ c_n σ^{n−2} v_n(y, 1) is real-analytic
   near {σ = 0} × [−K, K].
3. It satisfies h(y, 0) = A(y² − 2) and the exact symmetry h(−y, −σ) = h(y, σ).
4. The implicit function theorem at the simple roots ±√2 gives analytic branches
   y±(σ), and z± = z₀ + σ y±(σ). By (a) these are the only zeros in W. The
   symmetry gives y₋(σ) = −y₊(−σ).
5. So r/σ = y₊(σ) + y₊(−σ) is even in σ. Likewise σ(y₊(σ) + y₋(σ))/2 is even in σ.
   Both are therefore analytic in τ = σ².
6. Substituting y± = ±√2 + y₁σ + … into h = A(y²−2) + σB(y³−6y) + σ²C(y⁴−12y²+12) + O(σ³)
   gives y₁ = 2B/A for both branches, and the stated κ_r.

*(c), the elder argument.*
1. Here (A > 0) z₋ is a local maximum of F with value M, and z₊ a local minimum
   with value m₀. F increases on [z₀ − δ, z₋], decreases on [z₋, z₊] and increases
   on [z₊, z₀ + δ].
2. Since ∂_z² H ≥ A on W, ∫_{z₊}^{z₀+δ} H ≥ (A/6)(z₀ + δ − z₊)³ ≥ Aδ³/48 for
   small τ. Meanwhile M − m₀ = −∫_{z₋}^{z₊} H = O(τ^{3/2}). Hence
   F(z₀ + δ) > M, and symmetrically F(z₀ − δ) < m₀.
3. For λ ∈ (m₀, M], the component of {F ≥ λ} through z₋ is an interval inside W
   containing no other critical point.
4. At λ = m₀ it merges with the component through z₊. That component contains
   z₀ + δ, where F > M, so it was born earlier. By the elder rule the bar of z₋ is
   [m₀, M].

*The value of ℓ.*
1. Write ℓ = σ³ |J(σ)| with J(σ) = ∫_{y₋(σ)}^{y₊(σ)} h(y, σ) dy. The symmetry
   gives J(−σ) = J(σ), so ℓ/τ^{3/2} is analytic in τ.
2. Also J(0) = ∫_{−√2}^{√2} A(y² − 2) dy = −(8√2/3) A.
3. For the τ-coefficient, use the exact identity for consecutive zeros a < c,
   ∫_a^c H = −½ ∫_a^c (u − a)(c − u) ∂_u² H du
   ([`PERSISTENCE_BRIDGE.md`](../../docs/research-translation/20260930/PERSISTENCE_BRIDGE.md)
   lines 94–99).
4. Then ∂_u² H = 2A + 6Bx + C(12x² − 24τ) + O(τ^{3/2}), with x = u − z₀ = m + s and
   m = 2(B/A)τ + O(τ²). The needed moments are ∫(r²/4 − s²) ds = r³/6,
   ∫(r²/4 − s²) s ds = 0 and ∫(r²/4 − s²) s² ds = r⁵/120.
5. Together with r² = 8τ(1 + O(τ)) these give κ_ℓ. Then κ_t follows from
   r³ = (8τ)^{3/2} g^{3/2}. ∎

**Checks.** `dbn/kappa_check.py` (scratch copy: `zeta/synth/kappa_check.py`)
integrates a truncated heat-polynomial solution with (A, B, C, D) = (1, 0.3, 0.2, −0.15). At τ = 10⁻⁴ the four quantities match their
predictions:

| Quantity | Measured | Predicted |
|---|---|---|
| κ_r | 0.61974 | 0.62 |
| κ_ℓ | −1.37928 | −1.38 |
| κ_t | −0.44979 | −0.45 |
| midpoint / τ | 0.60024 | 2B/A = 0.6 |

The differences are O(τ). §5.2 shows the same on actual de Bruijn–Newman
collisions.

**Remarks.**
1. *Generic geometry.* ℓ = (|F'''|/12) r³ (1 + O(r²)) holds for every
   nondegenerate fold. Here F''' = 2A, so k = |A|/6. Neither ζ nor the heat
   equation enters this relation.
2. *What the heat equation adds.* First, transversality is automatic:
   ∂_t H = −2A ≠ 0 at the double zero, so the discriminant is crossed
   transversally and the pair always opens in forward time. Second, the opening
   law r² = 8τ has no free constant.
3. *Consequence for H_t.* Folds of H_t occur only at t₀ ≤ Λ: for t > Λ all zeros
   are simple [CSV 1994]. If RH holds, no fold event occurs for t > 0, so a
   vineyard of F_t = ∫ H_t over t ≥ 0 has no births or deaths inside a window.
   The only changes are pairing switches and zeros crossing the window ends.
4. *Persistence language is not new.* Describing this as one birth in a vineyard
   is standard Cerf theory [Cerf 1970; Cohen-Steiner–Edelsbrunner–Morozov 2006].

## 4. Other elementary observations

These are new but elementary. They were checked numerically and were not searched
for in the literature.

1. **ζ lives along the scale direction.** Let N_F(L, d) = Σ_{j∈ℤ^d} e^{−2π²|j|²/L²}
   be the project's Fourier normalizer, which equals Θ(2π/L²)^d. Then
   ∫_0^∞ (N_F(L, d) − 1) L^{−s−1} dL = ½ (2π²)^{−s/2} Γ(s/2) Z_d(s/2).
   For d = 1 this is (2π²)^{−s/2} Γ(s/2) ζ(s) (for Re s > 1).
   **[num]** Checked for d = 1 at s = 3 and 2+i, d = 2 at s = 3, and d = 4 at s = 6.
   ζ appears only after integrating over **all torus sizes**, a direction the
   project disclaims.
2. **The periodization is invisible locally.** K_L(z) = Π_i ϑ₃(πz_i/L, q)/ϑ₃(0, q)
   with q = e^{−2π²/L²}. At L = 24 it differs from the planar squared exponential
   by O(e^{−L²/2} poly(L)), about 10⁻¹²⁵ in the jets at a point; K'', K⁗ and K⁽⁶⁾
   at 0 are −1, 3 and −15 to working precision. The project uses Poisson/Jacobi
   only to make every Fourier weight positive. The modular symmetry u ↦ 1/u plays
   no role.
3. **The barcode of ∫H depends on how H is normalized, not only on its zeros.**
   Multiplying H by a positive weight w keeps the critical points but changes bar
   lengths, and can change pairings. On Hardy-Z windows **[num]**:
   - with the Ξ weight instead of Hardy normalization, 80/113 pairs ([600, 900])
     and 82/106 ([5000, 5200]) are shared;
   - with Ξ's actual sign, 0 pairs are shared, because superlevel and sublevel
     roles swap;
   - with a mild weight e^{0.01t}, 112/113 and 104/106 are shared.

   Short bars from close pairs are robust: for C¹ weights, ℓ_w = w(x₀) ℓ (1 + O(r)).
   So the short-bar **exponent** is unchanged, but its coefficient picks up a
   w^{−1}-weighted average.
4. **The lobe area is a time derivative.** For consecutive zeros a < c of H_t,
   ∫_a^c H_t = ½ ∫_a^c (x − a)(c − x) ∂_t H_t dx. This is the integration-by-parts
   identity used in the proof of Proposition 1, combined with the heat equation. **[num]** Both sides equal
   −6.31861166707933·10⁻⁴ between 2γ₁ and 2γ₂.

## 5. Numerical results so far

Everything here is exploratory. The data were inspected before any prediction was
frozen, so nothing below is a confirmation. Nothing is interval-certified.

### 5.1 De Bruijn–Newman identities

See §1.2. Two independent implementations of H_t agree:
- Gauss–Legendre quadrature in mpmath (`claims/dbn_common.py`);
- exact fixed-point trapezoid rule with step 1/160 and cut-off u ≤ 1.25,
  80–110 digits (`fold/hflow.py`).

They agree on the (9/10) collision below to about 27 significant digits in t₀ and
29 in z₀. Recomputing with step 1/256, u ≤ 1.5 and 110 digits moves t* by
≤ 10⁻⁴⁰ and x₀ by ≤ 10⁻³⁹.

### 5.2 Two actual collisions of H_t zeros

The method follows each adjacent pair's critical point c_k(t) and records the sign
change of H_t(c_k(t)). It bisects to about 10⁻⁴⁵, then solves H = H_z = 0 by 2D
Newton. Direct zero counts confirm both events:
- window z ≤ 103: 10 real zeros at t = −1.87, but 8 at −1.89;
- window z ≤ 270: 46 at t = −0.38, but 44 at −0.39.

| Pair (zeros at t = 0) | t* | x₀ = z₀ | A = H_zz/2 | B/A | C/A | κ_r | κ_ℓ |
|---|---|---|---|---|---|---|---|
| 9/10 (γ = 48.0052, 49.7738) | −1.877909072972535264767 | 99.328469305575750948 | 2.76275·10⁻¹⁶ | −0.419058 | 0.0629861 | −0.0992746 | 0.448990 |
| 34/35 (γ = 111.0295, 111.8747) | −0.384561116627981414865 | 223.275780185615519914 | −1.76353·10⁻³⁶ | −0.486922 | 0.0700905 | −0.193823 | 0.749687 |

**What the windows show.**
- The (9/10) collision is the first one among the lowest 10 zeros. Later events
  there: 4/5 at −3.9548, 7/8 at −4.0338 and 2/3 at −12.4945.
- The (34/35) collision is the first among the lowest 46 zeros. Later events:
  27/28 at −0.8811, 45/46 at −0.9600, 24/25 at −1.1439, 40/41 at −1.1519,
  19/20 at −1.2755 and 13/14 at −1.2853.
- The tracker stopped at t ≈ −1.355 because of a bookkeeping failure in the code.
  This is not a mathematical event.
- Which collision is "first" depends on the window. If RH holds, close pairs at
  greater height collide ever closer to t = 0⁻ [Rodgers–Tao].

**Fold-law relative errors.** τ = t − t*. The rows "t = 0" use the actual zeta
zeros.

| τ | pair 34/35: r²/(8τ) − 1 | pair 34/35: ℓ/(\|A\|r³/6) − 1 | pair 9/10: r²/(8τ) − 1 | pair 9/10: ℓ/(\|A\|r³/6) − 1 |
|---|---|---|---|---|
| t = 0 (τ = 0.3846 or 1.878) | −7.14% | +33.1% | −16.7% | +128% |
| 0.1 | −1.92% | +7.77% | −0.99% | +4.6% |
| 10⁻³ | −1.94·10⁻⁴ | +7.50·10⁻⁴ | −9.93·10⁻⁵ | +4.49·10⁻⁴ |
| 10⁻⁶ | −1.94·10⁻⁷ | +7.50·10⁻⁷ | — | — |
| 10⁻¹² | −1.94·10⁻¹³ | +7.50·10⁻¹³ | −9.93·10⁻¹⁴ | +4.49·10⁻¹³ |

**How well the theory fits.**
- Over τ ≤ 10⁻⁴, both errors scale as τ^{1.000000}.
- They agree with κτ to a relative 6·10⁻⁶ – 3.5·10⁻⁵, which is the size of the
  O(τ²) term.
- The midpoint drift 2(B/A)τ agrees to the same accuracy.
- Below t*, Im z / √(2(t* − t)) = 1.00097, 1.0000097 and 1.0000001 at
  t* − t = 10⁻², 10⁻⁴ and 10⁻⁶.

**Persistence check** (pair 9/10, τ = +0.01, sampled antiderivative): there is
exactly one finite bar in the window, with length 1.04454·10⁻¹⁸. The exact lobe
area is 1.0450·10⁻¹⁸. At τ = −0.01 there is no finite bar.

**Reading.**
- The fold law describes H_t only near the collision time. At t = 0 the low zeta
  zeros are far from that regime.
- t* = −0.38456 for the (34/35) pair matches the historical lower bound ≈ −0.385
  (attributed in search snippets to Norfolk–Ruttan–Varga). We have not confirmed
  that those authors used this pair.
- These collisions only re-derive bounds Λ ≥ t* that Rodgers–Tao's Λ ≥ 0 supersedes.

### 5.3 Short-bar statistics: Gaussian process, CUE and zeta

**Notation and estimator.**
- s is the short-bar CDF slope (N(ℓ) ~ ℓ^s); the density exponent is s − 1.
- Bar lengths are in units of the median bar.
- s is the Hill (maximum-likelihood power-law) estimate on all bars below a cutoff.
- Intervals are 95% cluster-bootstrap CIs, B = 400. A cluster is one Gaussian
  torus, one CUE matrix, or a block of 2000 zeta lobes. Analysis RNG seed 424242.
- Barcode: superlevel H0 of the antiderivative on the alternating sequence of
  critical values, by the elder rule. The infinite bar and bars born at window
  endpoints are dropped.

**Samples.**

- **Zeta.** Riemann–Siegel main sum plus the corrections C₀, C₁, C₂.
  - The correction functions are Chebyshev fits to mpmath derivatives, with fit
    residual ≤ 3·10⁻¹⁵.
  - Maximum |error| against `mpmath.siegelz` at 60 random points: 3.4·10⁻¹⁰ on
    [10⁴, 10⁵] and 6.5·10⁻⁹ on [10⁵, 10⁶]. Validation RNG seed 20261012.
  - Zeros are bracketed on a grid of 20 points per mean spacing, plus a near-miss
    search. That search recovered 4 and 52 close pairs hidden inside one grid cell.
  - Zero counts are 127,927 and 1,609,077. Both equal
    `mpmath.nzeros(hi) − nzeros(lo)` exactly.
  - Lobes use 16-point Gauss–Legendre quadrature. Relative error against
    `mpmath.quad(siegelz)` is 3·10⁻¹¹ to 5·10⁻⁹ on typical lobes. On the
    worst-conditioned lobe (a pair 0.003 apart in t at t ≈ 663318.508) it is
    1.6·10⁻⁵.
  - Scale: t is unfolded by the smooth counting function θ(t)/π + 1, and the
    integrand is Z(t)/√(log(t/2π)). Raw Z gives the same exponents.
- **Gaussian.** The 1D case of the project's covariance, applied to H = F'.
  - Fourier weights q_n ∝ exp(−ω² n²/2) with ω = 1/M, M = 1000 (L = 2π·1000), so
    the covariance of H is ≈ e^{−x²/2}.
  - 500 samples, seeds 10000–10499, FFT on 2¹⁹ points. F is the exact spectral
    antiderivative.
  - Zeros come from cubic Hermite interpolation; 17 cells were flagged as possible
    hidden pairs. Lobe-area relative error against direct trigonometric sums is
    6·10⁻¹¹.
- **CUE.** Haar U(N) matrices, with Z_U(θ) = Π_j 2 sin((θ − θ_j)/2) evaluated as a
  direct product.
  - N = 12: 86,016 matrices. N = 128: 8,064 matrices. Each gives 1,032,192 lobes.
  - Master seed 1. Unfolded to mean gap 1.
  - Quadrature with 16 versus 32 nodes differs by ≤ 2·10⁻¹³.

**Short-bar CDF slope s.** Prediction: −1/3 density gives s = 2/3, and density ℓ⁰
gives s = 1.

| Sample | Lobes / bars | s (cut 0.1) | s (cut 0.01) | s (cut 0.003) | LS fit on [10⁻³, 10⁻²] |
|---|---|---|---|---|---|
| Gaussian | 998,836 / 498,674 | 0.631 [0.627, 0.636] | 0.650 [0.641, 0.659] | 0.657 [0.643, 0.671] | 0.643 [0.631, 0.657] |
| CUE N=128 | 1,032,192 / 501,103 | 0.791 [0.786, 0.796] | 0.935 [0.920, 0.951] | 0.971 [0.942, 1.001] | 0.949 [0.926, 0.972] |
| CUE N=12 | 1,032,192 / 386,701 | 0.901 [0.893, 0.909] | 0.980 [0.954, 1.007] | 1.019 [0.973, 1.070] | 0.982 [0.948, 1.018] |
| ζ, t ∈ [10⁴, 10⁵] | 127,926 / 63,962 | 1.002 [0.977, 1.029] | 1.049 [0.968, 1.145] | 1.001 [0.873, 1.147] | 1.034 [0.921, 1.173] |
| ζ, t ∈ [10⁵, 10⁶] | 1,609,076 / 804,536 | 0.947 [0.941, 0.953] | 0.980 [0.959, 1.001] | 0.972 [0.938, 1.009] | 0.978 [0.950, 1.007] |

At cut 0.01, the ζ [10⁵, 10⁶] density exponent is −0.020 [−0.041, +0.001]. The
Gaussian exponent is −0.350 [−0.359, −0.341].

The Gaussian interval at cut 0.01 just excludes −1/3. The local slopes on
[10⁻⁴, 10^−3.5] and [10^−3.5, 10⁻³] are 0.678 and 0.681, and the cut-0.003 interval
contains 2/3. Finite-ℓ corrections are expected: the 1D theorem has a C₁ ℓ^{1/4} term. Its sign and size
for this F have not been computed, and the hypotheses of the 1D theorem were not
checked verbatim for an antiderivative with a zero mode.

**Small-gap CDF slope (= β + 1, cut r < 0.2).**

| Sample | Slope | 95% CI |
|---|---|---|
| Gaussian | 1.982 | [1.965, 2.001] |
| CUE N=128 | 2.993 | [2.930, 3.053] |
| CUE N=12 | 3.003 | [2.940, 3.067] |
| ζ [10⁴, 10⁵] | 3.108 | [2.914, 3.350] |
| ζ [10⁵, 10⁶] | 2.930 | [2.882, 2.983] |

Two-term GUE expectations at cuts 0.4, 0.3, 0.2 and 0.15 are 2.84, 2.91, 2.96 and
2.98.

**Fold check.** For gaps r < 0.35, the slope of log(lobe area) on log(gap) is
3.000 (Gaussian), 3.032 (CUE 128), 3.025 (CUE 12), 3.010 (ζ A) and 3.009 (ζ B).

**What makes zeta's short bars short.** Take the 8,352 ζ [10⁵, 10⁶] bars below
0.01 × median:
- 100% are an adjacent maximum–minimum pair, and 100% have a gap in the lowest
  gap decile;
- only 3.8% are in the lowest local-amplitude decile (10% would be chance);
- the mean log₁₀ shortfall is −2.70: gap −0.87, peak/local amplitude −1.90, local
  amplitude +0.05, shape +0.02;
- Spearman correlation with gap is 0.82; with local amplitude, −0.02.

**Matched lower tail (q = 0.01).** For ζ [10⁵, 10⁶], s(lobe area) = 0.9796 and
s(gap³) = 0.9793, a difference of 0.0004 [−0.0088, 0.0097]. For CUE N=128 the
difference is −0.034 [−0.052, −0.016]. There, small-amplitude marks are a real
secondary effect: short bars are 3.3× over-represented in the lowest amplitude
decile. This is the §D2 mark mechanism acting at finite ℓ, and it explains the
slower convergence of CUE N=128.

**Reading.** The same cubic fold, with quadratic rather than linear pair
repulsion, moves the density exponent from −1/3 to ≈ 0. This is exactly the
transfer rule in Math- `collision_mechanism` §D, (D1)/(D2), with α = 2, m = 3,
β = 1. It needs the mark condition E[κ^{−1}] < ∞, which here is a statement about
how small |Z''| can be at close pairs. That condition is not proved for ζ;
numerically it does no harm at these heights. Fits over moderate lengths give
misleading intermediate exponents: for example CUE N=128 below 0.1 × median gives
density ≈ −0.21. The heights 10⁴–10⁶ are modest.

### 5.4 Small Hardy-Z windows

| Window | Zeros (Riemann–von Mangoldt main term) | Finite bars | Single-lobe | Slope log(area) on log(gap) | Residual sd (total) |
|---|---|---|---|---|---|
| [600, 900], step 0.02 | 228 (228.0) | 113 | 80.5% | 2.995 | 0.257 (1.264) |
| [5000, 5200], step 0.015 | 214 | 106 | 82.1% | 3.054 | 0.351 (1.394) |

The ten shortest bars in [600, 900] all have ℓ/gap³ between 0.50 and 1.01. On
normalization dependence, see §4.3.

## 6. Conjecture and a theorem target

These are labelled targets, not results.

**Conjecture Z.** This is exploratory and rests on two unproved inputs: GUE
behaviour of small zero gaps, and E[κ^{−1}] < ∞ for the contact mark in the chosen
normalization. Take the unit-mean-spacing antiderivative of Z(t)/√(log(t/2π)) on
windows at height T. Then over the short-bar range the expected short-bar density
is ≍ c(T) ℓ⁰. Contrast this with C₀ ℓ^{−1/3} for smooth stationary Gaussian
processes (Math- d1 theorems).

The coefficient c(T) need not converge. Local amplitudes have log-normal spread of
variance ~ ½ log log T (Selberg's central limit theorem). By analogy with negative
discrete moments of ζ'(ρ) [Gonek 1989; Hejhal 1989; Hughes–Keating–O'Connell 2000,
from memory], c(T) may drift slowly, perhaps like a power of log T. Amplitude-
normalized bars (lobe area divided by local amplitude) are the better candidate for
a T-stable coefficient.

The known random-matrix fact behind this is that P(|ζ'(ρ)| small) is governed by
the r² repulsion. Hughes–Keating–O'Connell's discrete-moment formula has its first
pole at k = −3/2, from memory. The short-bar law is a different marginal of the
same small-gap behaviour.

**Theorem target T-CUE (fixed N).** Let U be Haar on U(N) with N ≥ 2, let
Z_U(θ) = Π 2 sin((θ − θ_j)/2), and let F_U = ∫ Z_U on a fixed arc, with bars born
at the arc ends excluded. Show that the expected short-bar density satisfies
ν_N(ℓ) → c_N ∈ (0, ∞) as ℓ ↓ 0, with

    c_N = (1/3) ∫ b₀ κ^{−1} dλ

in the notation of Math- `collision_mechanism` §D, (D1), with α = 2, m = 3, β = 1.

- **Mark law.** Heuristically, under the Palm measure at a coincidence θ₁ = θ₂,
  a third eigenvalue near the pair has density ∝ |θ₃ − θ₀|⁴. With κ ∝ |θ₃ − θ₀|
  this gives P(κ < ε) ≍ ε⁵ (q = 4 in §D2), so E[κ^{−1}] < ∞.
- **Work needed.** Discharge §D hypotheses 1–4 (domination and uniform comparison
  when a third eigenvalue approaches), plus the elder argument of Proposition 1(c).
- **Sanity check.** N = 2 can be done in closed form.
- **Harder follow-up.** An N → ∞ sine-kernel version for amplitude-normalized bars.

## 7. Open questions, ranked

1. **T-CUE.** Is c_N explicit and positive for fixed N? Does
   (ν_N(ℓ) − c_N) behave like a power of ℓ? Compare with the existing CUE N=12
   and N=128 data at the level of the **coefficient**, not only the slope.
2. **A frozen zeta comparison.** Predeclare, per `experiments/universality/PROTOCOL.md`
   lines 53–64 ("altered pair-count tilt"): the amplitude-normalized short-bar
   coefficient on fresh windows (for example [10⁶, 10⁷], or published zero tables
   at large height) against CUE with N ≈ log(T/2π). Fix the windows, cut-offs,
   estimator and decision rule before generating data.
3. **Mark sensitivity.** Which barcode features of ∫H are invariant under positive
   smooth reweighting, and how does the short-bar coefficient transform? This is
   the deterministic analogue of the amplitude counterexample.
4. **Collision–bar correspondence at t = 0.** For close pairs, ℓ(0) ≈
   (8√2/3)|A| |t_c|^{3/2}, where t_c < 0 is the backward collision time. This is a
   persistence restatement of the Lehmer-pair mechanism and is known in substance.
   Testing it at large height would need Polymath15's effective approximation of
   H_t.

## 8. Explicit non-claims

- Nothing here proves, supports or approaches RH, or bounds Λ beyond known results.
  The step the note calls "decisive", ruling out a positive-time transition, is
  **equivalent to RH** (it is Λ ≤ 0), not merely "tied to" it.
- The −1/3 exponent does not explain, select or relate to the critical line. Zeta's
  short bars follow ≈ ℓ⁰, not ℓ^{−1/3}.
- The project's theorems do not apply to ζ. They concern random laws on closed
  manifolds, admitted through stated interfaces; ζ is a single deterministic
  function.
- The theta link is classical and passes through a Mellin transform over L. The
  project's fixed-L construction carries no information about ζ's zeros.
- Theta structure plus a functional equation selects no critical line (Epstein
  counterexamples).
- Persistence on the real axis is not needed to detect off-line zeros. Counting
  real zeros against N(T) from the argument principle (Turing's method) already
  does this; it is how RH is verified numerically.
- The local collision picture and its persistence phrasing are not new
  mathematics.
- No exponent is estimated from the small windows (~110 bars). The slopes in §5.3
  are exploratory and not confirmations.
- The 2025 material in `history/2025/` is not prior work of this project.

## 9. Reproduce

The scripts live under `dbn/`, `theta/` and `stats/` (sources: scratch workdirs
`claims/`, `fold/`, `stats/`). Requirements are pinned in `requirements.txt`:
`numpy==2.5.3` (the same pin as `experiments/periodic_h0`) and `mpmath==1.4.1`.
The runs here used Python 3.13.16.

The exact-algebra tests (`test_exact.py`) need only the standard library:

    python3 -B -E -S -m unittest discover -s experiments/zeta_fold_exploration -p 'test_exact*.py'

Numeric checks skip when numpy or mpmath is absent. Approximate runtimes:

- `dbn/check_dbn_basic.py`: about 75 s.
- `dbn/check_collision.py`: about 95 s.
- `dbn/validate_H0.py`, `dbn/track.py`, `dbn/fold_check.py` (pair34_35 and
  pair9_10): minutes.
- `theta/check_theta.py` and `stats/check_gue_transfer.py`: seconds.
- `stats/sample_Z.py 600 900 0.02` and `stats/sample_Z.py 5000 5200 0.015`:
  several minutes each. Then run `stats/check_Z_persistence.py` on each output.
- `stats/rs_zeta.py`, `stats/gauss_proc.py 500 4`, `stats/cue.py 12 86016` and
  `stats/cue.py 128 8064`, then `stats/analyze.py` and `stats/matched_tail.py`:
  about 7 minutes in total.

Raw `.npz` samples (about 190 MB) are not committed. Their SHA-256 values are
listed in `results/MANIFEST.json`, and the scripts regenerate them. `results/`
holds the JSON and CSV summaries quoted above.

## 10. Review

A nonauthor read from a non-Anthropic provider is requested, in four slices:

- **A.** Proposition 1 and its coefficients.
- **B.** The corrections and the citations, especially every item marked "from
  memory".
- **C.** The numerics, rerun independently.
- **D.** A wording audit of the non-claims.

A same-provider read earns no organizational-independence credit.
