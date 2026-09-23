<!-- W2 FREEZE | body-sha256 (over everything below this line): 26e2d1dd3b36f1f7d3cc71f519a156dd5dee241854c2ba9bcd0bce36626ffc9f | W2_sanity.py sha256: ede20064b4f6242346ac02c0a4ac09b6c7b98f073c2f4e16ce6e5fadc9ff4b45 | run1.log sha256: 2570958be883cd893dc790259c4d0e4aea69a10b282b849515c8613de207e693 -->
# W2 — Symbolic WP/Kac–Rice Derivation (SIDE24/q0 lower-rate repair, workstream W2)

**Status:** independent first-principles derivation. No KIMI WP-resolution artifact
(KIMI-DER-025, verify_wp_witness_v1.py, THM-023 drafts) was read or reused; the only
external inputs are (i) the model statement in the work order, (ii) the verbatim
rigidity-zone quotes from C030/C031 reproduced below, and (iii) published measured
constants of C030/C031 used ONLY as a-posteriori cross-validation probes (never as
ingredients of any bound). Companion fail-closed script: `W2_sanity.py` (114 checks,
all PASS; any FAIL invalidates this document).

---

## 0. The rigidity zone, verbatim from the sources

From **C030 CountingLemmas Package.md** (freeze ae20f4e3…f4b), Lemma WP, verbatim:

> "E[N_ws(B₃∖collars) | 9 pins] ≤ **0.21·r³** at r = 0.025 (= 1.28·ℓ), three-zone:
> rigidity zone d ≲ 1.5 contributes ~3e-15 (superexponential kill, below); transition
> (1.5–2.5) 1.9e-6 (ℓ-scaling verified: rung ratio 7.76 ≈ 8); far (2.5–3) via the
> derived unconditional ρ_sad(1.2) = 0.030449 × measured enhancement ≤ 2 × TV."

From **C030**, the discovery paragraph, verbatim:

> "Var(f(y′) | 9 pins) at r = 0.025: **0.018 / 0.55 / 0.976 at d = 1/2/3.**"
> "(i) the band intensities are superexponentially killed (e^{−(b−m)²/2v}) throughout
> d ≲ 1.5 — the assemblies strengthen;"

From **C031_LBRATE_Integration.md** (freeze e165821b…be), §3 table, verbatim:

> "| — window-pass channel | E[N_ws(B₃∖collars) \| 9] ≤ **0.213·r³** (rigidity 3.6e-15
> + transition 1.95e-6 + far 1.37e-6 at r = 0.025; ℓ-ratio verified 7.76/8.19 ≈ 8) |
> derived-structure + measured constants (Lemma WP) | C030 (freeze ae20f4e3…f4b) |
> KR validity; collar exclusion |"

**What d means / zone boundary.** Neither C030 nor C031 ever prints a formal
definition "d := …". The consistent reading across both documents — "Var(f(y′)|9 pins)
… at d = 1/2/3", "Lemma FD: TV ≤ 2.2% at d ≥ 3, ≤ 1e-4 at d = 5" (C031 §3), "Rice …
2.43 on ∂B₅" (C031 §4) — is:

- d(y) = Euclidean distance from the r-clustered pin set {M, S, Y} (equivalently, up
  to O(r) = 0.025, from M or from the cluster centroid — indistinguishable at the
  printed precision);
- B_d = the ball of radius d about the cluster/arch station, so the WP counting region
  is B₃ ∖ collars, decomposed as **rigidity zone {d ≲ 1.5}**, transition {1.5 < d ≤ 2.5},
  far {2.5 < d ≤ 3};
- the rigidity kill mechanism is the Gaussian window factor e^{−(b−m(y))²/2v(y)} with
  m(y) = E[f̃(y)], v(y) = Var(f̃(y)) the 9-pin-conditional mean and variance.

**Evidence for this reading (our own probe, W2_sanity.py Part 4):** recomputing
v(y) = Var(f(y) | 9 pins) from first principles at r = 0.025 gives, along the arch ray
from M (direction (−0.76, 0.24)/0.797), v = **0.0177 / 0.5575 / 0.9776** at d = 1/2/3 —
matching C030's printed 0.018/0.55/0.976 to max relative deviation 1.7% (those are
measured-grade numbers). Along the x-axis the values are 0.0204/0.575/0.980 (dev 14%);
perpendicular to the cluster they are 0.076/0.75/0.99 (no match). So "d" is distance
from the cluster with the printed triple evaluated on (approximately) the arch ray.
**Honest gap:** the exact center and boundary convention of B₃ are not printed in
C030/C031; this is immaterial for W2 because every bound below is **pointwise in y**
and valid on the whole torus; the zone boundary matters only for W3's integration
domain. All integration domains below: Z_r = B₃ ∖ collars with rigidity sub-zone
Z_rig = {y ∈ B₃ ∖ collars : d(y) ≲ 1.5}.

**Budget context.** 0.213·r³ at r = 0.025 is 3.33e-6; the registered zone split is
rigidity 3.6e-15, transition 1.95e-6, far 1.37e-6. Any replacement bound chain must
therefore resolve the pointwise intensity to ≲1e-23-class in the rigidity zone (our
recommended chain achieves 5.0e-23 at the probe point y = (1.0, 0.3), see §7).

---

## 1. Conventions (all choices stated; used throughout)

- **C1 (coordinates).** ℝ² with components (x₁, x₂). Multiindices α = (α₁,α₂),
  |α| = α₁+α₂, ∂^α = ∂₁^{α₁}∂₂^{α₂}.
- **C2 (kernel/covariance derivative rule).** With s = x − y and K(x,y) = K(s),
  ∂_{yᵢ} = −∂_{sᵢ}, hence
  Cov(∂^α f(x), ∂^β f(y)) = ∂_x^α ∂_y^β K(x,y) = **(−1)^{|β|} ∂_s^{α+β} K(s)**.
  Sanity: Cov(f(x), ∂₁₁f(y))|_{x=y} = +K1″(0)·K1(0) = −μ₂ (sign verified, P1).
- **C3 (lattice kernel, exact).** κ_j = jπ/12, w_j = e^{−κ_j²/2}, Z = Σ_j w_j,
  K1(s) = Z⁻¹ Σ_j w_j e^{iκ_j s}, K(s) = K1(s₁)K1(s₂). All spectral sums are
  truncated at |j| ≤ 48 with proved tail bound 3.1e-35 (P0). Lattice moments equal
  the continuum Gaussian moments to <1e-12: μ₂ = 1, μ₄ = 3, μ₆ = 15 (P0).
- **C4 (jet ordering).** At a test point y: J(y) = (F, G₁, G₂, A, B, C) :=
  (f, ∂₁f, ∂₂f, ∂₁₁f, ∂₂₂f, ∂₁₂f)(y). The Hessian matrix is [[A,C],[C,B]], so
  **det H = AB − C²**, tr H = A + B.
- **C5 (typing/orientation).** Saddle ⟺ det H < 0 ⟺ eigenvalues of opposite sign
  ⟺ Morse index 1 (a.s. Morse wherever nondegenerate). The pin-station typing of
  C031 (det M > 0 & tr M < 0 maximum; det S < 0, det Y < 0 saddles) concerns the pin
  values only; the counted event at y is exactly {det H̃(y) < 0}.
- **C6 (Kac–Rice Jacobian/multiplicity).** For an a.s.-C² Gaussian field g on a
  domain D with (∇g(y), vec H^g(y)) nondegenerate for a.e. y,
  E[# {y ∈ D : ∇g = 0, type/event}] = ∫_D p_{∇g(y)}(0) E[|det H^g(y)| 1{event} | ∇g(y)=0] dy.
  The Jacobian of the zero-counting map y ↦ ∇g(y) is |det H^g(y)|; each regular zero
  counted once (no factor 2, no sign, because |det| is used); degenerate zeros
  (det = 0) are a.s. absent (Bulinskaya) under the same nondegeneracy. The
  conditioned field f̃ = f | pins is Gaussian with analytic mean and covariance, so
  C²/analyticity is inherited; the required nondegeneracy det Σ_{G,H}(y) > 0 on Z_r
  is the C030/C031-named "KR validity; collar exclusion" item (at our probe point the
  conditioned 4×4 covariance has min eigenvalue 2.1e-5 > 0).
- **C7 (conditioning order).** First condition on the 9 pins (values v), then on
  ∇f̃(y) = 0. By the block-Schur identity this equals one-shot conditioning on
  (pins, ∇f(y) = 0); verified to 1e-16 (P4). Gaussian conditioning throughout:
  X | Y = y ~ N(μ_X + Σ_{XY}Σ_{YY}^{-1}(y−μ_Y), Σ_{XX} − Σ_{XY}Σ_{YY}^{-1}Σ_{YX}).
- **C8 (window).** Open interval W = (b−ℓ, b), b = 6/5, ℓ = r³/6 exactly; boundary
  is Lebesgue-null and immaterial under Gaussian densities.
- **C9 (pin ordering/values).** Station-major: pins P = (f,∂₁,∂₂)(M),
  (f,∂₁,∂₂)(S), (f,∂₁,∂₂)(Y); values v = (b,0,0, b−ℓ,0,0, μ_t,0,0). C031 writes the
  Y-value as v* = clip(μ_t); at r = 0.025, μ_t = 1.19999869865649 ∈ (b−ℓ, b), so the
  clip is inactive (computed from first principles, §2.4).
- **C10 (TDC transform).** T = (A+B)/2 (half-trace), D = (A−B)/2, C = C. Then
  **det H = T² − D² − C²** (since AB − C² = (T+D)(T−D) − C²). In (T,D,C) coordinates
  det = zᵀ M z with M = diag(1, −1, −1). The transform is orthogonal up to the
  constant Jacobian 1 (map (A,B) ↦ (T,D) has |det| = 1/2 on the (A,B)-plane, a
  constant that cancels in all densities/expectations; we only use it as a linear
  reparametrization of the Gaussian vector).

---

## 2. The exact conditional Gaussian law

### 2.1 Pin covariance and conditional jet law (Theorem W2-1)

Let Σ_PP ∈ ℝ^{9×9}, (Σ_PP)_{ij} = Cov(P_i, P_j) via C2/C3; k_J(y) ∈ ℝ^{6×9},
k_J(y)_{aj} = Cov(∂^{α_a}f(y), P_j); Σ_JJ the unconditional same-point covariance
(computed exactly from C3). **Claim:** the conditioned jet J̃(y) = J(y) | (P = v) is
Gaussian with

  μ_J(y) = k_J(y) Σ_PP^{-1} v,          Σ̃_JJ(y) = Σ_JJ − k_J(y) Σ_PP^{-1} k_J(y)ᵀ.

*Proof.* (J(y), P) is jointly Gaussian (linear functionals of a Gaussian field);
apply C7. Σ_PP is symmetric PD — verified numerically by Cholesky at 50-digit
precision (P4); analytically, distinct derivative functionals of a field with full
spectral support are a.s. linearly independent. ∎

### 2.2 The ∇ = 0 conditioning (Theorem W2-2)

Partition J̃ = (F; G; H), G = (G₁,G₂). Then

  p_{∇f̃(y)}(0) = (2π)^{-1} (det Σ_GG)^{-1/2} exp(−½ μ_Gᵀ Σ_GG^{-1} μ_G),

and (F, H) | (pins, G = 0) ~ N(μ_{FH·G}, Σ_{FH·G}) with

  μ_{FH·G} = μ_{FH} − Σ_{FH,G} Σ_GG^{-1} μ_G,
  Σ_{FH·G} = Σ_{FH,FH} − Σ_{FH,G} Σ_GG^{-1} Σ_{G,FH},

all blocks taken from (μ_J, Σ̃_JJ). *Proof:* C7 applied to the already-conditioned
Gaussian; equivalence with one-shot conditioning on (pins, G=0) is the block-matrix
Schur-complement identity (verified to 1.1e-16 in max-abs, P4). ∎

**Closed form of the estimand.** With W = (b−ℓ, b),

  ρ_WP(y;r) = p_{∇f̃(y)}(0) · 𝔈(y),
  𝔈(y) := E[ |det H| 1{det H<0} 1{F∈W} ],   (F,H) ~ N(μ_{FH·G}, Σ_{FH·G})   (4-dim),

and I_WP(r) = ∫_{Z_r} ρ_WP(y;r) dy. This is exact — no approximation has occurred.

### 2.3 Unconditional reference structure (Theorem W2-3; the "isotropic slice")

Unconditionally (or conditioned on F = u only), at any fixed y:

  F ⟂ G, G ⟂ H, Cov(F, H) = (−μ₂, −μ₂, 0), Var G = μ₂ I₂,
  H | F=u :  A, B iid N(−μ₂u, μ₄−μ₂²),  C ~ N(0, μ₂₂) independent,

with μ₂ = 1, μ₄ = 3, μ₂₂ = 1 (continuum values, lattice-equal to 1e-12; all six
identities verified in P1, e.g. Schur complement of F in H is diag(2,2,1) to 1e-11).
Hence in TDC coordinates: **T ~ N(−u,1), D ~ N(0,1), C ~ N(0,1), independent.**

### 2.4 First-principles cross-validations (probes, not ingredients)

- μ_t := E[f(Y) | 6 pins at M,S, ∇f(Y)=0] computed from §2.1 at r = 0.025:
  (μ_t − b)/ℓ = **−0.4997159**, vs the C031 exact r→0 constant −0.4999290
  (deviation 2.1e-4, consistent with an O(r) approach to the limit). This validates
  the entire pin-covariance + conditional-mean pipeline against an independent exact
  constant.
- Horizon variances: §0 (matches 0.018/0.55/0.976 to 1.7% on the arch ray).
- Var(f(y) | 9 pins, ∇f(y)=0) probe: 0.00042/0.285/0.975 at d=1/2/3 (arch ray) vs
  C030's DF row 0.001/0.278/0.974 (matches at d=2,3; d=1 differs in absolute terms
  at the 6e-4 level — measured-grade discrepancy or d-convention; recorded, not used).
- Two-stage = one-shot conditioning to 1e-16 (algebraic identity, P4).

### 2.5 WARNING (structural fact that shapes everything below)

Under the 9-pin conditioning the isotropic independence of §2.3 is **destroyed**:
at the probe point y = (1.0, 0.3),

  Σ(T,D,C | pins, ∇=0) = [[0.2499, −0.1365, −0.1298],
                          [−0.1365, 0.1704,  0.1129],
                          [−0.1298, 0.1129,  0.1554]],

|Cov TD|+|Cov TC|+|Cov DC| = 0.379 > 0 (P4). Any bound that reuses the
unconditional T ⟂ (D,C) independence or the unit variances under the pins is
**invalid** (registered as I2 in §6). All valid bounds must use the general
quadratic-form machinery of §4.

---

## 3. Closed forms in the isotropic slice (valid unconditionally / given F only)

**Lemma W2-4 (χ²₂ key identity).** If R² ~ χ²₂ (density ½e^{−r/2}, r ≥ 0), then for
a ≥ 0:  P(R² > a) = e^{−a/2}  and  E[(R² − a)₊] = 2e^{−a/2}.
*Proof.* P(R²>a) = ∫_a^∞ ½e^{−r/2}dr = e^{−a/2}. E[(R²−a)₊] = ∫_0^∞ s·½e^{−(s+a)/2}ds
(substituting s = r−a) = e^{−a/2}·E[Exp(1/2)] = 2e^{−a/2}. ∎

**Lemma W2-5.** For T ~ N(m, 1): E[e^{−T²/2}] = 2^{-1/2} e^{−m²/4}.
*Proof.* φ(t−m)e^{−t²/2} = (2π)^{-1/2} exp(−(t−m/2)² − m²/4); integrating over t gives
(1/√2)e^{−m²/4}. ∎ (Verified P2 for m = 0, −1.2, 2.3.)

**Theorem W2-6 (isotropic saddle/max closed forms).** Under the law of §2.3 given
F = u (so T ~ N(−u,1), D,C ~ N(0,1) independent, R² := D²+C² ~ χ²₂ independent of T):

  (i)   E[det H | u] = u² − 1.
  (ii)  P(det H < 0 | u) = 2^{-1/2} e^{−u²/4}.
  (iii) E[|det H| 1{det H<0} | u] = √2 · e^{−u²/4}.
  (iv)  E[|det H| 1{H ≺ 0} | u] = E[ 1{T<0} (T² − 2 + 2e^{−T²/2}) ]  (1D quadrature).

*Proof.* (i) E[T²] − E[R²] = (u²+1) − 2. (ii) P(R² > T²) = E[P(R²>T²|T)] =
E[e^{−T²/2}] by W2-4; apply W2-5 with m = −u. (iii) |det|1{det<0} = (R²−T²)₊;
E[(R²−T²)₊|T] = 2e^{−T²/2} by W2-4; E[2e^{−T²/2}] = 2·2^{-1/2}e^{−u²/4} by W2-5.
(iv) H ≺ 0 ⟺ det>0 ∧ tr<0 ⟺ T < −√(D²+C²)... precisely det>0 ∧ T<0 ⟺ T²>R² ∧ T<0;
(T²−R²)₊ = T²−2+2e^{−T²/2} on {T²<R²... }: E[(T²−R²)1{R²<T²}|T] =
T²(1−e^{−T²/2}) − (2−(T²+2)e^{−T²/2}) = T²−2+2e^{−T²/2}, using
E[R²1{R²<a}] = 2−(a+2)e^{−a/2} (direct integration). ∎

*Numerical verification (P2):* (i)–(iii) by 3e6-sample MC at u = 0, 0.7, 1.2, all
within 4 MC standard errors. Cross-checks against C030/C031 measured constants
(probes only): ρ_sad(1.2) = (2π)⁻¹φ(1.2)·√2 e^{−0.36} = **0.030493** vs printed
0.030449 (rel 1.5e-3); (iv) at u=1.2 gives **1.41363** (quadrature) vs printed
E-term 1.41350 ± 0.0017; ρ_mx(1.2) = (2π)⁻¹φ(1.2)·1.41363 = **0.043689** vs printed
0.043685 (rel 9e-5). Agreement at the measured-constant noise level.

---

## 4. Rigorous UPPER bounds on 𝔈(y) (each step proved)

Setup for §4: (F,H) ~ N(μ, Σ), the 4-dim law of Theorem W2-2; X := |det H| ≥ 0,
A := {det H < 0}, B := {F ∈ W}. All means/covariances below are the conditioned ones;
we write m = μ_H, S = Σ_HH, s_F² = Σ_FF, c = Σ_FH (row), and drop the y-argument.

### 4(a). The corrected global Cauchy–Schwarz (Theorem W2-7)

  𝔈 = E[X 1_A 1_B] ≤ √(E[X² 1_A]) · √(P(B)) ≤ **√(E[X²]) · √(min(P(A), P(B)))**
        = √( E[det²H] · min(P(det H<0), P(F∈W)) ).

*Proof.* First inequality: Cauchy–Schwarz on the product (X1_A)·(1_B). Second:
E[X²1_A] ≤ E[X²] (integrand ≥ 0), and symmetrically
E[X1_A1_B] ≤ √(E[X²1_B])√(P(A)) ≤ √(E[X²])√(P(A}); taking the better of the two
gives the min inside a single square root. ∎

**Why the "dropped inner root" variant is invalid (Proposition W2-8).** The
candidate inequality E[X1_A1_B] ≤ √(E[X²])·P(B) is **false** in general.
*Counterexample 1 (abstract):* X = 1_B, A = Ω, P(B) = 1/4: LHS = 1/4, but
√(E X²)·P(B) = (1/2)(1/4) = 1/8 < 1/4. *Counterexample 2 (in-context):* X = |det H|
with the isotropic H of §2.3, B = {X ≥ q_{0.75}} (P(B) = 1/4), A = Ω: LHS/RHS =
1.69 (measured, P3). The invalid step is treating E[X1_B] ≤ √(EX²)·P(B) — CS only
gives √(EX²)·√(P(B)); the correlation of X with 1_B can saturate √P(B)/P(B) =
1/√P(B) → ∞ as P(B) → 0. This is exactly the defective-inequality class flagged in
the work order. ∎

**The exact second moment (Proposition W2-9).** For H ~ N(m,S) in (A,B,C)
coordinates, E[det²H] = I(A,A,B,B) − 2I(A,B,C,C) + I(C,C,C,C), where for any indices
i,j,k,l, writing z = H:

  I(i,j,k,l) = mᵢmⱼmₖmₗ + (Sᵢⱼmₖmₗ + Sᵢₖmⱼmₗ + Sᵢₗmⱼmₖ + Sⱼₖmᵢmₗ + Sⱼₗmᵢmₖ + Sₖₗmᵢmⱼ)
               + (SᵢⱼSₖₗ + SᵢₖSⱼₗ + SᵢₗSⱼₖ).

*Proof.* det² = A²B² − 2ABC² + C⁴; Isserlis' formula for the fourth moment of a
noncentral Gaussian (differentiate the mgf E[e^{t·z}] = e^{t·m + tᵀSt/2} four times
at t = 0). ∎ Verified against 6×10⁵-sample MC on 6 random (m,S) instances (P3, all
within 4 se). Companions (same proof): E[det H] = m_A m_B + S_AB − m_C² − S_CC, and

  Var(det H) = 2 tr((MS)²) + 4 mᵀ M S M m,   M = QF matrix of det (below),   (★)

the standard Gaussian quadratic-form variance (proof: Wick contraction; verified P3).

Here and below, in whichever coordinate basis, det H = zᵀ M z with symmetric
M = [[0,½,0],[½,0,0],[0,0,−1]] in (A,B,C) coordinates, resp. M = diag(1,−1,−1) in
(T,D,C) coordinates.

### 4(b). Exact slice factorizations (Theorem W2-10)

By disintegration of the joint Gaussian (tower property; both directions):

  **(F-slice, 1D):**  𝔈 = ∫_{b−ℓ}^{b} φ(u; μ_F, s_F²) · G(u) du,
     G(u) := E[|det H| 1{det H<0} | F = u],
     H | F=u ~ N( m_H(u), S_{H·F} ),  m_H(u) = m + (cᵀ/s_F²)(u − μ_F),
     S_{H·F} = S − cᵀc/s_F²   (Schur complement; independent of u).

  **(H-slice, 3D):**  𝔈 = E_H[ |det H| 1{det H<0} · Pw(u(H)) ],
     u(h) := E[F | H=h] = μ_F + c S^{-1}(h − m),
     s_f² := Var(F|H) = s_F² − c S^{-1}cᵀ   (the "residual s_f"),
     Pw(u) = Φ((b−u)/s_f) − Φ((b−ℓ−u)/s_f).

*Proof.* Both are the tower property E[·] = E[E[·|F]] resp. E[E[·|H]] with the
Gaussian conditional laws (C7). Degenerate case s_f = 0: F = u(H) a.s., so
Pw(u) = 1_W(u) (an indicator); the formula remains valid with this convention. ∎
The F-slice identity is verified numerically against direct MC (rel dev 1e-3, P3b).

The scalar u of the work order ("u = E[f | H, ∇f=0]") is exactly u(h) above; since
F | H is 1-D Gaussian, u(h) plus the constant s_f capture the entire window factor.

### 4(c). Per-slice rigorous bounds on G(u)

Fix the slice; write H ~ N(m, S) (3-dim, general), Q := det H = zᵀMz,
δ := E[Q] = mᵀMm + tr(MS), V := Var(Q) (★). Let L Lᵀ = S (Cholesky),
Λ = diag(λ₁,λ₂,λ₃) the eigenvalues of LᵀML, U the eigenvector matrix,
ŵ = UᵀLᵀMm ∈ ℝ³, and c_neg := max(0, −minᵢ λᵢ).

**W2-11 (crude positive-part bound, valid).** G(u) = E[(D²+C²−T²)₊|u] ≤
E[D²+C²|u] = m_D² + S_DD + m_C² + S_CC (in TDC coordinates).
*Proof.* (x−y)₊ ≤ x for x ≥ 0. ∎ Sharp when T ≈ 0 and det<0 typical; loose when the
saddle event is rare.

**W2-12 (in-slice Cauchy–Schwarz, valid).** G(u) ≤ √( E[det²|u] · P(det<0|u) ),
with E[det²|u] by W2-9 applied to (m_H(u), S_{H·F}). *Proof:* W2-7 with B = Ω in the
slice. ∎ Strictly sharper than the global W2-7 whenever the window and the
H-distribution are not perfectly aligned (proved: integrating W2-12 with the true
P(det<0|u) and applying CS to the integral returns W2-7 with P(A∩B) in place of
min(PA,PB) — hence slice-CS ≤ global CS; with rigorous upper bounds on P(det<0|u)
the ordering is empirical, see probe §7).

**W2-13 (Cantelli, valid when δ(u) > 0).** P(Q < 0 | u) ≤ V/(V + δ²).
*Proof.* One-sided Chebyshev: P(Q < 0) = P(δ − Q > δ) ≤ E[(δ−Q)₊²]... precisely
Cantelli's inequality P(Q − δ ≤ −t) ≤ V/(V+t²) at t = δ (apply
P(Y ≥ a) ≤ (σ²)/(σ²+a²) to Y = δ−Q). ∎ Polynomial decay; never beats W2-14 by much
when δ²/V is large, but can beat it when δ²/V ≲ 1 (e.g. δ²/V = 0.5: Cantelli 0.667
vs Bernstein-form 0.88). Both valid ⇒ their min is valid.

**W2-14 (exact-mgf Chernoff bound, valid; the recommended tail bound).**
For all θ ≥ 0 with 1 + 2θλᵢ > 0 (i = 1,2,3):

  P(Q < 0 | u) ≤ e^{−θ mᵀMm} · ∏ᵢ (1+2θλᵢ)^{-1/2} · exp( 2θ² ŵᵢ² / (1+2θλᵢ) ),

and the inf over admissible θ is a valid bound.
*Proof.* P(Q<0) = P(e^{−θQ} > 1) ≤ E[e^{−θQ}] (Markov, θ ≥ 0). Write z = m + Lg,
g ~ N(0,I): Q = mᵀMm + 2ŵᵀ(Uᵀg) + Σᵢ λᵢ (Uᵀg)ᵢ² after rotating g ↦ Uᵀg (orthogonal,
law preserved). Factor per coordinate:
E[e^{−2θŵᵢg − θλᵢg²}] = (1+2θλᵢ)^{-1/2} exp(2θ²ŵᵢ²/(1+2θλᵢ})) for 1+2θλᵢ > 0
(complete the square; the condition is exactly integrability). Product over i and
multiply by e^{−θmᵀMm}. ∎ Verified: mgf formula vs MC at s = −0.05, +0.03 on 6
random instances (P3); bound ≥ true probability on all instances; bound ≤ W2-15
(below) on all instances.

**W2-15 (explicit-constant Bernstein form, valid, closed form).** If δ > 0:

  P(Q < 0 | u) ≤ exp( − δ² / ( 4 (V + 2 c_neg · δ) ) ).

*Proof (complete).* For 0 ≤ θ ≤ 1/(4c_neg) we bound log E[e^{−θQ}] termwise.
(i) λᵢ ≥ 0: −½ log(1+2θλᵢ) ≤ −θλᵢ + θ²λᵢ² since log(1+x) ≥ x − x²/2 for x ≥ 0.
(ii) λᵢ < 0: with y = 2θ|λᵢ| ≤ 2θc_neg ≤ 1/2, −½log(1−y) = ½Σ_{k≥1} y^k/k ≤
½y + ½y²·Σ_{k≥2}y^{k−2}/k ≤ ½y + ½y² = θ|λᵢ| + 2θ²λᵢ², using
Σ_{k≥2}y^{k-2}/k ≤ 1/(2(1−y)) ≤ 1.
(iii) 2θ²ŵᵢ²/(1+2θλᵢ) ≤ 2θ²ŵᵢ² (λᵢ ≥ 0) resp. ≤ 4θ²ŵᵢ² (λᵢ < 0, since 1/(1−y) ≤ 2).
Summing: log E[e^{−θQ}] ≤ −θ(mᵀMm + Σλᵢ) + θ²(2‖λ‖² + 4‖ŵ‖²) = −θδ + θ²V,
because Σλᵢ = tr(LᵀML) = tr(MS), δ = mᵀMm + tr(MS), and
V = 2‖λ‖² + 4‖ŵ‖² (direct Wick computation equals (★): Var(Q) =
2tr((MS)²) + 4mᵀMSMm = 2tr(Λ²) + 4‖ŵ‖²). Markov: P(Q<0) ≤ exp(−θδ + θ²V).
Choose θ* = δ/(2(V + 2c_neg δ)); admissible since θ* ≤ δ/(4c_neg δ) = 1/(4c_neg).
Then −θ*δ + θ*²V = −δ²(V + 4c_negδ)/(4(V+2c_negδ)²) ≤ −δ²/(4(V+2c_negδ))
because V + 4c_negδ ≥ V + 2c_negδ. ∎ Verified on all random instances (P3):
bound ≥ true P, and bound ≥ W2-14 (i.e., W2-14 is always at least as tight, as it
must be since it takes the exact inf).

**W2-16 (combined per-slice bound, valid).**
G(u) ≤ min( E[D²+C²|u],  √( E[det²|u] · min(1, Cant(u), Cher(u)) ) ),
where Cant(u) = W2-13 if δ(u) > 0 else 1, Cher(u) = inf-θ form of W2-14 (or the
closed-form W2-15 for a fully explicit chain). *Proof:* each ingredient valid;
min of valid upper bounds is valid. ∎

### 4(d). Sharper structural view (what is and isn't available)

Under the whitening of W2-14, Q = mᵀMm + 2ŵ·g + Σᵢ λᵢ gᵢ² exactly: det is an
**independent sum of shifted noncentral χ²₁ pieces**, with eigenvalues
λ = (λ₁,λ₂,λ₃) of mixed sign (M = diag(1,−1,−1) has signature (1,2), and S ≻ 0 ⇒
LᵀML has the same signature by Sylvester's law of inertia — so exactly one
λᵢ > 0 and two λᵢ < 0; a useful structural fact: the saddle event is exactly
"the negative-eigenvalue directions beat the positive one").

- If (and only if) ŵ = 0 and λ = (1,−1,−1)·σ² we recover the isotropic closed forms
  of §3. Under the 9-pin conditioning neither holds (§2.5), so **no exact closed
  form for G(u) exists in the conditioned problem** — G(u) is a genuine 3-D Gaussian
  integral; the valid rigorous handles on it are exactly W2-11/12/16 (upper),
  §5 (lower), or interval-arithmetic numerical integration (W3).
- A useful exact identity remains: det < 0 ⟺ D² + C² > T², i.e.
  P(det<0|u) = P( χ̃²₂(corr) > T² ) with (T,D,C) jointly Gaussian — W2-14 is the
  sharp generic bound on precisely this event; Sylvester signature means the event
  is genuinely "tail in ≥ 2 directions", which is why single-eigenvalue bounds are
  loose and the full mgf is needed.

---

## 5. Rigorous LOWER bounds on 𝔈(y) on a compact subregion

**Theorem W2-17 (box lower bound; valid for any admissible box).** Let
R = ∏ᵢ [mᵢ − aᵢ, mᵢ + aᵢ] ⊂ ℝ³ be a box in H-space centered at the slice mean
m = m_H(u₀) (or at the unsliced mean — see variants), with halfwidths aᵢ > 0, such
that

  det h ≤ −δ < 0  for all h ∈ R,   for some certified δ > 0.   (adm)

Then, with W = (b−ℓ, b), w̄ = b − ℓ/2 its midpoint, and
Pw(h) := P(F ∈ W | H = h) = Φ((b − μ_{F|H}(h))/s_f) − Φ((b−ℓ − μ_{F|H}(h))/s_f),

  𝔈(y) ≥ δ · P(H ∈ R) · min_{v ∈ vert(R)} Pw(v)
        ≥ δ · ∏ᵢ ( 2Φ(aᵢ/σᵢ) − 1 ) · min_{v ∈ vert(R)} Pw(v),

where σᵢ² = Sᵢᵢ and vert(R) is the 8 vertices.

*Proof of each step.*
(1) On R, |det h| 1{det h<0} = −det h ≥ δ by (adm), and 1{F∈W} ≥ 1{F∈W}; hence
E[|det|1{det<0}1{F∈W}] ≥ δ·E[1_R(H)1_W(F)] = δ·E[ 1_R(H) Pw(H) ] (tower).
(2) E[1_R Pw] ≥ P(R)·inf_{h∈R} Pw(h) (Pw ≥ 0).
(3) inf over the box is attained at a vertex: μ_{F|H}(h) = μ_F + cS^{-1}(h−m) is
affine in h, so Δ(h) := |μ_{F|H}(h) − w̄| is convex, hence maximized on the compact
box at a vertex; and Pw depends on h only through Δ, decreasing in Δ — because for a
fixed symmetric interval about w̄ and a symmetric unimodal (Gaussian) density,
d/dΔ Pw = φ((ℓ/2 − Δ̃)/s_f) − φ((−ℓ/2 − Δ̃)/s_f)-type derivative is negative for
Δ > 0 (standard; direct computation). Hence inf_{R} Pw = min_{vert(R)} Pw. ∎
(4) P(H ∈ R) ≥ ∏ᵢ P(|Hᵢ−mᵢ| ≤ aᵢ) = ∏ᵢ (2Φ(aᵢ/σᵢ)−1) is **Šidák's inequality**
(Šidák 1967: for centered jointly Gaussian Z, P(∩{|Zᵢ|≤aᵢ}) ≥ ∏P(|Zᵢ|≤aᵢ)); applies
since R is centered at the mean. ∎

**Certifying (adm) in closed form.** In (A,B,C) coordinates, det = AB − C². Since R
is a product box, sup_{R} det = (max over the 4 (A,B)-vertices of AB) − (min over
the C-endpoints of C², i.e. 0 if 0 ∈ [c₁,c₂] else min(c₁²,c₂²)); (adm) holds iff
this number ≤ −δ. A convenient construction: pick a sign pattern with
|m_C| − a_C =: r_C > 0 and (|m_A|+a_A)(|m_B|+a_B) =: p_AB; then δ = r_C² − p_AB
(>0 required). In TDC coordinates: det ≤ (|m_T|+a_T)² − (max(0,|m_D|−a_D))² −
(max(0,|m_C|−a_C))², same recipe.

**Variants (all valid).** (i) Unsliced version: apply W2-17 directly to the 4-dim
(F,H) law with R a box in H-space and Pw as above — this is the form stated.
(ii) Box not centered at the mean: replace step (4) by any valid Gaussian
orthant/rectangle lower bound (e.g. Šidák after recentering to the mean and
shrinking: R' = centered box ⊂ R with halfwidths aᵢ − |mᵢ − cᵢ| if positive).
(iii) Union of boxes: sum the bounds (disjoint boxes) — monotone convergence keeps
validity; this refines toward the true value.
*Numerical verification (P3b):* random 4-dim instance, box with certified
δ = 7.04: LHS (MC) = 4.065 ≥ bound 0.080 ✓; vertex-min property verified against
2e4 random interior points; Šidák verified on 3 random instances (P3).

**Role.** This is the refutation instrument for any proposed upper budget U: if a
certified (δ, R) at some y ∈ Z_r gives p_G(0)·δ·P(R)·min Pw integrated over a
subregion exceeding U, the budget is dead — with every factor closed-form and
interval-certifiable. It is loose by construction (throws away all mass outside R
and all size-|det| variation inside R) but nonzero whenever the Gaussian law is
nondegenerate, which is exactly what a falsification needs.

---

## 6. Validity ledger

### PROVEN VALID (proofs above; numerically verified in W2_sanity.py)

| # | Statement | Sharp where | Loose where |
|---|-----------|-------------|-------------|
| V1 | Global CS with min inside the root (W2-7) | X 1_A ∝ 1_B a.s. | rare-window × decorrelated H (loses corr factor √(P(A∩B)/min)) |
| V2 | Exact E[det²] by Isserlis (W2-9); E[det], Var(det) (★) | exact identities | — |
| V3 | F-slice / H-slice factorizations (W2-10) | exact identities | — |
| V4 | In-slice CS (W2-12) | within-slice |det| ∝ 1{det<0} | tail-dominated slices |
| V5 | Crude G(u) ≤ E[D²+C²|u] (W2-11) | saddle event typical, T≈0 | rare-event slices |
| V6 | Cantelli (W2-13), needs δ(u)>0 | δ²/V ≲ 1 | δ²/V ≫ 1 (polynomial vs exponential) |
| V7 | Chernoff inf-θ with exact mgf (W2-14) | universally; tight up to a Mills-type polynomial prefactor | loses that prefactor only |
| V8 | Explicit Bernstein exp(−δ²/4(V+2cδ)) (W2-15) | same regimes as V7, fully closed-form | constant-factor looser than V7 (θ grid vs sub-optimal θ*) |
| V9 | min-combination (W2-16) | — | — |
| V10 | Box lower bound + Šidák + vertex-min (W2-17) | box captures the bulk of det<0∩window mass | else conservative (but always valid, always positive under nondegeneracy) |
| V11 | Isotropic closed forms (W2-6) — **only** unconditional / given F alone | exact identities in their domain | inapplicable under pins (see I2) |
| V12 | Sylvester signature: exactly one λᵢ > 0, two λᵢ < 0 (§4d) | structural fact | — |

### PROVEN INVALID (counterexamples exhibited; do not use)

| # | Statement | Kill |
|---|-----------|------|
| I1 | E[X1_A1_B] ≤ √(E[X²])·P(B) — "dropped inner root" | W2-8: abstract counterexample (ratio √2 at P(B)=1/4 → ∞ as P(B)→0); in-context isotropic ratio 1.69 (P3) |
| I2 | Reusing unconditional independence T ⟂ (D,C), unit variances, or the W2-6 closed forms under the 9-pin conditioning | measured Σ(T,D,C | pins,∇=0) has |off-diag| sum 0.379, variances (0.25, 0.17, 0.16) ≠ 1 (P4) |
| I3 | Any bound of the form 𝔈 ≤ √(E[det²])·P(A∩B)-without-root or P(A)·P(B)-style factorizations | same mechanism as I1 (correlation can saturate 1/√P) |
| I4 | Replacing P(det<0|u) by 1{δ(u) ≤ 0} ("sign of the mean decides") | Q is a quadratic form, not linear: P(det<0|u) > 0 even when δ(u) > 0 (V7 bounds it, MC confirms nonzero mass) |

### Relative strength (probe-verified, §7)

Slice chain (V3+V16) ≪ global chain (V1+V7) in the rigidity zone (5.0e-23 vs 3.5e-12
at the probe point — 11 orders, because the global chain wastes the exact φ-window
integration); V7 ≤ V8 always (proved and observed); min(V6,V7) captures the better
of the polynomial/exponential regimes; V5 wins in slices where det<0 is typical.

---

## 7. Recommended bound chain for W3 (rigorous numerics)

**Upper chain (recommended; every ingredient closed-form except a 1-D integral):**

  ρ_WP(y;r) ≤ p_G(0) · min{ UB_global(y), UB_slice(y) },

  UB_global(y) = √( E[det²] · min( P_W, Cher ) ),     P_W = Φ((b−μ_F)/s_F) − Φ((b−ℓ−μ_F)/s_F),

  UB_slice(y) = ∫_{b−ℓ}^{b} φ(u; μ_F, s_F²) · min( E[D²+C²|u], √( E[det²|u] · min(1, Cant(u), Cher(u)) ) ) du,

with p_G(0), μ_F, s_F, μ_G, Σ_GG from Theorems W2-1/2; E[det²], E[det²|u] from
W2-9; Cant(u) = V/(V+δ²)·1{δ>0} + 1·1{δ≤0}; Cher(u) = W2-14 on a θ-grid certified
by interval arithmetic, or the fully closed-form W2-15; all Gaussian CDFs evaluated
with rigorous bounds. Numerical quadrature of the 1-D integral with interval
arithmetic (integrand smooth, explicitly bounded derivatives; window length
ℓ = r³/6 tiny ⇒ a few dozen nodes suffice with certified remainder).
Then I_WP(r) ≤ ∫_{Z_r} UB(y) dy (quadrature + certified sup/mesh argument — the
"sup-over-zone" step remains the C030/C031-named formality, outside W2's scope).

**Why this is tight enough.** At probe y = (1.0, 0.3) (rigidity zone, r = 0.025):
μ_F = 1.4294 (mean pulled above b), s_F = 0.0257 ⇒ P_W = 2.0e-22 (exact window
kill); E[det] = 4.78 > 0, Chernoff ≤ 0.073; UB_global = 3.5e-12;
**UB_slice = 5.0e-23** — eleven orders under the registered rigidity budget 3.6e-15
and ~17 orders under the 3.3e-6 total. The slice chain has the right structure to
reproduce (and improve) the rigidity-kill mechanism e^{−(b−m)²/2v} with no measured
constants anywhere.

**Lower chain (refutation instrument):** W2-17 with certified (δ, R) per §5, summed
over a disjoint box family if needed; every factor closed-form. Use against any
proposed replacement of the 0.213r³ budget.

**Fallback/refinement:** G(u) is a 3-D Gaussian integral; W3 may evaluate it
directly with interval arithmetic (tensor-product tanh-sinh on (T,D,C) whitened
coordinates; integrand |det|1{det<0} has an explicit Lipschitz bound). This turns
UB_slice into an equality up to quadrature certificates.

---

## 8. Executable falsifier (what kills this derivation)

Any ONE of the following, independently reproduced, refutes the corresponding part
(F1–F4 refute everything downstream; F5–F6 refute the model interface):

- **F1.** Recompute symbolically (e.g. SymPy integration) the isotropic slice:
  E[|det H|1{det H<0} | f=u] with H|f=u ~ (N(−u,2), N(−u,2), N(0,1)) indep. If the
  result is not √2·e^{−u²/4}, §3 is dead.
- **F2.** Recompute the Gaussian quadratic-form mgf E[e^{s·zᵀMz}] for a 1-dof case
  by hand and show W2-14's formula differs — kills V7/V8 and the recommended chain.
- **F3.** Exhibit any nondegenerate Gaussian (m,S) with certified P(det<0) exceeding
  W2-15's exp(−δ²/4(V+2cδ)) (e.g. by quadrature at high precision) — kills V8.
- **F4.** Show E[det²] (W2-9) wrong on any Gaussian instance via symbolic fourth
  moments — kills V1/V2/V4.
- **F5.** Independently recompute (μ_t − b)/ℓ at r = 0.025 from the model (e.g. via
  the Poisson-summed kernel or a finer lattice); a value outside
  −0.4997159 ± 2e-3 kills the conditioning pipeline (§2.1, §2.4).
- **F6.** Independently recompute Var(f(y)|9 pins) at d = 1/2/3 on any ray; values
  outside ±50% of the C030 triple (0.018/0.55/0.976) everywhere would kill the
  kernel/covariance conventions (C1–C3).
- **F7 (instant).** `python3 W2_sanity.py` — exit code ≠ 0 (any of the 114 ck()
  checks FAIL) invalidates this document wholesale. The script is fail-closed and
  uses no bare asserts.

---

## Appendix A. Sanity script and run record

`W2_sanity.py` (same directory). Structure: P0 kernel/truncation/spectral moments;
P1 unconditional BF covariance structure; P2 isotropic closed forms vs MC +
C030/C031 measured-constant cross-checks; P3 general quadratic-form machinery
(Isserlis E[det], E[det²], Var; mgf; Cantelli/Chernoff/Bernstein validity; global
CS; defective-inequality counterexample; Šidák); P3b lower-bound chain, vertex-min,
slice identity; P4 the actual 9-pin pipeline at r = 0.025 in 50-digit mpmath
(Σ_PP Cholesky-PD, μ_t, horizon scan, DF probe, two-stage=one-shot, TDC
correlation, bound values at the probe point, conditioned-law CS check).
Run record: `run1.log` — **114 PASS, 0 FAIL**, exit 0. numpy 2.2.5, scipy 1.16.2,
mpmath 1.3.0, CPython 3.12.

## Appendix B. Zone quotes and budget arithmetic

Registered budgets at r = 0.025 (r³ = 1.5625e-5): total 0.213·r³ = 3.33e-6;
rigidity 3.6e-15; transition 1.95e-6; far 1.37e-6 (C031 §3, quoted verbatim in §0).
The window length ℓ = r³/6 = 2.6042e-6 at r = 0.025.
