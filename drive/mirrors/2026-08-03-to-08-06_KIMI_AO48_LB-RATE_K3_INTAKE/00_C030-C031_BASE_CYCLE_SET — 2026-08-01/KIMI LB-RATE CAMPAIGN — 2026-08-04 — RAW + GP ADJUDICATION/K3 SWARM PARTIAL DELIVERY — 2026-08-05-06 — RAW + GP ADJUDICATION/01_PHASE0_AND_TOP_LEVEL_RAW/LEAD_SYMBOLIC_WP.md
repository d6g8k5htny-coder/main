# LEAD SYMBOLIC WP DERIVATION (second independent derivation for gate G1)

Author: lead coordinator (KIMI line). Date: 2026-08-05.
Independence statement: this derivation was produced in-session on 2026-08-05 BEFORE workstream W2 was
dispatched, from the exact kernel up, and is frozen here unchanged for the G1 comparison. It does not
consult W2's document. Conventions follow the manuscript's §2 (estimand definitions restated below).

## 1. Estimand and conventions

Field: normalized periodized Bargmann–Fock field on T²₂₄ (Var 1, factorized kernel K = K1⊗K1,
spectral (π/12)ℤ² masses e^{−|k|²/2} / Z², Z = Σ_j e^{−j²/2}; wrapped dual
K1(s) = (24/(√(2π)Z)) Σ_n e^{−(s+24n)²/2}). Pins: (f, ∂₁f, ∂₂f) at M = (−r/2,0), S = (r/2,0),
Y = M + r(−0.76, 0.24); values f(M) = b = 6/5, f(S) = b − ℓ (ℓ = r³/6), f(Y) = μ_t
(μ_t = E[f(Y) | 6 pins at M,S, ∇f(Y) = 0]), gradients 0. Conditioned field f̃ = f | pins.
Estimand (typed, saddles det H < 0, Jacobian |det H|, multiplicity 1, conditioning on ∇f̃(y) = 0 only):

    ρ_WP(y; r) = p_{∇f̃(y)}(0) · E[ |det H_y| · 1{det H_y < 0} · 1{b−ℓ < f̃(y) < b} | ∇f̃(y) = 0 ],
    I_WP(r) = ∫_Z ρ_WP(y; r) dy,   Z the rigidity zone {d ≲ 1.5} about the pin cluster.

## 2. Exact conditional law (two-stage ≡ one-shot)

Let Z = (f(y), ∇f(y), H_y) ∈ ℝ⁶, P the 9 pins. Joint Gaussianity gives (Kronecker):
Σ_ZZ, Σ_ZP, Σ_PP all exact kernel evaluations; Σ_PP ≻ 0 (spectral nondegeneracy, full Fourier support).
E[Z|P], Cov(Z|P) = Σ_ZZ − Σ_ZP Σ_PP⁻¹ Σ_PZ. Second stage: condition on ∇f̃(y) = 0 by another
Kronecker step on the (f, H) marginal given ∇: μ̃ = E[f|P,∇=0], ṽ = Var(f|P,∇=0), ν = E[H|P,∇=0],
Σ_H = Cov(H|P,∇=0), and the cross vector c = Cov(f, H | P, ∇=0). Two-stage ≡ one-shot (tower property
of Gaussian conditioning); verified numerically to 1e-16 in-session.

## 3. The valid inequality chains

(A) Corrected Cauchy–Schwarz (valid):
    E[|det H|·1{det<0}·1{f∈W}] ≤ √(E[det²H]) · √(P(det<0 ∩ W)) ≤ √(E[det²H]) · √(min(P(det<0), P(W))).
    Dropping the inner root is invalid: for p ∈ (0,1), p < √p; counterexample in-context at the witness
    point (−0.04, −0.58), r = 0.025: dropped-root 1.31310e-4 vs valid 2.30247e-2 (violation ratio 175.3).
    E[det²H] is exact by Isserlis: with C_{ij} = Σ_H,ij + ν_iν_j,
    E[det²H] = M(00,22) − 2M(01,12) + M(11,11), M(ij,kl) = C_{ij}C_{kl} + C_{ik}C_{jl} + C_{il}C_{jk}.

(B) The u-slice exact factorization (valid, structural): define u = E[f | H, P, ∇=0] = μ̃ + β·(H − ν),
    β = Σ_H⁻¹ c, so u ~ N(μ̃, s_u²) with s_u² = cᵀΣ_H⁻¹c, and f | (H, P, ∇=0) ~ N(u, s_f²),
    s_f² = ṽ − s_u² > 0. Then, exactly,
        E[|det H|·1{det<0}·1{f∈W} | P, ∇=0] = ∫_ℝ φ(u; μ̃, s_u²) · G(u) · Pw(u) du,
        G(u) = E[|det H|·1{det H<0} | u],   Pw(u) = Φ((u−(b−ℓ))/s_f) − Φ((u−b)/s_f),
    since P(f∈W | H,∇=0) depends on H only through u. This reduces the 4D expectation to an exact 1D
    integral of a 3D-conditional expectation. (Independently derived in-session 2026-08-05 before W2.)

(C) Per-slice rigorous bounds on G(u) (valid): with a(u) = E[H | u] = ν + (c/s_u²)(u − μ̃) and
    V = Cov(H | u) = Σ_H − ccᵀ/s_u² (u-independent), exact by Gaussian conditioning:
        G(u) ≤ √(E[det² | u]) · √(P(det<0 | u))   (in-slice CS),
        P(det<0 | u) ≤ Cantelli(u) = Vdet(u)/(Vdet(u) + max(Edet(u),0)²)   when Edet(u) > 0 (≤ 1 else),
    with E[det² | u] and Edet(u) = E[det H | u], Vdet(u) = Var(det H | u) exact by Isserlis on
    N(a(u), V) (Vdet(u) is u-independent up to the mean shift; computed exactly). Quadratic-form tails
    (Hanson–Wright class with explicit constants; Laurent–Massart) apply to P(det<0|u) when Edet(u) > 0
    (W11 methods). The det = T² − D² − C² form is available; under the 9-pin conditioning (T,D,C) are
    correlated (measured, in-session), so the isotropic closed forms do NOT transfer — quadratic-form
    machinery is required, in agreement with W2's structural finding (stated here from my own pipeline).

(D) Lower bound on a compact subregion (valid): for any box R in H-space with det H ≤ −δ < 0 on R,
        E[|det H|·1{det<0}·1{f∈W}] ≥ δ · P(H ∈ R) · min_{v ∈ Vert(R)} Pw(u(v)),
    using positivity and the exact window factor; P(H ∈ R) for the 3D Gaussian is any certified
    lower bound (Šidák/product or quadrant bound). Every step elementary.

## 4. What is estimate vs enclosure in the lead's existing numbers

The two-sided evaluation I_true ≈ 4.7602e-6 at r = 0.025 used (B) with Gauss–Hermite inner
expectations and Riemann spatial grids (mesh stability 0.1/0.05 = 2.8%, strip 0.0125, two engines
agreeing 1.6% at the dominant point): a derived-on-grid ESTIMATE, not an enclosure. The advertised
"certified upper bound" 1.30399e-5 used the dropped-root factor (invalid, A-counterexample above) and
is quarantined. The corrected global CS (A) is expected at ~1e-2 zone level (too weak); the per-slice
chain (B)+(C) is the lead's recommended enclosure path, consistent with W2's recommendation (G1
comparison item).

## 5. Falsifier

An independent symbolic re-derivation showing (i) the u-slice factorization (B) does not hold as stated,
(ii) the conditional laws of §2 are wrong beyond 1e-50 cross-representation tolerance, or (iii) any of
(A)/(C)/(D) is not a valid inequality in the Gaussian class, kills the corresponding section.

Frozen. Body hash computed at delivery into MANIFEST.
