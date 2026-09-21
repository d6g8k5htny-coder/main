# KIMI-DER-027a — FAR-FIELD MONOTONE DECAY ENVELOPE (AO48-WO-063 Task 4a)

**Verdict: PROVED for the load-bearing value-variance channel — all recorded constants
(d = 3: ≤ 2.24%; d = 5: ≤ 1e-4; d = 6.1: ≤ 7.5e-13 < 1e-4) reproduced at certified
grade, with a derived, certified-monotone envelope beyond d₀ = 3 and a derived
variance floor. The value-law envelope is likewise PROVED (certified TV bound ≤ 1e-4
at d = 5, monotone envelope), and the recorded 6-pin value-law numbers 7.7e-5 / 8.1e-5
are matched at measured grade (deviations 0.39% / 0.19% below the recorded figures;
the certified bounds 7.879e-5 / 8.304e-5 sit 2.3–2.5% above them, as any honest
sup-certificate must). ONE explicit isolation: the tasked witness value v* = −1/2 at Y
is macroscopically inconsistent with the 6-pin mean at Y and breaks value-law
stabilization (certified |μ|̄(5) = 1.09 / 4.50; Section 6); the variance channel is
unaffected (pinned values never enter Var).**

## Inputs (hashes echoed)
- Work order: `AO48-WO-063 ... 2026-08-04.md`, sha256
  `e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2` (verified against upload).
- `C022 Observed Update.json`, sha256 `9bc0647b885931ba4dc862bedfb879f9bd12c4cd6d39f51c6e98e54c7f75d5cb`
  (present in /mnt/agents/upload/ at 9,435 B; contradicts the WO's "absent" listing — noted for reconciliation).
- `c022 fd.json`, sha256 `394a95990644cd49221a894e2b3a5954a01dad70142f75c49f0cd6ffe9b3290a` (7,007 B).
- LB-3 S2–S4 recomputation class: not in the intake; the recorded constants are taken from
  the task statement (2.24% at d = 3; 7.7e-5/8.1e-5 6-pin value law at d = 5; variance
  enveloped < 1e-4 from d = 6.1, worst case ≤ 7.5e-13).

## 1. Model (exact)
Field: periodized side-24 2D Bargmann–Fock field with spectral lattice (π/12)ℤ²,
masses e^{−|k|²/2}, normalized covariance
  C(u) = Z₃₀⁻¹ Σ_{|k|≤30} e^{−|k|²/2} cos(k·u),  C(0) = 1   (EXACT definition).
Certified truncation tail (this work, DPS80): T₀/Z₃₀ ≤ 3.02e-196 < 4.4e-185 (recorded
bound) ≪ 1e-60 (task bar). Poisson summation (exact identity) gives the wrapped
representation C_untr(u) = c₀ Σ_{n∈ℤ²} e^{−|u+24n|²/2}; wrapped sums (|n|∞ ≤ 2) with
certified image remainder ε_img (order 0: 2.16e-31; order 1: 5.17e-30) plus the
truncation correction 2T_{|j|}/Z₃₀ are the certified kernel used throughout.
Cross-representation agreement spectral↔wrapped: 2.62e-81 at the check point (< 1e-50,
Task-1 discipline), 2.4e-16 across 8 points at float64 class.

Pins (tasked conditioning): (f, ∇f) at M = (−r/2, 0), S = (+r/2, 0), values
β = (6/5, 0, 0, 6/5 − r³/6, 0, 0), plus one witness value pin at
Y = r·(−63/50, 6/25) with value v* = −1/2 (EXACT). Rungs r = 1/20, 1/40 (EXACT).
The variance channel is independent of the pinned *values* (Gaussian conditioning);
the value-law channel depends on β. The recorded "6-pin" constants correspond to the
ensemble without the Y pin; the tasked 7-pin ensemble includes it. Both are certified.

Domain: cluster annuli A_d = {y ∈ torus : dist(y, hull(pins)) ≥ d}, d ≤ 12
(toroidal representative |uᵢ| ≤ 12).

## 2. Objects and channels (precise definitions)
With v(y) = vector of covariances of f(y) with the pins, G = pin Gram matrix
(entries (−1)^{|α|} ∂^{α+β} C, Hermite-exact), Schur complement:
- Δ(y) := 1 − Var(f(y)|pins) = v(y)ᵀ G⁻¹ v(y)  — the LOAD-BEARING value-variance channel.
- μ(y) = βᵀ G⁻¹ v(y), σ²(y) = 1 − Δ(y); value-law channel T(y) = TV(N(μ, σ²), N(0,1)).
Δ̄(d) := sup_{A_d} Δ, and likewise T̄(d). Both are non-increasing in d by domain nesting.

## 3. Derived envelope (constants derived, not fitted)
Certified Gram data (DPS80, Weyl-residual certification λ̂ = λ̃₁ − ‖R‖_F − 1e-70):
  λ_min(G): 2.5677067e-10 (7-pin, r=1/20); 4.7566847e-12 (7-pin, r=1/40);
            3.2552060e-10 (6-pin, r=1/20); 5.0862628e-12 (6-pin, r=1/40);
  inverse residuals ‖G⁻¹ − M‖_F ≤ 9.58778e-61 / 3.76507e-58 / 7.08626e-61 / 4.86464e-58
  (same order); moment-coefficient certification error cWerr ≤ 8.63e-58.

Power–Gaussian sup functions (EXACT, derived): P_{i,k}(d) = sup_{|u|≥d} |u₁|ⁱ|u₂|ᵏ e^{−|u|²/2}
equals i^{i/2}k^{k/2}e^{−(i+k)/2} for d² ≤ i+k (unique interior critical point
u₁²=i, u₂²=k) and d^{i+k}(i/(i+k))^{i/2}(k/(i+k))^{k/2}e^{−d²/2} for d² > i+k (boundary
circle maximum at cos²θ = i/(i+k)); continuous, globally non-increasing.

Kernel bounds (derived): |∂^{m}C(u)| ≤ Σ_{i,k} |He_{m₁}[i] He_{m₂}[k]| P_{i,k}(d) + corrections
(Hermite coefficient triangle — EXACT integer polynomials; corrections (1+τ)ε_img + 2T/Z₃₀).

Schur-complement moment expansion (derived): with M = QνQᵀ the certified inverse and
W = ν^{1/2}Qᵀ, and Taylor expansion of ∂^{α_p}C(y−x_p) about y through order J = 8,
  Δ(y) ≤ Σ_q ( Σ_{|m|≤J+2} |c_{q,m}| B_m(d) + R_q(d) )² + 1.01·‖G⁻¹−M‖_F·‖v‖²_crude,
where c_{q,m} = Σ_p W_{qp}(−1)^{|α_p|}(−x_p)^{m−α_p}/(m−α_p)! are EXACT-structural moments
of the dual basis (computed DPS80, certification error ≤ 8.6e-58), B_m(d) the kernel
bounds above, and R_q(d) the certified factorial Taylor tail (orders 9–30, ratio-audited).
Value law: |μ(y)| ≤ Σ_m |cw_m| B_m(d) + R^w(d) + ε-residual, cw from Mβ.
Every term is non-increasing in d; hence the envelope Ê(d) is non-increasing on (0, 12],
in particular beyond the stated radius **d₀ = 3**.

## 4. Certified exact suprema at the recorded knots (mesh + outward pad)
At each knot d ∈ {3, 5, 6.1}: float64 wrapped-kernel evaluation of Δ (and |μ|) on a
(θ, t) mesh of A_d ∩ {t ≤ d+1.25} (4096 × 625 cells), certified geometry pad
pad = (max mesh |∇F|)·ρ + H_cert(d)·ρ² with H_cert the derived envelope bound on
|∇²Δ| (same moment machinery, derivative shifts ≤ 2), plus float64 pipeline pad
1e-3·mesh_max + 10×|f64 − DPS80| at the argmax (observed discrepancies ≤ 8.4e-8);
beyond t = d+1.25 the analytic envelope Ê(d+1.25) dominates. Argmax points
re-evaluated at DPS80 (agreement ≪ pad, displayed per knot).

Certified values (DPS80/F64PAD classes, from the transcript; pads and DPS80 anchors
displayed there):

| quantity | 7-pin r = 1/20 | 7-pin r = 1/40 | 6-pin r = 1/20 | 6-pin r = 1/40 |
|---|---|---|---|---|
| Δ̄(3) certified | 0.0194319 | 0.0204043 | 0.0193899 | 0.0203911 |
| Δ̄(5) certified | 3.45513e-8 | 3.81227e-8 | 3.43132e-8 | 3.80577e-8 |
| Δ̄(6.1) certified | 5.28475e-13 | 5.96702e-13 | 5.22296e-13 | 5.95071e-13 |
| Ê(3) analytic | 0.120636 | 0.0888361 | 0.0541125 | 0.0539988 |
| Ê(5) analytic | 1.39668e-7 | 9.80575e-8 | 6.38662e-8 | 6.3629e-8 |
| Ê(6.1) analytic | 1.94556e-12 | 1.32433e-12 | 8.89022e-13 | 8.84675e-13 |
| Ê overhead at d=3 | 6.208× | 4.354× | 2.791× | 2.648× |

Recorded checks: Δ̄(3) ≤ 2.24e-2 PASS both rungs; Δ̄(5) ≤ 1e-4 PASS; Δ̄(6.1) ≤ 7.5e-13
and < 1e-4 PASS. Analytic envelope certified non-increasing on the grid 3.0 → 12.0
(step 0.1) for all four configurations, and dominates the certified exact sup at every
knot (join checks PASS).

**Theorem (value-variance channel).** For both rungs and the tasked 7-pin conditioning
(any pinned values, in particular v* = −1/2): for all y with dist(y, hull(pins)) ≥ d,
3 ≤ d ≤ 12,
  Var(f(y)|pins) ≥ 1 − Δ̄(3) ≥ 1 − 0.0204043 = 0.9795957 (derived constant),
  Δ̄(d) ≤ min(Ê(d), certified knot curve) with Ê derived as in §3 and certified
  non-increasing on [3, 12],
and the recorded constants hold: Δ̄(3) ≤ 2.24e-2, Δ̄(5) ≤ 1e-4, Δ̄(6.1) ≤ 7.5e-13.

## 5. Value-law channel at d = 5 (6-pin, recorded constants)
Certified |μ|̄(5) (mesh + pad) and σ_inf = √(1 − Δ̄(5)); certified
T̄(5) ≤ |μ|̄/(σ_inf√(2π)) + TV_σ(σ_inf), with exact DPS80 TV at the certified argmax
(exact quadratic density-crossing solve):
- r = 1/20: certified |μ|̄(5) = 1.97474e-4, certified T̄(5) = 7.87891e-5 ≤ 1e-4;
  exact TV at certified argmax y = (5.026, 0) is 7.66991e-5 — recorded 7.7e-5:
  **MATCH**, deviation 0.39% below the recorded figure.
- r = 1/40: certified |μ|̄(5) = 2.08127e-4, certified T̄(5) = 8.30397e-5 ≤ 1e-4;
  exact TV at certified argmax y = (5.014, 0) is 8.08430e-5 — recorded 8.1e-5:
  **MATCH**, deviation 0.19% below the recorded figure.
Margin note (honest, non-load-bearing): the certified sup bounds (7.879e-5 / 8.304e-5)
sit 2.3–2.5% above the recorded measured numbers — expected direction for a
sup-certificate over the continuum annulus (recorded values are measured extrema).

## 6. Isolation: witness pin v* = −1/2 breaks value-law stabilization
For the tasked 7-pin ensemble with v* = −1/2 (the value −0.5 is macroscopically
inconsistent with the 6-pin conditional mean ≈ 1.2 at Y, forcing dual weights ~1e5):
certified |μ|̄(5) = 1.09176 (r = 1/20) and 4.50189 (r = 1/40) — the one-point
value-law TV at d = 5 is ≈ 0.44 / ≈ 0.99. The recorded value-law constants therefore
do NOT extend to the v* = −1/2 conditioning; the variance channel (the load-bearing
one) is unaffected (values never enter Var). If the intended witness value is the C022
barrier value b − ℓ/2 ≈ 1.2 (the 9-pin y* pin), the d = 5 cluster-domain value-law TV
is ≈ 8.46e-5 (r = 1/20) / ≈ 8.23e-5 (r = 1/40) (float64 recomputation, this work) —
still ≤ 1e-4 and at the recorded 8.1e-5 scale. Counterexample of record for the
isolation: the S8 mesh argmax on {dist(y, hull{M,S,Y}) ≥ 5} (transcript), where the
certified |μ| suprema 1.09176 / 4.50189 are attained within the displayed pads; for
the 6-pin value law the certified argmax points are y = (5.026, 0) (r = 1/20) and
y = (5.014, 0) (r = 1/40).

## 7. Hypotheses
H1. The model of §1 (side-24 periodized spectral field, |k| ≤ 30 truncation with the
    certified tail) is the intended "exact" field — EXACT definition, no hypothesis.
H2. Poisson summation identity for the lattice Gaussian (standard, exact).
H3. Recorded constants taken from the task statement (LB-3 recomputation absent from intake).
H4. Gaussian conditioning (Schur complement) — exact for the Gaussian field.
No probabilistic or fitted input; every constant is derived from C's exact structure
(Gaussian masses, Hermite-exact derivatives, Schur complement with certified inverse).

## 8. Dependency table
| item | hash (sha256) | use |
|---|---|---|
| AO48-WO-063 work order | e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2 | task definition |
| C022 Observed Update.json | 9bc0647b885931ba4dc862bedfb879f9bd12c4cd6d39f51c6e98e54c7f75d5cb | recorded gamma_LOC corollary, Lemma FD context |
| c022 fd.json | 394a95990644cd49221a894e2b3a5954a01dad70142f75c49f0cd6ffe9b3290a | prior per-d tables (cross-check only) |
| verify_farfield_envelope_v1.py | e898527dc10ad01e50d1e457c8e7166f7744e7f606d0f435deb5557a89539e6f | certificate |
| transcript (.out.txt / .O.out.txt) | 266b0f442b323598b28bd08fa1f96e5201e186fa8e18429219d2484bc7087beb (both; byte-identical, 8900 B each) | audit |

## 9. Falsifier
The theorem dies if: (i) any independent certified recomputation of Δ̄(d) or T̄(d) at a
knot exceeds the displayed certified bound by more than the displayed pad; (ii) any point
y with dist(y, hull) ≥ 3 exhibits Δ(y) > Ê(dist); (iii) the certified λ̂ > 0 fails for
any Gram matrix (conditioning degenerate); (iv) the spectral tail bound T₀/Z₃₀ is violated;
(v) the two transcripts (normal / python3 -O) differ in any byte.

## 9b. Remaining gap (one paragraph)
The certificate is complete for the load-bearing channel, but three gaps of record
remain. (i) The recorded d = 3 bound 2.24% is certified by the validated mesh layer
(certified Δ̄(3) = 0.0194 / 0.0204), not by the closed-form envelope: the derived
analytic Ê(3) = 0.054–0.121 carries a 2.6–6.2× overhead from the Hermite coefficient
triangle and the per-direction sup reduction, so Ê is the monotone/tail carrier while
the knots are carried by direct certified computation; tightening Ê to mesh sharpness
would require direction-resolved Hermite sups (exact per-θ polynomial sup functions),
not the per-radius power–Gaussian sups used here. (ii) The mesh layer is float64 with
certified geometry pads (max-mesh-gradient × cell radius + H_cert·ρ²) anchored by DPS80
recomputation at every argmax (observed discrepancies ≤ 8.4e-8, covered 10× in the
pad); a full outward-rounded interval evaluation at every mesh point would raise the
rigor class but not any displayed constant. (iii) The honest discrepancy of record —
full one-point TV including the conditional mean is 6.6%–31% at d ≥ 3 — is not
re-certified here (it is non-load-bearing since C031 R3′ removed the additive-TV
usage); this certificate covers the load-bearing variance channel and the 6-pin value
law at the recorded knots only, and it shows the value-law constants are fragile to
the witness value: v* = −1/2 destroys them (§6), so any future use of the 7-pin
ensemble must pin Y at a consistent value (b − ℓ/2 keeps TV ≈ 8.5e-5 ≤ 1e-4 at d = 5).

## 10. Precision labels
EXACT: a = π/12, L = 24, r = 1/20, 1/40, b = 6/5, ℓ = r³/6, Y = r(−63/50, 6/25),
v* = −1/2, Hermite polynomials and power–Gaussian sup formulas, Poisson identity.
DPS80: all mpmath dps-80 quantities with displayed certified error bounds (tails, kernel
values, λ̂, inverse residuals, moment coefficients, envelope values, knot pads, exact TVs).
F64PAD: float64 mesh extrema with certified outward pads (validated at argmax by DPS80).
