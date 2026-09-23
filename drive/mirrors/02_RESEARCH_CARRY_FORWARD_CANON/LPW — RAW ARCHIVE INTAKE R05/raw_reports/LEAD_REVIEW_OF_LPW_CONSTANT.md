# LEAD REVIEW — LPW_CONSTANT certified explicit pair (c ≥ 6.239e-44, r₀ = 1/2414592)

Reviewer: lead coordinator (KIMI line). Date: 2026-09-13.
Target: LPW_CONSTANT package (LPW_CONSTANT_REPORT.md 289d9f40…; lpw_constant.py e258322c…;
falsify.py 435b40ce…; transcripts byte-identical both modes; RECEIPT.json 6332afe0…).
Contract: 03_CONSTANT_CERTIFICATE_CONTRACT.md (acceptance contract, verified against it item by item).

## VERDICT: PASS WITHIN SCOPE — the certified explicit pair stands at the contract's standard.

**1 − q(r, 6/5) ≥ c·r³ for all 0 < r ≤ r₀, with c ≥ 260/(3790446482793·2⁴⁰·10²¹) = 6.239e-44 ≥ 10^{−44} > 0 and r₀ = 1/2414592 = 1/(256·9432) ≈ 4.14149e-7 > 0.**

## Verification performed by the lead (independent of the author's and the agent's own checks)

1. **Clean-room re-execution of the certifier and falsifier (my own run):** lpw_constant.py exits 0
   with ALL_CHECKS_PASSED; falsify.py exits 0 with F1–F8 all PASS, including the four contract
   mutation defects (eigenfloor_up, hessian_power, box_width, planar_moments) each rejected with
   nonzero exit in both modes; transcripts byte-identical across python and python -O
   (sha256 d0cb9259…, matching the package's recorded value).
2. **Changed compact set (contract item 1):** JBOX(R) = [−(10+2δ)R, 0] × [2−δ, 2+δ] × [−δ, δ]²
   contains every thin box E_r for 0 < r ≤ R = 1e-5, by exact rational containment (q ∈
   2r[−5−δ, −5+δ] ⊂ [−(10+2δ)R, 0]; A, B, D_3 intervals unchanged). Stated explicitly, with all
   dependent bounds (density floor, B_4, conditional means) computed on this set — exactly what the
   contract requires; the original larger K_J is not silently retained.
3. **Covariance interval (contract item 2):** the interval modulus ‖Γ_r − Γ_0‖_F ≤ L_1·R,
   L_1 = 19.071156, comes from ANALYTIC profile-derivative bounds (|g_i| ≤ pb_i, |g_i′| ≤ pb1_i via
   |sin θ| ≤ |θ| and |∫₀^θ t sin t dt| ≤ min(θ²/2, |θ|³/3)) — not sampled rungs. The endpoint floor
   31/250 is used correctly (its ten Sylvester minors were independently reproduced by me and by the
   agent's F2). Then λ_min(Γ_r) ≥ 31/250 − L_1R = 0.123809288439 > 0 uniformly on [0, R], inherited by
   the six-pin block (Cauchy interlacing) and by the Schur complement (variational form of the Schur
   lower bound; the parenthetical S^{−1} = (Γ^{−1})_{JJ} is the standard identity, consistent).
   r_G := R = 1e-5 is thereby certified.
4. **Density floor (contract boxed formula):** R_J = 2.000977042 (sup |j| over JBOX(R), exact),
   M = 1.222575834 (sup |μ_r|, exact |μ_0| = a₂b plus interval correction), Λ = 3.000… (Gershgorin on
   the r-independent Cov J = diag(a₄, a₄a₂/4, a₂a₄/4, a₆/36), whose off-diagonals vanish by parity at
   t = 0 — independently confirmed by me), exponent (R_J + M)²/(2λ) ≤ 41.96491748 (recomputed:
   (3.22356)²/0.247618577 = 41.965 ✓). Then m = (2π)^{-2}Λ^{-2}exp(−41.96491748) with
   (2π)^{-2} > 1/40 exact → certified floor m ≥ 10^{−21} (estimate 1.676e-21; cross-check
   m ≤ 1.2832e-3 OK). Directions correct throughout (upper Λ, upper exponent, lower m).
5. **B_3/B_4 (contract §4):** A_3 = 71.10596, A_4 = 262.81357 (representer bounds with exact moments
   and trace sups from the same interval modulus); global moments E‖f‖_{C³}⁴ ≤ μ_4S_3⁴ =
   1.6146672e12, E‖f‖_{C⁴} ≤ μ_1S_4 = 3258.2936 with certified Fourier tails; Isserlis
   E‖U_r‖⁴ ≤ 38.23; sup‖(u_r, j)‖ ≤ 2.3701308; B_3 ≤ 3.790446483e12 ≤ 3790446482793,
   B_4 ≤ 4715.749569 (recomputed components consistent: 3258.29 + 262.81357·(3.175 + 2.3701) ≈
   4715.75); K ≤ 9432, C_Z ≤ 15161785931172. Residual never assumed stationary (the contract's §4
   regression-operator form is used).
6. **Final arithmetic (already independently consistent):** K = max(1, 2B_4) ≤ 9432;
   1/(256·9432) = 1/2414592 < R = 1e-5 so the K-term binds; 256·K·r₀ = 1 exactly;
   16δ + 8Kr₀ = 1/64 + 1/32 = 3/64 < 1/16 (exact Fractions); path clearance 103/3840 > 0;
   exponent 1 + 4 − 2 = 3; weight base (81/16)(1673/128) ≥ 65; 65·16 = 1040 = 4·260;
   volume 32δ⁴r; c = 260·10^{−21}/(3790446482793·2^{40}) = 6.239277637492383e-44 ≥ 6.239e-44
   (downward-safe; the author's own independent arithmetic check agrees).
7. **Exact finite-torus moments (contract item 2):** only moment forms (never planar rationals);
   1 − a₂ = 9.65254179896e-123 > 0 certified at 150 dps in both spectral and 576-sum forms
   (independently reproduced by me earlier); a₄ = 3 + 5.501948825e-120 (positive, the Return-03
   correction, reproduced); endpoint Schur/mean identities in a₂, a₄, a₆ verified to 1.6e-61.

## Scope and boundary (stated exactly)

- The certified pair applies on 0 < r ≤ r₀ = 1/2414592 (the K-term binds). It coexists with the
  author-side's deliberate fallback (r_* = 10^{−28}, c_* = 10^{−1235}, analytically verified): the
  certified c dominates everywhere the two overlap; the fallback extends the tiny-r regime.
- It does NOT identify a limiting coefficient, prove an asymptotic formula, or supply the missing 2D
  upper bound. The matching 2D upper theorem is a separate open dependency (Return-04 work order).
- The RB rung-table errata and the corrected rb_r2_part2_spectral_v2.py (new version; predecessor
  frozen) are consistent with this certificate's T_r route (sine terms correct; corrected density
  1.2832254e-3 reproduced to 2.2e-8 relative; both modes byte-identical).
- Status firewall: this review verifies the certificate. It changes no LS-CTL Boolean, eligibility
  predicate, theorem status, RP status, AO48 operator record, or q0 package status. Any status change
  requires separate operator adjudication (operator permission for evidence-qualified transitions is
  recorded as granted in Return-04; the mathematical predicates are the remaining gates).

## Falsifier

An interval-rigorous recomputation producing, on the same domain and law, a density floor below
10^{−21} at a certified point of JBOX(R) × [0, R], a covariance eigenvalue below the uniform floor
0.123809288439 on [0, 1e-5], or a conditional moment bound exceeding B_3/B_4's values, kills the
corresponding input and, if load-bearing, the certified pair. The four contract mutations must
continue to be rejected (rerun falsify.py: any zero exit).

SHA-256 of this review body (text after this line is excluded):
cd2c6363a9a3a12ec88b408ce5f202335ce040e3352e2c7ed8a8d94d2fad6bec
