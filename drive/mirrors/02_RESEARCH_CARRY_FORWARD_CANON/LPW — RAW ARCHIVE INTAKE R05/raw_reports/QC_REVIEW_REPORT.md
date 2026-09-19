# QC-RETURN03 — INDEPENDENT NUMERIC VERIFICATION REPORT

**Verifier:** adversarial verification agent (verification-only; no implementation files touched).
**Target:** `INBOX_LPW_Return_03/LPW_Return_03_2026-09-12/03_EXPLICIT_CONSTANTS_CANDIDATE.md`
(sha256 `18a6e27cb945c7fc80975e20f775724989de0d368ce7a6f2ee29f11bea165d70`), proposing
`r_* = 10^{-28}`, `c_* = 10^{-1235}`.
**Scope delegated:** numeric evidence + independent reproduction of the computable components
(items 1–7). The analytic chain Q1–Q7 at derivation level is the lead's verdict
(`LEAD_ANALYTIC_VERDICT.md`); this report is the numeric-evidence complement.

## OVERALL RESULT: PASS WITHIN SCOPE on all seven delegated items (Q1–Q7 numeric components).

No discrepancy, no bound violation, and no sign error was found anywhere. 335/335 fail-closed
checks PASS; byte-identical output under `python3` and `python3 -O`; deterministic on rerun;
6 mutation families correctly fail with exit 1 (including under `-O`).

## Verification instrument

`qc_return03_numeric.py` (sha256 `f3622d3cb1c5cf7d1d6f0cb4916449564965b23fe70003e818e16fa160fc281d`),
written from scratch from the derivation documents. **No author code was imported, read into
the computation, or reused** (the author's `tools/verify_return03.py` was read only to confirm
which constants it claims, never executed as part of this review).

- Exact field: periodized side-24 Bargmann–Fock, covariance `K(z) = k(z1)k(z2)`,
  spectral lattice `(π/12)ℤ²`, masses `exp(−|k|²/2)`, Z-normalized.
- Truncation `|k| ≤ 30` (lattice index `|n| ≤ 114`) with an **in-program certified tail**:
  first omitted shell `n = 115`, geometric ratio bound `ρ ≤ 3.910e-4 < 1/2`, certified tail
  for every derivative order ≤ 8: **T ≤ 2.0062e-185 < 1e-120 < 1e-60** (fail-closed gates).
  All truncated quantities carry certified error < 1e-100; every acceptance margin exceeds 1e-3.
- Precision: mpmath 100 dps throughout (requirement ≥ 60 dps); the endpoint-moment block runs
  at 220 dps because the planar deviations are O(1e-117)–O(1e-123), invisible at 100 dps.
- Gates: `ck()` → `SystemExit` on failure; zero bare asserts; JSON stdout with sorted keys.
- Eigenvalue certification: mpmath `eigsy` + per-pair residual gates (< 1e-80), orthonormality
  gates, trace-consistency gates; **plus full Sylvester leading-minor positive-definiteness
  certificates** for every claimed matrix interval (Γ_r − (13/125)I, 81I − Γ_r, Σ_r − (1/10)I,
  9I − Σ_r, at both rungs). Margins are ~2e-2 vs. certified arithmetic slack ~1e-85.
- **Independent second code path (Poisson dual):** every load-bearing number was recomputed
  with the image-sum representation `k24(t) = Σ_n exp(−(t+24n)²/2)/Σ_n exp(−288n²)` with
  termwise Hermite derivatives — a different formula, different summation, different
  cancellation structure. Agreement to all 25–40 displayed digits on: a_{2j} deviations,
  λ_min(Γ₀), λ spectra at r = 1/1600, the L² errors, μ_r, Σ_r spectrum, and the corner
  density minimum (see below).

## Item 1 — L² Hermite error table (§2): PASS

Exact L² errors of the six coordinates of `V_r − V_0` on the exact field, vs displayed bounds:

| rung r | coordinate | exact L² error | displayed bound | bound/exact |
|---|---|---|---|---|
| 1/4 | value | 1.03266e-4 | 3.35693e-4 | 3.251 |
| 1/4 | x-gradient | 6.20331e-5 | 3.13314e-3 | 50.51 |
| 1/4 | avg y-gradient | 1.34440e-2 | 8.59375e-2 | 6.392 |
| 1/4 | div-x/2 | 1.32490e-2 | 1.43229e-2 | **1.0811** |
| 1/4 | div-y | 1.00310e-2 | 2.86458e-2 | 2.856 |
| 1/4 | cubic | 7.95650e-3 | 1.43229e-1 | 18.00 |
| 1/10 | div-x/2 | 2.13238e-3 | 2.29167e-3 | **1.0747** |
| 1/20 | div-x/2 | 5.33545e-4 | 5.72917e-4 | **1.0738** |
| 1/40 | div-x/2 | 1.33414e-4 | 1.43229e-4 | **1.0736** |
| 1/100 | div-x/2 | 2.13476e-5 | 2.29167e-5 | **1.0735** |

(All other coordinates/rungs in `OUTPUT_NORMAL.json`; every one below its bound.) The total:
E‖V_r−V_0‖² exact vs bound `(186461/73728)r²`, e.g. r=1/4: 5.2022e-4 vs 1.58065e-1 (margin 304);
r=1/100: 1.3489e-9 vs 2.52904e-4 (margin 1.87e5). And `(186461/73728) = 2.529 < 4` confirmed
exactly (Fraction identity `121·1541/73728 = 186461/73728`).

**Adversarial note (not a defect):** the divided-x-gradient/2 bound `11r²/48` is genuinely
tight — its leading error term is the deterministic first Taylor term `f_xxxx(0)r²/48` with
L² norm `√a8·r²/48`, so the margin ratio is exactly `11/√105 → 1.0735`. The bound passes
because `a8 < 106 < 121`; a coefficient below ~10.25 would have failed. Similarly the row-1
(value) margin is 3.22 because the two leading remainder contributions cancel
(`1/384 − 1/192 = −1/384`), independently confirming the derivation's cancellation structure;
the x-gradient (row 2) error is empirically ≈ 0.016·r⁴ (its cubic terms cancel, so the
O(r³) bound gains a factor ∝ 1/r: margins 50 → 1252 across the rungs), and the cubic
(row 6) error is empirically ≈ 0.128·r² against the O(r) bound 55r/96 (margins 18 → 447).

## Item 7 — L² Taylor remainder inputs (§2): PASS

Exact L² remainders vs displayed bounds (both signs ±r/2 identical by symmetry):

| rung r | statistic | exact | bound | margin |
|---|---|---|---|---|
| all 5 | f(±r/2) degree-3 | ≈ √a8·(r/2)⁴/24 | 11r⁴/384 | **1.0735–1.0755** |
| all 5 | f_x(±r/2) degree-2 | ≈ √a8·(r/2)³/6 | 11r³/48 | **1.0735–1.0763** |
| all 5 | f_y(±r/2) degree-2 | ≈ √(a6a2)·(r/2)³/6 | 11r³/48 | 2.840 |
| all 5 | f_y(±r/2) degree-1 (row-3 input) | e.g. 2.1643e-3 at r=1/10 | 11r²/8 | 6.35 |

The same `11/√a8 = 1.0735` tightness applies — passes with positive margin at every rung.

## Item 3 — Endpoint moments (§1): PASS

Certified-truncation moments (220 dps block; certified tail 2.01e-185):

- `a2 = 1 − 9.65254179895991305500968634889e-123` — matches the author's MC-RETURN03
  140-digit diagnostic to all 36 displayed digits; also `a2 < 1` (exact sign confirmed).
- `a4 = 3 + 5.50194882540715044135552121887e-120` — matches; `a4 > 3` (the corrected sign).
- `a6 = 15 − 3.11951811112966366103108544392e-117` — matches; `a6 < 15`.
- `a8 = 105 + 1.75612…e-114`.
- All deviations < 1e-98 as claimed. Bounds: `0 < a2 < 2`, `0 < a4 < 4`, `0 < a6 < 16`,
  `0 < a8 < 106` all hold (the last is the binding one: `a8 ≈ 105`). Interior caps used in §6
  confirmed: `99/100 < a2 < 101/100`, `a4 < 301/100`.
- `sup_{z,|α|≤4} ‖∂^α f‖²_L² = max_{i+j≤4} a_{2i}a_{2j} = a8 = 105.000…`, so the sup norm is
  **√105 = 10.24695076595960 ≤ 11** ✓ (stationarity makes it z-independent). Order-4 product
  caps `a6a2 < 32`, `a4² < 16` confirmed.

## Item 2 — Covariance interval (§3, §6): PASS

Exact ten-vector covariance Γ_r on the exact field (spectral and image paths agree to 25 digits).

r = 1/1600 (claimed boundary):
- `λ_min(Γ_r) = 0.12655342940792017 > 13/125 = 0.104` — margin 2.255e-2, certified both by
  residual-controlled eigensolve and by all ten Sylvester minors of Γ_r − (13/125)I > 0.
- `λ_max(Γ_r) = 3.5670440911 ≤ 81`; `tr Γ_r = 10.08333 ≤ 81` (81I − Γ_r PD certified).
- Schur complement Σ_r of J given U_r: eigenvalues `{1/6, 1/2, 0.50000004883, 2.0}` ⊂
  `[1/10, 9]` — `(1/10)I ⪯ Σ_r ⪯ 9I` certified (8 Sylvester minor gates).
- Conditional mean at the exact observed `u_r`: `μ_r = (−1.19999999997965, 0, 4.8828126e-8, 0)`,
  `‖μ_r‖ = 1.20000 < 3`; the displayed β₀ formula matches the exact endpoint regression
  `Cov(J,U₀)A₀⁻¹` to 5.7e-101; `‖μ_r − β₀u_r‖ = 1.64e-15 ≤ 6/5`; `‖β₀u_r‖ = 1.20 < 7/5`;
  `‖u_r‖ = 1.24544 < 2`.
- Perturbation evidence: exact `‖Γ_r − Γ_0‖_op = 1.4018e-7` vs the proved cap `32r = 0.02`
  (the true modulus is ~1.4e5 times smaller; the cap is analytic, verified by the lead).
- Endpoint: `λ_min(Γ_0) = 0.1265534496672894796441563 > 31/250 = 0.124` — this independently
  reproduces Kimi's diagnostic central value 0.126553449667289 (and exceeds the disputed
  rounded floor 0.1265534497 by only ~3e-11, consistent with the supplement's rounding
  correction). `‖V_0‖_L2 = √tr Γ_0 = 3.17543 < 7`; `tr D = 4.91667 < 76/9 < 9`.

r = 1/800 (control, outside the claimed interval): `λ_min(Γ_r) = 0.1265533686 > 13/125`,
`λ_max = 3.5670 ≤ 81`, Σ_r ∈ [1/6, 2] ⊂ [1/10, 9] — the claimed bounds still hold with
margin (evidence of slack, not part of the claim).

## Item 4 — Density floor (§6): PASS

At r = 1/1600, exact conditional density of J | (U_r = u_r) over the original box
`K_J = [−11,1]×[2−δ,2+δ]×[−δ,δ]²`, δ = 1/1024:

- **Minimum at corner `(q,A,B,D3) = (−11, 2+δ, −δ, −δ)`; log₁₀(density) = −13.22305593722707.**
  (Consistent with the supplement's planar-reference diagnostic 5.98e-14 at the same corner.)
- Since `−2 log φ = Q(j) + const` with Q a **convex** quadratic (Σ_r⁻¹ PD), its maximum over
  the box is attained at a vertex: the corner minimum is the certified global box minimum.
  Adversarial scans agree exactly: 32 edges × 21 points and a 5⁴ = 625-point grid both give
  min = −13.22305593722707 (never below the corner value).
- Displayed floor `m_* = 10^{−1129}` is **1115.78 orders of magnitude below** the computed
  minimum (task expectation "hundreds of orders" satisfied). It is also ~637 orders below the
  derivation's own natural floor from the same formula, `(2π)⁻²9⁻²e⁻¹¹²⁵ = 10^{−492.09}` —
  i.e., m_* is a valid, deliberately weak lower choice, exactly as stated.

## Item 5 — B_3/B_4 chain inputs (§5, evidence): PASS

At r = 0.025 (eight z-samples × all 15 multi-indices |α| ≤ 4):
- `max ‖∂^α C_r(z)‖₂ = 15.3704 ≤ 99` (max at α = (0,4), z = 0; bound 11·9 has margin 6.4×).
- Ten-vector moments: `E‖V‖ ≤ √tr Γ = 3.17533 ≤ 9` (Jensen); exact Isserlis
  `E‖V‖⁴ = (tr Γ)² + 2 tr(Γ²) = 142.2279 ≤ 3(tr Γ)² = 304.9846`.
- Exact integer/Fraction components re-verified: `99·10 = 990 < 1000`, `3·81² < 18⁴`,
  `10²³ + 1000(9+14) < 10²⁴ = B₄`, `(1+10²³+1000·20)⁴ = (2·10²³)⁴ = 16·10⁹² < 10⁹⁶ = B₃`,
  `‖j‖² < 11² + (2+δ)² + 2δ² < 144`, `‖(u_r,j)‖ < 14`, `‖β₀‖_F² < 4`, `‖β₀u_r‖ ≤ 5353/4000 < 7/5`.
- Q4 (§4) exact arithmetic re-verified: `(n+1)⁴ ≤ 24·C(n+4,4)` (n ≤ 200), `S = 48·65⁵`,
  `4S² = 12407264266410000000000 < 10²³` (matches the displayed integer exactly).

## Item 6 — Final arithmetic (§7, exact rational): PASS

All in `Fraction`/integers/powers of ten (no floats anywhere near tiny constants):
- `256·K·r_* = 32/625 = 0.0512 < 1` (K = 2·10²⁴, r_* = 10⁻²⁸ ≤ R = 1/1600).
- `16δ + 8Kr_* = 689/40000 = 1/64 + 1/625 < 3/64 < 1/16` (exact equalities).
- `260·10¹⁰ = 2.6e12 > 2⁴⁰ = 1099511627776`; hence `260·δ⁴ = 65/274877906944 = 2.3647e-10 > 10⁻¹⁰`.
- Coefficient ledger `65·16/4 = 260`; exponent chain `−1129 − 96 − 10 = −1235`; and the full
  exact-rational chain `260·2⁻⁴⁰·10⁻¹¹²⁹·10⁻⁹⁶ > 10⁻¹²³⁵ = c_*` (margin factor ≈ 2.36), with
  `c_*` stored as `{base:10, exponent:−1235}` — never materialized as a float.
- Symbolic identities (sympy, independent of numerics): `T_r(b,0,0,b−r³/6,0,0)ᵀ =
  (b−r³/12,−r²/4,0,0,0,1/3)ᵀ` exactly; `det T_r = −r⁻⁵`.

## Independent reproduction statement

The reviewer program was written solely from the derivation documents (candidate, endpoint
supplement, moment correction). Field covariances were built two independent ways —
oscillatory spectral sums over `(π/12)ℤ²` and Poisson-dual Gaussian image sums — agreeing to
25–40 digits on every load-bearing quantity (moments and their 1e-117-scale deviations,
endpoint and finite-r spectra, L² errors, conditional mean/covariance, density minimum).
Author-side diagnostics independently reproduced: the three MC-RETURN03 140-digit moment
decimals (36/36 digits) and Kimi's endpoint eigenvalue central value 0.126553449667289.
The author's companion checker was not executed or reused.

## Evidence vs proof labels

- **Computed exactly (certified truncation + certified linear algebra):** all numbers in
  Items 1–7 above at the stated rungs/points; Sylvester PD certificates for all four matrix
  interval claims at both rungs; exact rational/integer arithmetic of §4–§7; symbolic u_r
  and det T_r identities.
- **Evidence only (finite rungs/samples, as delegated):** the five L²-error rungs, the two
  covariance rungs, the eight representer z-samples, and the edge/grid density scans support
  but do not by themselves prove uniform claims on `[0, R]` or the whole torus. The uniform
  statements rest on the analytic arguments (integral-remainder kernel, variational Schur
  bound, covariance perturbation expansion, Markov/geometry) verified at derivation level by
  the lead — outside this report's numeric scope.
- **Convexity certificate:** for Item 4 the corner minimum is the exact global box minimum
  (convex quadratic maximized at vertices), so the density-floor comparison is not merely
  sampled evidence.

## Per-Q disposition (numeric scope)

| Q | Disposition | Basis |
|---|---|---|
| Q1 (moments, L² bound 11) | **PASS WITHIN SCOPE** | Item 3: all caps hold; tightest is a8 < 106 (true 105+1.8e-114); sup = 10.24695 < 11; tail 2.01e-185 certified |
| Q2 (six L² errors) | **PASS WITHIN SCOPE** | Items 1+7: all 30 coordinate checks + 20 remainder checks below bounds; tightest margin ratio 1.0735 (div-x/2 and deg-3/deg-2 x-remainders), driven by 11 vs √a8; cancellations confirmed structurally |
| Q3 (covariance interval, Schur) | **PASS WITHIN SCOPE** | Item 2: λ_min = 0.1265534 > 0.104 and Σ_r ∈ [1/6, 2] ⊂ [1/10, 9], Sylvester-certified at r = 1/1600 and 1/800; endpoint floor reproduced; true perturbation 1.4e-7 ≪ 32r |
| Q4 (Fourier C⁴ majorant) | **PASS WITHIN SCOPE** | Exact integer components re-verified (incl. the displayed 23-digit integer); analytic inequality chain per lead verdict (no numeric target delegated) |
| Q5 (representer, residual, B3/B4) | **PASS WITHIN SCOPE** | Item 5: max representer norm 15.37 ≤ 99; E‖V‖ ≤ 3.175 ≤ 9; E‖V‖⁴ = 142.23 ≤ 3(trΓ)²; B₃/B₄ cap arithmetic exact and correctly distinguished |
| Q6 (density floor, tiny constants) | **PASS WITHIN SCOPE** | Item 4: computed min 10⁻¹³·²²³ at corner (−11, 2+δ, −δ, −δ), convexity-certified; m_* = 10⁻¹¹²⁹ below it by 1115.8 orders; exact power-of-ten representation enforced (underflow mutation fails closed) |
| Q7 (final powers, linkage) | **PASS WITHIN SCOPE** | Item 6: every rational equality/inequality exact; c_* exponent chain −1129−96−10 = −1235 with factor 2.36 margin; topology/weight linkage itself is analytic (lead scope) |

No AMEND REQUIRED, no REFUTED, no CANNOT VERIFY items in the delegated numeric scope.

## Reviewer's adversarial notes (not defects)

1. The bounds `11r⁴/384`, `11r³/48`, and the div-x/2 row bound `11r²/48` are nearly sharp:
   their margin is exactly `11/√a8 ≈ 1.0735`. Any future "optimization" that reduced the
   derivative sup bound from 11 toward √106 ≈ 10.296 would break them (10.296 < √105·(1+ε)
   fails — the leading Taylor coefficient is exactly √a8 and a8 > 105).
2. λ_min(Γ₀) = 0.1265534496672895 exceeds Kimi's printed lower endpoint 0.1265534497 by
   only ~3e-11 — the supplement's rounding correction is justified; the proved floor remains
   the conservative 31/250.
3. The conditional-mean perturbation cap `‖μ_r − β₀u_r‖ ≤ 6/5` has ~15 orders of slack
   (true 1.6e-15 at R); the density exponent cap N = 1125 vs true quadform ≈ 56.0 at the minimizing corner; m_* is
   ~1116 orders below the true floor. All deliberate, all disclosed in the derivation.
4. Truncation |k| ≤ 30 was certified in-program (tail 2.01e-185); a second fully independent
   image-sum code path reproduced every reported number, so no single implementation artifact
   can account for agreement.

## Artifacts

- `qc_return03_numeric.py` — reviewer program (sha256 `f3622d3c…fc281d`)
- `OUTPUT_NORMAL.json` = `OUTPUT_OPTIMIZED.json` (byte-identical, sha256 `3311b40f…1cbdd1`), 335 checks
- `receipts/HASHES.json`, `receipts/ENVIRONMENT.json`, `receipts/MUTATION_TESTS.json`
- `mutation_tests/mut_1..5.py` — fail-closed demonstrations (all exit 1)
