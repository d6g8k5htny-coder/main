# CL-RNU-001-v1.0 — D3-LEMMA-RN-UNIF(r = 0.05): ENGINE LOCATED, BLOCKER ROOT-CAUSED, FIX VALIDATED, CLOSURE COSTED

AUTHOR: Claude (Anthropic) · CREATED: 2026-09-16 · CLASS: LANE STATUS + DIAGNOSIS (foundations, D3 lane)
STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE — the lemma is NOT closed by this document.
SUPERSEDES: CL-OBL-001 §2 and Appendix A (my six-step work order), which assumed no engine existed.
RECEIPTS: `rnu_run1.txt` (engine run to its current end), `rnu_diag.py`/`rnu_white.py`/`rnu_env.py`/`rnu_land.py`/
  `rnu_fdcheck.py`/`rnu_meanfix.py`/`rnu_chi2_white_v2.py`/`rnu_dcheck.py` and their transcripts (this folder).

## 1. What exists that no state document mentions

`UPPER2D/D3_percolation/d3_rn_unif.py` — 2,240 lines, Kimi, last write 2026-09-15 07:43 (a `_dbg` twin at 07:24
differs only by memoization caches). Not in RETURN_06, not in any addendum, no transcript, no FREEZE, no
manifest row. It is the rung-part closure engine for D3-LEMMA-RN-UNIF and it was mid-build when Kimi's
tokens ran out. Its docstring names the two pieces exactly as the lemma is frozen:

- **Piece 1 (rigidity-decoupling):** far-zone uniformity of τ(y) = tr(Δᵀ Σ_pair⁻¹ Δ) and the κ pieces, via the
  rigid residual forms R_k = v_k·H_pair − a_k·C6 (exact 6-pin regression), their certified monomial moments
  with definite parity, and θ-free moment-series envelopes `env_form(k, γ, d, q)` bounding |∂^q F_{R_k,γ}(y)|
  uniformly on {|y| ≥ d}.
- **Piece 2 (certified Riemann sum):** the annulus crude spine ρ_spine = p_grad · window-cap / Z_lo on an
  adaptive polar grid with per-cell error bounds.

**What runs (verified here, 45 s to its current end):** hash pins (d3_perc, d3_amend, cov_exact, pin_transform,
reg_lemmas, H3_RUNG_FLOOR) verify; closed-form kernel validated against `cov_exact` to 2.3e-100; SPAIR0
spectrum {2.6e-10, 5.2e-7, 2.1e-6 | 2.5e-3, 2.5e-3, 4.0}, rigid/non-rigid gap ×1.2e3; residual-form moments and
parity certificates (vanishing orders < 1e-80); fast two-stage station reproduces the frozen station to 5e-96;
normalized pin frame eigenfloor 0.1293, ‖Gram(U)⁻¹‖ ≤ 7.73; **fast assembly at (5,0) reproduces the frozen v2
values κ_cross = 0.6499017772, κ_far = 0.6772849052**; an exact-derivative engine (`DS`/`DM` automatic
differentiation to second order) gives κ_far = 0.677284905216, |∇κ| = 2.34273, ‖∇²κ‖ = 7.19436 at (5,0).

**Where it stops:** after four probe evaluations of `kp1_point`, immediately before
`run_certification(kap_fn_p1, KAP_BUDGET, …)` — the adaptive polar certifier is defined and never invoked.

## 2. Why it stops there — root cause, measured

`kp1_point` returns (κ₀, g₀, h₀, t₀) = value plus certified gradient / Hessian / third-derivative bounds, and the
certifier closes a cell iff κ₀ + g₀·hw + 3h₀hw²/2 + 9t₀hw³/6 ≤ 0.68 − 0.0002. At the worst point:

    probe y=(5,0): κ₀ = 0.67728491   g₀ = 1.04e17   h₀ = 1.69e37   t₀ = 8.22e57

against exact |∇κ| = 2.34, ‖∇²κ‖ = 7.19. **The variation bounds are ~1e17 too loose; no cell can close at any
half-width above the 4e-4 refinement floor, so the run would fail closed at the first cell.** That is why the
call is absent.

Decomposition of g₀ (`rnu_diag.py`): τ pieces honest (τ₀ = 3.53e-4, ∂τ ≤ 2.4e-3, ∂²τ ≤ 0.29); ratio pieces
honest (∂R_pg ≤ 7e-6, ∂R_wm ≤ 1.2e-3); **the χ² piece is the whole failure**: `chi2_grad_bound` = 1.57e14 with
χ² = 1.94e-6, feeding ∂κ_pair = κ_pair·∂χ²/(2χ²) = 1.0e17. Mechanism: the bound sums absolute values of four
terms (∂ log det Σ_pair, ∂ log det M, quadratic-form terms) that individually carry ‖Σ_pair⁻¹‖ ≈ 3.8e9 and
cancel exactly in the true derivative. The crude Hessian/third scales of the Wick-moment pieces
(`_rel_der_scale`: h = 5.6e4, t = 2.7e7 at d = 5; growing with (2d+10)^k to t = 5.6e102 at d = 11.6) are the
second-order problem behind it.

## 3. The fix, validated numerically (`rnu_white.py`)

χ² is the χ²-divergence between two 6-dim Gaussians, N(μ9p(y), Σ_pair(y)) and N(μ6, Σ6), and is invariant
under a common affine map. Whiten by the **y-free** Σ6 = SPAIR0 (verified `kit.Hcov == SPAIR0` exactly):
W = Λ₀^{−1/2} V₀ᵀ, ‖W Σ6 Wᵀ − I‖ = 1.1e-88. Then

    χ²(y) computed in the whitened frame = 1.94428568743e-6 = engine value   (rel. diff 1.6e-84)

and in that frame at (5,0): ‖I − Σ′‖_F = 2.3e-5, ‖m′‖ = 1.4e-3 — the rigid directions are gone because the
pins explain them (the engine's own thesis for τ, now applied to χ²). **True |∇χ²| at (5,0), central FD at
dps 100 with h = 1e-20: 1.563e-5** (vs the engine's bound 1.57e14 — 1e19 slack). Hence true |∇κ_pair| =
0.0101, consistent with the exact D-engine total 2.34.

The whitened residual-form envelopes are honest in magnitude (`rnu_env.py`, d = 5): values 0.002–0.07,
gradients 0.02–0.8, Hessians 0.2–9, thirds 2–100 (per (k, γ), after the 1/√λ_k scaling); RSS over all 18
(k, γ) pairs: at d ≥ 6: {0.0015, 0.019, 0.25, 3.4} for q = 0..3; at d ≥ 8: {7.8e-9, 1.2e-7, 2.1e-6, 3.5e-5};
at d ≥ 10: ~1e-15..1e-12. They are loose against the exact values (×3–50 on values, ×10–900 on gradients,
because parity zeros are invisible to a termwise bound) but they are the right *size* to close.

## 3a. Engine defect found while validating the fix (E-RNU-1) — and the exact engine confirmed sound

Chain-ruling the whitened log q with the engine's "exact" inputs gave |∇χ²| = 0.01245 against FD 1.563e-5.
FD-checking the inputs (`rnu_fdcheck.py`): `cov9_grad_exact` matches FD to 1.1e-43 (exact);
**`mean_grad_exact` does not** — pair-mean components off by up to 185× (idx 0: FD −2.37e-6 vs −4.39e-4),
nonzero where symmetry forces zero (idx 1, 4, 7). Cause, from the code: dTY6/dy omits −TC·G6inv·dYCᵀ in every
row and −dTC·G6inv·YCᵀ in the Y-target rows. A corrected `mean_grad_fixed` (`rnu_meanfix.py`) matches FD to
3.4e-42 at three stations; with it the whitened exact gradient reproduces FD to **1e-39 relative** at four
stations (`rnu_chi2_white_v2.py`): |∇χ²|(5,0) = 1.5630931e-5, exact |∇κ_pair| = 0.0101202.

Consequence: every `kp1_point` piece that consumed `dmu` (gCm, gky, gkp, and the ratio bounds) was silently
wrong as well as loose. None of this touched any frozen certificate — `mean_grad_exact` is used only inside
the unrun Piece-1 net machinery — but it would have poisoned the certification had the χ² blow-up not stopped
the run first.

**The exact-derivative engine is independent of the bug and correct:** `kappa_far_ds` gradient and Hessian
match central FD (h = 1e-15, dps 100) to 1.3e-26 at (5,0), 3.3e-26 at (4.8,1.6), 3.1e-26 at (5.1,−0.3)
(`rnu_dcheck.py`). It is the right cell-center evaluator.

## 4. The landscape, and why the lemma is certifiable at the rung

κ_far(d, θ) (`rnu_land.py`), θ measured from the pair axis:

| d \ θ | 0° | 4° | 8° | 15° | 30° | 45° | 90° |
|---|---|---|---|---|---|---|---|
| 5.00 | **0.677** | 0.669 | 0.644 | 0.570 | 0.364 | 0.209 | 0.050 |
| 5.05 | 0.569 | 0.562 | 0.541 | 0.478 | 0.304 | 0.173 | 0.041 |
| 5.10 | 0.476 | 0.470 | 0.452 | 0.400 | 0.254 | 0.143 | 0.033 |
| 5.20 | 0.330 | 0.326 | 0.314 | 0.277 | 0.174 | 0.097 | 0.021 |
| 5.50 | 0.101 | 0.100 | 0.096 | 0.085 | 0.053 | 0.028 | 0.005 |
| 6.00 | 0.011 | 0.011 | 0.010 | 0.009 | 0.006 | 0.003 | 4e-4 |

The sup over the zone is attained on its boundary at (5,0) — the frozen probe point — and falls at ≈ −2.2 per
unit d radially and ≈ −0.006 per degree angularly. θ → −θ is an exact symmetry (checked to 6 digits);
θ → 180° − θ is **not** (0.6264 vs 0.6122) — the full zone must be covered, halved by the one true symmetry.
Beyond d = 6 the budget slack is > 0.66 and cells can be coarse; the delicate patch {κ > 0.6} is
d ∈ [5, 5.035] × |θ| ≤ 12° — area ≈ 0.04.

**Cost model:** `kappa_far_ds` (exact value/gradient/Hessian) = **0.51 s per point at dps = 100**; dps ≤ 30
trips a fail-closed eigsy residual check (4.4e-25 vs 1e-25), so dps stays ≥ ~40. With exact center data and a
valid third-derivative envelope T, a cell closes iff |∇κ|hw + ½‖∇²κ‖hw² + T·hw³/6 ≤ cap − κ(c) − 0.0002.

| cap consumed by D1 | margin at (5,0) | Ĩ_hi coefficient (exact) | hw at the peak (T ≈ 1e7 / T ≈ 1e2) | est. points in the delicate patch |
|---|---|---|---|---|
| 0.68 (current) | 0.00272 | 650.1826 | 7e-4 / 1.1e-3 | ~5,000 / ~2,000 (≈ 40 / 17 min) |
| 0.69 | 0.01272 | 650.2079 | 1.8e-3 / 5e-3 | ~600 / ~100 |
| 0.70 | 0.02272 | 650.2332 | 2.3e-3 / 9e-3 | ~300 / ~40 |

The lever is real and nearly free: consuming κ_far ≤ 0.69 at v2.3 moves the headline from 650.1827 to
650.2079 (0.004%, inside a 507× honest width) and cuts the certification cost ~8×. I recommend it, but the
lemma is certifiable at 0.68 as well — the patch is small because the decay is fast.

## 5. What remains to close Piece 1 (ordered; all executable without Kimi)

1. **Whitened χ² derivative bounds** — the exact first derivative is DONE (`rnu_chi2_white_v2.py`, FD-validated
   to 1e-39 with `mean_grad_fixed`). Remaining: over-cell bounds at orders 2–4 from the whitened closed form
   log q(A, m′), A = I − Σ′, with `env_form` (q ≤ 4) on the whitened residual covariances — loose but valid, and
   O(1)–O(100) in size where the old bound was 1e14.
   **Recommended certification form (route D):** exact derivatives to order 3 at the cell centre (extend `DS` to
   a third-order `DS3`; ~40 lines of mechanical arithmetic rules) plus a crude-but-valid fourth-order envelope
   T₄(d₀); a cell then closes iff |∇κ|hw + ½‖∇²κ‖hw² + ‖∇³κ‖hw³/6 + T₄hw⁴/24 ≤ cap − κ(c) − 0.0002. At hw = 7e-4
   even T₄ ≈ 1e12 costs only ~1e-2 — the crude envelopes become harmless one order higher; at T₄ ≈ 1e9 they
   cost 1e-5.
2. **Uniform third-derivative envelope T(d₀)** for κ_far over {|y| ≥ d₀} — the one genuinely new construction.
   Route A: formal chain rule through the composition κ_far = R_pg R_wm((1+κ_pair)(1+κ_y)+κ_cross) − 1 with
   `env_form` (q ≤ 3) on the whitened residual covariances and the y-jet blocks; the engine's existing t-pieces
   are this route with the χ² piece broken and the Wick scales crude — fixing (1) and replacing `_rel_der_scale`
   with exact-Hessian-at-center + `env_form`-based Lipschitz correction is the path of least construction.
   Route B (no third derivatives): run the `DS` gradient engine on `mpmath.iv` intervals over each cell; the
   whitened formulation removes the only ill-conditioned inverse; the 3×3 y-jet and Bures pieces need
   interval-safe replacements (Bures trace via a certified eigenvalue enclosure). Route B is cleaner but touches
   more of the engine.
3. **Certification run** with the fixed `kp1_point` (or the interval certifier), polar cover d ∈ [5, 17], θ-halved.
   Expected: a few hundred coarse cells for d ≥ 6, ~10³–10⁴ fine cells in the delicate patch; wall clock under
   two hours at 0.5 s/point.
4. **Freeze**: both-mode transcript, MUT-RN-1..5, FREEZE record with a rule-id per CL-REG-001; then D1 consumes
   "D3-LEMMA-RN-UNIF(r = 0.05) Piece 1 CERTIFIED" and the rung part of the named hypothesis (1)(b) closes.

Piece 2 (the certified Riemann sum for the annulus spine) has its fast two-stage Schur evaluator validated to
1e-90 in the engine already; its driver is likewise unwritten. It is the smaller of the two.

## 6. Register action (proposed)

D3-LEMMA-RN-UNIF: OPEN, "no lane executing" → **OPEN, lane engine EXISTS (Kimi, 09-15 07:43, unfrozen),
blocker root-caused (χ² gradient bound; crude Wick scales), fix validated, closure costed (< 2 h compute at
the rung once steps 1–2 land).** Owner for steps 1–2: any family with the tree; I am proceeding.

## 7. Falsification

Any of: the whitened χ² failing to reproduce the engine's χ² at a station (identity is exact, so a mismatch
indicates `kit.Hcov ≠ SPAIR0` there); a certified cell bound below a sampled κ_far value inside the cell; a
zone point with κ_far > 0.68 (which would falsify the consumed cap itself, not just the lemma).
