# CL-RNU-003-v1.0 — RN-UNIF(r = 0.05): PIECE 1 CERTIFICATION RUNNING WITH TIGHT BOUNDS; PIECE 2 CHARACTERISED (near-pin crude bounds fail; residual-coordinate route named)

AUTHOR: Claude (Anthropic) · CREATED: 2026-09-17 · CLASS: LANE PROGRESS (D3 / RN-UNIF) · STATUS: PROPOSED
AUTHORITY: none · CANONICAL IMPACT: NONE yet (Piece 1 closes when the run completes and is frozen; Piece 2 open)
RECEIPTS (bundle CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip, folder RN_UNIF_2026-09-16/): `rnu_t4.py` (Taylor-model
arithmetic, whitened + normalized-frame assembly), `rnu_certify.py` (adaptive polar certifier, checkpointed),
`certify_cells.jsonl` + `certify_queue.json` (run state), `rnu_spine.py` (Piece-2 integrand, validated), traces
`rnu_c4trace`, `rnu_rwm_trace`, `rnu_decomp`, `rnu_chi2_closed`, `rnu_chi2_trace`, `rnu_spine_feas`.

## 1. Piece 1 — the run

Target: sup over the far zone {y ∈ T², |y| ≥ 5} of κ_far(y) ≤ 0.68 − 0.0002, at the rung r = 0.05, by an
adaptive polar cover of 5 ≤ d ≤ 17 (≥ 12√2), θ ∈ [0, π] (reflection across the pair axis is an exact symmetry:
pins at (±0.025, 0), covariant constraint/target sets, isotropic kernel — verified to 100 digits).

Per cell (centre c, polar box with radial half-extent a, tangential b, circumradius ρ):
    sup_cell κ ≤ R_pg,hi · R_wm,hi · ((1 + κ_pair,hi)(1 + κ_y,hi) + κ_cross,hi) − 1,
each factor's hi = exact value at c + exact directional Taylor terms to order 3 on the rotated box +
(2/3)·C4·ρ⁴ with C4 a valid crude sup of the fourth derivative over the box (Taylor-model arithmetic, §2);
the sqrt-singular pieces (χ², w2, τ) enter through the highs of their non-singular ingredients.
Fail-closed: any arithmetic check inside a cell (interval through 0, non-PD, divergence) refines the cell;
a cell below hw = 1.5e-4 aborts the run. Checkpointed per closed cell; budgeted chunks because background
processes die at tool-call boundaries here (the same operational failure W3 suffered).

**State at this writing: 1,170 cells closed, 561 base cells pending, max certified bound 0.679737** (attained
by a boundary cell closed under the earlier, looser bounds; all cells closed since the tightening are below
0.6788). Coverage so far: d ∈ [5, 5.09], θ ∈ [0, 8.1°] — the peak region. Cells closed after the tightening
have hw between 4e-3 and 1.7e-2 (four to sixteen times the earlier sizes).

## 2. What had to be fixed to make the bounds tight — three parity/cancellation failures, all measured

The crude fourth-order envelope is only useful if it does not lose cancellations that the exact quantities
depend on. Three did, and each was traced to a specific line:

| piece | crude C4 before | after | fix (exact identity on the exact part; only the crude part changes) |
|---|---|---|---|
| all pin conditioning | (G6⁻¹-conditioned) | O(1)-conditioned | `dstation_norm`: X G6⁻¹ Yᵀ = (X Tᵀ) G_N⁻¹ (Y Tᵀ)ᵀ in the H2 normalized pin frame (eigenfloor 0.1293, ‖G_N⁻¹‖ ≤ 7.73); R_pg C4 535 → 1.2, w2 9.4e4 → 4e2 |
| R_wm (window-mass ratio) | 3.2e7 | 163 | wm = Φ(β) − Φ(α), β − α = ℓ/σ_t = 2e-5, wm/wm0 ≈ 1: bound via wm = ℓ·(1/σ_t)·∫₀¹ φ(α + sℓ/σ_t) ds (product form); local sups of |φ⁽ᵐ⁾| |
| χ² | 5.5e4 | 0.14 | closed form of the whitened log-quotient, **logq = −½ log det(I − A²) + mᵀ(I + A)⁻¹m** with A = I − Σ′ (verified vs the engine's formula to 1e-101; χ² ≈ ‖m′‖² to 99.99%); m′ = F_w,jet·(YY6⁻¹ rv) built from the parity-aware residual-form covariances (not W times raw rows); C4 of logq from ½tr A² + mᵀm − mᵀAm plus generating-function majorants for the tails |

Effect on the boundary cell d ∈ [5, 5.002], |θ| ≤ 0.05°: bound 0.68132 → **0.67736** against the true sup
0.67728 (7.4e-5 excess, all of it the honest radial Taylor term). The whitened, normalized-frame mirror still
reproduces the engine's value, gradient, Hessian and third derivatives to 8.6e-82 at every station tested.

What still limits cell size: the moment pieces C_mom and Ey (crude C4 ≈ 1e4–3e4 through the degree-8 Wick
recursion), which cap ρ near 0.01 in the inner shell. Throughput only.

## 3. Piece 2 — the annulus crude-spine integral: integrand certified, domain not yet

The frozen I_ann = Num_ann/Z_lo consumed Num_ann = 1.714838453e-5, a polar-**trapezoid** assembly of
exact-kernel stations on the D4 net (13 radii × 5 angles): an evidence assembly, not an upper bound. The
lemma's Piece 2 is the certified interval-box Riemann upper bound Σ area(box)·sup_box(p_grad·I_cap).

Built (`rnu_spine.py`): the integrand p_grad(y)·wm(y)·sup_{v∈window} g_env(y, v) as Taylor-model objects,
with the window sup of g_env(v) = (E dM⁴ E dS⁴)^{1/4}√(E dy⁴) taken **exactly**: each moment is a polynomial of
degree ≤ 8 in v (the 9-pin mean is affine in v), recovered from 9 evaluations by an exact Vandermonde solve
and bounded on the window by Σ|c_k|(ℓ/2)^k — replacing the frozen cap's displayed "20 × spread" margin by a
proof. **Validation: the exact part reproduces the frozen D4 table at all 65 stations to 4.7e-5 relative
(display precision).**

**Finding (kills the direct route inside d < 3):** the crude Taylor-model bounds are unusable near the
pins. Measured ratio bound/centre at θ = 45°:

| d | hw = 0.02 | 0.01 | 0.005 |
|---|---|---|---|
| 1.0 | fail | fail | 2.8e19 |
| 2.0 | fail | 1.6e3 | 9.6 |
| 3.0 | 543 | 4.8 | 1.21 |
| 4.0 | 5.7 | 1.23 | 1.01 |

The mechanism is the near-zone analogue of the far-zone rigidity problem: within ~2 of the pins the field is
nearly determined by the six pins, so every conditional variance is a tiny difference of O(1) terms, and the
crude rules cannot see the cancellation. The far zone was fixed by the residual forms + moment-series
envelopes; the near zone needs its own decoupling.

**Route (named, not built):** Taylor-residual coordinates via the integral-remainder representation of the
field about the pins — f(y) − f(S) − (y−S)·∇f(S) = ∫₀¹(1−t)(y−S)ᵀH(S + t(y−S))(y−S) dt — so that
Var(f(y) | jets at S, M) and the cross-covariances are explicit double integrals of kernel fourth derivatives
along segments: manifestly O(|y−S|⁴) sums of small terms, which crude rules bound at the right magnitude. The
segment integrals of Gaussian×polynomial have erf closed forms. Alternative route: extend the C1/H5 chart's
machine cover to an absolute radius d₀ ≈ 3 (the OBL-H5-REMOTE-THRESHOLD option already on the register) and
certify only d ∈ [d₀, 5] here, where the Taylor-model bounds are within 1.2× at hw = 0.01.

Contribution at stake: from the frozen table, the ring 0.1 ≤ d ≤ 3 carries roughly 30% of Num_ann; the
certified value will exceed the trapezoid evidence value by the sup/mean looseness (target ≤ 1–2%), so
I_ann/r³ should land near 18 rather than 17.68, B_remote near 22.3, Ĩ_hi's coefficient near 650.5 — the
constant moves at the 0.05% level and the named hypothesis (1)(b) closes at the rung.

## 4. Register (proposed)

D3-LEMMA-RN-UNIF(r = 0.05): Piece 1 — certification RUNNING with bounds tight to 1e-4 at the peak
(1,170/≈2,000+ cells); Piece 2 — integrand certified and validated, domain certification blocked inside
d ≈ 3 by near-pin crude-bound failure; two routes named. Nothing here changes a frozen carrier.
