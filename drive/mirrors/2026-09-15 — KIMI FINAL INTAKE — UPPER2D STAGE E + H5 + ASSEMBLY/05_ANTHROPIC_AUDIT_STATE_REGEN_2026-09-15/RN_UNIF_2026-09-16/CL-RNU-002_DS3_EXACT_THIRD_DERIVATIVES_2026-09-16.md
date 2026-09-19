# CL-RNU-002-v1.0 — RN-UNIF route D, step 2 DONE: exact third derivatives of κ_far (DS3), validated

AUTHOR: Claude (Anthropic) · CREATED: 2026-09-16 · CLASS: LANE PROGRESS (D3 / RN-UNIF) · STATUS: PROPOSED
AUTHORITY: none · CANONICAL IMPACT: NONE (the lemma is not closed; this is certified-numerics infrastructure)
RECEIPTS: `rnu_ds3.py` (the engine extension), `rnu_ds3_transcript.txt` (validation), `rnu_ds3_scan_transcript.txt`
  (evidence-grade derivative scan). Runs against `d3_rn_unif.py` unmodified (monkey-patched at import).

## What was built
`DS3` extends the engine's `DS` automatic-differentiation class with the four third-order partials
(t_xxx, t_xxy, t_xyy, t_yyy): product rule and Faà di Bruno to third order for +, −, ×, ÷, sqrt, exp, log, Φ.
Three hand-coded second-order pieces of the engine are replaced by generic DS3-arithmetic versions so the
whole of `kappa_far_ds` lifts to third order automatically:
- `DM.inv` → Gauss–Jordan on DS3 entries (bug found and fixed while validating: an entry with value exactly 0
  but nonzero derivatives must still be eliminated — at symmetric stations the skip corrupted h_yy by 4.5e-6);
- `DM.det_ds` → LU pivot product on DS3 entries;
- `bures_trace_ds` (tr A^{1/2}, eigen-perturbation to 2nd order) → Denman–Beavers iteration on DS3 matrices,
  converged to 1e-95 (checked independently against the engine's eigen version and an FD of tr√A: agreement
  to all displayed digits, including at the near-degenerate spectrum {0.9999992, 3.9992652, 4.0} at (5,0));
- kernel entries `de1`/`d_entry` → third-order kernel derivatives via the engine's own `kplane`.

## Validation (three stations; all at dps 100)
| station | DS3 vs engine DS (value, ∇, ∇²) | DS3 ∇³ vs FD of the engine's exact ∇² | cost |
|---|---|---|---|
| (5, 0) | 2.0e-87 | 2.4e-24 | 0.90 s |
| (4.8, 1.6) | 8.6e-88 | 9.3e-25 | 1.17 s |
| (5.1, −0.3) | 9.9e-87 | 3.5e-24 | 1.15 s |

## What it shows (evidence-grade scan, 40 stations on the delicate patch, `rnu_ds3_scan_transcript.txt`)
Exact |∇κ| ≤ 2.34, ‖∇²κ‖ ≤ 7.19, ‖∇³κ‖ ≤ 18.5, and FD-estimated ‖∇⁴κ‖ ≲ 32 — all attained at the boundary
axis point (5,0) and decaying fast (at d = 6: ‖∇³κ‖ ≤ 1.0; at d = 7: ≤ 0.01). Against the engine's crude
third-order envelope t₀ = 8.2e57 at the same point, the exact value is 1e56 smaller. A cell at the peak closes
under the cap 0.68 with exact derivatives to order 3 at hw ≈ 7e-4 for ANY valid fourth-order envelope up to
T₄ ≈ 1e8 (T₄·hw⁴/24 = 1e-6 at T₄ = 1e8), so the certification no longer depends on the envelope being tight —
only on it being valid.

## What remains for a CERTIFIED closure of Piece 1 (unchanged order; two items)
1. A valid uniform fourth-order envelope T₄(d₀) for κ_far on {|y| ≥ d₀}. Because the χ² piece is 1e19 loose in
   the unwhitened frame at first order (CL-RNU-001 §2), any crude envelope must be built in the whitened frame
   (χ² ↦ log q(A, m′), A = I − Σ′); the other pieces' crude scales (Wick moments, Bures, ratio pieces) may be taken
   from the engine's existing `_rel_der_scale`/`enorm_scales` machinery extended one order, since looseness is
   now harmless. Estimated: ~200 lines; validation target: T₄ ≥ 32 everywhere it is evaluated (the scan), never
   below the FD estimate.
2. The certification run: polar adaptive cover of {5 ≤ |y| ≤ 17}, θ-halved by the exact θ→−θ symmetry, cell
   bound κ(c) + |∇κ|hw + ½‖∇²κ‖hw² + ‖∇³κ‖hw³/6 + T₄hw⁴/24 ≤ 0.68 − 0.0002; ~5,000 fine cells near the axis at
   0.9 s each plus a few hundred coarse cells; ≈ 1.5 h wall clock. Then both-mode transcript, MUT-RN-1..5,
   FREEZE with a rule-id, and D1 consumes the rung part.

## Falsification
Any station where DS3 (value, ∇, ∇²) disagrees with the engine's `DS` beyond 1e-60, or where DS3's ∇³
disagrees with an FD of the engine's exact ∇² beyond the FD truncation (h² · ‖∇⁵κ‖ ≈ 1e-22 at h = 1e-12).
