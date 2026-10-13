**Object:** `CAP-FIRST-EXIT-LEAN-20261002` v2.1. **Stacked on Math-#260** (v2.0, head `ea40607`), which is stacked on #259, #258, #257, #255 and #253.
- **Author:** Anthropic / Claude, session `017Mi3hx…`.
- **Scientific effect NONE.** No register, catalog, STATUS, PROOF_INDEX, GRAPH or `formal/` change.
- Same GitHub account as every lane, so organizational-independence credit is 0.
- **The author will not merge.** Merge order: #253, #255, #257, #258, #259, #260, then this PR.

## What changes

v2.0's `ridge_capHyp_C4` still took two kinds of input that SP does not state as such:
- six derivative-bound hypotheses besides the block bound `hT`: `hHxb`, `hHrb` (L3), `hwcv`, `hwcc` (L5–L6) and `hG0cv`, `hG0cc` (L1–L2);
- every bound `m` on a collar `[−2r − ε, 2r + ε] × B̄(0, 2r)`, where SP bounds them on `D = [−2r, 2r] × B̄(0, 2r)`. This is the first I3 item, noted in K2's finding 2 on #258.

v2.1 removes both. It adds 8 declarations:

| Declaration | Statement | SP |
|---|---|---|
| `hessian_x_bound` | `∂_x D_y² f (x, 0)[v, w] = T3(x, 0)[(0, v), (0, w), (1, 0)]`, by uniqueness of derivatives within an interval, so `‖∂_x D_y² f‖ ≤ m` from the block bound | L3's input (`hHxb`) |
| `hessian_r_bound` | `∂_ρ D_y² f (x, sy)[v, w] = T3(x, sy)[(0, v), (0, w), (0, y)]` along the radius, within `[0, 1]`, so `‖∂_ρ D_y² f‖ ≤ m‖y‖` | L3's input (`hHrb`) |
| `axis_convexConcave` | `|∂_x² (Df(·, 0) u)| ≤ B` on `[a, c]` ⇒ `Df(·, 0) u ± (B/2)t²` convex / concave. With `u = (0, v)`, `B = m‖v‖` this gives `hwcv`, `hwcc`; with `u = (1, 0)`, `B = m` (SP: `|f_xxx| ≤ M₃`) it gives `hG0cv`, `hG0cc` | L1–L2, L5–L6 inputs |
| `block_nonneg`, `block_bound_collar` | **the continuity step.** The block bound `m` on `D`, together with some bound `K ≥ 0` on the `x`-derivative `T4` of `T3` in block form on a collar of width `ε₀`, gives the block bound `m + Kε` on the collar of width `ε ≤ ε₀`. The proof applies the mean value inequality from the nearest point of `D` | I3 |
| `ridge_capHyp_C5` | `ridge_capHyp_C4` with those six hypotheses derived, plus `|f_xxx| ≤ m` on the pin segment (`hTe`; the block bound `hT` excludes `f_xxx`, as K3's finding 8 notes) | §2, L1–L20 |
| `ridge_capHyp_D` | **the cap from SP's bounds on `D`.** On `D`: `m`, `|f_xxx| ≤ m` on the pin segment, `n`, (H1), (H2). On a collar of width `ε₀`: only qualitative data (derivatives, the strict derivative of `∂_y f`, some `K`). Because (H1) is strict, it picks `ε ∈ (0, ε₀]` with `8r(m + Kε)² < λ` and concludes `ridge_capHyp_C4`'s statement on `(−2r − ε, 2r + ε)` | §2 (H1), (H2), `M₃` and `M₄` on `D`, `f ∈ C⁴` near `D` |
| `h1Toy_D` | `x³/3 − x/4 − 20y²` meets every hypothesis (`T4 = 0`, `K = 0`, `ε₀ = 1/2`) | non-vacuity |

**What remains (I3).**
- That SP's `f ∈ C⁴` near `D` supplies the qualitative collar data and some finite `K`, by compactness of a closed collar. Lean takes them as hypotheses.
- I1 (coordinates), the persistence-module bookkeeping, L14 and L15 (not used by the cap theorem), L23 and the probabilistic statements, as before.

**Unchanged.**
- The 113 v2.0 declarations are byte-identical. The only removed line is in the header comment.
- The toolchain, pins and workflow are unchanged.
- The branch also carries, by merge commit `3a5688a`, the K1–K4 wording AMENDs applied on #257–#260.

## Validation (local)

At `a0cce67` (the Lean commit) and `3a5688a` (docs merge, Lean byte-identical):
- `gate.py` in source mode: PASS, 121 targets. `MANIFEST.json` SHA-256 at `3a5688a`: `fb74ff5f86ee019e83541199148de8628ca42280be68a918644e1d02ed58f4c8`.
- Unit tests: OK in `-B -S` and in `-B -O -S`.
- `gate.py --execute` at `a0cce67`: PASS.
  - It runs a fresh build and a `leanchecker` replay.
  - It audits the axioms of all 121 declarations exactly: each uses only `propext`, `Classical.choice` and `Quot.sound`.
  - All three negative controls are REJECTED.
- The hosted checks run on this PR.

## Review requested

A nonauthor, non-Claude delta alignment read under ALIGNMENT §5 should cover:
- the six v2.1 rows of §2 (`hessian_x_bound` to `ridge_capHyp_D`, seven theorems);
- the `h1Toy_D` sentence of §3;
- the revised SP row and I3.

Points worth checking:
- the block form of `hT4` against SP's `f ∈ C⁴` near `D` (a full bound `‖D⁴f‖ ≤ K'` on a compact collar implies it with `K = K'`, since the 7 mixed terms of a multilinear form sum to the block weight);
- that `hTe`, `|f_xxx| ≤ m` on the pin segment, is what v2.0's `hG0cv`/`hG0cc` encoded, and is implied by SP's `m = M₃`;
- that the derivatives within `[−2r − ε, 2r + ε]` and `[0, 1]` in `hessian_x_bound` and `hessian_r_bound` cover the collar endpoints and the sphere `‖y‖ = 2r`.

This does not close Math-#193.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
