**Object:** `CAP-FIRST-EXIT-LEAN-20261002` v1.5, additive to the merged v1.4 (Math-#246, main `0fda855`). **Author:** Anthropic / Claude, session `017Mi3hx…`. **Scientific effect NONE.** No register, catalog, STATUS, PROOF_INDEX, GRAPH or `formal/` change. Same GitHub account as every lane, so organizational-independence credit is 0. **The author will not merge.** Pickup: Math-#193 [5972637588](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-5972637588).

## What changes

In v1.4, `ridge_capHyp` took four things as bare inputs: the transverse maximum (`hT`), its uniqueness (`hTs`), the curved side (`hside`) and the ridge equation `g' = F`. SP derives these deterministically from L4 and L5. v1.5 proves those derivations, and L1, L2 and L5, in a new `Slice` section and two toy sections: 21 declarations in all. Ten were added in `83f7d87` and `a932946`, before any review pickup. **The head is now frozen for review at `a932946`.**

| Declaration | Statement | SP |
|---|---|---|
| `strongConcave_le_of_hasFDerivAt` | `StrongConcaveOn K δ φ`, `p ∈ K` and `HasFDerivAt φ L p` give `φ y ≤ φ p + L(y − p) − (δ/2)‖y − p‖²` on `K` | the engine of L7, L8 and L18 |
| `slice_le` | at a critical point, `φ y ≤ φ p − (δ/2)‖y − p‖²` | L7, L16, L19 |
| `slice_ridge_bound` | `δ‖h(x)‖ ≤ ‖w(x)‖` | L8 |
| `slice_curved` | on `‖y‖ = R`, the slice is `< s` once `‖w‖ ≤ ω ≤ Rδ`, `g ≤ b` and `2δ(b − s) < (Rδ − ω)²` | L18 |
| `slice_hasFDerivAt`, `ridge_hasDerivAt`, `ridge_inputs_of_joint` | if `∂_y f = 0` along the ridge, then `g' = ∂_x f(x, h(x))`; joint derivatives and a differentiable `h` supply `hh`, `hg` and the slice critical points | L7 (ridge equation), L11 |
| `ridge_capHyp_of_slices` | (C1)–(C3) and (C2⁺) on the cylinder, with `hT`, `hTs`, `hside` and `hM` replaced by L4 (`hconc`), the critical points `h(x) ∈ B̄(0, 2r)`, L5 (`‖w‖ ≤ ω`), `ω ≤ 2rδ` and `2δ(b − s) < (2rδ − ω)²` | L4, L5, L7, L8, L16–L19 |
| `convexOn_add_node`, `two_node_bound`, `slice_gradient_bound` | if `u` vanishes at `a < c` and `|u''| ≤ m` (weak form: `u ± (m/2)t²` convex or concave), then `|u(x)| ≤ (m/2)|(x − a)(x − c)|`, inside and outside `[a, c]`. For the slice derivatives this gives `‖w(x)‖ ≤ (m/2)|x² − r²/4| ≤ 2mr²` | L5 |
| `ridge_capHyp_sp` | the cap from SP's normalized hypotheses: `m ≥ 2`, `δ > rm(8m − 5)` with `hconc`, the weak third-derivative bound on `w`, `h(∓r/2) = 0`, L11, L12 and gap `≤ r³/6`. `ω`, `hωr` and `hcurv` are derived | L2, L4, L5, L7, L8, L11, L12, L16–L20 |
| `spToyF`, `spToy_capHyp` | `x³/3 − x/4 − 12y²` (`m = 2`, `δ = 24 > 22`, gap `r³/6`) meets every hypothesis of `ridge_capHyp_sp` | non-vacuity at SP's normalization |
| `hermite_gap_le`, `hermite_m_ge_two` | if `g' = G` vanishes at `a < c` and `|G''| ≤ m` (weak form), then `g(a) − g(c) ≤ m(c − a)³/12`. At the normalized gap `r³/6`, this forces `m ≥ 2` | L1, L2 |
| `ridge_capHyp_normalized`, `spToy_normalized` | `ridge_capHyp_sp` with `m ≥ 2` derived from the critical pins along `y = 0`, the third-derivative bound and the gap exactly `r³/6`; `spToyF` meets every hypothesis | §2 normalization, L1, L2 |
| `sp_curved_side_constants` | SP's constants (`m ≥ 2`, `δ > rm(8m − 5)`, `ω ≤ 2mr²`, gap `r³/6`) meet both conditions | L2, L4, L5, L18 |
| `ridgeToy_joint`, `ridgeToy_slices` | the ridge toy (`δ = 2`, `ω = 0`) meets every input of the slice route, which re-derives (C1)–(C3) and (C2⁺) | non-vacuity |

**I3 after v1.5.** The interface narrows to:
- L3, L4 and L6: the increments, the transverse Hessian bound that produces `δ`, and the `w'` bound;
- the existence and implicit-function regularity of the transverse maximizer `h`;
- L9–L12: the `‖h'‖` and `F''` bounds.

**Unchanged.**
- The 53 v1.4 declarations are byte-identical. The only removed lines in `CapFirstExit.lean` are three lines of the header comment.
- The toolchain, Mathlib pin, dependency revisions, pins on main and the workflow are all unchanged.

## Validation (local, at `a932946`)

- `gate.py` (source mode): PASS, 74 targets. `MANIFEST.json` SHA-256 `2cae887eafacd30a0d027e5fd0f4de4e641d4ce8c97eeef4253be7e8e712584e`.
- Unit tests: OK in `-B -S` and in `-B -O -S`.
- `gate.py --execute`: PASS.
  - Exact Lean `v4.34.1`, with clean pinned dependencies.
  - Fresh build and `leanchecker` replay.
  - Exact axiom audit of all 74 declarations: each uses only `propext`, `Classical.choice` and `Quot.sound`.
  - The three negative controls (`sorry`, a custom axiom, `native_decide`) are all REJECTED.
- Hosted checks: 4/4 green at `9e4ec3f` (`kernel-evidence`, `formal / formal-evidence`, `downstream-replay`, `math-downstream-gates`). 4/4 green again at `83f7d87`, and 4/4 green at the frozen head `a932946`. Codex reviewed `9e4ec3f` with no findings.

## Review requested

This needs a nonauthor, non-Claude **delta alignment review** under the contract in ALIGNMENT §5.
- **Rows to assess:** the fourteen new rows of ALIGNMENT §2, the v1.5 toy paragraphs in §3 and the revised I3 in §4.
- **v1.4 rows:** the v1.4 review [5396110365](https://github.com/d6g8k5htny-coder/Math-/pull/246#pullrequestreview-5396110365) still covers them once the reviewer confirms by diff that they are unchanged.
- **Points worth checking:**
  - Mathlib's `StrongConcaveOn K δ` (modulus `δ/2 · r²`) against SP's `D_y² f ≤ −δ I`;
  - the operator norm of `w x` against SP's `‖∇_y f(x, 0)‖` (they agree for Euclidean `E`);
  - `hcurv` against SP's L18 comparison `(δ/2)(2r − ‖h‖)² > b − s`.

This does not close Math-#193: the D1 kernel formalization remains partial, and external human review is absent.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3

---
_Generated by [Claude Code](https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3)_





---
_Generated by [Claude Code](https://claude.ai/code)_
