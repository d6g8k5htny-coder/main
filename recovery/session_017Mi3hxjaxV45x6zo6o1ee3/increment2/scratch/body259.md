**Object:** `CAP-FIRST-EXIT-LEAN-20261002` v1.9. **Stacked on Math-#258** (v1.8, head `5c66d15`), which is stacked on #257, #255 and #253.
- **Author:** Anthropic / Claude, session `017Mi3hx…`.
- **Scientific effect NONE.** No register, catalog, STATUS, PROOF_INDEX, GRAPH or `formal/` change.
- Same GitHub account as every lane, so organizational-independence credit is 0.
- **The author will not merge.** Merge order: #253, #255, #257, #258, then this PR.

## What changes

Through v1.8, L12's convexity `hconv`, that `F − (x + r/2)(x − r/2)/8` is convex for `F x = ∂_x f(x, h x)`, was the one analytic hypothesis of the cap. v1.9 derives it, proving SP's L6, L9 and L12 with SP's constants. It adds 15 declarations.

**L6: `‖w'‖ ≤ 2mr`**, in `convex_deriv_sign`, `two_node_deriv_bound` and `slice_gradient_deriv_bound`.
- Take L5's convexity pair (weak `|w''| ≤ m`) and `w(∓r/2) = 0`. Then `‖w'(x)‖ ≤ m · max(|x|, r/2)`.
- Outside the pins, the chord through them has slope 0. Between them, the chord to the nearer pin plus L5's bound on `w(x)` gives `mr/2`.
- SP averages over the pin interval with an integral kernel. The two maxima on `[−2r, 2r]` agree.

**L9: `‖h'‖ ≤ 2s + 2s²`, with `s = mr/δ`**, in `schur_max`, `ridge_second_identity`, `dx_grad_increment` and `l9_arith`.
- The differentiated ridge equation is `D²f[(1, h'), (0, ·)] = 0`. It gives `δ‖h'‖² ≤ ∂_x∇_y f [h'] ≤ ‖∂_x∇_y f‖‖h'‖`, with no matrix inverse.
- `‖∂_x∇_y f(x, h)‖ ≤ ‖w'(x)‖ + m‖h‖` (L3 in `y`).
- Together with L8's `δ‖h‖ ≤ 2mr²`, this is SP's `u = 2/k + 2/k²`, with `k` replaced by `δ/(rm) > 8m − 5`.

**L12, in `ridge_hconv`.**
- `F' = D²f[(1, h'), (1, 0)]` is the maximum of `ζ ↦ D²f[(1, ζ), (1, ζ)]` (`schur_max`, completing the square).
- So for `x₁ < x₂` and `η = h'(x₁)`: `F'(x₂) − F'(x₁) ≥ D²f(x₂)[(1, η), (1, η)] − D²f(x₁)[(1, η), (1, η)]`.
- The mean value inequality bounds the right side below by `(x₂ − x₁) · min T[(1, η), (1, η), (1, h')]`. This is the same third-derivative expression that SP's CAP (11) bounds.
- No second derivative of `h` is used. SP's route needs `h ∈ C²` for the second ridge identity.
- The bound itself comes from `trilinear_lower`, `l12_lower` and `l12_constants`: `f_xxx − m((1 + u)³ − 1) ≥ 7/4 − (48/121)(3 + 72/121 + 576/14641) > 1/4`. This is SP's `2554128/1771561` margin.

**Capstone: `ridge_capHyp_C3`.**
- It takes `ridge_capHyp_source`'s source data and adds, on the collar:
  - a second derivative `D2f`;
  - third derivatives `T3 p a b` of the scalar maps `p ↦ D²f p [a, b]`;
  - the bound `m` on every third-derivative block other than `f_xxx`, in a combined form equivalent to SP's blocks;
  - `f_xxx ≥ 7/4`.
- It proves that the ridge exists and that every ridge gives `h(∓r/2) = 0` and (C1)–(C3), (C2⁺) on the cap. **`hconv` is no longer a hypothesis.**
- `ridge_source_facts` packages the v1.8 ridge facts. The split keeps each declaration within the kernel's default heartbeat budget; `set_option` is forbidden by the gate.
- `transverse_hessian_eq` ties v1.8's `H` to `D2f`. Second-derivative symmetry comes from Mathlib's `second_derivative_symmetric_of_eventually`.

**Toy: `h1Toy_C3`.** `x³/3 − x/4 − 20y²` meets every hypothesis, with `D²f = diag(2x, −40)` and only `f_xxx = 2` nonzero. The theorem gives its ridge and cap with no `hconv` supplied.

## What remains (I3)

- SP's L4 bound `f_xxx ≥ 7/4` on `D`. SP derives it from L2's second statement (some point has `f_xxx = 2`) and (H2) `rn ≤ 1/20`. It is a hypothesis here, on `D` only. The stacked #260 derives it.
- The source data themselves, and the continuity step from SP's bounds on `D` to the collar.
- I1 (coordinates), the persistence-module bookkeeping, L23 and the probabilistic statements, as before.

**Unchanged.**
- The 93 v1.8 declarations are byte-identical. Only the header comment changed outside the two inserted blocks.
- The toolchain, pins and workflow are unchanged.

## Revision `18bada8` (Codex review on `ed4aede`)

[4175121617](https://github.com/d6g8k5htny-coder/Math-/pull/259#discussion_r4175121617) (P1) is fixed, replied to and resolved:
- `ridge_capHyp_C3` now takes `f_xxx ≥ 7/4` only on SP's domain `D = [−2r, 2r] × B̄(0, 2r)`, where L4 proves it.
- Before, it assumed the bound on the whole collar.
- The proof uses the bound only at the ridge points over `[−2r, 2r]`.
- The declaration list is unchanged.

## Validation (local, at `18bada8`)

- `gate.py` (source mode): PASS, 108 targets. `MANIFEST.json` SHA-256 `0ae0fb4016c335aa6490d4e8a9496976778705625fa149742257e9c55bd00195`.
- Unit tests: OK in `-B -S` and in `-B -O -S`.
- `gate.py --execute`: PASS.
  - Fresh build and `leanchecker` replay.
  - Exact axiom audit of all 108 declarations: each uses only `propext`, `Classical.choice` and `Quot.sound`.
  - The three negative controls are REJECTED.
- The hosted checks are green on `18bada8` (4/4).

## Review requested

A nonauthor, non-Claude delta alignment read under ALIGNMENT §5, covering:
- the nine new §2 rows, `convex_deriv_sign` to `ridge_capHyp_C3`;
- the `h1Toy_C3` sentence of §3;
- the revised SP row and I3.

Points worth checking:
- that the Schur-maximum route bounds the same quantity as SP's CAP (11)–(12);
- that the combined block bound matches SP's "third-derivative blocks bounded by `m`";
- that `f_xxx ≥ 7/4` is the only analytic input left.

This does not close Math-#193.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3


---
_Generated by [Claude Code](https://claude.ai/code)_
