**Object:** `CAP-FIRST-EXIT-LEAN-20261002` v1.7. **Stacked on Math-#255** (v1.6, head `1cbbbf9`), which is stacked on Math-#253 (v1.5); the base branch is #255's head branch.
- **Author:** Anthropic / Claude, session `017Mi3hx…`.
- **Scientific effect NONE.** No register, catalog, STATUS, PROOF_INDEX, GRAPH or `formal/` change.
- Same GitHub account as every lane, so organizational-independence credit is 0.
- **The author will not merge.** Merge order: #253, then #255, then this PR. Each retargets as its base lands.

## What changes

Through v1.6, the ridge `h` was an input: `h x` was assumed to be a critical point in the ball, and `h` was assumed differentiable. SP's L7 derives both, with the implicit function theorem doing the regularity. v1.7 proves them (7 declarations):

| Declaration | Statement | SP |
|---|---|---|
| `slice_critical_unique` | a `δ`-strongly concave slice, `δ > 0`, has at most one critical point in `K` | L7 (uniqueness) |
| `slice_critical_exists` | if the slice is strongly concave and differentiable on `B̄(0, R)` and `‖Dφ(0)‖ < δR`, a critical point exists in the open ball (the maximum over the compact ball cannot be on the sphere, because there the radial derivative is `≤ w(p) − δR² < 0`) | L7 (interior maximum) |
| `isInvertible_of_negDef` | in finite dimension, `A v v ≤ −δ‖v‖²` makes `A : E →L E →L ℝ` invertible | L7 (`D_y² f` invertible) |
| `ridge_hasStrictFDerivAt` | Mathlib's `implicitFunctionOfProdDomain` for `G = ∂_y f = 0` gives a critical point near `h x0`, which equals `h` by uniqueness. So `HasStrictFDerivAt h (−(DG∘inr)⁻¹ ∘ (DG∘inl)) x0`. No continuity of `h` is assumed | L7 (IFT), L9's formula for `h'` |
| `ridge_differentiable` | `h` is strictly differentiable on `[−2r, 2r]`, given the slice data on an open `U ⊇ [−2r, 2r]` (SP: `f ∈ C⁴` near `D`). This supplies `hh'` of `ridge_inputs_of_joint` | L7 (`h ∈ C¹`) |
| `ridge_exists` | a ridge map `h` exists on `U`, by choice; it is unique by `slice_critical_unique` | L7 ("call it `h(x)`") |
| `h1Toy_ridge_differentiable` | every hypothesis of `ridge_differentiable` holds for `h1ToyF` (`δ = 40`, `U = (−3, 3)`) | non-vacuity |

**I3 after v1.7.** The interface narrows to:
- L6 and L9–L12, the quantitative `w'`, `‖h'‖` and `F''` bounds that give `hconv`;
- the derivative data these consume: third-derivative bounds `m`, and the strict differentiability of `∂_y f` along the ridge.

Composing `ridge_differentiable` with `ridge_inputs_of_joint` and `ridge_capHyp_H1` is described in ALIGNMENT, not packaged as a new capstone.

**Unchanged.**
- The 81 v1.6 declarations are byte-identical; only the header comment changed in `CapFirstExit.lean`.
- The toolchain, pins and workflow are unchanged.

## Validation (local, at `f0920c5`)

- `gate.py` (source mode): PASS, 88 targets. `MANIFEST.json` SHA-256 `f3be0b4c39e75af1768c4e5e2e822c955e8e39ee2b8dedceaaf4b34db41d4483`.
- Unit tests: OK in `-B -S` and in `-B -O -S`.
- `gate.py --execute`: PASS.
  - Fresh build and `leanchecker` replay.
  - Exact axiom audit of all 88 declarations: each uses only `propext`, `Classical.choice` and `Quot.sound`.
  - The three negative controls are REJECTED.
- The hosted checks run on this PR.

## Review requested

A nonauthor, non-Claude delta alignment read under ALIGNMENT §5, covering:
- the six v1.7 rows of §2 (`slice_critical_unique` to `ridge_exists`);
- the `h1Toy_ridge_differentiable` sentence of §3;
- the revised I3.

Points worth checking:
- the open-set hypothesis `U ⊇ [−2r, 2r]` against SP's "near `D`";
- the strict hypothesis `‖Dφ(0)‖ < δR` against SP's `‖w‖ · 2r < δ · 4r²`;
- that the IFT derivative formula matches L9.

This does not close Math-#193.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
