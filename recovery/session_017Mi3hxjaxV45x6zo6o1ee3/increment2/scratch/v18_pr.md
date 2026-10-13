**Object:** `CAP-FIRST-EXIT-LEAN-20261002` v1.8. **Stacked on Math-#257** (v1.7, head `192be0d`), which is stacked on #255, which is stacked on #253. The base branch is #257's head branch.
- **Author:** Anthropic / Claude, session `017Mi3hx…`.
- **Scientific effect NONE.** No register, catalog, STATUS, PROOF_INDEX, GRAPH or `formal/` change.
- Same GitHub account as every lane, so organizational-independence credit is 0.
- **The author will not merge.** Merge order: #253, #255, #257, then this PR. Each retargets as its base lands.

## Why

Codex's review of #257 ([4174905561](https://github.com/d6g8k5htny-coder/Math-/pull/257#discussion_r4174905561)) found a gap:
- `slices_strongConcave` and `ridge_capHyp_H1` cover only `x ∈ [−2r, r/2]`;
- the L7 lemmas need the slice data on an open `U ⊇ [−2r, 2r]`;
- no declaration composed L7 with the cap.

#257 now lists that gap as unformalized. This PR proves it, as promised in the reply on that thread.

## What changes (5 declarations, additive)

| Declaration | Statement | SP |
|---|---|---|
| `hessian_increment_D` | L3 on the collar `x ∈ [−2r − ε, 2r + ε]` | L3 on `D` |
| `transverse_hessian_le_D` | for `0 < ε ≤ r/2`, `H(p) ≤ −(λ − 5rm)` on the collar: there `|x + r/2| ≤ 3r`, so the increment is at most `5r` | L4 ("`9r/2 < 5r`" on `D`) |
| `slices_strongConcave_D` | every slice over the collar is `(λ − 5rm)`-strongly concave on `B̄(0, 2r)` | L4 |
| `ridge_capHyp_source` | the composed route, from source data on the collar (below) | §2, (H1), L1–L5, L7, L8, L11, L16–L20 |
| `h1Toy_source` | `x³/3 − x/4 − 20y²` meets every hypothesis (`ε = 1/2`); the theorem gives its ridge and its cap | non-vacuity |

**`ridge_capHyp_source` takes, on the collar × `B̄(0, 2r)`:**
- a joint derivative `Df`;
- a strict derivative `DG` of the transverse gradient `Df ∘ inr`;
- the transverse Hessian `H`, with L3's third-derivative bounds `m`;
- (H1) `8rm² < λ`, with `H(M) ≤ −λ`;
- the critical pins `Df(∓r/2, 0) = 0`;
- the weak bounds `|∂_x³ f(·, 0)| ≤ m` on `[−r/2, r/2]` (L1) and `|∂_x² ∂_y f(·, 0)| ≤ m` on the collar (L5);
- the gap `f(M) − f(S) = r³/6`.

**It proves:**
1. a ridge exists: on `(−2r − ε, 2r + ε)`, `h x` is a critical point of its slice in the open ball;
2. for every such ridge (unique) with L12's `hconv` for `F x = ∂_x f(x, h x)`: `h(∓r/2) = 0`, and (C1)–(C3) and (C2⁺) hold on the cap with `M = (−r/2, 0)` and `S = (r/2, 0)`.

**How the proof goes.**
- L1–L2 give `m ≥ 2`, and L3–L4 on the collar give `δ = λ − 5rm > 11rm`.
- L5 on the collar gives `‖∂_y f(x, 0)‖ ≤ 3mr² < 2rδ`, which yields existence via `ridge_exists`.
- Uniqueness at the critical pins gives `h(∓r/2) = 0`.
- L8 gives the open ball on `[−2r, 2r]`.
- `DG ∘ inr = H` holds by uniqueness of derivatives, so it is negative definite. `ridge_differentiable` then gives `h'`.
- `ridge_inputs_of_joint` gives L11, and `ridge_capHyp_H1` gives the cap.

**What remains (I3).**
- L6 and L9–L12, which give `hconv`.
- The source data themselves. SP has `f ∈ C⁴` near `D` and bounds the third derivatives on `D`. On a thin enough collar, the bound `m + η` holds by continuity, and (H1) stays strict for small `η`. Lean does not prove this step, and the README and I3 say so.

**Unchanged.**
- The 88 v1.7 declarations are byte-identical. In `CapFirstExit.lean` only the header comment changed outside the two inserted blocks.
- The toolchain, pins and workflow are unchanged.

## Validation (local, at `5c66d15`)

- `gate.py` (source mode): PASS, 93 targets. `MANIFEST.json` SHA-256 `2caeb033bf1ceca6467ce30d0f6058cbdbb10ff38e73ada963768d001bb2b969`.
- Unit tests: OK in `-B -S` and in `-B -O -S`.
- `gate.py --execute`: PASS.
  - Fresh build and `leanchecker` replay.
  - Exact axiom audit of all 93 declarations: each uses only `propext`, `Classical.choice` and `Quot.sound`.
  - The three negative controls are REJECTED.
- The hosted checks are green on `5c66d15` (4/4), and the branch is clean.

## Review requested

A nonauthor, non-Claude delta alignment read under ALIGNMENT §5, covering:
- the four new §2 rows, `hessian_increment_D` to `ridge_capHyp_source`;
- the `h1Toy_source` sentence of §3;
- the revised SP row, the `slices_strongConcave` and `ridge_differentiable` notes, and I3.

Points worth checking:
- the collar `[−2r − ε, 2r + ε]` against SP's "near `D`". The bounds `m` are assumed on the collar, while SP bounds them on `D`;
- that quantifying over every ridge in the second conjunct matches SP's `h`. It is unique by `slice_critical_unique`;
- that `hconv` is the only cap input left besides the source data.

This does not close Math-#193.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
