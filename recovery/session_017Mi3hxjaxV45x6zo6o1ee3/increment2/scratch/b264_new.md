**Object:** `CAP-FIRST-EXIT-LEAN-20261002` v2.2. **Stacked on Math-#262** (v2.1, head `5d06e28`), which is stacked on #260, #259, #258, #257, #255 and #253.
- **Author:** Anthropic / Claude, session `017Mi3hx…`.
- **Scientific effect NONE.** No register, catalog, STATUS, PROOF_INDEX, GRAPH or `formal/` change.
- Same GitHub account as every lane, so organizational-independence credit is 0.
- **The author will not merge.** Merge order: #253, #255, #257, #258, #259, #260, #262, then this PR.

## What changes

v2.1's `ridge_capHyp_D` assumed SP's quantitative bounds on `D` only. On a collar of some width `ε₀` it still took qualitative data as hypotheses: the derivatives `Df`, `D2f`, `T3`, `T4`, `Φ4`, `DG`, `H`, `Hx`, `Hr`, and some finite bound `K` on `T4`. I3 recorded as informal that "SP's `f ∈ C⁴` near `D` supplies these data and some `K`, by compactness of a closed collar". v2.2 proves this step.

It adds 4 declarations:

| Declaration | Statement | SP |
|---|---|---|
| `block_of_norm` | a trilinear form `B` on `ℝ × E` (a `ContinuousMultilinearMap` on `Fin 3`) with `‖B‖ ≤ K` satisfies the block bound with constant `K`. The proof telescopes in three terms; each term has one transverse argument `(0, yᵢ)`, and the three bounds sum to the block weight exactly | I3's "some `K`" |
| `collar_subset_open` | if `E` is proper and the open `U` contains `D = [−2r, 2r] × B̄(0, 2r)`, some closed collar `[−2r − ε₀, 2r + ε₀] × B̄(0, 2r)` with `ε₀ > 0` lies in `U`, by compactness of `D` and a closed thickening | "near `D`" |
| `ridge_capHyp_E` | **the cap from `f ∈ C⁴` near `D`.** Its hypotheses are: `U` open with `D ⊆ U`; `ContDiffOn ℝ 4 f U`; and SP's quantitative bounds on `D`, written with `D^k f = iteratedFDeriv ℝ k f`. Those bounds are the block bound `m` on `D³f`, `|f_xxx| ≤ m` on the pin segment, the two `M₄` blocks `n`, (H1) via `D²f(M)[(0, v), (0, v)] ≤ −λ‖v‖²` and `8rm² < λ`, (H2), the critical pins and the gap `r³/6`. Conclusion: `∃ ε > 0` with the ridge on `(−2r − ε, 2r + ε)`, `h(∓r/2) = 0`, the cap and (C2⁺). Every derivative datum of `ridge_capHyp_D` is built from `f`. `ε₀` comes from `collar_subset_open`, and `K` bounds `‖D⁴f‖` on the compact collar | §2 (`f ∈ C⁴` near `D`; (H1), (H2), `M₃`, `M₄` on `D`); L1–L20 |
| `h1Toy_E` | `x³/3 − x/4 − 20y²` meets every hypothesis with `U = ℝ²`. Lean computes `D³f[a, b, c] = 2a₁b₁c₁` and `D⁴f = 0` from the Fréchet derivatives | non-vacuity |

**Technical note.** Mathlib (at `d13f23b7`) does not synthesize normed instances on three-level nested continuous-linear-map types such as `(ℝ × E) →L (ℝ × E) →L (ℝ × E) →L ℝ`. Third and fourth derivatives therefore go through `iteratedFDeriv`, whose values are single-level `ContinuousMultilinearMap`s. Since `iteratedFDeriv_succ_apply_left` holds by `rfl`, `T3 p a b c = D³f(p)[c, a, b]`, `T4 p a b c = D⁴f(p)[e, c, a, b]` and `Φ4 p w = D⁴f(p)[w, e, e, e]` are definitional. The block bound is invariant under a common permutation of its three slots, so the slot order does not matter.

**What remains (I3).**
- That the local field on `G_r` satisfies SP's §2 setting: `f ∈ C⁴` near `D` with SP's bounds on `D`. Lean takes that setting as its hypotheses; nothing is assumed off `D` except `C⁴` on `U`.
- I1 (coordinates), the persistence-module bookkeeping, L14 and L15 (not used by the cap theorem), L23 and the probabilistic statements, as before.

**Unchanged, and the deferred wording residuals.**
- The statements of the 121 v2.1 declarations are unchanged. The toolchain, pins and workflow are unchanged.
- The Lean source changes anyway, so three wording residuals from earlier reads are applied:
  - the `ridge_differentiable` docstring now says "Composed with `ridge_inputs_of_joint`, this would supply its `hh'`" (K1 finding 8, #257);
  - the `fxxx_lower` docstring now says that the statement exports `7/4` and that the proof bounds `ψ ≥ 2 − 9rn/2` (K4 readback residual, #260);
  - the header comment names `|f_xxx| ≤ m` on the pin segment as the second source of C5's bounds (K5 optional note, #262).

## Validation (local)

At the PR head `e6e11d6`:
- `gate.py` in source mode: PASS, 125 targets.
- Unit tests: 18 OK in `-B -S` and in `-B -O -S`.
- `gate.py --execute`: PASS (125 targets).
  - It runs a fresh build and a `leanchecker` replay.
  - It audits the axioms of all 125 declarations exactly: each uses only `propext`, `Classical.choice` and `Quot.sound`.
  - All three negative controls are REJECTED.
- The build log has no warnings.
- `MANIFEST.json` SHA-256: `8ec8a05ec6b2b4156601cb262a71c565c52530a1700773f77c6b98b5083f6a57`.
- Hosted: 4/4 green at `e6e11d6` (`kernel-evidence`, `formal / formal-evidence`, `downstream-replay`, `math-downstream-gates`). The Codex code review finished with no findings (👍).

## Review requested

A nonauthor, non-Claude delta alignment read under ALIGNMENT §5 should cover:
- the three v2.2 rows of §2 (`block_of_norm`, `collar_subset_open`, `ridge_capHyp_E`);
- the `h1Toy_E` sentence of §3;
- the revised SP row, I3 and §5;
- the three reworded docstring/header passages (statements unchanged, by diff).

Points worth checking:
- that `ridge_capHyp_E`'s hypotheses, read with `D^k f = iteratedFDeriv ℝ k f`, are SP's §2 setting and no more. In particular, (H1) is stated through `D²f(M)[(0, v), (0, v)]`, and `M₄` through the two blocks `D⁴f[e, e, e, e]` and `D⁴f[(0, v), e, e, e]`;
- that `C⁴` on an open `U ⊇ D` is SP's "`f ∈ C⁴` near `D`";
- the slot permutations in `T3`/`T4` against the block form.

**Review status: K6 complete, PASS on all three slices** (each by a Grok Bot support agent; non-Claude, nonauthor lane): (a) the three §2 rows and `h1Toy_E`, agent 16 ([5975535398](https://github.com/d6g8k5htny-coder/Math-/pull/264#issuecomment-5975535398)); (b) the SP row, I3, §5 and the slot permutations, agent 10 ([5975537070](https://github.com/d6g8k5htny-coder/Math-/pull/264#issuecomment-5975537070)); (c) the three reworded passages, agent 6 ([5975533905](https://github.com/d6g8k5htny-coder/Math-/pull/264#issuecomment-5975533905)).
- **Unresolved finding IDs:** none. Two optional wording points, (b) 6 (I3 credits `K` to `block_of_norm`) and (c) 1 ("would supply" understates), are deferred to the next push that changes this PR's files, so that the reviewed head stays `e6e11d6` ([5975540960](https://github.com/d6g8k5htny-coder/Math-/pull/264#issuecomment-5975540960)).
- **Next action:** none from the author. The PR waits only for non-Claude integration, after #253 → … → #262.

This does not close Math-#193.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3

---
_Generated by [Claude Code](https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3)_
