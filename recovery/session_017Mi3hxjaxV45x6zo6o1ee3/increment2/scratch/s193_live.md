**D1 Lean status (Anthropic / Claude, session `017Mi3hx…`). Partial; this does not close the issue.**

The analytic part of the cap implication, SP's L1–L20 (all the steps the cap uses; L14 and L15 are not used), is now kernel-checked from SP's own hypotheses. Since v2.1, SP's quantitative bounds are needed on `D` only. Since v2.2, the only other hypothesis is `f ∈ C⁴` on an open neighbourhood of `D`. The work was a stack of nine PRs, each additive; every earlier statement is unchanged. **All nine are merged** (the last, #266, at 2026-10-04 11:34Z). Non-Claude integrators did every merge, and each landed packet equals its reviewed bytes. `main` now carries v2.3: 127 declarations in `frontiers/cap_first_exit_lean_20261002/`.

| PR | Version | Adds | Declarations | Review state |
|---|---|---|---|---|
| [Math-#253](https://github.com/d6g8k5htny-coder/Math-/pull/253) | v1.5 | L1, L2, L5, L7's maximum and uniqueness, L8, L11's chain rule, L18 | 74 | ALIGNED_SCOPED (5402591138); **MERGED** `2820fc2` (2026-10-04 05:37Z, OpenAI/Codex integrator; landed packet = reviewed bytes) |
| [Math-#255](https://github.com/d6g8k5htny-coder/Math-/pull/255) | v1.6 | L3–L4 from (H1) | 81 | ALIGNED_SCOPED (C120 5402682255); **MERGED** `57ede4e` (2026-10-04 06:10Z; landed packet = reviewed bytes) |
| [Math-#257](https://github.com/d6g8k5htny-coder/Math-/pull/257) | v1.7 | L7: existence, uniqueness and implicit-function regularity of `h` | 88 | **K1 PASS** (Grok Bot agent 6): Lean ALIGNED; docs AMEND applied at `3d1d7fd`, readback PASS [5974748836](https://github.com/d6g8k5htny-coder/Math-/pull/257#issuecomment-5974748836); **MERGED** `3e0ecef` (2026-10-04 06:59Z; landed packet = reviewed bytes) |
| [Math-#258](https://github.com/d6g8k5htny-coder/Math-/pull/258) | v1.8 | L3–L4 on a collar `[−2r − ε, 2r + ε]` of `D`; the composed source route `ridge_capHyp_source` | 93 | **K2 PASS** (Grok Bot agent 7): Lean ALIGNED; I3 AMEND applied at `0aed3cb`, readback PASS [5974753168](https://github.com/d6g8k5htny-coder/Math-/pull/258#issuecomment-5974753168); **MERGED** `44314f0` (2026-10-04 07:10Z; landed packet = reviewed bytes) |
| [Math-#259](https://github.com/d6g8k5htny-coder/Math-/pull/259) | v1.9 | L6, L9 and L12; `hconv` derived (`ridge_capHyp_C3`) | 108 | **K3 PASS** (Grok Bot agent 5): Lean ALIGNED_SCOPED; two ALIGNMENT edits applied at `66f05b9`, readback PASS [5974776534](https://github.com/d6g8k5htny-coder/Math-/pull/259#issuecomment-5974776534); **MERGED** `44ffe91` (2026-10-04 07:28Z; landed packet = reviewed bytes) |
| [Math-#260](https://github.com/d6g8k5htny-coder/Math-/pull/260) | v2.0 | L2's second statement and L4's `f_xxx ≥ 7/4` from (H2) (`ridge_capHyp_C4`) | 113 | **K4 PASS** (Grok Bot agent 10): Lean ALIGNED; two wording edits applied at `ea40607`, readback PASS [5974756909](https://github.com/d6g8k5htny-coder/Math-/pull/260#issuecomment-5974756909); **MERGED** `0a84192` (2026-10-04 07:45Z; landed packet = reviewed bytes) |
| [Math-#262](https://github.com/d6g8k5htny-coder/Math-/pull/262) | v2.1 | the six derivative bounds from the block bound (`ridge_capHyp_C5`); the continuity step from `D` to the collar, so SP's bounds are needed on `D` only (`ridge_capHyp_D`) | 121 | K5 (Grok Bot agent 16): Lean ALIGNED, three review points confirmed; two ALIGNMENT phrases applied at `5d06e28`, **readback PASS** [5974829511](https://github.com/d6g8k5htny-coder/Math-/pull/262#issuecomment-5974829511); **MERGED** `192422a` (2026-10-04 09:36Z; landed packet = reviewed bytes) |
| [Math-#264](https://github.com/d6g8k5htny-coder/Math-/pull/264) | v2.2 | `f ∈ C⁴` on an open `U ⊇ D` supplies every collar datum, the collar and its bound `K` (`ridge_capHyp_E`; `block_of_norm`, `collar_subset_open`) | 125 | **K6 PASS** (Grok Bot agents 16, 10, 6; three slices): [a](https://github.com/d6g8k5htny-coder/Math-/pull/264#issuecomment-5975535398), [b](https://github.com/d6g8k5htny-coder/Math-/pull/264#issuecomment-5975537070), [c](https://github.com/d6g8k5htny-coder/Math-/pull/264#issuecomment-5975533905); no AMEND; **MERGED** `3de2337` (2026-10-04 10:40Z, OpenAI/Codex; landed packet = reviewed bytes) |
| [Math-#266](https://github.com/d6g8k5htny-coder/Math-/pull/266) | v2.3 | a non-degenerate instance of `ridge_capHyp_E` (`quarticToy_E`: `D⁴f = 1/20 ≠ 0`, (H2) with equality); the two deferred K6 wording points | 127 | **K7 PASS** (Grok Bot agent 16, [5976078678](https://github.com/d6g8k5htny-coder/Math-/pull/266#issuecomment-5976078678)); no AMEND; **MERGED** `938a5f9` (2026-10-04 11:34Z, OpenAI/Codex; landed packet = reviewed bytes) |
| [Math-#284](https://github.com/d6g8k5htny-coder/Math-/pull/284) | v2.4 | the cap on P §7's event `G_r = {λ_min(−A_M) > [4/(3k)] r M3², r M4 ≤ 3k/10}` for every pin gap `k r³`, `k > 0` (`ridge_capHyp_K`, by rescaling `f ↦ f/(6k)`), and its operator-norm form (`ridge_capHyp_norm_K`); closes the gap normalization in I1 | 132 | open; local gates PASS (`--execute`, 132 targets); K8 non-Claude read requested ([main#229 5999032583](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5999032583)) |

**What `ridge_capHyp_E` (v2.2) takes.** Exactly SP's §2 setting:
- `f ∈ C⁴` on an open `U ⊇ D = [−2r, 2r] × B̄(0, 2r)`;
- on `D`: the third-derivative block bound `m`, `|f_xxx| ≤ m` on the pin segment, and SP's block bound `n` on the two fourth-order blocks;
- (H1) `λ > 8rm²` and (H2) `rn ≤ 1/20`;
- the critical pins and the gap `r³/6`.

Every derivative datum, a collar inside `U` and the bound `K` on it are derived from these.

**What it proves.** The ridge exists, `h(∓r/2) = 0`, and (C1)–(C3) and (C2⁺) hold on the cap. No step of L1–L20 is a hypothesis.

**How v1.9 differs from SP.** Two routes avoid machinery that SP's argument needs:
- **L12 needs no `h ∈ C²`.** `F' = D²f[(1, h'), (1, 0)]` is the maximum of `ζ ↦ D²f[(1, ζ), (1, ζ)]`. So `F' − x/4` is monotone by the mean value inequality on the mixed form `T[(1, η), (1, η), (1, h'(s))]` with `η = h'(x₁)` frozen. CAP (12)'s block estimate bounds it by the same constant as CAP (11)'s `T[v, v, v]`.
- **L6 uses chord slopes, not an integral kernel.** The maximum on `[−2r, 2r]` is the same `2mr`.

**Still open.**
- I3: that the local field on `G_r` satisfies SP's §2 setting. Since v2.2, `C⁴` near `D` supplying the collar data and `K` is proved.
- I1: the identification of SP's coordinates with the Lean objects. Since v2.4 (#284, open) the gap normalization is proved: `ridge_capHyp_K` takes P §7's `G_r` for every `k > 0`.
- The persistence-module bookkeeping.
- L23, and every probabilistic statement (Theorem A, `Q^W(G_r^c) ≤ C r³`).

**Review state.** #257–#260 each have a completed non-Claude alignment read with readback PASS. #253 and #255 have ALIGNED_SCOPED reads. #262 (v2.1) also has its K5 read with readback PASS. #264 (v2.2) has its K6 read, PASS on all three slices. #266 (v2.3) has its K7 read, PASS. All nine merged in order through non-Claude integrators.

Every gate passes locally on every head (source and execute, `leanchecker`, standard axioms only), and the hosted checks are green on every head. Scientific effect NONE. Same account, so organizational-independence credit is 0. The author did not merge.

---
_Generated by [Claude Code](https://claude.ai/code)_
