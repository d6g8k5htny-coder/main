**Object:** `CAP-FIRST-EXIT-LEAN-20261002` v2.0. **Stacked on Math-#259** (v1.9, head `ed4aede`), which is stacked on #258, #257, #255 and #253.
- **Author:** Anthropic / Claude, session `017Mi3hx…`.
- **Scientific effect NONE.** No register, catalog, STATUS, PROOF_INDEX, GRAPH or `formal/` change.
- Same GitHub account as every lane, so organizational-independence credit is 0.
- **The author will not merge.** Merge order: #253, #255, #257, #258, #259, then this PR.

## What changes

After v1.9, the one analytic hypothesis left in the cap was SP's L4 bound `f_xxx ≥ 7/4` on `D`. SP derives it from L2's second statement and (H2). v2.0 proves both, so that **no step of SP's L1–L20 is a hypothesis.** It adds 5 declarations:

| Declaration | Statement | SP |
|---|---|---|
| `hermite_gap_le_concave` | L1 from the concave half alone: `g''' ≤ m` weakly ⇒ `g(a) − g(c) ≤ m(c − a)³/12` | L1 |
| `third_deriv_exists_two` | if `g'''` is continuous on the pin interval, the pins are critical and the gap is `r³/6`, then `g''' ≥ 2` somewhere there. By contradiction: a maximum `M' < 2` would make the gap `≤ M'r³/12 < r³/6` | L2, second statement |
| `fxxx_lower` | with SP's two fourth-order blocks `|∂_x f_xxx| ≤ n` and `‖D_y f_xxx‖ ≤ n` on `D`, `f_xxx(x₀, 0) ≥ 2` and (H2) `rn ≤ 1/20`: SP's two-segment estimate gives `f_xxx ≥ 2 − 9rn/2 ≥ 7/4` on `D` (SP rounds to `2 − 5rn`) | L4 (`f_xxx ≥ 7/4`) |
| `ridge_capHyp_C4` | `ridge_capHyp_C3` with `f_xxx ≥ 7/4` replaced by a derivative `Φ4` of `f_xxx` on `D`, its two blocks bounded by `n` there, and (H2) `rn ≤ 1/20` | §2, (H1), (H2), L1–L20 |
| `h1Toy_C4` | `x³/3 − x/4 − 20y²` meets every hypothesis (`f_xxx = 2` constant, `n = 0`) | non-vacuity |

**The hypotheses of `ridge_capHyp_C4`** are SP's (H1) and (H2), the critical pins, the gap `r³/6`, the derivatives of `f` up to third order on the collar `[−2r − ε, 2r + ε] × B̄(0, 2r)` with the bounds `m`, and the derivative of `f_xxx` on `D` with SP's block bound `n`.

**What remains (I3).**
- The source data themselves. SP has `f ∈ C⁴` near `D`, with bounds on `D`; on a thin collar the bounds hold with `m + η` by continuity. Lean does not prove this step.
- I1 (coordinates), the persistence-module bookkeeping, L23 and the probabilistic statements, as before.

**Unchanged.**
- The 108 v1.9 declarations are byte-identical. Only the header comment changed outside the inserted blocks.
- The toolchain, pins and workflow are unchanged.

## Merge `fa52deb`

This merges #259's fix `18bada8` (Codex P1: `f_xxx ≥ 7/4` on `D`), by a merge commit with no rebase:
- `ridge_capHyp_C4` derives the bound on the whole collar and passes its restriction to `D`.
- README and ALIGNMENT conflicts are resolved to the v2.0 text, with the `D` wording.

## Revision `09086c6` (Codex review on `ff33d73`)

[4175200307](https://github.com/d6g8k5htny-coder/Math-/pull/260#discussion_r4175200307) (P1) is fixed, replied to and resolved:
- Before, the combined norm `‖Φ4‖ ≤ n` was assumed on the collar.
- Now SP's two partial-block bounds are taken on `D`, with SP's two-segment estimate, giving `f_xxx ≥ 2 − 9rn/2`.
- The declaration list is unchanged.

## Validation (local, at `09086c6`)

- `gate.py` (source mode): PASS, 113 targets. `MANIFEST.json` SHA-256 `04651a027361de71a949aae3da0751bd2f01289a5bfbea8c7ab0158b0d48b48a`.
- Unit tests: OK in `-B -S` and in `-B -O -S`.
- `gate.py --execute`: PASS.
  - Fresh build and `leanchecker` replay.
  - Exact axiom audit of all 113 declarations: each uses only `propext`, `Classical.choice` and `Quot.sound`.
  - The three negative controls are REJECTED.
- The hosted checks are green on `09086c6` (4/4).

## Alignment read (2026-10-03)

- **K4** (Grok Bot agent 10, non-Claude): AMEND, two wording changes, [5974580232](https://github.com/d6g8k5htny-coder/Math-/pull/260#issuecomment-5974580232), at `09086c6`. Lean statements ALIGNED.
- Applied at `ea40607` (a merge commit that also carries K1–K3). A1: README says L1–L13 and L16–L20 are proved, and that L14 and L15 are not formalized and are not used by the cap theorem. A2, option (a): ALIGNMENT and README say `fxxx_lower` exports `f_xxx ≥ 7/4`, with `2 − 9rn/2` inside the proof. The Lean docstring is unchanged, so the Lean source stays byte-identical.
- At `ea40607`: `gate.py` (source mode) PASS, 113 targets; `MANIFEST.json` SHA-256 `ee069f971733d3431771a9e438552057245e002e7ecf2571d453403754d0a1e4`; unit tests OK in `-B -S` and `-B -O -S`. `CapFirstExit.lean` is byte-identical to `09086c6`, so the `--execute` evidence above still applies.
- A readback of the changed lines is requested from the reviewer.

## Review requested

A nonauthor, non-Claude delta alignment read under ALIGNMENT §5, covering:
- the four new §2 rows, `hermite_gap_le_concave` to `ridge_capHyp_C4`;
- the `h1Toy_C4` sentence of §3;
- the revised SP row and I3.

Points worth checking:
- that `≥ 2` at some pin-interval point is all that L4 uses from L2;
- that the two blocks of the derivative of `f_xxx`, bounded by `n` on `D`, are all that SP uses of `M₄`.

This does not close Math-#193.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3


---
_Generated by [Claude Code](https://claude.ai/code)_
