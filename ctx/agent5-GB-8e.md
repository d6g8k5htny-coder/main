# ctx: agent 5 — GB-8e (RETRO-MATH-BIND-01)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8e (D1–D2 lifetime/remainder) per CoS assignment main#475/6369972075. The unit list comes from `docs/retro/RETRO-MATH-BIND-01.slices.json` on main#527 (head 8630f8d8). PICKUP: main#475/6370725095.
- Data file: `docs/retro/GB-8e.agent5.node-candidates.json`. This is the proposal format of #534, **not** the v2 bindings file.
- Per GB-5 (#522, e1d1ebc2) Q6, no v2 record is added. None of the 13 units has a node in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes; 0 reference these paths or their directory names). `architecture/graph-artifact-bindings.v2.json` is unchanged.
- Each unit is pinned as `kind: tree` at the Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`, with the tree sha plus a per-file git blob sha and sha256. The sha256 values were computed from transient raw reads at the pinned commit, and every git blob sha was recomputed from the bytes and matched the tree (141/141). Implementation pins cover code inside the unit (`.py`, `.lean`, `.sh`) plus the workflows at the cut whose text mentions the unit path (role `workflow-reference`), each pinned by blob and sha256. First-add and last-change commits are recorded as provenance (history at the cut).
- `proposed_node_id` values (prefix `cand.`) are **proposals derived from the path**. They aren't in the graph; creating them is a separate reviewed GRAPH.json change.

## Units (13)
| unit | tree sha @9f3fb691 | files | impl pins | proposed node | Drive leads |
|---|---|---|---|---|---|
| `coefficients/side24_remainder_20260929` | `fa06b652d5433120b7eabd3a8b612ff53dce60e1` | 6 | 3 | `cand.coefficients.side24-remainder` | 2 |
| `companions/d2_atom_integral_20261006` | `5afa4e69a16e0ac81f15b1d7221aa093c644a471` | 7 | 6 | `cand.companions.d2-atom-integral` | 2 |
| `companions/d2_moment_bridge_20261006` | `374a339bb466e0da0f7edfd334c705fe9f294022` | 6 | 6 | `cand.companions.d2-moment-bridge` | 2 |
| `companions/d2_square_schur_20261006` | `c3165be6f4f6c17e6dce96e332edf194eaace9cc` | 7 | 7 | `cand.companions.d2-square-schur` | 2 |
| `frontiers/c7_elder_two_thirds_20261007` | `c005fde11094bbb72de3f2889a7af817cc28aeba` | 15 | 3 | `cand.frontiers.c7-elder-two-thirds` | 0 |
| `frontiers/c8_lifetime_difference_constant_20261001` | `4c33a54bb1b6408b30c3610432e900e68e9db69c` | 8 | 4 | `cand.frontiers.c8-lifetime-difference-constant` | 0 |
| `frontiers/c8_torus_vanishing_remainder_20261001` | `04fc5584c8b8268d9b61c44e6c97085a8a249237` | 9 | 5 | `cand.frontiers.c8-torus-vanishing-remainder` | 0 |
| `frontiers/d1_line_20261004` | `e8cd84e3742bd87a2d24e1c4ddae00732a90051b` | 5 | 2 | `cand.frontiers.d1-line` | 0 |
| `frontiers/d1_sharp_remainder_20261001` | `3bb8701027fc7f202d4fc60003b2e118502b8ec8` | 5 | 3 | `cand.frontiers.d1-sharp-remainder` | 0 |
| `frontiers/d1_third_order_law_20261001` | `d1c60cff48361a11e1f2770b33b6ee0146555afb` | 5 | 6 | `cand.frontiers.d1-third-order-law` | 0 |
| `frontiers/finite_l_d2_coefficient_20261018` | `99373bcddd581e727b35156c54733b7312a6131a` | 36 | 6 | `cand.frontiers.finite-l-d2-coefficient` | 0 |
| `frontiers/fixed_r_inverse_lifetime_20260930` | `1a0e07aaeb6c34a915055003f238b6d34de1dae8` | 11 | 7 | `cand.frontiers.fixed-r-inverse-lifetime` | 0 |
| `frontiers/lifetime_c3r_rate_20261013` | `bc35c180babeaea219162cae22ce31e1d0db5e35` | 21 | 4 | `cand.frontiers.lifetime-c3r-rate` | 0 |

Status: 13 tree-pinned; 0 absent; 0 v2 records added; 13 unbound_artifact; 0 existing-node matches.

- `frontiers/finite_l_d2_coefficient_20261018`: no workflow at the cut mentions this path, so its implementation pins are in-unit code only.

## Overlap check
- GB-7c (#530) binds `coefficients/side24_v1` only. GB-8e's `coefficients/side24_remainder_20260929` is a distinct tree.
- GB-7f (#529) binds `imports/lifetime_parent_20260925/*` and `reviews/d1_section9_borel_repair_20260925/REPAIR.md`. None of the GB-8e units are in it.
- No Codex-stalled open-PR directory is bound. All 13 units are landed at the cut.

## Drive / Dropbox
- Drive (read-only title search) queries: lifetime, remainder, d2. The title-matching leads are listed as `external_unverified`: side24 remainder repair docs (LS-DER-064, LS-REQ-037) and D2 companion carriers (C248 source-binding repair, S_D2 Wave0 evidence). Their bytes were **not fetched** and **not imported** under `imports/historical/GB-8e/` (NOT EXECUTED). They are never bound.
- Dropbox: no access (no connector), and I didn't work around that.

## Checks
- The data file parses as JSON, and its counts match the unit list (13/13). The blob shas were re-derived from bytes (141/141 match).
- `tools/graph_binding_check.py` and pytest were **NOT EXECUTED** by the author: there's no repository checkout on the box for this lane. The v2 file is untouched, so the checker baseline (bound 21, nodes_not_bound 28, stale 0, missing 0, unknown_node 0 per #534) is expected to be unchanged. CI on this PR is the check of record.

## NOT EXECUTED
- Drive byte fetch and import; per-unit Drive queries; math content review of any unit; manifest re-verification (left to the owner lanes); running the in-unit scripts or workflows.

Integrator: agent 6 (non-author). Merge commit only. No self-merge.

## Amendment 2026-10-23 ~19:00Z (OP-PRIVACY, per CoS norm main#539/6372418882)
- All 6 Google Drive file IDs in `docs/retro/GB-8e.agent5.node-candidates.json` (4 `drive_leads[].file_id` values plus 2 in the lane-level note) are replaced with `WITHHELD (OP-PRIVACY)`. Titles, dates and match text are kept. Pins, units and counts are unchanged.
- This is an additive commit. The IDs remain in public predecessor commit `89c527cde116e5b14fb3a513087e44cf8f7bad1e`; any history action is Dylan's decision. I took none.
