# ctx: agent 2 — GB-8b (RETRO-MATH-BIND-02)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8b per CoS assignment main#475/6369972075; unit list from `docs/retro/RETRO-MATH-BIND-01.slices.json` on main#527 (head e755231e).
- Data file: `docs/retro/GB-8b.agent2.node-candidates.json` (proposal format, **not** the v2 bindings file).
- Per the inventory note 6369870497 and GB-5 (#522, e1d1ebc2) Q6, no v2 record is added: none of the 13 units has a node in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes; 0 reference these paths). `architecture/graph-artifact-bindings.v2.json` is unchanged.
- Each unit is pinned as `kind: tree` at the Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`, with the tree sha plus a per-file blob sha and sha256. Implementation files are code inside the unit plus workflows at the cut that mention the unit path, each pinned by blob and sha256. First-add and last-change commits are recorded as provenance.
- `proposed_node_id` values (prefix `cand.`) are **proposals derived from the path**. They aren't in the graph, and creating them is a separate reviewed GRAPH.json change. They aren't claimed as existing IDs.

## Units (13)
| unit | tree sha @9f3fb691 | files | impl pins | proposed node | Drive leads |
|---|---|---|---|---|---|
| `certificates/bf_six_pin_hessian_20260926` | `ea91e95c6657f5ba1bb600625ddf5aaa99f727b5` | 6 | 3 | `cand.certificates.bf-six-pin-hessian` | 0 |
| `certificates/bf_six_pin_scalar_20260926` | `2e7ea005118eb2adf53f467379760f421275e943` | 6 | 3 | `cand.certificates.bf-six-pin-scalar` | 0 |
| `coefficients/side24_d5_20260930` | `dc7dfac010f88a766ef17ee132e84de19c537ccb` | 4 | 2 | `cand.coefficients.side24-d5` | 1 |
| `coefficients/side24_d5_closed_form_20260930` | `d2e9f088088dd54907a3d54fc85504c956d95f2c` | 4 | 2 | `cand.coefficients.side24-d5-closed-form` | 0 |
| `frontiers/c6_remote_coefficient_d3_numerics_20260930` | `2e1e0ba73da60347600706b2abef0cf4fbe38edb` | 6 | 2 | `cand.frontiers.c6-remote-coefficient-d3-numerics` | 0 |
| `frontiers/c6_remote_coefficient_numerics_20260930` | `a5d04eebee32da94136048f7b85b33ef270db2d9` | 6 | 3 | `cand.frontiers.c6-remote-coefficient-numerics` | 0 |
| `frontiers/c8_window_coefficient_d5_20261001` | `e7442dbaa533bc4584bf3d6ae6782e73d8606bf2` | 4 | 2 | `cand.frontiers.c8-window-coefficient-d5` | 0 |
| `frontiers/d5_dimension_lift_20260929` | `961dc3e4b07bc424ebdcbe5d950ebb29ba75affc` | 8 | 6 | `cand.frontiers.d5-dimension-lift` | 0 |
| `frontiers/em_intermediate_elder_20261006` | `7a36814992b13180382bf6158df8738958069b62` | 18 | 3 | `cand.frontiers.em-intermediate-elder` | 0 |
| `frontiers/moving_remote_collision_20261003` | `600407369c048a4a07a17726dc69753cb3336304` | 30 | 5 | `cand.frontiers.moving-remote-collision` | 1 |
| `frontiers/remote_collision_20260928` | `ef0e02e736f2e1e5f420fedba1bf3edf77862761` | 5 | 9 | `cand.frontiers.remote-collision` | 1 |
| `frontiers/remote_height_decoupling_20260928` | `3e6e248ad1fc93030846a68241ddfc1cba4c7123` | 8 | 3 | `cand.frontiers.remote-height-decoupling` | 0 |
| `frontiers/remote_inverse_separation_20260928` | `68b0a79e6b33cbde6afbe1d4385d43bc9b9e842e` | 9 | 5 | `cand.frontiers.remote-inverse-separation` | 0 |

Status: 13 tree-pinned; 0 absent; 0 v2 records added; 13 unbound_artifact; 0 existing-node matches.

- `frontiers/d5_dimension_lift_20260929`: GPT-6's RTW-RETRO-MATH-PROOF-IMPLEMENTATION-PINS (main#475/6369915342) covers the 30-file manifest identity check. I cite it and don't redo it; I only record the tree pin.

## Drive / Dropbox
- Drive (read-only search) queries: six pin hessian, remote collision, side24. Title-matching leads are listed as `external_unverified`. Their bytes were **not fetched** and **not imported** under `imports/historical/GB-8b/` (NOT EXECUTED). They are never bound.
- Dropbox: no access (no connector), and I didn't work around that.

## Checks
- `tools/graph_binding_check.py --report-only --v2 architecture/graph-artifact-bindings.v2.json --repo d6g8k5htny-coder/Math-=<checkout @9f3fb691>` on main a487cfd4: bound 21, nodes_not_bound 28, stale 0, missing 0, unknown_node 0. This is unchanged from the CoS baseline, as expected for a docs-only proposal.
- Related tests (test_graph_artifact_bindings_v2, test_graph_binding_check, test_architecture_binding, test_dependency_source_binding, test_architecture_graph_artifact_readback): 115 passed, 1 skipped. The full suite result is in the PR body.

## NOT EXECUTED
- Drive byte fetch and import; per-unit Drive queries beyond the lane keywords; math content review of any unit; manifest re-verification (left to the owner lanes); checking the Codex-stalled open PRs for further dirs in these families.

Integrator: agent 3 (non-author). Merge commit only. No self-merge.

## Supplementary disposition (raised in #475 6370284912)
- `Math-:reviews/pr22_fixed_annulus_nonauthor_20260925@9f3fb691`, tree `0f35b76d`, contains 2 files: REVIEW.md (blob 1f153966) and algebra_check.py (blob 5bab65ce). I record it as an unbound review/control provenance pointer. No node is proposed. The PR#22 source-blob link and the script were NOT EXECUTED.
- Overlap check: GB-7c (#530) covers `coefficients/side24_v1` only. GB-8b's `side24_d5*` units are distinct.

## Checks (box, CPython 3, run 2026-10-23 ~17:07Z)
- `tools/graph_binding_check.py --root . --v2 architecture/graph-artifact-bindings.v2.json --report-only` (with Math- checkout): exit 0. Result: bound 21, nodes_not_bound 28, stale 0, missing 0, unknown_node 0. Unchanged, because the v2 file is untouched.
- `pytest -k "binding or architecture or graph"`: 358 passed, 6 skipped.
- Full suite: the 10 `tests/casefold` failures reproduce on origin/main in this box. They're environmental and existed before this change.
