# ctx: agent 1, GB-8s (organizational, NOT A RESULT)

Lane GB-8s, "Misc (governance/audit/imports/experiments)", per main#475 6369972075 row 22 / docs/retro/RETRO-MATH-BIND-01.slices.json. PICKUP main#475 6372550123. Integrator agent 2. credit 0 · OBL OPEN · no flags · sci NONE.

## What was done
- Pinned all 13 GB-8s units as `kind: tree` at the frozen Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`. Each unit has a per-file blob and sha256 in `docs/retro/GB-8s.agent1.inventory.json`, schema `retro-inventory/proposal (NOT v2, NOT A RESULT)`.
- **No v2 binding records were added, and no node ids were proposed.** No GB-8s unit is referenced by any of the 49 nodes in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (searched each node for the unit directory name: 0 matches). Under GB-5 and advisory 6370183473, every unit is therefore **unbound** with the reason `inventory_unit_without_graph_node`, not `broken_binding`. A node can only come from a reviewed Math- GRAPH PR first. Main's bindings must not invent nodes.
- Classification ceiling: AUTHOR_SIDE_CANDIDATE. Nothing is reviewed, accepted or promoted by this inventory.

| unit (frontiers/…) | tree sha @9f3fb691 | files | impl (in-unit code + workflow refs) | first commit (UTC) | last change | GRAPH node |
|---|---|---|---|---|---|---|
| cluster_exponential_weight_20260930 | 3e079f6e4fef389d5adffbc5b322e89e2cc11a2e | 14 | 6 + 1 wf | 0e70540b (2026-09-30) | d3d91251 | none |
| contact_kernel_tail_20260925 | 4b0c522ca56a6be2a712242ef34ebb1cfc402e66 | 7 | 3 + 1 wf | 90e0743a (2026-09-25) | 90e0743a | none |
| extreme_cluster_sampling_20260930 | 69b378737c85129c886ef719d068ba6d62af42c7 | 15 | 6 + 3 wf | 7ab0d31b (2026-09-30) | 56a7904f | none |
| local_cluster_tightness_20260929 | 7ec99c587e85642e503dda28a8a31b7dfc5db20b | 11 | 5 + 1 wf | e7f89290 (2026-09-29) | e7f89290 | none |
| local_pairing_20260930 | 39a3020d0610ae4a3bbca6a3c65cdb6a5b294c3b | 8 | 3 + 1 wf | ec968cb1 (2026-09-30) | 78b3eea5 | none |
| ordinary_short_bar_occurrence_20261001 | 007e442f6630d181734a024e50ee84d44c8d538d | 7 | 3 + 2 wf | c4637773 (2026-10-01) | c4637773 | none |
| ordinary_short_bar_occurrence_all_d_20261001 | 98781eb8951ce834c35bc56fc8ffebaf77632257 | 6 | 3 + 1 wf | cf95d824 (2026-10-01) | 3133181f | none |
| p15_bipartite_overlap_20260928 | 71d34331da10f7969670aa876a2acbd45ce818cb | 10 | 3 + 1 wf | 23b75589 (2026-09-28) | 23b75589 | none |
| p15_tail_load_20260928 | b4fd184fb61d37368bfd65708c36ec462f05f7d6 | 9 | 3 + 1 wf | 245f3d77 (2026-09-28) | 245f3d77 | none |
| price_budget_20260924 | 023206d94f94f0d6596dcf6fcc5ffbf17af49669 | 6 | 3 + 6 wf | 1dfd49e8 (2026-09-24) | 1dfd49e8 | none |
| radial_rate_20260930 | b02e9b07774d75acbefe7d80019b7b32013cf2e0 | 12 | 4 + 4 wf | 71772a89 (2026-09-30) | 71772a89 | none |
| regular_superlevel_geometry_20261020 | 58a4c92539a7fffcc84707fd7515164c4fe125b5 | 38 | 11 + 1 wf | 9f2f79bb (2026-10-20) | 363a533e | none |
| replacement_bar_occurrence_20261001 | bbfb5944108bd3185f9533369ad4dce68e44a983 | 9 | 2 + 1 wf | a38449c3 (2026-10-01) | a38449c3 | none |

Status: 13 tree-pinned; 0 absent; 0 v2 records added; 13 unbound (inventory_unit_without_graph_node); 0 existing-node matches; 152 file pins.

- Method: the tree sha comes from `git rev-parse 9f3fb691:<unit>`. Files come from `git ls-tree -r`, with sha256 taken over the blob bytes. "Implementation" means in-unit code files plus `.github/workflows/*` at 9f3fb691 whose text names the unit path (role `workflow-reference`). First and last commits come from `git log 9f3fb691 -- <unit>`. Nothing tracks a branch tip. The Math- main tip at the time of the run was still 9f3fb691.
- Codex-stalled overlap: none. All 13 units are landed at the cut, and none appears in `docs/retro/OPEN_MATH_PR_CANDIDATES_20261023.json`.

## Sources consulted
- GitHub only: a local clone of public Math- (objects checked with git) and main. **Google Drive was not consulted** because it isn't needed for in-repo tree pins, and no Drive identifiers, links or owner info appear anywhere in this PR (OP-PRIVACY). Dropbox: no access (no connector); I did not try to work around that.
- No Codex-stalled PRs were touched or bound. main#549 was not touched.

## Context from prior agent-1 lanes
- GB-8a (#533): 3 per-file v2 records at dbe6c0a1. Checker baseline afterwards: bindings 36, bound 24/49.
- GB-COV-01 (6367145303, corrected 6368905569): the checker counts a tree with a matching sha as bound.
- Box test env: `pytest tests` has 10 environmental failures in tests/casefold/test_source_isolation_casefold.py on clean main. They are unrelated to this lane.
