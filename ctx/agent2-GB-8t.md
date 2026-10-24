# ctx: agent 2 — GB-8t (RETRO-MATH-BIND-02)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8t per CoS assignment main#475/6369972075; unit list from `docs/retro/RETRO-MATH-BIND-01.slices.json` on main#527 (head e755231e).
- Data file: `docs/retro/GB-8t.agent2.node-candidates.json` (proposal format, **not** the v2 bindings file).
- Per the inventory note 6369870497 and GB-5 (#522, e1d1ebc2) Q6, no v2 record is added: none of the 13 units has a node in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes; 0 reference these paths). `architecture/graph-artifact-bindings.v2.json` is unchanged.
- Each unit is pinned as `kind: tree` at the Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`, with the tree sha plus a per-file blob sha and sha256. Implementation files are code inside the unit plus workflows at the cut that mention the unit path, each pinned by blob and sha256. First-add and last-change commits are recorded as provenance.
- `proposed_node_id` values (prefix `cand.`) are **proposals derived from the path**. They aren't in the graph, and creating them is a separate reviewed GRAPH.json change. They aren't claimed as existing IDs.

## Units (13)
| unit | tree sha @9f3fb691 | files | impl pins | proposed node | Drive leads |
|---|---|---|---|---|---|
| `frontiers/short_bar_endpoint_law_20261001` | `b84ebbbb55c010767187b5797bcc4e66eba4626d` | 1 | 2 | `cand.frontiers.short-bar-endpoint-law` | 0 |
| `frontiers/source_smooth_realization_20261011` | `90153c9da6e9ac03960d6a7d3cc394eb988d94f8` | 21 | 15 | `cand.frontiers.source-smooth-realization` | 0 |
| `frontiers/spectral_cluster_closure_20260929` | `cd60823ee3670fd2a0c0a79c13330200fc382f23` | 16 | 11 | `cand.frontiers.spectral-cluster-closure` | 0 |
| `frontiers/subpin_collision_20261002` | `29965e0ea0c90a941356ea0440affda2122f4618` | 5 | 1 | `cand.frontiers.subpin-collision` | 1 |
| `frontiers/two_scale_cluster_geometry_20260929` | `e09d618cfa520fe07dd085c1a487e9d798fbc535` | 14 | 9 | `cand.frontiers.two-scale-cluster-geometry` | 0 |
| `frontiers/unique_replacement_bar_intensity_20261001` | `fddc85298c8121b05d28690c84f0dcf1ddcf1251` | 6 | 3 | `cand.frontiers.unique-replacement-bar-intensity` | 0 |
| `frontiers/unrestricted_selection_difference_20260929` | `59260ecda2b9aaf321b5504a1954131d512bb353` | 4 | 4 | `cand.frontiers.unrestricted-selection-difference` | 0 |
| `frontiers/window_multiplicity_laws_20260928` | `88d63cc679305641a8b8901bc1fbcad53fba2c42` | 10 | 2 | `cand.frontiers.window-multiplicity-laws` | 0 |
| `imports/gp_data_114_embedded_20260930` | `7eca4729b091d16ca033a4da08167a2e6d37950c` | 10 | 6 | `cand.imports.gp-data-114-embedded` | 3 |
| `imports/hardening_ebedb780` | `5f678eb5338f4c7b9613952cac8925b83e20831d` | 13 | 2 | `cand.imports.hardening-ebedb780` | 0 |
| `imports/transverse_contact_library_20260927` | `34abbbdef0a144849eec53dfe2485b84d670ff6b` | 13 | 2 | `cand.imports.transverse-contact-library` | 0 |
| `imports/upper2d_h5_ledgers_20260926` | `43b7a72b12d6e16352862181aea84cdbb9f72d74` | 55 | 2 | `cand.imports.upper2d-h5-ledgers` | 1 |
| `imports/upper2d_stage_e_20260926` | `97d505ba459443599a1d1afffb3ff27b9d5149da` | 17 | 11 | `cand.imports.upper2d-stage-e` | 0 |

Status: 13 tree-pinned; 0 absent; 0 v2 records added; 13 unbound_artifact; 0 existing-node matches.

## Drive / Dropbox
- Drive (read-only search) queries: upper2d (title), GP-DATA-114 (title), source smooth realization (title, 0 hits). Title-matching leads are listed as `external_unverified`. Their bytes were **not fetched** and **not imported** under `imports/historical/GB-8t/` (NOT EXECUTED). They are never bound.
- Dropbox: no access (no connector), and I didn't work around that.

## Checks
- `tools/graph_binding_check.py --report-only --v2 architecture/graph-artifact-bindings.v2.json --repo d6g8k5htny-coder/Math-=<checkout @9f3fb691>` on main a487cfd4: bound 21, nodes_not_bound 28, stale 0, missing 0, unknown_node 0. This is unchanged from the CoS baseline, as expected for a docs-only proposal.
- Related tests (test_graph_artifact_bindings_v2, test_graph_binding_check, test_architecture_binding, test_dependency_source_binding, test_architecture_graph_artifact_readback): 115 passed, 1 skipped. The full suite result is in the PR body.

## NOT EXECUTED
- Drive byte fetch and import; per-unit Drive queries beyond the lane keywords; math content review of any unit; manifest re-verification (left to the owner lanes); checking the Codex-stalled open PRs for further dirs in these families.

Integrator: agent 3 (non-author). Merge commit only. No self-merge.

## Checks (box, CPython 3)
- `tools/graph_binding_check.py --root . --v2 architecture/graph-artifact-bindings.v2.json --report-only` (with Math- checkout): exit 0 on main `09ae8fd3`. Result: bound 24, nodes_not_bound 25, stale 0, missing 0, unknown_node 0. Same as main, because this PR doesn't touch the v2 file.
- `pytest -k "binding or architecture or graph"`: 359 passed, 6 skipped (on `09ae8fd3`). Full suite (earlier, on a487cfd4): 1249 passed, 11 skipped, 10 failed. All 10 failures are in `tests/casefold` and reproduce on origin/main in this box, so they're environmental and existed before this change.
