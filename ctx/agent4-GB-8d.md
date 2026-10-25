# ctx: agent 4 — GB-8d (RETRO-MATH-BIND-02)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8d per CoS assignment main#475/6369972075 (PICKUP 6370598807); unit list from `docs/retro/RETRO-MATH-BIND-01.slices.json` on main#527 (head e755231e). Same format as GB-8b (#534).
- Data file: `docs/retro/GB-8d.agent4.node-candidates.json` (proposal format, **not** the v2 bindings file).
- No v2 record is added: none of the 12 units has a node in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes; 0 reference these paths). GB-5 (#522, e1d1ebc2) Q6. `architecture/graph-artifact-bindings.v2.json` is unchanged.
- Each unit is pinned as `kind: tree` at Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb` (tree sha + per-file blob sha + sha256). Implementation = code files inside the unit plus workflows at the cut that mention the unit path, each pinned by blob and sha256. First-add and last-change commits recorded as provenance.
- `proposed_node_id` values (`cand.` prefix) are proposals derived from the path, not existing IDs; creating them is a separate reviewed GRAPH.json change.
- `referenced_unit_paths`: other Math- unit paths each unit's files mention (regex scan, textual mention only). It's a navigation hint, not a verified review or consumption relation.

## Units (12)
| unit | tree sha @9f3fb691 | files | impl pins | proposed node | referenced units | implementation |
|---|---|---|---|---|---|---|
| `reviews/remote_distance_moments_claude_20260929` | `1df1cf0cb869d86f89076427dd6d2584a93bc00f` | 4 | 2 | `cand.reviews.remote-distance-moments-claude` | 4 | pinned |
| `reviews/remote_height_claude_20260928` | `af9cd19ac3a47740ebfddcc85856d7e909a7f2fc` | 4 | 2 | `cand.reviews.remote-height-claude` | 3 | pinned |
| `reviews/remote_height_marks_claude_20260929` | `68536332e8e89db35156ab9b4b77aeef875065dd` | 4 | 2 | `cand.reviews.remote-height-marks-claude` | 6 | pinned |
| `reviews/remote_inverse_separation_claude_20260929` | `5497e9268316b31c7c32d374b9d6f8345d25a1a3` | 4 | 2 | `cand.reviews.remote-inverse-separation-claude` | 4 | pinned |
| `reviews/remote_mixed_inverse_claude_20260929` | `2f932c1a09a24b67f7194212b5b57975ca2b64ae` | 4 | 2 | `cand.reviews.remote-mixed-inverse-claude` | 1 | pinned |
| `reviews/remote_pair_law_claude_20260928` | `c864d077f0956604052721fe454e5292c9ba0de3` | 4 | 2 | `cand.reviews.remote-pair-law-claude` | 4 | pinned |
| `frontiers/rn_mesoscopic_20260925` | `8e1ffe9ee63b6ac170cec62fb255daa79bfb94d6` | 2 | 0 | `cand.frontiers.rn-mesoscopic` | 0 | **MISSING** |
| `reviews/d5_c6_workflow_repair_20261005` | `4102da67c021a908a7955154c973478097e7a51b` | 1 | 0 | `cand.reviews.d5-c6-workflow-repair` | 3 | **MISSING** |
| `reviews/d5_consumption_bridge_20260928` | `7a9b0bb1f970bd72349b955ee4c4eced567c7ff6` | 4 | 0 | `cand.reviews.d5-consumption-bridge` | 3 | **MISSING** |
| `reviews/d5_dimension_lift_erratum_20261005` | `c20625f56b889f6071ee521fafbffecd22da94b9` | 1 | 0 | `cand.reviews.d5-dimension-lift-erratum` | 4 | **MISSING** |
| `reviews/d5_intermediate_window_grok_20260928` | `6cd958b9bd690f5cc46bf07111eea0f5c409d4d5` | 3 | 0 | `cand.reviews.d5-intermediate-window-grok` | 1 | **MISSING** |
| `reviews/d5_microdisk_20260926` | `6a193fc25b764238b9940fec09f42f3c96cfdff8` | 2 | 0 | `cand.reviews.d5-microdisk` | 0 | **MISSING** |

Status: 12 tree-pinned; 0 absent; 0 v2 records added; 12 unbound_artifact; 0 existing-node matches.

## Missing evidence (explicit)
- 6 units have **no implementation pin**: `frontiers/rn_mesoscopic_20260925`, `reviews/d5_c6_workflow_repair_20261005`, `reviews/d5_consumption_bridge_20260928`, `reviews/d5_dimension_lift_erratum_20261005`, `reviews/d5_intermediate_window_grok_20260928`, `reviews/d5_microdisk_20260926`. They contain no code file and no workflow at the cut mentions them. Recorded as `implementation_status: MISSING_EVIDENCE`. Not filled in by guessing.
- `frontiers/rn_mesoscopic_20260925` and `reviews/d5_microdisk_20260926` also mention no other unit path.
- The six `*_claude_*` review units each carry one checker script and one dedicated workflow; whether those workflows ran green at the cut is NOT EXECUTED.

## Overlap
- `reviews/d5_dimension_lift_erratum_20261005` and `reviews/d5_c6_workflow_repair_20261005` mention `frontiers/d5_dimension_lift_20260929`, a frontier unit outside this lane. I only tree-pin the review dirs and don't check that frontier unit.
- No unit is in GB-8c's list. No Codex-stalled PR touches these units as far as I saw.

## Drive / Dropbox
- Drive (read-only title search): `rn_mesoscopic`, `remote height`, `microdisk`. 0 leads. Not proof of absence. Nothing imported under `imports/historical/GB-8d/`.
- Dropbox: no access (no connector), not worked around.

## NOT EXECUTED
- Math content review of any unit; per-unit workflow run history; Drive full-text queries beyond the three above; GRAPH.json node creation.

Integrator: agent 5 (non-author). Merge commit only. No self-merge.
