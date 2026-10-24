# ctx: agent 14, GB-8n (organizational, NOT A RESULT)

Lane GB-8n of RETRO-MATH-BIND-01 (assignment main#475 comment 6369972075; inventory in PR #527).
Region: Sard / planar / fold / jet / third-order. 15 units. Integrator: agent 16 (not author).

## What this PR does
- Adds `docs/retro/GB-8n-agent14.proposal.json`: one node-candidate proposal per unit (proposed node id, source file, implementation paths), each pinned `owner/repo:path@commit#blob` with sha256, plus the unit tree sha.
- Pins are at the frozen cuts: main `a487cfd4`, Math- `9f3fb691`. Nothing tracks a branch tip.
- **No v2 binding records are added.** Per RETRO-MATH-BIND-01 (d), "No v2 records until the node exists at a pinned GRAPH commit." None of these 15 node ids exist in GRAPH.json at dbe6c0a1 (the v2 node_source) or at 9f3fb691, and no existing node references these paths. So all 15 stay **unbound**, reason: the node does not exist at a pinned GRAPH commit yet. Adding the nodes is a Math-side GRAPH.json change and needs its own nonauthor review.
- Classification is capped: `math.*` → AUTHOR_SIDE_CANDIDATE, `eng.*` (the main/experiments units) → ENGINEERING_CONTROL. Credit 0, OBL OPEN, no flags, no status promotion.

## Units
| Unit | Proposed node | Class | Tree @cut | Source pin | Impl pins |
|---|---|---|---|---|---|
| `experiments/periodic_h0` | `eng.periodic-h0` | ENGINEERING_CONTROL | `337cf6479b7a` (117 files) | `d6g8k5htny-coder/main:experiments/periodic_h0/README.md@a487cfd4#3c7f747360e7` | 4 |
| `experiments/zeta_fold_exploration` | `eng.zeta-fold-exploration` | ENGINEERING_CONTROL | `60538dd7a097` (117 files) | `d6g8k5htny-coder/main:experiments/zeta_fold_exploration/CLAIMS.md@a487cfd4#cd9362f79f17` | 3 |
| `frontiers/c2_finite_jet_transfer_20261001` | `math.c2-finite-jet-transfer` | AUTHOR_SIDE_CANDIDATE | `3e57957a81d1` (10 files) | `d6g8k5htny-coder/Math-:frontiers/c2_finite_jet_transfer_20261001/PROOF.md@9f3fb691#54cc4a1ae274` | 4 |
| `frontiers/c8_normalizer_floor_planar_20260930` | `math.c8-normalizer-floor-planar` | AUTHOR_SIDE_CANDIDATE | `f734b522b747` (6 files) | `d6g8k5htny-coder/Math-:frontiers/c8_normalizer_floor_planar_20260930/NOTE.md@9f3fb691#d321d52e83e9` | 2 |
| `frontiers/c8_theorem_a_constants_planar_20260930` | `math.c8-theorem-a-constants-planar` | AUTHOR_SIDE_CANDIDATE | `95d1459bc55e` (6 files) | `d6g8k5htny-coder/Math-:frontiers/c8_theorem_a_constants_planar_20260930/NOTE.md@9f3fb691#49ebecf43c0e` | 2 |
| `frontiers/candidate_third_order_20261001` | `math.candidate-third-order` | AUTHOR_SIDE_CANDIDATE | `790cbb8ba316` (5 files) | `d6g8k5htny-coder/Math-:frontiers/candidate_third_order_20261001/PROOF.md@9f3fb691#70ca57ef0519` | 3 |
| `frontiers/equal_height_mass_value_20261001` | `math.equal-height-mass-value` | AUTHOR_SIDE_CANDIDATE | `313a0cc3610a` (7 files) | `d6g8k5htny-coder/Math-:frontiers/equal_height_mass_value_20261001/NOTE.md@9f3fb691#95d622c88de2` | 4 |
| `frontiers/planar_compact_k_chain_20261006` | `math.planar-compact-k-chain` | AUTHOR_SIDE_CANDIDATE | `dd21bd892a5b` (48 files) | `d6g8k5htny-coder/Math-:frontiers/planar_compact_k_chain_20261006/README.md@9f3fb691#bce18f372442` | 4 |
| `frontiers/planar_cubic_cluster_20260929` | `math.planar-cubic-cluster` | AUTHOR_SIDE_CANDIDATE | `69069bce69f5` (12 files) | `d6g8k5htny-coder/Math-:frontiers/planar_cubic_cluster_20260929/PROOF.md@9f3fb691#bb446d08db8a` | 4 |
| `frontiers/planar_rejected_endpoint_margin_20261003` | `math.planar-rejected-endpoint-margin` | AUTHOR_SIDE_CANDIDATE | `8e52b7d9223d` (13 files) | `d6g8k5htny-coder/Math-:frontiers/planar_rejected_endpoint_margin_20261003/README.md@9f3fb691#0cb64d1d8d6c` | 3 |
| `frontiers/planar_soft_layer_chain_20261003` | `math.planar-soft-layer-chain` | AUTHOR_SIDE_CANDIDATE | `6abbf06fe3d7` (122 files) | `d6g8k5htny-coder/Math-:frontiers/planar_soft_layer_chain_20261003/README.md@9f3fb691#f24b6697d1b6` | 4 |
| `frontiers/sard_g_a2_parameter_20260929` | `math.sard-g-a2-parameter` | AUTHOR_SIDE_CANDIDATE | `4734aebcbae2` (14 files) | `d6g8k5htny-coder/Math-:frontiers/sard_g_a2_parameter_20260929/PROOF.md@9f3fb691#36bbc0c30a45` | 4 |
| `frontiers/sard_g_a2_regularity_20260929` | `math.sard-g-a2-regularity` | AUTHOR_SIDE_CANDIDATE | `49020546a60b` (4 files) | `d6g8k5htny-coder/Math-:frontiers/sard_g_a2_regularity_20260929/PROOF.md@9f3fb691#8ea2c3537508` | 2 |
| `frontiers/sard_g_robust_charts_20260929` | `math.sard-g-robust-charts` | AUTHOR_SIDE_CANDIDATE | `50faccb4c2d6` (9 files) | `d6g8k5htny-coder/Math-:frontiers/sard_g_robust_charts_20260929/PROOF.md@9f3fb691#7e2237063a8d` | 3 |
| `frontiers/separated_fold_multiplicity_20261002` | `math.separated-fold-multiplicity` | AUTHOR_SIDE_CANDIDATE | `5af92f150002` (10 files) | `d6g8k5htny-coder/Math-:frontiers/separated_fold_multiplicity_20261002/PROOF.md@9f3fb691#026cf5237510` | 3 |

## Verification
- Every blob and sha256 was recomputed from `git show <cut>:<path>` and `git rev-parse <cut>:<path>`. Every tree sha came from `git ls-tree <cut>`. File counts at the cut match the inventory for all 15 units. All 15 inventory commits resolve. All impl paths exist at the cut.
- The checker (`tools/graph_binding_check.py --report-only`) gives the same results before and after: bound 21, unbound_nodes 16, nodes_not_bound 28, stale 0, missing 0, unknown_node 0. `tests/test_graph_artifact_bindings_v2.py`: 7 passed, before and after.
- The source file chosen for each unit is the top-level README/PROOF/NOTE/CLAIMS markdown. This is an organizational choice and the integrator or reviewer can change it.

## Context notes
- No Drive or historical-import finds were made for this lane, so `imports/historical/GB-8n/` is not created. Dropbox: no access (no connector), per the rules. Nothing was worked around.
- Codex-stalled Math- PRs: none were bound or touched here.
- Overlap check: no unit is in GB-7c/FC-10, GB-7d/e/f, or the GPT-6 RTW / FILE-OBJECTS leases.
- Agent 14 also holds integrator duties for GB-8a (agent 1) and GB-8m (agent 13). Those need nonauthor, merge-commit-only, SHA-guarded merges, with no self-merge. Note: section (d) of RETRO-MATH-BIND-01.md lists agent 2 as the GB-8a integrator. The later #475 table supersedes it.
- `docs/retro/` is created here because #527 is not merged at a487cfd4. If #527 merges first, this adds alongside it with no conflict.

credit 0 · sci NONE · eng≠discharge · OBL OPEN · NOT A RESULT
