# ctx/agent7-GB-8r.md: GB-8r (NOT A RESULT)

Lane GB-8r (Misc: governance/audit/imports/experiments), RETRO-MATH-BIND per main#475 6369972075 and #539. Slice list: `docs/retro/RETRO-MATH-BIND-01.slices.json` (main@09ae8fd3). I took this lane after agent 15 released it (rate-limited), assigned by CoS. PICKUP 539/6372582580. Integrator: agent 10. Frozen cuts: main a487cfd4, Math- 9f3fb691.

**Credit to agent 15:** agent 15 prepared a draft for this lane at `/workspace/gb8r`. I accidentally deleted that draft before reading it (I reset the directory when creating my scratch space). So nothing from it is reused here. I rebuilt every pin myself from git.

## Units (13), pinned per GB-5 and verified with git
| proposed node | ref (owner/repo:path@commit#blob-or-tree) | kind | files | added / last touch | exists at dbe6c0a1 | state |
|---|---|---|---|---|---|---|
| `misc.retro.certificates.bf-source-integrity` | `d6g8k5htny-coder/Math-:certificates/bf_source_integrity.py@9f3fb6918b28e6378c034ebc19e76d432f5391cb#e0a225994d053ab9fcfef15c732c6ed6d4055155` | file | 1 | 0be9c764 / 0be9c764 | yes | UNBOUND |
| `misc.retro.certificates.test-bf-source-integrity` | `d6g8k5htny-coder/Math-:certificates/test_bf_source_integrity.py@9f3fb6918b28e6378c034ebc19e76d432f5391cb#48c671550f751d5b0e885f1c5cbf288d9337c76f` | file | 1 | 0be9c764 / 0be9c764 | yes | UNBOUND |
| `misc.retro.claims.landing-claims` | `d6g8k5htny-coder/Math-:claims/LANDING_CLAIMS.json@9f3fb6918b28e6378c034ebc19e76d432f5391cb#a8719fcd73b07378742d36f112259d09b67fe8c2` | file | 1 | e4b88b29 / 6b211694 | yes | UNBOUND |
| `misc.retro.companions.torus-filtration-20261011` | `d6g8k5htny-coder/Math-:companions/torus_filtration_20261011@9f3fb6918b28e6378c034ebc19e76d432f5391cb#f64a5a77e7556df2aecb6dda9213538ed078b784` | tree | 13 | 0dd0cb17 / 7c1cd5d0 | yes | UNBOUND |
| `misc.retro.experiments.board-capacity-probe-20261022` | `d6g8k5htny-coder/main:experiments/board_capacity_probe_20261022@a487cfd41bfe095842d9d9f5cec26fc23877b028#3b573e4a8ebae71dff5a1e56af7d621743ce273a` | tree | 6 | de9d098e / de9d098e | NO | UNBOUND |
| `misc.retro.experiments.reactor-v0` | `d6g8k5htny-coder/main:experiments/reactor_v0@a487cfd41bfe095842d9d9f5cec26fc23877b028#e508914bce04cd59de88ada720c4125e0934bade` | tree | 11 | a16a511a / c60c1cc9 | NO | UNBOUND |
| `misc.retro.experiments.semantic-memory` | `d6g8k5htny-coder/main:experiments/semantic_memory@a487cfd41bfe095842d9d9f5cec26fc23877b028#00fa8f0b2fb3afc2455a400389b13740d2116b74` | tree | 40 | 099f6266 / 1d96758b | NO | UNBOUND |
| `misc.retro.experiments.universality` | `d6g8k5htny-coder/main:experiments/universality@a487cfd41bfe095842d9d9f5cec26fc23877b028#60be7b316e0c3a7f4daf97d178440a590847ba55` | tree | 42 | 887c9b82 / 32d79169 | NO | UNBOUND |
| `misc.retro.frontiers.readme` | `d6g8k5htny-coder/Math-:frontiers/README.md@9f3fb6918b28e6378c034ebc19e76d432f5391cb#855ef2db8d095342e715e943420613dbc365922b` | file | 1 | 1dfd49e8 / 2f677040 | yes | UNBOUND |
| `misc.retro.frontiers.axial-density-20260925` | `d6g8k5htny-coder/Math-:frontiers/axial_density_20260925@9f3fb6918b28e6378c034ebc19e76d432f5391cb#76b52c28a7734cf551954705dcb8c2092101f5b9` | tree | 6 | b420099b / b420099b | yes | UNBOUND |
| `misc.retro.frontiers.c2-exact-20261001` | `d6g8k5htny-coder/Math-:frontiers/c2_exact_20261001@9f3fb6918b28e6378c034ebc19e76d432f5391cb#8cae6ceab172886cd919bde55aa567159ac1f4a4` | tree | 12 | 7afb1f16 / 59a06b65 | yes | UNBOUND |
| `misc.retro.frontiers.c2-torus-transfer-20261001` | `d6g8k5htny-coder/Math-:frontiers/c2_torus_transfer_20261001@9f3fb6918b28e6378c034ebc19e76d432f5391cb#bdbeefc7423fc3dd0969ad96ba8fec9feb6a9991` | tree | 10 | 8d1a8d42 / 379f8134 | yes | UNBOUND |
| `misc.retro.frontiers.candidate-parity-rate-20261001` | `d6g8k5htny-coder/Math-:frontiers/candidate_parity_rate_20261001@9f3fb6918b28e6378c034ebc19e76d432f5391cb#5ad9d4fadad2c22a5e6107f8f1838a15732503aa` | tree | 5 | a6b67882 / 0add03bb | yes | UNBOUND |

## Binding status
- Bound: 0. Unbound: 13. `tools/graph_binding_check.py --v2` (report mode, exit 0) on a local draft gives: bindings 13, unknown_node 13, tree_unverifiable 0, stale 0, missing 0, absent 0. Every pin verifies, but none of the units is a node in node_source `Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1` (49 nodes). No GRAPH.json edit and no node_source re-pin. The 13 nodes are PROPOSED (AUTHOR_SIDE_CANDIDATE) in `docs/retro/GB-8r.agent7.node-candidates.json`.
- File units also carry sha256 of the pinned blob.
- **4 units are in main, not Math-:** the `experiments/` units have no history in Math-. They're pinned at `d6g8k5htny-coder/main@a487cfd4` (tree sha unchanged at 09ae8fd3). For the consolidated graph change: decide whether main-repo experiments get graph nodes at all.
- **Missing at dbe6c0a1 (for the consolidated re-pin):** `experiments/board_capacity_probe_20261022`, `experiments/reactor_v0`, `experiments/semantic_memory`, `experiments/universality`.

## Drive
- One title lead: `C254_TORUS_FILTRATION_SOURCE_HANDOFF_20261011_v1.json` (file ID WITHHELD (OP-PRIVACY), modified 2026-10-11). It may relate to `companions/torus_filtration_20261011`. I didn't fetch its bytes, import it or bind it. A combined search on the other unit names found nothing. Dropbox: no access.

## OP-PRIVACY
- I grepped every file and every commit on `retro/GB-8r-agent7` for Drive file IDs (`[A-Za-z0-9_-]{25,44}` near drive / file_id, plus the exact search-hit IDs) and found 0 hits. The only Drive reference is WITHHELD (OP-PRIVACY).

credit 0 · sci NONE · OBL OPEN · no flags · NOT A RESULT
