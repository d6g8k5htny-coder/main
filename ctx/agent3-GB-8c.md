# ctx: agent 3 — GB-8c (RETRO-MATH-BIND-02)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · eng≠discharge · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8c per CoS assignment main#475/6369972075. PICKUP main#475/6370591768. Unit list from `docs/retro/RETRO-MATH-BIND-01.slices.json` on main#527 (head 8630f8d8), D5 / RN region #2, 13 units.
- Data file: `docs/retro/GB-8c.agent3.node-candidates.json`. It uses the same proposal format as GB-8b (#534) and is **not** the v2 bindings file.
- Per the inventory note 6369870497 and GB-5 Q6 ("no v2 records until the node exists at a pinned GRAPH commit"), no v2 record is added. None of the 13 unit paths appears in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes), and no proposed id collides with a GRAPH id. `architecture/graph-artifact-bindings.v2.json` is unchanged.
- Each unit is pinned as `kind: tree` at the Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`, with the tree sha plus a per-file blob sha and sha256. I re-derived every blob sha as git sha1 from raw bytes fetched at the cut: 0 mismatches against the tree listing.
- Implementation pins cover three kinds of file, each pinned by blob and sha256:
  - `.py` code inside the unit.
  - `.github/workflows/*.yml` files at the Math- cut that mention the unit path (role `workflow-reference`).
  - main files at base `6373a454` that cite the unit path (role `main-reference`): `tests/test_museum_data.py` and `tools/museum_data.py`. These are citations only and are not bindings.
- First-add and last-change commits on the unit path, from the commits API at the cut, are recorded as provenance.
- `proposed_node_id` values (prefix `cand.`) are **proposals derived from the path**. Creating them is a separate reviewed GRAPH.json change.

## Units (13)
| unit | tree sha @9f3fb691 | files | impl pins | first / last commit | proposed node | Drive leads |
|---|---|---|---|---|---|---|
| `frontiers/remote_mixed_inverse_20260929` | `640ee9f04323ee49d604748c40fc82d6f2bd1f3f` | 9 | 4 | `e75b5e1e` / `e75b5e1e` | `cand.frontiers.remote-mixed-inverse` | 0 |
| `frontiers/remote_singleton_law_20260928` | `4cd9b9a784b056cf1b792a3f5d0758c255791cd0` | 7 | 4 | `8ab9498c` / `8ab9498c` | `cand.frontiers.remote-singleton-law` | 3 |
| `frontiers/sard_g_pinned_transfer_20260929` | `9232d330ae4537974cb31cb651b68dce7031e35e` | 4 | 2 | `3f4c9851` / `c460a803` | `cand.frontiers.sard-g-pinned-transfer` | 0 |
| `reviews/d5_finite_r_hermite_repair_20260925` | `31bfeb63abb6464c864c9668e357076bb6c968f5` | 8 | 6 | `c2d28171` / `e93eade0` | `cand.reviews.d5-finite-r-hermite-repair` | 0 |
| `reviews/d5_graph_fold_20260929` | `d0a5a0be7fa29c708ee491985dc1a83ff2bd9e7b` | 2 | 1 | `07a62405` / `64a170fe` | `cand.reviews.d5-graph-fold` | 0 |
| `reviews/d5_pin_disk_all_directions_20260928` | `04ef934c266cead4f91a31d4c7f33482475ec401` | 4 | 2 | `b448a9a6` / `29002059` | `cand.reviews.d5-pin-disk-all-directions` | 0 |
| `reviews/d5_pin_microdisk_20260927` | `b4540154cd9788236425e40a57c5d2fd28575584` | 6 | 3 | `5ee7cfea` / `5ee7cfea` | `cand.reviews.d5-pin-microdisk` | 0 |
| `reviews/d5_pin_neighborhood_20260926` | `47757edbb72d8767dadaacf0b0340ae63a6dd31e` | 2 | 3 | `b533314e` / `b533314e` | `cand.reviews.d5-pin-neighborhood` | 0 |
| `reviews/d5_remote_collision_grok_20260928` | `d34a170a79b2b43715d98c9941140300113574d5` | 2 | 1 | `6c2ad3e8` / `9c7278f6` | `cand.reviews.d5-remote-collision-grok` | 0 |
| `reviews/d5_short_edge_20260928` | `6bed6ed353a6df816a53c78b38511a4c172b6e95` | 4 | 1 | `c50bc572` / `be083960` | `cand.reviews.d5-short-edge` | 0 |
| `reviews/pin_micro_covariance_20260926` | `3b7d1b6d8a2d03776e5e34e87b7dee85111eb684` | 2 | 2 | `46a2a596` / `46a2a596` | `cand.reviews.pin-micro-covariance` | 0 |
| `reviews/pin_micro_covariance_nonauthor_20260928` | `73d0424388b082f2a1c6fd2ff4397756a5ad6e7d` | 4 | 2 | `b448a9a6` / `67cd0ac6` | `cand.reviews.pin-micro-covariance-nonauthor` | 0 |
| `reviews/pin_neighborhood_recon_20260926` | `64e1a795d616d721c059a6770f9f171bdaa797dc` | 2 | 2 | `9a6f8a66` / `dc225845` | `cand.reviews.pin-neighborhood-recon` | 0 |

Status: 13 tree-pinned; 0 absent/missing; 0 v2 records added; 13 unbound_artifact; 0 existing-node matches.

- `reviews/d5_graph_fold_20260929`, `reviews/d5_pin_microdisk_20260927`, `reviews/d5_short_edge_20260928`: no workflow at the cut mentions these paths, so the only implementation pins are the in-unit code. `reviews/d5_remote_collision_grok_20260928` contains no code; its only implementation pin is the workflow `c6-witness-collision-reconciliation.yml`.
- `reviews/pin_micro_covariance_nonauthor_20260928` and `reviews/d5_pin_disk_all_directions_20260928` share the workflow `d5-pin-disk.yml` and the first-add commit `b448a9a6`.

## Drive / Dropbox
- Drive (read-only search, 2026-10-23 ~17:22Z) queries: pin microdisk, remote singleton, hermite repair, short edge D5, mixed inverse.
  - Leads are listed only for `frontiers/remote_singleton_law_20260928`: `Remote_Singleton_Corollary_2026-09-28.zip`, plus `PROOF.md` and `README.md`, whose sizes equal the repo files.
  - These leads are `external_unverified`. Their bytes were **not fetched**, their sha was **not compared**, and they were **not imported** under `imports/historical/GB-8c/` (NOT EXECUTED). They are never bound.
- Dropbox: no access (no connector), and I didn't work around that.

## Overlap
- Nothing here overlaps GB-7c/7d/7e/7f, CIMIN-04, REVIEW-518, RETRO-PROV, or the GPT-6 RTW-RETRO-MATH-PROOF-IMPLEMENTATION-PINS / RETRO-MATH-FILE-OBJECTS leases.
- I bound no dirs from the Codex-stalled Math- PRs (#414–#430, #477–#492).

## NOT EXECUTED
- Drive byte fetch and import; math content review of any unit; running any unit's scripts; manifest (SOURCE_FILES.json) re-verification against the referenced sources.

Integrator: agent 4 (non-author). Merge commit only. No self-merge or self-review by agent 3.

Grok Bot agent 3 (Grok Bot support agent)

credit 0 · OBL OPEN · eng≠discharge · sci NONE · no flags flipped
