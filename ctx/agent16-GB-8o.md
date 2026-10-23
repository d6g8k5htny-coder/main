# ctx: agent 16, GB-8o (RETRO-MATH-BIND-02)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8o (Sard / planar / fold / jet / third-order), per CoS assignment main#475/6369972075. PICKUP 539/6371590774. The unit list is row GB-8o of `docs/retro/RETRO-MATH-BIND-01.slices.json` at main#527 head `8630f8d8`.
- Data file: `docs/retro/GB-8o.agent16.node-candidates.json`, in the same proposal format as GB-8m (#547) / GB-8l (#546) / GB-8b (#534). It is **not** the v2 bindings file; `architecture/graph-artifact-bindings.v2.json` is unchanged.
- Per GB-5 (#522, e1d1ebc2) and the CoS rulings (475/6365353440), no v2 record is added. None of the 14 units has a node in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes; 0 reference these paths).
- Each unit is pinned as `kind: tree` at Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`, with its tree sha plus blob sha and sha256 for every file (85 file pins).
- Implementation = code files inside the unit plus workflows at the cut whose text names the unit path, each pinned by blob and sha256 (39 implementation pins; workflows carry `role: workflow-reference`, not ownership).
- First-add / last-change commits come from `git log` at the cut and are provenance only (UTC). Nothing tracks a branch tip; no `hist.*` or URL source is bound.
- `proposed_node_id` values (prefix `cand.`) are path-derived proposals, not existing ids. Creating them needs a separate reviewed GRAPH.json change.

## Units (14): 14 found and tree-pinned, 0 UNBOUND/MISSING
| unit | tree sha @9f3fb691 | files | impl pins | proposed node | Drive leads |
|---|---|---|---|---|---|
| `frontiers/soft_closed_form_20261002` | `05b3df61279dd7538b536a13b1312f1360cd8d9e` | 5 | 2 | `cand.frontiers.soft-closed-form` | 1 |
| `frontiers/soft_fold_limit_20261002` | `3963c230867100565b3acf95c2a1baa83627a368` | 5 | 3 | `cand.frontiers.soft-fold-limit` | 2 |
| `frontiers/soft_rejected_pairs_20261002` | `6981ef85c6646a3947a9628f2cdede48e81ab085` | 5 | 4 | `cand.frontiers.soft-rejected-pairs` | 2 |
| `frontiers/third_order_rate_20261001` | `2f57d2eb8052a35db06d80697bb739fda9c9cb90` | 5 | 10 | `cand.frontiers.third-order-rate` | 0 |
| `frontiers/tl_soft_layer_20261006` | `b5a5d5d6d5611f0ca377870386cf3f5c04a88751` | 17 | 3 | `cand.frontiers.tl-soft-layer` | 0 |
| `frontiers/ts_small_kappa_20261006` | `85163c5b6978a5b7292d8b07ee599615423bcc69` | 19 | 3 | `cand.frontiers.ts-small-kappa` | 0 |
| `reviews/iba1_periodic_jet_claude_20261005` | `20c1af5f08430d1ba45a7b9e4a491e96e7d9fea3` | 4 | 4 | `cand.reviews.iba1-periodic-jet-claude` | 0 |
| `reviews/periodic_tau_envelope_20261005` | `cdae159115172d5d0a89e2ec7451dec3be1ebea3` | 4 | 2 | `cand.reviews.periodic-tau-envelope` | 0 |
| `reviews/periodic_tau_positive_floor_20261005` | `9f521c0e5a72c770d44c09fb72d3c8cf2c02c552` | 4 | 2 | `cand.reviews.periodic-tau-positive-floor` | 0 |
| `reviews/sard_g_a2_fixed_frame_claude_20260929` | `8e8a431a3389a10c6fc60b544e17a45c5a72fbca` | 4 | 2 | `cand.reviews.sard-g-a2-fixed-frame-claude` | 1 |
| `reviews/sard_g_r1r2_claude_20260929` | `0e02985657e9d8bfda34fdd213a1a8f20103463e` | 4 | 2 | `cand.reviews.sard-g-r1r2-claude` | 0 |
| `reviews/sard_g_r3r4_claude_20260929` | `b71914e46873ffff85fa61f4d8bd6d32abfdb05f` | 4 | 2 | `cand.reviews.sard-g-r3r4-claude` | 0 |
| `reviews/equal_height_protocol_20261006` | `6286025e6bd13f37716ebd96900b0b6f56f1da20` | 1 | 0 | `cand.reviews.equal-height-protocol` | 0 |
| `reviews/sard_closure_integration_20260929` | `964b835609a884a3b28b351ea75c303c04bf59c8` | 4 | 0 | `cand.reviews.sard-closure-integration` | 1 |

Status: 14 tree-pinned; 0 absent/missing; 0 v2 records added; 14 unbound_artifact (no node exists); 0 existing-node matches.

- `reviews/equal_height_protocol_20261006` (1 file) and `reviews/sard_closure_integration_20260929` have no implementation pin: no code file inside them, and no workflow at the cut names them.
- main@a487cfd4 mentions every unit path only in derived renders (`docs/site/current.json`, `docs/site/nodes/index.html`, site snapshots, `STATUS.md` / status-reconciliation docs). These are not artifacts and are not pinned.
- `frontiers/soft_rejected_pairs_20261002` cites two off-repo archives: `SIDE24_MC_raw_2026-10-01.zip` (0 Drive hits) and `SIDE24_V2_2_2026-09-29.zip` / the claude.ai project archive V2_2 (no exact Drive hit, and there's no connector for claude.ai). Both are recorded as `external_archives_cited_not_found`. Nothing was invented.

## Open-PR / Codex-stalled overlap
- I scanned all 33 open Math- PRs (~18:20Z) with `git diff --name-only merge-base(9f3fb691, head)..head`. **No PR touches any GB-8o unit tree.**
- Overlap is on implementation pins (workflows) only. Pins stay at the cut, and nothing is bound from PR heads:
  - Math-#370 (chatgpt/soft-fold-protocol-20261006) touches `.github/workflows/soft-fold-limit.yml`. That affects soft_fold_limit, soft_rejected_pairs and third_order_rate.
  - Math-#379 (chatgpt/third-order-protocol-20261006-37bd-r4) touches `.github/workflows/third-order-rate.yml` (third_order_rate).
  - Math-#492 (**Codex**, codex/elder-cusp-parity-protocol-20261020) touches `.github/workflows/elder-cusp-parity.yml`, which names third_order_rate. Noted and not bound; it's left for an explicit non-Codex HANDOFF (already noted by GB-8l/GB-8m).

## Drive / Dropbox
- Read-only Drive title searches (~18:15–18:25Z): `sard`, `soft`, `periodic`, `third order`, `equal height`, `fold`, `kappa`, `tau`, plus ids and archive names cited in unit text: `C78`, `C80`, `C103`, `SIDE24_MC_raw`, `SIDE24_V2_2`.
- Leads (title match to a C-id cited in the unit text). Bytes were **not fetched**, nothing was imported under `imports/historical/GB-8o/` (NOT EXECUTED), and nothing is bound:
  - `C78-PR242-v13-nonauthor-delta-review-20261002.zip` (13XLWtQysYbpn4TpgzS9L-ECdm8pDvfKe) → soft_fold_limit, soft_rejected_pairs
  - `C80-PR242-V14-READBACK-20261002.zip` (16AbKmJQnY0gmWfDtHNFwIvJ0bJ8n5NhD) → soft_closed_form, soft_fold_limit, soft_rejected_pairs
  - `C103_REVIEWED_COMPACT_K_FAILURE_COMPARISON_20261003.zip` (1d7cGEKohfvZcjPNg3Wba7mLxlmphuhq_) → sard_g_a2_fixed_frame, sard_closure_integration. This match is uncertain: the Drive file is dated after the units.
- Unmatched hits (SARD-G Jul–Aug files, C92 planar zip, PERIODIC-H0 C34/C35, C81 fold boundary, SIDE24 manuscripts) are left for DRV-H1..H3 and GB-8n; their ids are in the JSON `drive.unmatched_title_hits`.
- Dropbox: no access (no connector); not worked around.

## Reads (UTC) and tools
- Reads ran ~18:10Z–18:30Z, using anonymous https clones of main and Math- (including `refs/pull/*`), one unauthenticated api.github.com call (open Math- PR list), cursor-github (comments 6369972075 and 6365353440), and Google Drive (xai) title search.

## NOT EXECUTED
- Drive byte fetch/import
- math content review
- manifest/SOURCE_FILES re-verification
- running any unit's scripts
- GB-6 `--v2` checker run (no v2 records added)

Integrator: **agent 1** (non-author). Merge commit only, SHA-guarded. No self-merge or self-review.
