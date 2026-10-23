# ctx: agent 9 — GB-8i (RETRO-MATH-BIND-02)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8i (D3 / coefficients, side24 + LM006), per CoS assignment main#475/6369972075. Unit list from `docs/retro/RETRO-MATH-BIND-01.slices.json` (blob `3c9f374d`, landed in main `a258693a` via #527). Board traffic continues on main#539.
- Data file: `docs/retro/GB-8i.agent9.node-candidates.json` (proposal format, same shape as GB-8b's #534, **not** the v2 bindings file).
- Per GB-5 (#522, e1d1ebc2) Q6, no v2 record is added: none of the 12 units appears in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes; checked by path search of the GRAPH.json bytes at that commit: 0 hits). `architecture/graph-artifact-bindings.v2.json` is unchanged.
- Each unit is pinned `kind: tree` at the unit's last-change commit with its tree sha; all 12 tree shas equal the tree at the Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb` (`tree_sha_matches_cut: true`). Each file carries git blob sha and sha256 read from the cut. Implementation pins are code inside the unit (`owner/repo:path@commit#blob` + sha256); workflow refs are the Math- workflows at the cut that name the unit dir. Nothing tracks a branch tip.
- `cand.*` ids are path-derived proposals, flagged PROPOSED_NOT_IN_NODE_SOURCE. Creating them is a separate reviewed Math- GRAPH.json change.

## Units (12)
| unit | tree sha | files | impl pins | workflows | proposed node |
|---|---|---|---|---|---|
| `frontiers/c8_window_coefficient_d6_20261001` | `e675cda0ee8437d74eee5e406c6b4705d6fe0b31` | 4 | 1 | 1 | `cand.frontiers.c8-window-coefficient-d6` |
| `frontiers/c8_window_coefficient_d7_20261001` | `6d3cd23892dd7cf723f5b54f343a1a7a74069778` | 4 | 1 | 1 | `cand.frontiers.c8-window-coefficient-d7` |
| `frontiers/c8_window_coefficient_planar_20260930` | `0bfa9cc94ae66951e3e70ed501e599565e4595cd` | 4 | 1 | 1 | `cand.frontiers.c8-window-coefficient-planar` |
| `frontiers/cusp_coefficient_certified_20261001` | `deeecf43f839b86f0f45da7d4aa813601f42f52b` | 16 | 10 | 4 | `cand.frontiers.cusp-coefficient-certified` |
| `frontiers/qs_d3_soft_layer_chain_20261005` | `e8790be272efb823f72bb1be14be6382598c4796` | 146 | 13 | 1 | `cand.frontiers.qs-d3-soft-layer-chain` |
| `frontiers/side24_periodization_20260929` | `7bde1b7d225da191762dd943b5e36c4336866a7d` | 4 | 1 | 2 | `cand.frontiers.side24-periodization` |
| `frontiers/strict_unique_replacement_coefficient_20261003` | `94ad7d85ad6095f4eca987cc63bafa4bdc2d8031` | 5 | 1 | 1 | `cand.frontiers.strict-unique-replacement-coefficient` |
| `frontiers/third_order_coefficient_20261001` | `d82a0e6453dd0aec432c3205ef1aa950f1620af5` | 5 | 1 | 2 | `cand.frontiers.third-order-coefficient` |
| `reviews/cap_i4_d3_spectral_adapter_20261005` | `36f9e833a0b332d8829a676aa7849102d43dde3b` | 6 | 3 | 1 | `cand.reviews.cap-i4-d3-spectral-adapter` |
| `reviews/lm006_common_regression_rate_20261006` | `16d32428cd97d3204aee18bafea96458b4d0ce30` | 4 | 2 | 0 | `cand.reviews.lm006-common-regression-rate` |
| `reviews/lm006_cubic_sign_band_20261006` | `3426a81e17b836b14da80c95785c8a359a02e231` | 4 | 2 | 0 | `cand.reviews.lm006-cubic-sign-band` |
| `reviews/lm006_endpoint_extension_20261005` | `34a3aa3f551c52d7ed992548cc39657a47dabdfa` | 4 | 2 | 0 | `cand.reviews.lm006-endpoint-extension` |

Status: 12 tree-pinned, 12 tree-matches-cut, 0 absent, 0 deleted-in-history files, 0 existing-node matches, 0 v2 records added, 12 unbound_artifact.
- The three `reviews/lm006_*` units have no workflow at the cut naming them; their only implementation pins are the in-unit scripts.

## Verification (box, 2026-10-23 ~17:35Z)
- Independent re-read of every pin against a local Math- clone: each unit tree sha at cut and at last commit, every file blob sha + sha256 at the cut, every implementation blob at its pinned commit. 12/12 OK, 0 mismatches.
- GRAPH.json @ dbe6c0a1: none of the 12 unit dir names occur in it.

## Overlap / leases
- No overlap with GB-7c (#530, `coefficients/side24_v1`), GPT-6 RTW-RETRO-MATH-EXECUTABLE-LINK-01 (6369972429 / 6370135751: side24_v1, third_order_rate, candidate_third_order, elder_third_order; `third_order_coefficient` here is a different dir), or the D3 source-identity addendum 527/6370174759 (`coefficients/` packets; my units are under frontiers/ and reviews/).
- Related open Math- PRs, not bound (pin rule: only landed bytes): #366, #372, #380, #385 (LM006 coefficient certificate line), #316 (D3 L4 replay). **#492** (elder cusp workflow hardening) is in the Codex-stalled range #477–#492 and touches the cusp family; noted only, left for an explicit non-Codex HANDOFF.

## Drive / Dropbox
- Drive read-only title search, queries: LM006, periodization, window coefficient. Six title-match leads are recorded in the JSON `drive.leads` as `external_unverified` (Drive file ids only). Bytes **not fetched**, **not imported** under `imports/historical/GB-8i/` (NOT EXECUTED). Never bound.
- Dropbox: no access (no connector), not worked around.

## NOT EXECUTED
- Math content review of any unit; running any unit script; Drive byte fetch/import; per-unit Drive queries beyond the lane keywords; checking Codex-stalled PR dirs.

## Process
- Author: Grok Bot agent 9. **Integrator: agent 10** (non-author), merge commit only, SHA-guarded. No self-merge, no self-review.
- An earlier attempt at this lane (prepared ~11:48am CT) stopped on request limits before anything was posted or pushed; this PR is the first publication.
