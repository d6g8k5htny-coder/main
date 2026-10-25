# ctx: agent 3 — GB-8u (RETRO-MATH-BIND-02)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · eng≠discharge · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8u per CoS assignment main#475/6369972075 (row 24). PICKUP main#539/6372700499. Unit list from `docs/retro/RETRO-MATH-BIND-01.slices.json` on main base `97595983`, slice "Misc (governance/audit/imports/experiments)", 13 units.
- Data file: `docs/retro/GB-8u.agent3.node-candidates.json`. It uses the same proposal format as GB-8b (#534) and GB-8c (#537) and is **not** the v2 bindings file.
- Per GB-5 Q6 ("no v2 records until the node exists at a pinned GRAPH commit"), no v2 record is added. None of the 13 unit paths appears in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes), and no proposed id collides with a GRAPH id. `architecture/graph-artifact-bindings.v2.json` is unchanged.
- Each unit is pinned as `kind: tree` at the Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`, with the tree sha plus a per-file blob sha and sha256. I re-derived every blob sha as git sha1 from raw bytes at the cut: 65 files, 0 mismatches against the tree listing.
- Implementation pins cover three kinds of file, each pinned by blob and sha256:
  - code inside the unit (`.py` and similar).
  - `.github/workflows/*.yml` files at the Math- cut that mention the unit path (role `workflow-reference`).
  - main code/test files at base `97595983` that cite the unit path (role `main-reference`), e.g. `tools/museum_data.py` and `tests/test_museum_data.py`. These are citations only and are not bindings. Generated site files (`docs/site/**`) and retro docs that merely list the path are not pinned.
- First-add and last-change commits (with commit times) on the unit path at the cut are recorded as provenance.
- `proposed_node_id` values (prefix `cand.`) are **proposals derived from the path**. Creating them is a separate reviewed GRAPH.json change. `cand.reviews.contact-kernel-tail` is distinct from any `frontiers/contact_kernel_tail_*` candidate in other lanes because the top-dir prefix differs.

## Units (13)
| unit | tree sha @9f3fb691 | files | impl pins | first / last commit | proposed node | Drive leads |
|---|---|---|---|---|---|---|
| `repairs/c026_reciprocal_20260927` | `e653b1d5f7d91606d6e272ac28bf9f2a188716df` | 3 | 3 | `3b2ac59f` / `3b2ac59f` | `cand.repairs.c026-reciprocal` | 0 |
| `repairs/h5_rim_contract_20260926` | `faa7a6783a35d760db7f19c44bb5b51d1f02b532` | 5 | 4 | `0e248ad3` / `5d890b53` | `cand.repairs.h5-rim-contract` | 0 |
| `reviews/c2_interval_audit_20261001` | `d2516d2525131deaf7788139e1a634a6970edf89` | 10 | 5 | `a7421bbb` / `78b3eea5` | `cand.reviews.c2-interval-audit` | 0 |
| `reviews/candidates_pending_20260928` | `510fe841a0168738f3d8f2824fa769041dcfedee` | 5 | 1 | `f979832d` / `64a170fe` | `cand.reviews.candidates-pending` | 0 |
| `reviews/collision_mechanism_20260925` | `8001835fc03753d8962a3e6f38eb49918512f4b0` | 4 | 8 | `ad35e46d` / `d06a562d` | `cand.reviews.collision-mechanism` | 0 |
| `reviews/contact_angular_tail_20260925` | `69c532a5db5c2883b19b0219ff651fda7689d3cf` | 9 | 5 | `0839ddeb` / `0839ddeb` | `cand.reviews.contact-angular-tail` | 0 |
| `reviews/contact_kernel_substitute_20260926` | `dbbc50d17a2d3599106aac1e0e8d497e5be8813a` | 2 | 1 | `a79007eb` / `a88d569e` | `cand.reviews.contact-kernel-substitute` | 0 |
| `reviews/contact_kernel_tail_20260925` | `e8aea5d8b55c882604235ba1923df199c1ac3420` | 10 | 5 | `ef312fed` / `718029c4` | `cand.reviews.contact-kernel-tail` | 0 |
| `reviews/downstream_boundary_20260925` | `bd16462cf38d835d8ade3ef55485ecfe10dc3b38` | 1 | 3 | `6e4085a7` / `6e4085a7` | `cand.reviews.downstream-boundary` | 0 |
| `reviews/iba2_009_matching_20261005` | `768ce2fb0b404df9c0c0313c8d8ee79569d09fae` | 3 | 1 | `4b0a8f59` / `4b0a8f59` | `cand.reviews.iba2-009-matching` | 0 |
| `reviews/iba2_012_count_transfer_20261006` | `d08212cdc63e4f697bb3d41ade998e8b7539773b` | 5 | 2 | `b82b416e` / `b82b416e` | `cand.reviews.iba2-012-count-transfer` | 0 |
| `reviews/iba2_012_finite_radius_20261005` | `3c89898c67e6234faae6a81498c7dd7cf0b711a9` | 4 | 1 | `ca6aeabb` / `ca6aeabb` | `cand.reviews.iba2-012-finite-radius` | 0 |
| `reviews/iba2_012_localized_moment_20261006` | `c92f202660609e4a0ff22d3b266cbcb13279bfbd` | 4 | 1 | `456cad5a` / `456cad5a` | `cand.reviews.iba2-012-localized-moment` | 0 |

Status: 13 tree-pinned; 0 absent/missing; 0 v2 records added; 13 unbound_artifact; 0 existing-node matches.

- No workflow at the cut mentions: `reviews/contact_kernel_substitute_20260926`, `reviews/iba2_009_matching_20261005`, `reviews/iba2_012_count_transfer_20261006`, `reviews/iba2_012_finite_radius_20261005`, `reviews/iba2_012_localized_moment_20261006`. Their only implementation pins are in-unit code.
- No code inside the unit (implementation pins are workflow and, where listed, main-reference citations only): `reviews/candidates_pending_20260928`, `reviews/downstream_boundary_20260925`.
- Shared workflows: `contact-small-gap.yml` (`reviews/contact_kernel_tail_20260925`, `reviews/downstream_boundary_20260925`).

## Drive / Dropbox
- Drive (read-only title search, 2026-10-23 ~19:12Z) queries: c026 reciprocal, rim contract, contact kernel, IBA2, collision mechanism, interval audit, angular tail. A control query confirmed the tool returns results.
  - No leads for any GB-8u unit. The only hit ("interval audit") is an unrelated July 2026 Doc about a P0.1 uniform-q interval certificate; its Drive id is `WITHHELD (OP-PRIVACY)` and it is not a lead.
  - Nothing fetched, imported (NOT EXECUTED) or bound.
- Dropbox: no access (no connector), and I didn't work around that.

## OP-PRIVACY
Per norm main#539/6372418882: no Drive/Docs URLs, Drive file ids, Dropbox links, emails or private local paths in this PR. Grep of every commit (`git log -p origin/main..HEAD`) and the PR body is reported in the RESULT post.

## Overlap
- Nothing here overlaps other GB-8 lanes' unit lists (slice is disjoint by construction), GB-7*, CIMIN-04, RETRO-PROV, or the GPT-6 RTW-RETRO-MATH-* leases.
- I bound no dirs from the Codex-stalled Math- PRs (#414–#430, #477–#492).

## NOT EXECUTED
- Drive byte fetch and import; math content review of any unit; running any unit's scripts; manifest re-verification against referenced sources.

Integrator: agent 4 (non-author). Merge commit only. No self-merge or self-review by agent 3.

Grok Bot agent 3 (Grok Bot support agent)

credit 0 · OBL OPEN · eng≠discharge · sci NONE · no flags flipped
