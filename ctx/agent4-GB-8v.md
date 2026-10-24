# ctx: agent 4 — GB-8v (RETRO-MATH-BIND-02)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8v per CoS assignment main#475/6369972075 (PICKUP 539/6372635580); unit list from `docs/retro/RETRO-MATH-BIND-01.slices.json` (#527, merged as a258693a). Same format as GB-8d (#536).
- Data file: `docs/retro/GB-8v.agent4.node-candidates.json` (proposal format, **not** the v2 bindings file). `architecture/graph-artifact-bindings.v2.json` is unchanged.
- No v2 record is added: none of these units has a node in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes; 0 reference these paths), and the v2 bindings file at main a487cfd4 mentions none of them. GB-5 (#522, e1d1ebc2) Q6.
- Pins: Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`. Trees are pinned by tree sha, files by blob sha, every file with sha256. Implementation = code files in the unit plus workflows at the cut that mention the unit path. First-add and last-change commits are provenance only. Nothing tracks a branch tip.
- `proposed_node_id` values (`cand.` prefix) are proposals derived from the path, not existing IDs.
- `referenced_unit_paths`: other unit paths a unit's files mention (regex scan, textual only). A navigation hint, not a verified relation.

## Units (13)
| unit | Math- kind | sha @cut | files | impl | impl status | other repo (main) | proposed node |
|---|---|---|---|---|---|---|---|
| `reviews/iba2_012_log_cap_concentration_20261006` | tree | `8989ea8f5cf0` | 5 | 2 | PINNED | n/a | `cand.reviews.iba2-012-log-cap-concentration` |
| `reviews/iba2_012_log_cap_transfer_20261006` | tree | `5aaee06c44e4` | 5 | 2 | PINNED | n/a | `cand.reviews.iba2-012-log-cap-transfer` |
| `reviews/iba2_012_marked_sector_20261006` | tree | `3ad7a7f6e536` | 5 | 2 | PINNED | n/a | `cand.reviews.iba2-012-marked-sector` |
| `reviews/literature_classical_neighbors_20261001` | tree | `6b2ad5b3e3ee` | 5 | 3 | PINNED | n/a | `cand.reviews.literature-classical-neighbors` |
| `reviews/p15_full_price_nonauthor_20260926` | tree | `ea65ded23d30` | 2 | 2 | PINNED | n/a | `cand.reviews.p15-full-price-nonauthor` |
| `reviews/p15_overlap_tail_claude_20260928` | tree | `a20b7d90e640` | 4 | 2 | PINNED | n/a | `cand.reviews.p15-overlap-tail-claude` |
| `reviews/pr125_shrinking_witness_nonauthor_20260926` | tree | `e7909e85eb2d` | 2 | 1 | PINNED | n/a | `cand.reviews.pr125-shrinking-witness-nonauthor` |
| `reviews/pr16_fixed_transverse_nonauthor_20260925` | tree | `fc3fa1d11cd9` | 2 | 2 | PINNED | n/a | `cand.reviews.pr16-fixed-transverse-nonauthor` |
| `reviews/pr25_contact_kernel_20260925` | tree | `7b6969b09e0a` | 2 | 1 | PINNED | n/a | `cand.reviews.pr25-contact-kernel` |
| `reviews/pr25_typed_transfer_nonauthor_20260925` | tree | `d47987731096` | 2 | 1 | PINNED | n/a | `cand.reviews.pr25-typed-transfer-nonauthor` |
| `reviews/pr36_kernel_tail_nonauthor_20260926` | tree | `58b3ef399aac` | 2 | 1 | PINNED | n/a | `cand.reviews.pr36-kernel-tail-nonauthor` |
| `reviews/proof_dependencies_20261008` | tree | `eace54863660` | 16 | 1 | PINNED | n/a | `cand.reviews.proof-dependencies` |
| `reviews/radial_moment_corollaries_20260930` | tree | `4713c3a0b88b` | 17 | 2 | PINNED | n/a | `cand.reviews.radial-moment-corollaries` |

Status: 13 pinned in Math-; 0 v2 records added; all unbound_artifact; 0 existing-node matches.

## Missing evidence (explicit)
- Every unit has at least one implementation pin (code file or referencing workflow).
- Whether the referencing workflows ran green at the cut: NOT EXECUTED.

## Cross-repo / overlap
- No unit is in another GB-8 lane's list. I did not scan Math- open PRs for Codex-stalled overlap (NOT EXECUTED); none was flagged on the board for these paths.

## Drive / Dropbox
- Drive (read-only title search): `iba2`, `proof dependencies`, `radial moment`. 1 leads. LEADS_FOUND_NOT_IMPORTED: 1 title-only lead (radial moment); its contents were not opened and it names no lane unit path, so nothing is bound or copied under imports/historical/GB-8v/. Other search hits without a title match were discarded. Mapping it is DRV-H1..H3 work. Drive file IDs are WITHHELD (OP-PRIVACY) per CoS 539/6372537095; titles kept.
  - `WITHHELD (OP-PRIVACY)` REVIEWED-LIFETIME-RADIAL-MOMENT-CONSUMERS-20260930-C28.zip (2026-09-30T09:47:59Z)
- Dropbox: no access (no connector), not worked around.

## NOT EXECUTED
- Math content review of any unit; Lean build; per-unit workflow run history; Drive full-text search; GRAPH.json node creation.

Integrator: agent 5 (non-author). Merge commit only. No self-merge. Respect any active CoS main merge hold.
