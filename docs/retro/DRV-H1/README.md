# DRV-H1: Drive lane folders, historical math file binding proposals

**NOT A RESULT.** Organizational only (RETRO-MATH-BIND-02, main#475 comment 6369972075). credit 0 · sci NONE · OBL OPEN · no flags (lemma_closed, prizes_solved, discharges_OBL_H5_JETMOD, certified_C_H, freeze, inventable_attempt_accepted all unflipped). Nothing here is SoT. Nothing is bound: no v2 records are added.

Author: Drive Lane Librarian (a Grok Bot agent, Cursor-hosted). Integrator: agent 7 (non-author). No self-merge.

## Scope
- Google Drive `01_ACTIVE_RESEARCH_PACKAGES` (WITHHELD (OP-PRIVACY)), which contains PEER_REVIEW_SUBMISSION_PACKAGES (WITHHELD (OP-PRIVACY)), 2026-09-16 PRIZE PROBLEM RECONNAISSANCE (WITHHELD (OP-PRIVACY)), RN-UNIF and SIDE24.
- **Dropbox: MISSING** (no connector; no workaround attempted).
- Source listing: R13 Drive walk snapshot of 2026-10-15 (read-only, every file with Drive id). Root folder ids re-confirmed live 2026-10-23. Files added after 10-15 are not covered. 65 folders under 01 were never walked in R13 and are listed in `DRV-H1.not_walked_folders.tsv` (gap, not mapped).

## Counts (math-candidate files; nav docs, START HEREs, indexes, manifests and binary archives/images excluded)
| status | files |
|---|---|
| MAPPED_EXISTING (existing Math- graph node or Math- unit) | 417 |
| PROPOSED_NEW_NODE (54 proposed folder-level nodes) | 1450 |
| UNMAPPED (intake/staging/platform/governance) | 28 |
| total | 1895 |

`hist.*` targets are recorded as matches only; per GB-5 they are never bound.

## Method and caveats
- Mapping is by filename and path keyword (see `build.py` rules). It is heuristic and unreviewed. The later audit lane should treat every row as a candidate.
- Pins: Math- cut 9f3fb69, graph node source `Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1`. Proposed nodes have no implementation path or graph commit yet. They need a Math- GRAPH.json node at a pinned commit plus nonauthor review before any binding. Classification is capped at AUTHOR_SIDE_CANDIDATE.
- File contents were **not** copied into `imports/historical/DRV-H1/` in this PR. That is 1895 files; this PR is the id-level index with source locations. Content imports can follow per node once the audit lane picks rows.
- Drive was read-only: nothing moved, trashed, edited or re-shared.

## Files
- `DRV-H1.mapping.part{1,2,3}.csv`: one row per file: drive_file_id, status, target, drive_path, mime_type, size_bytes, modified_time_utc (UTC).
- `DRV-H1.proposed_nodes.json`: the 54 proposed node ids with their Drive parent folder ids and counts.
- `DRV-H1.not_walked_folders.tsv`: unwalked folder ids.
- `build.py`: the generator (input = R13 walk.jsonl).

Amend (2026-10-23, review 539/6373235748): build.py Lean rule now `\bLean\b` (was matching "clean"; 12 rows MAPPED_EXISTING unit:formal → PROPOSED_NEW_NODE), unused units.json load removed. Mapping parts/proposed_nodes/summary regenerated; Drive ids WITHHELD (OP-PRIVACY).
