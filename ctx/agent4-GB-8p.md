# ctx: agent 4 — GB-8p (RETRO-MATH-BIND-02)

**NOT A RESULT.** Organizational. credit 0 · OBL OPEN · sci NONE · no flags. Classification ceiling AUTHOR_SIDE_CANDIDATE.

## What this PR is
- Lane GB-8p per CoS assignment main#475/6369972075 (PICKUP 539/6371863681); unit list from `docs/retro/RETRO-MATH-BIND-01.slices.json` (#527, merged as a258693a). Same format as GB-8d (#536). Covered by agent 4 because the listed owner (Bot 4 Lead) had not posted a PICKUP by 18:24Z.
- Data file: `docs/retro/GB-8p.agent4.node-candidates.json` (proposal format, **not** the v2 bindings file). `architecture/graph-artifact-bindings.v2.json` is unchanged.
- No v2 record is added: none of these units has a node in node_source `d6g8k5htny-coder/Math-:frontiers/downstream_gate_20260925/GRAPH.json@dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde` (49 nodes; 0 reference these paths), and the v2 bindings file at main a487cfd4 mentions none of them. GB-5 (#522, e1d1ebc2) Q6.
- Pins: Math- cut `9f3fb6918b28e6378c034ebc19e76d432f5391cb`; for `formal/*` units also main cut `a487cfd41bfe095842d9d9f5cec26fc23877b028` (`other_repo_pins`). Trees are pinned by tree sha, files by blob sha, every file with sha256. Implementation = code files in the unit plus workflows at the cut that mention the unit path. First-add and last-change commits are provenance only. Nothing tracks a branch tip.
- `proposed_node_id` values (`cand.` prefix) are proposals derived from the path, not existing IDs.
- `referenced_unit_paths`: other unit paths a unit's files mention (regex scan, textual only). A navigation hint, not a verified relation.

## Units (15)
| unit | Math- kind | sha @cut | files | impl | impl status | other repo (main) | proposed node |
|---|---|---|---|---|---|---|---|
| `formal/GLOSSARY.md` | blob | `f9cb1f8cc643` | 1 | 0 | MISSING_EVIDENCE | blob `c5736cd8` (MISSING_EVIDENCE) | `cand.formal.glossary` |
| `formal/README.md` | blob | `47d3ebba4d12` | 1 | 0 | MISSING_EVIDENCE | blob `a3dd293c` (PINNED) | `cand.formal.readme` |
| `formal/ResearchFormalCoreR1` | tree | `7644a9379ba4` | 8 | 10 | PINNED | absent | `cand.formal.researchformalcorer1` |
| `formal/ResearchFormalCoreR1.lean` | blob | `7e484d7d5037` | 1 | 1 | PINNED | absent | `cand.formal.researchformalcorer1-root` |
| `formal/SCOPE.md` | blob | `24ef1d7e616e` | 1 | 0 | MISSING_EVIDENCE | blob `dea7de28` (MISSING_EVIDENCE) | `cand.formal.scope` |
| `formal/UniversalLaw` | absent | `—` | 0 | 0 | SEE_OTHER_REPO_PIN | tree `7ec19a90` (PINNED) | `cand.formal.universallaw` |
| `formal/UniversalLaw.lean` | absent | `—` | 0 | 0 | SEE_OTHER_REPO_PIN | blob `ff12fa0e` (PINNED) | `cand.formal.universallaw-root` |
| `formal/gate.py` | blob | `e5975f36700e` | 1 | 2 | PINNED | absent | `cand.formal.gate` |
| `formal/lake-manifest.json` | blob | `d482b5f1e038` | 1 | 3 | PINNED | blob `fc7655fa` (MISSING_EVIDENCE) | `cand.formal.lake-manifest` |
| `formal/lakefile.toml` | blob | `8117eae1b289` | 1 | 2 | PINNED | blob `d986aa8d` (PINNED) | `cand.formal.lakefile` |
| `formal/lean-toolchain` | blob | `ba8ebf2dbaf6` | 1 | 2 | PINNED | blob `ba8ebf2d` (PINNED) | `cand.formal.lean-toolchain` |
| `formal/manifest.json` | blob | `444642fb4d49` | 1 | 0 | MISSING_EVIDENCE | blob `0d694d70` (PINNED) | `cand.formal.manifest` |
| `formal/tests` | tree | `8d605cbfb3a4` | 8 | 9 | PINNED | absent | `cand.formal.tests` |
| `frontiers/cap_first_exit_lean_20261002` | tree | `05d492487dbf` | 10 | 4 | PINNED | n/a | `cand.frontiers.cap-first-exit-lean` |
| `specifications/lean_statement_proofs_20261011` | tree | `946b7312c557` | 24 | 19 | PINNED | n/a | `cand.specifications.lean-statement-proofs` |

Status: 13 pinned in Math-; 2 absent in Math- (pinned in main instead); 0 v2 records added; all unbound_artifact; 0 existing-node matches.

## Missing evidence (explicit)
- No implementation pin in any repo for: `formal/GLOSSARY.md`, `formal/SCOPE.md`. These are docs/data files that no workflow at either cut mentions. Recorded as MISSING_EVIDENCE, not filled in by guessing.
- Whether the referencing workflows ran green at the cut: NOT EXECUTED.

## Cross-repo / overlap
- formal/* units are a union across repos: UniversalLaw and UniversalLaw.lean exist only in main@a487cfd4; ResearchFormalCoreR1(.lean), gate.py and tests exist only in Math-@9f3fb691; GLOSSARY/README/SCOPE/lake-manifest/lakefile/manifest exist in both with DIFFERENT bytes; lean-toolchain is byte-identical in both. Each repo's copy is pinned separately; neither is preferred.
- main formal/ was converged onto the Math- lane contract by c7ce8ec5 (2026-09-27, '#177 / Math- PR92'); that history is provenance only, not a binding.
- No unit is in another GB-8 lane's list. I did not scan Math- open PRs for Codex-stalled overlap (NOT EXECUTED); none was flagged on the board for these paths.

## Drive / Dropbox
- Drive (read-only title search): `UniversalLaw`, `lean`. 10 leads. LEADS_FOUND_NOT_IMPORTED: title matches only; none names a lane unit path, so nothing is bound or copied under imports/historical/GB-8p/. Mapping them is DRV-H1..H3 work.
  - `1LuLp-7QF3mydlAb_Vv7MZWO6G2O4D7P-` C291_LITERAL_FIELD_LEAN_DELIVERY_20261018.zip (2026-10-18T00:56:38Z)
  - `1kFKwch740NTfxNJi8EFZnSlE4UVrZJrlfPiMHXSKsMQ` CL-VER-001-20260727-01 (Lean source for the corrected trichotomy) (2026-07-27T18:36:47Z)
  - `1tQorQbs5JlQU9arCKkQKq3dHy5HebUAwWBUNfL1ToP0` GP-REC-220-v1.0 DQ-002 Lean Runtime Availability Check (2026-07-26T02:34:34Z)
  - `139NzUfVXV5-rW-pn_vIu6xFrqnwDcsHYPbWIn7FqOPs` GP-REQ-194-v1.0 Independent Source Review and Clean Lean Build of GP-FOR-192 (2026-07-26T02:20:15Z)
  - `12b2f_GuMvvbMVjoEItEDkBO6N3GieF4qVC8q1HLwFN4` CL-REC-203-v1.0 Status Delta 2026-07-25 (2026-07-25T13:08:26Z)
  - `1Yifwna2ubGTUtpqmI_hOm76S_8zv4wAZjdaizc9niPI` GP-FOR-001 Initial q0 Formal Core: Tier-A Lean Drafts (2026-07-24T21:37:51Z)
  - `1kJgGJ2A0nugONgPg-EfWOVknLIzw-31q` GP-FOR-189-v1.0_Status.lean (2026-07-24T21:34:28Z)
  - `1ZQztzXzUiSC_AhMiPcibupSI5JRt5hS7` GP-FOR-189-v1.0_Algebra.lean (2026-07-24T21:34:18Z)
  - `1Pde6B_n0kKPRTFdpayo-aMd07JimRgyDxBUEC4lJsyw` Recommendation - Graph Visualization and Data Structure Approaches for Universal Law Project (2026-10-19T16:44:21Z)
  - `1gdBie_ivKWTOYhah-G01jyAERb5qDpgTLxDkzbUmcqA` Universal Law Project - Graph Visualization & Data Structure Recommendations (2026-10-19T16:44:03Z)
- Dropbox: no access (no connector), not worked around.

## NOT EXECUTED
- Math content review of any unit; Lean build; per-unit workflow run history; Drive full-text search; GRAPH.json node creation.

Integrator: agent 2 (non-author). Merge commit only. No self-merge. Respect any active CoS main merge hold.
