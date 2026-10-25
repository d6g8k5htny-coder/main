# Dependency index: handoff source implementations joined to canonical records

This is the retroactive accounting the owner asked for: each mathematical record's source implementation, so the dependency work knows which artifact backs which record. Compiled by Anthropic Claude 01Cz on 25 October 2026 from the #605 audit results. It is data only. It copies statuses verbatim and changes none. Scientific effect NONE.

| File | What it is | Source |
|---|---|---|
| `D5_RECORD_SOURCE_INDEX.tsv` | 24 canonical records (23 with the D5 prefix, plus C6-PALM-Q) mapped to their direct source ID, source SHA-256, anchor lines and necessary premises, with the preserved acceptance, import, obligation and certificate statuses | Codex Capability Frontier v22 D5 accounting, main#605 comment 6400144048 |
| `V21_SUPPLEMENT_IMPLEMENTATION_INDEX.tsv` | The 14 v21 supplement members, which support the D5 collision cover as analytic candidates and are not canonical records. Each row gives the member's dependency scope, related checks and the Claude01Cz challenge result | Codex v21 map, main#605 comment 6399998554, and Claude01Cz challenge 6400100046 |
| `v22_D5_accounting_6400144048.json` | Exact JSON from that comment: 56,177 bytes, SHA-256 `6a5f6f2d0572cdb54d157c1345580d8f46bce96f552b95f05fbd3de641a8fa63` | as above |
| `v21_source_map_6399998554.json` | Exact JSON from that comment: 34,645 bytes, SHA-256 `7657e3f777a116271fe8b28cd81ec4bd76de338f3b4b5f74ade29327c102f1a5` | as above |

## Full coverage, added at 16:45Z

The canonical dataset now lives at branch `codex/605-v22-canonical-dataset-20261025`, commit `11f7b82e`, delivered by Capability Frontier in main#605 comment 6400298942. This lane rehashed it: 396,561 bytes, SHA-256 `4cfae309b9da989ca6c95881f207c0a097a4dde836f45eacbe0e78d7faf9e4d7`, matching exactly.

`ALL_104_RECORDS_SOURCE_INDEX.tsv` has one row per canonical record. Each row gives:
- topic, kind and title;
- preserved statuses, copied verbatim: project acceptance, obligation, certificate, import, and whether the source is authenticated;
- necessary premises (108 distinct links, matching the dataset) and external imports;
- the pinned source repository, commit and path;
- whether that source resolves in Git, with its bytes, SHA-256 and Git blob ID;
- whether a #605 source-accounting result covers the record.

`SOURCE_RESOLUTION_33.json` holds the fetch results for the 33 unique sources.

| Result | Count |
|---|---:|
| Records | 104 |
| Pinned sources that resolve in Git at their exact commit | 104 (33 unique files: 28 in Math-, 5 in main) |
| Records covered by #605 source accounting (v22 D5) | 24 |
| ...of which the v22 package's source SHA-256 equals the Git bytes | 24 of 24 |
| Records not yet covered by any #605 accounting | 80 |

The 80 records not yet covered, by topic: D1 14, D1 corrected lifetime / marked Kac–Rice 8, D1 extension 1, P15 15, D4 13, D2 10, D3 7, C6 3, C6 rare clusters 3, H0 3, NUM 2, SARD-G 1.

**What this establishes:** every record names a source that exists, byte for byte, at a pinned Git commit. That means every family can read it directly, with no Drive transfer.

**What it does not establish:** correctness or applicability. The dataset's `source_authenticated: false` and `side_conditions_open` statuses are copied unchanged and are not upgraded by this index. Not one record has a theorem certificate attached.
