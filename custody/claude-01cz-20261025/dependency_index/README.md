# Dependency index: handoff source implementations joined to canonical records

This is the retroactive accounting the owner asked for: each mathematical record's source implementation, so the dependency work knows which artifact backs which record. Compiled by Anthropic Claude 01Cz on 25 October 2026 from the #605 audit results. It is data only. It copies statuses verbatim and changes none. Scientific effect NONE.

| File | What it is | Source |
|---|---|---|
| `D5_RECORD_SOURCE_INDEX.tsv` | 24 canonical records (23 with the D5 prefix, plus C6-PALM-Q) mapped to their direct source ID, source SHA-256, anchor lines and necessary premises, with the preserved acceptance, import, obligation and certificate statuses | Codex Capability Frontier v22 D5 accounting, main#605 comment 6400144048 |
| `V21_SUPPLEMENT_IMPLEMENTATION_INDEX.tsv` | The 14 v21 supplement members, which support the D5 collision cover as analytic candidates and are not canonical records. Each row gives the member's dependency scope, related checks and the Claude01Cz challenge result | Codex v21 map, main#605 comment 6399998554, and Claude01Cz challenge 6400100046 |
| `v22_D5_accounting_6400144048.json` | Exact JSON from that comment: 56,177 bytes, SHA-256 `6a5f6f2d0572cdb54d157c1345580d8f46bce96f552b95f05fbd3de641a8fa63` | as above |
| `v21_source_map_6399998554.json` | Exact JSON from that comment: 34,645 bytes, SHA-256 `7657e3f777a116271fe8b28cd81ec4bd76de338f3b4b5f74ade29327c102f1a5` | as above |

**Coverage: 24 of 104 records have a mapped implementation.**

The other 80 records are not yet mapped by any #605 result. The canonical dataset, `d4-far-audit/canonical_dataset.json` (396,561 bytes, SHA-256 `4cfae309…e4d7`), is inside the v22 archive and is not in Git. Once it is committed here as text, the remaining rows can be listed and assigned.

A record with no implementing member is a real gap. It is listed, never silently dropped.
