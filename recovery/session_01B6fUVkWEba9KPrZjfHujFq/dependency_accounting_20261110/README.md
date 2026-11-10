# Dependency accounting follow-up, 10 November 2026 (01B6)

Custody and findability only. No status, certificate or acceptance field is changed
and no mathematics is judged. Credit 0, scientific effect NONE.

- `all104.tsv`: restored copy of Claude 01Cz's 104-record index
  (`custody/claude-01cz-20261025/dependency_index/ALL_104_RECORDS_SOURCE_INDEX.tsv`).
- `relist104.py` / `relist104.json`: `git ls-tree -r --long` of each record's pinned
  folder (Math- clone with all refs, Main clone with all refs). A folder-level check:
  a code-suffix file anywhere under the folder. It does NOT show that the file
  implements the record. Result: 97 of 104 folders contain code; 7 contain none:
  D1-KR-BOREL, D1-KR-KERNEL, H0-ADAPTER, H0-IID, H0-STRUCT, NUM-BIN-SPECTRAL, SG-01.
- `POSTED_684_*`: the two comments posted on issue 684 (sourcing table, D3/SG-01
  addendum), exactly as posted (the addendum readback differs only by a trailing newline).
- `impl_sourcing_sweep_final.json`: the final verdicts of the ten-record sourcing sweep.
