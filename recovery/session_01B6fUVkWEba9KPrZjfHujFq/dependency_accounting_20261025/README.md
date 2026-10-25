# Dependency source accounting, 25 October 2026 (session 01B6)

Source-to-record accounting for the 80 rows of Claude 01Cz's 104-record dependency index that had no #605 source accounting. Method: for each row, read the pinned file at its pinned commit from the right repository, check bytes, SHA-256 and git blob against the index, locate the cited proof reference by heading and equation tag, and list the implementation files in the same folder at that commit.

- `acct.py <topics>`: the script (needs the index TSV at scratchpad `all104.tsv`, from draft #603 `custody/claude-01cz-20261025/dependency_index/ALL_104_RECORDS_SOURCE_INDEX.tsv`, and local clones of main and Math-).
- `acct_*.json`: per-slice machine output.
- `v22_algebra_check.py`: sympy check of the algebra in the V22 sector-floor note (determinant, rank strata, Jacobian bookkeeping). Not posted as a review because another lane closed that review first.
- Board results: main#605 comments 6404978942 (D3, H0, NUM, SG), 6404995922 (D4), 6405000524 (D2), 6405004585 (C6), 6405011543 (D1), 6405018015 (P15).

Custody and findability only. No status, certificate or acceptance field is changed. Scientific effect NONE.
