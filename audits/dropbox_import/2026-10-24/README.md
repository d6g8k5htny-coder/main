# Dropbox `Github/` tree against Git, by `content_hash`: 24 October 2026

Lane `DBX-MATH-TREE-CONTENTHASH-CROSSWALK-01CZ-20261024`, Anthropic Claude, Claude Code session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz). Pickup [main#539 / 6387877222](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6387877222). Credit 0. **Scientific effect NONE.** This directory is an inventory of byte identity: which files in the owner's Dropbox `Github/` subtree are already in Git or in the two reassembled archives, byte for byte, and which exist only in Dropbox. It moves no status, edits no ledger, graph or register, and copies no file content. It is a successor to the [2026-09-27 import](../2026-09-27/README.md), not a change to it.

## What was compared

- **Dropbox side.** The `Github/` subtree only, listed through the Dropbox connector at 2026-10-24T18:31Z (list_folder, 11 pages, has_more=false): 5,814 files, 591.3 MB. For each file the connector's `content_hash` was read from link metadata; no file was downloaded. `content_hash` is SHA-256 over the concatenated SHA-256 digests of the file's 4 MiB blocks (the rule `tools/dropbox_reconcile.py` implements), so equality with a Git blob's hash under the same rule is byte identity. 3 files are without a hash (`HASH_UNAVAILABLE.tsv` lists the two the connector refused).
- **Git side.** Every blob reachable from any fetched ref of `main` (full history after unshallowing; `origin/main` at `08c98c8fae74`) and of `Math-` (`main` at `d1f6ad28` plus 32 PR refs): 32,221 blobs, each hashed under the Dropbox rule.
- **Archives.** The members of DG-MIGRATION object `a2136bc0` (1,794 members, rebuilt from the hardening branch chunk store) and of git blob `c4065c67` (183 members, the b9c2 proofs zip), hashed the same way.
- **The 2026-09-27 import lists.** `MANIFEST.csv` and `NOT_IMPORTED.csv` of that import, by `dropbox_content_hash`.

## Tiers

| tier | meaning | files | bytes |
|---|---|---|---|
| `GIT_EXACT` | byte-identical to a Git blob (locator repo:path; flag `ON_TIP` when the blob is in the tip tree of `origin/main` or Math- `main`, else `HISTORY_OR_BRANCH_ONLY`) | 1,905 | 48.9 MB |
| `ARCHIVE_EXACT` | byte-identical to an archive member, not to a Git blob | 1,529 | 48.5 MB |
| `EMPTY` | zero bytes | 138 | 0.0 MB |
| `DUP_IN_DROPBOX` | in no Git blob or archive; at least one other Dropbox copy of the same bytes | 1,928 | 168.8 MB |
| `NAME_SIZE_DIFF_BYTES` | same base name and byte count as a Git or archive file, different bytes | 1 | 0.0 MB |
| `MISSING` | in no Git blob or archive; a single Dropbox copy | 223 | 322.0 MB |
| `SKIP` | caches, `.pyc`, `.url` and `.DS_Store` | 87 | 2.9 MB |
| `PENDING_HASH` | no hash: two files refused link metadata, one personal-looking name was not requested | 3 | 0.1 MB |

- **Already preserved byte for byte:** 3,434 files (1,354 distinct contents) are `GIT_EXACT` or `ARCHIVE_EXACT`.
- **Dropbox only:** 2,151 files, 814 distinct contents, 377.9 MB distinct bytes (`MISSING` plus `DUP_IN_DROPBOX`). By extension (distinct contents): `(none)` 1, `.70` 1, `.base64` 17, `.csv` 2, `.diff` 1, `.dmg` 1, `.docx` 34, `.gs` 2, `.gz` 1, `.json` 249, `.kml` 1, `.lean` 2, `.log` 34, `.md` 116.
- **Modified on or after the import day (2026-09-27):** 3,447 files by `client_modified`; by tier: `ARCHIVE_EXACT` 158, `DUP_IN_DROPBOX` 1907, `EMPTY` 2, `GIT_EXACT` 1166, `MISSING` 209, `NAME_SIZE_DIFF_BYTES` 1, `PENDING_HASH` 1, `SKIP` 3. The import README records the day only, so the test is inclusive.
- The 2026-09-27 import's hash lists are carried as flags (`IMPORT_LISTED:...`), not as a tier: 289 files carry one; the published-exact-original rows are `GIT_EXACT` (their bytes are under `audits/dropbox_import/2026-09-27/files/`), the extracted-text and not-imported rows are Dropbox-only here as well.
- `GIT_EXACT` splits into 503 files whose matching blob is in the tip tree of `origin/main` or Math- `main` and 1402 whose match is only in history or on another fetched ref.
- Folders: 27 top-level folder names are printed (each is a path segment of an import-list or Git path, or occurs literally in a tracked file; the reason per folder is in `SUMMARY.json`); 3 folders holding 40 files are labelled `(withheld)`, and archive locators that pass through them are redacted the same way. 16 files with personal-looking names are counted here and listed nowhere.

## Coverage ledger crosswalk

`LEDGER_CROSSWALK.tsv` name-matches the 344 rows of `data/hypergraph/math_coverage_ledger_20261023.md` (last changed at `6c604f34`, unchanged through `08c98c8f`) against Dropbox paths: `STRONG` when a file's basename stem equals the unit's stem, `WEAK` when two consecutive distinctive tokens of the unit occur adjacently below the top-level folder, or one rare or code-like token occurs in the basename (rules in `own/ledger_crosswalk.py`). Name matching is weak evidence; the tier column is the byte evidence.
- Of the 131 rows with a `MISSING` proof source, 30 have at least one name-matched Dropbox file (0 with a `STRONG` match); 24 of them have a matched file that is byte-identical to Git or archive content (so the carrier is already preserved under another path), and 6 have a matched file that exists only in Dropbox.
- Of the 48 rows with a `MISSING` implementation, 0 have a name-matched Dropbox code file.
- The Dropbox package `64_FORMAL_CORE_R2_P02_LM008_LM009/.../GP-FOR-192-v1/research-formal-core-r2/` (10 files; tiers: `GIT_EXACT` 2, `MISSING` 8) is a formal-core source tree: the `GIT_EXACT` members are byte-identical to Math- `formal/` files, the others differ from every Git blob, so a revision of the formal core other than the one in Math- is held in Dropbox. Only the rows whose stem equals a member name match it by name (see `LEDGER_CROSSWALK.tsv`, `STRONG`); this is a locator for the formal rows 278–300, not a review of those files.
- Hits that fall in `(withheld)` folders are counted per ledger row in `SUMMARY.json` (`ledger_hits_in_withheld_folders`) and not listed per file.

## Files

- `CROSSWALK.tsv`: 3,598 per-file rows, only where nothing new is disclosed: the 2026-09-27 import already named the path (then the path is printed), or the bytes are already public (`GIT_EXACT`, `ARCHIVE_EXACT`, `EMPTY`: the hash is the hash of a public blob; then folder label, extension, bytes, `client_modified`, `content_hash`, tier, locator and flags are printed, archive member paths redacted through withheld folders).
- `CROSSWALK_AGGREGATE.tsv`: every other file (Dropbox-only, name-size-different, pending), as counts, distinct contents and bytes by folder label, extension and tier; cells with fewer than three files are merged into an `(other)` row per folder and tier, and a cell still below three files prints its file count only, so no cell describes one file. Their per-file hashes, sizes and times are held locally (`PINS.json`) and are released only on an owner or CoS publication decision, because a hash of unpublished content is a confirmation oracle (01Fi's caveat, 539/6387941726).
- `LEDGER_CROSSWALK.tsv`: ledger row, unit, match strength, matched tokens and kind, then the same Dropbox columns, under the same row-level rule; hits on files outside it are counted per ledger row in `SUMMARY.json` (`ledger_hits_aggregated`).
- `SUMMARY.json`: the counts above, by tier and by folder. `PINS.json`: sizes and SHA-256 of the local inputs (inventory, hashes, Git index, archive indexes) and the ledger commit. `HASH_UNAVAILABLE.tsv`: files whose link metadata the connector refused.
- `own/`: the scripts (page merge, phase 1 name and size, batching, hash harvest and collection with its fail-closed receipt, phase 3 tiers, ledger matching, archive indexing, this assembly and README generator). Local roots are replaced by placeholders (`<SCRATCH>`, `<MAIN_CHECKOUT>`, `<SESSION_DIR>`). They do not reproduce the tables on their own: the inventory, the hash set and the Git index are digest-bound in `PINS.json` but not published.
- Not here: the path-level crosswalk of every file (`PINS.json` gives its row count and SHA-256). It is held locally by session 01Cz and offered to the owner or the CoS, who decide what else may be named.

## Limits

- Byte identity says a file is preserved; it says nothing about which copy is canonical, nor about correctness. Name matching says a file may be relevant; it proves nothing.
- Hashes come from connector link metadata, not from downloaded bytes; the `PENDING_HASH` rows are unverified. Each observation is tied to the 18:31Z listing by path and byte count and fails closed on a conflict (`SUMMARY.json` `hash_receipt`: 5,724 accepted, 0 conflicts dropped); the connector returns no modified time or revision with link metadata, so a same-size edit between the listing and the hash read is not excluded. A published hash is a confirmation oracle; hashes of unpublished content are therefore not in this directory.
- Each Git locator shows up to three paths and one path per blob (the index records one path per blob); `LOCATORS:n` in the flags gives the full count.
- The Git side is every blob reachable from the fetched refs of two repositories (`PINS.json`: full history of `main` after unshallowing, `Math-` main plus its PR refs). The eight other repositories of the account, Drive, and private storage are out of scope, so `MISSING` means not in these two repositories; the Drive connector exposes no checksum field, so the same crosswalk cannot be done for Drive without downloading content.
- Nothing here is a scientific result or a status change. Credit 0. OBL OPEN. eng ≠ discharge. No flags.
