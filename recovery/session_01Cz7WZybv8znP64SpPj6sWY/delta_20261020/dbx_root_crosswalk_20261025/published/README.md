# Dropbox outside `Github/`: byte-identity crosswalk, 25 October 2026

Data only. Scientific effect NONE. Credit 0. Completes the Dropbox side of the retroactive accounting that `audits/dropbox_import/2026-10-24` (main#594, the `Github/` subtree) began, with the same indexes and the same `content_hash` rule; it creates no second catalog.

## Scope and inputs

- every file in the owner's Dropbox outside the Github/ subtree that main#594 covered: six top-level folders (labelled A to F), the root of the shared namespace that also carries Github/, and loose files at the account root; listed 2026-10-24 21:4xZ to 2026-10-25 00:0xZ; Github/ entries reached by the recursive listing were excluded.
- **Files:** 1,089 in all; 808 with a `content_hash` read from link metadata (no file downloaded); 281 personal or other-material files counted only (link files, personal documents, an app bundle, the owner's other research); 0 empty; 0 without a hash.
- **Reference sides.** The 32,221 fetched Git blobs of main and Math- (Dropbox `content_hash` per blob), the two reassembled archives (DG-MIGRATION object `a2136bc0`, git blob `c4065c67`), and the 5,724 `Github/` hashes of main#594. Tiers: `GIT_EXACT`, `ARCHIVE_EXACT`, `GITHUB_TREE_COPY` (same bytes as a `Github/` file), `ROOT_DUP` (same bytes as another file outside `Github/` only), `ROOT_ONLY` (found nowhere else).

## Tiers

| tier | files | files in published parts | bytes of published parts (rounded to 100 KB) |
|---|---:|---:|---:|
| `ARCHIVE_EXACT` | 10 | 10 | 0.4 MB |
| `GITHUB_TREE_COPY` | 628 | 628 | 11.2 MB |
| `GIT_EXACT` | 89 | 89 | 14.4 MB |
| `NOT_HASHED` | 281 | 278 | 36.3 MB |
| `ROOT_DUP` | 8 | 6 | 0.4 MB |
| `ROOT_ONLY` | 73 | 70 | 32.1 MB |

A tier byte total covers only the parts that print bytes elsewhere in this directory (per-file rows and aggregate cells of three files or more); files in count-only cells are excluded, so no withheld size can be recovered by differencing.

`GIT_EXACT` by location: BRANCH_OR_HISTORY 4, DRIVE_MIRROR_BRANCH 68, TIP 17. 705 hashed files are byte-identical to a `Github/` file (so already covered by main#594's tiers).

**Found nowhere else (`ROOT_ONLY`):** 73 files, 73 distinct contents, by folder label account root (loose files) 7, folder A 64, folder D 1, shared namespace root 1; by class json 1, markdown 2, office 1, other 2, python 2, text 63, zip 2. `ROOT_DUP`: 8 files, 4 distinct contents.

## Folders (synthetic labels; the label-to-folder map is local)

| label | files | hashed | years with three or more files (omitted where the folder has per-file rows) |
|---|---:|---:|---|
| account root (loose files) | 22 | 11 | (omitted) |
| folder A | 434 | 428 | (omitted) |
| folder B | 42 | 42 | (omitted) |
| folder C | 17 | 17 | (omitted) |
| folder D | 10 | 3 | 2025 |
| folder E (counted only) | 22 | 0 | 2025 |
| folder F (counted only) | 235 | 0 | 2026 |
| shared namespace root | 307 | 307 | (omitted) |

Labels are synthetic (the label-to-folder map is local); years list only years with three or more files and are omitted for a folder that has per-file rows; no month, day or time is published.

## Coverage ledger

The main#594 name rule applied to the paths below each folder gives 449 hits over 114 of the 344 ledger units (33 `MISSING`-proof units: rows 43, 59, 78, 137, 143, 144, 145, 146, 158, 161, 162, 169, 176, 177, 189, 190, 192, 195, 204, 206, 210, 212, 214, 222, 223, 225, 284, 285, 290, 291, 301, 311, 330); by class WEAK_LONE 433, WEAK_PAIR 16; by tier of the hit item GITHUB_TREE_COPY 290, GIT_EXACT 159. Name evidence only. Published hits cover only items in the `GIT_EXACT`, `ARCHIVE_EXACT` and `GITHUB_TREE_COPY` tiers (1 hit on other items stays local, as do the matched tokens, which are fragments of Dropbox names); `LEDGER_ROOT.tsv` is one line per unit, `LEDGER_ROOT_ROWS.tsv` lists only hits on bytes-public items with the match class and no tokens.

## Files

- `ROOT_CROSSWALK.tsv`: 99 per-file rows, all for bytes already public (`GIT_EXACT`, `ARCHIVE_EXACT`): folder label, class, bytes, `content_hash`, tier, Git blob or archive locator, ledger rows. No Dropbox path, name or time.
- `ROOT_AGGREGATE.tsv`: the 990 other files by folder label, tier and class; classes below three files are merged into `(other)` per folder and tier, and a cell still below three prints its count only.
- `LEDGER_ROOT.tsv`, `LEDGER_ROOT_ROWS.tsv`, `SUMMARY.json` (with the hash receipt), `PINS.json` (inputs and local files by hash), `own/` (the scripts, local roots replaced).

## Limits

- Byte identity says a file is preserved somewhere; it says nothing about canonical copies, review status or correctness. The `Github/` tiers of main#594 are not repeated here.
- The hash receipt ties each `content_hash` to the listing by path and byte count only; the connector returns no revision with link metadata, so a same-size edit between the listing and the hash read is not excluded.
- Files and folders are counted and labelled, never named; the labels are synthetic and the label-to-folder map, the path-level inventory and the hashes stay local with the author and are offered to the owner or the CoS.
- Not checked here: the `Github/` subtree itself (main#594), files changed after the listing, and the content of the counted-only files.
