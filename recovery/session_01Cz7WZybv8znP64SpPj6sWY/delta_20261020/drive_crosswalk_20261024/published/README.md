# Drive byte-identity crosswalk, 24 October 2026

Data only. Scientific effect NONE. Credit 0. Not a review, a binding or a status change: Drive sources stay UNBOUND (GB-5). It joins the existing 2026-09-17 Drive catalog to Git, to the Dropbox `Github/` hashes of `audits/dropbox_import/2026-10-24` (main#594) and to two reassembled archives; it creates no second catalog.

## Inputs

- **Drive side.** drive/inventory.jsonl, drive/source_map/Archive_Members.csv and drive/deltas/2026-09-18..20 in the drive/ tree at a01c72f1 (Drive-GitHub hardening branch); the 2026-09-17 accessibility publication: 4,456 items plus 49 delta items (one record per Drive id; an id listed in a delta and in the inventory, or in two deltas, is counted once); 3,735 files outside the personal lane (17 files in `03_PERSONAL_AND_EARLIER_RESEARCH` are counted and not listed). 2,085 files carry a SHA-256 in that publication (binary originals and stored delta objects), 2,076 of them outside the personal lane; native Google Docs carry none.
- **Rule.** For a non-empty file of 4 MiB or less the Dropbox `content_hash` is the SHA-256 of its SHA-256 (an empty file's is the SHA-256 of nothing) (the rule `tools/dropbox_reconcile.py` implements), so the published digests are matched to the Dropbox table without any download. Git blobs and archive members are matched by SHA-256 directly, so the 4 MiB limit affects only the Dropbox column (17 files).
- **Git side.** Every blob reachable from the fetched refs of main (all remote branches, unshallowed) and Math- (main and PR refs): 32,221 blobs, the same index as main#594. `git_where` says whether a matching blob is on a tip as of the index tips listed in `PINS.json` (`TIP`), found in Git only under `drive/` paths (`DRIVE_MIRROR_BRANCH`: the mirror tree, on main from 18 to 23 September 2026 and on the hardening branches; `f35eef1b` reverted it from main) or elsewhere in history (`BRANCH_OR_HISTORY`).
- **Liveness.** A connector search for items with `modifiedTime` after 2026-09-17T00:00Z (ten windows, metadata only). That cutoff precedes the snapshot taken that day, so an item modified on 17 September is `MODIFIED_ON_SNAPSHOT_DAY` (digest `UNCERTAIN_SNAPSHOT_DAY`); an item whose `modifiedTime` equals its `createdTime` was never modified after creation (`SAME_AS_CREATED`, digest `CURRENT`); an item modified on 18 September or later is `MODIFIED_AFTER_SNAPSHOT_DAY`, and its snapshot digest is `CURRENT` only if a dated delta (18–20 September) re-hashed it at or after that modification, otherwise `STALE` or `STALE_SAME_SIZE` (same byte count; moves and renames bump the time; identity not re-verified). Items not in the search are `NOT_MODIFIED_SINCE_2026-09-17`. Items unknown to the snapshot are `NEW_SINCE_SNAPSHOT` and counted only.

## Tiers (files outside the personal lane)

| tier | files | files in published parts | bytes of published parts (rounded to 100 KB) |
|---|---:|---:|---:|
| `ARCHIVE_EXACT` | 24 | 24 | 42.1 MB |
| `DBX_EXACT` | 293 | 244 | 8.4 MB |
| `DRIVE_ONLY` | 87 | 64 | 27.7 MB |
| `DRIVE_ONLY_DBX_UNKNOWN_OVER_4MIB` | 10 | 8 | 112.8 MB |
| `GIT_EXACT` | 1,662 | 1,662 | 66.3 MB |
| `NO_DIGEST` | 1,659 | 1,570 | 26.2 MB |

A tier byte total covers only the parts that print bytes elsewhere here (rows with public bytes, and aggregate cells of three files or more); files in count-only cells and title-public rows without public bytes are left out, so no withheld size can be recovered by differencing.

`GIT_EXACT` by location: BRANCH_OR_HISTORY 45, DRIVE_MIRROR_BRANCH 1,497, TIP 120. `NO_DIGEST` are native Google Docs and a few delta rows without a stored object; `FOLDER` rows are not files.

**Dropbox overlap.** 882 Drive files are byte-identical to a Dropbox `Github/` file. Of the 814 distinct contents that main#594 found only in Dropbox, 287 also exist in Drive at the snapshot.

**Archive members.** The publication lists 11,649 members of 73 Drive archive carriers with payload digests: 8,042 are themselves in Git, Dropbox or an archive, 340 are preserved because their carrier is, and 3,267 are found in none of the three places (2,530 inside DRIVE_ONLY carriers, 737 inside DRIVE_ONLY_DBX_UNKNOWN_OVER_4MIB carriers; the over-4 MiB carriers' Dropbox identity is not derivable, nor is that of the 5 members themselves over 4 MiB). Carriers by tier: ARCHIVE_EXACT 1, DBX_EXACT 2, DRIVE_ONLY 32, DRIVE_ONLY_DBX_UNKNOWN_OVER_4MIB 9, GIT_EXACT 29.

**Liveness.** The ten listing agents each reported their window complete (True; a self-report, see Limits); 2,387 live items seen. Snapshot files by live status: MODIFIED_AFTER_SNAPSHOT_DAY 43, MODIFIED_ON_SNAPSHOT_DAY 40, NOT_MODIFIED_SINCE_2026-09-17 3,386, SAME_AS_CREATED 266. Hashed files by digest state: CURRENT 2,033, STALE_SAME_SIZE 35, UNCERTAIN_SNAPSHOT_DAY 8. New since the snapshot: 2,000 items (72 folders), 493 with a title already in the DRV maps, 1,462 under a parent the snapshot does not know; counted by class in `SUMMARY.json`, not listed.

## Coverage ledger

The 344-row coverage ledger has 131 units with a `MISSING` proof source; 83 of them had no Dropbox name hit of either strength in main#594 (48 had one: 30 printed at row level there, 18 only in its aggregated `ledger_hits`). The same name rule applied to Drive paths below the hub gives 480 hits over all 344 rows, 32 `MISSING`-proof rows among them, 13 of those without a Dropbox name hit (rows 51, 82, 131, 166, 183, 185, 188, 190, 195, 198, 221, 283, 309). None is `STRONG`; by class the hits are WEAK_ADJ 1, WEAK_LONE 381, WEAK_PAIR 98, and 33 of them match only in a folder segment. Of the 13 rows without a Dropbox name hit, 12 are reached only by one rare token in a prose title (`WEAK_LONE`, the weakest class: 241 of the lone-token hits fall on titles of more than eight tokens); none is unit-specific. Of the 113 hits on `MISSING`-proof rows, the items are DBX_EXACT 11, DRIVE_ONLY 3, GIT_EXACT 25, NO_DIGEST 74. The rule is the main#594 rule byte for byte, applied to the path below the hub and its package folder (title only where a delta item has no path); it is name evidence only, labelled per hit in `LEDGER_DRIVE_ROWS.tsv` (`match`, `where`).

## Join to the DRV maps

251 snapshot files carry a title that `docs/retro/DRV-H2/DRV-H2.map.csv` (135 rows) or `docs/retro/DRV-H3/DRV-H3_drive_map.csv` (651 rows) already prints; 167 of them have a snapshot digest, printed for the 101 whose bytes are already public (`GIT_EXACT`, `ARCHIVE_EXACT`) and withheld for the rest (DBX_EXACT 46, DRIVE_ONLY 19, DRIVE_ONLY_DBX_UNKNOWN_OVER_4MIB 1; held locally). The maps keep their objects; nothing here binds.

## Files

- `DRIVE_CROSSWALK.tsv`: 1,836 per-file rows under the row-level rule (per-file lines only where the title is already on main (DRV-H2/DRV-H3 maps) or the bytes are already public (GIT_EXACT, ARCHIVE_EXACT); digests only for public bytes; every other file is counted only, in DRIVE_AGGREGATE.tsv by hub x tier x class over those files alone, where classes below 3 files are merged into (other) per hub and tier and a cell still below 3 files prints its count and no bytes; the same rule for carriers and for ledger hits; 03_PERSONAL items counted only).
- `DRIVE_AGGREGATE.tsv`: the 1,899 files that have no per-file row, by hub, tier and class (so no cell can be differenced against the per-file rows); cells below three files are merged into `(other)` per hub and tier, and a cell still below three prints its file count only. Delta manifest folders are one generic hub label.
- `LEDGER_DRIVE.tsv`: one line per ledger unit with hits (counts by strength and tier); `LEDGER_DRIVE_ROWS.tsv`: per-item lines for public items only. `ARCHIVE_CARRIERS.tsv`: 36 per-carrier lines where the title is on main or the bytes are public, the other carriers aggregated by tier.
- `SUMMARY.json`, `PINS.json` (input blobs and local files by hash), `own/` (the scripts, with local roots replaced).

## Limits

- Byte identity says a file is preserved somewhere; it says nothing about canonical copies, review status or correctness.
- Digests come from the 2026-09-17 publication and the 18–20 September deltas; the connector exposes no checksum, so nothing modified after the last delta (2026-09-20) has a current digest here, and no file was downloaded.
- The live listing is metadata from a modified-after search: it cannot show deletions or trashing (`NOT_MODIFIED_SINCE_2026-09-17` means not re-modified, not still present), its completeness is the listing agents' self-report (one window returned no items), duplicate lines across page boundaries were de-duplicated by id, and items whose parent the snapshot does not know are counted without a hub.
- Byte totals are rounded to 100 KB and tier totals cover published parts only, so a withheld file's size cannot be recovered by differencing; the RESULT post of 22:41Z quoted whole-tier totals (rounded), which are not combinable with this directory's cells for that purpose either, because the cells here exclude the files whose bytes those totals also covered only in aggregate.
- Items created after the snapshot are counted, not identified; their names are not public and their parents are often folders the snapshot does not know.
- Row-level hashes of non-public files and the id-level crosswalk are held locally by the author and offered to the owner or the CoS.
