# Shared custody: data held by Claude session 01Cz, 25 October 2026

Data only. Scientific effect NONE. Nothing here is a proof, a review verdict or a status change.

The owner asked that scientific data never be held by a single model or family. This directory publishes everything the Anthropic Claude session `session_01Cz7WZybv8znP64SpPj6sWY` built locally that was not already public: the full path-level and id-level data behind its Dropbox and Drive byte-identity crosswalks, its reference indexes, and its RN-UNIF search hits. The same README and manifest, with the smaller tables, are also in the owner's Google Drive and Dropbox; this GitHub copy is the complete one.

## Contents

- **`dropbox_github_tree/`** (8 files): Dropbox `Github/` tree (5,814 files), path-level: inventory, per-file tier and content_hash, ledger join (the full data behind main#594, which published only import-named rows).
- **`dropbox_outside_github/`** (6 files): Dropbox outside `Github/` (808 hashed of 1,089 files), path-level: inventory, per-file tiers, content hashes, ledger hits, real folder names behind the synthetic labels (the full data behind 539/6398421067).
- **`drive_crosswalk/`** (7 files): Drive 2026-09-17 publication crosswalk, id-level: per-file and per-archive-member tiers, carriers, items created since 17 September, ledger hits, and the 24 October live listing (the full data behind 539/6390454043).
- **`indexes/`** (6 files): Reference indexes: every fetched Git blob of main and Math- with its Dropbox content_hash (32,221), the tip set, and member indexes of DG-MIGRATION object a2136bc0 and git blob c4065c67.
- **`rn_unif_search_20261025/`** (1 files): Raw name hits for the RN-UNIF originals search (86/6398780395).
- **`scripts/`** (14 files): The scripts that produced these tables (local roots replaced by <SCRATCH>).

42 files, 33,540,970 bytes. `MANIFEST.tsv` gives the size and SHA-256 of every file.

## What was removed, and why

Only material that is not scientific data was left out:
- 281 Dropbox rows and 23 Drive rows that the crosswalks had already flagged as personal documents (identity, tax, financial, medical and similar files, and the owner's personal Drive folder), plus 5 new Drive items under that folder. Their counts stay in the published summaries.
- Email addresses, replaced by `<email>`.
- No secrets, tokens or credentials were present; a pattern scan found none.

Drive ids and Dropbox namespace paths are kept, per the owner's 25 October direction relayed by the CoS (539/6398833595), so any agent with access can open the files.

## How the data was made

- Dropbox `content_hash` values were read from link metadata, without downloading. Git and archive sides use the same Dropbox block hash, computed from the bytes.
- Drive digests come from the 2026-09-17 accessibility publication (`drive/` tree at `a01c72f1`); the live listing was paged on 24 October.
- Ledger joins are name evidence only, by the main#594 rule (STRONG, WEAK_ADJ, WEAK_PAIR, WEAK_LONE).
- The published, privacy-reduced versions and their review records are on the main#454 retention branch (`claude/recovery-01cz-delta-20261020`).

## Limits

- Snapshots: Dropbox listed 24–25 October, Drive listed 24 October. Files changed afterwards are not reflected.
- Byte identity says a file is preserved somewhere. It says nothing about which copy is canonical, review status or correctness.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
