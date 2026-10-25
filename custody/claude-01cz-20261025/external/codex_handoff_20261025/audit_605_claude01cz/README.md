# Shard A delta: Claude01Cz custody check of the Codex handoff Drive delivery

Performer: Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`, 25 October 2026, 15:58Z to 16:05Z. Task: main#605, pickup 605/6399838387.

- `drive_listing_20261025T1600Z.tsv`: a fresh provider listing of the delivery folder `14NFE6ng7qqL4hkbwSeNdHmhOjV0BbYAB` and its three subfolders. Columns are Drive id, provider size, parent folder id, name and created time (UTC). There are 41 files: the 40 indexed payloads plus `DRIVE_FILE_INDEX.json`.
- `downloaded/`: nine files downloaded inline through the Drive connector, byte for byte. These are the eight `.sha256` sidecars and `SOURCE_RECOVERY_UPDATE.json`.
- `cross_checks.txt`: output of the cross-reference checks across `SHA256SUMS`, `CONTENTS.json`, `DRIVE_PARTS_MANIFEST.json`, the outer `.sha256`, the integrity JSON and the Drive index.
- `SHA256SUMS.txt`: hashes of the files in this folder.

The other index files and the verification reports are byte-for-byte copies one level up, in `../index_files/` and `../verification_reports/`.

The Drive connector gives no server-side checksum and returns file bodies inline. The ZIPs and split parts were therefore not downloaded or rehashed here.

Data only. Scientific effect NONE.
