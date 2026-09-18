# Drive source map

Content-addressed index of the Google Drive shared drive, from the
`07_MODEL_ACCESSIBILITY — READING COPIES AND SOURCE MAP` publication of
2026-09-17.

| File | Rows | What it holds |
|---|---:|---|
| `inventory.jsonl` | 4,456 | one JSON object per Drive item: id, title, path, mimeType, bytes, sha256, context, access_status, link |
| `source_map/Files.csv` | 4,456 | the original accessibility table, including reading-copy links |
| `source_map/Archive_Members.csv` | 11,649 | every member of every archive carrier, with payload SHA-256 and byte count |
| `source_map/Payloads.csv` | 4,020 | distinct source payloads by SHA-256, with occurrence counts |
| `source_map/Exceptions.csv` | 137 | retained exceptions: empty bodies, read failures, compiled caches, DOCX layout limits |

Totals: 3,714 files and 742 folders, 319 MB, 77 archive carriers, 4,020 distinct
payloads, 2,059 items carrying a source SHA-256.

## Queries

```bash
python3 tools/drive_index.py stats
python3 tools/drive_index.py find D1_ASSEMBLY
python3 tools/drive_index.py id 1v4z492iAzk5NcOrR47IJHGkIgfsRACpC
python3 tools/drive_index.py sha 0c9446b7          # file or archive-member payload
python3 tools/drive_index.py tree 01_ACTIVE_RESEARCH_PACKAGES --depth 2
python3 tools/drive_index.py archive RN3-20260917  # list a carrier's members
python3 tools/drive_index.py exceptions READ_FAILED
```

## What this is and is not

It **is** a metadata and digest snapshot that lets any reader resolve a title, a
path, a Drive ID or a SHA-256 to a specific source, including content sealed
inside ZIP carriers.

It **is not** proof of anything mathematical, and it does not override a
register status. Catalog presence or omission never establishes a claim. The
`99_DO_NOT_OPEN` vault appears in the inventory as metadata only; its contents
are deliberately not mirrored, per the standing order that models must not open
it for authority.

New or changed sources require a scoped refresh. Check the original source ID
and SHA-256 before reusing anything derived from this snapshot.
