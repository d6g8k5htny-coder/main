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

Totals: 3,714 files and 742 folders, 319 MB, 77 archive carriers (the completion
report's count; the inventory marks 76 items `ARCHIVE_INDEXED` and
`Archive_Members.csv` lists members for 73), 4,020 distinct payloads over the whole
source map (2,975 among archive members), 2,059 items carrying a source SHA-256.

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

## After the snapshot: `deltas/`

The Drive changed on 2026-09-18 (a folder restructure, a proposed Drive–GitHub
execution contract, a reusable-operations guide, and the register workbook).
Those changes are carried under [`deltas/`](deltas/README.md) as dated,
byte-exact, manifest-verified copies plus `PATH_CHANGES.jsonl`; `tools/drive_index.py`
overlays the path changes by default (`--snapshot` shows the export as
published). 238 of the 4,456 inventory rows carry a stale `path` since that
restructure; their ids, byte counts and digests are unchanged, and the export is
not edited.

## Curated copies: `mirrors/`

[`mirrors/`](mirrors/) holds byte-exact copies of selected Drive objects, laid out
as `mirrors/<Drive lane folder>/<path inside the lane>/`, one `_MANIFEST.jsonl`
per directory that holds files (Drive id, title, path, `dest`, `bytes`,
`sha256`, `exact`, inventory digest and byte count), verified by
`tools/verify_manifests.py` in CI. A row with `exact: true` hashes to the
digest the 2026-09-17 inventory declares for that id; a row with `exact: false`
is a text export of a native Google Doc, for which no payload digest exists
anywhere in the corpus — a reading copy, not the object. Each lane directory has
a README quoting the source's own status banners verbatim. Mirrored so far: the
reviews lane (21 cold-review packets, the post-ratification theorem package, the
Theorem B retraction folder, the terminal closure records and gate documents as
reading copies), the status layer of the prize reconnaissance lane, the SIDE24
3D ratification chain (`AO48-OPR-045`, `AO48-AUD-044/043/036`, `AO48-REC-034/035`,
the folder README; `AO48-AUD-033` tree-only after a hash mismatch), the 2026-08
KIMI/AO48 LB-RATE and K3 intake (22 files, 8 manifests) and, from the
`HOLD_NOT_FOR_SUBMISSION` lane, the `CLOSE-20260917-b9c2` carrier byte-exact.
**A mirror is a copy: it is not review, replay, endorsement or promotion, and it
moves no status.**
