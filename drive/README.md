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

Totals: 3,714 files and 742 folders, 319 MB, 77 archive carriers — the 76 items
the inventory marks `ARCHIVE_INDEXED` plus the `BINARY_UNRENDERED`
`S2-DATA-002-v1.0_result_carrier.zip` (`1mYHVSdVk57CM9h6L3NFR9_G2EFXKPwhj`) for
which `Archive_Members.csv` also lists members; 72 of the 76 have member rows
and the other four are single-file `.gz` uploads — 4,020 distinct payloads over
the whole source map (2,974 distinct 64-hex digests among archive members; the
five `READ_FAILED` member rows carry an empty digest cell), 2,059 items carrying
a source SHA-256. Until 2026-09-19 this sentence quoted 77 as the completion
report's number, said members were listed for 73, and counted the empty cell as
a 2,975th digest.

## Provenance of these files

`inventory.jsonl` is derived from `Files.csv`: one JSON object per `Files.csv`
row, keyed by Drive ID, carrying eight of its columns (title, path, mimeType,
bytes, sha256, context, access_status, link) — 4,456 of the 4,456 rows
agree on id, title, path, digest and byte count (a `null` digest where the CSV
cell is empty). The committed bytes:

| file | bytes | SHA-256 |
|---|---:|---|
| `inventory.jsonl` (derived) | 2,807,455 | `48766f1807efaba61e9ecc2a54d358d779393d06cce39d0d7b2bd1f2f2636b66` |
| `source_map/Files.csv` | 2,503,504 | `4bad6b97c96a86382e200d31e50ac06932fdbbc8ad26a568559664441769c48e` |
| `source_map/Archive_Members.csv` | 6,241,565 | `32e360568ae835f9b78ceeded659e8adda3c17fb7c73420a7d59252177f52888` |
| `source_map/Payloads.csv` | 2,957,503 | `1b1b94a784ebc87cadee1098cde995c05dd6b5d541f55c6943fc8a48a9e43f4e` |
| `source_map/Exceptions.csv` | 53,212 | `625ba2d02bab66c6731f2b7823edd8077890f7a816f9e7b3776b69b6f59841ac` |

These four CSVs are four of the accessibility publication's ten audit files;
`tools/verify_manifests.py` does not cover them (their identity rests on the
digests above, recorded 2026-09-19).

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
`tools/verify_manifests.py` in CI, and every quotation in those READMEs is
verified against the stored bytes by `tools/mirror_quotes_check.py`
(see [`docs/MIRROR_QUOTES.md`](../docs/MIRROR_QUOTES.md)). A row with
`exact: true` hashes to the digest the 2026-09-17 inventory declares for that
id; a row with `exact: false` is a text export of a native Google Doc, for which
no payload digest exists anywhere in the corpus — a reading copy, not the object.
A reading copy that renders nothing — empty, whitespace only, or nothing but a
UTF-8 byte-order mark — is carried **only** where `drive/inventory.jsonl` itself
records that Drive id as `EMPTY_NATIVE_BODY`, which eight of the 4,456 rows do.
There the blank result is the one the inventory predicts and the manifest row
says so in words; anywhere else a blank export is a failed fetch rather than a
rendering, and `tools/verify_manifests.py` refuses it. The corroboration is read
from the inventory, never from the row, so a manifest cannot talk its own file
into being acceptable.
Each lane directory has a README quoting the source's own status banners verbatim.

**What is mirrored is listed in [`MIRRORS.md`](MIRRORS.md), per lane, with counts
generated from the manifests by `tools/mirrors_index_check.py` and refused by CI if
they drift.** Until 2026-09-20 this section instead carried a hand-written sentence
beginning "Mirrored so far:" that named six lanes; the tree held twenty-one, and no
check could tell.

**A mirror is a copy: it is not review, replay, endorsement or promotion, and it
moves no status.**
