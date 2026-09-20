# Research source search — 2026-09-20

Open the [Drive search navigator](https://docs.google.com/spreadsheets/d/1x9aKuv9pQGD6kwwa3ZmehSyEnVX2ZlhhcY2OehjMCqI/edit).
Its Guide, Files and Archive members tabs are a generated view of the same
custody records used by `tools/drive_index.py`. The governing research register
and original source organization remain unchanged.

## Active research packages

[01_ACTIVE_RESEARCH_PACKAGES](https://drive.google.com/drive/folders/1cOnYpeU3Fdt06jJ8Np9wmNmJ8F4KWzzm)
is already represented in GitHub by **2,944 stored files/exports totaling
323,502,833 bytes**, each rechecked for exact stored size and SHA-256 in this
pass. These comprise 1,306 native exports, 310 existing exact files and 1,328
exact source files. The path-associated snapshot also records 424 folders and
40 no-open holds, for 3,408 objects. The holds remain metadata-only.

This is not a fresh traversal of all 3,408 objects. The permitted membership
graph reaches 3,365 records: 2,944 payloads, 418 folders and three held boundary
records. Another 37 held file records and six folders belong to untraversed
held subtrees. Native export bytes do not preserve all native comments,
revision history, behavior or semantics. See the exact
[active-package audit](context/ACTIVE_RESEARCH_PACKAGES_AUDIT_20260920.md).

RN5's corrected source informs the exact affine moment and interval-family
implementations. Preferred DS3, its drift reports and the whitening envelope
provide source/context and retrieval controls. The indexed `rnu_white.py` is
an explicitly dated author-side reconstruction, not recovery of a missing
historical Claude module. Source inclusion never promotes a mathematical claim.

## Find the right object

In Sheets, use **Edit → Find and replace → All sheets**, or filter a header.
Search a distinctive title, a Drive ID, or a full 64-character SHA. Retain all
occurrences; a matching archive member has a carrier ID and its own member
path and digest. A carrier digest identifies the container, not its members.

From the repository root:

```sh
python tools/drive_index.py find rnu_ds3.py
python tools/drive_index.py search Cholesky --json
python tools/drive_index.py sha ac89f60b8206 --mentions --json
python tools/drive_index.py id 1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5
python tools/drive_index.py find 01_ACTIVE_RESEARCH_PACKAGES --json
```

`find` searches file and archive metadata. `search` also searches eligible
stored UTF-8 text and paragraph text extracted from DOCX exports. A content
result carries its source digest and extraction rule. `sha` searches recorded
byte identities and returns both file and archive hits. `--mentions` separately
labels documents that merely mention the queried digest. Invalid hashes are
refused. Prefixes need 8–64 hexadecimal characters; multiple distinct matching
digests are ambiguous and exit with status 2. No identity match exits 1.
Multiple occurrences of one digest are not ambiguity and remain separate.
`--snapshot` retains the original September 17 metadata view.

## Scope and identity types

The default view has **4,934 file/folder identities**: 4,928 reconciled objects
plus six later deliveries with verified raw readbacks. It includes all **11,649
archive-table occurrences**, preserving row identities even where carrier,
path or content repeats. These totals describe different record types.

Text extraction covers **3,140 files**. It excludes 1,012 records by scope,
461 without stored payloads, 301 unsupported formats and 20 above the
5,000,000-byte extraction limit. Held, personal, vault, quarantine, legacy and
history content is not opened. Archive member contents are not text-indexed.
Metadata visibility grants no content or scientific authority.

On the first content query, a disposable local cache is reconstructed from
already-stored, size/hash-verified source bytes. Its complete decompressed
identity must match the committed extraction report. Later queries reuse the
cache after the same digest and source/scope checks. This avoids committing a
second 100 MB text copy. The cache is not a new governing catalog. Rebuilding
inputs deliberately requires regenerating the dated report, not silently
accepting an old text cache.

`RAW_FILE_SHA256`, `NATIVE_EXPORT_SHA256`, `ARCHIVE_MEMBER_SHA256` and
`SOURCE_MAP_REPORTED_SHA256` are distinct. Native export digests identify
exported containers, not native revision identity or body equivalence. Older
reported digests remain labeled and searchable. Exact UTC observation strings
are preserved in the native navigator without date-number conversion.

The reconciliation membership cutoff is 2026-09-20T00:50:55.975Z. Later
output records carry individual readback times. The navigator and this pass's
own final artifacts are intentionally outside that input snapshot; the final
delivery manifest provides their identities without recursive re-ingestion.

## Proven retrieval traps

Drive keyword search placed RN5 reading copies above its original. A live full
RN5 SHA query found hash-mention documents and missed the source itself. Live
exact-name searches missed `rnu_ds3.py`, although its direct ID and parent
listing returned it. An empty search therefore does not establish absence.
Use a known ID or a bounded permitted folder listing to resolve uncertainty.

The fixed SHA lookup returns the preferred DS3 live-file record and archive
member together. The 7,201-byte superseded version remains distinct from the
9,704-byte preferred source. Four indexed `rnu_white.py` occurrences share
13,344 bytes and SHA
`859e1963b1b7c84f2c1f14b4b163fb5575ad41c5394d27c51c0d1ceff70d2dc3`;
they are occurrences of the dated reconstruction. See the preserved
[discovery](context/SHA_KEYWORD_DISCOVERY_20260920.md) and
[addendum](context/SHA_KEYWORD_DISCOVERY_ADDENDUM_20260920.md).

Search results, hash matches and tests confer no scientific approval. There
is no claim of account-wide completeness, a complete RN spatial cover, novelty,
general utility or organizational independence.
