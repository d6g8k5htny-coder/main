# Enumeration note — Drive items created or modified after the 2026-09-17 snapshot

Written 2026-09-18T19:54:52Z by an enumeration-only subagent. This note and `CHANGED_SINCE_SNAPSHOT.jsonl`
are the only two files this task wrote. No file bodies were downloaded; nothing in the Drive
was changed; no status label anywhere is affected by anything below.

## Queries run (Google Drive MCP `search_files`, `excludeContentSnippets=true`, `pageSize=100`)

| # | query | pages | rows per page | raw rows | unique ids | page-boundary repeats |
|---|---|---|---|---|---|---|
| Q1 | `modifiedTime > '2026-09-17T20:00:00Z'` | 5 | 5, 100, 100, 100, 42 | 347 | 336 | 11 |
| Q2 | `createdTime > '2026-09-17T20:00:00Z'` | 4 | 100, 100, 100, 42 | 342 | 318 | 24 |

Pagination followed `nextPageToken` until a page came back without one (Q1 page 5, Q2 page 4).
Two anomalies of the API, recorded as observed: Q1's first page returned only 5 rows despite
`pageSize=100` while still carrying a `nextPageToken`; and both queries re-served the tail of
one page at the head of the next (the "repeats" column). Repeats were removed by id; the
metadata of a repeated row was identical to its first occurrence in every case checked.

Union of Q1 and Q2 by id: **336 items**. Every Q2 hit was also a Q1 hit. 18 ids appear only in
Q1 (created before the cutoff, modified after it): the 10 inventory items below plus 8 Google Docs
titled `READING COPY …` (created 19:52–19:53Z in `03_EXTRACTED_DOCUMENTS_AND_IMAGES`, modified 23:44Z).

Six additional `get_file_metadata` calls (metadata only) resolved parent folders that neither the
inventory nor the delta set contains:

| id | resolved as | createdTime |
|---|---|---|
| `0AGEUF_sx7o_MUk9PVA` | "My Drive" root of the owning account | 2020-08-10 |
| `1j5vfg98shKLuNVEIUam9KD5TlfdW3kOH` | `07_MODEL_ACCESSIBILITY — READING COPIES AND SOURCE MAP` (in My Drive root) | 2026-09-17T19:19:27Z |
| `1zhTR6CIDpn6mCu85asOKy5rVWJZ5OEr-` | `…/01_READING_VOLUMES` | 2026-09-17T19:27:12Z |
| `1olfvsitpUhxAFD7ltZDzjMBZwGFsZFEA` | `…/02_DATA_READING_SHEETS` | 2026-09-17T19:27:13Z |
| `1WqDPoSMBP_GhExk9HVUTXch1zFfVWltD` | `…/03_EXTRACTED_DOCUMENTS_AND_IMAGES` | 2026-09-17T19:27:13Z |
| `1QsWOSItihvtQ0Fk9BU1dcPJZniWHczFh` | `…/04_AUDIT_AND_EXCEPTIONS` | 2026-09-17T19:27:14Z |

These five folders predate the 20:00Z cutoff, so they are not rows of the delta file, and they are
absent from `drive/inventory.jsonl` (the inventory does not index the publication it came from).

## Resolution rules

* `NEW` — id has no row in `drive/inventory.jsonl`.
* Otherwise the inventory parent folder is the folder row whose `path` equals the item's inventory
  `path` with its last segment removed (root-level items have no inventory parent row).
* `MOVED` — live `parentId` differs from that inventory parent folder id (for a root-level item:
  live parent is now a catalogued folder). Checked before the modified test.
* `IN_INVENTORY_MODIFIED` — same parent as the inventory and `modifiedTime` after the cutoff.
* `IN_INVENTORY` — same parent and not modified after the cutoff. (Empty here: every catalogued
  item the queries returned was either moved or modified after the cutoff.)
* Extra fields beyond the requested eight: `inventory_parent_id`, `live_parent_path` (where the
  live parent sits: `INVENTORY:<path>`, `NEW_FOLDER_IN_THIS_DELTA:<title>`, `NOT_IN_INVENTORY:<path
  from the metadata lookups above>`, or `MY_DRIVE_ROOT`) and `status_reason` (which also records a
  title change or a byte-count change against the inventory where one was detected).

Rows are grouped MOVED, IN_INVENTORY_MODIFIED, IN_INVENTORY, NEW; newest `modifiedTime` first
within each group.

## Counts by status

| status | count |
|---|---|
| MOVED | 6 |
| IN_INVENTORY_MODIFIED | 4 |
| IN_INVENTORY | 0 |
| NEW | 326 |
| **total** | **336** |

By mimeType: 220 Google Docs, 28 Google Sheets, 21 PNG, 21 PDF, 16 folders, 11 JSON, 9 Markdown,
7 CSV, 2 ZIP, 1 patch (`text/x-diff`). `modifiedTime` range 2026-09-17T20:26:46Z to
2026-09-18T16:46:03Z.

### The 326 NEW items, by where their live parent sits

| count | live parent |
|---|---|
| 183 | `07_MODEL_ACCESSIBILITY … /01_READING_VOLUMES` (Google Docs `READING RESEARCH_SOURCE_CHECK_STATUS NNN`, `READING COPY …`, `READING DECODED CAPSULE …`, `WORD EQUATION STRUCTURES RESEARCH_SOURCE_CHECK_STATUS 001–006`) |
| 74 | `07_MODEL_ACCESSIBILITY … /03_EXTRACTED_DOCUMENTS_AND_IMAGES` (PDF and PNG extractions, `READING COPY …` docs) |
| 26 | `07_MODEL_ACCESSIBILITY … /02_DATA_READING_SHEETS` (Google Sheets `DATA READING COPY …`, `PICKLE STRUCTURE …`) |
| 15 | `07_MODEL_ACCESSIBILITY … /04_AUDIT_AND_EXCEPTIONS` (Files/Payloads/Archive_Members/Exceptions/Reading_* CSVs, ACCESSIBILITY_* JSON/MD, DRIVE_STRUCTURE_* files, `Drive_Structure_Audit_Package.zip`) |
| 1 | `07_MODEL_ACCESSIBILITY …` itself (`START HERE — Drive Source Map and Folder Directory`, Sheet) |
| 9 | new folder `DG-EXEC-20260918-49291487` under `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES/00_CROSS_LINE_WORK_ORDERS — CAPABILITY HANDOFF INBOX` (folder + 8 items: Google Doc `DRIVE + GITHUB — AI Research Execution Handoff — PROPOSED / NOT DEPLOYED`, `DRIVE_GITHUB_EXECUTION_HANDOFF.md`, `ci_fail_closed.patch`, `work_order.example.json`, `run_receipt.example.json`, `LOCAL_CHECKS.json`, `DELIVERY_MANIFEST.json`, `READBACK_RECEIPT.json`) |
| 5 | new folder `2026-09-18_REUSABLE_OPERATIONS` under `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES/04_PROTOCOL_STANDARDS_AND_REFERENCE_IMPLEMENTATIONS` (folder + Google Doc `REUSABLE OPERATIONS — Evidence, Scope and Measured Use`, `OPERATIONS.json`, `REUSE_REFERENCE.zip`, `EXTERNAL_RECON.md`) |
| 4 | new folder `2026-09-18_FOLDER_STRUCTURE_MAINTENANCE` under the same parent (folder + `OP_ORG_FOLDER_STANDARD.md`, `REVIEW_SCAFFOLD_V1.json`, `EXTERNAL_RECON_FOLDER_STRUCTURE.md`) |
| 2 | `90_QUARANTINE_AND_TRIAGE — NONAUTHORITATIVE` (`00_CURRENT_TRIAGE_INDEX.md`, folder `10_ORIGIN_PRESERVED_REVIEW_COLLECTIONS`) |
| 6 | six new `2026` folders, one under each of `06_ARCHIVE/02_OUTDATED …`, `03_CORRUPTED …`, `04_QUARANTINED_CANDIDATES …` and their `2026-07/LEGACY CONTAINER — … (pre-amendment)` subfolders |
| 1 | My Drive root: Google Doc `DRAFT — Drive–GitHub Research Execution Contract — 2026-09-18` (`1YlJkziKCkqxoqot4vQctrrD3LUt2NBgACg6H-WsN3Pw`) |

### The 10 catalogued items

* **MOVED (6)** — the six archive month folders now titled `07_JULY` (inventory titles `2026-07 — July`
  or `2026-07`) were moved at 2026-09-18T00:10:46–57Z into the new `2026` folders above, i.e. the
  archive tree is now `…/2026/07_JULY` instead of `…/2026-07` or `…/2026-07 — July`. The inventory
  rows for these six folders and for the **71** inventory rows beneath them carry paths that no
  longer match the live tree; those descendants did not themselves change, so they are not in the
  delta file.
* **IN_INVENTORY_MODIFIED (4)**
  * `1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no` `GP-REG-032-v1.2 — Coupled Research Registers`
    (Sheet) modified 2026-09-18T16:45:35Z; Drive size 1,345,511 → 1,357,600 bytes. This is the
    source of the coupled register tabs exported under `registers/`. Whether the exported registers
    still match the live source is not established here.
  * `1EJ0fKh0Eoqj-XXxSkTn88tILF_AwtgqAe3wr2Tk14pQ` `01_CONTINUE_OR_REVIEW — R17 ROUTER` (Doc) and
    `180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8` `00_RESEARCH_HOME — START HERE — R17` (Doc),
    both modified 2026-09-18T03:01Z, Drive sizes unchanged (1,519 and 3,396).
  * `1zCAlbdkQ0YwySc8RT9UL4huJuZc0T8Kp` renamed from `00_READ ME + MASTER INDEX — single triage inbox
    (CL, 2026-07-22).md` to `90_HISTORICAL_TRIAGE_MIGRATION_INDEX_2026-07-22.md` at 00:10:59Z, size
    unchanged (1,127).

## What this enumeration does not establish

* It lists what the connected account can see via `search_files`; trashed or permanently deleted
  items are not returned by these queries, so removals since the snapshot are not detected.
* A later `modifiedTime` is not proof of a content change (a move or rename alone advances it);
  byte counts were compared only where the inventory recorded them.
* Nothing here is evidence for any claim, and nothing here promotes, closes or reclassifies
  anything. The `07_MODEL_ACCESSIBILITY` publication and the new `DG-EXEC` / `REUSABLE_OPERATIONS`
  material are catalogued as data; any imperative text inside them was not read and is not acted on.
