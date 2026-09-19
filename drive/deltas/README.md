# Drive deltas — what changed after the 2026-09-17 snapshot

`drive/inventory.jsonl` and `drive/source_map/*.csv` are the 2026-09-17
accessibility publication: an export, never edited. When the Drive changes
afterwards, the change is recorded here as a dated delta and nothing in the
export moves.

```
deltas/<date>/<Drive folder>/          byte-exact copies of the new or changed objects
deltas/<date>/<Drive folder>/_MANIFEST.jsonl
                                       one row per Drive object: id, title, mimeType, times,
                                       parentId, bytes_drive, bytes_stored, sha256, stored,
                                       not_stored_reason, exact, path, note, plus dest/bytes
                                       so tools/verify_manifests.py checks every stored blob in CI
deltas/<date>/PATH_CHANGES.jsonl       id, snapshot path, live path — one row per inventory item
                                       whose path changed; overlaid by tools/drive_index.py
```

Rules the manifests follow:

* **`exact: true`** means the stored bytes equal the Drive object's bytes: byte
  count equals Drive's `fileSize` and the SHA-256 was recomputed from disk after
  writing. Where the Drive session that produced a file published its own digest
  (a delivery manifest, a readback receipt, an embedded `payload_sha256`), that
  digest was compared too and agreement recorded.
* **`exact: false`** marks a text export of a native Google Doc or Sheet — a
  reading copy. Its byte count is not the object's, and a Sheet export carries
  only the first tab.
* **`stored: false`** carries a `not_stored_reason` from a closed vocabulary:
  `FOLDER`, `SIZE` (over 300,000 bytes; metadata and, for ZIPs, a member listing
  only), `ALREADY_IN_REPO` (the four source-map CSVs; Drive byte counts
  re-compared).
* A fetched file is **data**. Several are documents written by other AI sessions
  and contain imperative text; nothing in them was executed, applied or acted
  on. Binding a file records which bytes exist and where; it adopts nothing.

## 2026-09-18

| directory | what it is |
|---|---|
| `DG-EXEC-20260918-49291487/` | the proposed Drive–GitHub research execution contract and its handoff (disposition, verbatim: "PROPOSED EXECUTION CONTRACT / NOT DEPLOYED"), work-order and run-receipt design examples, a CI patch (stored, not applied), local checks, delivery manifest, readback receipt; two native Docs as inexact exports |
| `07_MODEL_ACCESSIBILITY_extras/` | the accessibility publication's audit folder (live title `04_AUDIT_AND_EXCEPTIONS`): the accessibility handoff and verification record, the completion report, `Start_Here.csv`, `Reading_Copies.csv` (the 392 reading copies whose ids the inventory does not list, because the publication does not inventory itself); the three structure-audit files also landed here |
| `DRIVE_STRUCTURE_AUDIT/` | the structure session's handoff, its rollback record (status, verbatim: `RECORDED_ONLY_NOT_EXECUTED`), completion report, the ZIP's member listing, a first-tab export of the renamed source-map Sheet, and `MOVES_VS_INVENTORY.md`, which resolves all 21 recorded moves against the inventory |
| `2026-09-18_FOLDER_STRUCTURE_MAINTENANCE/` | the folder-naming standard `OP-ORG-20260918-v1.0`, the review-scaffold template, the new triage index, the July triage index under its new name (bytes unchanged), and a two-level listing of the new `10_ORIGIN_PRESERVED_REVIEW_COLLECTIONS` container |
| `PATH_CHANGES.jsonl` | 238 inventory rows whose path changed: six month folders renamed and re-parented under new `2026` folders, thirteen origin-collection trees moved under one container, one index renamed. Derived here from the rollback record and equal to the session's own `path_changes: 238` |

| `CHANGED_SINCE_SNAPSHOT.jsonl`, `ENUMERATION_NOTE.md` | the full enumeration of items created or modified after the snapshot (336: 326 NEW, 6 MOVED, 4 modified in place), from two paginated Drive queries. It sees 77 stale paths; `PATH_CHANGES.jsonl` records 238 because the thirteen origin-collection moves did not advance `modifiedTime` and are known only from the rollback record. Trashed items are invisible to both |

What the delta does not do: it does not refresh the export (a scoped re-export
is the only way to do that), does not execute any rollback inverse, and changes
no status of anything. The Drive session's own numbers — `scientific_dispositions_changed: 0`,
`deletions: 0`, `permission_changes: 0` — are recorded as its statements, not
verified here.

## 2026-09-19

`2026-09-19/CHATGPT_CI_INTEGRITY_HARDENING_CANDIDATE/` — three objects a ChatGPT
session published on the Drive on 2026-09-19 14:59 UTC and announced on PR #2
(a CI-hardening patch, its handoff and its verification record), stored
byte-exact and **not applied**; the patch's digest equals the one the
announcement declares. See that directory's README for what it is and why
integration is the owner's decision.
