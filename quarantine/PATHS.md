# Vault and quarantine paths (non-activating)

Map for an agent reading this tip. It does not open Drive, does not fetch
bytes, and does not move a status. **Quarantine is not a source of truth.**
Engineering hygiene is not mathematical discharge.

`tools/vault_hygiene_check.py` recomputes the facts below from
`drive/inventory.jsonl`, `drive/vault_tree.txt`, the manifests, and the dated
coverage ledger. A drift fails that check.

## The letters `DO_NOT_OPEN` are not the vault

| Path token | What it is on this tip | What it is not |
|---|---|---|
| segment starting `99_DO_NOT_OPEN` | The vault folder `99_DO_NOT_OPEN — SUPERSEDED_MIRRORS_DEAD_ENDS_AND_TRAP_COPIES` (Drive `1VTiBaRlBvHGptiXqoEli4E5630nO7mNb`) plus five native Docs. Inventory metadata only. No stored manifest row. | A fetch list. A reading copy of one of those five Docs would open the vault. |
| segment starting `00_DO_NOT_OPEN_MANIFEST` | One Doc outside the vault, Drive `1tYi29H_hsTPwi2Tj75OlYiz8sGv2JJwGT_k7FyMgNd4`. Held as a reading copy (`exact: false`) under `drive/mirrors/01_ACTIVE_RESEARCH_PACKAGES — ROOT (the vault manifest)/`. | The object. The vault. A digest (the inventory declares none). |
| `DO_NOT_OPEN_BEFORE_HASH_FREEZE` | A different folder, under carry-forward canon. Not the vault. | A reason to open `99_DO_NOT_OPEN`. |
| titles containing `DO_NOT_OPEN` | `24_ADDENDUM — DO_NOT_OPEN vault routing` and `25_ADDENDUM — DO_NOT_OPEN live manifest update`, both stored reading copies in the peer-review lane. | Live authority for vault contents. |

Addendum 24 names manifest id `1dUylnMeC2QGTteIQBfxB4glJlxNGfwQMpQrV0vDU7as`
under a heading that says the id is outside the vault. `drive/vault_tree.txt`
places that id inside the vault, titled `ZZ_SUPERSEDED — DO_NOT_OPEN_MANIFEST
shell (pre-entry-rules)`. Following 24's id opens the vault. Addendum 25 names
`1tYi29H_hsTPwi2Tj75OlYiz8sGv2JJwGT_k7FyMgNd4` and calls 24's id superseded.
The peer-review lane README records the same contradiction and does not repair
24's body. This map does not repair it either.

## Tip paths

| Path | Role | Not |
|---|---|---|
| `quarantine/EXCLUSIONS.json` | Logical-exclusion metadata (carrier, member, digest, scope). | Bodies. A source of truth. An instruction to copy an excluded payload into `engine/`, `research/`, `packages/`, or `claims/`. |
| `quarantine/README.md` | How those exclusions are classed. | Evidence. |
| `drive/vault_tree.txt` | Seven metadata rows: the vault folder, the five vault Docs, and the external manifest. No digest column. | Contents. A fetch list. |
| `drive/mirrors/01_ACTIVE_RESEARCH_PACKAGES — ROOT (the vault manifest)/` | Reading copy of the external manifest. Holding it opens nothing inside the vault. | A mirror of the vault. |
| `drive/mirrors/90_QUARANTINE_AND_TRIAGE/` | Notices, reading copies of routing receipts, and index rows. The lane README says the excluded payloads were not ported. | The excluded bodies. A source of truth. Citeable evidence. |
| `registers/json/quarantine_index.json` | Register export the exclusion list is checked against. | A promotion of quarantined material. |

`tools/quarantine_check.py` still owns the exclusion register: agreement with
the register, archive-member digests, no excluded digest in a manifest, and
bound-member annotations. This map does not add a row to `EXCLUSIONS.json` and
does not mark any carrier present that the inventory does not already name.

## Dated ledger versus this tip

`drive/deltas/2026-09-19/DG-MIGRATION-20260919/coverage.jsonl` marks 40 rows
`HELD_NO_OPEN` with the reason "Standing DO_NOT_OPEN boundary; content not
fetched." That reason is a substring match on `DO_NOT_OPEN` in the path. Split
by the path, not by the reason text:

| Class | `HELD_NO_OPEN` rows | Stored manifest row on this tip |
|---|---:|---|
| vault native Docs | 5 | none |
| external manifest | 1 | the reading copy above. "content not fetched" is true of that dated pass and false of this tip |
| `DO_NOT_OPEN_BEFORE_HASH_FREEZE` | 32 | 31 stored. One id, `1cPEMAUyvOlqH8-PT7TnOV420uYHDUYn8`, is indexed tree-only (`BULK_DATA_OVER_STORE_SIZE_LIMIT`). This check does not fetch it and does not call it a vault object |
| other titles containing `DO_NOT_OPEN` | 2 | both addenda, as reading copies |

The vault folder itself is `FOLDER_METADATA`, not `HELD_NO_OPEN`. Storing a
hash-freeze object or an addendum did not open `99_DO_NOT_OPEN`. The ledger is
left as the dated record it is.

## Non-activation

Copying a vault path or a `90_QUARANTINE_AND_TRIAGE` path into `engine/`,
`research/`, `packages/`, or `claims/` does not make it authoritative. The
hygiene check refuses such a path. A stored manifest row whose Drive id is a
vault id is refused even when `drive_path` omits `99_DO_NOT_OPEN`.

A pass of the check, and this file existing, discharges nothing.

Inventable receipts under `docs/math_status_probes/` are under the same
bound. A vault id or a quarantine path named from that lane does not become
an inventable source of truth. A green run of this check or of
`tools/quarantine_check.py` does not discharge OBL-H5-JETMOD. The reader
note is the vault-path section of `docs/math_status_probes/README.md`.
Receipt bytes stay as already recorded. This map still adds no exclusion.
