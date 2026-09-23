# COLLISION PROPOSAL 2026-09-19b — third successor: the fourteen duplicate primary keys first surfaced by keying the relations, review_ledger and definitions tabs

> **THIS IS A PROPOSAL. IT REQUIRES OPERATOR ACTION.**
> **Nothing has been repaired.** No register row was edited, merged, deleted, reordered or
> reclassified. The 2026-09-18 export remains faithful: `registers/source/`, `registers/json/`
> and `registers/csv/` are byte-identical to `git HEAD`, and `tools/collision_proposal_check.py`
> `--proposal registers/collision_proposal_2026-09-19b.json` fails if that stops being true.

> **Status discipline.** Every claim, premise and obligation stands exactly as its source
> records it; nothing in this document moves one, and every gate remains where it was. No
> original prize problem is solved. The 2D upper/lower tracks are not composed with the 3D
> lifetime track anywhere in this document.

## What this is

`registers/KNOWN_FINDINGS.json` records, under its section `findings_first_keyed_2026-09-19`, **fourteen duplicate primary keys in three id-bearing tabs of the source registers** that `tools/registers_check.py` had never keyed until 2026-09-19 and that both exports had carried all along: eleven Relation IDs in the Relation Index (`REL-036`, `REL-037`, `REL-038`, `REL-039`, `REL-040`, `REL-048`, `REL-049`, `REL-132`, `REL-133`, `REL-134`, `REL-EC021-CLS141`), two Review IDs in the Review Independence ledger (`REV-P12-GP-006`, `REV-P02-GP-INTERVAL-001`) and one Definition ID in the Definition Registry (`DEF-049`). This document proposes, for each of the fourteen, a remedy drawn from the protocols' own text and from the workbook's own executed precedents. It executes none of them.

It is the **third numbered proposal** and a numbered successor to [`registers/COLLISION_PROPOSAL_2026-09-19.md`](COLLISION_PROPOSAL_2026-09-19.md) / [`registers/collision_proposal_2026-09-19.json`](collision_proposal_2026-09-19.json) (the 2026-09-19 proposal, 7 records over `findings_first_visible_in_2026-09-18_export`, SHA-256 `3fd2a40ff0d2cde28c4189c8cffacd4d2758395a1940cd45a3f202f336165eb5`, 133,840 bytes), which is itself a numbered successor to [`registers/COLLISION_PROPOSAL.md`](COLLISION_PROPOSAL.md) / [`registers/collision_proposal.json`](collision_proposal.json) (the 2026-09-18 proposal, 16 records over `findings`, SHA-256 `6055dad5ec41a8d5eaa59ef005bb2e84622217fe177649be7c2e34c5b6a28a82`, 368,189 bytes). Under CLAUDE.md rule 8 both predecessors are frozen: neither is edited, superseded, amended or reissued here, and this document carries `supersedes: null`. `successor_of` names the 2026-09-19 document by SHA-256; `predecessor_chain` lists both, and the checker walks the chain, verifies every level's digest on disk, and refuses any identifier either predecessor issued. Together the three documents cover all 37 findings in `KNOWN_FINDINGS.json`, each exactly once; `tests/test_collision_proposal.py` asserts that.

The machine-readable form is [`registers/collision_proposal_2026-09-19b.json`](collision_proposal_2026-09-19b.json). The checker is `tools/collision_proposal_check.py --proposal registers/collision_proposal_2026-09-19b.json`; the tests are `tests/test_collision_proposal.py`.

### Source of record

| | |
|---|---|
| Export | `registers/source/GP-REG-032_v1.2_export_2026-09-18.xlsx` |
| SHA-256 | `c3229ecefc642f3e32f23cb8320fb66236d11affb83f67bd135153d068a3f460` |
| Bytes | 1,919,741 |
| Provenance | `registers/source/SOURCES.json` (exact: false — an xlsx export is a rendering of the native Sheet) |
| JSON tabs quoted | `registers/json/relations.json` (workbook sheet 'Relation Index'), `registers/json/review_ledger.json` ('Review Independence'), `registers/json/definitions.json` ('Definition Registry') — generated from the xlsx by `tools/registers_import.py`, whose SHEETS map gives the sheet names |
| Row canonicalisation | `sha256(json.dumps(row, ensure_ascii=False, separators=(",",":")))` |
| Findings file | `registers/KNOWN_FINDINGS.json`, section `findings_first_keyed_2026-09-19` (14 findings) |
| Records | 14 (one per finding key) |
| Predecessor | `registers/collision_proposal_2026-09-19.json`, SHA-256 `3fd2a40ff0d2cde28c4189c8cffacd4d2758395a1940cd45a3f202f336165eb5`, 133,840 bytes — frozen, not edited |
| Predecessor's predecessor | `registers/collision_proposal.json`, SHA-256 `6055dad5ec41a8d5eaa59ef005bb2e84622217fe177649be7c2e34c5b6a28a82`, 368,189 bytes — frozen, not edited |

Every row quoted below is the list of cell strings exactly as the named JSON tab holds it at the stated 0-based row index, with the SHA-256 of its canonical serialisation recorded in the JSON. The 2026-09-17 markdown export delivered the relations tab only as a prefix and is not cited. Row indices are the 0-based indices `tools/registers_check.py` prints.

`tools/collision_proposal_check.py --proposal registers/collision_proposal_2026-09-19b.json` holds this companion to the live rows, not to the JSON proposal alone: every fenced row block below must equal the canonical serialisation of the live `registers/json/<tab>.json` row, every `rows[i], canonical N bytes, SHA-256` line must match that row, every `Cell count` line and every materiality paragraph must appear verbatim as recomputed, and every summary-table row must name the record's two rows, its keeper (the earlier row), its reidentified row (the later) and its successor id. The document-level `summary_counts` and the classification lists in the JSON are likewise recomputed from the records' cells. A number that appears here and is not recomputed by the checker is a defect of the checker, not a fact.

### Which cells the drive-object rule reads, per tab

`both_rows_cite_one_drive_object` is true exactly when every cell the tab's rule names agrees whole-cell, with no normalisation; the checker holds the same lists (`DRIVE_OBJECT_RULE`) and requires each record's `drive_object_rule_cells` to equal them.

| Tab | Cells the rule reads | Why |
|---|---|---|
| `relations` (Relation Index) | `Source URL` | The Drive object the relation is asserted from. Rows also carry a `Target URL`; it is compared separately (`target_url_cells_agree`) and reported, never folded into the flag. The relation a row describes is its (`Source object`, `Relation type`, `Target object`) triple, reported under `relation_triple_cells_agree`; the defect class must say `SAME_RELATION` exactly when that triple agrees. |
| `review_ledger` (Review Independence) | `Exact Object ID` | The tab carries no `Source`, `Source URL` or `Drive ID` column; the object a row reviews is named by `Exact Object ID` and by nothing else the export holds. `drive_source_cited_by_both_rows` is therefore null, and the checker requires it to be null for this tab. |
| `definitions` (Definition Registry) | `Source URL` | The Drive document the definition is drawn from. |

### Why the locators REL, RVL and DEF

A successor key `<id>@<LOCATOR>-R<row>` must say which tab's row it locates, so the locator is tab-scoped (`AIDX-R39` would name Artifact Index row 39, a different row of a different tab). `REL` and `DEF` are the first three letters of the machine tab names `relations` and `definitions`; `RVL` abbreviates `review_ledger` and is used instead of `REV` because `REV` is that tab's own id prefix. `REL` and `DEF` happen to coincide with their tabs' id prefixes (`REL-036`, `DEF-049`); in the `@…-R<n>` position that is harmless (no id in the export has that shape, and every compound key is checked against every whole cell and token of `registers/json/`), and it is recorded as a residual question in case the operator prefers locators that cannot be read as id prefixes. THE LOCATORS ARE AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM (`CL-AUD-LSDER036-20260727-01@1zMq504`, `CL-CLWO-20260727-07@1jqT3KJ`), NOT A LITERAL QUOTATION; ratifying the row-locator form once ratifies it for all three proposals.

### Reviewer standing and independence

- **Session family:** Anthropic. **Role:** PROPOSAL AUTHOR — NOT A REVIEWER.
- **`independence_credit = 0`.** This session is Anthropic-family. OP-PROT-019-v1.1 R17 §4 permits a fresh nonauthor session of any provider to perform technical review, but records organizational independence separately and at ZERO for a same-provider reviewer. Independently of that, a PROPOSAL is not a review at all: it licenses no technical verdict and no independence credit on any object, for any provider.
- **Every independence-requiring gate REMAINS OPEN.** Every independence-requiring gate in this program REMAINS OPEN, unchanged by this document, regardless of any technical judgement expressed here. Nothing in this proposal is a technical pass, and no technical pass would satisfy an independence predicate in any case (R17 §4: 'Never relabel an independence-required theorem terminal solely because its technical review passed.'). The Independence Score and Independence Class cells of the four Review Independence rows quoted here are the ledger's own words, transcribed and not evaluated.
- **Authorship of the objects examined.** The Relation Index and Definition Registry rows carry no Org cell; the four Review Independence rows carry Reviewer Line 'GP' and a Reviewer cell naming the GP line's product, quoted verbatim in the rows and not repeated in any prose of this document. None records the family of this session. This changes nothing: the credit recorded here is zero because a proposal licenses zero, not because of provider matching.

### What this document does NOT establish

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

### The collision registry this proposal writes to

The workbook's append-only collision registry is the 'Duplicate Flags' tab of GP-REG-032-v1.2 (registers/json/duplicate_flags.json), which already carries 31 collision clusters including DUP-ID-GP-DER-044, the collision OP-CNS-001 §2 names by hand, and DUP-CST073-20260727 / DUP-ID-HELP-BOARD-20260724, the workbook's own records of one primary key on two different bodies.

`OP-CNS-001-R0.2 §2` requires that *"A later collision is preserved and disambiguated; historical
artifacts are not silently rewritten"*, that the program *"Maintain an append-only collision
registry"*, and that *"Gaps or duplicates require an additive provenance disclosure."* The remedy
below is that remedy, not a substitute for it.

### Operation discipline

No operation in this proposal edits, merges, deletes or reorders any existing row in any register. Cell rewrites in the SOURCE workbook that a full remedy would eventually require are listed separately under each record's non_additive_followups, marked operator_reserved, and are explicitly NOT proposed for autonomous execution.

- Allowed executable operations: `APPEND_ROW`.
- Every record's `operations` block contains appends only.
- Cell rewrites that a full remedy would eventually need appear under `non_additive_followups`,
  marked `operator_reserved: true`, and are **explicitly not proposed for autonomous execution**.
  They would change the live workbook and break `tools/registers_import.py --check` until a fresh
  dated export is taken. That is the operator's call, not this document's.

### Batch-level artifacts that would also be appended

- **NCC-REG032-COLLISION-20260919-002** — Records that the whole batch changes no mathematical statement, no schema, no verifier and no status. Precedent: NCC-P01-TR007-COLLISION, the certificate the workbook issued for its own TR-P01-007 collision repair; NCC-REG032-COLLISION-20260918-001 and NCC-REG032-COLLISION-20260919-001, the identifiers the two earlier proposals proposed for their own batches (proposed, not created).
- **GP-COR-212** — Next unused number above GP-COR-211, the identifier the 2026-09-19 proposal proposed (the complete 2026-09-18 export's highest assigned GP-COR is GP-COR-210). GP-COR-204 and GP-COR-211 remain the earlier frozen documents' proposed identifiers and are not reissued here. Precedent: GP-COR-143-v1.0 (the 2026-07-23 sweep of duplicate rows across five tabs, Relation Index row 239), GP-COR-190 (Help Board key collisions), GP-COR-192 (P02-LM012 multi-ID collision), cited in the Duplicate Flags rows for those repairs.
- Both are PROPOSED identifiers. Neither is created by this document. Whether the three proposals' batches should share one correction record when executed is an operator decision recorded under residual_questions_document_level.

### How the fourteen pairs classify

| Class | Pairs |
|---|---|
| Same Drive object, same relation, different scope / effect / provenance / date cells | REL-EC021-CLS141 (rows 232 and 235) |
| Different objects under one identifier | REL-036 (rows 35 and 39); REL-037 (rows 36 and 40); REL-038 (rows 37 and 41); REL-039 (rows 38 and 42); REL-040 (rows 43 and 44); REL-048 (rows 52 and 58); REL-049 (rows 53 and 59); REL-132 (rows 138 and 141); REL-133 (rows 139 and 142); REL-134 (rows 140 and 143); REV-P12-GP-006 (rows 15 and 17); REV-P02-GP-INTERVAL-001 (rows 19 and 21); DEF-049 (rows 48 and 50) |
| Exact duplicate rows | none |

The same-object bucket is named 'same_object_different_cells' rather than the earlier documents' 'same_object_different_status_text' because the one pair in it, REL-EC021-CLS141, carries one Status ('TERMINAL') in both rows and differs in other cells. REL-EC021-CLS141 was tested for cell-for-cell identity because KNOWN_FINDINGS describes the rows as identical apart from Last reviewed; the test is recorded in the record (exact_duplicate_row: false, six of fifteen cells differ, nine agree; the counts are recomputed by the checker from the quoted rows), so the EXACT_DUPLICATE_ROW class and the VOID-DUPLICATE disposition are discussed in that record and not applied anywhere in this proposal. The thirteen different-object pairs are one register key on two different relations, reviews or definitions: for the Relation Index the Source URL cells differ in every one of the ten, for the Review Independence ledger the Exact Object ID cells differ in both, for the Definition Registry the Source URL cells differ.

### Relation to the earlier proposals, and what differs

- `supersedes: null`; `successor_of` names the frozen 2026-09-19 document with its SHA-256, and `predecessor_chain` lists it and the frozen 2026-09-18 document it names in turn.
- `findings_source_section: findings_first_keyed_2026-09-19`; the predecessors cover `findings_first_visible_in_2026-09-18_export` and `findings`.
- `source_of_record.kind: xlsx_export_json_rows`, as the 2026-09-19 document; three JSON tabs are quoted instead of two.
- Successor identifiers follow the predecessors' shape (`<id>@<LOCATOR>-R<row>`) with the tab-scoped locators `REL`, `RVL` and `DEF`; none collides with any identifier in `registers/json/` or with any successor, cluster or batch identifier either predecessor issued (checked mechanically, chain two deep).
- Thirteen of the fourteen pairs are one key on two different objects, the case the 2026-09-19 document met once (`GP-REQ-194-v1.0`); the class and disposition of that record are reused. The one same-object pair differs in cells other than Status, so the same-object bucket is named `same_object_different_cells` rather than `..._different_status_text`.
- The correction-record identifier is `GP-COR-212`, above the 2026-09-19 document's `GP-COR-211`; the no-change certificate is `NCC-REG032-COLLISION-20260919-002`.

### Residual questions at document level

- Whether the fourteen appends of this proposal, the seven of the 2026-09-19 proposal and the sixteen of the 2026-09-18 proposal are executed as one batch under one no-change certificate and one correction record, or as three, is an operator decision; the identifiers proposed here are distinct so that any choice remains open.
- Whether the row-locator discriminator (@REL-R<n>, @RVL-R<n>, @DEF-R<n>, alongside @AIDX-R<n> and @EVL-R<n>) is ratified is one decision for all three proposals; this document does not ratify it. REL and DEF coincide with their tabs' id prefixes; the operator may prefer locators that cannot be read as id prefixes (for instance RIX and DFR), in which case the fourteen successor ids change shape but nothing else in this document changes.
- Thirteen of the fourteen pairs are one key on two different objects, for which the workbook's own executed remedy elsewhere was to renumber the later row to the next unused number of its series (DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726). The row-locator key is adopted here only so that the three proposals issue keys of one shape; the renumbering alternative is recorded in each record and is the operator's to prefer.
- The bare ids REV-P12-GP-006 and REV-P02-GP-INTERVAL-001 are cited by other tabs (Evidence Lineage rows 36, 37, 39, 47, 48; Relation Index row 22; Transition Alarms row 59); re-keying a later row leaves those citations as they are, and for REV-P02-GP-INTERVAL-001 three of them name the later row's object. A cross-tab citation repair is part of any full remedy and is not proposed here.
- The keeper rule identifies the exported row order with the confirmed append position. The REL-EC021-CLS141 and REV-P12-GP-006 records show regions whose row order is not chronological by the rows' own date cells; the operator should confirm append order from the workbook's revision history before executing any record.
- registers/KNOWN_FINDINGS.json described REL-EC021-CLS141 as an exact duplicate row apart from its review date, and registers/README.md repeated that wording, until the commit that adds this document; the cells give six differing cells, and both now record the recomputed counts, each saying what it said before. Those are this repository's own rationale and prose, not exported data. The finding key itself is correct and unchanged, and the register is untouched.
- The task text that commissioned this document named registers/SOURCES.json as the digest record; the file is registers/source/SOURCES.json. The digest and byte count recorded here were re-computed from the xlsx on disk and agree with that file. The same task text described REL-EC021-CLS141 as identical except for its 'Last reviewed' cell; the cells were read and the record follows the cells.

## Part 1 — Eleven duplicate Relation IDs in the Relation Index

Ten of the eleven pairs are **two different relations under one identifier**: different Source objects, Relation types, Target objects and Source URLs, each in a block of consecutive rows that was numbered from a number an earlier block had already taken (rows 35–38 versus 39–43 and row 44; rows 52–53 versus 58–59; rows 138–140 versus 141–143). That is the case `OP-CNS-001 §2`'s GP-DER-044 example and acceptance test T7 describe, and the case the 2026-09-19 proposal met once. Both relations are preserved in every pair; the identifier is disambiguated. The eleventh pair, `REL-EC021-CLS141`, is the other case — **one relation from one Drive document registered twice**, with six cells differing and neither row an exact copy of the other — and it is the unswept face of an EC-021 double registration the workbook corrected in four other tabs on 2026-07-23.

| Relation ID | Rows | Class | Keeper row | Reidentified row | Proposed successor row key |
|---|---|---|---|---|---|
| `REL-036` | 35, 39 | DIFFERENT OBJECTS | 35 | 39 | `REL-036@REL-R39` |
| `REL-037` | 36, 40 | DIFFERENT OBJECTS | 36 | 40 | `REL-037@REL-R40` |
| `REL-038` | 37, 41 | DIFFERENT OBJECTS | 37 | 41 | `REL-038@REL-R41` |
| `REL-039` | 38, 42 | DIFFERENT OBJECTS | 38 | 42 | `REL-039@REL-R42` |
| `REL-040` | 43, 44 | DIFFERENT OBJECTS | 43 | 44 | `REL-040@REL-R44` |
| `REL-048` | 52, 58 | DIFFERENT OBJECTS | 52 | 58 | `REL-048@REL-R58` |
| `REL-049` | 53, 59 | DIFFERENT OBJECTS | 53 | 59 | `REL-049@REL-R59` |
| `REL-132` | 138, 141 | DIFFERENT OBJECTS | 138 | 141 | `REL-132@REL-R141` |
| `REL-133` | 139, 142 | DIFFERENT OBJECTS | 139 | 142 | `REL-133@REL-R142` |
| `REL-134` | 140, 143 | DIFFERENT OBJECTS | 140 | 143 | `REL-134@REL-R143` |
| `REL-EC021-CLS141` | 232, 235 | same relation, different scope / effect / provenance / date cells | 232 | 235 | `REL-EC021-CLS141@REL-R235` |

**Keeper rule, applied mechanically to all eleven.** Earlier confirmed append position. `R17 §3`: *"Use the confirmed append position to order contenders, not a self-reported timestamp."* This is the rule both earlier proposals applied, reused unchanged. Ten pairs carry one `Last reviewed` date in both rows, so the date could not order them in any case; the eleventh (`REL-EC021-CLS141`) has its append order and its date order inverted, and the record says why that needs operator confirmation.

**Keeping the bare key is a KEY assignment, not a currency verdict.** It does not make the keeper row's text current, nor the reidentified row's text stale, and for the ten different-object pairs it ranks nothing.

### 1.1 `REL-036` — rows 35, 39

> Finding key: `relations: duplicate key 'REL-036' at rows 35 and 39`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 35: Source object 'File 4 Section 3.2 equations (3.1)–(3.4)'; Relation type 'DEFINES'; Target object 'Corrected six-pin pair frame'; Status 'ACTIVE'. Row 39: Source object 'GP-AUD-061-v1.0'; Relation type 'PROPOSES_AMENDMENT_TO'; Target object 'GP-DATA-054-v1.0 pairing instrument'; Status 'AUTHOR-REPAIR-CANDIDATE'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 35 → https://drive.google.com/file/d/1lbZzqFOwMubvnD0KBqT9pTlIIIXxUnTA/view; row 39 → https://docs.google.com/document/d/1dPk0sZ14zLbkXxiTvbggVWtOmQRS7Fym3rk9oO-vv1Q/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 35 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[35], canonical 616 bytes, SHA-256 `550fcb4d0d4cd8ad24e4335a14ea232a0e04cd4b833ff888186e0c07097248f5`:

```json
["REL-036","File 4 Section 3.2 equations (3.1)–(3.4)","[[SRC:FILE4-PAIR-PALM-3.1-3.4]]","DEFINES","[[REL:DEFINES]]","Corrected six-pin pair frame","[[DEF:CORRECTED_SIX_PIN_PAIR_FRAME]]","Provides the authoritative raw ordering, corrected value-difference coordinate, symmetric gradient sums, and r-normalized differences.","Primary formula evidence","No closure or theorem authority by itself","https://drive.google.com/file/d/1lbZzqFOwMubvnD0KBqT9pTlIIIXxUnTA/view","https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit","File 4 PDF, visually checked page 4","ACTIVE","2026-07-21"]
```

#### Row 39 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[39], canonical 656 bytes, SHA-256 `e48de74bb35c31455d1c884ef8d16496077cdc18e7e6457281fe7bb2ee667d8f`:

```json
["REL-036","GP-AUD-061-v1.0","[[OBJ:GP-AUD-061-v1.0]]","PROPOSES_AMENDMENT_TO","[[REL:PROPOSES_AMENDMENT_TO]]","GP-DATA-054-v1.0 pairing instrument","[[OBJ:GP-DATA-054-v1.0]]","Resolves the artifact-ID collision by retaining GP-AUD-059 for the v1.1 author receipt and retaining GP-AUD-061 for the stale-export/current-v1.0 correction.","Author repair specification and claimed implementation","No production or theorem authority","https://docs.google.com/document/d/1dPk0sZ14zLbkXxiTvbggVWtOmQRS7Fym3rk9oO-vv1Q/edit","https://docs.google.com/document/d/1n5rmtvI8Xn5GIX1TUGI4KLOcYg4cC91lI6gjPQOSjdE/edit","TR-P12-008","AUTHOR-REPAIR-CANDIDATE","2026-07-21"]
```

#### Where the two rows differ

| Field | Row 35 (keeper) | Row 39 (reidentified) |
|---|---|---|
| **Source object** | File 4 Section 3.2 equations (3.1)–(3.4) | GP-AUD-061-v1.0 |
| **Source hyper-tag** | [[SRC:FILE4-PAIR-PALM-3.1-3.4]] | [[OBJ:GP-AUD-061-v1.0]] |
| **Relation type** | DEFINES | PROPOSES_AMENDMENT_TO |
| **Relation hyper-tag** | [[REL:DEFINES]] | [[REL:PROPOSES_AMENDMENT_TO]] |
| **Target object** | Corrected six-pin pair frame | GP-DATA-054-v1.0 pairing instrument |
| **Target hyper-tag** | [[DEF:CORRECTED_SIX_PIN_PAIR_FRAME]] | [[OBJ:GP-DATA-054-v1.0]] |
| **Exact scope / meaning** | Provides the authoritative raw ordering, corrected value-difference coordinate, symmetric gradient sums, and r-normalized differences. | Resolves the artifact-ID collision by retaining GP-AUD-059 for the v1.1 author receipt and retaining GP-AUD-061 for the stale-export/current-v1.0 correction. |
| **Evidentiary effect** | Primary formula evidence | Author repair specification and claimed implementation |
| **Authority effect** | No closure or theorem authority by itself | No production or theorem authority |
| **Source URL** | https://drive.google.com/file/d/1lbZzqFOwMubvnD0KBqT9pTlIIIXxUnTA/view | https://docs.google.com/document/d/1dPk0sZ14zLbkXxiTvbggVWtOmQRS7Fym3rk9oO-vv1Q/edit |
| **Target URL** | https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit | https://docs.google.com/document/d/1n5rmtvI8Xn5GIX1TUGI4KLOcYg4cC91lI6gjPQOSjdE/edit |
| **Provenance** | File 4 PDF, visually checked page 4 | TR-P12-008 |
| **Status** | ACTIVE | AUTHOR-REPAIR-CANDIDATE |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 35 and 39 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations. Row 35 relates 'File 4 Section 3.2 equations (3.1)–(3.4)' DEFINES 'Corrected six-pin pair frame' (Status 'ACTIVE', Provenance 'File 4 PDF, visually checked page 4', Source URL a Drive PDF 1lbZzqFOwMubvnD0KBqT9pTlIIIXxUnTA); row 39 relates 'GP-AUD-061-v1.0' PROPOSES_AMENDMENT_TO 'GP-DATA-054-v1.0 pairing instrument' (Status 'AUTHOR-REPAIR-CANDIDATE', Provenance 'TR-P12-008', Source URL document 1dPk0sZ14zLbkXxiTvbggVWtOmQRS7Fym3rk9oO-vv1Q). Only Relation ID and Last reviewed ('2026-07-21') agree; thirteen of fifteen cells differ, including every cell that identifies the relation. This is one identifier assigned twice, not two readings of one relation; there is no text to adjudicate between the rows.

**Corroborating register evidence.** Rows 35–38 carry REL-036, -037, -038, -039 as one topical block (the EC-014 corrected six-pin pair frame: Provenance 'File 4 PDF, visually checked page 4', 'EV-EC014-GP-RECON', 'EC-014', 'OP-PROT-004; GP-REQ-057 acceptance rule'); rows 39–43 carry REL-036, -037, -038, -039, -040 as a second topical block (the GP-DATA-054 pairing-instrument incident: Provenance 'TR-P12-008' in rows 39, 40, 41, 43 and 'TR-P12-009' in row 42); row 44 carries REL-040 a second time (its only other occurrence is row 43) and row 45 continues at REL-041. Both blocks carry Last reviewed '2026-07-21', so the date cannot order them; the second block re-used the numbers the first had just taken. OP-CNS-001 §2: 'Failed create/upload attempts should be verified absent before retry' and 'Concurrent work should disclose line/session identity.'

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 35** keeps `REL-036`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 39 → `REL-036@REL-R39`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-036@1dPk0sZ for row 39; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 39 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL036-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 35 (2026-09-18 xlsx export) — REL-036 — 'File 4 Section 3.2 equations (3.1)–(3.4)' DEFINES 'Corrected six-pin pair frame' — Status: ACTIVE |
| `File B` | Relation Index row 39 (2026-09-18 xlsx export) — REL-036 — 'GP-AUD-061-v1.0' PROPOSES_AMENDMENT_TO 'GP-DATA-054-v1.0 pairing instrument' — Status: AUTHOR-REPAIR-CANDIDATE |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://drive.google.com/file/d/1lbZzqFOwMubvnD0KBqT9pTlIIIXxUnTA/view versus https://docs.google.com/document/d/1dPk0sZ14zLbkXxiTvbggVWtOmQRS7Fym3rk9oO-vv1Q/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 35 as REL-036; register row 39 additively under compound key REL-036@REL-R39; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 39, column 'Relation ID'; `REL-036` → `REL-036@REL-R39`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number (the remedy Duplicate Flags DUP-CST073-20260727 and DUP-ID-HELP-BOARD-20260724 record for other tabs; the export's highest plain 'REL-<nnn>' is REL-225) is not decided here; that would change a key the tab's other rows may cite by number.
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.2 `REL-037` — rows 36, 40

> Finding key: `relations: duplicate key 'REL-037' at rows 36 and 40`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 36: Source object 'GP-DER-061-v1.0'; Relation type 'RECONSTRUCTS'; Target object 'Corrected six-pin pair frame'; Status 'ACTIVE-CANDIDATE'. Row 40: Source object 'GP-AUD-059-v1.0 amended author receipt'; Relation type 'AUTHOR_SELF_TESTS'; Target object 'GP-DATA-054-v1.1 claimed source'; Status 'AUTHOR-TEST-PASS / SOURCE-GATE-FAIL'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 36 → https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit; row 40 → https://docs.google.com/document/d/125eYLOvqR9BWBfG7713Jg8dg8ugYBDIkldDiAhv_bOA/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 36 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[36], canonical 574 bytes, SHA-256 `e73ee5afcf2a8c1bbc4392ad422baa297beaf01016cd67222f2b7f9043ff2e88`:

```json
["REL-037","GP-DER-061-v1.0","[[OBJ:GP-DER-061-v1.0]]","RECONSTRUCTS","[[REL:RECONSTRUCTS]]","Corrected six-pin pair frame","[[DEF:CORRECTED_SIX_PIN_PAIR_FRAME]]","Displays T_r entry by entry and proves det T_r=-r^-5 by block factorization and exact symbolic arithmetic.","Exact GP-side reconstruction evidence","Same-line only; cannot satisfy independent gate","https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit","https://drive.google.com/file/d/1lbZzqFOwMubvnD0KBqT9pTlIIIXxUnTA/view","EV-EC014-GP-RECON","ACTIVE-CANDIDATE","2026-07-21"]
```

#### Row 40 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[40], canonical 644 bytes, SHA-256 `61161cae31e9d99f46d5eb8db2cde862ac7437d3e0514f77e9e9440278857c1f`:

```json
["REL-037","GP-AUD-059-v1.0 amended author receipt","[[OBJ:GP-AUD-059-v1.0]]","AUTHOR_SELF_TESTS","[[REL:AUTHOR_SELF_TESTS]]","GP-DATA-054-v1.1 claimed source","[[OBJ:GP-DATA-054-v1.1]]","Reports 21 of 21 author-side controls, but the exact source cannot be reconstructed from the cited failed staging capsule.","Internal author evidence only","Cannot satisfy source identity or independent qualification","https://docs.google.com/document/d/125eYLOvqR9BWBfG7713Jg8dg8ugYBDIkldDiAhv_bOA/edit","https://docs.google.com/document/d/1Hur9mCBvTtaMN4jHEP94cB5t_E2YcHtS9xKo7WZK1cQ/edit","TR-P12-008","AUTHOR-TEST-PASS / SOURCE-GATE-FAIL","2026-07-21"]
```

#### Where the two rows differ

| Field | Row 36 (keeper) | Row 40 (reidentified) |
|---|---|---|
| **Source object** | GP-DER-061-v1.0 | GP-AUD-059-v1.0 amended author receipt |
| **Source hyper-tag** | [[OBJ:GP-DER-061-v1.0]] | [[OBJ:GP-AUD-059-v1.0]] |
| **Relation type** | RECONSTRUCTS | AUTHOR_SELF_TESTS |
| **Relation hyper-tag** | [[REL:RECONSTRUCTS]] | [[REL:AUTHOR_SELF_TESTS]] |
| **Target object** | Corrected six-pin pair frame | GP-DATA-054-v1.1 claimed source |
| **Target hyper-tag** | [[DEF:CORRECTED_SIX_PIN_PAIR_FRAME]] | [[OBJ:GP-DATA-054-v1.1]] |
| **Exact scope / meaning** | Displays T_r entry by entry and proves det T_r=-r^-5 by block factorization and exact symbolic arithmetic. | Reports 21 of 21 author-side controls, but the exact source cannot be reconstructed from the cited failed staging capsule. |
| **Evidentiary effect** | Exact GP-side reconstruction evidence | Internal author evidence only |
| **Authority effect** | Same-line only; cannot satisfy independent gate | Cannot satisfy source identity or independent qualification |
| **Source URL** | https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit | https://docs.google.com/document/d/125eYLOvqR9BWBfG7713Jg8dg8ugYBDIkldDiAhv_bOA/edit |
| **Target URL** | https://drive.google.com/file/d/1lbZzqFOwMubvnD0KBqT9pTlIIIXxUnTA/view | https://docs.google.com/document/d/1Hur9mCBvTtaMN4jHEP94cB5t_E2YcHtS9xKo7WZK1cQ/edit |
| **Provenance** | EV-EC014-GP-RECON | TR-P12-008 |
| **Status** | ACTIVE-CANDIDATE | AUTHOR-TEST-PASS / SOURCE-GATE-FAIL |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 36 and 40 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations. Row 36 relates 'GP-DER-061-v1.0' RECONSTRUCTS 'Corrected six-pin pair frame' (Status 'ACTIVE-CANDIDATE', Provenance 'EV-EC014-GP-RECON'); row 40 relates 'GP-AUD-059-v1.0 amended author receipt' AUTHOR_SELF_TESTS 'GP-DATA-054-v1.1 claimed source' (Status 'AUTHOR-TEST-PASS / SOURCE-GATE-FAIL', Provenance 'TR-P12-008'). The Source URL cells name two different Drive documents (1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8 and 125eYLOvqR9BWBfG7713Jg8dg8ugYBDIkldDiAhv_bOA). Only Relation ID and Last reviewed agree; thirteen of fifteen cells differ. One identifier assigned twice; nothing to adjudicate between the rows.

**Corroborating register evidence.** Second member of the two four-row blocks described under REL-036 (rows 35–38 versus rows 39–43). Row 36's Source URL (1LFs1…, GP-DER-061-v1.0) is the Target URL of row 35 and the Source URL of row 37, tying rows 35–37 to one derivation; row 40's Source URL (125eY…, GP-AUD-059-v1.0) is the Target URL of rows 42 and 44, tying rows 39–44 to the GP-DATA-054 incident.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 36** keeps `REL-037`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 40 → `REL-037@REL-R40`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-037@125eYLO for row 40; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 40 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL037-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 36 (2026-09-18 xlsx export) — REL-037 — 'GP-DER-061-v1.0' RECONSTRUCTS 'Corrected six-pin pair frame' — Status: ACTIVE-CANDIDATE |
| `File B` | Relation Index row 40 (2026-09-18 xlsx export) — REL-037 — 'GP-AUD-059-v1.0 amended author receipt' AUTHOR_SELF_TESTS 'GP-DATA-054-v1.1 claimed source' — Status: AUTHOR-TEST-PASS / SOURCE-GATE-FAIL |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit versus https://docs.google.com/document/d/125eYLOvqR9BWBfG7713Jg8dg8ugYBDIkldDiAhv_bOA/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 36 as REL-037; register row 40 additively under compound key REL-037@REL-R40; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 40, column 'Relation ID'; `REL-037` → `REL-037@REL-R40`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number is not decided here (see REL-036).
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.3 `REL-038` — rows 37, 41

> Finding key: `relations: duplicate key 'REL-038' at rows 37 and 41`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 37: Source object 'GP-DER-061-v1.0'; Relation type 'RESPONDS_TO'; Target object 'GP-REQ-057-v1.0'; Status 'ACTIVE'. Row 41: Source object 'GP-DATA-054-v1.1 failed staging capsule'; Relation type 'FAILS_PUBLICATION_GATE_FOR'; Target object 'GP-DATA-054-v1.1 claimed source'; Status 'ACTIVE-CRITICAL-BLOCKER'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 37 → https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit; row 41 → https://docs.google.com/document/d/1Hur9mCBvTtaMN4jHEP94cB5t_E2YcHtS9xKo7WZK1cQ/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 37 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[37], canonical 554 bytes, SHA-256 `b0ddd70fdb62ee03cf61a2a5114c5cbe98df0e826b75632f36ff33a6437d8565`:

```json
["REL-038","GP-DER-061-v1.0","[[OBJ:GP-DER-061-v1.0]]","RESPONDS_TO","[[REL:RESPONDS_TO]]","GP-REQ-057-v1.0","[[OBJ:GP-REQ-057-v1.0]]","Discharges the matrix-visibility and GP-side derivation tasks while preserving the request’s non-GP and second-line requirements.","Reduces remaining distance; zero independent-review credit","No queue closure","https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit","https://docs.google.com/document/d/1V-2e8oeYrf2Amgm2wzkSMAlpXDTQWgxEC5tHSTFKVLs/edit","EC-014","ACTIVE","2026-07-21"]
```

#### Row 41 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[41], canonical 702 bytes, SHA-256 `9bb3c10efa2e7f9ba8b853d200b0608b3d1e22da16aa6f5a194534e8b754a190`:

```json
["REL-038","GP-DATA-054-v1.1 failed staging capsule","[[OBJ:GP-DATA-054-v1.1-FAILED-STAGING]]","FAILS_PUBLICATION_GATE_FOR","[[REL:FAILS_PUBLICATION_GATE_FOR]]","GP-DATA-054-v1.1 claimed source","[[GATE:PAIRING_V11_SOURCE_IDENTITY]]","Contains only a terminal Base64 fragment and END marker; lacks BEGIN marker and complete payload.","Direct source-publication failure evidence","Blocks independent reconstruction, execution, and production","https://docs.google.com/document/d/1Hur9mCBvTtaMN4jHEP94cB5t_E2YcHtS9xKo7WZK1cQ/edit","https://docs.google.com/document/d/1dPk0sZ14zLbkXxiTvbggVWtOmQRS7Fym3rk9oO-vv1Q/edit","TR-P12-008; fresh download and full Doc read","ACTIVE-CRITICAL-BLOCKER","2026-07-21"]
```

#### Where the two rows differ

| Field | Row 37 (keeper) | Row 41 (reidentified) |
|---|---|---|
| **Source object** | GP-DER-061-v1.0 | GP-DATA-054-v1.1 failed staging capsule |
| **Source hyper-tag** | [[OBJ:GP-DER-061-v1.0]] | [[OBJ:GP-DATA-054-v1.1-FAILED-STAGING]] |
| **Relation type** | RESPONDS_TO | FAILS_PUBLICATION_GATE_FOR |
| **Relation hyper-tag** | [[REL:RESPONDS_TO]] | [[REL:FAILS_PUBLICATION_GATE_FOR]] |
| **Target object** | GP-REQ-057-v1.0 | GP-DATA-054-v1.1 claimed source |
| **Target hyper-tag** | [[OBJ:GP-REQ-057-v1.0]] | [[GATE:PAIRING_V11_SOURCE_IDENTITY]] |
| **Exact scope / meaning** | Discharges the matrix-visibility and GP-side derivation tasks while preserving the request’s non-GP and second-line requirements. | Contains only a terminal Base64 fragment and END marker; lacks BEGIN marker and complete payload. |
| **Evidentiary effect** | Reduces remaining distance; zero independent-review credit | Direct source-publication failure evidence |
| **Authority effect** | No queue closure | Blocks independent reconstruction, execution, and production |
| **Source URL** | https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit | https://docs.google.com/document/d/1Hur9mCBvTtaMN4jHEP94cB5t_E2YcHtS9xKo7WZK1cQ/edit |
| **Target URL** | https://docs.google.com/document/d/1V-2e8oeYrf2Amgm2wzkSMAlpXDTQWgxEC5tHSTFKVLs/edit | https://docs.google.com/document/d/1dPk0sZ14zLbkXxiTvbggVWtOmQRS7Fym3rk9oO-vv1Q/edit |
| **Provenance** | EC-014 | TR-P12-008; fresh download and full Doc read |
| **Status** | ACTIVE | ACTIVE-CRITICAL-BLOCKER |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 37 and 41 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations. Row 37 relates 'GP-DER-061-v1.0' RESPONDS_TO 'GP-REQ-057-v1.0' (Status 'ACTIVE', Provenance 'EC-014'); row 41 relates 'GP-DATA-054-v1.1 failed staging capsule' FAILS_PUBLICATION_GATE_FOR 'GP-DATA-054-v1.1 claimed source' (Status 'ACTIVE-CRITICAL-BLOCKER', Provenance 'TR-P12-008; fresh download and full Doc read'). Source URLs differ (1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8 versus 1Hur9mCBvTtaMN4jHEP94cB5t_E2YcHtS9xKo7WZK1cQ). Only Relation ID and Last reviewed agree; thirteen of fifteen cells differ. One identifier assigned twice.

**Corroborating register evidence.** Third member of the two blocks described under REL-036. Row 41's Source URL (1Hur9…) is the Target URL of row 40, and row 41's Status 'ACTIVE-CRITICAL-BLOCKER' is repeated by row 43 (REL-040, the compressed-capsule face of the same publication-gate failure).

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 37** keeps `REL-038`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 41 → `REL-038@REL-R41`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-038@1Hur9mC for row 41; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 41 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL038-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 37 (2026-09-18 xlsx export) — REL-038 — 'GP-DER-061-v1.0' RESPONDS_TO 'GP-REQ-057-v1.0' — Status: ACTIVE |
| `File B` | Relation Index row 41 (2026-09-18 xlsx export) — REL-038 — 'GP-DATA-054-v1.1 failed staging capsule' FAILS_PUBLICATION_GATE_FOR 'GP-DATA-054-v1.1 claimed source' — Status: ACTIVE-CRITICAL-BLOCKER |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit versus https://docs.google.com/document/d/1Hur9mCBvTtaMN4jHEP94cB5t_E2YcHtS9xKo7WZK1cQ/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 37 as REL-038; register row 41 additively under compound key REL-038@REL-R41; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 41, column 'Relation ID'; `REL-038` → `REL-038@REL-R41`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number is not decided here (see REL-036).
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.4 `REL-039` — rows 38, 42

> Finding key: `relations: duplicate key 'REL-039' at rows 38 and 42`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 38: Source object 'EC-014'; Relation type 'REQUIRES_REVIEW_FROM'; Target object 'Two distinct lines among CL, AO48, CW/C047R'; Status 'OPEN-RESPONSE-REQUIRED'. Row 42: Source object 'GP-AUD-061-v1.0'; Relation type 'DISTINCT_FROM'; Target object 'GP-AUD-059-v1.0 amended author receipt'; Status 'ACTIVE-COLLISION-RESOLVED'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 38 → https://docs.google.com/document/d/1V-2e8oeYrf2Amgm2wzkSMAlpXDTQWgxEC5tHSTFKVLs/edit; row 42 → https://docs.google.com/document/d/1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 38 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[38], canonical 659 bytes, SHA-256 `48685093b31b71494653a654559a3cb5da82c7562eb6b5da2eae40fe8788262a`:

```json
["REL-039","EC-014","[[QUEUE:EC-014]]","REQUIRES_REVIEW_FROM","[[REL:REQUIRES_REVIEW_FROM]]","Two distinct lines among CL, AO48, CW/C047R","[[ROSTER:ACTIVE_NONAUTHOR_LINES]]","One line independently reconstructs all matrix and determinant details; a different line checks mutation, density, and excluded-scope boundary.","Explicit responses required; silence has no effect","Human approval remains later","https://docs.google.com/document/d/1V-2e8oeYrf2Amgm2wzkSMAlpXDTQWgxEC5tHSTFKVLs/edit","https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit","OP-PROT-004; GP-REQ-057 acceptance rule","OPEN-RESPONSE-REQUIRED","2026-07-21"]
```

#### Row 42 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[42], canonical 609 bytes, SHA-256 `7ef0dce0e8308f2fe9994e795414cd33edd751c9e541f34c8ed1e18a9a1f2e40`:

```json
["REL-039","GP-AUD-061-v1.0","[[OBJ:GP-AUD-061-v1.0]]","DISTINCT_FROM","[[REL:DISTINCT_FROM]]","GP-AUD-059-v1.0 amended author receipt","[[OBJ:GP-AUD-059-v1.0]]","Resolves concurrent artifact-ID collision: GP-AUD-061 preserves the stale-export/current-v1.0 adversarial correction; GP-AUD-059 identifies the later v1.1 author receipt.","Prevents evidence-lineage conflation","NONE","https://docs.google.com/document/d/1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ/edit","https://docs.google.com/document/d/125eYLOvqR9BWBfG7713Jg8dg8ugYBDIkldDiAhv_bOA/edit","TR-P12-009","ACTIVE-COLLISION-RESOLVED","2026-07-21"]
```

#### Where the two rows differ

| Field | Row 38 (keeper) | Row 42 (reidentified) |
|---|---|---|
| **Source object** | EC-014 | GP-AUD-061-v1.0 |
| **Source hyper-tag** | [[QUEUE:EC-014]] | [[OBJ:GP-AUD-061-v1.0]] |
| **Relation type** | REQUIRES_REVIEW_FROM | DISTINCT_FROM |
| **Relation hyper-tag** | [[REL:REQUIRES_REVIEW_FROM]] | [[REL:DISTINCT_FROM]] |
| **Target object** | Two distinct lines among CL, AO48, CW/C047R | GP-AUD-059-v1.0 amended author receipt |
| **Target hyper-tag** | [[ROSTER:ACTIVE_NONAUTHOR_LINES]] | [[OBJ:GP-AUD-059-v1.0]] |
| **Exact scope / meaning** | One line independently reconstructs all matrix and determinant details; a different line checks mutation, density, and excluded-scope boundary. | Resolves concurrent artifact-ID collision: GP-AUD-061 preserves the stale-export/current-v1.0 adversarial correction; GP-AUD-059 identifies the later v1.1 author receipt. |
| **Evidentiary effect** | Explicit responses required; silence has no effect | Prevents evidence-lineage conflation |
| **Authority effect** | Human approval remains later | NONE |
| **Source URL** | https://docs.google.com/document/d/1V-2e8oeYrf2Amgm2wzkSMAlpXDTQWgxEC5tHSTFKVLs/edit | https://docs.google.com/document/d/1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ/edit |
| **Target URL** | https://docs.google.com/document/d/1LFs1OlJwHvPTjby9Vjfua3X9L3iQNey_gRu0crd0og8/edit | https://docs.google.com/document/d/125eYLOvqR9BWBfG7713Jg8dg8ugYBDIkldDiAhv_bOA/edit |
| **Provenance** | OP-PROT-004; GP-REQ-057 acceptance rule | TR-P12-009 |
| **Status** | OPEN-RESPONSE-REQUIRED | ACTIVE-COLLISION-RESOLVED |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 38 and 42 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations. Row 38 relates 'EC-014' REQUIRES_REVIEW_FROM 'Two distinct lines among CL, AO48, CW/C047R' (Status 'OPEN-RESPONSE-REQUIRED', Provenance 'OP-PROT-004; GP-REQ-057 acceptance rule'); row 42 relates 'GP-AUD-061-v1.0' DISTINCT_FROM 'GP-AUD-059-v1.0 amended author receipt' (Status 'ACTIVE-COLLISION-RESOLVED', Provenance 'TR-P12-009'). Source URLs differ (1V-2e8oeYrf2Amgm2wzkSMAlpXDTQWgxEC5tHSTFKVLs versus 1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ). Only Relation ID and Last reviewed agree; thirteen of fifteen cells differ. One identifier assigned twice. Row 42 is itself the register's record that an artifact-ID collision (GP-AUD-059 / GP-AUD-061) was resolved; it was filed under a colliding relation id.

**Corroborating register evidence.** Fourth member of the two blocks described under REL-036. Row 42's Exact scope / meaning ('Resolves concurrent artifact-ID collision: GP-AUD-061 preserves the stale-export/current-v1.0 adversarial correction; GP-AUD-059 identifies the later v1.1 author receipt') and row 44's ('Resolves the second artifact-ID collision ... reidentifying the stale-export correction as GP-AUD-062') show the block was written while that artifact-ID collision was being repaired; the relation ids were not checked against rows 35–38 in the same pass.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 38** keeps `REL-039`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 42 → `REL-039@REL-R42`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-039@1xQmawV for row 42; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 42 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL039-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 38 (2026-09-18 xlsx export) — REL-039 — 'EC-014' REQUIRES_REVIEW_FROM 'Two distinct lines among CL, AO48, CW/C047R' — Status: OPEN-RESPONSE-REQUIRED |
| `File B` | Relation Index row 42 (2026-09-18 xlsx export) — REL-039 — 'GP-AUD-061-v1.0' DISTINCT_FROM 'GP-AUD-059-v1.0 amended author receipt' — Status: ACTIVE-COLLISION-RESOLVED |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://docs.google.com/document/d/1V-2e8oeYrf2Amgm2wzkSMAlpXDTQWgxEC5tHSTFKVLs/edit versus https://docs.google.com/document/d/1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 38 as REL-039; register row 42 additively under compound key REL-039@REL-R42; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 42, column 'Relation ID'; `REL-039` → `REL-039@REL-R42`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number is not decided here (see REL-036).
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.5 `REL-040` — rows 43, 44

> Finding key: `relations: duplicate key 'REL-040' at rows 43 and 44`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 43: Source object 'GP-DATA-054-v1.1 compressed capsule'; Relation type 'FAILS_PUBLICATION_GATE_FOR'; Target object 'GP-DATA-054-v1.1 claimed exact source'; Status 'ACTIVE-CRITICAL-BLOCKER'. Row 44: Source object 'GP-AUD-062-v1.0'; Relation type 'DISTINCT_FROM'; Target object 'GP-AUD-059-v1.0 amended 21-control receipt'; Status 'ACTIVE-COLLISION-RESOLVED'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 43 → https://docs.google.com/document/d/1NJaNSyoe63iizZS9bB2OK02Hz46p4bwzUW3_iVCEDbo/edit; row 44 → https://docs.google.com/document/d/1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 43 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[43], canonical 711 bytes, SHA-256 `4d1fb40629fa6e3ab96f1c9e763b98f719e133a6f1af567ad83950ff0d11bf66`:

```json
["REL-040","GP-DATA-054-v1.1 compressed capsule","[[OBJ:GP-DATA-054-v1.1-COMPRESSED-CAPSULE]]","FAILS_PUBLICATION_GATE_FOR","[[REL:FAILS_PUBLICATION_GATE_FOR]]","GP-DATA-054-v1.1 claimed exact source","[[GATE:PAIRING_V11_SOURCE_IDENTITY]]","Google Docs text is empty and text/plain download is three bytes; no compressed payload is reconstructable.","Direct second publication-surface failure evidence","Blocks independent reconstruction and qualification","https://docs.google.com/document/d/1NJaNSyoe63iizZS9bB2OK02Hz46p4bwzUW3_iVCEDbo/edit","https://docs.google.com/document/d/1tvskI0aup4X8Qt2v_v_C7M0C8L1W1ssYJiSbHRR4Uhg/edit","TR-P12-008; GP-AUD-063; fresh download","ACTIVE-CRITICAL-BLOCKER","2026-07-21"]
```

#### Row 44 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[44], canonical 615 bytes, SHA-256 `eafb5c4e9eccc46c257fca15d949eb0c8533d3686028032204ffcf6c5c6d9ebd`:

```json
["REL-040","GP-AUD-062-v1.0","[[OBJ:GP-AUD-062-v1.0]]","DISTINCT_FROM","[[REL:DISTINCT_FROM]]","GP-AUD-059-v1.0 amended 21-control receipt","[[OBJ:GP-AUD-059-v1.0]]","Resolves the second artifact-ID collision by retaining GP-AUD-059 for the v1.1 author receipt and reidentifying the stale-export correction as GP-AUD-062.","Provenance and retrieval correction","NONE","https://docs.google.com/document/d/1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ/edit","https://docs.google.com/document/d/125eYLOvqR9BWBfG7713Jg8dg8ugYBDIkldDiAhv_bOA/edit","DEF-001 Drive object identity","ACTIVE-COLLISION-RESOLVED","2026-07-21"]
```

#### Where the two rows differ

| Field | Row 43 (keeper) | Row 44 (reidentified) |
|---|---|---|
| **Source object** | GP-DATA-054-v1.1 compressed capsule | GP-AUD-062-v1.0 |
| **Source hyper-tag** | [[OBJ:GP-DATA-054-v1.1-COMPRESSED-CAPSULE]] | [[OBJ:GP-AUD-062-v1.0]] |
| **Relation type** | FAILS_PUBLICATION_GATE_FOR | DISTINCT_FROM |
| **Relation hyper-tag** | [[REL:FAILS_PUBLICATION_GATE_FOR]] | [[REL:DISTINCT_FROM]] |
| **Target object** | GP-DATA-054-v1.1 claimed exact source | GP-AUD-059-v1.0 amended 21-control receipt |
| **Target hyper-tag** | [[GATE:PAIRING_V11_SOURCE_IDENTITY]] | [[OBJ:GP-AUD-059-v1.0]] |
| **Exact scope / meaning** | Google Docs text is empty and text/plain download is three bytes; no compressed payload is reconstructable. | Resolves the second artifact-ID collision by retaining GP-AUD-059 for the v1.1 author receipt and reidentifying the stale-export correction as GP-AUD-062. |
| **Evidentiary effect** | Direct second publication-surface failure evidence | Provenance and retrieval correction |
| **Authority effect** | Blocks independent reconstruction and qualification | NONE |
| **Source URL** | https://docs.google.com/document/d/1NJaNSyoe63iizZS9bB2OK02Hz46p4bwzUW3_iVCEDbo/edit | https://docs.google.com/document/d/1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ/edit |
| **Target URL** | https://docs.google.com/document/d/1tvskI0aup4X8Qt2v_v_C7M0C8L1W1ssYJiSbHRR4Uhg/edit | https://docs.google.com/document/d/125eYLOvqR9BWBfG7713Jg8dg8ugYBDIkldDiAhv_bOA/edit |
| **Provenance** | TR-P12-008; GP-AUD-063; fresh download | DEF-001 Drive object identity |
| **Status** | ACTIVE-CRITICAL-BLOCKER | ACTIVE-COLLISION-RESOLVED |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 43 and 44 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations and are adjacent. Row 43 relates 'GP-DATA-054-v1.1 compressed capsule' FAILS_PUBLICATION_GATE_FOR 'GP-DATA-054-v1.1 claimed exact source' (Status 'ACTIVE-CRITICAL-BLOCKER', Provenance 'TR-P12-008; GP-AUD-063; fresh download'); row 44 relates 'GP-AUD-062-v1.0' DISTINCT_FROM 'GP-AUD-059-v1.0 amended 21-control receipt' (Status 'ACTIVE-COLLISION-RESOLVED', Provenance 'DEF-001 Drive object identity'). Source URLs differ (1NJaNSyoe63iizZS9bB2OK02Hz46p4bwzUW3_iVCEDbo versus 1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ). Only Relation ID and Last reviewed agree; thirteen of fifteen cells differ. One identifier assigned twice.

**Corroborating register evidence.** Row 43 closes the second block (REL-036…040 at rows 39–43); row 44 is a single further row that took REL-040 again before row 45 resumed at REL-041. Row 44's Source URL (1xQma…, GP-AUD-062-v1.0 per its Source object) equals row 42's Source URL although row 42's Source object reads 'GP-AUD-061-v1.0': the two rows cite one Drive document under two artifact ids, which is the GP-AUD-061/062 reidentification row 44 itself describes. Not adjudicated here.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 43** keeps `REL-040`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 44 → `REL-040@REL-R44`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-040@1xQmawV for row 44; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 44 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL040-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 43 (2026-09-18 xlsx export) — REL-040 — 'GP-DATA-054-v1.1 compressed capsule' FAILS_PUBLICATION_GATE_FOR 'GP-DATA-054-v1.1 claimed exact source' — Status: ACTIVE-CRITICAL-BLOCKER |
| `File B` | Relation Index row 44 (2026-09-18 xlsx export) — REL-040 — 'GP-AUD-062-v1.0' DISTINCT_FROM 'GP-AUD-059-v1.0 amended 21-control receipt' — Status: ACTIVE-COLLISION-RESOLVED |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://docs.google.com/document/d/1NJaNSyoe63iizZS9bB2OK02Hz46p4bwzUW3_iVCEDbo/edit versus https://docs.google.com/document/d/1xQmawVY4d-lFSKaTxI5IR_J2zYRFm8VR77OVUwzAUQQ/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 43 as REL-040; register row 44 additively under compound key REL-040@REL-R44; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 44, column 'Relation ID'; `REL-040` → `REL-040@REL-R44`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number is not decided here (see REL-036).
- Rows 42 and 44 cite one Source URL (1xQma…) under Source objects 'GP-AUD-061-v1.0' and 'GP-AUD-062-v1.0'; whether that document's declared id is 061 or 062 is an Artifact Index question outside this proposal.
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.6 `REL-048` — rows 52, 58

> Finding key: `relations: duplicate key 'REL-048' at rows 52 and 58`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 52: Source object 'CL-AUD-044-v1.0'; Relation type 'CORROBORATES'; Target object 'GP-DER-047-v1.0 deterministic layer'; Status 'ACTIVE'. Row 58: Source object 'CL-AUD-036-v1.0'; Relation type 'SATISFIES_CONDITION_OF'; Target object 'HA-006 conditional approval'; Status 'TERMINAL-CONDITION-SATISFIED'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 52 → https://docs.google.com/document/d/1O72ptuojwXcy2K8tg3xNNnnthmIgBscpboliDY7Lnd4/edit; row 58 → https://docs.google.com/document/d/12VDhZV85MvuBppaI5-SgBwyFULyTP-mknE526C7U1Vw/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 52 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[52], canonical 617 bytes, SHA-256 `7280490582d85940f830cf1872a0a834f162c3bb069b9e2c6ac2a7e9538b59e7`:

```json
["REL-048","CL-AUD-044-v1.0","[[OBJ:CL-AUD-044-v1.0]]","CORROBORATES","[[REL:CORROBORATES]]","GP-DER-047-v1.0 deterministic layer","[[OBJ:GP-DER-047-v1.0]]","Reexecutes the frozen degree-four certificate on 864 adversarial configurations with zero counterexamples; does not verify Gaussian/Palm or exact-law layers.","Raises component confidence only","No theorem promotion","https://docs.google.com/document/d/1O72ptuojwXcy2K8tg3xNNnnthmIgBscpboliDY7Lnd4/edit","https://docs.google.com/document/d/1jm1WxNmDd_rBLZA9FQsc2Nlzb9_PPCuEBV0fX3mCL1c/edit","CL execution response; GP additive relation","ACTIVE","2026-07-21"]
```

#### Row 58 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[58], canonical 605 bytes, SHA-256 `1f6e771e2441e355661bfd4a425e459019a8f87809b67fcf5b3e8023352706b6`:

```json
["REL-048","CL-AUD-036-v1.0","[[OBJ:CL-AUD-036-v1.0]]","SATISFIES_CONDITION_OF","[[REL:SATISFIES_CONDITION_OF]]","HA-006 conditional approval","[[OBJ:HA-006]]","Itemized APPROVE 7/7 after independent re-execution; batch wrapper deduplicated; EC-006 precision caveat incorporated.","Completes human closure gate for seven exact objects","No authority beyond named proposals","https://docs.google.com/document/d/12VDhZV85MvuBppaI5-SgBwyFULyTP-mknE526C7U1Vw/edit","https://docs.google.com/document/d/1Ttw7OS1vFAE2x7V3tkMFBEBELVrIK5ukRxKG2ObFT6Y/edit","GP-REQ-065","TERMINAL-CONDITION-SATISFIED","2026-07-21"]
```

#### Where the two rows differ

| Field | Row 52 (keeper) | Row 58 (reidentified) |
|---|---|---|
| **Source object** | CL-AUD-044-v1.0 | CL-AUD-036-v1.0 |
| **Source hyper-tag** | [[OBJ:CL-AUD-044-v1.0]] | [[OBJ:CL-AUD-036-v1.0]] |
| **Relation type** | CORROBORATES | SATISFIES_CONDITION_OF |
| **Relation hyper-tag** | [[REL:CORROBORATES]] | [[REL:SATISFIES_CONDITION_OF]] |
| **Target object** | GP-DER-047-v1.0 deterministic layer | HA-006 conditional approval |
| **Target hyper-tag** | [[OBJ:GP-DER-047-v1.0]] | [[OBJ:HA-006]] |
| **Exact scope / meaning** | Reexecutes the frozen degree-four certificate on 864 adversarial configurations with zero counterexamples; does not verify Gaussian/Palm or exact-law layers. | Itemized APPROVE 7/7 after independent re-execution; batch wrapper deduplicated; EC-006 precision caveat incorporated. |
| **Evidentiary effect** | Raises component confidence only | Completes human closure gate for seven exact objects |
| **Authority effect** | No theorem promotion | No authority beyond named proposals |
| **Source URL** | https://docs.google.com/document/d/1O72ptuojwXcy2K8tg3xNNnnthmIgBscpboliDY7Lnd4/edit | https://docs.google.com/document/d/12VDhZV85MvuBppaI5-SgBwyFULyTP-mknE526C7U1Vw/edit |
| **Target URL** | https://docs.google.com/document/d/1jm1WxNmDd_rBLZA9FQsc2Nlzb9_PPCuEBV0fX3mCL1c/edit | https://docs.google.com/document/d/1Ttw7OS1vFAE2x7V3tkMFBEBELVrIK5ukRxKG2ObFT6Y/edit |
| **Provenance** | CL execution response; GP additive relation | GP-REQ-065 |
| **Status** | ACTIVE | TERMINAL-CONDITION-SATISFIED |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 52 and 58 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations. Row 52 relates 'CL-AUD-044-v1.0' CORROBORATES 'GP-DER-047-v1.0 deterministic layer' (Status 'ACTIVE', Provenance 'CL execution response; GP additive relation'); row 58 relates 'CL-AUD-036-v1.0' SATISFIES_CONDITION_OF 'HA-006 conditional approval' (Status 'TERMINAL-CONDITION-SATISFIED', Provenance 'GP-REQ-065'). Source URLs differ (1O72ptuojwXcy2K8tg3xNNnnthmIgBscpboliDY7Lnd4 versus 12VDhZV85MvuBppaI5-SgBwyFULyTP-mknE526C7U1Vw). Only Relation ID and Last reviewed agree; thirteen of fifteen cells differ. One identifier assigned twice; the TERMINAL status of row 58 is its own object's and is transcribed, not evaluated.

**Corroborating register evidence.** Rows 52–53 carry REL-048 and REL-049 (the CL-AUD-044/045 review of GP-DER-047, Provenance 'CL execution response; GP additive relation' and 'CL crosswalk; GP additive relation'); rows 54–57 continue at REL-050…053; rows 58–59 then carry REL-048 and REL-049 again (the HA-006 / GP-CLS-BATCH-017 closure pair, Provenance 'GP-REQ-065' and 'Closure Log entries GP-CLS-017-A through G'); row 60 resumes at REL-054. The second pair was appended after REL-053 had been issued but numbered from 048, i.e. from a stale view of the tab's highest number.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 52** keeps `REL-048`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 58 → `REL-048@REL-R58`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-048@12VDhZV for row 58; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 58 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL048-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 52 (2026-09-18 xlsx export) — REL-048 — 'CL-AUD-044-v1.0' CORROBORATES 'GP-DER-047-v1.0 deterministic layer' — Status: ACTIVE |
| `File B` | Relation Index row 58 (2026-09-18 xlsx export) — REL-048 — 'CL-AUD-036-v1.0' SATISFIES_CONDITION_OF 'HA-006 conditional approval' — Status: TERMINAL-CONDITION-SATISFIED |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://docs.google.com/document/d/1O72ptuojwXcy2K8tg3xNNnnthmIgBscpboliDY7Lnd4/edit versus https://docs.google.com/document/d/12VDhZV85MvuBppaI5-SgBwyFULyTP-mknE526C7U1Vw/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 52 as REL-048; register row 58 additively under compound key REL-048@REL-R58; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 58, column 'Relation ID'; `REL-048` → `REL-048@REL-R58`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number is not decided here (see REL-036).
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.7 `REL-049` — rows 53, 59

> Finding key: `relations: duplicate key 'REL-049' at rows 53 and 59`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 53: Source object 'CL-AUD-045-v1.0'; Relation type 'AUDITS'; Target object 'GP-DER-047-v1.0 dependency graph'; Status 'ACTIVE'. Row 59: Source object 'GP-CLS-BATCH-017-v1.0'; Relation type 'TERMINALLY_CLOSES'; Target object 'EC-005, EC-006, EC-007, EC-008, EC-010, EC-011, EC-012'; Status 'TERMINAL'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 53 → https://docs.google.com/document/d/1mhNO8GXWqxeMnqFyE5xpyZvr_OdvGOmJ_EsDeUjnUxY/edit; row 59 → https://docs.google.com/document/d/1Zm65j9BZe__z3P_8qP7rfZukRU-IWUE8h8bDqXdDsIQ/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 53 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[53], canonical 570 bytes, SHA-256 `5fa7ab1569d3da6224eeae12e3562569e5e400365d2a0805717c85d3e814c273`:

```json
["REL-049","CL-AUD-045-v1.0","[[OBJ:CL-AUD-045-v1.0]]","AUDITS","[[REL:AUDITS]]","GP-DER-047-v1.0 dependency graph","[[OBJ:GP-DER-047-v1.0]]","Crosswalks coordinates, pins, determinant weight, normalizer, threshold class, and remainder; isolates E1 and E2.","Identifies aligned and unverified dependency edges","No closure or promotion","https://docs.google.com/document/d/1mhNO8GXWqxeMnqFyE5xpyZvr_OdvGOmJ_EsDeUjnUxY/edit","https://docs.google.com/document/d/1jm1WxNmDd_rBLZA9FQsc2Nlzb9_PPCuEBV0fX3mCL1c/edit","CL crosswalk; GP additive relation","ACTIVE","2026-07-21"]
```

#### Row 59 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[59], canonical 694 bytes, SHA-256 `c6de7ac0d0b90f7d046500a2b2a9ad739560d3084b9f5f767932d79b56f30215`:

```json
["REL-049","GP-CLS-BATCH-017-v1.0","[[OBJ:GP-CLS-BATCH-017-v1.0]]","TERMINALLY_CLOSES","[[REL:TERMINALLY_CLOSES]]","EC-005, EC-006, EC-007, EC-008, EC-010, EC-011, EC-012","[[QUEUE:EC-005..EC-012-SELECTED]]","Enters seven exact terminal records and removes the items from active attention while preserving all provenance and reopening paths.","Terminal exact-object status","No theorem, machine, external-release or deletion authority","https://docs.google.com/document/d/1Zm65j9BZe__z3P_8qP7rfZukRU-IWUE8h8bDqXdDsIQ/edit","https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200012","Closure Log entries GP-CLS-017-A through G","TERMINAL","2026-07-21"]
```

#### Where the two rows differ

| Field | Row 53 (keeper) | Row 59 (reidentified) |
|---|---|---|
| **Source object** | CL-AUD-045-v1.0 | GP-CLS-BATCH-017-v1.0 |
| **Source hyper-tag** | [[OBJ:CL-AUD-045-v1.0]] | [[OBJ:GP-CLS-BATCH-017-v1.0]] |
| **Relation type** | AUDITS | TERMINALLY_CLOSES |
| **Relation hyper-tag** | [[REL:AUDITS]] | [[REL:TERMINALLY_CLOSES]] |
| **Target object** | GP-DER-047-v1.0 dependency graph | EC-005, EC-006, EC-007, EC-008, EC-010, EC-011, EC-012 |
| **Target hyper-tag** | [[OBJ:GP-DER-047-v1.0]] | [[QUEUE:EC-005..EC-012-SELECTED]] |
| **Exact scope / meaning** | Crosswalks coordinates, pins, determinant weight, normalizer, threshold class, and remainder; isolates E1 and E2. | Enters seven exact terminal records and removes the items from active attention while preserving all provenance and reopening paths. |
| **Evidentiary effect** | Identifies aligned and unverified dependency edges | Terminal exact-object status |
| **Authority effect** | No closure or promotion | No theorem, machine, external-release or deletion authority |
| **Source URL** | https://docs.google.com/document/d/1mhNO8GXWqxeMnqFyE5xpyZvr_OdvGOmJ_EsDeUjnUxY/edit | https://docs.google.com/document/d/1Zm65j9BZe__z3P_8qP7rfZukRU-IWUE8h8bDqXdDsIQ/edit |
| **Target URL** | https://docs.google.com/document/d/1jm1WxNmDd_rBLZA9FQsc2Nlzb9_PPCuEBV0fX3mCL1c/edit | https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200012 |
| **Provenance** | CL crosswalk; GP additive relation | Closure Log entries GP-CLS-017-A through G |
| **Status** | ACTIVE | TERMINAL |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 53 and 59 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations. Row 53 relates 'CL-AUD-045-v1.0' AUDITS 'GP-DER-047-v1.0 dependency graph' (Status 'ACTIVE', Provenance 'CL crosswalk; GP additive relation'); row 59 relates 'GP-CLS-BATCH-017-v1.0' TERMINALLY_CLOSES 'EC-005, EC-006, EC-007, EC-008, EC-010, EC-011, EC-012' (Status 'TERMINAL', Provenance 'Closure Log entries GP-CLS-017-A through G'). Source URLs differ (1mhNO8GXWqxeMnqFyE5xpyZvr_OdvGOmJ_EsDeUjnUxY versus 1Zm65j9BZe__z3P_8qP7rfZukRU-IWUE8h8bDqXdDsIQ); row 59's Target URL is the register workbook itself (spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no, fragment #gid=200012). Only Relation ID and Last reviewed agree; thirteen of fifteen cells differ. One identifier assigned twice.

**Corroborating register evidence.** Second member of the stale-numbering pair described under REL-048 (rows 52–53 versus rows 58–59). Row 59's terminal closure of seven EC items is the relation the register cites for GP-CLS-BATCH-017; its TERMINAL status is not affected by its key and is not evaluated here.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 53** keeps `REL-049`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 59 → `REL-049@REL-R59`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-049@1Zm65j9 for row 59; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 59 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL049-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 53 (2026-09-18 xlsx export) — REL-049 — 'CL-AUD-045-v1.0' AUDITS 'GP-DER-047-v1.0 dependency graph' — Status: ACTIVE |
| `File B` | Relation Index row 59 (2026-09-18 xlsx export) — REL-049 — 'GP-CLS-BATCH-017-v1.0' TERMINALLY_CLOSES 'EC-005, EC-006, EC-007, EC-008, EC-010, EC-011, EC-012' — Status: TERMINAL |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://docs.google.com/document/d/1mhNO8GXWqxeMnqFyE5xpyZvr_OdvGOmJ_EsDeUjnUxY/edit versus https://docs.google.com/document/d/1Zm65j9BZe__z3P_8qP7rfZukRU-IWUE8h8bDqXdDsIQ/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 53 as REL-049; register row 59 additively under compound key REL-049@REL-R59; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 59, column 'Relation ID'; `REL-049` → `REL-049@REL-R59`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number is not decided here (see REL-036).
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.8 `REL-132` — rows 138, 141

> Finding key: `relations: duplicate key 'REL-132' at rows 138 and 141`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 138: Source object 'GP-CLS-PROP-019-v1.0'; Relation type 'PACKAGES_TERMINAL_RETIREMENT_FOR'; Target object 'EC-016'; Status 'ACTIVE-CONDITIONAL-PACKAGE'. Row 141: Source object 'GP-DER-106-v1.0'; Relation type 'IMPLEMENTS_ROUTE_B_OF'; Target object 'GP-REQ-105-v1.0'; Status 'ACTIVE-COMPONENT'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 138 → https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit; row 141 → https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 138 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[138], canonical 726 bytes, SHA-256 `83ec98dba85f556956fdd8f235b6272422b833f494c36e90c2e39f2fab856034`:

```json
["REL-132","GP-CLS-PROP-019-v1.0","[[OBJ:GP-CLS-PROP-019-v1.0]]","PACKAGES_TERMINAL_RETIREMENT_FOR","[[REL:PACKAGES_TERMINAL_RETIREMENT_FOR]]","EC-016","[[QUEUE:EC-016]]","Binds the five rejected source/inference claims, exact evidence, retained diagnostics, dependencies, terminal label, gates, register actions, and reopening rule.","Makes the negative-result closure reviewable and prevents repeated scope assembly.","No terminal authority until outside and human gates pass.","https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit","https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200011","TR-EC016-001","ACTIVE-CONDITIONAL-PACKAGE","2026-07-22"]
```

#### Row 141 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[141], canonical 600 bytes, SHA-256 `6fd29add1f86ca4ac1b1470c46ff6bbcd5ec9b9519a058b54b2f598255fc24f8`:

```json
["REL-132","GP-DER-106-v1.0","[[OBJ:GP-DER-106-v1.0]]","IMPLEMENTS_ROUTE_B_OF","[[REL:IMPLEMENTS_ROUTE_B_OF]]","GP-REQ-105-v1.0","[[OBJ:GP-REQ-105-v1.0]]","Constructs a full nine-dimensional interval-certified degree-four box around PRCP-INTERIOR-001.","Discharges the GP-side degree-four continuum-box construction step","No mass, exact-field, r-uniformity, or theorem authority","https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit","https://docs.google.com/document/d/1kNp3vZxyA3kcorgwkdkL50MLwZQ7IiSayMh--aXZIiY/edit","TR-P01-002","ACTIVE-COMPONENT","2026-07-22"]
```

#### Where the two rows differ

| Field | Row 138 (keeper) | Row 141 (reidentified) |
|---|---|---|
| **Source object** | GP-CLS-PROP-019-v1.0 | GP-DER-106-v1.0 |
| **Source hyper-tag** | [[OBJ:GP-CLS-PROP-019-v1.0]] | [[OBJ:GP-DER-106-v1.0]] |
| **Relation type** | PACKAGES_TERMINAL_RETIREMENT_FOR | IMPLEMENTS_ROUTE_B_OF |
| **Relation hyper-tag** | [[REL:PACKAGES_TERMINAL_RETIREMENT_FOR]] | [[REL:IMPLEMENTS_ROUTE_B_OF]] |
| **Target object** | EC-016 | GP-REQ-105-v1.0 |
| **Target hyper-tag** | [[QUEUE:EC-016]] | [[OBJ:GP-REQ-105-v1.0]] |
| **Exact scope / meaning** | Binds the five rejected source/inference claims, exact evidence, retained diagnostics, dependencies, terminal label, gates, register actions, and reopening rule. | Constructs a full nine-dimensional interval-certified degree-four box around PRCP-INTERIOR-001. |
| **Evidentiary effect** | Makes the negative-result closure reviewable and prevents repeated scope assembly. | Discharges the GP-side degree-four continuum-box construction step |
| **Authority effect** | No terminal authority until outside and human gates pass. | No mass, exact-field, r-uniformity, or theorem authority |
| **Source URL** | https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit | https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit |
| **Target URL** | https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200011 | https://docs.google.com/document/d/1kNp3vZxyA3kcorgwkdkL50MLwZQ7IiSayMh--aXZIiY/edit |
| **Provenance** | TR-EC016-001 | TR-P01-002 |
| **Status** | ACTIVE-CONDITIONAL-PACKAGE | ACTIVE-COMPONENT |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 138 and 141 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations. Row 138 relates 'GP-CLS-PROP-019-v1.0' PACKAGES_TERMINAL_RETIREMENT_FOR 'EC-016' (Status 'ACTIVE-CONDITIONAL-PACKAGE', Provenance 'TR-EC016-001'); row 141 relates 'GP-DER-106-v1.0' IMPLEMENTS_ROUTE_B_OF 'GP-REQ-105-v1.0' (Status 'ACTIVE-COMPONENT', Provenance 'TR-P01-002'). Source URLs differ (1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag versus 1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI). Only Relation ID and Last reviewed ('2026-07-22') agree; thirteen of fifteen cells differ. One identifier assigned twice.

**Corroborating register evidence.** Rows 138–140 carry REL-132, -133, -134 as the EC-016 closure-package block (Source object 'GP-CLS-PROP-019-v1.0' in all three, Provenance 'TR-EC016-001' / 'HB-027; TR-EC016-001'); rows 141–143 carry REL-132, -133, -134 again as the Route-B degree-four box block (Provenance 'TR-P01-002', 'TR-P01-002', 'GP-AUD-107; GP-REQ-105'); rows 144–145 continue that second block at REL-135 and REL-136 with Provenance 'TR-P01-003'. Row 144 (REL-135) repeats row 141's triple 'GP-DER-106-v1.0' IMPLEMENTS_ROUTE_B_OF 'GP-REQ-105-v1.0' under its own id with Status 'ACTIVE-GP-SIDE-PASS' — a second registration of the same relation under a different key, which is not a key collision and is only noted.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 138** keeps `REL-132`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 141 → `REL-132@REL-R141`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-132@1ve_dMz for row 141; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 141 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL132-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 138 (2026-09-18 xlsx export) — REL-132 — 'GP-CLS-PROP-019-v1.0' PACKAGES_TERMINAL_RETIREMENT_FOR 'EC-016' — Status: ACTIVE-CONDITIONAL-PACKAGE |
| `File B` | Relation Index row 141 (2026-09-18 xlsx export) — REL-132 — 'GP-DER-106-v1.0' IMPLEMENTS_ROUTE_B_OF 'GP-REQ-105-v1.0' — Status: ACTIVE-COMPONENT |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit versus https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 138 as REL-132; register row 141 additively under compound key REL-132@REL-R141; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 141, column 'Relation ID'; `REL-132` → `REL-132@REL-R141`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number is not decided here (see REL-036).
- Row 144 (REL-135) asserts the same relation triple as row 141 under a distinct id and a different Status; whether that is a supersession or a duplicate registration is a content question for the operator, outside the fourteen findings.
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.9 `REL-133` — rows 139, 142

> Finding key: `relations: duplicate key 'REL-133' at rows 139 and 142`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 139: Source object 'GP-CLS-PROP-019-v1.0'; Relation type 'REQUESTS_REVIEW_FROM'; Target object 'AO48 or CW/C047R'; Status 'OPEN-RESPONSE-REQUIRED'. Row 142: Source object 'GP-AUD-107-v1.0'; Relation type 'VERIFIES_EXECUTION_OF'; Target object 'GP-DATA-106-v1.0'; Status 'ACTIVE-GP-SIDE-PASS'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 139 → https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit; row 142 → https://docs.google.com/document/d/1cCuANEtlp1bZDvckMCNNwGBfXZivCMUWrwxcwtccrRc/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 139 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[139], canonical 720 bytes, SHA-256 `d9476f7d20a083008e7ee391c849e1fbe95e9c8506138b6d461b17f5fe1f7b3d`:

```json
["REL-133","GP-CLS-PROP-019-v1.0","[[OBJ:GP-CLS-PROP-019-v1.0]]","REQUESTS_REVIEW_FROM","[[REL:REQUESTS_REVIEW_FROM]]","AO48 or CW/C047R","[[ROSTER:EC016-OUTSIDE-SCOPE]]","Requests a seven-question exact verdict on imported a_r, grid dimension, continuum inference, classifier completeness, box versus capture mass, retained evidence, and exclusions.","Creates the final technical review surface.","Silence, GP self-review, and CL source-line agreement do not satisfy.","https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit","https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200002","HB-027; TR-EC016-001","OPEN-RESPONSE-REQUIRED","2026-07-22"]
```

#### Row 142 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[142], canonical 588 bytes, SHA-256 `54f393f738de44f53deb3d9acf0511717604078c777e23c4a69b7a06ce14d45a`:

```json
["REL-133","GP-AUD-107-v1.0","[[OBJ:GP-AUD-107-v1.0]]","VERIFIES_EXECUTION_OF","[[REL:VERIFIES_EXECUTION_OF]]","GP-DATA-106-v1.0","[[OBJ:GP-DATA-106-v1.0]]","Confirms exact Drive source/receipt reconstruction, original-filename rerun, all positive margins, and five controls.","Same-line source and interval component receipt","Outside interval review remains open","https://docs.google.com/document/d/1cCuANEtlp1bZDvckMCNNwGBfXZivCMUWrwxcwtccrRc/edit","https://docs.google.com/document/d/1sAbtS40ioX91gxiYRpbTZVg2hmqDqQu-z5HklYMibe4/edit","TR-P01-002","ACTIVE-GP-SIDE-PASS","2026-07-22"]
```

#### Where the two rows differ

| Field | Row 139 (keeper) | Row 142 (reidentified) |
|---|---|---|
| **Source object** | GP-CLS-PROP-019-v1.0 | GP-AUD-107-v1.0 |
| **Source hyper-tag** | [[OBJ:GP-CLS-PROP-019-v1.0]] | [[OBJ:GP-AUD-107-v1.0]] |
| **Relation type** | REQUESTS_REVIEW_FROM | VERIFIES_EXECUTION_OF |
| **Relation hyper-tag** | [[REL:REQUESTS_REVIEW_FROM]] | [[REL:VERIFIES_EXECUTION_OF]] |
| **Target object** | AO48 or CW/C047R | GP-DATA-106-v1.0 |
| **Target hyper-tag** | [[ROSTER:EC016-OUTSIDE-SCOPE]] | [[OBJ:GP-DATA-106-v1.0]] |
| **Exact scope / meaning** | Requests a seven-question exact verdict on imported a_r, grid dimension, continuum inference, classifier completeness, box versus capture mass, retained evidence, and exclusions. | Confirms exact Drive source/receipt reconstruction, original-filename rerun, all positive margins, and five controls. |
| **Evidentiary effect** | Creates the final technical review surface. | Same-line source and interval component receipt |
| **Authority effect** | Silence, GP self-review, and CL source-line agreement do not satisfy. | Outside interval review remains open |
| **Source URL** | https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit | https://docs.google.com/document/d/1cCuANEtlp1bZDvckMCNNwGBfXZivCMUWrwxcwtccrRc/edit |
| **Target URL** | https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200002 | https://docs.google.com/document/d/1sAbtS40ioX91gxiYRpbTZVg2hmqDqQu-z5HklYMibe4/edit |
| **Provenance** | HB-027; TR-EC016-001 | TR-P01-002 |
| **Status** | OPEN-RESPONSE-REQUIRED | ACTIVE-GP-SIDE-PASS |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 139 and 142 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations. Row 139 relates 'GP-CLS-PROP-019-v1.0' REQUESTS_REVIEW_FROM 'AO48 or CW/C047R' (Status 'OPEN-RESPONSE-REQUIRED', Provenance 'HB-027; TR-EC016-001'); row 142 relates 'GP-AUD-107-v1.0' VERIFIES_EXECUTION_OF 'GP-DATA-106-v1.0' (Status 'ACTIVE-GP-SIDE-PASS', Provenance 'TR-P01-002'). Source URLs differ (1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag versus 1cCuANEtlp1bZDvckMCNNwGBfXZivCMUWrwxcwtccrRc). Only Relation ID and Last reviewed agree; thirteen of fifteen cells differ. One identifier assigned twice; row 139's open review request and row 142's same-line pass are each their own object's status and are transcribed, not evaluated.

**Corroborating register evidence.** Second member of the two three-row blocks described under REL-132 (rows 138–140 versus rows 141–143). Row 145 (REL-136) repeats row 142's triple 'GP-AUD-107-v1.0' VERIFIES_EXECUTION_OF with Target object 'GP-DATA-106-v1.0 source and receipt' and Provenance 'TR-P01-003'.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 139** keeps `REL-133`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 142 → `REL-133@REL-R142`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-133@1cCuANE for row 142; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 142 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL133-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 139 (2026-09-18 xlsx export) — REL-133 — 'GP-CLS-PROP-019-v1.0' REQUESTS_REVIEW_FROM 'AO48 or CW/C047R' — Status: OPEN-RESPONSE-REQUIRED |
| `File B` | Relation Index row 142 (2026-09-18 xlsx export) — REL-133 — 'GP-AUD-107-v1.0' VERIFIES_EXECUTION_OF 'GP-DATA-106-v1.0' — Status: ACTIVE-GP-SIDE-PASS |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit versus https://docs.google.com/document/d/1cCuANEtlp1bZDvckMCNNwGBfXZivCMUWrwxcwtccrRc/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 139 as REL-133; register row 142 additively under compound key REL-133@REL-R142; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 142, column 'Relation ID'; `REL-133` → `REL-133@REL-R142`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number is not decided here (see REL-036).
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.10 `REL-134` — rows 140, 143

> Finding key: `relations: duplicate key 'REL-134' at rows 140 and 143`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Row 140: Source object 'GP-CLS-PROP-019-v1.0'; Relation type 'REQUIRES_HUMAN_APPROVAL_FROM'; Target object 'Dylan M. Roy after EC-016 outside scope review'; Status 'OPEN-HUMAN-GATE-AFTER-TECHNICAL'. Row 143: Source object 'P01-ROUTEB-D4-BOX-001'; Relation type 'REQUIRES_REMAINING'; Target object 'Outside interval review, rigorous exact-law Palm mass, exact-field transfer, and r-interval continuation'; Status 'OPEN-MATERIAL-GATES'. Two different relations under one id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 140 → https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit; row 143 → https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit. Relation triple cells agree: false; Target URL cells agree: false.

#### Row 140 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[140], canonical 644 bytes, SHA-256 `0b2671ca63e4b6a3e5c06b123421d03cb40e8c3798cae276a2b468670fecf36f`:

```json
["REL-134","GP-CLS-PROP-019-v1.0","[[OBJ:GP-CLS-PROP-019-v1.0]]","REQUIRES_HUMAN_APPROVAL_FROM","[[REL:REQUIRES_HUMAN_APPROVAL_FROM]]","Dylan M. Roy after EC-016 outside scope review","[[GATE:EC016-HUMAN]]","Exact approval wording is required after an affirmative AO48/CW verdict; general work authorization does not count.","Keeps terminal authority explicit.","No current approval counted.","https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit","https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200011","TR-EC016-001","OPEN-HUMAN-GATE-AFTER-TECHNICAL","2026-07-22"]
```

#### Row 143 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[143], canonical 733 bytes, SHA-256 `43744cc35b6191d8d6e6e98d3561596655d6ccaec4e1176562886aff8b3121c0`:

```json
["REL-134","P01-ROUTEB-D4-BOX-001","[[OBJ:P01-ROUTEB-D4-BOX-001]]","REQUIRES_REMAINING","[[REL:REQUIRES_REMAINING]]","Outside interval review, rigorous exact-law Palm mass, exact-field transfer, and r-interval continuation","[[TASK:HB-026]] [[TASK:HB-024]] [[TASK:HB-025]]","Degree-four continuum inclusion is complete on GP line; probability and exact-field claims require the named remaining gates.","Prevents component pass from becoming P0.1 evidence beyond scope","No promotion","https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit","https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200002","GP-AUD-107; GP-REQ-105","OPEN-MATERIAL-GATES","2026-07-22"]
```

#### Where the two rows differ

| Field | Row 140 (keeper) | Row 143 (reidentified) |
|---|---|---|
| **Source object** | GP-CLS-PROP-019-v1.0 | P01-ROUTEB-D4-BOX-001 |
| **Source hyper-tag** | [[OBJ:GP-CLS-PROP-019-v1.0]] | [[OBJ:P01-ROUTEB-D4-BOX-001]] |
| **Relation type** | REQUIRES_HUMAN_APPROVAL_FROM | REQUIRES_REMAINING |
| **Relation hyper-tag** | [[REL:REQUIRES_HUMAN_APPROVAL_FROM]] | [[REL:REQUIRES_REMAINING]] |
| **Target object** | Dylan M. Roy after EC-016 outside scope review | Outside interval review, rigorous exact-law Palm mass, exact-field transfer, and r-interval continuation |
| **Target hyper-tag** | [[GATE:EC016-HUMAN]] | [[TASK:HB-026]] [[TASK:HB-024]] [[TASK:HB-025]] |
| **Exact scope / meaning** | Exact approval wording is required after an affirmative AO48/CW verdict; general work authorization does not count. | Degree-four continuum inclusion is complete on GP line; probability and exact-field claims require the named remaining gates. |
| **Evidentiary effect** | Keeps terminal authority explicit. | Prevents component pass from becoming P0.1 evidence beyond scope |
| **Authority effect** | No current approval counted. | No promotion |
| **Source URL** | https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit | https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit |
| **Target URL** | https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200011 | https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200002 |
| **Provenance** | TR-EC016-001 | GP-AUD-107; GP-REQ-105 |
| **Status** | OPEN-HUMAN-GATE-AFTER-TECHNICAL | OPEN-MATERIAL-GATES |

Identical in both rows: `Relation ID`, `Last reviewed`.

Cell count: 15 columns compared, 2 identical, 13 differing (rows 140 and 143 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different relations. Row 140 relates 'GP-CLS-PROP-019-v1.0' REQUIRES_HUMAN_APPROVAL_FROM 'Dylan M. Roy after EC-016 outside scope review' (Status 'OPEN-HUMAN-GATE-AFTER-TECHNICAL', Provenance 'TR-EC016-001'); row 143 relates 'P01-ROUTEB-D4-BOX-001' REQUIRES_REMAINING 'Outside interval review, rigorous exact-law Palm mass, exact-field transfer, and r-interval continuation' (Status 'OPEN-MATERIAL-GATES', Provenance 'GP-AUD-107; GP-REQ-105'). Source URLs differ (1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag versus 1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI). Only Relation ID and Last reviewed agree; thirteen of fifteen cells differ. One identifier assigned twice; both rows record open gates, and neither gate is moved by this document.

**Corroborating register evidence.** Third member of the two blocks described under REL-132. Row 143's Target hyper-tag '[[TASK:HB-026]] [[TASK:HB-024]] [[TASK:HB-025]]' names the Help Board items that hold its open gates; row 140's '[[GATE:EC016-HUMAN]]' names the human gate. Neither is affected by the key.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Relation Index's own executed form for a relation-ID collision is the suffix key: rows 183, 184, 191, 192 and 193 carry 'REL-177-COLLISION-PROVENANCE' through 'REL-181-COLLISION-PROVENANCE' with Status 'NONOPERATIVE-COLLISION-PROVENANCE' and Provenance 'Concurrent relation-ID collision; corrected by authoritative REL-177 and TR-P01-011/012'; rows 245–247 carry 'REL-20260723-143-01' etc. with Status 'VOID-COLLISION-COPY'; row 363 'REL-CLWO07-1JGG-DISTINCT-1JQT' records a stable-key collision as a relation ('STABLE_KEY_COLLISION_DISTINCT_FROM'). Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076; no test evidence or scientific status changed'), DUP-ID-HELP-BOARD-20260724 ('RESOLVED: preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190') and DUP-DISPATCH-DQ030-20260726 ('preserve row 30 as DQ-030; reidentify row 31 as DQ-032'). NOTE: the tab's own '-COLLISION-PROVENANCE' and 'VOID-COLLISION-COPY' forms carry a status verdict (nonoperative, void) that was true of those rows and is NOT asserted of either row here; the row-locator key asserts nothing about status.

**Which row keeps the original id, and why**

- Keeper: **row 140** keeps `REL-134`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 143 → `REL-134@REL-R143`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Relation Index (both rows carry a Source URL) and would read REL-134@1ve_dMz for row 143; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither relation's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 143 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-REL134-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT RELATIONS |
| `File A` | Relation Index row 140 (2026-09-18 xlsx export) — REL-134 — 'GP-CLS-PROP-019-v1.0' REQUIRES_HUMAN_APPROVAL_FROM 'Dylan M. Roy after EC-016 outside scope review' — Status: OPEN-HUMAN-GATE-AFTER-TECHNICAL |
| `File B` | Relation Index row 143 (2026-09-18 xlsx export) — REL-134 — 'P01-ROUTEB-D4-BOX-001' REQUIRES_REMAINING 'Outside interval review, rigorous exact-law Palm mass, exact-field transfer, and r-interval continuation' — Status: OPEN-MATERIAL-GATES |
| `Similarity / hash` | Same declared Relation ID; different Source URL (https://docs.google.com/document/d/1-5yFAbJbsC-tkHltrmm9vHq5iQuuFr_mXLafhRTBiag/edit versus https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit); Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Exact scope / meaning, Evidentiary effect, Authority effect, Source URL, Target URL, Provenance, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 140 as REL-134; register row 143 additively under compound key REL-134@REL-R143; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 143, column 'Relation ID'; `REL-134` → `REL-134@REL-R143`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused REL number is not decided here (see REL-036).
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.11 `REL-EC021-CLS141` — rows 232, 235

> Finding key: `relations: duplicate key 'REL-EC021-CLS141' at rows 232 and 235`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the relations tab was first keyed by tools/registers_check.py (until then no checker read its Relation ID column). Rows 232 and 235 are NOT an exact duplicate: over the tab's fifteen columns nine cells agree (Relation ID, Source object, Source hyper-tag, Relation type, Relation hyper-tag, Target object, Target hyper-tag, Source URL and Status) and six differ ('Exact scope / meaning', 'Evidentiary effect', 'Authority effect', the Target URL's sheet fragment #gid=200011 vs #gid=200012, 'Provenance', and 'Last reviewed' 2026-07-23 vs 2026-07-22), so one relation from one Drive document is registered twice with different scope, effect and provenance wording. Until 2026-09-20 this rationale said the rows were "identical apart from Last reviewed ... an exact duplicate row", which was written from a printout of four of the fifteen columns and is wrong. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__SAME_DRIVE_OBJECT__SAME_RELATION__DIFFERENT_SCOPE_EFFECT_PROVENANCE_AND_DATE_CELLS`. Rule cells read: `Source URL` — agree. Both rows cite Drive source https://docs.google.com/document/d/1Z8XWIS5gVvkiZL67sgx8Ca71LR2HFxU_96HULcuOCMI/edit. Relation triple cells agree: true; Target URL cells agree: false.

#### Row 232 — proposed KEEPER row, verbatim

`registers/json/relations.json` rows[232], canonical 646 bytes, SHA-256 `50c5b49a764153730dcc732f5ca0a06edea583009022025263bc3e795865b68d`:

```json
["REL-EC021-CLS141","GP-CLS-141-v1.0","[[OBJ:GP-CLS-141-v1.0]]","TERMINALLY_CLOSES","[[REL:TERMINALLY_CLOSES]]","EC-021","[[QUEUE:EC-021]]","Closes only uniform positive typed finite-Q4 Palm mass of the fixed-q corridor for sufficiently small r, without a numerical constant.","Retires the exact finite-Q4 mass question and HB-032.","Authority derives from HA-008 item-level approval.","https://docs.google.com/document/d/1Z8XWIS5gVvkiZL67sgx8Ca71LR2HFxU_96HULcuOCMI/edit","https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200011","GP-CLS-PROP-025; CL-AUD-074; EC-020; HA-008.","TERMINAL","2026-07-23"]
```

#### Row 235 — proposed REIDENTIFIED row, verbatim

`registers/json/relations.json` rows[235], canonical 615 bytes, SHA-256 `3287798bef20ac68d5a506a45a6cb055aa8777e230276a2be5a0b65b06c27c5f`:

```json
["REL-EC021-CLS141","GP-CLS-141-v1.0","[[OBJ:GP-CLS-141-v1.0]]","TERMINALLY_CLOSES","[[REL:TERMINALLY_CLOSES]]","EC-021","[[QUEUE:EC-021]]","Closes the exact finite-Q4 qualitative uniform positive-mass object and retires it from active attention.","Creates terminal reusable finite-dimensional input with strict scope firewall.","No exact-field or theorem authority.","https://docs.google.com/document/d/1Z8XWIS5gVvkiZL67sgx8Ca71LR2HFxU_96HULcuOCMI/edit","https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200012","GP terminal record after HA-008","TERMINAL","2026-07-22"]
```

#### Where the two rows differ

| Field | Row 232 (keeper) | Row 235 (reidentified) |
|---|---|---|
| **Exact scope / meaning** | Closes only uniform positive typed finite-Q4 Palm mass of the fixed-q corridor for sufficiently small r, without a numerical constant. | Closes the exact finite-Q4 qualitative uniform positive-mass object and retires it from active attention. |
| **Evidentiary effect** | Retires the exact finite-Q4 mass question and HB-032. | Creates terminal reusable finite-dimensional input with strict scope firewall. |
| **Authority effect** | Authority derives from HA-008 item-level approval. | No exact-field or theorem authority. |
| **Target URL** | https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200011 | https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no/edit#gid=200012 |
| **Provenance** | GP-CLS-PROP-025; CL-AUD-074; EC-020; HA-008. | GP terminal record after HA-008 |
| **Last reviewed** | 2026-07-23 | 2026-07-22 |

Identical in both rows: `Relation ID`, `Source object`, `Source hyper-tag`, `Relation type`, `Relation hyper-tag`, `Target object`, `Target hyper-tag`, `Source URL`, `Status`.

Cell count: 15 columns compared, 9 identical, 6 differing (rows 232 and 235 of `registers/json/relations.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows were compared cell for cell: they are NOT an exact duplicate, although registers/KNOWN_FINDINGS.json described them as 'identical apart from Last reviewed' until the commit that adds this document, where that rationale is corrected to the recomputed counts. Nine of fifteen cells agree — Relation ID, Source object 'GP-CLS-141-v1.0', Source hyper-tag, Relation type 'TERMINALLY_CLOSES', Relation hyper-tag, Target object 'EC-021', Target hyper-tag, Source URL (document 1Z8XWIS5gVvkiZL67sgx8Ca71LR2HFxU_96HULcuOCMI) and Status 'TERMINAL' — so both rows assert one relation from one Drive document. Six cells differ: Exact scope / meaning ('Closes only uniform positive typed finite-Q4 Palm mass of the fixed-q corridor for sufficiently small r, without a numerical constant.' versus 'Closes the exact finite-Q4 qualitative uniform positive-mass object and retires it from active attention.'), Evidentiary effect ('Retires the exact finite-Q4 mass question and HB-032.' versus 'Creates terminal reusable finite-dimensional input with strict scope firewall.'), Authority effect ('Authority derives from HA-008 item-level approval.' versus 'No exact-field or theorem authority.'), Target URL (one spreadsheet id 1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no with fragment '#gid=200011' versus '#gid=200012'; an xlsx export carries no sheet gids, so which tabs those fragments name cannot be read from the export), Provenance ('GP-CLS-PROP-025; CL-AUD-074; EC-020; HA-008.' versus 'GP terminal record after HA-008') and Last reviewed ('2026-07-23' versus '2026-07-22'). Two Authority effect texts about one terminal closure is a material content difference under OP-CNS-001 §1; the EXACT_DUPLICATE_ROW class and the VOID-DUPLICATE disposition therefore do NOT apply, and the single-cell description in the finding text is corrected by the cells, not by this document.

**Corroborating register evidence.** The EC-021 closure was registered twice across the workbook on 2026-07-22/23: Closure Log row 21 'GP-CLS-141-v1.0' and row 22 'VOID-DUPLICATE-GP-CLS-141-v1.0'; Transition Register row 68 'VOID-DUPLICATE-TR-P01-021-20260723'; Review Independence row 42 'VOID-DUPLICATE-REV-EC021-CL074'. GP-COR-143-v1.0 swept those on 2026-07-23 (Relation Index row 239, REL-COR143-REG032) and its target hyper-tags do not name the Relation Index, so rows 232 and 235 survived as the unswept face of the event. Row 232 (earlier append position) carries the later Last reviewed date '2026-07-23' and the fuller Provenance; row 235 (later append position) carries '2026-07-22' and 'GP terminal record after HA-008'. Rows 233, 234 and 236–238 between and after them carry Last reviewed '2026-07-23', '2026-07-22', '2026-07-22', '2026-07-22', '2026-07-22', so this region's row order is not chronological by the date cell. The keeper rule still selects by append position and says so.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Gaps or duplicates require an additive provenance disclosure.'
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' Noted BECAUSE IT DOES NOT APPLY: both rows cite one Drive document as Source URL and assert one relation triple, so this is a register-row duplication of one relation, not a second relation.
- OP-PROT-019-v1.1 R17 §3 — 'Multiple successors of one old head are a publication conflict: hold only that target, preserve both branches, and append a reconciliation selecting or merging them with reasons. Never resolve it by silently overwriting proof bytes.' and 'Use the confirmed append position to order contenders, not a self-reported timestamp.'
- OP-PROT-019-v1.1 R17 §6 — 'No permanent deletion in this workflow.' and 'Moving or renaming must not be represented as a scientific verdict.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The same EC-021 double registration left VOID-DUPLICATE rows in four other tabs, all voided additively by GP-COR-143-v1.0: Closure Log row 22 'VOID-DUPLICATE-GP-CLS-141-v1.0' (Item 'EC-021', Closed at '2026-07-23'), Operator Decisions row 20 'VOID-DUPLICATE-OD-EC021-001' (Decision Type 'DUPLICATE_REGISTER_ENTRY', Notes 'Voided additively by GP-COR-143-v1.0; exact-ID duplicate preserved as provenance.', its own Recorded UTC '2026-07-22'), Transition Register row 68 'VOID-DUPLICATE-TR-P01-021-20260723' (Transition Type 'DUPLICATE_REGISTER_ENTRY', New State 'VOID — AUTHORITATIVE EC-021 TRANSITION IS TR-P01-021 AT ROW 68', Status 'VOID-DUPLICATE / GP-COR-143-v1.0', Recorded UTC '2026-07-23') and Review Independence row 42 'VOID-DUPLICATE-REV-EC021-CL074' (Status '... GP-COR-143-v1.0', Recorded UTC '2026-07-23'). Relation Index row 239 'REL-COR143-REG032' records that sweep ('CORRECTS_DUPLICATE_ROWS_IN' GP-REG-032; 'Marks later duplicate rows void while preserving their text; first complete entry remains authoritative for each event') with Target hyper-tag '[[OBJ:GP-REG-032]][[TAB:CLOSURE_LOG]][[TAB:OPERATOR_DECISIONS]][[TAB:TRANSITION_REGISTER]][[TAB:REVIEW_INDEPENDENCE]][[TAB:ARTIFACT_INDEX]]' — the Relation Index is not among the tabs it names. This pair is therefore the face of that registration event the sweep did not reach. The VOID-DUPLICATE precedent is directly on point for the EVENT and is discussed here, but NOT applied: the sweep voided rows whose text repeated an authoritative row, whereas rows 232 and 235 differ in Exact scope / meaning, Evidentiary effect, Authority effect, Target URL fragment, Provenance and Last reviewed, and which of them is the 'first complete entry' is exactly what the export cannot show (the later-appended row carries the earlier Last reviewed date). Compound row keys are the workbook's own form: Artifact Index rows 645/646 'CL-AUD-LSDER036-20260727-01@1zMq504' / '@1AcCpCe'; Duplicate Flags DUP-CLWO-20260727-07-STABLEKEY-20260730 'register compound keys CL-CLWO-20260727-07@1jqT3KJ and @1jGggrx; preserve both documents'.

**Which row keeps the original id, and why**

- Keeper: **row 232** keeps `REL-EC021-CLS141`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by a self-reported timestamp, and here the two orders disagree: row 232 (earlier) carries Last reviewed '2026-07-23', row 235 (later) carries '2026-07-22'. The workbook's own sweep of this registration event (GP-COR-143-v1.0) kept the 'first complete entry' in each of the three tabs it reached; by append position that is row 232 here. The rule is the one both earlier proposals applied and is applied mechanically so that no pair is resolved by this session's reading of its text.
- Explicitly not implied: Keeping the bare key is a KEY assignment, not a currency verdict. It does not make row 232's Authority effect or Exact scope wording current, nor row 235's stale.

**Proposed successor identifier**

- Row 235 → `REL-EC021-CLS141@REL-R235`
- Naming rule: Compound row key <Relation ID>@REL-R<n>, the Relation Index analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. REL = Relation Index (the workbook's name for registers/json/relations.json), R<n> = the 0-based row index tools/registers_check.py prints. REL is the tab's machine name's first three letters; it also happens to be the tab's own id prefix ('REL-036'), which is harmless in this position because the locator sits after '@' and is followed by '-R<n>', a shape no Relation ID in the export has (tools/collision_proposal_check.py confirms 'REL-036@REL-R39' matches no whole cell and no token in registers/json/), but it is recorded as a residual question in case the operator prefers a locator that cannot be read as an id prefix. THE REL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: The reidentified row's own Last reviewed date (REL-EC021-CLS141@2026-07-22) would be fully source-derived but would invert the append order R17 §3 prescribes (the later-appended row carries the earlier date), and a Drive-ID discriminator is unavailable because both rows cite one Source URL. The tab's own '-COLLISION-PROVENANCE' suffix is not adopted because it asserts NONOPERATIVE status, which this proposal does not assert of either row.
- Explicitly not proposed: The relation's declared source document, target and Status are not changed; only the register row key of row 235 is disambiguated. The VOID-DUPLICATE disposition GP-COR-143-v1.0 applied to the same event in the Closure Log, Transition Register and Review Independence tabs is NOT proposed here, because those voided rows repeated an authoritative row's text whereas rows 232 and 235 differ in six cells including Authority effect; labelling either VOID would destroy recorded state and would decide the content adjudication this proposal leaves open. No new Relation ID is minted for the same relation.
- Proposed disposition label: `COLLISION-PRESERVED / CONTENT-ADJUDICATION-PENDING / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-RELATION-INDEX-RELEC021CLS141-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / SAME DRIVE OBJECT / SAME RELATION / DIFFERENT SCOPE, EFFECT, PROVENANCE AND DATE CELLS |
| `File A` | Relation Index row 232 (2026-09-18 xlsx export) — REL-EC021-CLS141 — 'GP-CLS-141-v1.0' TERMINALLY_CLOSES 'EC-021' — Status: TERMINAL |
| `File B` | Relation Index row 235 (2026-09-18 xlsx export) — REL-EC021-CLS141 — 'GP-CLS-141-v1.0' TERMINALLY_CLOSES 'EC-021' — Status: TERMINAL |
| `Similarity / hash` | Same declared Relation ID, same relation triple ('GP-CLS-141-v1.0' TERMINALLY_CLOSES 'EC-021'), same Source URL https://docs.google.com/document/d/1Z8XWIS5gVvkiZL67sgx8Ca71LR2HFxU_96HULcuOCMI/edit and same Status 'TERMINAL'; Exact scope / meaning, Evidentiary effect, Authority effect, Target URL, Provenance, Last reviewed differ (Target URL only in its '#gid=' fragment) |
| `Risk` | Citation ambiguity and stale-authority-text routing risk; the bare ID does not identify one row, and the two Authority effect cells disagree |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 232 as REL-EC021-CLS141; register row 235 additively under compound key REL-EC021-CLS141@REL-R235; open an OP-CNS-001 §1 content-adjudication item on the Exact scope / Evidentiary effect / Authority effect / Provenance difference; no merge, no deletion, no VOID label, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Relation Index |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Relation Index, row 235, column 'Relation ID'; `REL-EC021-CLS141` → `REL-EC021-CLS141@REL-R235`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Which of the two Authority effect texts ('Authority derives from HA-008 item-level approval.' / 'No exact-field or theorem authority.') and which Exact scope wording is the current reading of GP-CLS-141-v1.0's closure is the content-adjudication item; it is handed to the operator unresolved. Nothing here moves EC-021 or HA-008.
- Which workbook tabs the Target URL fragments '#gid=200011' and '#gid=200012' name cannot be read from the xlsx export (it carries no gids); the operator can resolve them against the live Sheet.
- registers/KNOWN_FINDINGS.json described this pair as 'identical apart from Last reviewed ... an exact duplicate row', and registers/README.md repeated that wording, until the commit that adds this document; the cells give six differing cells, and both now record the recomputed counts, each saying what it said before. Those are this repository's own rationale and prose, not exported data. The finding key is unchanged (it is the string tools/registers_check.py prints) and the register itself is untouched: no row of relations was edited, and which of the two rows should keep the identifier remains the operator's content adjudication.
- The region's row order is not chronological by Last reviewed (row 232 '2026-07-23' precedes row 234 '2026-07-22' and row 235 '2026-07-22'); the operator should confirm append order from the workbook's revision history before executing this record.
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

## Part 2 — Two duplicate Review IDs in the Review Independence ledger

Both pairs are **two reviews of two different exact objects under one review id**, recorded at one `Recorded UTC` by one reviewer line. This tab carries no URL or Drive ID column; the reviewed object is named by `Exact Object ID`, and in both pairs the two differ. Both reviews are preserved in each pair; the identifier is disambiguated; no score, class or status quoted from the ledger is evaluated. The bare ids are cited by other tabs, and for `REV-P02-GP-INTERVAL-001` three of those citations name the later row's object — recorded as a residual question, not repaired.

| Review ID | Rows | Class | Keeper row | Reidentified row | Proposed successor row key |
|---|---|---|---|---|---|
| `REV-P12-GP-006` | 15, 17 | DIFFERENT OBJECTS | 15 | 17 | `REV-P12-GP-006@RVL-R17` |
| `REV-P02-GP-INTERVAL-001` | 19, 21 | DIFFERENT OBJECTS | 19 | 21 | `REV-P02-GP-INTERVAL-001@RVL-R21` |

### 2.1 `REV-P12-GP-006` — rows 15, 17

> Finding key: `review_ledger: duplicate key 'REV-P12-GP-006' at rows 15 and 17`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the review_ledger tab was first keyed (its Review ID column was never read by the checker before). Row 15: Recorded UTC '2026-07-21T19:00:00Z'; Exact Object ID 'EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_CURRENT_7F_SOURCE'; Reviewer Line 'GP'; Review Role 'SAME-LINE CURRENT-SOURCE ADVERSARIAL EXECUTION'. Row 17: Recorded UTC '2026-07-21T19:00:00Z'; Exact Object ID 'EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_V1_1'; Reviewer Line 'GP'; Review Role 'AUTHOR AMENDMENT AND 21-CONTROL EXECUTION'. Two reviews of two different exact objects under one review id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Exact Object ID` — differ: row 15 → EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_CURRENT_7F_SOURCE; row 17 → EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_V1_1.

#### Row 15 — proposed KEEPER row, verbatim

`registers/json/review_ledger.json` rows[15], canonical 829 bytes, SHA-256 `09d423aee987440ef9010f220ca7496950a8ac2b51137dcf4e439ad78c285044`:

```json
["REV-P12-GP-006","2026-07-21T19:00:00Z","EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_CURRENT_7F_SOURCE","GP","OpenAI GPT-5.6 Thinking","SAME-LINE CURRENT-SOURCE ADVERSARIAL EXECUTION","Current capsule; GP-AUD-057; GP-AUD-056; GP-AUD-059; GP-DATA-060","Strictly decoded current 30,606-byte 7f source and direct runtime results","TRUE","TRUE","TRUE","FALSE","Same GP lineage and source; new failure-path fixtures and fresh current-state extraction","Strict current-source reconstruction, compile, frozen self-test run, mocked solver failure, identity-substitution census, singular-Hessian fixture","TRUE","FALSE","FALSE","NONE","FULL","TRUE","Current-source exact-byte and adversarial numerical qualification","0.08","LOW / SAME-LINE EXECUTION","PASS-SOURCE-IDENTITY / PASS-13-OF-13 / CONFIRM-3-OF-3-DEFECTS / OUTSIDE REVIEW OPEN"]
```

#### Row 17 — proposed REIDENTIFIED row, verbatim

`registers/json/review_ledger.json` rows[17], canonical 704 bytes, SHA-256 `c2911d076793d95d7dd3ae54111e65c33744724e3ae2e7d43f64af89e49dc6a3`:

```json
["REV-P12-GP-006","2026-07-21T19:00:00Z","EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_V1_1","GP","OpenAI GPT-5.6 Thinking","AUTHOR AMENDMENT AND 21-CONTROL EXECUTION","GP-AUD-056; GP-DATA-054-v1.0; GP-COR-058; amended source; GP-AUD-059; GP-REQ-056","Independent defect audit and exact amended local source","TRUE","TRUE","TRUE","TRUE","Same source-author line; inherited GP-AUD-056 target definitions and own fixture design","Source-level implementation, exact hashing, syntax check, and 21-control execution","TRUE","FALSE","FALSE","NONE","FULL","TRUE","Author amendment and adversarial fixture execution","0.10","LOW FOR QUALIFICATION / PASS AMENDMENT","PASS-AUTHOR-AMENDMENT / OUTSIDE REVIEW PENDING"]
```

#### Where the two rows differ

| Field | Row 15 (keeper) | Row 17 (reidentified) |
|---|---|---|
| **Exact Object ID** | EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_CURRENT_7F_SOURCE | EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_V1_1 |
| **Review Role** | SAME-LINE CURRENT-SOURCE ADVERSARIAL EXECUTION | AUTHOR AMENDMENT AND 21-CONTROL EXECUTION |
| **Sources Read** | Current capsule; GP-AUD-057; GP-AUD-056; GP-AUD-059; GP-DATA-060 | GP-AUD-056; GP-DATA-054-v1.0; GP-COR-058; amended source; GP-AUD-059; GP-REQ-056 |
| **Primary Sources Read** | Strictly decoded current 30,606-byte 7f source and direct runtime results | Independent defect audit and exact amended local source |
| **Same Fixture Reused** | FALSE | TRUE |
| **Inherited Assumptions** | Same GP lineage and source; new failure-path fixtures and fresh current-state extraction | Same source-author line; inherited GP-AUD-056 target definitions and own fixture design |
| **Independent Route** | Strict current-source reconstruction, compile, frozen self-test run, mocked solver failure, identity-substitution census, singular-Hessian fixture | Source-level implementation, exact hashing, syntax check, and 21-control execution |
| **Methodological Family** | Current-source exact-byte and adversarial numerical qualification | Author amendment and adversarial fixture execution |
| **Independence Score** | 0.08 | 0.10 |
| **Independence Class** | LOW / SAME-LINE EXECUTION | LOW FOR QUALIFICATION / PASS AMENDMENT |
| **Status** | PASS-SOURCE-IDENTITY / PASS-13-OF-13 / CONFIRM-3-OF-3-DEFECTS / OUTSIDE REVIEW OPEN | PASS-AUTHOR-AMENDMENT / OUTSIDE REVIEW PENDING |

Identical in both rows: `Review ID`, `Recorded UTC`, `Reviewer Line`, `Reviewer`, `Original Derivation Visible`, `Independent Reconstruction`, `Same Code Reused`, `Object Definition Checked`, `Algebra Only`, `Independent Notation`, `Observer Mode`, `Lineage Access`, `Internal Conclusions Visible`.

Cell count: 24 columns compared, 13 identical, 11 differing (rows 15 and 17 of `registers/json/review_ledger.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two reviews of two different exact objects. Row 15 reviews Exact Object ID 'EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_CURRENT_7F_SOURCE' in Review Role 'SAME-LINE CURRENT-SOURCE ADVERSARIAL EXECUTION' (Independence Score '0.08', Independence Class 'LOW / SAME-LINE EXECUTION', Status 'PASS-SOURCE-IDENTITY / PASS-13-OF-13 / CONFIRM-3-OF-3-DEFECTS / OUTSIDE REVIEW OPEN'); row 17 reviews Exact Object ID 'EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_V1_1' in Review Role 'AUTHOR AMENDMENT AND 21-CONTROL EXECUTION' (Independence Score '0.10', Independence Class 'LOW FOR QUALIFICATION / PASS AMENDMENT', Status 'PASS-AUTHOR-AMENDMENT / OUTSIDE REVIEW PENDING'). Thirteen of twenty-four cells agree, among them Recorded UTC '2026-07-21T19:00:00Z', Reviewer Line 'GP' and the Reviewer cell; eleven cells differ, among them Exact Object ID, Review Role, Sources Read, Same Fixture Reused ('FALSE' versus 'TRUE'), Independence Score, Independence Class and Status. This tab carries no URL; the object a row reviews is named by Exact Object ID, and the two differ. One review id assigned to two reviews; every score and class quoted is the ledger's own word and is not evaluated.

**Corroborating register evidence.** The bare id REV-P12-GP-006 is cited outside this tab: Evidence Lineage rows 36 (EV-P12-GP-AUDIT-059) and 37 (EV-P12-GP-DATA-060) carry Review ID 'REV-P12-GP-006' with Exact Object ID '..._CURRENT_7F_SOURCE', row 15's object; Relation Index row 22 (REL-023, Target object 'GP-DATA-054 current production qualification') carries Provenance 'TR-P12-006; REV-P12-GP-006'. Under the keeper rule row 15 keeps the bare id, so those three citations continue to name the object their own Exact Object ID cells name; that is an observation about cells, not a verdict on them. Row 16 between the pair (REV-EC014-GP-001) carries Recorded UTC '2026-07-21T19:20:00Z', later than row 17's '19:00:00Z', so the region's append order and timestamp order are not monotone.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Review Independence ledger's own executed form for a duplicate registration is the VOID-DUPLICATE row: row 41 'VOID-DUPLICATE-REV-EC019-CL072' (Review Role 'DUPLICATE REVIEW-DISCLOSURE ROW', Status 'VOID — AUTHORITATIVE EC-019 DISCLOSURE IS REV-EC019-CL072-OPT-A AT ROW 42; GP-COR-143-v1.0'), row 42 'VOID-DUPLICATE-REV-EC021-CL074', row 52 'VOID-DUPLICATE-REV-EC019-CLPREREG200', row 78 'VOID-DUPLICATE-REV-P01-GPAUD199-0055' and row 80 'VOID-DUPLICATE-EC019-CLAUD200'. Every one of those voided a second row about ONE review of ONE exact object; none applies to two reviews of two exact objects. Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076'), DUP-ID-HELP-BOARD-20260724 ('preserve first occurrence; assign next unused IDs to later rows') and DUP-DISPATCH-DQ032-20260726 ('Same primary key on distinct exact objects ... preserve row 31 as DQ-032; reidentify row 33 as DQ-035').

**Which row keeps the original id, and why**

- Keeper: **row 15** keeps `REV-P12-GP-006`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Recorded UTC, so the timestamp could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 17 → `REV-P12-GP-006@RVL-R17`
- Naming rule: Compound row key <Review ID>@RVL-R<n>, the Review Independence analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. RVL = the review_ledger tab (the workbook's sheet 'Review Independence'), R<n> = the 0-based row index tools/registers_check.py prints. RVL is chosen instead of REV because REV is the tab's own id prefix ('REV-P12-GP-006') and a locator should not be confusable with one, and instead of RI because two letters would not be tab-scoped enough to survive a future 'Review Queue' locator. THE RVL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is unavailable for the Review Independence ledger, which carries no URL or Drive ID column.
- Explicitly not proposed: Neither review's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 17 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-REVIEW-INDEPENDENCE-REVP12GP006-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT REVIEWS OF DIFFERENT EXACT OBJECTS |
| `File A` | Review Independence row 15 (2026-09-18 xlsx export) — REV-P12-GP-006 — Exact Object ID EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_CURRENT_7F_SOURCE — Review Role: SAME-LINE CURRENT-SOURCE ADVERSARIAL EXECUTION — Status: PASS-SOURCE-IDENTITY / PASS-13-OF-13 / CONFIRM-3-OF-3-DEFECTS / OUTSIDE REVIEW OPEN |
| `File B` | Review Independence row 17 (2026-09-18 xlsx export) — REV-P12-GP-006 — Exact Object ID EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_V1_1 — Review Role: AUTHOR AMENDMENT AND 21-CONTROL EXECUTION — Status: PASS-AUTHOR-AMENDMENT / OUTSIDE REVIEW PENDING |
| `Similarity / hash` | Same declared Review ID; no URL column in this tab; different Exact Object ID ('EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_CURRENT_7F_SOURCE' versus 'EXACT_FIELD_PAIRING_INSTRUMENT_GP_DATA_054_V1_1'); Exact Object ID, Review Role, Sources Read, Primary Sources Read, Same Fixture Reused, Inherited Assumptions, Independent Route, Methodological Family, Independence Score, Independence Class, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects; the bare id is cited by other tabs |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 15 as REV-P12-GP-006; register row 17 additively under compound key REV-P12-GP-006@RVL-R17; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Review Independence |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Review Independence, row 17, column 'Review ID'; `REV-P12-GP-006` → `REV-P12-GP-006@RVL-R17`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the later row should instead be renumbered to the next unused number of its series (the export's highest 'REV-P12-GP-<nnn>' is REV-P12-GP-007, so REV-P12-GP-008 is the next unused; the remedy Duplicate Flags DUP-CST073-20260727 and DUP-ID-HELP-BOARD-20260724 record) is not decided here.
- The three cross-tab citations of the bare id (Evidence Lineage rows 36, 37; Relation Index row 22) all name row 15's object by their own Exact Object ID or Target object cells; whether any of them intended row 17's review is not decidable from the export.
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 2.2 `REV-P02-GP-INTERVAL-001` — rows 19, 21

> Finding key: `review_ledger: duplicate key 'REV-P02-GP-INTERVAL-001' at rows 19 and 21`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the review_ledger tab was first keyed (its Review ID column was never read by the checker before). Row 19: Recorded UTC '2026-07-21T19:55:00Z'; Exact Object ID 'ADJACENCY_TO_ONE_CUBIC_TYPED_MS'; Reviewer Line 'GP'; Review Role 'AUTHOR-LINE EXACT-RATIONAL CERTIFICATE RESPONSE'. Row 21: Recorded UTC '2026-07-21T19:55:00Z'; Exact Object ID 'P02_DETERMINISTIC_INTERVAL_CERTIFICATE'; Reviewer Line 'GP'; Review Role 'INDEPENDENT IMPLEMENTATION / AUTHOR-LINE COMPONENT EXECUTION'. Two reviews of two different exact objects under one review id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Exact Object ID` — differ: row 19 → ADJACENCY_TO_ONE_CUBIC_TYPED_MS; row 21 → P02_DETERMINISTIC_INTERVAL_CERTIFICATE.

#### Row 19 — proposed KEEPER row, verbatim

`registers/json/review_ledger.json` rows[19], canonical 792 bytes, SHA-256 `9fd46da3be828dac8499ffe9bb7aa9c3974f82d9254d2c54bf0d087b5316c463`:

```json
["REV-P02-GP-INTERVAL-001","2026-07-21T19:55:00Z","ADJACENCY_TO_ONE_CUBIC_TYPED_MS","GP","OpenAI GPT-5.6 Thinking","AUTHOR-LINE EXACT-RATIONAL CERTIFICATE RESPONSE","CL-AUD-044 in full; frozen CL-DATA-044 source; GP-DER-047; existing P0.2 object and certificate chain","Frozen CL source and sample plan; GP-DER-047 deterministic inequalities","TRUE","FALSE","TRUE","TRUE","GP-DER-047 inequalities; CL frozen sample plan and float64 fixtures; declared exact six-pin object","Independently authored exact-rational continuous inequalities replacing finite grids, with new fail-closed negative controls","TRUE","TRUE","FALSE","NONE","FULL","TRUE","Same-line exact-rational deterministic certification","0.08","LOW FOR THEOREM / MODERATE FOR CERTIFICATE","PASS-CERTIFICATE / NON-GP-RERUN-PENDING"]
```

#### Row 21 — proposed REIDENTIFIED row, verbatim

`registers/json/review_ledger.json` rows[21], canonical 779 bytes, SHA-256 `d398dcdfe3ee25abfe86b20eb88c34c334db33dc88b02179a34c78c37bd9ac2f`:

```json
["REV-P02-GP-INTERVAL-001","2026-07-21T19:55:00Z","P02_DETERMINISTIC_INTERVAL_CERTIFICATE","GP","OpenAI GPT-5.6 Thinking","INDEPENDENT IMPLEMENTATION / AUTHOR-LINE COMPONENT EXECUTION","CL-DATA-044 raw source; CL-AUD-044; GP-DER-047; generated GP-DATA-066 and GP-AUD-067","Reference CL source and exact candidate theorem objects","TRUE","TRUE","FALSE","TRUE","Same theorem-author lineage, same frozen 864 parameter schedule, same declared constants and geometry","Custom exact-rational and nextafter interval arithmetic; continuous-box polynomial enclosures; full-circle Hessian certification","TRUE","FALSE","TRUE","NONE","FULL","TRUE","Outward-rounded interval certificate","0.27","LOW-MODERATE FOR COMPONENT / NOT THEOREM-INDEPENDENT","PASS-LOCAL / OUTSIDE EXECUTION PENDING"]
```

#### Where the two rows differ

| Field | Row 19 (keeper) | Row 21 (reidentified) |
|---|---|---|
| **Exact Object ID** | ADJACENCY_TO_ONE_CUBIC_TYPED_MS | P02_DETERMINISTIC_INTERVAL_CERTIFICATE |
| **Review Role** | AUTHOR-LINE EXACT-RATIONAL CERTIFICATE RESPONSE | INDEPENDENT IMPLEMENTATION / AUTHOR-LINE COMPONENT EXECUTION |
| **Sources Read** | CL-AUD-044 in full; frozen CL-DATA-044 source; GP-DER-047; existing P0.2 object and certificate chain | CL-DATA-044 raw source; CL-AUD-044; GP-DER-047; generated GP-DATA-066 and GP-AUD-067 |
| **Primary Sources Read** | Frozen CL source and sample plan; GP-DER-047 deterministic inequalities | Reference CL source and exact candidate theorem objects |
| **Independent Reconstruction** | FALSE | TRUE |
| **Same Code Reused** | TRUE | FALSE |
| **Inherited Assumptions** | GP-DER-047 inequalities; CL frozen sample plan and float64 fixtures; declared exact six-pin object | Same theorem-author lineage, same frozen 864 parameter schedule, same declared constants and geometry |
| **Independent Route** | Independently authored exact-rational continuous inequalities replacing finite grids, with new fail-closed negative controls | Custom exact-rational and nextafter interval arithmetic; continuous-box polynomial enclosures; full-circle Hessian certification |
| **Algebra Only** | TRUE | FALSE |
| **Independent Notation** | FALSE | TRUE |
| **Methodological Family** | Same-line exact-rational deterministic certification | Outward-rounded interval certificate |
| **Independence Score** | 0.08 | 0.27 |
| **Independence Class** | LOW FOR THEOREM / MODERATE FOR CERTIFICATE | LOW-MODERATE FOR COMPONENT / NOT THEOREM-INDEPENDENT |
| **Status** | PASS-CERTIFICATE / NON-GP-RERUN-PENDING | PASS-LOCAL / OUTSIDE EXECUTION PENDING |

Identical in both rows: `Review ID`, `Recorded UTC`, `Reviewer Line`, `Reviewer`, `Original Derivation Visible`, `Same Fixture Reused`, `Object Definition Checked`, `Observer Mode`, `Lineage Access`, `Internal Conclusions Visible`.

Cell count: 24 columns compared, 10 identical, 14 differing (rows 19 and 21 of `registers/json/review_ledger.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two reviews of two different exact objects. Row 19 reviews Exact Object ID 'ADJACENCY_TO_ONE_CUBIC_TYPED_MS' in Review Role 'AUTHOR-LINE EXACT-RATIONAL CERTIFICATE RESPONSE' (Independence Score '0.08', Independence Class 'LOW FOR THEOREM / MODERATE FOR CERTIFICATE', Status 'PASS-CERTIFICATE / NON-GP-RERUN-PENDING'); row 21 reviews Exact Object ID 'P02_DETERMINISTIC_INTERVAL_CERTIFICATE' in Review Role 'INDEPENDENT IMPLEMENTATION / AUTHOR-LINE COMPONENT EXECUTION' (Independence Score '0.27', Independence Class 'LOW-MODERATE FOR COMPONENT / NOT THEOREM-INDEPENDENT', Status 'PASS-LOCAL / OUTSIDE EXECUTION PENDING'). Ten of twenty-four cells agree, among them Recorded UTC '2026-07-21T19:55:00Z', Reviewer Line 'GP' and the Reviewer cell; fourteen cells differ, among them Exact Object ID, Review Role, Independent Reconstruction ('FALSE' versus 'TRUE'), Same Code Reused ('TRUE' versus 'FALSE'), Algebra Only, Independent Notation, Independence Score, Independence Class and Status. This tab carries no URL; the reviewed object is named by Exact Object ID, and the two differ. One review id assigned to two reviews; the scores and classes are the ledger's own words and are not evaluated.

**Corroborating register evidence.** The bare id REV-P02-GP-INTERVAL-001 is cited outside this tab for BOTH objects: Evidence Lineage row 39 (EV-P02-INTERVAL-068, Exact Object ID 'ADJACENCY_TO_ONE_CUBIC_TYPED_MS', row 19's object) and rows 47 (EV-P02-GP-DATA-066) and 48 (EV-P02-GP-AUD-067), both Exact Object ID 'P02_DETERMINISTIC_INTERVAL_CERTIFICATE', row 21's object; Transition Alarms row 59 (Code 'P02_INTERVAL_OUTSIDE_EXECUTION_PENDING', Exact object 'P02_DETERMINISTIC_INTERVAL_CERTIFICATE', Reference ID 'REV-P02-GP-INTERVAL-001'), again row 21's object. Re-keying row 21 alone would leave Evidence Lineage rows 47 and 48 and Alarms row 59 naming the keeper's id for row 21's object; a cross-tab citation repair is therefore part of any full remedy and is recorded below as a residual question, not proposed. Row 20 between the pair (REV-P02-CL-003) shares row 19's Recorded UTC and object.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Review Independence ledger's own executed form for a duplicate registration is the VOID-DUPLICATE row: row 41 'VOID-DUPLICATE-REV-EC019-CL072' (Review Role 'DUPLICATE REVIEW-DISCLOSURE ROW', Status 'VOID — AUTHORITATIVE EC-019 DISCLOSURE IS REV-EC019-CL072-OPT-A AT ROW 42; GP-COR-143-v1.0'), row 42 'VOID-DUPLICATE-REV-EC021-CL074', row 52 'VOID-DUPLICATE-REV-EC019-CLPREREG200', row 78 'VOID-DUPLICATE-REV-P01-GPAUD199-0055' and row 80 'VOID-DUPLICATE-EC019-CLAUD200'. Every one of those voided a second row about ONE review of ONE exact object; none applies to two reviews of two exact objects. Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076'), DUP-ID-HELP-BOARD-20260724 ('preserve first occurrence; assign next unused IDs to later rows') and DUP-DISPATCH-DQ032-20260726 ('Same primary key on distinct exact objects ... preserve row 31 as DQ-032; reidentify row 33 as DQ-035').

**Which row keeps the original id, and why**

- Keeper: **row 19** keeps `REV-P02-GP-INTERVAL-001`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Recorded UTC, so the timestamp could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 21 → `REV-P02-GP-INTERVAL-001@RVL-R21`
- Naming rule: Compound row key <Review ID>@RVL-R<n>, the Review Independence analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. RVL = the review_ledger tab (the workbook's sheet 'Review Independence'), R<n> = the 0-based row index tools/registers_check.py prints. RVL is chosen instead of REV because REV is the tab's own id prefix ('REV-P12-GP-006') and a locator should not be confusable with one, and instead of RI because two letters would not be tab-scoped enough to survive a future 'Review Queue' locator. THE RVL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is unavailable for the Review Independence ledger, which carries no URL or Drive ID column.
- Explicitly not proposed: Neither review's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 21 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-REVIEW-INDEPENDENCE-REVP02GPINTERVAL001-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT REVIEWS OF DIFFERENT EXACT OBJECTS |
| `File A` | Review Independence row 19 (2026-09-18 xlsx export) — REV-P02-GP-INTERVAL-001 — Exact Object ID ADJACENCY_TO_ONE_CUBIC_TYPED_MS — Review Role: AUTHOR-LINE EXACT-RATIONAL CERTIFICATE RESPONSE — Status: PASS-CERTIFICATE / NON-GP-RERUN-PENDING |
| `File B` | Review Independence row 21 (2026-09-18 xlsx export) — REV-P02-GP-INTERVAL-001 — Exact Object ID P02_DETERMINISTIC_INTERVAL_CERTIFICATE — Review Role: INDEPENDENT IMPLEMENTATION / AUTHOR-LINE COMPONENT EXECUTION — Status: PASS-LOCAL / OUTSIDE EXECUTION PENDING |
| `Similarity / hash` | Same declared Review ID; no URL column in this tab; different Exact Object ID ('ADJACENCY_TO_ONE_CUBIC_TYPED_MS' versus 'P02_DETERMINISTIC_INTERVAL_CERTIFICATE'); Exact Object ID, Review Role, Sources Read, Primary Sources Read, Independent Reconstruction, Same Code Reused, Inherited Assumptions, Independent Route, Algebra Only, Independent Notation, Methodological Family, Independence Score, Independence Class, Status differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects; the bare id is cited by other tabs |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 19 as REV-P02-GP-INTERVAL-001; register row 21 additively under compound key REV-P02-GP-INTERVAL-001@RVL-R21; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Review Independence |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Review Independence, row 21, column 'Review ID'; `REV-P02-GP-INTERVAL-001` → `REV-P02-GP-INTERVAL-001@RVL-R21`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Three cross-tab citations (Evidence Lineage rows 47 and 48; Transition Alarms row 59) name row 21's object under the bare id; after any re-keying of row 21 they would point at the keeper unless updated. Which citations to update, and whether that makes the keeper choice here the wrong way round (row 21's object is the one cited more often), is an operator decision recorded and not taken.
- Whether the later row should instead be renumbered within its series (the export has 'REV-P02-GP-008' and 'REV-P02-GP-DM2-001' but no second 'REV-P02-GP-INTERVAL-<nnn>', so REV-P02-GP-INTERVAL-002 is unassigned) is not decided here.
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

## Part 3 — One duplicate Definition ID in the Definition Registry

The pair is **two different definitions under one id** — two canonical terms, two definition texts, two Source URLs — inside a block (rows 48–52) that was numbered out of order in one pass. Both definitions are preserved; the identifier is disambiguated. Whether the later row's term duplicates the neighbouring `DEF-050` in substance is a content question left to the operator.

| Definition ID | Rows | Class | Keeper row | Reidentified row | Proposed successor row key |
|---|---|---|---|---|---|
| `DEF-049` | 48, 50 | DIFFERENT OBJECTS | 48 | 50 | `DEF-049@DEF-R50` |

### 3.1 `DEF-049` — rows 48, 50

> Finding key: `definitions: duplicate key 'DEF-049' at rows 48 and 50`

> Finding text as recorded in `KNOWN_FINDINGS.json`: Found 2026-09-19 when the definitions tab was first keyed (its Definition ID column was never read by the checker before). Row 48: Canonical term 'Certified capture mass'; Scope 'P0.1/P0.2 probability lower bounds'; Status 'ACTIVE'; Last reviewed '2026-07-22'. Row 50: Canonical term 'Interval-certified degree-four corridor box'; Scope 'Fixed field layer, chart, r, box, witness parameters, and implemented gate ledger'; Status 'ACTIVE-CANDIDATE'; Last reviewed '2026-07-22'. Two canonical terms under one definition id. Exported faithfully; OP-CNS-001 §2 requires collisions to be preserved and disambiguated in an append-only collision registry, not merged. Covered, one record per key, by registers/collision_proposal_2026-09-19b.json (companion registers/COLLISION_PROPOSAL_2026-09-19b.md); nothing has been repaired. Until 2026-09-20 this rationale ended "Not yet covered by any collision proposal".

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. Rule cells read: `Source URL` — differ: row 48 → https://docs.google.com/document/d/1kNp3vZxyA3kcorgwkdkL50MLwZQ7IiSayMh--aXZIiY/edit; row 50 → https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit.

#### Row 48 — proposed KEEPER row, verbatim

`registers/json/definitions.json` rows[48], canonical 770 bytes, SHA-256 `0b7c65d74109f4abf00d8b13c241405c36982672f309f1a9c170eaf499d3cbb4`:

```json
["DEF-049","Certified capture mass","The determinant-weighted probability mass of a region for which capture has been proved on every point under the identical governing law, branch convention, field layer, and r regime.","P0.1/P0.2 probability lower bounds","Coordinate-box membership mass, typed mass, finite-grid capture frequency, or Monte Carlo success rate is not certified capture mass without a proved inclusion.","capture lower bound; certified corridor mass; B subset A_r","[[DEF:CERTIFIED_CAPTURE_MASS]]","P0.1; Palm_weight; inclusion; exact_field; probability","GP-AUD-103-v1.0; GP-REQ-105-v1.0","Prevents box-mass laundering; no theorem authority","ACTIVE","2026-07-22","https://docs.google.com/document/d/1kNp3vZxyA3kcorgwkdkL50MLwZQ7IiSayMh--aXZIiY/edit"]
```

#### Row 50 — proposed REIDENTIFIED row, verbatim

`registers/json/definitions.json` rows[50], canonical 948 bytes, SHA-256 `2ca0febeff2dd4106e442fa0707dd93c1f0fa9f2bc09d5e96ba6a2c9548f1126`:

```json
["DEF-049","Interval-certified degree-four corridor box","A positive-width finite-jet parameter box for which interval sufficient conditions certify every declared degree-four typing, selected-branch cone, strip, transit, section, chart, and capture gate over the entire continuum box.","Fixed field layer, chart, r, box, witness parameters, and implemented gate ledger","Does not imply Gaussian/Palm mass, exact-field capture, exact-torus validity, r-interval uniformity, or a theorem without separate evidence.","Route-B box; interval corridor; continuum degree-four box","[[DEF:INTERVAL_CERTIFIED_DEGREE4_CORRIDOR_BOX]]","P0.1; RouteB; interval; degree4; continuum_box; capture","GP-DER-106-v1.0; GP-DATA-106-v1.0; GP-AUD-107-v1.0","Defines the component and its exclusions only; outside review and terminal authority absent","ACTIVE-CANDIDATE","2026-07-22","https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit"]
```

#### Where the two rows differ

| Field | Row 48 (keeper) | Row 50 (reidentified) |
|---|---|---|
| **Canonical term** | Certified capture mass | Interval-certified degree-four corridor box |
| **Exact operational definition** | The determinant-weighted probability mass of a region for which capture has been proved on every point under the identical governing law, branch convention, field layer, and r regime. | A positive-width finite-jet parameter box for which interval sufficient conditions certify every declared degree-four typing, selected-branch cone, strip, transit, section, chart, and capture gate over the entire continuum box. |
| **Scope** | P0.1/P0.2 probability lower bounds | Fixed field layer, chart, r, box, witness parameters, and implemented gate ledger |
| **Explicit exclusions** | Coordinate-box membership mass, typed mass, finite-grid capture frequency, or Monte Carlo success rate is not certified capture mass without a proved inclusion. | Does not imply Gaussian/Palm mass, exact-field capture, exact-torus validity, r-interval uniformity, or a theorem without separate evidence. |
| **Aliases / search terms** | capture lower bound; certified corridor mass; B subset A_r | Route-B box; interval corridor; continuum degree-four box |
| **Hyper-tag** | [[DEF:CERTIFIED_CAPTURE_MASS]] | [[DEF:INTERVAL_CERTIFIED_DEGREE4_CORRIDOR_BOX]] |
| **Object tags** | P0.1; Palm_weight; inclusion; exact_field; probability | P0.1; RouteB; interval; degree4; continuum_box; capture |
| **Authority source** | GP-AUD-103-v1.0; GP-REQ-105-v1.0 | GP-DER-106-v1.0; GP-DATA-106-v1.0; GP-AUD-107-v1.0 |
| **Authority effect** | Prevents box-mass laundering; no theorem authority | Defines the component and its exclusions only; outside review and terminal authority absent |
| **Status** | ACTIVE | ACTIVE-CANDIDATE |
| **Source URL** | https://docs.google.com/document/d/1kNp3vZxyA3kcorgwkdkL50MLwZQ7IiSayMh--aXZIiY/edit | https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit |

Identical in both rows: `Definition ID`, `Last reviewed`.

Cell count: 13 columns compared, 2 identical, 11 differing (rows 48 and 50 of `registers/json/definitions.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different definitions. Row 48 defines 'Certified capture mass' ('The determinant-weighted probability mass of a region for which capture has been proved on every point ...', Scope 'P0.1/P0.2 probability lower bounds', Authority source 'GP-AUD-103-v1.0; GP-REQ-105-v1.0', Status 'ACTIVE', Source URL document 1kNp3vZxyA3kcorgwkdkL50MLwZQ7IiSayMh--aXZIiY); row 50 defines 'Interval-certified degree-four corridor box' ('A positive-width finite-jet parameter box for which interval sufficient conditions certify every declared degree-four typing ...', Scope 'Fixed field layer, chart, r, box, witness parameters, and implemented gate ledger', Authority source 'GP-DER-106-v1.0; GP-DATA-106-v1.0; GP-AUD-107-v1.0', Status 'ACTIVE-CANDIDATE', Source URL document 1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI). Only Definition ID and Last reviewed ('2026-07-22') agree; eleven of thirteen cells differ, including the canonical term, the definition text and the Source URL. One definition id assigned twice; two terms, not two readings of one term.

**Corroborating register evidence.** Row 49 between the pair is DEF-050 'Full-dimensional interval-certified degree-four corridor box' with the same Source URL (1ve_dMz…), the same Authority source ('GP-DER-106-v1.0; GP-DATA-106-v1.0; GP-AUD-107-v1.0') and Scope 'P0.1 Route-B finite-dimensional continuum component' as row 50; rows 51 and 52 then carry DEF-052 and DEF-051, in that order, with Status suffix '-CORRECTED'. The block 048–052 was therefore numbered out of order in one pass; whether row 50's term duplicates DEF-050 in substance is a content question not decided here. Relation Index row 136 (REL-130, 'Coordinate-box mass P_MS(B)' DISTINCT_FROM 'Certified capture mass', Provenance 'GP-AUD-103 F4') and row 141 (REL-132, Target object 'GP-REQ-105-v1.0', Target URL 1kNp3v…) tie row 48's term and Source URL to GP-AUD-103 / GP-REQ-105; Artifact Index row 69 (GP-DER-106-v1.0, Source 1ve_dMz…) ties row 50's Source URL to GP-DER-106.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES BY ANALOGY: the two rows describe two different objects under one register key. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — 'duplicate flagging', 'provenance repair' and 'index maintenance' are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** The Definition Registry has no executed collision form of its own: no row carries a VOID, COLLISION or compound key. Rows 51 and 52 carry 'DEF-052' then 'DEF-051' with Status 'ACTIVE-CANDIDATE-CORRECTED' and 'ACTIVE-COMPONENT-CORRECTED', so the tab has absorbed corrections by status suffix and out-of-order numbering rather than by re-keying. Duplicate Flags precedents for one key on two different bodies: DUP-CST073-20260727 ('Exact Test ID collision; distinct test bodies ... Preserved both; reidentified later rows as CST-075 and CST-076') and DUP-ID-HELP-BOARD-20260724 ('preserve first occurrence; assign next unused IDs to later rows; cite GP-COR-190').

**Which row keeps the original id, and why**

- Keeper: **row 48** keeps `DEF-049`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp (the two rows carry one Last reviewed date, so the date could not order them in any case). The rule is the same one applied to every pair in all three proposals, so that the keeper of the bare key is never chosen by this session's reading of which object matters more. Keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two objects, does not execute, close or reopen either, and grants no review or independence credit to either.

**Proposed successor identifier**

- Row 50 → `DEF-049@DEF-R50`
- Naming rule: Compound row key <Definition ID>@DEF-R<n>, the Definition Registry analogue of the AIDX and EVL forms the 2026-09-18 and 2026-09-19 proposals issued for the Artifact Index and Evidence Lineage, so that every successor key in the three documents is read the same way. DEF = Definition Registry (registers/json/definitions.json), R<n> = the 0-based row index tools/registers_check.py prints. DEF is the tab's machine name's first three letters and is also the tab's id prefix ('DEF-049'); harmless after '@' and before '-R<n>' (no Definition ID in the export has that shape, and the checker confirms the compound key matches nothing), but recorded as a residual question alongside REL. THE DEF ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN COMPOUND-KEY FORM, NOT A LITERAL QUOTATION; it requires operator ratification, and ratifying the row-locator form once ratifies it for all three proposals.
- Alternative discriminator considered: Renumbering the later row to the next unused number of its series is the workbook's literal precedent for one key on two bodies (Duplicate Flags DUP-CST073-20260727, DUP-ID-HELP-BOARD-20260724, DUP-DISPATCH-DQ030-20260726) and is recorded under residual_questions as the alternative the operator may prefer; it is not adopted only so that the three proposals issue keys of one shape, and because a renumbered id would have to be propagated to every cell that cites the old number. A Drive-ID-prefix discriminator (the form of Artifact Index rows 645/646) is available for the Definition Registry (both rows carry a Source URL) and would read DEF-049@1ve_dMz for row 50; it is not adopted for the same one-shape reason.
- Explicitly not proposed: Neither definition's content is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 50 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-DEFINITION-REGISTRY-DEF049-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT DEFINITIONS |
| `File A` | Definition Registry row 48 (2026-09-18 xlsx export) — DEF-049 — 'Certified capture mass' — Status: ACTIVE |
| `File B` | Definition Registry row 50 (2026-09-18 xlsx export) — DEF-049 — 'Interval-certified degree-four corridor box' — Status: ACTIVE-CANDIDATE |
| `Similarity / hash` | Same declared Definition ID; different Source URL (https://docs.google.com/document/d/1kNp3vZxyA3kcorgwkdkL50MLwZQ7IiSayMh--aXZIiY/edit versus https://docs.google.com/document/d/1ve_dMzQd8QvkPpu55oReGwSRnQRWuNsbLcKWzYStGtI/edit); Canonical term, Exact operational definition, Scope, Explicit exclusions, Aliases / search terms, Hyper-tag, Object tags, Authority source, Authority effect, Status, Source URL differ |
| `Risk` | Citation ambiguity and wrong-object routing risk; the bare ID resolves to two different objects |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 48 as DEF-049; register row 50 additively under compound key DEF-049@DEF-R50; both objects stand with their own Status; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Definition Registry |
| `Detected at` | 2026-09-18 xlsx export; tab first keyed by tools/registers_check.py 2026-09-19; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Definition Registry, row 50, column 'Definition ID'; `DEF-049` → `DEF-049@DEF-R50`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether row 50 ('Interval-certified degree-four corridor box') is in substance the same definition as row 49 (DEF-050, 'Full-dimensional interval-certified degree-four corridor box', same Source URL and Authority source) is a content adjudication for the operator; this proposal keys the row, it does not merge or retire it.
- Whether the later row should instead be renumbered to the next unused DEF number (the export's highest plain 'DEF-<nnn>' is DEF-059, so DEF-060 is unassigned) is not decided here.
- Which of the two rows' text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- No Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct text where the rows describe one object, nor rank two objects that share one key; content adjudication is an operator matter under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party; the Independence Score and Independence Class cells quoted from the Review Independence ledger are transcribed, not evaluated.
- Does not establish that any listed relation, review or definition is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not repair the cross-tab citations of a bare colliding id (Evidence Lineage, Relation Index, Transition Alarms rows that name REV-P12-GP-006 or REV-P02-GP-INTERVAL-001); it records them and leaves them as they are.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

## Closing statement

1. **This is a proposal. It requires operator action.** Fourteen records, fourteen findings, one to one; together with
   the frozen 2026-09-19 and 2026-09-18 proposals, thirty-seven records over thirty-seven findings, none covered twice.
2. **Nothing has been repaired.** No register row was edited, merged, deleted, reordered or
   reclassified by this document. Every executable operation in it is an append to the workbook's
   append-only collision registry, and none of those appends has been performed either.
3. **The export remains faithful.** `registers/source/`, `registers/json/` and `registers/csv/` are
   byte-identical to `git HEAD`. `tools/collision_proposal_check.py` asserts this and exits nonzero
   if it ever stops being true.
4. **`independence_credit = 0`.** This session is Anthropic-family. OP-PROT-019-v1.1 R17 §4 permits a fresh nonauthor session of any provider to perform technical review, but records organizational independence separately and at ZERO for a same-provider reviewer. Independently of that, a PROPOSAL is not a review at all: it licenses no technical verdict and no independence credit on any object, for any provider.
5. **Every independence-requiring gate REMAINS OPEN**, unchanged by this document, regardless of any
   technical judgement expressed in it.
6. **No status is decided here.** The thirteen different-object records decide nothing about either
   object; the one same-relation record hands its `OP-CNS-001 §1` content-adjudication item to the
   operator unresolved.
