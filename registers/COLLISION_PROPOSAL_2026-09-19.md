# COLLISION PROPOSAL 2026-09-19 — successor: the seven defects first visible in the 2026-09-18 xlsx export

> **THIS IS A PROPOSAL. IT REQUIRES OPERATOR ACTION.**
> **Nothing has been repaired.** No register row was edited, merged, deleted, reordered or
> reclassified. The 2026-09-18 export remains faithful: `registers/source/`, `registers/json/`
> and `registers/csv/` are byte-identical to `git HEAD`, and `tools/collision_proposal_check.py`
> `--proposal registers/collision_proposal_2026-09-19.json` fails if that stops being true.

> **Status discipline.** Every claim, premise and obligation stands exactly as its source
> records it; nothing in this document moves one, and every gate remains where it was. No
> original prize problem is solved. The 2D upper/lower tracks are not composed with the 3D
> lifetime track anywhere in this document.

## What this is

`registers/KNOWN_FINDINGS.json` records, under its section `findings_first_visible_in_2026-09-18_export`, **seven structural defects in the source registers** that sit in rows the 2026-09-17 markdown export never delivered and that became visible only with the complete xlsx export of 2026-09-18: six duplicate artifact IDs in the Artifact Index (rows 206 and beyond of 761) and one duplicate Evidence ID in the Evidence Lineage (rows 137 and beyond of 485). This document proposes, for each of the seven, a remedy drawn from the protocols' own text and from the workbook's own executed precedents. It executes none of them.

It is a **numbered successor** to [`registers/COLLISION_PROPOSAL.md`](COLLISION_PROPOSAL.md) / [`registers/collision_proposal.json`](collision_proposal.json) (the 2026-09-18 proposal, 16 records over the `findings` section). Under CLAUDE.md rule 8 that document is frozen: it is not edited, superseded, amended or reissued here, and this document carries `supersedes: null`. Together the two documents cover all 23 findings in `KNOWN_FINDINGS.json`, each exactly once; `tests/test_collision_proposal.py` asserts that.

The machine-readable form is [`registers/collision_proposal_2026-09-19.json`](collision_proposal_2026-09-19.json). The checker is `tools/collision_proposal_check.py --proposal registers/collision_proposal_2026-09-19.json`; the tests are `tests/test_collision_proposal.py`.

### Source of record

| | |
|---|---|
| Export | `registers/source/GP-REG-032_v1.2_export_2026-09-18.xlsx` |
| SHA-256 | `c3229ecefc642f3e32f23cb8320fb66236d11affb83f67bd135153d068a3f460` |
| Bytes | 1,919,741 |
| Provenance | `registers/source/SOURCES.json` (exact: false — an xlsx export is a rendering of the native Sheet) |
| JSON tabs quoted | `registers/json/artifact_index.json`, `registers/json/evidence_lineage.json` (generated from the xlsx by `tools/registers_import.py`) |
| Row canonicalisation | `sha256(json.dumps(row, ensure_ascii=False, separators=(",",":")))` |
| Findings file | `registers/KNOWN_FINDINGS.json`, section `findings_first_visible_in_2026-09-18_export` (7 findings) |
| Records | 7 (one per finding key) |
| Predecessor | `registers/collision_proposal.json`, SHA-256 `6055dad5ec41a8d5eaa59ef005bb2e84622217fe177649be7c2e34c5b6a28a82`, 368,189 bytes — frozen, not edited |

Every row quoted below is the list of cell strings exactly as the named JSON tab holds it at the stated 0-based row index, with the SHA-256 of its canonical serialisation recorded in the JSON. The markdown export has no line for any of these rows and is not cited. Row indices are the 0-based indices `tools/registers_check.py` prints.

`tools/collision_proposal_check.py --proposal registers/collision_proposal_2026-09-19.json` holds this companion to the live rows, not to the JSON proposal alone: every fenced row block below must equal the canonical serialisation of the live `registers/json/<tab>.json` row, every `rows[i], canonical N bytes, SHA-256` line must match that row, every `Cell count` line and every materiality paragraph must appear verbatim as recomputed, and every summary-table row must name the record's two rows, its keeper (the earlier row), its reidentified row (the later) and its successor id. The document-level `summary_counts` and the classification lists in the JSON are likewise recomputed from the records' cells. A number that appears here and is not recomputed by the checker is a defect of the checker, not a fact.

### Reviewer standing and independence

- **Session family:** Anthropic. **Role:** PROPOSAL AUTHOR — NOT A REVIEWER.
- **`independence_credit = 0`.** This session is Anthropic-family. OP-PROT-019-v1.1 R17 §4 permits a fresh nonauthor session of any provider to perform technical review, but records organizational independence separately and at ZERO for a same-provider reviewer. Independently of that, a PROPOSAL is not a review at all: it licenses no technical verdict and no independence credit on any object, for any provider.
- **Every independence-requiring gate REMAINS OPEN.** Every independence-requiring gate in this program REMAINS OPEN, unchanged by this document, regardless of any technical judgement expressed here. Nothing in this proposal is a technical pass, and no technical pass would satisfy an independence predicate in any case (R17 §4: 'Never relabel an independence-required theorem terminal solely because its technical review passed.').
- **Authorship of the objects examined.** The Org cells of the fourteen rows examined record OpenAI-family, 'GP / OpenAI', 'OpenAI GP' or 'GP/LCR' provenance; they are quoted verbatim in each record's rows and none records the family of this session. This changes nothing: the credit recorded here is zero because a proposal licenses zero, not because of provider matching.

### What this document does NOT establish

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct status text; that is an operator content adjudication under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party.
- Does not establish that any listed artifact is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

### The collision registry this proposal writes to

The workbook's append-only collision registry is the 'Duplicate Flags' tab of GP-REG-032-v1.2 (registers/json/duplicate_flags.json), which already carries 31 collision clusters including DUP-ID-GP-DER-044, the collision OP-CNS-001 §2 names by hand.

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

- **NCC-REG032-COLLISION-20260919-001** — Records that the whole batch changes no mathematical statement, no schema, no verifier and no status. Precedent: NCC-P01-TR007-COLLISION, the certificate the workbook issued for its own TR-P01-007 collision repair; NCC-REG032-COLLISION-20260918-001, the identifier the 2026-09-18 proposal proposed for its own batch (proposed, not created).
- **GP-COR-211** — Next unused number above the exported GP-COR maximum in the complete 2026-09-18 export, GP-COR-210. The 2026-09-18 proposal's GP-COR-204 was computed against the markdown prefix, whose maximum was GP-COR-203; the complete export shows GP-COR-204 through GP-COR-209 unassigned and GP-COR-210 assigned. GP-COR-204 remains that frozen document's proposed identifier and is not reissued here. Precedent: GP-COR-190 (Help Board key collisions), GP-COR-192 (P02-LM012 multi-ID collision), cited in the Duplicate Flags rows for those repairs.
- Both are PROPOSED identifiers. Neither is created by this document. Whether the two proposals' batches should share one correction record when executed is an operator decision recorded under residual_questions_document_level.

### How the seven pairs classify

| Class | Pairs |
|---|---|
| Same Drive object, different status text | GP-DATA-168-v1.1 (rows 231 and 240); LS-AUD-002-v1.0 (rows 248 and 253); LS-COR-001-v1.0 (rows 249 and 254); LS-AUD-003-v1.0 (rows 250 and 255); LS-MAN-045-v1.0 (rows 544 and 545); EV-LS-REQ030 (rows 385 and 386) |
| Different objects under one identifier | GP-REQ-194-v1.0 (rows 325 and 373) |
| Exact duplicate rows | none |

EV-LS-REQ030 was tested for cell-for-cell identity because KNOWN_FINDINGS describes the rows as binding the same Drive ID and the same code fixture digest; the test is recorded in the record (exact_duplicate_row: false, eight of sixteen cells differ, eight agree; the counts are recomputed by the checker from the quoted rows), so the EXACT_DUPLICATE_ROW class and the VOID-DUPLICATE precedent are not used anywhere in this proposal.

### Relation to the 2026-09-18 proposal, and what differs

- `supersedes: null`; `successor_of` names the frozen predecessor with its SHA-256.
- `findings_source_section: findings_first_visible_in_2026-09-18_export`; the predecessor covers `findings`.
- `source_of_record.kind: xlsx_export_json_rows`; the predecessor quotes 1-based lines of the markdown export, which has no line for these rows.
- Successor identifiers follow the predecessor's shapes (`<id>@AIDX-R<row>` for the Artifact Index) and add the analogous `<id>@EVL-R<row>` for the Evidence Lineage; none collides with any identifier in `registers/json/` or with any successor the predecessor issued (checked mechanically).
- The correction-record identifier is `GP-COR-211`, not `GP-COR-205`: the complete export shows `GP-COR-210` assigned, which the markdown prefix did not deliver. The predecessor's `GP-COR-204` stands as that document's proposal.

### Residual questions at document level

- Whether the seven appends of this proposal and the sixteen of the 2026-09-18 proposal are executed as one batch under one no-change certificate and one correction record, or as two, is an operator decision; the identifiers proposed here are distinct so that either choice remains open.
- Whether the row-locator discriminator (@AIDX-R<n>, @EVL-R<n>) is ratified is one decision for both proposals; this document does not ratify it.
- The keeper rule identifies the exported row order with the confirmed append position. The LS-MAN-045-v1.0 record shows a region of the Artifact Index whose row order is not chronological; the operator should confirm append order from the workbook's revision history before executing any record.
- The task text that commissioned this document named registers/SOURCES.json as the digest record; the file is registers/source/SOURCES.json. The digest and byte count recorded here were re-computed from the xlsx on disk and agree with that file.

## Part 1 — Six duplicate artifact IDs in the Artifact Index (rows the markdown export never delivered)

Five of the six pairs share the property the 2026-09-18 proposal found in all of its six: **both rows cite the SAME Drive document.** Those five are duplicate *register rows* describing one artifact, and no new artifact ID is proposed for them; what is disambiguated is the register row key. The sixth pair, `GP-REQ-194-v1.0`, is the other case — **two different documents, two different titles, one identifier** — the case `OP-CNS-001 §2`'s GP-DER-044 example and acceptance test T7 describe. Both objects are preserved; the identifier is disambiguated.

| Artifact ID | Rows | Class | Keeper row | Reidentified row | Proposed successor row key |
|---|---|---|---|---|---|
| `GP-DATA-168-v1.1` | 231, 240 | same object, different status text | 231 | 240 | `GP-DATA-168-v1.1@AIDX-R240` |
| `LS-AUD-002-v1.0` | 248, 253 | same object, different status text | 248 | 253 | `LS-AUD-002-v1.0@AIDX-R253` |
| `LS-COR-001-v1.0` | 249, 254 | same object, different status text | 249 | 254 | `LS-COR-001-v1.0@AIDX-R254` |
| `LS-AUD-003-v1.0` | 250, 255 | same object, different status text | 250 | 255 | `LS-AUD-003-v1.0@AIDX-R255` |
| `GP-REQ-194-v1.0` | 325, 373 | DIFFERENT OBJECTS | 325 | 373 | `GP-REQ-194-v1.0@AIDX-R373` |
| `LS-MAN-045-v1.0` | 544, 545 | same object, different status text | 544 | 545 | `LS-MAN-045-v1.0@AIDX-R545` |

**Keeper rule, applied mechanically to all six.** Earlier confirmed append position. `R17 §3`: *"Use the confirmed append position to order contenders, not a self-reported timestamp."* This is the rule the 2026-09-18 proposal applied, reused unchanged. One pair (`LS-MAN-045-v1.0`) has its append order and its timestamp order inverted; the rule still selects by append position, and the record says why that needs operator confirmation.

**Keeping the bare key is a KEY assignment, not a currency verdict.** It does not make the keeper row's status text current, nor the reidentified row's status text stale.

### 1.1 `GP-DATA-168-v1.1` — rows 231, 240

> Finding key: `artifact_index: duplicate key 'GP-DATA-168-v1.1' at rows 231 and 240`

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__SAME_DRIVE_OBJECT__DIFFERENT_STATUS_TEXT`. Both rows cite Drive source https://docs.google.com/document/d/1BRAq7iDEf2S30avt3-5CiLLc39K8gVVn7ESn2wxwZwY/edit.

#### Row 231 — proposed KEEPER row, verbatim

`registers/json/artifact_index.json` rows[231], canonical 592 bytes, SHA-256 `d587c9ad622fb32731663cc1a438e51f40df5d225ec0b57306e0adea8b13486a`:

```json
["GP-DATA-168-v1.1","EC-019 T2 Corrected GP-Side Interval Preflight Source and Execution Receipt","GP / OpenAI","DATA / SOURCE / COMPUTATIONAL REPAIR","P0","EC-019; T2; intervals; controls; clean-room preflight","SAME-LINE REPAIR PASS / T2 OPEN","Factual execution only; canonical impact NONE; independence credit ZERO","Supersedes GP-DATA-168-v1.0 for same-line technical use; responds to GP-AUD-176","2026-07-24T19:49:45Z","https://docs.google.com/document/d/1BRAq7iDEf2S30avt3-5CiLLc39K8gVVn7ESn2wxwZwY/edit","Raw runtime .py/.json preserved separately; exact hashes recorded in receipt."]
```

#### Row 240 — proposed REIDENTIFIED row, verbatim

`registers/json/artifact_index.json` rows[240], canonical 585 bytes, SHA-256 `2ddb79d296d228bfd104e3dae916533099e15fa8705995e1b4e871cfc2ff0bec`:

```json
["GP-DATA-168-v1.1","EC-019 T2 Corrected GP-Side Interval Preflight Source and Execution Receipt","OpenAI GP","DATA / EXECUTION RECEIPT","EC-019","T2; interval repair; 19 margins; 5 controls; source publication","SAME-LINE RECEIPT PASS / SOURCE REQUIRED / T2 OPEN","No independent credit; no terminal effect","Successor to GP-DATA-168-v1.0; controlled by GP-AUD-176 addendum","2026-07-24T20:05:00Z","https://docs.google.com/document/d/1BRAq7iDEf2S30avt3-5CiLLc39K8gVVn7ESn2wxwZwY/edit","Declared 20,949-byte source is not reconstructably published in Drive; exact replay unavailable."]
```

#### Where the two rows differ

| Field | Row 231 (keeper) | Row 240 (reidentified) |
|---|---|---|
| **Org** | GP / OpenAI | OpenAI GP |
| **Class** | DATA / SOURCE / COMPUTATIONAL REPAIR | DATA / EXECUTION RECEIPT |
| **Priority** | P0 | EC-019 |
| **Topics / object tags** | EC-019; T2; intervals; controls; clean-room preflight | T2; interval repair; 19 margins; 5 controls; source publication |
| **Status** | SAME-LINE REPAIR PASS / T2 OPEN | SAME-LINE RECEIPT PASS / SOURCE REQUIRED / T2 OPEN |
| **Authority / canonical impact** | Factual execution only; canonical impact NONE; independence credit ZERO | No independent credit; no terminal effect |
| **Dependencies or supersession** | Supersedes GP-DATA-168-v1.0 for same-line technical use; responds to GP-AUD-176 | Successor to GP-DATA-168-v1.0; controlled by GP-AUD-176 addendum |
| **Modified UTC** | 2026-07-24T19:49:45Z | 2026-07-24T20:05:00Z |
| **Notes** | Raw runtime .py/.json preserved separately; exact hashes recorded in receipt. | Declared 20,949-byte source is not reconstructably published in Drive; exact replay unavailable. |

Identical in both rows: `Artifact ID`, `Title`, `Source`.

Cell count: 12 columns compared, 3 identical, 9 differing (rows 231 and 240 of `registers/json/artifact_index.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** Both rows cite one Drive document and one title, 'EC-019 T2 Corrected GP-Side Interval Preflight Source and Execution Receipt'. Row 231 classes the object 'DATA / SOURCE / COMPUTATIONAL REPAIR' with Status 'SAME-LINE REPAIR PASS / T2 OPEN' and Priority 'P0'; row 240 classes the same document 'DATA / EXECUTION RECEIPT' with Status 'SAME-LINE RECEIPT PASS / SOURCE REQUIRED / T2 OPEN' and puts 'EC-019' in the Priority column. The Notes cells contradict each other on the one point a reader would consult the row for: row 231 says 'Raw runtime .py/.json preserved separately; exact hashes recorded in receipt', row 240 says 'Declared 20,949-byte source is not reconstructably published in Drive; exact replay unavailable'. Nine of twelve cells differ. This is a material content difference under OP-CNS-001 §1, not a formatting variant.

**Corroborating register evidence.** Row 235 (GP-AUD-176-v1.0, 2026-07-24T20:00:00Z, Status 'T1 SURVIVES / GP PREFLIGHT AMEND REQUIRED / T2 OPEN') sits between the two rows and is the audit both rows name in their Dependencies cell ('responds to GP-AUD-176' / 'controlled by GP-AUD-176 addendum'). Row 240's 'SOURCE REQUIRED' and 'exact replay unavailable' postdate row 231's 'exact hashes recorded in receipt' by sixteen minutes of self-reported time. Whether that is a correction of row 231 or a second registration of a different reading is exactly the OP-CNS-001 §1 adjudication left open here.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Gaps or duplicates require an additive provenance disclosure.'
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' Noted BECAUSE IT DOES NOT APPLY: both rows cite one Drive ID, so this is a register-row duplication, not a second artifact identity. Minting a new declared artifact ID would create a second identity for one Drive object and is therefore NOT proposed.
- OP-PROT-019-v1.1 R17 §3 — 'Multiple successors of one old head are a publication conflict: hold only that target, preserve both branches, and append a reconciliation selecting or merging them with reasons. Never resolve it by silently overwriting proof bytes.' and 'Use the confirmed append position to order contenders, not a self-reported timestamp.'
- OP-PROT-019-v1.1 R17 §6 — 'No permanent deletion in this workflow.' and 'Moving or renaming must not be represented as a scientific verdict.'
- OP-PROT-012 §4 Class 1 — duplicate flagging, provenance repair and index maintenance are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** Duplicate Flags cluster DUP-REG-ARTIFACT-INDEX-GP197-20260724: 'DUPLICATE REGISTER PRIMARY KEYS / SAME DRIVE OBJECTS ... RESOLVED: first rows control; later rows reidentified VOID-DUPLICATE with unique keys; preserve additively.' Also SELFHEAL-LSMAN037-REG-001: 'Preserve earlier authoritative rows; reidentify later rows as VOID-DUPLICATE / ZERO CREDIT.' Compound row keys are the workbook's own form: Artifact Index rows 645 and 646 carry 'CL-AUD-LSDER036-20260727-01@1zMq504' and 'CL-AUD-LSDER036-20260727-01@1AcCpCe'; Duplicate Flags cluster DUP-CLWO-20260727-07-STABLEKEY-20260730 reads 'register compound keys CL-CLWO-20260727-07@1jqT3KJ and @1jGggrx; preserve both documents'. The 2026-09-18 proposal (registers/collision_proposal.json, records CP-AIDX-*) applied the same rule to six Artifact Index pairs of this class.

**Which row keeps the original id, and why**

- Keeper: **row 231** keeps `GP-DATA-168-v1.1`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp, and the in-register Artifact Index precedent (DUP-REG-ARTIFACT-INDEX-GP197-20260724, SELFHEAL-LSMAN037-REG-001) resolves same-Drive-object register-row duplicates by letting the first row control and reidentifying the later row. The rule is the one the 2026-09-18 proposal applied to its six Artifact Index pairs and is applied here mechanically so that no pair is resolved by this session's reading of its status text.
- Explicitly not implied: Keeping the bare key is a KEY assignment, not a currency verdict. It does not make the keeper row's status text current, nor the successor row's status text stale.

**Proposed successor identifier**

- Row 240 → `GP-DATA-168-v1.1@AIDX-R240`
- Naming rule: Compound row key <declared artifact ID>@<discriminator>, the form the workbook already uses for stable-key collisions (Artifact Index rows 645/646 'CL-AUD-LSDER036-20260727-01@1zMq504' and '@1AcCpCe'; Duplicate Flags clusters DUP-CLWO-20260727-07-STABLEKEY-20260730 etc.). In those instances the discriminator was a Drive ID prefix. Here both rows cite ONE Drive ID, so the Drive-ID discriminator is unavailable and the discriminator falls back to the register row locator: AIDX = Artifact Index, R<n> = the 0-based row index tools/registers_check.py prints. This is the fallback the 2026-09-18 proposal adopted for its six pairs (GP-DER-118-v1.2@AIDX-R95 etc.) and is reused unchanged so that the two proposals issue keys of one shape. THE FALLBACK IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN FORM, NOT A LITERAL QUOTATION; the abbreviation requires operator ratification, and ratifying it once ratifies it for both proposals.
- Alternative discriminator considered: The reidentified row's own Modified UTC would be fully source-derived. It is not adopted, for the reason the 2026-09-18 proposal gave (it fails on pairs whose two rows carry one Modified UTC) and for a second reason visible only in this batch: for LS-MAN-045-v1.0 the later-appended row 545 carries the EARLIER timestamp (18:15:00Z vs row 544's 18:20:00Z), so a timestamp discriminator would invert the append order R17 §3 prescribes.
- Explicitly not proposed: The artifact's DECLARED ID inside the Drive document is not changed. Only the register row key is disambiguated. The 'VOID-DUPLICATE' disposition used by DUP-REG-ARTIFACT-INDEX-GP197-20260724 and SELFHEAL-LSMAN037-REG-001 is NOT proposed here, because those clusters' rows were substantively identical ('same Drive IDs and same evidence/request roles', 'identical LS-MAN-037 export identity') whereas this pair differs materially in Status; labelling either row VOID would destroy recorded state.
- Proposed disposition label: `COLLISION-PRESERVED / CONTENT-ADJUDICATION-PENDING / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-ARTIFACT-INDEX-GPDATA168v11-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / SAME DRIVE OBJECT / DIFFERENT STATUS TEXT |
| `File A` | Artifact Index row 231 (2026-09-18 xlsx export) — GP-DATA-168-v1.1 — Status: SAME-LINE REPAIR PASS / T2 OPEN |
| `File B` | Artifact Index row 240 (2026-09-18 xlsx export) — GP-DATA-168-v1.1 — Status: SAME-LINE RECEIPT PASS / SOURCE REQUIRED / T2 OPEN |
| `Similarity / hash` | Same declared Artifact ID and same Drive source https://docs.google.com/document/d/1BRAq7iDEf2S30avt3-5CiLLc39K8gVVn7ESn2wxwZwY/edit; Org, Class, Priority, Topics / object tags, Status, Authority / canonical impact, Dependencies or supersession, Modified UTC, Notes differ |
| `Risk` | Citation ambiguity, double-count and stale-status routing risk; the bare ID does not identify one row |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 231 as GP-DATA-168-v1.1; register row 240 additively under compound key GP-DATA-168-v1.1@AIDX-R240; open an OP-CNS-001 §1 content-adjudication item on the Status difference; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Artifact Index |
| `Detected at` | 2026-09-18 xlsx export scan; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Artifact Index, row 240, column 'Artifact ID'; `GP-DATA-168-v1.1` → `GP-DATA-168-v1.1@AIDX-R240`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the 'SOURCE REQUIRED' / 'exact replay unavailable' reading of row 240 supersedes row 231's 'exact hashes recorded in receipt' is unresolved by the export alone and is not resolved here.
- The Priority cell of row 240 holds 'EC-019', a control-case label rather than a priority grade; whether that is a column-shift error in the source row is not decided here.
- Which Status text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- Neither Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct status text; that is an operator content adjudication under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party.
- Does not establish that any listed artifact is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.2 `LS-AUD-002-v1.0` — rows 248, 253

> Finding key: `artifact_index: duplicate key 'LS-AUD-002-v1.0' at rows 248 and 253`

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__SAME_DRIVE_OBJECT__DIFFERENT_STATUS_TEXT`. Both rows cite Drive source https://docs.google.com/document/d/123M_3Qsw_D3e1Slkh9r1SNUioXbM5IOwRa-IVuLOZzk/edit.

#### Row 248 — proposed KEEPER row, verbatim

`registers/json/artifact_index.json` rows[248], canonical 615 bytes, SHA-256 `577874442d06e253c62f273ddfc947fb514444059bb1246da3eeb880c8355af9`:

```json
["LS-AUD-002-v1.0","Same-Line Adversarial Review of P0.1 Domain D — Dynamics and Exact-Field Transfer","OpenAI / Lead Scientist","AUD / DYNAMICS / TRANSFER","P0","P0.1; Domain D; interval corridor; exact transfer; scaling","PASS SAME-LINE WITH SCALING WORDING CORRECTION / EXTERNAL REVIEW OPEN","Zero independence credit; canonical impact NONE","GP-DATA-178; GP-DER-045; endpoint blocks; GP-PROT-065","2026-07-24T20:22:00Z","https://docs.google.com/document/d/123M_3Qsw_D3e1Slkh9r1SNUioXbM5IOwRa-IVuLOZzk/edit","Physical Hessian error is r times scaled Jacobian error; frozen theorem uses correct H_exact=H4+rE."]
```

#### Row 253 — proposed REIDENTIFIED row, verbatim

`registers/json/artifact_index.json` rows[253], canonical 447 bytes, SHA-256 `ab54bebbe983362571626d3056d5f2f84962f179aa9be973e3c2fc92d0ab6405`:

```json
["LS-AUD-002-v1.0","Same-Line Adversarial Review of P0.1 Domain D","OpenAI","AUD","P0","P0.1; dynamics; C1 transfer; saddle cone; sink capture","SAME-LINE SURVIVES WITH CORRECTION","No canonical impact; zero independence credit","GP-DER-044; GP-DER-045; GP-PROT-065; frozen theorem","2026-07-24T20:40:00Z","https://docs.google.com/document/d/123M_3Qsw_D3e1Slkh9r1SNUioXbM5IOwRa-IVuLOZzk/edit","D1–D12 reconstruction; scaling wording correction"]
```

#### Where the two rows differ

| Field | Row 248 (keeper) | Row 253 (reidentified) |
|---|---|---|
| **Title** | Same-Line Adversarial Review of P0.1 Domain D — Dynamics and Exact-Field Transfer | Same-Line Adversarial Review of P0.1 Domain D |
| **Org** | OpenAI / Lead Scientist | OpenAI |
| **Class** | AUD / DYNAMICS / TRANSFER | AUD |
| **Topics / object tags** | P0.1; Domain D; interval corridor; exact transfer; scaling | P0.1; dynamics; C1 transfer; saddle cone; sink capture |
| **Status** | PASS SAME-LINE WITH SCALING WORDING CORRECTION / EXTERNAL REVIEW OPEN | SAME-LINE SURVIVES WITH CORRECTION |
| **Authority / canonical impact** | Zero independence credit; canonical impact NONE | No canonical impact; zero independence credit |
| **Dependencies or supersession** | GP-DATA-178; GP-DER-045; endpoint blocks; GP-PROT-065 | GP-DER-044; GP-DER-045; GP-PROT-065; frozen theorem |
| **Modified UTC** | 2026-07-24T20:22:00Z | 2026-07-24T20:40:00Z |
| **Notes** | Physical Hessian error is r times scaled Jacobian error; frozen theorem uses correct H_exact=H4+rE. | D1–D12 reconstruction; scaling wording correction |

Identical in both rows: `Artifact ID`, `Priority`, `Source`.

Cell count: 12 columns compared, 3 identical, 9 differing (rows 248 and 253 of `registers/json/artifact_index.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** Both rows cite one Drive document. Row 248 reads Status 'PASS SAME-LINE WITH SCALING WORDING CORRECTION / EXTERNAL REVIEW OPEN'; row 253 reads 'SAME-LINE SURVIVES WITH CORRECTION' and drops the 'EXTERNAL REVIEW OPEN' clause. The Dependencies cells name different objects ('GP-DATA-178; GP-DER-045; endpoint blocks; GP-PROT-065' vs 'GP-DER-044; GP-DER-045; GP-PROT-065; frozen theorem'). Only Artifact ID, Priority and Source agree. Dropping an 'EXTERNAL REVIEW OPEN' clause is a material content difference under OP-CNS-001 §1; it is not read here as closing anything.

**Corroborating register evidence.** Rows 248, 249 and 250 (LS-AUD-002-v1.0, LS-COR-001-v1.0, LS-AUD-003-v1.0; Modified UTC 20:22, 20:23, 20:24 on 2026-07-24; Org 'OpenAI / Lead Scientist') are three consecutive long-form registrations. Rows 251 through 256 (LS-REC-001-v1.0, LS-AUD-001-v1.0, LS-AUD-002-v1.0, LS-COR-001-v1.0, LS-AUD-003-v1.0, LS-DATA-002) all carry Modified UTC 2026-07-24T20:40:00Z and Org 'OpenAI': a single batch re-registration that re-entered the same three declared IDs in abbreviated form, with shortened titles and reworded Status cells. The three collisions are one event seen three times, and the same keeper rule resolves all three.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Gaps or duplicates require an additive provenance disclosure.'
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' Noted BECAUSE IT DOES NOT APPLY: both rows cite one Drive ID, so this is a register-row duplication, not a second artifact identity. Minting a new declared artifact ID would create a second identity for one Drive object and is therefore NOT proposed.
- OP-PROT-019-v1.1 R17 §3 — 'Multiple successors of one old head are a publication conflict: hold only that target, preserve both branches, and append a reconciliation selecting or merging them with reasons. Never resolve it by silently overwriting proof bytes.' and 'Use the confirmed append position to order contenders, not a self-reported timestamp.'
- OP-PROT-019-v1.1 R17 §6 — 'No permanent deletion in this workflow.' and 'Moving or renaming must not be represented as a scientific verdict.'
- OP-PROT-012 §4 Class 1 — duplicate flagging, provenance repair and index maintenance are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** Duplicate Flags cluster DUP-REG-ARTIFACT-INDEX-GP197-20260724: 'DUPLICATE REGISTER PRIMARY KEYS / SAME DRIVE OBJECTS ... RESOLVED: first rows control; later rows reidentified VOID-DUPLICATE with unique keys; preserve additively.' Also SELFHEAL-LSMAN037-REG-001: 'Preserve earlier authoritative rows; reidentify later rows as VOID-DUPLICATE / ZERO CREDIT.' Compound row keys are the workbook's own form: Artifact Index rows 645 and 646 carry 'CL-AUD-LSDER036-20260727-01@1zMq504' and 'CL-AUD-LSDER036-20260727-01@1AcCpCe'; Duplicate Flags cluster DUP-CLWO-20260727-07-STABLEKEY-20260730 reads 'register compound keys CL-CLWO-20260727-07@1jqT3KJ and @1jGggrx; preserve both documents'. The 2026-09-18 proposal (registers/collision_proposal.json, records CP-AIDX-*) applied the same rule to six Artifact Index pairs of this class.

**Which row keeps the original id, and why**

- Keeper: **row 248** keeps `LS-AUD-002-v1.0`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp, and the in-register Artifact Index precedent (DUP-REG-ARTIFACT-INDEX-GP197-20260724, SELFHEAL-LSMAN037-REG-001) resolves same-Drive-object register-row duplicates by letting the first row control and reidentifying the later row. The rule is the one the 2026-09-18 proposal applied to its six Artifact Index pairs and is applied here mechanically so that no pair is resolved by this session's reading of its status text.
- Explicitly not implied: Keeping the bare key is a KEY assignment, not a currency verdict. It does not make the keeper row's status text current, nor the successor row's status text stale.

**Proposed successor identifier**

- Row 253 → `LS-AUD-002-v1.0@AIDX-R253`
- Naming rule: Compound row key <declared artifact ID>@<discriminator>, the form the workbook already uses for stable-key collisions (Artifact Index rows 645/646 'CL-AUD-LSDER036-20260727-01@1zMq504' and '@1AcCpCe'; Duplicate Flags clusters DUP-CLWO-20260727-07-STABLEKEY-20260730 etc.). In those instances the discriminator was a Drive ID prefix. Here both rows cite ONE Drive ID, so the Drive-ID discriminator is unavailable and the discriminator falls back to the register row locator: AIDX = Artifact Index, R<n> = the 0-based row index tools/registers_check.py prints. This is the fallback the 2026-09-18 proposal adopted for its six pairs (GP-DER-118-v1.2@AIDX-R95 etc.) and is reused unchanged so that the two proposals issue keys of one shape. THE FALLBACK IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN FORM, NOT A LITERAL QUOTATION; the abbreviation requires operator ratification, and ratifying it once ratifies it for both proposals.
- Alternative discriminator considered: The reidentified row's own Modified UTC would be fully source-derived. It is not adopted, for the reason the 2026-09-18 proposal gave (it fails on pairs whose two rows carry one Modified UTC) and for a second reason visible only in this batch: for LS-MAN-045-v1.0 the later-appended row 545 carries the EARLIER timestamp (18:15:00Z vs row 544's 18:20:00Z), so a timestamp discriminator would invert the append order R17 §3 prescribes.
- Explicitly not proposed: The artifact's DECLARED ID inside the Drive document is not changed. Only the register row key is disambiguated. The 'VOID-DUPLICATE' disposition used by DUP-REG-ARTIFACT-INDEX-GP197-20260724 and SELFHEAL-LSMAN037-REG-001 is NOT proposed here, because those clusters' rows were substantively identical ('same Drive IDs and same evidence/request roles', 'identical LS-MAN-037 export identity') whereas this pair differs materially in Status; labelling either row VOID would destroy recorded state.
- Proposed disposition label: `COLLISION-PRESERVED / CONTENT-ADJUDICATION-PENDING / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-ARTIFACT-INDEX-LSAUD002v10-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / SAME DRIVE OBJECT / DIFFERENT STATUS TEXT |
| `File A` | Artifact Index row 248 (2026-09-18 xlsx export) — LS-AUD-002-v1.0 — Status: PASS SAME-LINE WITH SCALING WORDING CORRECTION / EXTERNAL REVIEW OPEN |
| `File B` | Artifact Index row 253 (2026-09-18 xlsx export) — LS-AUD-002-v1.0 — Status: SAME-LINE SURVIVES WITH CORRECTION |
| `Similarity / hash` | Same declared Artifact ID and same Drive source https://docs.google.com/document/d/123M_3Qsw_D3e1Slkh9r1SNUioXbM5IOwRa-IVuLOZzk/edit; Title, Org, Class, Topics / object tags, Status, Authority / canonical impact, Dependencies or supersession, Modified UTC, Notes differ |
| `Risk` | Citation ambiguity, double-count and stale-status routing risk; the bare ID does not identify one row |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 248 as LS-AUD-002-v1.0; register row 253 additively under compound key LS-AUD-002-v1.0@AIDX-R253; open an OP-CNS-001 §1 content-adjudication item on the Status difference; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Artifact Index |
| `Detected at` | 2026-09-18 xlsx export scan; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Artifact Index, row 253, column 'Artifact ID'; `LS-AUD-002-v1.0` → `LS-AUD-002-v1.0@AIDX-R253`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether row 253's omission of 'EXTERNAL REVIEW OPEN' was an abbreviation or a change of state is unresolved by the export alone and is not resolved here; nothing in this document treats the external review as anything but open.
- Which Status text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- Neither Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct status text; that is an operator content adjudication under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party.
- Does not establish that any listed artifact is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.3 `LS-COR-001-v1.0` — rows 249, 254

> Finding key: `artifact_index: duplicate key 'LS-COR-001-v1.0' at rows 249 and 254`

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__SAME_DRIVE_OBJECT__DIFFERENT_STATUS_TEXT`. Both rows cite Drive source https://docs.google.com/document/d/1XVnd9Hd7Ds1SO5_NH_44OSeLG9ZBxM3Ahiq9h7JAY-k/edit.

#### Row 249 — proposed KEEPER row, verbatim

`registers/json/artifact_index.json` rows[249], canonical 530 bytes, SHA-256 `a6bba99be40b0cab4997f84f73dbd5929414996e4cc55b9456f8e67a142c0893`:

```json
["LS-COR-001-v1.0","Repair of the GP-DER-046 GT5 Expected-Supremum Step","OpenAI / Lead Scientist","COR / GAUSSIAN PROCESS PROOF REPAIR","P0","P0.1; Domain T; GT5; Dudley; Borell-TIS","MATERIAL LOCAL ERROR REPAIRED / GT5 SURVIVES SAME-LINE / EXTERNAL REVIEW OPEN","Zero independence credit; canonical impact NONE","Corrects GP-DER-046 Section 7","2026-07-24T20:23:00Z","https://docs.google.com/document/d/1XVnd9Hd7Ds1SO5_NH_44OSeLG9ZBxM3Ahiq9h7JAY-k/edit","Replaces false sup|Z|<=supZ+sup(-Z) with signed-index Gaussian process."]
```

#### Row 254 — proposed REIDENTIFIED row, verbatim

`registers/json/artifact_index.json` rows[254], canonical 388 bytes, SHA-256 `96e6352ce1e184ba54570a410b94b80316d12a6c61629cae9040d67eaa447c2c`:

```json
["LS-COR-001-v1.0","Repair of GP-DER-046 GT5 Expected-Supremum Step","OpenAI","COR","P0","GT5; Dudley; Borell-TIS; signed Gaussian process","MATERIAL LOCAL PROOF ERROR REPAIRED","No canonical impact; correction attachment","GP-DER-046-v1.0","2026-07-24T20:40:00Z","https://docs.google.com/document/d/1XVnd9Hd7Ds1SO5_NH_44OSeLG9ZBxM3Ahiq9h7JAY-k/edit","Replaces false supremum inequality"]
```

#### Where the two rows differ

| Field | Row 249 (keeper) | Row 254 (reidentified) |
|---|---|---|
| **Title** | Repair of the GP-DER-046 GT5 Expected-Supremum Step | Repair of GP-DER-046 GT5 Expected-Supremum Step |
| **Org** | OpenAI / Lead Scientist | OpenAI |
| **Class** | COR / GAUSSIAN PROCESS PROOF REPAIR | COR |
| **Topics / object tags** | P0.1; Domain T; GT5; Dudley; Borell-TIS | GT5; Dudley; Borell-TIS; signed Gaussian process |
| **Status** | MATERIAL LOCAL ERROR REPAIRED / GT5 SURVIVES SAME-LINE / EXTERNAL REVIEW OPEN | MATERIAL LOCAL PROOF ERROR REPAIRED |
| **Authority / canonical impact** | Zero independence credit; canonical impact NONE | No canonical impact; correction attachment |
| **Dependencies or supersession** | Corrects GP-DER-046 Section 7 | GP-DER-046-v1.0 |
| **Modified UTC** | 2026-07-24T20:23:00Z | 2026-07-24T20:40:00Z |
| **Notes** | Replaces false sup\|Z\|<=supZ+sup(-Z) with signed-index Gaussian process. | Replaces false supremum inequality |

Identical in both rows: `Artifact ID`, `Priority`, `Source`.

Cell count: 12 columns compared, 3 identical, 9 differing (rows 249 and 254 of `registers/json/artifact_index.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** Both rows cite one Drive document. Row 249 reads Status 'MATERIAL LOCAL ERROR REPAIRED / GT5 SURVIVES SAME-LINE / EXTERNAL REVIEW OPEN'; row 254 reads 'MATERIAL LOCAL PROOF ERROR REPAIRED' and drops both the 'GT5 SURVIVES SAME-LINE' and the 'EXTERNAL REVIEW OPEN' clauses. Row 249's Notes state the mathematical content of the repair ('Replaces false sup|Z|<=supZ+sup(-Z) with signed-index Gaussian process.'); row 254's Notes abbreviate it to 'Replaces false supremum inequality'. Only Artifact ID, Priority and Source agree. This is a material content difference under OP-CNS-001 §1.

**Corroborating register evidence.** Rows 248, 249 and 250 (LS-AUD-002-v1.0, LS-COR-001-v1.0, LS-AUD-003-v1.0; Modified UTC 20:22, 20:23, 20:24 on 2026-07-24; Org 'OpenAI / Lead Scientist') are three consecutive long-form registrations. Rows 251 through 256 (LS-REC-001-v1.0, LS-AUD-001-v1.0, LS-AUD-002-v1.0, LS-COR-001-v1.0, LS-AUD-003-v1.0, LS-DATA-002) all carry Modified UTC 2026-07-24T20:40:00Z and Org 'OpenAI': a single batch re-registration that re-entered the same three declared IDs in abbreviated form, with shortened titles and reworded Status cells. The three collisions are one event seen three times, and the same keeper rule resolves all three.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Gaps or duplicates require an additive provenance disclosure.'
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' Noted BECAUSE IT DOES NOT APPLY: both rows cite one Drive ID, so this is a register-row duplication, not a second artifact identity. Minting a new declared artifact ID would create a second identity for one Drive object and is therefore NOT proposed.
- OP-PROT-019-v1.1 R17 §3 — 'Multiple successors of one old head are a publication conflict: hold only that target, preserve both branches, and append a reconciliation selecting or merging them with reasons. Never resolve it by silently overwriting proof bytes.' and 'Use the confirmed append position to order contenders, not a self-reported timestamp.'
- OP-PROT-019-v1.1 R17 §6 — 'No permanent deletion in this workflow.' and 'Moving or renaming must not be represented as a scientific verdict.'
- OP-PROT-012 §4 Class 1 — duplicate flagging, provenance repair and index maintenance are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** Duplicate Flags cluster DUP-REG-ARTIFACT-INDEX-GP197-20260724: 'DUPLICATE REGISTER PRIMARY KEYS / SAME DRIVE OBJECTS ... RESOLVED: first rows control; later rows reidentified VOID-DUPLICATE with unique keys; preserve additively.' Also SELFHEAL-LSMAN037-REG-001: 'Preserve earlier authoritative rows; reidentify later rows as VOID-DUPLICATE / ZERO CREDIT.' Compound row keys are the workbook's own form: Artifact Index rows 645 and 646 carry 'CL-AUD-LSDER036-20260727-01@1zMq504' and 'CL-AUD-LSDER036-20260727-01@1AcCpCe'; Duplicate Flags cluster DUP-CLWO-20260727-07-STABLEKEY-20260730 reads 'register compound keys CL-CLWO-20260727-07@1jqT3KJ and @1jGggrx; preserve both documents'. The 2026-09-18 proposal (registers/collision_proposal.json, records CP-AIDX-*) applied the same rule to six Artifact Index pairs of this class.

**Which row keeps the original id, and why**

- Keeper: **row 249** keeps `LS-COR-001-v1.0`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp, and the in-register Artifact Index precedent (DUP-REG-ARTIFACT-INDEX-GP197-20260724, SELFHEAL-LSMAN037-REG-001) resolves same-Drive-object register-row duplicates by letting the first row control and reidentifying the later row. The rule is the one the 2026-09-18 proposal applied to its six Artifact Index pairs and is applied here mechanically so that no pair is resolved by this session's reading of its status text.
- Explicitly not implied: Keeping the bare key is a KEY assignment, not a currency verdict. It does not make the keeper row's status text current, nor the successor row's status text stale.

**Proposed successor identifier**

- Row 254 → `LS-COR-001-v1.0@AIDX-R254`
- Naming rule: Compound row key <declared artifact ID>@<discriminator>, the form the workbook already uses for stable-key collisions (Artifact Index rows 645/646 'CL-AUD-LSDER036-20260727-01@1zMq504' and '@1AcCpCe'; Duplicate Flags clusters DUP-CLWO-20260727-07-STABLEKEY-20260730 etc.). In those instances the discriminator was a Drive ID prefix. Here both rows cite ONE Drive ID, so the Drive-ID discriminator is unavailable and the discriminator falls back to the register row locator: AIDX = Artifact Index, R<n> = the 0-based row index tools/registers_check.py prints. This is the fallback the 2026-09-18 proposal adopted for its six pairs (GP-DER-118-v1.2@AIDX-R95 etc.) and is reused unchanged so that the two proposals issue keys of one shape. THE FALLBACK IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN FORM, NOT A LITERAL QUOTATION; the abbreviation requires operator ratification, and ratifying it once ratifies it for both proposals.
- Alternative discriminator considered: The reidentified row's own Modified UTC would be fully source-derived. It is not adopted, for the reason the 2026-09-18 proposal gave (it fails on pairs whose two rows carry one Modified UTC) and for a second reason visible only in this batch: for LS-MAN-045-v1.0 the later-appended row 545 carries the EARLIER timestamp (18:15:00Z vs row 544's 18:20:00Z), so a timestamp discriminator would invert the append order R17 §3 prescribes.
- Explicitly not proposed: The artifact's DECLARED ID inside the Drive document is not changed. Only the register row key is disambiguated. The 'VOID-DUPLICATE' disposition used by DUP-REG-ARTIFACT-INDEX-GP197-20260724 and SELFHEAL-LSMAN037-REG-001 is NOT proposed here, because those clusters' rows were substantively identical ('same Drive IDs and same evidence/request roles', 'identical LS-MAN-037 export identity') whereas this pair differs materially in Status; labelling either row VOID would destroy recorded state.
- Proposed disposition label: `COLLISION-PRESERVED / CONTENT-ADJUDICATION-PENDING / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-ARTIFACT-INDEX-LSCOR001v10-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / SAME DRIVE OBJECT / DIFFERENT STATUS TEXT |
| `File A` | Artifact Index row 249 (2026-09-18 xlsx export) — LS-COR-001-v1.0 — Status: MATERIAL LOCAL ERROR REPAIRED / GT5 SURVIVES SAME-LINE / EXTERNAL REVIEW OPEN |
| `File B` | Artifact Index row 254 (2026-09-18 xlsx export) — LS-COR-001-v1.0 — Status: MATERIAL LOCAL PROOF ERROR REPAIRED |
| `Similarity / hash` | Same declared Artifact ID and same Drive source https://docs.google.com/document/d/1XVnd9Hd7Ds1SO5_NH_44OSeLG9ZBxM3Ahiq9h7JAY-k/edit; Title, Org, Class, Topics / object tags, Status, Authority / canonical impact, Dependencies or supersession, Modified UTC, Notes differ |
| `Risk` | Citation ambiguity, double-count and stale-status routing risk; the bare ID does not identify one row |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 249 as LS-COR-001-v1.0; register row 254 additively under compound key LS-COR-001-v1.0@AIDX-R254; open an OP-CNS-001 §1 content-adjudication item on the Status difference; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Artifact Index |
| `Detected at` | 2026-09-18 xlsx export scan; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Artifact Index, row 254, column 'Artifact ID'; `LS-COR-001-v1.0` → `LS-COR-001-v1.0@AIDX-R254`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether row 254's omission of 'EXTERNAL REVIEW OPEN' was an abbreviation or a change of state is unresolved by the export alone and is not resolved here; nothing in this document treats the external review as anything but open.
- The correctness of the repair the row describes is not examined here; both rows are register entries about a document that was not downloaded for this proposal.
- Which Status text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- Neither Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct status text; that is an operator content adjudication under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party.
- Does not establish that any listed artifact is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.4 `LS-AUD-003-v1.0` — rows 250, 255

> Finding key: `artifact_index: duplicate key 'LS-AUD-003-v1.0' at rows 250 and 255`

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__SAME_DRIVE_OBJECT__DIFFERENT_STATUS_TEXT`. Both rows cite Drive source https://docs.google.com/document/d/1X9eNFpg7Or3HTLqKSIuslVd6vw3yLSw_xqo-BHdohGs/edit.

#### Row 250 — proposed KEEPER row, verbatim

`registers/json/artifact_index.json` rows[250], canonical 565 bytes, SHA-256 `157aa1c1300c1424b059fd6daa7e213f00e126e2fa4b27a877a7de62a7f00159`:

```json
["LS-AUD-003-v1.0","Same-Line Adversarial Review of P0.1 Domain T — GT5 and Probability Synthesis","OpenAI / Lead Scientist","AUD / PROBABILITY SYNTHESIS","P0","P0.1; Domain T; exact-law tail; joint event; Palm quotient","PASS AFTER REPAIR SAME-LINE / EXTERNAL REVIEW OPEN","Zero independence credit; canonical impact NONE","GP-DER-046; LS-COR-001; accepted G/D interfaces","2026-07-24T20:24:00Z","https://docs.google.com/document/d/1X9eNFpg7Or3HTLqKSIuslVd6vw3yLSw_xqo-BHdohGs/edit","Original written Domain T fails locally; repaired GT5 and synthesis survive."]
```

#### Row 255 — proposed REIDENTIFIED row, verbatim

`registers/json/artifact_index.json` rows[255], canonical 382 bytes, SHA-256 `d4b83cf85aacd39ed3cbb2496c46deeda677fb26d28425a65b223a7b846c98e8`:

```json
["LS-AUD-003-v1.0","Same-Line Adversarial Review of P0.1 Domain T","OpenAI","AUD","P0","P0.1; GT5; probability synthesis; Palm quotient","PASS AFTER REPAIR ON SAME LINE","No canonical impact; zero independence credit","LS-COR-001; frozen theorem","2026-07-24T20:40:00Z","https://docs.google.com/document/d/1X9eNFpg7Or3HTLqKSIuslVd6vw3yLSw_xqo-BHdohGs/edit","T1–T9 reconstruction"]
```

#### Where the two rows differ

| Field | Row 250 (keeper) | Row 255 (reidentified) |
|---|---|---|
| **Title** | Same-Line Adversarial Review of P0.1 Domain T — GT5 and Probability Synthesis | Same-Line Adversarial Review of P0.1 Domain T |
| **Org** | OpenAI / Lead Scientist | OpenAI |
| **Class** | AUD / PROBABILITY SYNTHESIS | AUD |
| **Topics / object tags** | P0.1; Domain T; exact-law tail; joint event; Palm quotient | P0.1; GT5; probability synthesis; Palm quotient |
| **Status** | PASS AFTER REPAIR SAME-LINE / EXTERNAL REVIEW OPEN | PASS AFTER REPAIR ON SAME LINE |
| **Authority / canonical impact** | Zero independence credit; canonical impact NONE | No canonical impact; zero independence credit |
| **Dependencies or supersession** | GP-DER-046; LS-COR-001; accepted G/D interfaces | LS-COR-001; frozen theorem |
| **Modified UTC** | 2026-07-24T20:24:00Z | 2026-07-24T20:40:00Z |
| **Notes** | Original written Domain T fails locally; repaired GT5 and synthesis survive. | T1–T9 reconstruction |

Identical in both rows: `Artifact ID`, `Priority`, `Source`.

Cell count: 12 columns compared, 3 identical, 9 differing (rows 250 and 255 of `registers/json/artifact_index.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** Both rows cite one Drive document. Row 250 reads Status 'PASS AFTER REPAIR SAME-LINE / EXTERNAL REVIEW OPEN'; row 255 reads 'PASS AFTER REPAIR ON SAME LINE' and drops the 'EXTERNAL REVIEW OPEN' clause. The Dependencies cells differ ('GP-DER-046; LS-COR-001; accepted G/D interfaces' vs 'LS-COR-001; frozen theorem'). Only Artifact ID, Priority and Source agree. This is a material content difference under OP-CNS-001 §1.

**Corroborating register evidence.** Rows 248, 249 and 250 (LS-AUD-002-v1.0, LS-COR-001-v1.0, LS-AUD-003-v1.0; Modified UTC 20:22, 20:23, 20:24 on 2026-07-24; Org 'OpenAI / Lead Scientist') are three consecutive long-form registrations. Rows 251 through 256 (LS-REC-001-v1.0, LS-AUD-001-v1.0, LS-AUD-002-v1.0, LS-COR-001-v1.0, LS-AUD-003-v1.0, LS-DATA-002) all carry Modified UTC 2026-07-24T20:40:00Z and Org 'OpenAI': a single batch re-registration that re-entered the same three declared IDs in abbreviated form, with shortened titles and reworded Status cells. The three collisions are one event seen three times, and the same keeper rule resolves all three.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Gaps or duplicates require an additive provenance disclosure.'
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' Noted BECAUSE IT DOES NOT APPLY: both rows cite one Drive ID, so this is a register-row duplication, not a second artifact identity. Minting a new declared artifact ID would create a second identity for one Drive object and is therefore NOT proposed.
- OP-PROT-019-v1.1 R17 §3 — 'Multiple successors of one old head are a publication conflict: hold only that target, preserve both branches, and append a reconciliation selecting or merging them with reasons. Never resolve it by silently overwriting proof bytes.' and 'Use the confirmed append position to order contenders, not a self-reported timestamp.'
- OP-PROT-019-v1.1 R17 §6 — 'No permanent deletion in this workflow.' and 'Moving or renaming must not be represented as a scientific verdict.'
- OP-PROT-012 §4 Class 1 — duplicate flagging, provenance repair and index maintenance are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** Duplicate Flags cluster DUP-REG-ARTIFACT-INDEX-GP197-20260724: 'DUPLICATE REGISTER PRIMARY KEYS / SAME DRIVE OBJECTS ... RESOLVED: first rows control; later rows reidentified VOID-DUPLICATE with unique keys; preserve additively.' Also SELFHEAL-LSMAN037-REG-001: 'Preserve earlier authoritative rows; reidentify later rows as VOID-DUPLICATE / ZERO CREDIT.' Compound row keys are the workbook's own form: Artifact Index rows 645 and 646 carry 'CL-AUD-LSDER036-20260727-01@1zMq504' and 'CL-AUD-LSDER036-20260727-01@1AcCpCe'; Duplicate Flags cluster DUP-CLWO-20260727-07-STABLEKEY-20260730 reads 'register compound keys CL-CLWO-20260727-07@1jqT3KJ and @1jGggrx; preserve both documents'. The 2026-09-18 proposal (registers/collision_proposal.json, records CP-AIDX-*) applied the same rule to six Artifact Index pairs of this class.

**Which row keeps the original id, and why**

- Keeper: **row 250** keeps `LS-AUD-003-v1.0`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp, and the in-register Artifact Index precedent (DUP-REG-ARTIFACT-INDEX-GP197-20260724, SELFHEAL-LSMAN037-REG-001) resolves same-Drive-object register-row duplicates by letting the first row control and reidentifying the later row. The rule is the one the 2026-09-18 proposal applied to its six Artifact Index pairs and is applied here mechanically so that no pair is resolved by this session's reading of its status text.
- Explicitly not implied: Keeping the bare key is a KEY assignment, not a currency verdict. It does not make the keeper row's status text current, nor the successor row's status text stale.

**Proposed successor identifier**

- Row 255 → `LS-AUD-003-v1.0@AIDX-R255`
- Naming rule: Compound row key <declared artifact ID>@<discriminator>, the form the workbook already uses for stable-key collisions (Artifact Index rows 645/646 'CL-AUD-LSDER036-20260727-01@1zMq504' and '@1AcCpCe'; Duplicate Flags clusters DUP-CLWO-20260727-07-STABLEKEY-20260730 etc.). In those instances the discriminator was a Drive ID prefix. Here both rows cite ONE Drive ID, so the Drive-ID discriminator is unavailable and the discriminator falls back to the register row locator: AIDX = Artifact Index, R<n> = the 0-based row index tools/registers_check.py prints. This is the fallback the 2026-09-18 proposal adopted for its six pairs (GP-DER-118-v1.2@AIDX-R95 etc.) and is reused unchanged so that the two proposals issue keys of one shape. THE FALLBACK IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN FORM, NOT A LITERAL QUOTATION; the abbreviation requires operator ratification, and ratifying it once ratifies it for both proposals.
- Alternative discriminator considered: The reidentified row's own Modified UTC would be fully source-derived. It is not adopted, for the reason the 2026-09-18 proposal gave (it fails on pairs whose two rows carry one Modified UTC) and for a second reason visible only in this batch: for LS-MAN-045-v1.0 the later-appended row 545 carries the EARLIER timestamp (18:15:00Z vs row 544's 18:20:00Z), so a timestamp discriminator would invert the append order R17 §3 prescribes.
- Explicitly not proposed: The artifact's DECLARED ID inside the Drive document is not changed. Only the register row key is disambiguated. The 'VOID-DUPLICATE' disposition used by DUP-REG-ARTIFACT-INDEX-GP197-20260724 and SELFHEAL-LSMAN037-REG-001 is NOT proposed here, because those clusters' rows were substantively identical ('same Drive IDs and same evidence/request roles', 'identical LS-MAN-037 export identity') whereas this pair differs materially in Status; labelling either row VOID would destroy recorded state.
- Proposed disposition label: `COLLISION-PRESERVED / CONTENT-ADJUDICATION-PENDING / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-ARTIFACT-INDEX-LSAUD003v10-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / SAME DRIVE OBJECT / DIFFERENT STATUS TEXT |
| `File A` | Artifact Index row 250 (2026-09-18 xlsx export) — LS-AUD-003-v1.0 — Status: PASS AFTER REPAIR SAME-LINE / EXTERNAL REVIEW OPEN |
| `File B` | Artifact Index row 255 (2026-09-18 xlsx export) — LS-AUD-003-v1.0 — Status: PASS AFTER REPAIR ON SAME LINE |
| `Similarity / hash` | Same declared Artifact ID and same Drive source https://docs.google.com/document/d/1X9eNFpg7Or3HTLqKSIuslVd6vw3yLSw_xqo-BHdohGs/edit; Title, Org, Class, Topics / object tags, Status, Authority / canonical impact, Dependencies or supersession, Modified UTC, Notes differ |
| `Risk` | Citation ambiguity, double-count and stale-status routing risk; the bare ID does not identify one row |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 250 as LS-AUD-003-v1.0; register row 255 additively under compound key LS-AUD-003-v1.0@AIDX-R255; open an OP-CNS-001 §1 content-adjudication item on the Status difference; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Artifact Index |
| `Detected at` | 2026-09-18 xlsx export scan; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Artifact Index, row 255, column 'Artifact ID'; `LS-AUD-003-v1.0` → `LS-AUD-003-v1.0@AIDX-R255`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether row 255's omission of 'EXTERNAL REVIEW OPEN' was an abbreviation or a change of state is unresolved by the export alone and is not resolved here; nothing in this document treats the external review as anything but open.
- Which Status text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- Neither Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct status text; that is an operator content adjudication under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party.
- Does not establish that any listed artifact is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.5 `GP-REQ-194-v1.0` — rows 325, 373

> Finding key: `artifact_index: duplicate key 'GP-REQ-194-v1.0' at rows 325 and 373`

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__DIFFERENT_OBJECTS`. The rows cite DIFFERENT Drive sources: row 325 → https://docs.google.com/document/d/1mM355KwRN4ZiYCB9FHftl5gdiIT1BuYoQQhGMQKgynQ/edit; row 373 → https://docs.google.com/document/d/139NzUfVXV5-rW-pn_vIu6xFrqnwDcsHYPbWIn7FqOPs/edit.

#### Row 325 — proposed KEEPER row, verbatim

`registers/json/artifact_index.json` rows[325], canonical 529 bytes, SHA-256 `9b7aec57ef6bc2089d069e0ada188339e2337f33d990f9bfb5154ec8d470ed59`:

```json
["GP-REQ-194-v1.0","Joint Independent Review of Shared Taylor Transfer — P0.1-D and P02-LM-004","GP / OpenAI","REQ","P0","P0.1-D; P02-LM-004; shared Taylor transfer; separate application verdicts","OPEN ASSIGNMENT / NO REVIEW CREDIT UNTIL EXECUTED","Review routing only; no canonical impact","GP-COR-194; LCR-REQ-007; LCR-REQ-039","2026-07-24T23:55:00Z","https://docs.google.com/document/d/1mM355KwRN4ZiYCB9FHftl5gdiIT1BuYoQQhGMQKgynQ/edit","One shared interface reconstruction; separate fixed-q/adaptive application verdicts"]
```

#### Row 373 — proposed REIDENTIFIED row, verbatim

`registers/json/artifact_index.json` rows[373], canonical 608 bytes, SHA-256 `2e85e4021ddaabf63e3854570ba38cb33e6236f9959254bee9624a0cd5130172`:

```json
["GP-REQ-194-v1.0","Independent Source Review and Clean Lean Build of GP-FOR-192","GP/LCR","REQ","P0.2","Lean; Mathlib; clean build; independent source review","OPEN ASSIGNMENT / RAW ZIP PRESENT / CLEAN BUILD REQUIRED","Review routing only; canonical impact NONE","GP-FOR-192-v1.0; GP-REC-193-v1.0; raw ZIP Drive ID 1hTeWNKxLcmXEB2i5enAdmzXFknUxwTL6","2026-07-25","https://docs.google.com/document/d/139NzUfVXV5-rW-pn_vIu6xFrqnwDcsHYPbWIn7FqOPs/edit","R1 directly executable from Drive; remaining R2-R12 include manifest, toolchain freeze, clean lake build, statement correspondence, and promotion firewall"]
```

#### Where the two rows differ

| Field | Row 325 (keeper) | Row 373 (reidentified) |
|---|---|---|
| **Title** | Joint Independent Review of Shared Taylor Transfer — P0.1-D and P02-LM-004 | Independent Source Review and Clean Lean Build of GP-FOR-192 |
| **Org** | GP / OpenAI | GP/LCR |
| **Priority** | P0 | P0.2 |
| **Topics / object tags** | P0.1-D; P02-LM-004; shared Taylor transfer; separate application verdicts | Lean; Mathlib; clean build; independent source review |
| **Status** | OPEN ASSIGNMENT / NO REVIEW CREDIT UNTIL EXECUTED | OPEN ASSIGNMENT / RAW ZIP PRESENT / CLEAN BUILD REQUIRED |
| **Authority / canonical impact** | Review routing only; no canonical impact | Review routing only; canonical impact NONE |
| **Dependencies or supersession** | GP-COR-194; LCR-REQ-007; LCR-REQ-039 | GP-FOR-192-v1.0; GP-REC-193-v1.0; raw ZIP Drive ID 1hTeWNKxLcmXEB2i5enAdmzXFknUxwTL6 |
| **Modified UTC** | 2026-07-24T23:55:00Z | 2026-07-25 |
| **Source** | https://docs.google.com/document/d/1mM355KwRN4ZiYCB9FHftl5gdiIT1BuYoQQhGMQKgynQ/edit | https://docs.google.com/document/d/139NzUfVXV5-rW-pn_vIu6xFrqnwDcsHYPbWIn7FqOPs/edit |
| **Notes** | One shared interface reconstruction; separate fixed-q/adaptive application verdicts | R1 directly executable from Drive; remaining R2-R12 include manifest, toolchain freeze, clean lake build, statement correspondence, and promotion firewall |

Identical in both rows: `Artifact ID`, `Class`.

Cell count: 12 columns compared, 2 identical, 10 differing (rows 325 and 373 of `registers/json/artifact_index.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows describe two different review requests. Row 325 is 'Joint Independent Review of Shared Taylor Transfer — P0.1-D and P02-LM-004' (Org 'GP / OpenAI', Priority 'P0', Modified UTC 2026-07-24T23:55:00Z, Source document 1mM355KwRN4ZiYCB9FHftl5gdiIT1BuYoQQhGMQKgynQ); row 373 is 'Independent Source Review and Clean Lean Build of GP-FOR-192' (Org 'GP/LCR', Priority 'P0.2', Modified UTC '2026-07-25', Source document 139NzUfVXV5-rW-pn_vIu6xFrqnwDcsHYPbWIn7FqOPs). Only Artifact ID and Class ('REQ') agree; the Title, Org, Priority, Topics, Status, Authority, Dependencies, Modified UTC, Source and Notes cells all differ. This is not two readings of one object but one identifier assigned twice: the GP-DER-044 case OP-CNS-001 §2 names by hand, and acceptance test T7's case. There is no status text to adjudicate between the rows; each row's status is its own object's.

**Corroborating register evidence.** Row 324 is GP-COR-194-v1.0 at the same Modified UTC as row 325 (2026-07-24T23:55:00Z) and is named in row 325's Dependencies ('GP-COR-194; LCR-REQ-007; LCR-REQ-039'): the number 194 was taken by the joint-review request in the same registration event as its correction record. Row 372 is GP-REC-193-v1.0 ('2026-07-25') and row 374 is GP-AUD-195-v1.0 ('2026-07-25'); row 373's Dependencies name 'GP-FOR-192-v1.0; GP-REC-193-v1.0'. The Lean-build request therefore took 194 as the next number after 193 in its own sequence, without seeing that 194 had been issued the evening before. OP-CNS-001 §2: 'Failed create/upload attempts should be verified absent before retry' and 'Concurrent work should disclose line/session identity.'

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical.' / 'The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' APPLIES DIRECTLY: the two rows cite two different Drive documents with two different titles. Both objects are preserved; neither is a duplicate of the other.
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.' Applies to the identifier only: the two objects are not to be adjudicated against each other, since they are not two readings of one thing.
- OP-PROT-019-v1.1 R17 §3 — 'Use the confirmed append position to order contenders, not a self-reported timestamp.' and 'Related topic does not imply duplicate object.'
- OP-PROT-019-v1.1 R17 §6 — 'Same titles, similar topics, different-format mirrors, and copied dependencies inside self-contained reproducibility packages are not sufficient grounds for removal.' and 'No permanent deletion in this workflow.'
- OP-PROT-012 §4 Class 1 — duplicate flagging, provenance repair and index maintenance are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** Duplicate Flags cluster DUP-CLWO-20260727-07-STABLEKEY-20260730: 'EXACT WORK-ORDER ID COLLISION ... Different operations; shared base ID CL-CLWO-20260727-07 ... RESOLVED: register compound keys CL-CLWO-20260727-07@1jqT3KJ and @1jGggrx; preserve both documents' (and the sibling clusters DUP-CLWO-20260727-03-STABLEKEY-20260730 and DUP-CLWO-20260727-05-STABLEKEY-20260730). Artifact Index rows 645 and 646 carry the executed form 'CL-AUD-LSDER036-20260727-01@1zMq504' and '@1AcCpCe'. Also Duplicate Flags cluster DUP-ID-GP-DER-044, the different-Drive-object collision OP-CNS-001 §2 names by hand. NOTE: in every one of those precedents BOTH colliding rows received a compound key; none kept the bare id. This proposal keeps the bare key on the earlier append position to stay uniform with the first proposal's keeper rule, and records the both-rows-re-keyed alternative as a residual question.

**Which row keeps the original id, and why**

- Keeper: **row 325** keeps `GP-REQ-194-v1.0`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp. The rule is the same one applied to every same-object pair in both proposals, so that the keeper of the bare key is never chosen by this session's reading of which request matters more. Unlike the same-object records, keeping the bare key here does not choose between two readings of one thing: both objects are preserved, each with its own Status, and neither is a duplicate of the other.
- Explicitly not implied: Keeping the bare key is a KEY assignment. It does not rank the two review requests, does not execute either, and grants no review credit to either.

**Proposed successor identifier**

- Row 373 → `GP-REQ-194-v1.0@AIDX-R373`
- Naming rule: Compound row key <declared artifact ID>@AIDX-R<n>, the same shape the 2026-09-18 proposal and the same-object records of this proposal issue, so that every successor key in both documents is read the same way. AIDX = Artifact Index, R<n> = the 0-based row index tools/registers_check.py prints. THIS IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN FORM, NOT A LITERAL QUOTATION; it requires operator ratification.
- Alternative discriminator considered: The workbook's literal form for a different-Drive-object collision is the Drive-ID-prefix discriminator, and here it IS available: GP-REQ-194-v1.0@139NzUf for row 373 (Drive ID 139NzUfVXV5-rW-pn_vIu6xFrqnwDcsHYPbWIn7FqOPs), and in the precedents BOTH rows were re-keyed (GP-REQ-194-v1.0@1mM355K for row 325). That form is more faithful to the precedent than the row locator and is recorded here as the alternative the operator may prefer; it is not adopted only so that the two proposals issue keys of one shape. A renumbering to the next unused GP-REQ number (the remedy the first proposal used for TR-P02-011 → TR-P02-028) is also possible but would change a declared id inside a Drive document not downloaded for this proposal.
- Explicitly not proposed: Neither document's DECLARED ID is changed, no row is labelled VOID or DUPLICATE (neither is), and no content adjudication between the two rows is opened: they are two objects, and each row's Status is its own object's. Only the register row key of row 373 is disambiguated.
- Proposed disposition label: `COLLISION-PRESERVED / TWO-OBJECTS-BOTH-PRESERVED / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-ARTIFACT-INDEX-GPREQ194v10-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / DIFFERENT DRIVE OBJECTS / TWO DISTINCT REVIEW REQUESTS |
| `File A` | Artifact Index row 325 (2026-09-18 xlsx export) — GP-REQ-194-v1.0 — Status: OPEN ASSIGNMENT / NO REVIEW CREDIT UNTIL EXECUTED |
| `File B` | Artifact Index row 373 (2026-09-18 xlsx export) — GP-REQ-194-v1.0 — Status: OPEN ASSIGNMENT / RAW ZIP PRESENT / CLEAN BUILD REQUIRED |
| `Similarity / hash` | Same declared Artifact ID only; different Drive sources (https://docs.google.com/document/d/1mM355KwRN4ZiYCB9FHftl5gdiIT1BuYoQQhGMQKgynQ/edit vs https://docs.google.com/document/d/139NzUfVXV5-rW-pn_vIu6xFrqnwDcsHYPbWIn7FqOPs/edit), different Titles; Title, Org, Priority, Topics / object tags, Status, Authority / canonical impact, Dependencies or supersession, Modified UTC, Source, Notes differ |
| `Risk` | Wrong-object citation and duplicate-assignment risk; a review routed by the bare ID could land on the wrong request (OP-CNS-001 §2 GP-DER-044 case; §7 T7) |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows and both documents; keep row 325 as GP-REQ-194-v1.0; register row 373 additively under compound key GP-REQ-194-v1.0@AIDX-R373; cite each object by full title and Drive ID per OP-CNS-001 §2; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Artifact Index |
| `Detected at` | 2026-09-18 xlsx export scan; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Artifact Index, row 373, column 'Artifact ID'; `GP-REQ-194-v1.0` → `GP-REQ-194-v1.0@AIDX-R373`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Which of the two objects should carry the bare id GP-REQ-194-v1.0 is decided here only by append position. The workbook's own different-object precedents (DUP-CLWO-20260727-07-STABLEKEY-20260730 and siblings; Artifact Index rows 645/646) re-keyed BOTH rows with a Drive-ID-prefix discriminator and kept the bare id on neither. The operator may prefer that form (GP-REQ-194-v1.0@1mM355K for row 325 and GP-REQ-194-v1.0@139NzUf for row 373); it is recorded as the alternative and is not adopted here only so that the two proposals issue keys of one shape.
- Whether the Lean-build request (row 373) should instead be renumbered to the next unused GP-REQ number, the remedy the 2026-09-18 proposal used for two distinct Transition Log transitions sharing one key (TR-P02-011 → TR-P02-028), is not decided here; a renumbering changes the declared id inside a Drive document that was not downloaded for this proposal.
- Neither review request is examined for whether it was executed; both Status cells read 'OPEN ASSIGNMENT' and are transcribed, not evaluated.
- Which Status text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- Neither Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct status text; that is an operator content adjudication under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party.
- Does not establish that any listed artifact is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

### 1.6 `LS-MAN-045-v1.0` — rows 544, 545

> Finding key: `artifact_index: duplicate key 'LS-MAN-045-v1.0' at rows 544 and 545`

Defect class: `DUPLICATE_REGISTER_PRIMARY_KEY__SAME_DRIVE_OBJECT__DIFFERENT_STATUS_TEXT`. Both rows cite Drive source https://docs.google.com/document/d/1fJbcQOpV18vpAWTL8jbJwYQwd1g6NZZSiINTYbEutNg/edit.

#### Row 544 — proposed KEEPER row, verbatim

`registers/json/artifact_index.json` rows[544], canonical 748 bytes, SHA-256 `d875f663f7d25639cba8c91c28aa7ed4e7e930bffc2804a23c32f054764ec951`:

```json
["LS-MAN-045-v1.0","Theorem B Collision-Safe Same-Run Successor after Dual LS-MAN-044 Race and Populated LS-REQ-030","OpenAI GPT-5.6 Thinking / Drive Self-Healing Audit","MAN / FULL SAME-RUN LEAF VERIFICATION / CONCURRENCY RECOVERY","P0 / OPS","Theorem B package; four-Claude integration; DQ-056; collision recovery","CURRENT PACKAGE-STATE MANIFEST / FINAL 72 CHILDREN / ROOT NONE / SEALED NO","Package integrity only; zero scientific and organizational-independence credit","Supersedes LS-MAN-043 current role and both LS-MAN-044 attempts","2026-07-27T18:20:00Z","https://docs.google.com/document/d/1fJbcQOpV18vpAWTL8jbJwYQwd1g6NZZSiINTYbEutNg/edit","Body 5,906 / SHA ff4b5b6e…0e0f; collision-safe 72-child endpoint; theorem remains retracted."]
```

#### Row 545 — proposed REIDENTIFIED row, verbatim

`registers/json/artifact_index.json` rows[545], canonical 652 bytes, SHA-256 `4d130831ab3f902a6f1257b7285b272f94f7132539f1ac3c870897dce835881b`:

```json
["LS-MAN-045-v1.0","Theorem B Collision-Safe Same-Run Successor after Dual LS-MAN-044 Race","OpenAI GPT-5.6 Thinking","MAN / FULL SAME-RUN LEAF VERIFICATION / CONCURRENCY RECOVERY","P0 / OPS","Theorem B; 72-child state; DQ-056; four-Claude integration","CURRENT PACKAGE-STATE MANIFEST / ROOT NONE / SEALED NO","Package integrity only; zero scientific and independence credit","Supersedes LS-MAN-043 and both LS-MAN-044 attempts","2026-07-27T18:15:00Z","https://docs.google.com/document/d/1fJbcQOpV18vpAWTL8jbJwYQwd1g6NZZSiINTYbEutNg/edit","Body 5,906 / SHA ff4b5b6e…0e0f; 39 exact objects attempted; zero mismatch or silent skip; final 72 children."]
```

#### Where the two rows differ

| Field | Row 544 (keeper) | Row 545 (reidentified) |
|---|---|---|
| **Title** | Theorem B Collision-Safe Same-Run Successor after Dual LS-MAN-044 Race and Populated LS-REQ-030 | Theorem B Collision-Safe Same-Run Successor after Dual LS-MAN-044 Race |
| **Org** | OpenAI GPT-5.6 Thinking / Drive Self-Healing Audit | OpenAI GPT-5.6 Thinking |
| **Topics / object tags** | Theorem B package; four-Claude integration; DQ-056; collision recovery | Theorem B; 72-child state; DQ-056; four-Claude integration |
| **Status** | CURRENT PACKAGE-STATE MANIFEST / FINAL 72 CHILDREN / ROOT NONE / SEALED NO | CURRENT PACKAGE-STATE MANIFEST / ROOT NONE / SEALED NO |
| **Authority / canonical impact** | Package integrity only; zero scientific and organizational-independence credit | Package integrity only; zero scientific and independence credit |
| **Dependencies or supersession** | Supersedes LS-MAN-043 current role and both LS-MAN-044 attempts | Supersedes LS-MAN-043 and both LS-MAN-044 attempts |
| **Modified UTC** | 2026-07-27T18:20:00Z | 2026-07-27T18:15:00Z |
| **Notes** | Body 5,906 / SHA ff4b5b6e…0e0f; collision-safe 72-child endpoint; theorem remains retracted. | Body 5,906 / SHA ff4b5b6e…0e0f; 39 exact objects attempted; zero mismatch or silent skip; final 72 children. |

Identical in both rows: `Artifact ID`, `Class`, `Priority`, `Source`.

Cell count: 12 columns compared, 4 identical, 8 differing (rows 544 and 545 of `registers/json/artifact_index.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** Both rows cite one Drive document, one Class and one Priority. Row 544 (Modified UTC 2026-07-27T18:20:00Z) reads Status 'CURRENT PACKAGE-STATE MANIFEST / FINAL 72 CHILDREN / ROOT NONE / SEALED NO' and Notes '... collision-safe 72-child endpoint; theorem remains retracted.'; row 545 (Modified UTC 2026-07-27T18:15:00Z) reads Status 'CURRENT PACKAGE-STATE MANIFEST / ROOT NONE / SEALED NO' and Notes '... 39 exact objects attempted; zero mismatch or silent skip; final 72 children.' The Title of row 544 adds 'and Populated LS-REQ-030'. Both Status cells claim to be the CURRENT manifest; two rows cannot both be current under one key. This is a material content difference under OP-CNS-001 §1. The earlier-appended row carries the LATER self-reported timestamp.

**Corroborating register evidence.** The Evidence Lineage collision EV-LS-REQ030 (rows 385 and 386, this proposal's record CP-EVL-EV-LS-REQ030) carries the same two timestamps, 2026-07-27T18:15:00Z and 18:20:00Z, and row 544's Title names 'Populated LS-REQ-030': the two collisions are two register faces of one registration event that wrote twice, five minutes apart. Evidence Lineage rows 387 and 388 (EV-CONCURRENT-LSMAN044-CLAUDE, EV-CONCURRENT-LSMAN044-OPENAI, both 18:20:00Z, Evidence Role 'HISTORICAL_CONCURRENT_MANIFEST_ATTEMPT') record the 'Dual LS-MAN-044 Race' both Titles name. Artifact Index row 543 (GP-REC-SELFHEAL-20260727-27, 19:15:00Z) precedes both rows in append order while carrying a later timestamp, so the tab's row order in this region is not chronological.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Gaps or duplicates require an additive provenance disclosure.'
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' Noted BECAUSE IT DOES NOT APPLY: both rows cite one Drive ID, so this is a register-row duplication, not a second artifact identity. Minting a new declared artifact ID would create a second identity for one Drive object and is therefore NOT proposed.
- OP-PROT-019-v1.1 R17 §3 — 'Multiple successors of one old head are a publication conflict: hold only that target, preserve both branches, and append a reconciliation selecting or merging them with reasons. Never resolve it by silently overwriting proof bytes.' and 'Use the confirmed append position to order contenders, not a self-reported timestamp.'
- OP-PROT-019-v1.1 R17 §6 — 'No permanent deletion in this workflow.' and 'Moving or renaming must not be represented as a scientific verdict.'
- OP-PROT-012 §4 Class 1 — duplicate flagging, provenance repair and index maintenance are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** Duplicate Flags cluster DUP-REG-ARTIFACT-INDEX-GP197-20260724: 'DUPLICATE REGISTER PRIMARY KEYS / SAME DRIVE OBJECTS ... RESOLVED: first rows control; later rows reidentified VOID-DUPLICATE with unique keys; preserve additively.' Also SELFHEAL-LSMAN037-REG-001: 'Preserve earlier authoritative rows; reidentify later rows as VOID-DUPLICATE / ZERO CREDIT.' Compound row keys are the workbook's own form: Artifact Index rows 645 and 646 carry 'CL-AUD-LSDER036-20260727-01@1zMq504' and 'CL-AUD-LSDER036-20260727-01@1AcCpCe'; Duplicate Flags cluster DUP-CLWO-20260727-07-STABLEKEY-20260730 reads 'register compound keys CL-CLWO-20260727-07@1jqT3KJ and @1jGggrx; preserve both documents'. The 2026-09-18 proposal (registers/collision_proposal.json, records CP-AIDX-*) applied the same rule to six Artifact Index pairs of this class.

**Which row keeps the original id, and why**

- Keeper: **row 544** keeps `LS-MAN-045-v1.0`.
- Reason: Earlier confirmed append position. R17 §3 orders contenders by append position, not by self-reported timestamp, and the in-register Artifact Index precedent (DUP-REG-ARTIFACT-INDEX-GP197-20260724, SELFHEAL-LSMAN037-REG-001) resolves same-Drive-object register-row duplicates by letting the first row control and reidentifying the later row. The rule is the one the 2026-09-18 proposal applied to its six Artifact Index pairs and is applied here mechanically so that no pair is resolved by this session's reading of its status text.
- Explicitly not implied: Keeping the bare key is a KEY assignment, not a currency verdict. It does not make the keeper row's status text current, nor the successor row's status text stale.

**Proposed successor identifier**

- Row 545 → `LS-MAN-045-v1.0@AIDX-R545`
- Naming rule: Compound row key <declared artifact ID>@<discriminator>, the form the workbook already uses for stable-key collisions (Artifact Index rows 645/646 'CL-AUD-LSDER036-20260727-01@1zMq504' and '@1AcCpCe'; Duplicate Flags clusters DUP-CLWO-20260727-07-STABLEKEY-20260730 etc.). In those instances the discriminator was a Drive ID prefix. Here both rows cite ONE Drive ID, so the Drive-ID discriminator is unavailable and the discriminator falls back to the register row locator: AIDX = Artifact Index, R<n> = the 0-based row index tools/registers_check.py prints. This is the fallback the 2026-09-18 proposal adopted for its six pairs (GP-DER-118-v1.2@AIDX-R95 etc.) and is reused unchanged so that the two proposals issue keys of one shape. THE FALLBACK IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN FORM, NOT A LITERAL QUOTATION; the abbreviation requires operator ratification, and ratifying it once ratifies it for both proposals.
- Alternative discriminator considered: The reidentified row's own Modified UTC would be fully source-derived. It is not adopted, for the reason the 2026-09-18 proposal gave (it fails on pairs whose two rows carry one Modified UTC) and for a second reason visible only in this batch: for LS-MAN-045-v1.0 the later-appended row 545 carries the EARLIER timestamp (18:15:00Z vs row 544's 18:20:00Z), so a timestamp discriminator would invert the append order R17 §3 prescribes.
- Explicitly not proposed: The artifact's DECLARED ID inside the Drive document is not changed. Only the register row key is disambiguated. The 'VOID-DUPLICATE' disposition used by DUP-REG-ARTIFACT-INDEX-GP197-20260724 and SELFHEAL-LSMAN037-REG-001 is NOT proposed here, because those clusters' rows were substantively identical ('same Drive IDs and same evidence/request roles', 'identical LS-MAN-037 export identity') whereas this pair differs materially in Status; labelling either row VOID would destroy recorded state.
- Proposed disposition label: `COLLISION-PRESERVED / CONTENT-ADJUDICATION-PENDING / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-ARTIFACT-INDEX-LSMAN045v10-20260919 |
| `Detection type` | DUPLICATE REGISTER PRIMARY KEY / SAME DRIVE OBJECT / DIFFERENT STATUS TEXT |
| `File A` | Artifact Index row 544 (2026-09-18 xlsx export) — LS-MAN-045-v1.0 — Status: CURRENT PACKAGE-STATE MANIFEST / FINAL 72 CHILDREN / ROOT NONE / SEALED NO |
| `File B` | Artifact Index row 545 (2026-09-18 xlsx export) — LS-MAN-045-v1.0 — Status: CURRENT PACKAGE-STATE MANIFEST / ROOT NONE / SEALED NO |
| `Similarity / hash` | Same declared Artifact ID and same Drive source https://docs.google.com/document/d/1fJbcQOpV18vpAWTL8jbJwYQwd1g6NZZSiINTYbEutNg/edit; Title, Org, Topics / object tags, Status, Authority / canonical impact, Dependencies or supersession, Modified UTC, Notes differ |
| `Risk` | Citation ambiguity, double-count and stale-status routing risk; the bare ID does not identify one row |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 544 as LS-MAN-045-v1.0; register row 545 additively under compound key LS-MAN-045-v1.0@AIDX-R545; open an OP-CNS-001 §1 content-adjudication item on the Status difference; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Artifact Index |
| `Detected at` | 2026-09-18 xlsx export scan; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Artifact Index, row 545, column 'Artifact ID'; `LS-MAN-045-v1.0` → `LS-MAN-045-v1.0@AIDX-R545`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Here append position (row 544 first) and self-reported timestamp (row 545 earlier) disagree. R17 §3 says append position controls and the rule is applied mechanically; but the export cannot show whether the sheet was ever re-sorted in this region (row 543 carries 19:15:00Z, later than both), so the operator should confirm the append order from the workbook's revision history before executing.
- Whether the 'FINAL 72 CHILDREN' reading of row 544 is the current manifest state is unresolved by the export alone and is not resolved here.
- Which Status text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- Neither Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct status text; that is an operator content adjudication under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party.
- Does not establish that any listed artifact is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

## Part 2 — One duplicate Evidence ID in the Evidence Lineage

`KNOWN_FINDINGS.json` describes the two `EV-LS-REQ030` rows as binding the same Drive ID and the same code fixture digest, which raised the question whether they are an exact duplicate — the `OP-PROT-019 §6` `EXACT_DUPLICATE` case, for which the tab has its own executed `VOID-DUPLICATE` remedy (row 334). They were compared cell for cell: **they are not.** Eight of sixteen cells differ, among them Evidence Role, Review ID (`DQ-057` vs `NONE`) and Status. The pair is therefore treated exactly as the same-object Artifact Index pairs are, with the Evidence Lineage analogue `<Evidence ID>@EVL-R<row>` of the row-locator key.

| Evidence ID | Rows | Class | Keeper row | Reidentified row | Proposed successor row key |
|---|---|---|---|---|---|
| `EV-LS-REQ030` | 385, 386 | same object, different role and status text | 385 | 386 | `EV-LS-REQ030@EVL-R386` |

The checker holds this row, like the six of Part 1, to the record's cells: the two row indices, the keeper (the earlier row), the reidentified row (the later) and the successor id are recomputed, not read from here.

### 2.1 `EV-LS-REQ030` — rows 385, 386

> Finding key: `evidence_lineage: duplicate key 'EV-LS-REQ030' at rows 385 and 386`

Defect class: `DUPLICATE_EVIDENCE_PRIMARY_KEY__SAME_DRIVE_OBJECT__DIFFERENT_ROLE_AND_STATUS_TEXT`. Both rows cite Drive source https://docs.google.com/document/d/1BdLp0MKHE3RLu7bgI-g97DCQxVQI1NbnM414EfC4hw4/edit. Cell-for-cell identical: **false**.

#### Row 385 — proposed KEEPER row, verbatim

`registers/json/evidence_lineage.json` rows[385], canonical 516 bytes, SHA-256 `c7d25bce83677c51da0367cba5e1123915570e61aa91351039b9d9ef490bcd3c`:

```json
["EV-LS-REQ030","2026-07-27T18:15:00Z","LS-REQ-030-v1.0","1BdLp0MKHE3RLu7bgI-g97DCQxVQI1NbnM414EfC4hw4","LS-REQ-030-v1.0","EXACT_REVIEW_REQUEST_FOR_LS_DER036","TRUE","FALSE","LS-DER-036; LS-DER-035; LS-DER-023/034","SHA256:ccf625dd77ae6c7b8f575682740b46359ad8685e34799108edf80909992311df","Populated exact-scope review packet","Routing only / zero independence credit","DQ-057","C WHOLE NATIVE EXPORT","https://docs.google.com/document/d/1BdLp0MKHE3RLu7bgI-g97DCQxVQI1NbnM414EfC4hw4/edit","PASS / OPEN REVIEW ROUTE"]
```

#### Row 386 — proposed REIDENTIFIED row, verbatim

`registers/json/evidence_lineage.json` rows[386], canonical 550 bytes, SHA-256 `032fc21dbfd3073fdce84a98ba1ab6ea2a436def82be601d464f872533e06eb0`:

```json
["EV-LS-REQ030","2026-07-27T18:20:00Z","LS-REQ-030-v1.0","1BdLp0MKHE3RLu7bgI-g97DCQxVQI1NbnM414EfC4hw4","LS-REQ-030-v1.0","ORGANIZATIONALLY_DISTINCT_REVIEW_REQUEST","TRUE","FALSE","LS-DER-036; LS-DER-035; LS-DER-023/034/021","SHA256:ccf625dd77ae6c7b8f575682740b46359ad8685e34799108edf80909992311df","R0–R14 and M1–M12 exact review ledger","OpenAI routing / zero review credit","NONE","C WHOLE NATIVE TEXT EXPORT","https://docs.google.com/document/d/1BdLp0MKHE3RLu7bgI-g97DCQxVQI1NbnM414EfC4hw4/edit","PASS CONTENT BINDING / EXTERNAL REVIEW OPEN"]
```

#### Where the two rows differ

| Field | Row 385 (keeper) | Row 386 (reidentified) |
|---|---|---|
| **Recorded UTC** | 2026-07-27T18:15:00Z | 2026-07-27T18:20:00Z |
| **Evidence Role** | EXACT_REVIEW_REQUEST_FOR_LS_DER036 | ORGANIZATIONALLY_DISTINCT_REVIEW_REQUEST |
| **Derives From** | LS-DER-036; LS-DER-035; LS-DER-023/034 | LS-DER-036; LS-DER-035; LS-DER-023/034/021 |
| **Computational Fixture ID** | Populated exact-scope review packet | R0–R14 and M1–M12 exact review ledger |
| **Methodological Family** | Routing only / zero independence credit | OpenAI routing / zero review credit |
| **Review ID** | DQ-057 | NONE |
| **Evidence Class** | C WHOLE NATIVE EXPORT | C WHOLE NATIVE TEXT EXPORT |
| **Status** | PASS / OPEN REVIEW ROUTE | PASS CONTENT BINDING / EXTERNAL REVIEW OPEN |

Identical in both rows: `Evidence ID`, `Exact Object ID`, `Drive ID`, `Artifact ID`, `Primary Evidence`, `Summary Only`, `Code Fixture ID`, `Source URL`.

Cell count: 16 columns compared, 8 identical, 8 differing (rows 385 and 386 of `registers/json/evidence_lineage.json`; the checker recomputes both lists and these counts from the live rows).

**Materiality.** The two rows were compared cell for cell: they are NOT an exact duplicate. Eight of sixteen cells agree — Evidence ID, Exact Object ID 'LS-REQ-030-v1.0', Drive ID '1BdLp0MKHE3RLu7bgI-g97DCQxVQI1NbnM414EfC4hw4', Artifact ID, Primary Evidence 'TRUE', Summary Only 'FALSE', Code Fixture ID 'SHA256:ccf625dd77ae6c7b8f575682740b46359ad8685e34799108edf80909992311df' and Source URL — so both rows bind one Drive object with one content digest. Eight cells differ: Recorded UTC (18:15:00Z vs 18:20:00Z), Evidence Role ('EXACT_REVIEW_REQUEST_FOR_LS_DER036' vs 'ORGANIZATIONALLY_DISTINCT_REVIEW_REQUEST'), Derives From, Computational Fixture ID, Methodological Family, Review ID ('DQ-057' vs 'NONE'), Evidence Class ('C WHOLE NATIVE EXPORT' vs 'C WHOLE NATIVE TEXT EXPORT') and Status ('PASS / OPEN REVIEW ROUTE' vs 'PASS CONTENT BINDING / EXTERNAL REVIEW OPEN'). A Review ID that is 'DQ-057' in one row and 'NONE' in the other is a material content difference under OP-CNS-001 §1; the EXACT_DUPLICATE_ROW class and the VOID-DUPLICATE disposition therefore do NOT apply.

**Corroborating register evidence.** The Artifact Index collision LS-MAN-045-v1.0 (rows 544 and 545, this proposal's record CP-AIDX-LS-MAN-045-v1.0) carries the same two timestamps, and row 544's Title reads '... and Populated LS-REQ-030'. The two collisions are one registration event. Evidence Lineage row 384 (EV-CONCURRENT-DUP-LSDER030-1V9VGW, 18:00:00Z) and rows 387/388 (EV-CONCURRENT-LSMAN044-CLAUDE / -OPENAI, 18:20:00Z) show the tab was absorbing concurrent writes in the same quarter hour. Row 385's Review ID 'DQ-057' is the review route the Artifact Index rows 645/646 (CL-AUD-LSDER036-20260727-01@...) name as 'DQ-057 v1.1 review remains open'.

**Protocol clause that governs**

- OP-CNS-001-R0.2 §2 — 'A later collision is preserved and disambiguated; historical artifacts are not silently rewritten.' / 'Maintain an append-only collision registry.' / 'Gaps or duplicates require an additive provenance disclosure.'
- OP-CNS-001-R0.2 §1 — 'Material content differences open a content-adjudication item rather than being silently merged.'
- OP-CNS-001-R0.2 §7 T7 — 'same declared ID with different Drive IDs remains separate.' Noted BECAUSE IT DOES NOT APPLY: both rows cite one Drive ID and one content digest, so this is a register-row duplication of one evidence binding, not two pieces of evidence.
- OP-PROT-019-v1.1 R17 §3 — 'Multiple successors of one old head are a publication conflict: hold only that target, preserve both branches, and append a reconciliation selecting or merging them with reasons. Never resolve it by silently overwriting proof bytes.' and 'Use the confirmed append position to order contenders, not a self-reported timestamp.'
- OP-PROT-019-v1.1 R17 §6 — 'No permanent deletion in this workflow.' and 'Moving or renaming must not be represented as a scientific verdict.'
- OP-PROT-012 §4 Class 1 — duplicate flagging, provenance repair and index maintenance are reversible operations; nothing here is Class 2 or Class 3.

**In-register precedent.** Evidence Lineage row 334, Evidence ID 'VOID-DUPLICATE-EV-LSMAN037-20260726T1943', Evidence Role 'DUPLICATE_REGISTRATION_RECEIPT', Evidence Class 'VOID DUPLICATE / ZERO EVIDENCE CREDIT', Status 'VOID — authoritative package-integrity evidence remains row 357; no duplicated credit' — the tab's own executed remedy for a duplicate Evidence Lineage registration (Duplicate Flags cluster SELFHEAL-LSMAN037-REG-001: 'Artifact Index row 517 / Evidence Lineage row 357' vs 'Artifact Index row 519 / Evidence Lineage row 359', 'Preserve earlier authoritative rows; reidentify later rows as VOID-DUPLICATE / ZERO CREDIT'). Also Evidence Lineage row 384, Evidence ID 'EV-CONCURRENT-DUP-LSDER030-1V9VGW', Evidence Role 'COLLISION_PROVENANCE_ONLY', Status 'PRESERVED / NONCONTROLLING / COLLISION CONTAINED' — the tab's Drive-ID-suffixed form for a collision whose rows cite different Drive IDs.

**Which row keeps the original id, and why**

- Keeper: **row 385** keeps `EV-LS-REQ030`.
- Reason: Earlier confirmed append position (and, here, also the earlier self-reported timestamp, which is noted but not relied on: R17 §3 orders contenders by append position). The tab's own precedent SELFHEAL-LSMAN037-REG-001 resolves a duplicate Evidence Lineage registration by 'Preserve earlier authoritative rows; reidentify later rows', and the same rule is the one applied to every Artifact Index pair in both proposals.
- Explicitly not implied: Keeping the bare key is a KEY assignment, not a currency verdict. It does not make the keeper row's status text current, nor the successor row's status text stale.

**Proposed successor identifier**

- Row 386 → `EV-LS-REQ030@EVL-R386`
- Naming rule: Compound row key <Evidence ID>@EVL-R<n>, the Evidence Lineage analogue of the AIDX form the 2026-09-18 proposal issued for the Artifact Index. EVL = Evidence Lineage, R<n> = the 0-based row index tools/registers_check.py prints. Why a separate abbreviation: a successor key must say which tab's row it locates, and 'AIDX-R386' would name Artifact Index row 386, which is a different row of a different tab; the locator is tab-scoped. Why not the tab's own forms: the VOID-DUPLICATE form (row 334) asserts 'ZERO EVIDENCE CREDIT' and 'no duplicated credit' about a row that is substantively identical to its keeper, which is false of row 386 (different Evidence Role, Review ID and Status); the Drive-ID-suffix form (row 384, EV-CONCURRENT-DUP-LSDER030-1V9VGW) discriminates by Drive ID, which is unavailable because both rows cite one Drive ID. THE EVL ABBREVIATION IS AN EXTRAPOLATION FROM THE WORKBOOK'S OWN FORM, NOT A LITERAL QUOTATION; it requires operator ratification.
- Alternative discriminator considered: The reidentified row's own Recorded UTC (EV-LS-REQ030@2026-07-27T18:20:00Z) would be fully source-derived and would work for this one pair. It is not adopted because the sibling Artifact Index pair of the same registration event (LS-MAN-045-v1.0, rows 544/545) has its append order and timestamp order inverted, and a rule that discriminates two faces of one event by two different principles would be a new source of confusion.
- Explicitly not proposed: The evidence's Exact Object ID, Drive ID and Code Fixture ID are not changed; only the register row key is disambiguated. The VOID-DUPLICATE disposition of Evidence Lineage row 334 is NOT proposed, because that row was a duplicate registration receipt with 'no duplicated credit' to protect, whereas rows 385 and 386 differ in Evidence Role, Review ID and Status; labelling either VOID would destroy recorded state. No new Evidence ID is minted for the same Drive object.
- Proposed disposition label: `COLLISION-PRESERVED / CONTENT-ADJUDICATION-PENDING / ZERO INDEPENDENCE CREDIT`

**Exact append-only collision-registry entry that would be added**

Operation `APPEND_ROW` → *Duplicate Flags (GP-REG-032-v1.2) — the workbook's append-only collision registry*. Appends one row; mutates nothing.

| Column | Value |
|---|---|
| `Cluster ID` | DUP-REG-EVIDENCE-LINEAGE-EVLSREQ030-20260919 |
| `Detection type` | DUPLICATE EVIDENCE PRIMARY KEY / SAME DRIVE OBJECT / DIFFERENT ROLE AND STATUS TEXT |
| `File A` | Evidence Lineage row 385 (2026-09-18 xlsx export) — EV-LS-REQ030 — Status: PASS / OPEN REVIEW ROUTE |
| `File B` | Evidence Lineage row 386 (2026-09-18 xlsx export) — EV-LS-REQ030 — Status: PASS CONTENT BINDING / EXTERNAL REVIEW OPEN |
| `Similarity / hash` | Same declared Evidence ID and same Drive source https://docs.google.com/document/d/1BdLp0MKHE3RLu7bgI-g97DCQxVQI1NbnM414EfC4hw4/edit; Recorded UTC, Evidence Role, Derives From, Computational Fixture ID, Methodological Family, Review ID, Evidence Class, Status differ |
| `Risk` | Citation ambiguity, double-count and stale-status routing risk; the bare ID does not identify one row |
| `Recommended action` | PROPOSED — NOT YET EXECUTED. Preserve both rows; keep row 385 as EV-LS-REQ030; register row 386 additively under compound key EV-LS-REQ030@EVL-R386; open an OP-CNS-001 §1 content-adjudication item on the Evidence Role / Review ID / Status difference; no merge, no deletion, no promotion, zero independence credit. |
| `Source folder` | GP-REG-032 Evidence Lineage |
| `Detected at` | 2026-09-18 xlsx export scan; proposal filed 2026-09-19 |

**Operator-reserved follow-up (NOT executed, NOT additive):** `REIDENTIFY_KEY_CELL` — GP-REG-032 source workbook, Evidence Lineage, row 386, column 'Evidence ID'; `EV-LS-REQ030` → `EV-LS-REQ030@EVL-R386`.

- Why it is not in the operations block: Writing a source cell is not additive. It would change the live workbook and therefore the next export's bytes; registers_import.py --check would fail until a fresh dated export is taken and registers/json and registers/csv are regenerated. It is listed so the proposal is complete, and is explicitly NOT proposed for autonomous execution.
- Authority required: Owner (Dylan Roy) or an operator acting under OP-PROT-012 §4 Class 1 with target identity, pre-change state and rollback path recorded.

**Residual questions — not decided here**

- Whether the Review ID of the evidence is DQ-057 (row 385) or NONE (row 386) is the content-adjudication item; it is handed to the operator unresolved. Nothing here awards, withdraws or routes any review credit.
- The evidence-lineage tab has no compound '@' key of its own; the EVL row-locator form is proposed by analogy with the AIDX form and requires the same operator ratification.
- Which Status text is current is NOT decided here.
- The keeper rule identifies the exported row order with the confirmed append position (R17 §3). The export cannot show whether rows were ever sorted, inserted or re-pasted; the identification is an assumption the operator can verify against the workbook's revision history, not a fact established here.
- Neither Drive body was downloaded for this proposal; the claim adjudicated is about register rows, not about document contents.

**What this record does not establish**

- Does not repair, promote, close, discharge, reclassify or retire any claim, premise, obligation or register row.
- Does not decide which of two colliding rows carries the currently correct status text; that is an operator content adjudication under OP-CNS-001 §1.
- Does not confer organizational-independence credit of any kind, on any object, to any party.
- Does not establish that any listed artifact is correct, complete, reviewed or novel.
- Does not change the 2026-09-18 export; registers/source/, registers/json/ and registers/csv/ are untouched and remain byte-faithful.
- Does not establish that the exported row order is the workbook's confirmed append order; the keeper rule takes the exported row order as the append position, and that identification is an operator-verifiable assumption, not a fact this document certifies.
- Does not compose the 2D upper/lower tracks with the 3D lifetime track, and solves no original prize problem.

---

## Closing statement

1. **This is a proposal. It requires operator action.** Seven records, seven findings, one to one; together with
   the frozen 2026-09-18 proposal, twenty-three records over twenty-three findings, none covered twice.
2. **Nothing has been repaired.** No register row was edited, merged, deleted, reordered or
   reclassified by this document. Every executable operation in it is an append to the workbook's
   append-only collision registry, and none of those appends has been performed either.
3. **The export remains faithful.** `registers/source/`, `registers/json/` and `registers/csv/` are
   byte-identical to `git HEAD`. `tools/collision_proposal_check.py` asserts this and exits nonzero
   if it ever stops being true.
4. **`independence_credit = 0`.** This session is Anthropic-family. OP-PROT-019-v1.1 R17 §4 permits a fresh nonauthor session of any provider to perform technical review, but records organizational independence separately and at ZERO for a same-provider reviewer. Independently of that, a PROPOSAL is not a review at all: it licenses no technical verdict and no independence credit on any object, for any provider.
5. **Every independence-requiring gate REMAINS OPEN**, unchanged by this document, regardless of any
   technical judgement expressed in it.
6. **No status is decided here.** Which of two colliding rows carries the currently correct status
   text is an `OP-CNS-001 §1` content-adjudication item in every one of the six same-object
   records, and it is handed to the operator unresolved in every one of them. The one
   different-objects record decides nothing about either object.
