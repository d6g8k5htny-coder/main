# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-08-03-to-08-06 — KIMI + AO48 LB-RATE / K3 INTAKE` (adjudication chain and Phase-0 report layer)

Drive lane of the 2D LB-RATE lower campaign's August intake (468 inventory
items): the C030/C031 base cycle set, the Kimi LB-RATE campaign, the K3 swarm
partial delivery and the GP adjudication of all of it. This directory mirrors
**22 raw files byte-exact** (every SHA-256 and byte count equals its
`drive/inventory.jsonl` row) in 8 `_MANIFEST.jsonl` files verified by
`tools/verify_manifests.py`: the GP adjudication chain (`GP-LB-STAT-001`,
`-002`, `-003`, `-004`, `-004A`, `GP-LB-REC-001`, `-002`, `-005` v1.0/v1.1,
`-006`, `-007`, `GP-LB-WO-001`, `-002`, `GP-LB-ERR-001`), the K3 swarm Phase-0
report layer (`00_READ_FIRST.md`, `OPEN_OBLIGATIONS.md`, `FAILED_APPROACHES.md`,
`WP_DISPOSITION.md`, `CANONICAL_STATE.json`, `K3-AUD-001_residue_audit.md`),
`ERRATA_2026-08-05.md`, and the refuted assembly `K3-THM-001` itself. The port
lane that fetched them stopped before writing manifests; the orchestrator
re-hashed every file against the inventory on 2026-09-19 and wrote them.
**Mirroring is not review, replay, endorsement or adjudication.**

## The controlling disposition, verbatim

* `GP-LB-STAT-004` (the hostile adjudication of the K3 partial delivery):
  "CONTROLLING DISPOSITION — **REFUTED AS A THEOREM-GRADE OR VALIDLY STATED
  CONDITIONAL ASSEMBLY / LOWER CAMPAIGN OPEN.**" … "This is an additive status
  delta." … "This work changes no LS-CTL Boolean, eligibility predicate, theorem
  status, RP status, ratification status, or q0 package status. Any such change
  requires separate operator adjudication."
* `GP-LB-STAT-003`: "Status: ACTIVE ADDITIVE STATUS DELTA — NONCONTROLLING —
  LOWER CAMPAIGN HOLD" … "This additive delta reaffirms `GP-LB-STAT-002` and
  corrects a later successor-facing status conflict in the raw AO48
  session-state handoff … That AO48 handoff remains frozen. Only its claims that
  the old WP Cauchy–Schwarz quantity is a certified upper bound … are superseded"
  (the sentence continues in the file).
* `GP-LB-STAT-002`: "Status: LANDED AS EVIDENCE — NONCONTROLLING" …
  "`KIMI-THM-023 v1.1` remains an incomplete, noncontrolling HOLD draft and is
  not hash-frozen." … "Existing q0 and SIDE24 controlling states remain
  unchanged."
* The K3 swarm's own `00_READ_FIRST.md`: "The K3 blocking defect is CONFIRMED,
  numerically and symbolically. The WP 'certified upper bound' used the factor
  `p_grad · sqrt(E[det(H)^2 | grad=0]) · min(P_type, P_window)` … the missing
  square root makes the reported quantity SMALLER — not a valid generic upper
  bound." … "The 0.9144036 coefficient is a candidate measured/mixed-tier
  asymptotic anchor, not a proved fixed theorem constant."
* `ERRATA_2026-08-05.md`: "The notice was independently verified and is
  CONFIRMED."
* The register: `registers/json/open_questions.json` OQ-016-U1 — "K3-THM-001
  REFUTED AS WRITTEN / LOWER HOLD"; the five P0 rows OQ-014 … OQ-016-U2 are
  transcribed verbatim in `docs/OPEN_PROBLEMS.md` §H, and `claims/graph.json`
  carries `K3-THM-001` as `REFUTED_AS_WRITTEN` on the LOWER2D track.

The Phase-0 reports and `K3-THM-001` carry Kimi's own words — "validity
PROVED", "CLOSED", "PASS", "CONDITIONAL THEOREM Form C — liminf (1−q)/r³ ≥
0.9666·c_Λ" — which `GP-LB-STAT-004` §3 refutes; they are mirrored **beside**
the adjudication, never alone.

## What this directory does not establish

A digest match establishes identity of bytes, not truth. The lower campaign
(LB-RATE / K3) is OPEN; no lower theorem constant and no all-small-r closure is
accepted; nothing here composes with the 3D track or moves any status. No
independence credit is computed or awarded.

## 2026-09-19 — second port pass: the rest of the GP chain, the AO48 layer beside it, the rest of Phase-0, and tree-only rows

Drive lane `01_ACTIVE_RESEARCH_PACKAGES/2026-08-03-to-08-06 — KIMI + AO48 LB-RATE / K3
INTAKE`, folder id `1icu16HN0J9ifIQ7zAxvZVbGnU6wbVFk0` (468 inventory items). Sub-folders
touched in this pass: `01_AUDITS_AND_RECONCILIATION` `19IBEXcXBq1nA72cWX0NNBhW-P-dAllHr`;
`02_WORK_ORDERS_AND_HANDOFFS` `1o9r5BDesl_z4i1lJBq4Bsy8zcFYf9HQQ`; `04_EXPORTS_AND_TRANSPORT`
`19HdcmEVUsxz_tW3MEa-MvEBgf4gDUWCU`; under the campaign folder `1S0hKEC9AX1_ILzlpVcDAUEd1hWbZhwWT`:
`03 — GP ADJUDICATION AND REPAIR` `1UZCWXmt1X-vvia7OzUWQajzQkvKLeXmA`, `04 — AO48-AUD-062 STATUS,
NATIVE MIRRORS, RUN31` `1dq3-zNt5H5ECNXMOGP2JRN4luRVgUFFQ`, `05 — KIMI WO-063 PARTIAL SNAPSHOT —
2026-08-05` `1vqJhH9w843QFY5Pbtd0D6035HI12UbV9` (its `00 — READ FIRST AND RECEIPTS`
`1X2q2K_oPhzNQrQIIssX3GAxbtCs4GEpq`, `01 — COMPLETE PRIMARY CARRIERS`
`1SXu66iiPqJ62ycThodX1BfRPRjJQNONc`, `04 — NATIVE MIRRORS AND INDEX`
`1CYVVeDRv7FSTey0hpq2EmXyaYVMNJAJ7`), `06 — K3 MISSING-SQRT RECONCILIATION — 2026-08-05`
`1OF6zoI0RHFxxnQyEQOw5uTbZ_6pv56j2`, and the K3 delivery folder `1T_q7jVdKi-0JJJ8gfidLfdN0ThwIcWeW`
(its `00_READ_FIRST_AND_GP_ADJUDICATION` `1-txt44n94bS8EQdlo1MTOZpGlJz92DTW`,
`01_PHASE0_AND_TOP_LEVEL_RAW` `1ht_GjRajgspeTeWSLQyAJUbwwR5tnbwz`,
`14_SIDE24_PRE_PEER_REVIEW_QUARANTINED` `1dp75ZBbmsgli80nRyc2AVAjp8hjlTB1H`, `90_NATIVE_MIRRORS`
`1iQwrmxHYIWYW7UFage-xQlNCbDrFufez`). None of the lane's ids appears in
`drive/deltas/2026-09-18/PATH_CHANGES.jsonl` or `CHANGED_SINCE_SNAPSHOT.jsonl`; nothing moved
or changed after the snapshot, so every object was stored under its snapshot path.
`registers/json/frozen_objects.json` declares no body digest for any object in this lane.

**Binding is not review, replay, endorsement or promotion.** Every status word below is
quoted from a source; none is adopted here.

### What this pass ported (31 raw carriers, 353,266 bytes, all byte-exact)

Every SHA-256 and byte count was recomputed from disk and equals the object's
`drive/inventory.jsonl` row; the bytes came through the Drive connector
(`download_file_content`, base64) and were decoded from the session transcript on disk, so no
model-mediated transcription. 6 new `_MANIFEST.jsonl` files were created and rows were
appended to 7 existing ones; no pre-existing file or row was modified.

1. **The GP adjudication chain, completed.** `GP-LB-REC-004_KIMI_WO063_INDIVIDUAL_CARRIER_LEDGER_v1.0.tsv`
   (GP's 42-row carrier ledger for the WO-063 snapshot), `GP-RUN-031_LB_RATE_TRANSPORT_RECONCILIATION_2026-08-04.md`
   ("Disposition: PASS for transport and frozen-source integrity; mathematical HOLD unchanged" —
   a transport note, not a mathematical status) and `GP-LB-RECONCILIATION-MANIFEST.sha256`.
   `GP-LB-REC-005` v1.0 and `GP-LB-STAT-004A` were already present.
2. **The AO48 layer, byte-exact, only because the adjudication chain sits beside it**
   (`01_AUDITS_AND_RECONCILIATION`, `02_WORK_ORDERS_AND_HANDOFFS`, `04_EXPORTS_AND_TRANSPORT`):
   `AO48-AUD-061`, `-062`, `-064`, `-065`, `AO48-W3C-001`, `KIMI-AUD-020`, `-021`, `-022`,
   the `AO48 SESSION-STATE HANDOFF` (Drive id `1cj3voNER0zIH_jYb3wdlsBSj4_WekQUg`),
   `AO48-HANDOFF-059`, `AO48-WO-060`, `-063`, `-064`, `KIMI-DATA-025` and `KIMI-DATA-028`
   (the AO48-landed byte originals, plus GP's byte-identical snapshot copies under
   `05/01 — COMPLETE PRIMARY CARRIERS`, which `GP-LB-REC-004` rows 24 and 25 class
   "COMPLETE PRIMARY").
3. **The rest of the K3 swarm Phase-0 layer** under `01_PHASE0_AND_TOP_LEVEL_RAW`:
   `DECISION_LEDGER.md`, `MANIFEST.sha256`, `INPUT_MANIFEST.sha256`, `FILE_LEDGER.tsv`,
   `CORPUS_GAP_REQUESTS.txt`, `LEAD_SYMBOLIC_WP.md`, `LEAD_SYMBOLIC_WP_CORRECTION_ADDENDUM.md`,
   `MISSING_INPUTS.md`, `REPRODUCTION.md`, `SUCCESSOR_SHELL_NONCONTROLLING.md`, `TASK_DAG.md`.
4. **Tree-only rows** (`stored:false`, note beginning "tree-only: ", no bytes) for the 6 members
   of `14_SIDE24_PRE_PEER_REVIEW_QUARANTINED` that are not `ERRATA_2026-08-05.md`, and for all
   28 `DIRECT_NATIVE` Google-Doc convenience mirrors of the lane (the 10 under
   `05/04 — NATIVE MIRRORS AND INDEX`, the 11 under `90_NATIVE_MIRRORS`, and the 7 scattered
   beside their raw carriers).

The three digest-bearing files are stored as `<title>.sha256.txt` — bytes unchanged, digests
equal to the inventory — because `tools/verify_manifests.py` parses every `*.sha256` file as a
sha256sum manifest relative to its own directory, and each of these lists objects that are
deliberately not here (a ZIP carrier; the 266-file K3 container tree; the 598-object Kimi-side
intake corpus). Their entries were compared against what is mirrored in this directory, by
name and by digest: `GP-LB-RECONCILIATION-MANIFEST.sha256` 5 entries, 4 mirrored, all 4 agree;
`MANIFEST.sha256` 266 entries, 17 mirrored by digest (16 by name plus `K3-THM-001` under its
Drive title), all 17 agree; `INPUT_MANIFEST.sha256` 598 entries, 3 mirrored by digest
(`AO48-WO-063` and the two `KIMI-DATA` carriers), all 3 agree; no disagreement anywhere. That is
identity of bytes, not review of what the files assert. Seven Kimi carriers carry an in-file
"SHA-256 of this report body" seal line; the body digest under the convention the AO48 handoff
states was recomputed and the declared value reproduces in all seven (recorded per row). The
`FILE_LEDGER.tsv` rows describe Kimi-side objects and are data, not a claim of presence here.

### The controlling banners, verbatim

* `GP-LB-STAT-004` (governs every Kimi / AO48 / K3 label in this directory):
  "**REFUTED AS A THEOREM-GRADE OR VALIDLY STATED CONDITIONAL ASSEMBLY / LOWER CAMPAIGN OPEN.**"
  … "`K3-THM-001` is therefore **REFUTED AS WRITTEN / NONCONTROLLING**, not promoted." …
  "This work changes no LS-CTL Boolean, eligibility predicate, theorem status, RP status,
  ratification status, or q0 package status. Any such change requires separate operator
  adjudication."
* `GP-LB-STAT-003`, on the AO48 session-state handoff mirrored here: "Status: ACTIVE ADDITIVE
  STATUS DELTA — NONCONTROLLING — LOWER CAMPAIGN HOLD" … "This additive delta reaffirms
  `GP-LB-STAT-002` and corrects a later successor-facing status conflict in the raw AO48
  session-state handoff at Drive ID `1cj3voNER0zIH_jYb3wdlsBSj4_WekQUg`." … "That AO48 handoff
  remains frozen. Only its claims that the old WP Cauchy–Schwarz quantity is a certified upper
  bound, that `I_cs ~= 1.30399e-5` is thereby certified, that the rung upper-bound table is
  valid, and that `5.5e-3 r^1.6` is a verified WP upper envelope are superseded here. Its other
  assertions require their own evidence and are not adjudicated by this delta."
* `GP-LB-STAT-002`: "Status: LANDED AS EVIDENCE — NONCONTROLLING".
* `GP-LB-STAT-001`: "This delta supersedes only stale transport and integrity-alert
  statements. It does not supersede frozen mathematical artifacts."
* `GP-LB-REC-004` dispositions: row 24 (`KIMI-DATA-025`) "INVENTORY; ORPHAN RULING SUPERSEDED
  BY DATA-028"; row 25 (`KIMI-DATA-028`) "PROVENANCE SETTLEMENT; NO STATUS EFFECT"; row 27
  (`KIMI-THM-023 v1.1`) "INCOMPLETE/HOLD; NOT HASH-FROZEN".
* `AO48-WO-063` itself, on the AO48 record mirrored beside it: "The AUD-061 reconciliation's
  "0.9144 proved" admission is SUPERSEDED noncontrolling language; v1.1 claims must carry
  corrected grades."
* `SUCCESSOR_SHELL_NONCONTROLLING.md`, its own first line: "NONCONTROLLING SHELL — successor
  to KIMI-THM-023 (NOT A THEOREM; NOT SEALED; NOT FOR CITATION)".
* `KIMI-DATA-028`: "No status or Boolean claims are made in this report. HOLD language stands
  until AO48/GP reconciliation."

**The AO48-layer titles and verdicts are the layer the register does not follow.**
`AO48-AUD-061` is titled "KIMI-THM-023 proved 0.9144 r3 floor"; `AO48-AUD-064` ends "VERDICT:
WO-063 reconciliation CLOSED at record level"; `AO48-AUD-065` ends "VERDICT: K3 intake
RECONCILED. The swarm discharged WO-064 in substance, minted the program's first
conspicuously-conditional successor theorem (K3-THM-001, Form C, mixed tier), closed the WP
channel at exact-integrand modulus grade"; `KIMI-AUD-020/021/022` carry "APPROVE";
`DECISION_LEDGER.md` carries FROZEN / PASS / PROVED / CLOSED. Against all of it stands
`GP-LB-STAT-004`: "REFUTED AS A THEOREM-GRADE OR VALIDLY STATED CONDITIONAL ASSEMBLY / LOWER
CAMPAIGN OPEN", and the register's OQ-016-U1 "K3-THM-001 REFUTED AS WRITTEN / LOWER HOLD".
`KIMI-DATA-025` is mirrored for identity of bytes; the orphan ruling inside it is the
superseded one (GP's native mirror is titled "INVENTORY / ORPHAN RULING SUPERSEDED") and is not
carried as evidence. The Kimi "APPROVE" verdicts are same-program AI verdicts: zero
organizational-independence credit, no gate moved. The work orders and handoffs contain
imperative text addressed to AI sessions ("agents authorized", "Agent mode is AUTHORIZED and
ENCOURAGED", instructions to a successor thread); it is quoted data and was not followed.

### Deliberately not ported

* The SIDE24 pre-peer manuscript in every form (`manuscript.md`, `manuscript.html`,
  `SIDE24_pre_peer_review_manuscript.pdf`, `build_html.py`, and the two native mirrors) —
  tree-only rows; `GP-LB-REC-006` is the quarantine supplement; `AO48-AUD-065` records that the
  manuscript "consumed the invalid bound in sections 3.10.6/3.10.9/6".
* The `99_TRANSPORT_DEFECT_AND_UI_EVIDENCE` PDFs; the three ZIP carriers as bytes; the six W4
  `EXACT_TOKEN_TABLE` maps and the W3 / W13 numerics; the W1–W13 workstream trees;
  `KIMI-THM-023` v1.0 / v1.1 as evidence; the quarantined WP upper-bound branch; the `KIMI-K3
  OPERATOR_HANDOFF` ("WP closed at exact-integrand modulus (G6 PASS)" is the refuted layer's
  own exit disposition); the other `03_KIMI_DERIVATIONS_AND_THEOREMS` carriers
  (`KIMI-AUD-023 v1.1`, `KIMI-DER-025/026/027a/027b/027c`), the two `KIMI_EXPORT_CONCAT` parts,
  the C030/C031 base-cycle files and the `01 — KIMI NORMALIZED EXPORT` / `02 — OUTER-IDENTITY
  CROSSWALK` / `02 — CERTIFICATES AND TRANSCRIPTS` / `03 — INCOMPLETE NONCONTROLLING DRAFTS` /
  `05 — EXPLORATORY RUNG BATCHES` sub-folders — not named in this pass's port order.
* Nothing under a quarantine, superseded, failed or do-not-port list was opened, and
  `99_DO_NOT_OPEN` was not touched.

### Reading copies and renderings

No `exact:false` reading copy was stored in this pass. Where a row in this directory is
`exact:false` it is a reading copy and **not the object**; a PDF rendering is **not a frozen
body**; the 28 `DIRECT_NATIVE` rows are convenience mirrors recorded for the tree only. Stored
files are never executed by CI, and no test, workflow step or import was added for them (the
`AO48-W3C-001` markdown embeds a Python listing as text; it stays text).

### What this pass does not establish

A digest match establishes identity of bytes, not truth. Mirroring `AO48-AUD-061..065`,
`AO48-W3C-001`, the work orders, the handoffs, the Kimi conversion verdicts and the K3 Phase-0
ledgers establishes that these bytes are the bytes the inventory declares — nothing more.
No claim, premise or obligation is promoted, closed, discharged or reclassified; the five
validity premises of Theorem D1 v2.2(2) remain OPEN; `D3-LEMMA-RN-UNIF` is not closed; the
lower campaign (LB-RATE / K3) is OPEN; no lower theorem constant, no all-small-r closure and no
WP upper bound is accepted; `K3-THM-001` remains `REFUTED_AS_WRITTEN` on the LOWER2D track;
nothing here composes with the 3D lifetime track; no independence credit is computed or
awarded; and no original prize problem is touched.
