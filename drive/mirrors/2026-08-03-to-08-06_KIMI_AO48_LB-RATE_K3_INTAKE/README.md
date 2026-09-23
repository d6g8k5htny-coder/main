# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-08-03-to-08-06 — KIMI + AO48 LB-RATE / K3 INTAKE` (adjudication chain and Phase-0 report layer)

Drive lane of the 2D LB-RATE lower campaign's August intake (468 inventory
items): the C030/C031 base cycle set, the Kimi LB-RATE campaign, the K3 swarm
partial delivery and the GP adjudication of all of it. **What this directory
holds is counted per lane in [`MIRRORS.md`](../../MIRRORS.md)**, generated
from these manifests by `tools/mirrors_index_check.py` and refused by CI if it drifts; every SHA-256 and
byte count in them equals its `drive/inventory.jsonl` row and is re-verified by
`tools/verify_manifests.py`. Until 2026-09-20 this paragraph gave a count of 22
raw files in 8 `_MANIFEST.jsonl` files in the present tense, which was the first
pass's figure and not the directory's; three further passes had landed by then.
The first pass mirrored the GP adjudication chain (`GP-LB-STAT-001`,
`-002`, `-003`, `-004`, `-004A`, `GP-LB-REC-001`, `-002`, `-005` v1.0/v1.1,
`-006`, `-007`, `GP-LB-WO-001`, `-002`, `GP-LB-ERR-001`), the K3 swarm Phase-0
report layer (`00_READ_FIRST.md`, `OPEN_OBLIGATIONS.md`, `FAILED_APPROACHES.md`,
`WP_DISPOSITION.md`, `CANONICAL_STATE.json`, `K3-AUD-001_residue_audit.md`),
`ERRATA_2026-08-05.md`, and the refuted assembly `K3-THM-001` itself. The port
lane that fetched them stopped before writing manifests; the orchestrator
re-hashed every file against the inventory on 2026-09-19 and wrote them.
**Mirroring is not review, replay, endorsement or adjudication.**

## The controlling disposition, verbatim

* `GP-LB-STAT-004` (the hostile adjudication of the K3 partial delivery), the
  paragraph under its heading `## CONTROLLING DISPOSITION`: "**REFUTED AS A
  THEOREM-GRADE OR VALIDLY STATED CONDITIONAL ASSEMBLY / LOWER CAMPAIGN OPEN.**"
  (until 2026-09-19 this README spliced heading and paragraph with a dash) … "This is an additive status
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
* The K3 swarm's own `00_READ_FIRST.md`, items 1 and 3, with the source's bold and
  inner quotation marks (until 2026-09-19 this README dropped the bold and turned the
  inner double quotes into single ones):
  > **The K3 blocking defect is CONFIRMED, numerically and symbolically.** The WP "certified upper bound" used the factor `p_grad · sqrt(E[det(H)^2 | grad=0]) · min(P_type, P_window)`. … Since p ≤ sqrt(p) on [0,1], the missing square root makes the reported quantity SMALLER — not a valid generic upper bound.
  >
  > **The 0.9144036 coefficient is a candidate measured/mixed-tier asymptotic anchor**, not a proved fixed theorem constant.

  (the source's hard line breaks are joined; "…" elides one sentence)
* `ERRATA_2026-08-05.md`: "The notice was independently verified and is
  CONFIRMED."
* The register: `registers/json/open_questions.json` OQ-016-U1 — "K3-THM-001
  REFUTED AS WRITTEN / LOWER HOLD"; the five P0 rows OQ-014 … OQ-016-U2 are
  transcribed verbatim in `docs/OPEN_PROBLEMS.md` §H, and `claims/graph.json`
  carries `K3-THM-001` as `REFUTED_AS_WRITTEN` on the LOWER2D track.

The Phase-0 reports and `K3-THM-001` carry Kimi's own words — "validity
PROVED", "CLOSED", "PASS", and the theorem file's Drive title, which begins
`K3-THM-001 - CONDITIONAL THEOREM Form C - liminf (1-q)∕r^3 >= 0.9666*c_Lambda`
(its body states the inequality as `\liminf_{r \to 0} \frac{1 - q(r, 6/5)}{r^3}
\;\ge\; \mathrm{AO}^0 \cdot c_\Lambda \;=\; 0.9666 \cdot c_\Lambda`; until
2026-09-19 this README gave a typographic rendering of the title inside quotation
marks as if it were a quotation) — which `GP-LB-STAT-004` §3 refutes; they are
mirrored **beside** the adjudication, never alone.

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
`AO48-AUD-061` is titled "KIMI-THM-023 proved 0.9144 r3 floor"; `AO48-AUD-064`'s verdict
sentence reads "VERDICT: WO-063 reconciliation CLOSED at record level with one material
amendment -- the quantitative WP upper-bound layer is DECERTIFIED (DEF-WP-CS-01, GP-flagged,
AO48-confirmed from landed source) and REOPENED as WO-064 Task A." (until 2026-09-19 this
README cut it after "record level" and called it the file's ending; it is neither);
`AO48-AUD-065`'s verdict paragraph opens "VERDICT: K3 intake RECONCILED. The swarm discharged
WO-064 in substance, minted the program's first conspicuously-conditional successor theorem
(K3-THM-001, Form C, mixed tier), closed the WP channel at exact-integrand modulus grade" (the
file's hard line breaks are joined here; its last sentence is "The program's rules -- freeze,
red-team, fail-closed, kill-equals-confirm -- ran at full scale and held."); `KIMI-AUD-020/021/022`
carry "APPROVE";
`DECISION_LEDGER.md` carries FROZEN / PASS / PROVED / CLOSED. Against all of it stands
`GP-LB-STAT-004`: "**REFUTED AS A THEOREM-GRADE OR VALIDLY STATED CONDITIONAL ASSEMBLY / LOWER
CAMPAIGN OPEN.**", and the register's OQ-016-U1 "K3-THM-001 REFUTED AS WRITTEN / LOWER HOLD".
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

## 2026-09-19 — third pass: independent re-verification of the second pass; no bytes added

Same Drive lane, folder id `1icu16HN0J9ifIQ7zAxvZVbGnU6wbVFk0`; the sub-folder ids are as listed
in the section above. This pass found the second pass's files already on disk (committed at
`b011cdd`) and re-verified them from scratch instead of re-fetching anything. It added no file and no manifest
row; this section is its only write.

What was recomputed here, from disk and from `drive/inventory.jsonl`, with nothing taken from
the section above on trust: all 85 rows of the 14 `_MANIFEST.jsonl` files carry the 14 declared
keys; the 53 `stored:true` rows are all `exact:true`, and each one's SHA-256 and byte count,
recomputed from the stored bytes, equals its inventory digest and byte count (629,828 bytes in
all, 353,266 of them from the second pass, against the 8,000,000-byte lane budget); the 32
`stored:false` rows all begin "tree-only: ", carry no bytes, and are exactly the 4 raw members of
`14_SIDE24_PRE_PEER_REVIEW_QUARANTINED` other than `ERRATA_2026-08-05.md` plus all 28
`DIRECT_NATIVE` objects of the lane (2 of which sit in that quarantined folder); no id is listed
twice; none of the lane's 468 ids appears in `drive/deltas/2026-09-18/PATH_CHANGES.jsonl` or
`CHANGED_SINCE_SNAPSHOT.jsonl`; none appears in `registers/json/frozen_objects.json`;
`python3 tools/verify_manifests.py` on this directory reports `manifests=14 verified=53
problems=0`. The three sha256sum-format files reproduce the comparison stated above
(`GP-LB-RECONCILIATION-MANIFEST.sha256` 5 entries / 4 mirrored / 4 agree; `MANIFEST.sha256` 266
/ 17 / 17; `INPUT_MANIFEST.sha256` 598 / 3 / 3; no disagreement). The seven in-file body seals
reproduce under the convention the AO48 session-state handoff states (`the sealed "body" = all
bytes strictly before the line containing "SHA-256 of this report body"`): the digest of those
bytes equals the 64-hex value on the line after the marker in every one of the seven files. The
banner sentences quoted above were located in the stored files they are attributed to; a
2026-09-19 re-verification found five of them not verbatim (a heading spliced to its paragraph,
two inner-quotation-mark substitutions, a typographic rendering of a title presented as a
quotation, and a verdict cut mid-sentence), and each was corrected in place the same day, with
the correction noted beside it. All of this is identity of bytes and of quotations; none of it
is review.

Objects in the folders this port order touches that neither pass stored, named here so the
omission is visible rather than silent (all remain fetchable by id; none is a hash mismatch):

* the seven `GP_FORENSIC_*` records beside `GP-LB-STAT-004` in `00_READ_FIRST_AND_GP_ADJUDICATION`
  (`GP_FORENSIC_extraction_record.json` `1agVhGqGk6R4g8TiWDgRQHoGDhkSlePIL` 154,905 B;
  `GP_FORENSIC_outer_inventory.csv` `1rEO7i5XQhFIsVCxwHkxA0tOqBnQkM54d` 154,917 B;
  `GP_FORENSIC_non_k3_meaningful.tsv` `1vutwXHja4bjaSKbuvWBkSO1tgk1Ztg3D` 74,107 B;
  `GP_FORENSIC_k3_classification.tsv` `1AG7ZdsbR0AcUTmYJsY0di8bUbv5SQUSl` 35,880 B;
  `GP_FORENSIC_outer_summary.json` `1ZP5qqsLQEZG3B_vqahpSFgG0gfYyW1Rn` 5,210 B;
  `GP_FORENSIC_key_k3_reports.tsv` `1PV0mnXFbZ45XTYU7dKGhTeGvirRbbW3c` 4,760 B;
  `GP_FORENSIC_phase0_research_carriers.tsv` `1smvY_6-GzueZ2E6GQI6VL4J78ZwOAd3-` 3,577 B;
  433,356 B together): GP's extraction and classification records of the K3 container, not the
  `GP-LB-STAT` / `-REC` / `-WO` / `-ERR` chain the port order names, so left for a later order;
* the sub-tree `MANIFEST.sha256` files under `01 — KIMI NORMALIZED EXPORT`
  (`1dPl3RRUQyMbGojK7DfqW1jVtPGoHULAF`), `04_W4_INDEPENDENT_ESTIMATE` (`1LjBYMWF4eS_5bLdKtyLvbKzmMl2yjRAJ`),
  `06_W6_UNIFORM_R_CONDITIONAL` (`1-sxjk4ORjW0FMaJcLK7G9dx1Eo3uuBsp`) and
  `09_W9_SCOPE_AUDITS/W9_scope_027a` (`1VH_VehD-3xzXO7-vosw2Id_WFILbnwpf`): the port order names
  only the top-level `01_PHASE0_AND_TOP_LEVEL_RAW` digest files, and the W-trees are outside it;
* `99_TRANSPORT_DEFECT_AND_UI_EVIDENCE/SUPERSEDED_TRANSPORT_PREVIEW__ERRATA_2026-08-05__U+FFFD-12.pdf`
  (`1lRzebef89hn7WQN_CTbv3HeQ2NyQPsaM`, 38,404 B): a `99_TRANSPORT_DEFECT` PDF, on the do-not-port
  list, and a superseded rendering in any case.

One fact about the second pass is recorded here because it is visible only on disk: its three
`.sha256` files are stored under the Drive title plus `.txt` (bytes unchanged, digests equal to
the inventory) for the reason its section states. Its 31 stored files and 63 manifest rows were
committed at `b011cdd` before this section was written; an earlier draft of this section said
they were uncommitted, which was wrong.

Reading copies (`exact:false`) are not the objects; there is none in this directory. PDF
renderings are not frozen bodies. The 28 `DIRECT_NATIVE` rows are convenience mirrors recorded
for the tree only. Nothing stored here is executed by CI, and this pass added no test, workflow
step or import.

### What this pass does not establish

Re-verification establishes that the bytes on disk are the bytes the 2026-09-17 inventory
declares and that the quotations above are verbatim; it establishes nothing about what those
bytes assert. No claim, premise or obligation is promoted, closed, discharged or reclassified;
the five validity premises of Theorem D1 v2.2(2) remain OPEN; `D3-LEMMA-RN-UNIF` is not closed;
the lower campaign (LB-RATE / K3) is OPEN under `GP-LB-STAT-004` ("**REFUTED AS A THEOREM-GRADE
OR VALIDLY STATED CONDITIONAL ASSEMBLY / LOWER CAMPAIGN OPEN.**"); the AO48-layer "proved" /
"CLOSED" / "APPROVE" / "RECONCILED" words remain the layer the register does not follow; the
seven reproduced body seals are identity of bodies, not review of them, and none is a
register-declared frozen-body digest; nothing here composes with the 3D lifetime track; no
independence credit is computed or awarded; no original prize problem is touched; and no gate
moves.

## 2026-09-20 — fourth pass: the workstream trees, the base cycle, and every remaining digest-bearing object

`drive/MIRRORS.md` counted this lane at 53 objects held of the 468 the inventory
gives it, with **330 digest-bearing items in neither column** — the largest such
gap in the repository. This pass closes the part of it that can be closed:
**311 raw files byte-exact** across 38 directories, and a tree-only row for
each of the 19 bulk objects left, so no digest-bearing item of this lane is
unaccounted for.

What arrives is the K3 swarm's own working trees, which no earlier pass named:
W2 symbolic, W3 numerics, W4 independent estimate, W5 forensic, W6 uniform-r
conditional, W7 gamma-loc minimality, W8 lambda non-closure, W9 scope audits,
W10 composition, W11 methods, W12 red team and W13 reproduction — together with
the C030/C031 base cycle the campaign starts from, the remaining
`03_KIMI_DERIVATIONS_AND_THEOREMS` carriers, the `01 — KIMI NORMALIZED EXPORT`,
`02 — OUTER-IDENTITY CROSSWALK`, `02 — CERTIFICATES AND TRANSCRIPTS`,
`03 — INCOMPLETE NONCONTROLLING DRAFTS` and `05 — EXPLORATORY RUNG BATCHES`
sub-folders, and the `99_TRANSPORT_DEFECT_AND_UI_EVIDENCE` previews.

**The controlling disposition is unchanged by any of it.** `GP-LB-STAT-004`
still governs every label in this lane, the lower campaign is still open, and
holding a workstream's files is not holding its conclusions. Several of the
folder names carry the source's own verdict on its contents —
`03_W3_NUMERICS_INCOMPLETE`, `13_W13_REPRODUCTION_STALE_SNAPSHOT`,
`03 — INCOMPLETE NONCONTROLLING DRAFTS` — and those words are the source's, kept
because removing them would make the tree read better than the thing it copies.
W9's scope audits and W7's minimality work include `mutation_workspace`
directories: a mutation file is a deliberately altered copy kept to show that a
check can fail, and it is not a result of anything.

### How the bytes got here

Each file was fetched through the Drive connector, and the base64 it returned
was decoded to disk by a script reading it back out of the session transcript,
or, for the files large enough that the harness spilled the tool result to disk
instead of inlining it, out of that file. Either way no model retyped a byte. A
file was written **only** when its SHA-256 and byte count already equalled the
ones `drive/inventory.jsonl` declares for that Drive id; a mismatch writes
nothing. Every stored file was then re-hashed from disk a second time by the
process that wrote the manifest rows, so the digest in a row below is computed
here and not copied from the fetch. Across all 311 files the decoder reported zero mismatches, zero ambiguities and nothing missing.

Six objects are stored under their Drive title plus `.txt`:
`MANIFEST.sha256` and its siblings are the **source's** sha256sum manifests of
the **source's** tree, and under their own names `tools/verify_manifests.py`
reads them as manifests of *this* repository and reports every file they list
that this repository does not hold. The bytes are unchanged and their digests
are the inventory's. This is the convention this lane's earlier pass had already
set; this pass had to rediscover it, because the verifier said so.

### Not stored, and why

The 19 objects left are bulk: two wedge tables of 13.5 MB and 13.2 MB, a
7.0 MB zip, four more tables over 1 MB, the administrative UI evidence PNG, the
forensic inventories and the concatenated Kimi exports. Their rows are tree-only
and carry the inventory's own digest and byte count, so a later pass can fetch
and prove them; nothing here claims to hold them, and no row computes a digest
for a file this repository does not have. The store limit this pass applied is
65,536 bytes, chosen because the lane's bulk is machine-generated scan output
and its documents are not.

The SIDE24 pre-peer manuscript and the other objects the earlier passes recorded
tree-only stay tree-only; `14_SIDE24_PRE_PEER_REVIEW_QUARANTINED` was not
opened, and neither was `99_DO_NOT_OPEN`. No object in this pass appears on any
list in `quarantine/EXCLUSIONS.json` — `tools/quarantine_check.py` is what
enforces that, not this sentence.

Until 2026-09-20 the **Deliberately not ported** section above listed the
W1–W13 workstream trees, the C030/C031 base-cycle files and those five
sub-folders, and gave as the reason that they were "not named in this pass's
port order". That was a statement about the third pass's scope and not a
prohibition, and this pass ports them.

### In-file seals

The lane's files declare digests of each other, and now that the bytes are here
those declarations are checkable against them. **Seventy-two reproduce and one
does not.** The reproducing ones are 69 lines of the source's own sha256sum
manifests, two digests a document declares for a sibling file, and one
body-seal digest over a file's own text below its marker line.

Until 2026-09-20 this paragraph gave sixty-four and 61. The checker that
produced those numbers under-counted: it resolved each declared path by its last
component, so a line naming a file inside a subdirectory was looked up as a
same-named file beside the manifest. On this lane that only lost matches. On the
canon lane the same bug invented nine mismatches that do not exist, which is how
it was found; the path is now resolved relative to the manifest that declares it,
and the one real mismatch below is unaffected because its line names a bare
filename in the manifest's own directory.

The one that does not is in
`04_W4_INDEPENDENT_ESTIMATE/MANIFEST.sha256`, which declares `probes_laws.txt`
at `3cb2a8af0ad6…`. The file this repository holds for that Drive id hashes to
`2f9d27d4bffa…` at 3,991 bytes, which is exactly the digest and byte count the
2026-09-17 inventory declares for it. So the copy here is the Drive object, and
it is the W4 bundle's own manifest that disagrees with the Drive. That id is in
neither `PATH_CHANGES.jsonl` nor `CHANGED_SINCE_SNAPSHOT.jsonl`, so the object
did not move or change between the snapshot and this port. Which of the two is
right is the source's question. It is **recorded here and not resolved**, and
nothing was altered to make them agree.

A further 1,136 lines of those manifests, and two of the in-document sibling
digests, name files this repository does not hold. That is not a defect and not
a mismatch: they are the source's manifests of the source's tree, and this
repository holds part of that tree. Only a line naming a file that is here can
reproduce or fail, and every such line is counted above.

### What this pass does not establish

Nothing about the mathematics. Every number above counts files, bytes and
digests. A byte-exact copy says these are the bytes the 2026-09-17 inventory
declares for that Drive id and says nothing about whether the document is
correct, current or authoritative — and in this lane the adjudication that
governs them all found the assembly they support refuted. Mirroring a
workstream is not reviewing it, replaying it, endorsing it or adjudicating it.
No status label moved, no obligation was discharged, no gate moved, and the
imperative text these files address to other threads is quoted data, not
instructions followed here.
