# Issue #309 reconciled fact sheet: Evidence Role dropdown (GP-REG-032, "Evidence Lineage" column F)

I checked everything below myself, read-only. Scripts were run with `python3 -I` from /tmp and live in `$SCRATCH/i309/work/critic/`:

| Script | SHA-256 |
|---|---|
| `critic_xlsx.py` | `58de77ac…` |
| `critic_xlsx_b.py` (boolean-typed variant) | `b8669c12…` |
| `critic_csv_md.py` | `08ca7fa6…` |
| `critic_diff.py` | `83dd4beb…` |
| `critic_rows.py` | `34019e68…` |
| `critic_all_dv.py` | `63c46bbc…` |
| `critic_misc.py` | `1d4d36fb…` |

Outputs are `xlsx_out.json` (`28f25b57…`) and `xlsx_out_b.json` (`9aaa47bd…`).

All `a01c72f1` paths below are at `a01c72f1937875b94bfd78e183a6dbbdb2b04da4`. For every parsed export, `git hash-object` of the extracted copy equals the blob id. `origin/main` is `157b6a9753fce110bd0769418b10e201720d24ad`.

## 0. Reconciliation of the five agent reports

**Counts: parser-A and parser-B agree on every count in every file, and I got the same values myself.** Their actual differences, with the value I re-derived:

| # | Item | parser-A | parser-B / other agent | Re-derived value (raw XML) |
|---|---|---|---|---|
| D1 | definedNames in the workbook | "The workbook has no definedNames" | governing-sweep: 26; B: not stated | **26 in each of the three xlsx files.** 23 are `_xlnm._FilterDatabase` and 3 are `Z_…_.wvu.FilterData`. The one on this sheet is `_xlnm._FilterDatabase` localSheetId=21 = `'Evidence Lineage'!$A$1:$P$2000`. None is a role list. **parser-A is wrong**, but no count changes. |
| D2 | Byte span of "the validation" | 378983..379311 | B: the same | That span is the `<dataValidations>` wrapper. The `<dataValidation>` element itself is at [379000, 379293) in 09-18, [386745, 387038) in R1 and [386718, 387011) in R2. |
| D3 | Top-10 outside labels | 8-way tie at count 3 | B breaks the tie by first appearance | This is only a tie-break convention. Ranks 9-16 tie at 3; report the tie, not a cut. |
| D4 | Data extent | N=517 / 529 | B scans to `max_row`=2001 | The method differs but the counts are the same. There are 2001 `<row>` elements and 2001 F `<c>` elements, of which 486 have `<v>` (header plus 485). |

**Cross-checks that hold:**
- R1 versus 09-18: 192 cells differ, all in rows 518-529.
- R2 versus R1: 0 cells differ in A:P.
- The row/F fingerprints match parser-A: `d9731cb3…` for 09-18 and `34835a26…` for R1 and R2.
- The CSV equals the 485 non-empty 09-18 rows across A:P with 0 mismatches, but only once boolean cells (`t="b"`) are typed as TRUE/FALSE. With raw `<v>` values of 1/0, 748 G/H cells would differ.

**Corrections to the other agents:**
- **Governing-sweep's row numbers are CSV line numbers, not sheet rows.**
  - "rows 349 (EV-OPPROT015) and 355 (EV-OPPROT016)" are sheet rows **F372 and F378**.
  - "EXACT_DERIVATION row 52" is sheet row **F53** (EV-P02-DM2-071, G=TRUE).
- **Governing-sweep, Run Log:** "Run Log rows 2 and 11 say 'Automation not yet installed'" is only half right. Only `'Run Log'!G2` says that. G11 says "Apps Script runtime, OAuth grants, first live refresh, and hourly triggers were not installed by the Drive connector; those remain deployment steps".
- **Consumers, line numbers:** in `registers/collision_proposal_2026-09-19.json` the role strings are at lines **1461**, 1481, **1503**, 1523, **1555-1556** and 1605, not 1460/1502/1554.
- **Consumers, row numbers:** "rows 385/386" and "row 334" there, and "Evidence Lineage rows 385 and 386" in `registers/KNOWN_FINDINGS.json` line 32, are **0-based `registers/json` row indices**. `tools/registers_check.py` lines 125-130 use `enumerate(t["rows"])`.
  - Indices 385/386 are sheet rows **410/411** (CSV lines 387/388), both EV-LS-REQ030.
  - Index 334 is sheet row **359**.
- **Three row-numbering systems are in use** (sheet row, CSV line, 0-based JSON index). Any data note must say which one it uses.

## 1. Counts table

The role list is the 11 options. "In list" means an exact, case-sensitive match after stripping.
- No value changes when stripped.
- No case-insensitive match exists outside the list.
- No formulas are present.

| Source (identity) | Rows with any A:F | Populated F | In list | Outside | Distinct | Distinct outside | Blank F on populated row | Last data row |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `registers/source/GP-REG-032_v1.2_export_2026-09-18.xlsx` (blob `e45ed49d…`, sha256 `c3229ece…a460`, 1,919,741 B) | 485 | 485 | 56 | 429 | 354 | 347 | 0 | 517 |
| `…_2026-09-23_R1.xlsx` (blob `e11a7876…`, sha256 `6cec54ce…1721469`, 1,979,318 B) | 497 | 497 | 67 | 430 | 355 | 348 | 0 | 529 |
| `…_2026-09-23_R2.xlsx` (blob `cc79f573…`, sha256 `97940217…bef9db`, 1,985,161 B) | 497 | 497 | 67 | 430 | 355 | 348 | 0 | 529 |
| `registers/csv/evidence_lineage.csv` (blob `cb1a1545…`, sha256 `b7597cbf…705d`; 486 CRLF lines, no BOM, all 16 fields) | 485 | 485 | 56 | 429 | 354 | 347 | 0 | n/a |
| `registers/source/GP-REG-032_v1.2_export_2026-09-17.md` (blob `4a24f53a…`, sha256 `57d9078b…8b71`) | 137 | 137 | 19 | 118 | 103 | n/a | 0 | prefix only |

**Per-option use:**

| Option | 09-18 and CSV | R1 and R2 | md prefix |
|---|---:|---:|---:|
| PRIMARY_DERIVATION | 7 | 18 | 4 |
| EXECUTABLE_RECEIPT | 14 | 14 | 9 |
| RAW_DATA | 17 | 17 | 0 |
| FORMAL_PROOF | 7 | 7 | 1 |
| INDEPENDENT_RECONSTRUCTION | 5 | 5 | 4 |
| SUMMARY | 2 | 2 | 0 |
| GOVERNANCE_PRIMARY | 4 | 4 | 1 |
| NAVIGATION, BALLOT, REGISTER_COPY, REVIEW_OF_REVIEW | 0 each | 0 each | 0 each |

Those last four are also absent as exact strings from `xl/sharedStrings.xml` in all three workbooks.

**Layout and distribution, 09-18:**
- 31 rows inside the data range are empty in all columns: 51, 85-100, 212-215, 307, 314, 423, 424, 430, 431, 434-437. No row is hidden.
- In-list rows: 2-8, 10, 13, 20, 26, 30, 33, 36, 41, 50, 59, 83, 153, 191, 196, 263, 372, 377, 378, 447, 448, 477-505.
- Outside-label frequencies: one label at 10, one at 8, one at 7, three at 6, one at 5, one at 4, eight at 3, 22 at 2, and 309 singletons.
- Top outside labels: CONTROL_PLANE_REPAIR_RECEIPT 10, INDEPENDENT_REVIEW_REQUEST 8, INDEPENDENT_REVIEW 7, then INSTRUMENT_SOURCE, SAME_LINE_ADVERSARIAL_AUDIT and SAME_LINE_HOSTILE_AUDIT at 6 each.
- Two values contain U+2013: F188 and F464.
- Column A repeats one key: EV-LS-REQ030 appears at both F410 and F411.

**What R1 changes (R2 is identical to R1):**
- It appends rows 518-529.
- F518:F528 are all PRIMARY_DERIVATION.
- F529 is SOURCE_ADMISSION_REPAIR, which is outside the list.

**The CSV** equals the 09-18 import (SOURCES.json line 10 names 09-18 as `current_import_source`). It does not contain R1's 12 rows; SOURCES.json lines 91 and 113 say R1 and R2 are "NOT YET" the import source.

**The md file** is a truncated rendering: lines 1780-1933 cover sheet rows 2-155, with 0 F mismatches. SOURCES.json lines 25 and 47-50 record `TRUNCATED_RENDERING` and `evidence_lineage` prefix_rows 137 of total_rows 485.

**GOVERNANCE_PRIMARY rows (09-18):**

| Row | Evidence ID | Recorded UTC | G (Primary Evidence) |
|---|---|---|---|
| F2 | EV-CAI-GUIDANCE | 2026-07-21T15:40:00Z | FALSE |
| F372 | EV-OPPROT015 | — | TRUE |
| F377 | EV-OPPROT015-V11 | — | FALSE |
| F378 | EV-OPPROT016 | — | TRUE |

**Column G is not type-uniform:** 286 boolean-true, 88 boolean-false, 76 string "TRUE", 28 string "FALSE", and 7 prose values (for example "YES — …").

## 2. Validation configuration per export

**Sheet location (all three workbooks):** `xl/workbook.xml` has `<sheet name="Evidence Lineage" sheetId="22" r:id="rId26">`, and `xl/_rels/workbook.xml.rels` points rId26 to `worksheets/sheet22.xml`.

| Export | sheet22.xml SHA-256 | Bytes |
|---|---|---:|
| 09-18 | `9d4c4aaec6f66aad7114f97e5528ecc4fe3657163da8c480e159ff8b7dd53140` (matches the issue) | 397,403 |
| R1 | `a83817229abfcf7a3c32385e1ea239c8601d774e532773289329da7129169b66` | 405,185 |
| R2 | `b0fba3791673160bba5d970cc9cee44c8bb963cbc2374d9ef83cc5ffd2c482e2` | 405,158 |

**The validation element** is byte-identical in all three files and is the only one in the sheet:

```
<dataValidation type="list" allowBlank="1" showErrorMessage="1" sqref="F2:F2001"><formula1>&quot;PRIMARY_DERIVATION,EXECUTABLE_RECEIPT,RAW_DATA,FORMAL_PROOF,INDEPENDENT_RECONSTRUCTION,SUMMARY,NAVIGATION,BALLOT,REGISTER_COPY,REVIEW_OF_REVIEW,GOVERNANCE_PRIMARY&quot;</formula1></dataValidation>
```

**Attributes not present:**
- No operator, formula2, showInputMessage, showDropDown or errorStyle (the OOXML default errorStyle is stop).
- No prompt, promptTitle, error or errorTitle text.

**Everything else around it:**
- **List source:** an inline literal, not a range or a defined name.
- **No extension validations:** no `extLst` and no x14 validations in the sheet, and 0 x14 validations workbook-wide.
- **No other sheet controls:** no conditional formatting, no `<sheetProtection>` and no `<dimension>`.
- **Filter:** `autoFilter ref="$A$1:$P$2000"`. Note it ends at row 2000 while the validation ends at 2001.
- **Sheet relationships:** only one drawing (`drawing22.xml`, with 0 anchors) and hyperlinks (492 in 09-18, 493 in R1 and R2). There is no comment or VML relationship on sheet22.
- **docProps:** none.

**Coverage:** every populated F cell is inside F2:F2001; there is none below row 2 or after row 2001. F594 is empty in every export. The issue's live F594 coverage gap cannot be tested from these files.

**Workbook-wide (09-18):** there are 9 `<dataValidation>` elements. Only sheet22's uses the role list.

| Range | Options | Populated | In | Out | Distinct | Attributes |
|---|---:|---:|---:|---:|---:|---|
| Review Queue!D2:D26 | 7 | 25 | 25 | 0 | 4 | allowBlank only (**no showErrorMessage**) |
| Transition Register!D2:D1000 | 2 | 82 | 82 | 0 | 2 | allowBlank, showErrorMessage |
| Transition Register!AJ2:AJ1000 | 5 | 82 | 3 | 79 | 78 | same |
| Review Independence!R2:R1000 | 4 | 134 | 101 | 33 | 13 | same |
| Evidence Lineage!F2:F2001 | 11 | 485 | 56 | 429 | 354 | same |
| Global Object Audit!G2:G500 | 3 | 29 | 5 | 24 | 22 | same |
| Operator Decisions!E2:E500 | 11 | 45 | 13 | 32 | 34 | same |
| P0.2 Review Manifest!P6:P100 | 2 | 11 | 11 | 0 | 2 | same |
| P0.2 Review Manifest!O6:O100 | 5 | 11 | 10 | 1 | 4 | same |

**Native Google Sheets readbacks (data, not governance):**
- `drive/deltas/2026-09-20/DG-RECON-20260920/payloads/1Z7UImHcliMI_iSU2jEfZjYu73VUrVDra.json` (blob `c7c3571f…`, sha256 `9df62ddb…eec`), lines 457-530: row 518 has `ONE_OF_LIST` with the same 11 values in the same order, `"strict": true` and `"showCustomUi": true`.
- `drive/search/2026-09-20/deliveries/RECON_RN_REGISTRATION_20260920_v1.json` (blob `da4937fa…`, sha256 `540bf83b…081a`), lines 640-678: the same rule for EV-DG-RECON-RN-MATH-20260920-v1, which is R1 row 519.

**The issue body** states the same frozen figures: 1,919,741 B, sha `c3229ece…`, member sha `9d4c4aae…`, F2:F2001, 11 options, allowBlank=1, showErrorMessage=1, 485/56/429. I fetched it via `gh api` (issue JSON sha256 `ca30c8ff…`, body sha256 `b3dd63d7…`, 6 comments).

## 3. Governing declarations

**Bottom line: none of the sources I opened declares the 11-option list authoritative, stale, legacy, closed, to be expanded or undecided.** The only explicit statement on its status is a non-decision (see "None" at the end of this section).

**NORMATIVE, but scoped to primary-evidence admissibility or to other things:**

1. **Source Manifest:** `drive/mirrors/14_COORDINATION_AUTOMATION_SPINE/04.4_MANIFEST_STAGING/01_AUTOMATION_CONTRACTS_AND_SOURCE/01_V1_2_ARCHITECTURE_CONTRACT_AND_MANIFEST/GP-AUTO-034-v1.2 — Source Manifest and Critical Patch Review.export.txt` (blob `528f20b7…`, sha256 `0108b667…e8cc`). The `_MANIFEST.jsonl` row marks it `exact:false`, "READING COPY, NOT THE OBJECT".
   - Line 121: "primary_evidence_ids = Evidence Lineage".
   - Lines 144-147, "PRIMARY-EVIDENCE RULE": "…assigned SUMMARY, NAVIGATION, BALLOT, REGISTER_COPY, or REVIEW_OF_REVIEW roles, cannot satisfy primary evidence. Admissible primary roles are PRIMARY_DERIVATION, EXECUTABLE_RECEIPT, RAW_DATA, FORMAL_PROOF, and INDEPENDENT_RECONSTRUCTION."
   - **Verified textual fact:** dropdown options 1-10 are exactly these 10 tokens, in this line's order (the five admissible roles, then the five excluded ones). GOVERNANCE_PRIMARY is appended at position 11 and appears nowhere in this text.
   - It does not say the role list is closed.
2. **Operator approval:** `registers/csv/operator_decisions.csv` lines 2-3 (blob `b6948b03…`, sha256 `6b14b738…a618`), which are `'Operator Decisions'!A2:A3` in 09-18.
   - OD-CAI-001: AUTHORITY, "Dylan-approved governance and coordination architecture", scope "GP-AUTO-034-v1.2-CAI only; no mathematical truth authority".
   - OD-CAI-002: CANONICAL_IMPACT APPROVED, "Approves architecture as the governing coordination system".
   - Both bind Drive id `1VgVzVTz…`, which is the Source Manifest's id. They approve the architecture, not a vocabulary.
3. **Architecture Contract:** `…/GP-AUTO-034-v1.2 — Coupled Advancement Installation and Architecture Contract.export.txt` (blob `e0e38f17…`, sha256 `9342f4aa…83e3`).
   - Line 27: "evidence role … come[s] from their dedicated registers".
   - Line 78: "Summaries, ballots, navigation artifacts, register copies, and review-of-review documents cannot be primary evidence."
4. **DEF-008:** `registers/csv/definitions.csv` line 9 = `'Definition Registry'!A9` (blob `a8820175…`, sha256 `a1d08483…f144`), status ACTIVE.
   - It defines primary evidence. E9: "Secondary descriptions and governance outcomes are excluded from the primary-evidence role."
   - It says nothing about the list.
   - **Tension (interpretive, not verified as a defect):** F372 and F378 carry GOVERNANCE_PRIMARY with G=TRUE. GOVERNANCE_PRIMARY is also not among line 147's admissible primary roles.
5. **Config row 66:** `registers/csv/automation_config.csv` line 66 = `'Config'!A66` (blob `d2f2935c…`). EVIDENCE_LINEAGE_SHEET … "Summaries cannot be primary evidence", "Required 16-field schema". It is about the column count only.
   - Note: CSV line 217 is not sheet row 217. `'Config'!A217` is FROZEN_OBJECTS_SHEET.
6. **OP-GDN-002:** `governance/protocols/OP-GDN-002.md` line 40 (blob `984389f0…`, sha256 `994c2198…`), with the Drive mirror's line 52 (blob `cc8d3845…`) worded the same: "Primary derivations, executable receipts, and raw evidence must remain identifiable." No list.
7. **R17:** `governance/protocols/OP-PROT-019-v1.1_R17.md` line 43 (blob `c08a3c10…`) gives a closed set, but for Review Queue technical status, a different column.
8. **Where authority lives:**
   - `AGENTS.md` (a01, blob `7aa52775…`) line 12: "Drive is the governing record."; line 17: "GitHub is the execution workspace, not a competing governing register."
   - `CLAUDE.md` (a01, blob `810d7f66…`) lines 125-128: "Exports are regenerated, never hand-edited … A defect in the source is repaired at source."

**DESCRIPTIVE (three items downgraded from NORMATIVE):**
- Architecture Contract line 45: "Evidence Lineage distinguishes primary derivations, … review-of-review artifacts." These are 10 categories in prose; GOVERNANCE_PRIMARY is absent.
- **Cross-Model Review Ledger, CM-046 (downgraded):** `governance/protocols/history/Cross-Model Review Ledger v1.0 — 2026-07-20.export.txt` (blob `501c4c2b…`, sha256 `e282d09a…`; the same blob is in the drive mirror). It is an implementation record in a `history/` file.
  - Line 2672: canonical impact "APPROVED FOR GOVERNANCE AND COORDINATION ONLY".
  - Line 2688: "Primary-evidence role … come from their dedicated registers".
  - Line 2695: "Evidence Lineage separating primary evidence from summaries, ballots, navigation copies, register copies, and review-of-review artifacts."
- **GP-AUTO-034 v1.3 drafts R0.1-R0.4 (downgraded):** these are non-governing draft code. R0.4 is blob `ec03e02d…`, sha256 `11ba6ccd…`.
  - Lines 4-5 mark the status "… DO-NOT-INSTALL".
  - Lines 2162-2165 hold `PRIMARY_EVIDENCE_ROLES` = the 5 admissible roles plus EXACT_DERIVATION and PRIMARY_NUMERICAL_SOURCE.
  - Lines 2503-2507 only test membership in that primary set.
  - Earlier drafts have the same list: R0.3 `9afc9f39…` lines 2154-2156, R0.2 `6ba54268…` lines 2142-2144, R0.1 `846c264c…` lines 2170-2172.
  - No draft has GOVERNANCE_PRIMARY, REGISTER_COPY, REVIEW_OF_REVIEW or BALLOT. NAVIGATION appears only as folder and document identifiers.
  - EXACT_DERIVATION is used at F53; PRIMARY_NUMERICAL_SOURCE in no row.
- **Live Registers leaf card:** `drive/mirrors/14_COORDINATION_AUTOMATION_SPINE/04.1_LIVE_REGISTERS/00_LEAF_CARD — 04.1 Live Registers (CL-NAV-036.12).export.txt` (blob `843ae52f…`). Line 2: "AUTHORITY: none"; line 12 describes the sheet.
- **`registers/README.md`** (a01, blob `5ab722d4…`, sha256 `97985dcc…`):
  - Line 259 says "GP-AUTO-034 was never installed or run". This is the README's own reading of the Run Log text; Run Log G2/G11 are quoted in section 0.
  - Line 266 describes the tab.
  - Lines 290-310 list the CI invariants; none concerns Evidence Role.
  - Line 314: "A register row is a transcription of the Sheet, not a verdict."
- **Metadata Schema:** `registers/csv/metadata_schema.csv` line 59 (blob `c2d64baf…`) concerns the artifact-card field `primary_evidence_ids`. No row of that file mentions "role", so no schema row describes the Evidence Lineage columns.
- **Transition Register:** `'Transition Register'!AE3` (09-18) requires "evidence roles" to exist but gives no vocabulary.
- **Current main:** `origin/main:docs/WORKSPACE.md` (blob `1da250d1…`) line 28 calls the a01 branch "legacy … frozen"; lines 51-53 say the registers "provide research memory and Work Events".
- **Executable never recovered:** GP-REQ-171 (blob `93e30251…`) lines 19-26 still request recovery of the v1.2 executable, `GP-AUTO-034-v1.2_COUPLED_ADVANCEMENT.gs`, sha256 `1062a63c…`.

**None:**
- **No text defines GOVERNANCE_PRIMARY.** `git grep -i 'governance[ _-]primary'` at a01 finds it only in 4 data files: the two readback JSONs, `registers/csv/evidence_lineage.csv` and `registers/json/evidence_lineage.json`. It finds nothing on origin/main.
- **The only explicit status statement is a comment, and it is a non-decision:** issue #309 comment 6061785988 (GROK-H04, 2026-10-08T14:13:25Z, body sha256 `7fb07263…`).
  - "The 11-option dropdown is **not adopted** as the authoritative Evidence Role enum by this note." "Contract status: NOT_ESTABLISHED."
  - "A strict consumer remains blocked until a separate explicit decision …"

## 4. Consumers (verified)

**No code at a01 or on origin/main reads Evidence Role by name, checks it against the list, normalizes it or branches on it.**

- **`tools/registers_check.py`** (blob `02edf165…`, sha256 `1fa5443c…`): 0 matches for "role".
  - Lines 61-65: the value sets cover only R17 statuses and quarantine classes.
  - Lines 66-73: `evidence_lineage` is in KEYED (column 0 must be unique).
  - Lines 125-136: duplicate-key and review_queue checks.
- **`tools/registers_import.py`** (blob `9cce441c…`, sha256 `dd0668dc…`): 0 matches for "validation" or "role". Line 369 reads `m:sheetData` only.
- **`registers/CONSUMERS.json`** (blob `cfbd8d9d…`, sha256 `ed5980ea…`): lines 219-234 class `evidence_lineage` as MACHINE with 6 machine and 3 prose consumers. Generic files are at lines 51-56. Its own `does_not_establish` field says "Nothing here reads, checks or grades any register cell".
- **Other files with 0 "role" matches:** `tools/collision_proposal_check.py` (`5ddad6c6…`), `tests/test_collision_proposal.py` (`7b9632ec…`), `tests/test_register_source_selection.py` (`4134f69f…`), `tests/test_registers.py` (`a8542a11…`) and `tools/consumers_check.py` (`cded96cc…`).
- **The test lets R1's out-of-list row through:** `test_register_source_selection.py` line 25 expects `evidence_lineage: 12` added rows (asserted at line 162). Lines 186-188 compare the full problem set, so R1's out-of-list F529 is not a problem the checker reports.
- **Recorded refusals agree:** SOURCES.json line 96 lists the checkers that refused R1; no refusal concerns Evidence Role.
- **CI** (`.github/workflows/ci.yml`, blob `f3b71fa7…`) runs the importer (`--check`, line 27), the checker (line 29), the collision check (lines 57-61) and the consumers check (line 63).
- **origin/main:** outside `incoming/`, the role terms match only the archived log `audits/dropbox_import/2026-09-27/files/Github/master-final-v1/master-final-v1/02.log` (blob `d5cb5e4e…`), line 14, a duplicate-key line. `incoming/` has 0 matches, and there are no spreadsheets on origin/main.

## 5. De facto uses since 8 October

These are agents' self-reports in GitHub comments. I opened 10 of the 37 the de-facto agent cited (bodies are `c<id>.json` in my work folder) and checked the quoted text.

**Verified: comments that treat the list as binding (none cites #309):**
- **309/6055621753** (2026-10-08T08:10:01Z): "The new receipt uses EXECUTABLE_RECEIPT; its attached list admits that role."
- **307/6112183920** (10-11): "F categories outside that strict enum" for row768 HOSTED_BEHAVIORAL_FAILURE_ORIGINALS and row772; "EXECUTABLE_RECEIPT is a candidate only"; row774 "uses admitted PRIMARY_DERIVATION".
- **307/6112344525** (10-11): F777 HOSTED_TEST_ORIGINALS "while that same cell's strict enum excludes it"; proposes EXECUTABLE_RECEIPT or RAW_DATA.
- **307/6112411510** (10-11): rows 768/772/777 "use inadmissible labels"; plans REGISTER_COPY correction rows, with RAW_DATA as the originals' role.
- **307/6112550048** (10-11): "Append-only role corrections EL781/782/783 plusWE1758 preserve invalid historical EL768/772/777 while assigning current original roleRAW_DATA and correction roleREGISTER_COPY".
- **275/6206016655** (C284, 2026-10-16T01:35:17Z, 25,694 B, sha256 `5d955d0d…effa9`):
  - "some strings did not satisfy the intended types and the existing role dropdown"; "F Evidence Role is the existing allowed REGISTER_COPY".
  - "No schema, allowlist, formula, scientific grade or general workflow was changed"; "F still has the same strict11-role validation".
  - Its write requests target sheetId 2015177386, rowIndex 829-831 (F830:F832), column 5.
  - Its JSON has 6 `ONE_OF_LIST` blocks with the same 11 values, each `strict:true` and `showCustomUi:true`.
  - It contains 0 mentions of 309.

**Verified: comments that treat the list as not binding or undecided:**
- 309/6061785988 (H04), quoted in section 3.
- **275/6055622308** (10-08): "do not infer taxonomy adoption from this thread, a green historical checker, or silence."
- **419/6215221806** (10-16): "main#309 — only a scoped routing audit completed (6062809044); the taxonomy clarification is not done."
- **309/6305395660** (10-21): this workflow's own read-only pickup.

**What this shows:**
- Out-of-list strings were written into cells whose native rule is reported as `strict:true` (rows 768, 772 and 777, and the original F830:F832 values). The write path that allowed this is not stated anywhere.
- No record gives the live sheet's total row count or current validation range after the A1:F692 snapshot.

## 6. Open unknowns and challenges a skeptical reader would raise

**Unknowns:**
- **Live sheet, unread by rule:** current row count, current validation ranges, and the F594 gap. The issue's 659/182/476/1/31 and 395-distinct counts, and its claim that all 485 historical values match the live rows, are unverified here.
- **Origin of GOVERNANCE_PRIMARY:** no text defines it. Its first row, F2, was recorded at 2026-07-21T15:40:00Z, the same `Recorded UTC` as OD-CAI-001/002. CM-046 is dated 15:45 UTC. That the option was added during the CAI implementation is an inference only.
- **The v1.2 executable** (`.gs`, sha256 `1062a63c…`) is unrecovered. It is the most likely place where the dropdown was defined, and it is unread.
- **Enforcement mechanics:** it is not established whether the exported `showErrorMessage=1` corresponds to native `strict:true` (only rows 518, 519 and 830-832 have native readbacks), or how 429-476 out-of-list values coexist with a strict rule (API writes, paste, or the rule being added after the values).
- **Edit history:** who added or changed the validation, and when. No revision history was read.

**Challenges:**
- **Row numbering:** three systems are in use (sheet row, CSV line, 0-based JSON index), and several agent reports mixed them up (section 0).
- **Strictness does not explain conformance:** the only fully conformant list, Review Queue!D2:D26, has no showErrorMessage, while Evidence Lineage has it and is 88% non-conformant.
- **The closest rule is about primary evidence:** it is an admissibility rule from which the first 10 options are taken token for token, not a declared full role taxonomy. It appears only in reading copies (`exact:false`) of native Docs. The operator approval covers "GP-AUTO-034-v1.2-CAI only", an architecture, not a list.
- **Practice is not adoption:** the binding treatment since 11 October (corrections to REGISTER_COPY and RAW_DATA) never cites #309 or H04, and H04 and C184 both say adoption must not be inferred from practice or silence.
- **Unused options:** 4 of the 11 options are unused in the frozen data. REGISTER_COPY's first use is reported only in live-sheet comments.
- **The CSV and JSON are the 09-18 import:** the registers do not contain R1/R2's 12 rows.

## 7. Not read by me

- The live Google Sheet and all Drive or private data.
- Other branches, the Math- repo, and PR review comments.
- The 27 cited de-facto comments I did not open, including 307/6069528800, 6086353524, 6112031335, 6112557783, 6112811005-6113685231, 6060611750, 6115443907 and 6136820683; 419/6206044524, 6292075720 and 6305398393; and 309/6062763924 and 6062809044.
- Comments created before 8 October.
- The xlsx parts for styles, theme and sheetViews, and other sheets' comments.
- Mirror files beyond the lines cited.
- The 234 binary files at a01 that the text search skips.
- `docs/FINDINGS_2026-09-18.md`, `docs/OPEN_PROBLEMS.md` and `registers/COLLISION_PROPOSAL_2026-09-19.md` (prose consumers) in full.
- The `EXPORT_DIFF` and `registers/json` files other than through the agents' reports.
