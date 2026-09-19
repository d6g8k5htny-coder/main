# Coupled research registers

The Drive's `GP-REG-032-v1.2 — Coupled Research Registers` workbook (Drive id
`1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no`), exported 2026-09-18 as
`.xlsx` and split into 44 machine-readable tabs.

* `source/GP-REG-032_v1.2_export_2026-09-18.xlsx` — the current export
  (1,919,741 bytes, SHA-256 `c3229ece…3460`; workbook `modifiedTime`
  2026-09-18T16:45:35Z). **This is the import source in this repository.**
  Never hand-edit the JSON or CSV; edit nothing, or take a new dated export and
  re-run the importer.
* `source/GP-REG-032_v1.2_export_2026-09-17.md` — the earlier markdown export,
  kept byte for byte. It is no longer the import source (see *The truncation
  defect* below) but it stays: exports are never edited or deleted.
* `source/SOURCES.json` — provenance of both exports as data (Drive id, title,
  `modifiedTime`, export method and MIME, bytes, SHA-256, `exact: false`).
  Both are renderings of a native Google Sheet, not the object's bytes.
* `json/<tab>.json` — `{tab, sheet_index, header, rows}`.
* `csv/<tab>.csv` — the same rows, header padded to full width.
* `KNOWN_FINDINGS.json` — defects that exist in the **source** registers,
  allowlisted so CI stays green without editing exported data.
* `EXPORT_DIFF_2026-09-17_to_2026-09-18.json` — the mechanical, every-column
  diff between the two exports (see *What moved between the two exports*).

Regenerate and verify:

```bash
python3 tools/registers_import.py          # rewrite json/ and csv/ from the xlsx
python3 tools/registers_import.py --check  # fail if they drift from the export
python3 tools/registers_check.py           # structural invariants
```

The importer reads the workbook with the standard library only (`zipfile` +
`xml.etree`), fails closed (exit 2) if the sheet count or any sheet name differs
from its mapping, and renders every cell as a string: booleans as `TRUE`/`FALSE`,
integer-valued numbers plainly, and any other numeric literal exactly as the
cell holds it — an `activity_log` "Modified UTC" cell holding a spreadsheet
date serial such as `46223.95347222222` is emitted as that string. Formulas are
not evaluated (their cached value is used), merged ranges are not expanded, and
in-cell newlines are preserved. Its docstring states the empty-row and
empty-cell rule.

## Cross-register observations (recorded, not repaired)

Contradictions between exported tabs, or between a tab and the Drive, that no
single-row rule of `tools/registers_check.py` can see. Since 2026-09-19 each of
the five below is a record in `KNOWN_FINDINGS.json` (section
`observations_cross_register`, ids `OBS-2026-09-19-01…05`) that names the cells,
inventory rows and path-change rows it rests on, and the checker re-verifies
every one of those bindings on every run, so an observation cannot outlive its
evidence; each record carries a proposed repair for the owner and none is
applied. The exports are not edited; these are for the owner to resolve at the
source.

* `consensus_ballot_retired` still names "Current governance | OP-PROT-009",
  while `operator_decisions` rows `OD-OP-PROT-010-001` and
  `OD-NOVOTE-20260724-001` say OP-PROT-009 has "no current governing force" and
  the Drive object is titled `HISTORICAL — OP-PROT-009 — No-Vote Mandate`.
* "OP-PROT-003" denotes two Drive objects (the Operating-Philosophy relay
  `1pEGZoTd…` and the majority-rule record `1pVAG4d0…`); `relations.json`
  REL-OPP008-002 links the latter, `governance/protocols/history/` holds the
  former.
* `relations.json` rows REL-036 ("Primary formula evidence", ACTIVE) and REL-087
  ("Exact definition-level law identification", ACTIVE-CANDIDATE / REVIEW-OPEN)
  cite `1lbZzqFOwMubvnD0KBqT9pTlIIIXxUnTA`, whose 2026-09-17 inventory path
  begins `02_LEGACY_Q0_ARCHIVE` — a lane `CLAUDE.md` rule 10 gives zero
  evidentiary authority. The rows predate the object's current placement; no
  claim, module or graph node in this repository cites it.
* `file_catalog` row `1zCAlbdkQ0YwySc8RT9UL4huJuZc0T8Kp` carries the title
  "00_READ ME + MASTER INDEX — single triage inbox (CL, 2026-07-22).md"; the
  Drive renamed it `90_HISTORICAL_TRIAGE_MIGRATION_INDEX_2026-07-22.md` after
  the snapshot (`drive/deltas/2026-09-18/PATH_CHANGES.jsonl` carries the move;
  bytes unchanged).
* One of the seventeen `legacy_primary_file_ids` of the Fresh Start 2.0
  allowlist (`1mFrNQxV9mzwwHwM7V4QMPPAA0vDvN07gCm50WIfUvpc`) is absent from the
  2026-09-17 inventory.

## Three tabs the checker never keyed (found 2026-09-19)

`tools/registers_check.py` checked primary-key uniqueness in twelve tabs and
never read the id column of `relations` (Relation ID), `review_ledger` (Review
ID) or `definitions` (Definition ID). Keying them surfaced 14 duplicate keys
that both exports had carried all along: eleven in `relations` (`REL-036…040`,
`REL-048`, `REL-049`, `REL-132…134`, each id naming two different relations,
and `REL-EC021-CLS141`, an exact duplicate row apart from its review date),
two in `review_ledger` (`REV-P12-GP-006`, `REV-P02-GP-INTERVAL-001`, one
review id over two exact objects each) and one in `definitions` (`DEF-049`,
"Certified capture mass" and "Interval-certified degree-four corridor box").
They are allowlisted row by row in `KNOWN_FINDINGS.json` section
`findings_first_keyed_2026-09-19` and are **not yet covered by any collision
proposal**: the 2026-09-18 proposal is frozen and the 2026-09-19 successor
covers only the seven xlsx-first keys, so a further numbered successor is the
place for these. `context_snapshot` (fifteen rows under two snapshot ids),
`activity_log` (Artifact ID) and `alarms` (Reference ID) also repeat values in
their id-like columns and are deliberately still unkeyed: those columns index
events, not objects. Nothing has been repaired.

## The truncation defect of the markdown rendering

The connector's markdown-table rendering used for the 2026-09-17 export returns
only a **prefix** of a large sheet. Seven tabs were incomplete in this
repository from its first commit until this refresh:

| Tab | Rows the markdown export delivered | Rows in the workbook |
|---|---|---|
| `file_catalog` | 310 | 2,952 |
| `activity_log` | 241 | 518 |
| `artifact_index` | 206 | 761 |
| `transition_log` | 67 | 82 |
| `review_ledger` | 101 | 138 |
| `evidence_lineage` | 137 | 485 |
| `relations` | 164 | 367 |

The prefixes it did deliver agree cell for cell with the xlsx export, apart from
rendering artifacts (markdown escaping of `<`, `` ` `` and `~`; `[merged] `
repeated across merged cells; date serials rounded to five decimals; one mangled
emoji; in-cell newlines flattened). A markdown rendering of the same live
workbook taken minutes before the xlsx export showed the identical cut, so the
cut is a property of that rendering, not of an edit to the Sheet. Seven
duplicate-key defects in the newly delivered rows are recorded in
`KNOWN_FINDINGS.json`, in their own section
(`findings_first_visible_in_2026-09-18_export`); they are defects of the source
workbook, not of the importer. They are not covered by
`collision_proposal.json`, whose source of record is the markdown export and
whose checker quotes colliding rows as lines of that export; under CLAUDE.md
rule 8 that 2026-09-18 proposal is frozen and not edited. They are covered
instead by its numbered successor, `collision_proposal_2026-09-19.json` with
companion `COLLISION_PROPOSAL_2026-09-19.md` (`supersedes: null`,
`successor_of` naming the predecessor by SHA-256), whose source of record is
the xlsx export and whose seven records quote each colliding row cell for
cell from `json/artifact_index.json` and `json/evidence_lineage.json`.
`python3 tools/collision_proposal_check.py` checks the first proposal and
`python3 tools/collision_proposal_check.py --proposal registers/collision_proposal_2026-09-19.json`
the second; together the two cover all 23 findings, each exactly once
(`tests/test_collision_proposal.py`). In the second, the checker also binds
each record to its finding key (the tab, the identifier and exactly the two
rows the key names), recomputes the identical/differing field lists and the
cell counts from the live rows, and holds keeper order, same-object class,
successor-id suffix and proposed cluster ids to the cells and to the
predecessor; it also recomputes the document-level `summary_counts` and
classification lists from the records' cells, binds each operator-reserved
follow-up to the identifier its record proposes, requires every append to
target the Duplicate Flags registry, and compares the companion's verbatim
row blocks, byte-count/digest lines, cell-count lines, materiality
paragraphs and summary-table rows with the live JSON rows. Each of those
checks has a CLI negative control in `tests/test_collision_proposal.py`,
added after an adversarial pass showed the earlier checker let a falsified
summary or a tampered companion block through. Both are proposals: nothing has been
repaired, every successor identifier is proposed and none is created, and the
independence-requiring gates remain open.

## What moved between the two exports

`EXPORT_DIFF_2026-09-17_to_2026-09-18.json` is generated by
`python3 tools/registers_import.py --diff-exports` from the two committed
exports alone (the markdown-era parser is retained in the importer for that one
purpose and never writes `json/`). It pairs rows by key column and occurrence,
lists **every** differing cell of every tab present in both exports, and
classifies each one by a stated rule: a `rendering_artifact` of the markdown
export (97 cells: five-decimal serials, `[merged] ` repetition, backslash
escapes, one mojibake emoji, two date serials the markdown displayed as
`YYYY-MM-DD HH:MM`), an `extension` (1 cell — the `lpw_fold_dispositions`
record link ending `/vie` in the markdown and `/view` in the xlsx; an edit or a
cut, which the exports cannot tell apart) or a `content_change` (50 cells),
with a `status_column` flag from a rule on header names. The four status-word
changes are `RV-LM004-MAIN` Technical status `NEEDS_RECONCILIATION → PASS_TECHNICAL`
and Aging action `ESCALATE → EXTERNAL ONLY`, and `RV-RN-ALIGN` Technical status
`NEEDS_RECONCILIATION → AMEND` and Independence status
`NO_CREDIT_ASSIGNED → AUTHOR_SIDE / ZERO ORG CREDIT`; the other content changes
are the `Age days` column (+1 on every pre-existing route), the reviewer,
verdict-artifact, last-review, next-action, exposure and scope columns of those
two routes, `RV-RN3` Next action, two `start_here` rows, one `start_here` row
removed (`RN overlap`), 4,134 rows added across 13 tabs (2,642 of them
`file_catalog` rows beyond the markdown prefix) and the two new tabs.
`tests/test_registers.py` asserts that the committed file equals the
recomputation, so the change list cannot be curated by hand. The diff decides
nothing: every word in it is the register's, and a `PASS_TECHNICAL` is a
same-line technical pass at zero organizational independence credit.

## The tabs

Row counts are the data rows of the 2026-09-18 export (header excluded).

| # | Tab | Rows | What it is |
|---|---|---|---|
| 0 | `start_here` | 29 | R17 compact entry: rules, destinations, exact next reads |
| 1 | `review_queue` | 25 | exact review objects with technical status, independence, age basis, next action |
| 2 | `file_catalog` | 2,952 | metadata snapshot of Drive files; `drive/inventory.jsonl` is the accessibility source map |
| 3 | `quarantine_index` | 22 | exclusions with class, reason, successor and restoration test (five RN5 rows added 2026-09-17) |
| 4 | `work_events` | 35 | append-only claim / publication coordination log |
| 5 | `research_state_dashboard` | 31 | the live scientific dashboard (RN, H5, q0 verifier, P0.1, P14/P15) |
| 6 | `open_questions` | 20 | OQ-001… decision classes and live state |
| 7 | `help_board` | 84 | cross-model requests and offered capacity |
| 8 | `activity_log` | 518 | artifact-level change feed |
| 9 | `artifact_index` | 761 | artifacts with class, status, authority and dependencies |
| 10 | `context_snapshot` | 15 | governance key/value snapshot |
| 11 | `metadata_schema` | 72 | required fields for artifact cards |
| 12 | `automation_config` | 249 | GP-AUTO-034 settings and safety switches |
| 13 | `duplicate_flags` | 31 | detected same-title / same-hash clusters |
| 14 | `run_log` | 32 | receipts of manual connector sessions, closure watches, a self-healing audit and a custody reconciliation, in the sheet reserved for automation runs; by the rows' own text ("Automation not yet installed"; "Apps Script runtime, OAuth grants, first live refresh, and hourly triggers were not installed") GP-AUTO-034 was never installed or run, and a `SUCCESS` here is a manual session's result |
| 15 | `consensus_ballot_retired` | 2 | retired no-vote ballot matrix (no live force) |
| 16 | `easy_closure_queue` | 32 | closure candidates with evidence grade and remaining work |
| 17 | `closure_log` | 34 | terminal closure records with correction path |
| 18 | `transition_log` | 82 | material transitions with evidence added/removed |
| 19 | `no_change_certificates` | 5 | OP-GDN-002 §1 no-change certificates |
| 20 | `review_ledger` | 138 | review records: reviewer line, sources read, independence |
| 21 | `evidence_lineage` | 485 | evidence rows binding Drive IDs to exact objects |
| 22 | `global_object_audit` | 29 | "are we proving the intended theorem" audits |
| 23 | `operator_decisions` | 45 | operator decisions with scope |
| 24 | `alarms` | 182 | severity-coded alarms |
| 25 | `architecture_metrics` | 73 | outcome metrics (uncertainty retired, etc.) |
| 26 | `definitions` | 61 | canonical operational definitions |
| 27 | `relations` | 367 | typed relations between objects |
| 28 | `autonomy_control` | 54 | control-plane version and autonomy budget |
| 29 | `active_work_claims` | 71 | legacy claim surface (still a collision input) |
| 30 | `dispatch_queue` | 101 | rank-sorted task board |
| 31 | `task_intake` | 34 | successor / repair / review intake |
| 32 | `cold_start_tests` | 78 | entry-behaviour test battery (CST) |
| 33 | `prompt_intent_tests` | 108 | entry-behaviour test battery (PIT) |
| 34 | `p02_exact_hash_review_manifest` | 15 | P0.2 exact-hash review manifest |
| 35 | `capability_records` | 4 | per-provider tool capability probes |
| 36 | `work_orders` | 18 | cross-model work orders |
| 37 | `frozen_objects` | 195 | frozen bodies: binding class, expected bytes, expected SHA-256 (seven RN5 class-D rows added 2026-09-17) |
| 38 | `identity_drift_watch` | 169 | drift watches over those frozen bodies |
| 39 | `task_gates` | 44 | per-task gate policy rows |
| 40 | `cold_start_control_view` | 43 | derived cold-start control view |
| 41 | `lpw_fold_dispositions` | 10 | the ten LPW-fold object dispositions |
| 42 | `reusable_operations` | 15 | reusable operations OP01–OP15 with reuse state, exact source and current review / authority (tab added 2026-09-18); transcribed cell for cell into `engine/operations/REGISTRY.json`, where four displayed identities are re-run in exact arithmetic — a trial is a record, not evidence, and Utility/Novelty stay the register's words |
| 43 | `operation_trials` | 0 | operation trial ledger: header only at this export (tab added 2026-09-18); `engine/operations/trials/` holds records in exactly this eighteen-column shape, append-only, each dated by the runner's clock and none of them evidence |

## Invariants CI enforces

1. The JSON and CSV are exactly what the importer produces from the source
   export — no silent hand edits, and no stale file the source no longer
   produces.
2. Primary keys are unique in the keyed tabs (modulo `KNOWN_FINDINGS.json`).
3. `review_queue` technical statuses are from the R17 set: `READY`,
   `IN_REVIEW`, `PASS_TECHNICAL`, `AMEND`, `FAIL`, `CANNOT_VERIFY`,
   `NEEDS_RECONCILIATION`.
4. `quarantine_index` classes are from the OP-PROT-019 §6 table.
5. Any `frozen_objects` row with a 64-hex SHA-256 has a positive byte count.
6. **`work_events` is append-only**: rows present in the parent commit must be
   present, unchanged, at the same positions.

## What a refresh does not establish

A register row is a transcription of the Sheet, not a verdict. The 2026-09-18
refresh moved two technical statuses in `review_queue` (`RV-LM004-MAIN` to
`PASS_TECHNICAL`, `RV-RN-ALIGN` to `AMEND`) and added one route
(`RV-RN5-MOMENT-REPAIR`, `READY`); those are the register's words. A
`PASS_TECHNICAL` here is a same-line technical pass at zero organizational
independence credit — the register says so in the row — and nothing in this
directory promotes, closes, discharges or reclassifies any claim.
