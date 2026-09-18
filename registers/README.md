# Coupled research registers

The Drive's `GP-REG-032-v1.2 — Coupled Research Registers` workbook, exported
2026-09-17 and split into 42 machine-readable tabs.

* `source/GP-REG-032_v1.2_export_2026-09-17.md` — the verbatim workbook export.
  **This is the source of truth in this repository.** Never hand-edit the JSON
  or CSV; edit nothing, or re-export and re-run the importer.
* `json/<tab>.json` — `{tab, sheet_index, header, rows}`.
* `csv/<tab>.csv` — the same rows, header padded to full width.
* `KNOWN_FINDINGS.json` — defects that exist in the **source** registers,
  allowlisted so CI stays green without editing exported data.

Regenerate and verify:

```bash
python3 tools/registers_import.py          # rewrite json/ and csv/
python3 tools/registers_import.py --check  # fail if they drift from the export
python3 tools/registers_check.py           # structural invariants
```

## The tabs

| Tab | What it is |
|---|---|
| `start_here` | R17 compact entry: rules, destinations, exact next reads |
| `review_queue` | 24 exact review objects with technical status, independence, age basis, next action |
| `file_catalog` | metadata snapshot of Drive files (310 rows); superseded in coverage by `drive/inventory.jsonl` |
| `quarantine_index` | 17 exclusions with class, reason, successor and restoration test |
| `work_events` | append-only claim / publication coordination log |
| `research_state_dashboard` | the live scientific dashboard (RN, H5, q0 verifier, P0.1, P14/P15) |
| `open_questions` | OQ-001… decision classes and live state |
| `help_board` | cross-model requests and offered capacity |
| `activity_log` | artifact-level change feed |
| `artifact_index` | 206 artifacts with class, status, authority and dependencies |
| `context_snapshot` | governance key/value snapshot |
| `metadata_schema` | required fields for artifact cards |
| `automation_config` | GP-AUTO-034 settings and safety switches |
| `duplicate_flags` | detected same-title / same-hash clusters |
| `run_log` | automation run receipts |
| `consensus_ballot_retired` | retired no-vote ballot matrix (no live force) |
| `easy_closure_queue` | closure candidates with evidence grade and remaining work |
| `closure_log` | terminal closure records with correction path |
| `transition_log` | 67 material transitions with evidence added/removed |
| `no_change_certificates` | OP-GDN-002 §1 no-change certificates |
| `review_ledger` | 101 review records: reviewer line, sources read, independence |
| `evidence_lineage` | 137 evidence rows binding Drive IDs to exact objects |
| `global_object_audit` | "are we proving the intended theorem" audits |
| `operator_decisions` | 45 operator decisions with scope |
| `alarms` | 182 severity-coded alarms |
| `architecture_metrics` | outcome metrics (uncertainty retired, etc.) |
| `definitions` | 61 canonical operational definitions |
| `relations` | 164 typed relations between objects |
| `autonomy_control` | control-plane version and autonomy budget |
| `active_work_claims` | legacy claim surface (still a collision input) |
| `dispatch_queue` | rank-sorted task board |
| `task_intake` | successor / repair / review intake |
| `cold_start_tests`, `prompt_intent_tests` | entry-behaviour test batteries |
| `p02_exact_hash_review_manifest` | P0.2 exact-hash review manifest |
| `capability_records` | per-provider tool capability probes |
| `work_orders` | cross-model work orders |
| `frozen_objects` | 188 frozen bodies: binding class, expected bytes, expected SHA-256 |
| `identity_drift_watch` | 162 drift watches over those frozen bodies |
| `task_gates` | per-task gate policy rows |
| `cold_start_control_view` | derived cold-start control view |
| `lpw_fold_dispositions` | the ten LPW-fold object dispositions |

## Invariants CI enforces

1. The JSON and CSV are exactly what the importer produces from the source
   export — no silent hand edits.
2. Primary keys are unique in the keyed tabs (modulo `KNOWN_FINDINGS.json`).
3. `review_queue` technical statuses are from the R17 set: `READY`,
   `IN_REVIEW`, `PASS_TECHNICAL`, `AMEND`, `FAIL`, `CANNOT_VERIFY`,
   `NEEDS_RECONCILIATION`.
4. `quarantine_index` classes are from the OP-PROT-019 §6 table.
5. Any `frozen_objects` row with a 64-hex SHA-256 has a positive byte count.
6. **`work_events` is append-only**: rows present in the parent commit must be
   present, unchanged, at the same positions.
