# Which register tabs this repository actually reads

`tools/registers_import.py` imports all 44 tabs of the `GP-REG-032-v1.2`
export and `tools/registers_check.py` checks every one of them structurally —
by walking `registers/json/*.json`, without naming any tab. So a tab can be
exported, imported, structurally checked and still be read by **nothing**: no
checker consults its cells, no module transcribes a row, no document quotes it.
That difference was invisible until 2026-09-20. `registers/CONSUMERS.json`
records it per tab and `tools/consumers_check.py` recomputes the scan on every
run, so the record cannot drift from the tree.

| class | tabs | what it means |
|---|---:|---|
| `MACHINE` | 23 | a checker, test, lane, claim node or pinned proposal names the tab and reads or pins its cells |
| `PROSE_ONLY` | 7 | quoted in Markdown, read by nothing |
| `UNREAD` | 14 | named nowhere outside the generic machinery |

Four files are **generic machinery** and are attributed to no tab, because they
touch every tab by construction: `tools/registers_import.py`, its
`registers/EXPORT_DIFF_2026-09-17_to_2026-09-18.json`, `tests/test_registers.py`
and the tab list in `registers/README.md`. Three more are machinery *about* the
map — the map itself, `tools/consumers_check.py` and `tests/test_consumers.py` —
and are excluded for the same reason: this document names `run_log` and
`alarms` to describe them, which is not reading them.

## The rule, so the count can be checked

* **Machine consumer** — a file under `tools/`, `tests/`, `engine/`,
  `research/` or `claims/`, or a non-export JSON file under `registers/`, whose
  bytes contain `<tab>.json` or the quoted token `"<tab>"` / `'<tab>'`.
* **Prose mention** — a Markdown file whose bytes contain `<tab>.json` or the
  backticked token `` `<tab>` ``. A bare word is deliberately not enough:
  "definitions" and "relations" are ordinary English, and matching them would
  invent consumers that do not exist. A control in `tests/test_consumers.py`
  pins that.
* **Generic** — a file naming at least 30 of the 44 tabs.
* The class is **derived** from the scan, never typed. A control flips a
  recorded class and the checker refuses it.

## The 14 tabs nothing reads

Each is imported, exported to JSON and CSV, and structurally checked; none is
read, quoted or transcribed anywhere in this repository.

| tab | rows | first columns |
|---|---:|---|
| `architecture_metrics` | 73 | Metric, Numerator, Denominator, Value |
| `capability_records` | 4 | Capability Record ID, Provider / Line, Session / Tool Context |
| `cold_start_tests` | 78 | Test ID, Generic Prompt, Search Query, Bootstrap Search Rank |
| `context_snapshot` | 15 | Snapshot ID, Category, Key, Value |
| `global_object_audit` | 29 | Audit ID, Recorded UTC, Exact Object ID, Original Scientific Question |
| `help_board` | 84 | Item ID, Type, Priority, Topic |
| `identity_drift_watch` | 169 | Watch ID, Object ID, Drive ID, Binding Class |
| `metadata_schema` | 72 | Field, Required, Type, Allowed values / example |
| `prompt_intent_tests` | 108 | Test ID, Class, Raw Prompt / Scenario, Speech Act |
| `research_state_dashboard` | 31 | (a display sheet; its header row is prose) |
| `start_here` | 29 | RESEARCH HOME · R17, Purpose / rule, Open / exact range, Status |
| `task_gates` | 44 | Gate ID, Dispatch ID, Origin / Intake ID, Gate Policy Version |
| `task_intake` | 34 | Intake ID, Created UTC, Creator Session / Provider, Origin Type |
| `work_orders` | 18 | Work Order ID, Requested UTC, Requesting Session, Provider |

Seven more are quoted but not read: `active_work_claims` (71 rows), `alarms`
(182), `autonomy_control` (54), `cold_start_control_view` (43),
`no_change_certificates` (5), `p02_exact_hash_review_manifest` (15) and
`run_log` (32). `alarms` joined them on 2026-09-20, when the
12_P1.1_LAW_SPECIFIC_Q_MACHINE lane README quoted it; until then nothing in the
repository named it and it sat in the table above.

That is 21 tabs, 1,190 rows, whose content reaches nothing in this repository.
(Until 2026-09-20 this line said 1,181 rows. Re-added from the per-tab counts above,
which are each correct, the total is 1,190; the old figure was an arithmetic slip and
not a changed export.)
The mathematics is not among them: the tabs the claim graph, the lanes, the
carriers, the review routes and the collision proposals depend on —
`review_queue`, `frozen_objects`, `operator_decisions`, `open_questions`,
`easy_closure_queue`, `quarantine_index`, `artifact_index`,
`evidence_lineage`, `relations`, `definitions`, `review_ledger`,
`duplicate_flags`, `transition_log`, `closure_log`, `work_events`,
`activity_log`, `dispatch_queue`, `automation_config`, `file_catalog`,
`lpw_fold_dispositions`, `reusable_operations`, `operation_trials`,
`consensus_ballot_retired` — are all `MACHINE`.

## Keeping it honest

```bash
python3 tools/consumers_check.py            # recompute and compare; nonzero on drift
python3 tools/consumers_check.py --write    # regenerate the map deliberately
```

Adding a consumer, removing one, adding a tab to the export or renaming a file
makes the map stale and the checker says so by name. The map is therefore a
statement about the tree at a commit, not a claim about the future.

## What this does not establish

Nothing here reads, checks or grades a single register cell. A tab with many
consumers is not more correct than a tab with none; an `UNREAD` tab is not
defective, missing, or a defect in the source — it is simply not used by this
repository yet, and several of these tabs are operational scaffolding (help
board, task intake, prompt tests) that a git mirror has no reason to consume.
No class in the map is a status, a promotion, a review verdict or a licensing
predicate, and no claim, premise or obligation moves because of it. The map
measures this repository's coverage of the export — not the export's coverage
of the Drive, and not the truth of anything either one says.
