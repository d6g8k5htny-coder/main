# Literal register-tab references in the repository corpus

`tools/registers_import.py` imports all 44 exported tabs and
`tools/registers_check.py` checks their structure generically. This separate
map records which in-scope files name individual tabs. Its labels describe
textual attribution; they do not prove runtime reads or complete dependencies.

`registers/CONSUMERS.json` is a derived view. `tools/consumers_check.py`
recomputes it and refuses drift without changing any source register or status.

| class | tabs | derived meaning |
|---|---:|---|
| `MACHINE` | 23 | a non-generic file under a code root, or non-export register JSON, contains a literal tab token |
| `PROSE_ONLY` | 3 | only in-scope Markdown names the tab |
| `UNREAD` | 18 | no in-scope non-generic file names the tab |

## Scan boundary

The generated map declares its exact `scan_scope`. The code roots are
`tools/`, `tests/`, `engine/`, `research/` and `claims/`; non-export JSON under
`registers/` is also included. The prose roots are those five plus `docs/`,
`governance/`, `reviews/`, `packages/`, `sandbox/`, `registers/`, `README.md`
and `CLAUDE.md`.

`drive/`, `legacy/`, `quarantine/` and `recovery/` are excluded before body
reads, as are symlink files and directories. No quarantine metadata is needed.
`UNREAD` therefore means unnamed in this corpus, not absent from excluded
sources. Files larger than 3,000,000 bytes and exported register bodies are
outside the scan; export filenames establish the tab set. Failed file reads
contribute empty text. This is a bounded text index, not a dependency verifier.

## Attribution rules

A machine token is `<tab>.json` or a quoted token `"<tab>"` / `'<tab>'`.
A prose token is `<tab>.json` or a backticked tab name. A bare word is not
enough. Code-root files are scanned literally, including comments and
Markdown, so a match does not establish that any register cell was executed,
read or validated.

Files naming at least 30 tabs are recorded as generic machinery and excluded
from per-tab attribution. The map, checker, its controls and this document
are excluded as map machinery. Classes and counts are derived, never typed
into the map by hand. The current result has four generic files.

The three prose-only tabs are `autonomy_control`, `no_change_certificates`, `run_log`.
The eighteen tabs without a qualifying reference are `active_work_claims`, `alarms`, `architecture_metrics`, `capability_records`, `cold_start_control_view`, `cold_start_tests`, `context_snapshot`, `global_object_audit`, `help_board`, `identity_drift_watch`, `metadata_schema`, `p02_exact_hash_review_manifest`, `prompt_intent_tests`, `research_state_dashboard`, `start_here`, `task_gates`, `task_intake`, `work_orders`.

## Regeneration and checks

```bash
python3 tools/consumers_check.py
python3 tools/consumers_check.py --write
```

Regenerate after changing eligible consumers or the declared corpus. The
checker and tests detect additions, removals, invented classes, omitted
consumers and a changed scope. Synthetic controls prove excluded bodies are
not read and an allowed-path symlink cannot reopen one.

## What this does not establish

Nothing here reads, checks or grades a register cell. An `UNREAD` tab is not
defective. No class is a status, promotion, review verdict or licensing
predicate; no claim, premise or obligation moves. The map describes literal
names within the declared corpus at one tree, not the export's coverage of
Drive, runtime dependency completeness, or the truth of any referenced data.

## Cold-start navigation

`cold_start_control_view`, `cold_start_tests`, and the related UNREAD tabs
`start_here` and `prompt_intent_tests` are UI/ops cold-start surfaces: entry
routing and entry-behaviour tests. They are not a mathematics source of truth.
`UNREAD` means unnamed in this corpus. An `UNREAD` tab is not defective.
This section names those tabs only to route a reader. It does not add a
consumer, and it does not move any class: this document is map machinery and
stays outside the scan.

Repository entry is the root [`README.md`](../README.md), which points on to
[`docs/RESEARCH_MAP.md`](RESEARCH_MAP.md) and
[`docs/OPEN_PROBLEMS.md`](OPEN_PROBLEMS.md). The docs index is
[`docs/README.md`](README.md). The STATUS packet is
[`docs/math_status/`](math_status/README.md): an OPEN/HOLD display, not a
second claim log. Exported tab roles are listed in
[`registers/README.md`](../registers/README.md).

A cold-start smoke pass and a green dashboard do not imply `lemma_closed`,
do not imply `discharges_OBL_H5_JETMOD`, and do not discharge
`D3-LEMMA-RN-UNIF`. Green CI is a run. It is not obligation discharge.
`OBL-H5-JETMOD` stays OPEN.
