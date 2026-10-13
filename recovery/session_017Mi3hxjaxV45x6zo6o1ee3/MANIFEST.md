# Recovery checkpoint — Anthropic Claude session `session_017Mi3hxjaxV45x6zo6o1ee3`

Owner-requested preservation under main#307 comments 6136156988 and 6136210973 (13 October 2026).
Performer: Anthropic Claude, Claude Code cloud session `session_017Mi3hxjaxV45x6zo6o1ee3`, delegated for
Dylan Roy. Organizational-independence credit 0. Scientific effect NONE. This directory preserves work already
performed; it promotes no scientific status, changes no accepted proof, and is not a review, receipt or
integration artifact. Everything here is marked as one of: LANDED (on a default branch), POSTED (a GitHub
comment or PR body), LOCAL-DIAGNOSTIC (ran here, not hosted evidence), or DRAFT.

Privacy: no credentials, no personal data beyond the owner's name, no private account or host identifiers.
The private delivery material of main#365 (Drive locators, custody note, permission listings) is excluded on
purpose; see section C.

## A. Already recoverable on GitHub (no upload needed)

| Item | State | Where |
|---|---|---|
| main#383 succession guidance (`governance/OP-WORKFLOW-20260930.md` lines 55–83, PR template line) | LANDED `f6911c250179fbb31da12507b9f54de4f3dd01c4`; blobs `169c2733…`, `dee09f0b…` | PR main#383; AMEND 6134414047 → commit `7f7468aa` → PASS 6134536932; author handoff 6134594626; landing check 6134642596 |
| main#282 custodian tasks C202-T1/T2 (17 thread resolutions, body header) | POSTED; PR LANDED `b7deb198…` by the T3 integrator | readback 282/6134573901; release 282/6134585682; whole-run rerun receipt 282/6134909547 |
| meta-framework#15 decision and #26 compatibility controls | LANDED meta main `ff97ad37…` (`tests/test_contract_compatibility.py` `e14438d1…`) | decision 15/6133399141, corrected 15/6133564511; PR #26 |
| meta-framework#27 nonauthor read (integer-hex identity repair) | POSTED; PR LANDED `daa4be26…` | 27/6134767062 |
| Math-#443 nonauthor read (required-checks guide correction) | POSTED; PR LANDED `8acc4856…` | 443/6135831692 |
| Math-#444 nonauthor review (C6 residual child-result contract) | POSTED AMEND_REQUIRED at head `ad666712…`; author amendment pending | 444/6136005124 (pickup 6135947838); evidence files in section B |
| main#365 integrator private handoff | DELIVERED to the owner-only Drive location; R17 Work Events rows 1917–1919 | 365 thread (CLAIM/DELIVERY/RELEASE readbacks 6133190178, 6133264891, 6133279256); locators withheld by policy |
| Toolchain readiness and executor measurements | POSTED | 307/6133905412, 6134096177, 6134118186, 6134233426, 6134406393 |
| Earlier rounds of this session (26 September – 12 October: D1/D5/C6/C7/C8 packets, SIDE24 coefficients, Lean packet stack v1.5–v2.6, register alignment records, audits, closure-proposal records) | LANDED as Math- and main PRs named in the session's task record; audits posted on main#229 | the raw local copies are indexed in `SCRATCHPAD_INDEX.tsv` as published-equivalent and are not re-uploaded |

## B. Local material published in this checkpoint (the delta)

Each file's SHA-256 is in `SHA256SUMS.txt` beside this manifest.

### `math444_review/` — evidence behind AMEND 444/6136005124 (priority item)
- `replay_c6_residual_contract.py` (LOCAL-DIAGNOSTIC tool, standard library only): replays the pinned
  checker and all 15 mutants in both Python modes through the PR's helper exactly as the amended workflow
  does; `--simulate-installed` writes the checker's own final build into
  `frontiers/downstream_gate_20260925/GRAPH.json` for the run and restores it byte for byte.
- `replay_baseline_state.log`: run at Math- PR #444 head `ad6667124c71483796ce2176f72475199c8890b1`
  (base `7def48d4…`), Python 3.11.15, 13 October 01:38Z. Result: zero-mutant output equals `RESULTS.json`,
  all 15 masks equal the frozen map, all 31 child results accepted, in both modes.
- `replay_installed_state_simulated.log`: same head, simulated installed register. Result: zero-mutant output
  equals `RESULTS_INSTALLED.json` byte for byte; three mutants per mode differ from the frozen map
  (`stale-fingerprint`, `record-unreviewed`: LIVE added and `register_state: partial`; `installed-drift`:
  GATE absent) and are refused by the helper. This is the reproducible basis of finding A444-01.
- `verdict_as_posted_6136005124.md`, `pickup_as_posted_6135947838.md`: the posted texts.
- Reproduce: in a full checkout of the PR head, `python3 -B -S replay_c6_residual_contract.py` and
  `python3 -B -S replay_c6_residual_contract.py --simulate-installed`. The hosted job logs could not be
  fetched from this session (the GitHub proxy refuses the blob-storage redirect), so the reconciliation is
  against the live checker, not the log text.

### `metaframework15_decision_support/` — reading aids behind decision 15/6133399141 and 15/6133564511
- `parity_probe.py` (LOCAL-DIAGNOSTIC, stdlib): subset JSON-Schema evaluator versus the Python validators;
  `parity_probe.out`: 91 cases, 32 differences (21 schema-stricter, 11 reader-stricter) at meta main before
  #26; `mutant_controls.txt`: mutants M1–M9 used to check the compatibility tests. Not a full validator.

### `main282_reconciliation/` — raw outcomes behind readback 282/6134573901
- `first_resolve_outcome.json`, `batch_resolve_outcomes.json`: the exact resolve calls and readbacks
  (23:24–23:26Z, 12 October); `threads_parsed.json`: the 25 native threads as read before resolution;
  `t2_patch.py`: the body-header builder used once for T2; `t1_readback_as_posted_6134573901.md`.

### `main383_author_tools/` — author-side tooling for main#383
- `bind_checks.py` and the two check-run tables bound on the PR (`eb385803`, `7f7468aa`).

### `toolchain_readiness/` — LOCAL-DIAGNOSTIC logs behind reports 307/6134096177 and 307/6134233426
- `mathlib_cache_get.log`: `lake exe cache get` for Math- `formal/` at Mathlib `d13f23b7…`, success.
- `math_gate_execute.log`: Math- `formal/gate.py --execute` at `40440ed8…`; `leanchecker` killed, exit 137,
  memory cgroup limit 14,343,688,192 bytes (13.49 GB anon-rss at the kill). FAILED ATTEMPT, capacity only.
- `math_gate_partial.log`: the same gate with only the leanchecker step replaced by a logged SKIP
  (`gate_partial_LOCAL_DIAGNOSTIC_ONLY.py`, a local copy that is NOT the project gate and must not be
  installed); build, axiom audit, type checks and negative controls passed. Not a receipt.
- `torus_replay.log`, `torus_replay_peak.txt`: `companions/torus_filtration_20261011/replay.sh` with
  `LEAN_NUM_THREADS=2`, exit 0 in 5 m 5 s, peak cgroup memory 12,645,441,536 bytes. The locally produced
  receipt file is deliberately NOT published (local receipts are not hosted evidence).
- Pins: elan 4.2.4; `leanprover/lean4:v4.34.1`; Lake 5.0.0; Mathlib `d13f23b723b8a846827a245b89c10fc7d3f11612`;
  Python 3.11.15; container 15 GB RAM, 4 CPUs, no swap.

### `SCRATCHPAD_INDEX.tsv`
Path, size, SHA-256 and mtime of every other local file of this session's scratch area (2,271 files), with
directory summaries for 21 bulky or excluded directories. Everything listed there and not in this delta is
either published-equivalent (a landed packet or a posted record), a disposable checkout or cache, or
excluded for privacy; any named entry can be uploaded on request.

## C. Excluded on purpose
- `d365/` (main#365 private delivery): bundle, custody note template and permission observations contain
  owner-only Drive locators. Recoverable in the owner-only Drive folder recorded in the custody note; the
  public identities (bundle `main365-integrator-private-handoff.zip`, 19,034 bytes, SHA-256
  `afd1fff87a67bcc5450c5cec430657777ef0b2ca49d45ced307feb6b53c629ce`) are already on main#365.
- Local Lean receipts (`.lake/torus-filtration-evidence/receipt.json`, partial-gate outputs): by the
  formal lane's rule they are not evidence; the hosted runs are the receipts.
- Clones, worktrees, Mathlib cache, Python virtual environments and replay output trees: regenerable from
  the pinned identities above.

## D. Remaining obligations, blockers, owner, next step
- Held by this session: nothing. Open on its side: none.
- Pending elsewhere: Math-#444 author amendment for A444-01 (owner: GPT-6 author); a bounded delta read
  follows, by this session if the coordinator routes it. Next reproducible step for whoever amends:
  run the two replay commands above after the change; both must end with "all child results accepted".
- Standing offers: bounded local executor for Mathlib-importing replays that fit this container (companion
  replay yes, full Math- `formal/` leanchecker no), scoped evidence only.
- Blockers: none. Platform limits observed: no scheduling from this session; hosted job logs not fetchable.
- Latest successful checkpoint: this directory, commit on branch `claude/nice-maxwell-7i7ok0` of `main`
  (commit id in the hub post that announces it).
