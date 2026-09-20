# Automated closure work

`python tools/closure_pipeline.py run --output-dir /absolute/path/to/new-directory`
executes the complete reviewed CI check list once, including mutation controls and
the full pytest suite. Use Python 3.11 with pytest installed. The output parent must
exist and the output directory must be absent and outside the repository. Optional
`--timeout-seconds` and `--budget-seconds` bound execution. This command delegates
execution and source-drift detection to [run_checks.py](VERIFICATION_RUNNER.md).

`check-plan` validates the inventory without running checks. `plan` prints its
source-bound JSON. CI uses `check-plan`, so it never recursively launches itself.
The plan covers all records from the dated claim graph, lane descriptions, Easy
Closure Queue, Review Queue and Open Questions exports. It retains full records,
original next actions, dependencies, status layers, source paths and SHA-256
identities. This is an explicitly bounded repository inventory; fresh Drive
registers may have newer states. A separate pinned observation layer includes
20 Open Questions, 32 Easy Closure, 84 Help Board, 28 Review Queue and 10 LPW
Current State records observed on September 20. Source/row identity preserves
duplicate IDs across layers. P0.1 version-routing conflicts are recorded for
reconciliation; no automatic authority precedence is inferred. It does not search all Drive proofs or infer
that all listed rows are open. Terminal rows remain terminal in the projection.

Each supported command becomes an executable machine-verification task. Explicit
relevance links connect RN checks to A5, Hermite checks to A1, and the LPW amplitude
check to its constant repair. These links identify useful evidence; they are not
theorem-discharge rules. The runner never executes arbitrary carrier IDs or
commands found in register cells. In particular, a superseded carrier being
listed among historical A5 inputs does not make it eligible for execution.

OQ-013 also carries separately recorded Lean build evidence from the recovered
GP-FOR-192 source. `check-plan` verifies the pinned bundle and its receipt/member
identities; it does not launch Lean or claim that compilation happened during
this Python run. The bundle preserves the original failure, sole `noncomputable`
repair, exact toolchain/dependency locks, 13 theorem axiom reports, two rejected
false statements and explicit counterexamples. Replay uses the bundled optional
Lean runner. The original GP-FOR-001 recovery and statement-scope review remain
separate. See [the recovered Lean package](../research/formal/candidates/LEAN_RECOVERY_VERIFICATION_20260920_v1.md).

The external run directory contains:

- `plan.json`: exact inventory, tasks, original records and input identities.
- `verification/report.json`, logs and `pytest.xml`: actual execution evidence.
- `closure-report.json`: per-task outcomes and per-item resolution candidates,
  bound to both preceding artifacts and a canonical payload hash, with exact
  source next actions and unmet-condition prose retained as structured fields.

The current version reruns the complete catalog; it does not reuse cached
receipts or claim incremental proof checking. A task completes only when its explicit command ran successfully with recorded
log identity against the plan's unchanged inputs. Missing checks, wrong bytes,
failed checks and exhausted budgets cannot yield a successful complete run.
Items without an exact adapter remain `UNMAPPED_EXACT_OBLIGATION`; a failed test
is an execution result, not a mathematical counterexample. All scientific
records remain `SOURCE_RECORD_UNCHANGED`. Independent review, missing analytic
hypotheses and identification with the actual random field cannot be supplied
by a passing synthetic test. The pipeline never rewrites governing statuses.

Engineering work can be completed directly: existing A1/A5 repository notes were
corrected to acknowledge the implemented reference kernels and cover ledger.
Their frozen source quotations and OPEN scientific statuses were preserved.
Further closure adapters should specify the exact predicate, assumptions,
source bytes, executable verifier, negative controls and remaining admission
requirements before adding a relevance link. General proof search and status
promotion by confidence weights are unsupported.

Outputs use exclusive creation and read-only filesystem modes after publication.
They are not cryptographically authenticated or protected from modification by
the filesystem owner; hashes permit later identity comparison. Timestamps are
local observations. Coauthor checks carry zero organizational independence.
