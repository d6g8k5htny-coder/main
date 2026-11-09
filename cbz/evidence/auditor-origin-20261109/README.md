# Auditor assertion-origin repair: preserved observations

The base Main commit is `05f2289478a0ee9492adc1eb501466e0f66866df`. At that
commit the Lean grader was blob `faab4ff7e4b515320a95238168ed2d641c226677`
and the A3 grader was blob `b7b56615fa6b11b88d74800478e156be883bedc5`.
Both treated the innermost *formatted traceback filename* as origin authority.
In actual subprocess probes on CPython 3.12.14, an unchanged subject passed,
while an `AssertionError` raised by the subject itself under a directory named
`tests` was incorrectly accepted as a kill by each grader in each Python mode.
`red-results-*.json`, `*-subject-*.stderr` and `initial-probe.py.txt` preserve
those observations and source hashes. `raw-failure-logs.zip` preserves the exact `red-controls-*.log` red
adverse tests and hashes. These failures affect the graders' mutation-credit soundness;
they do not refute a mathematical statement or erase earlier executions.

The repair uses the existing `tests/cbz_inventory_origin.py` recorder, which
captures actual traceback code identities before unittest cleans the exception.
A selected synchronous test-body failure is a kill only if all later frames
are explicitly authorized standard unittest assertion helpers. A clean
selected-test baseline is required for each actual source mutant. Subject
bytes are hashed before and after the run. Missing or inconsistent provenance,
subject/helper/nested callback exceptions, unexpected skips, errors and source
drift are non-kills. The recorder is for trusted synchronous unittest tests; it
is not a hostile-code sandbox, a Lean kernel, a proof checker or a federation
certificate. Existing historical mutation outcomes are not rebound.

The initial adapter self-mutation attempt failed in both modes because the
copied adapter inferred the temporary directory as its test checkout; full
failed logs are retained inside `raw-failure-logs.zip`; the explicit
limitations note is in `attempts.log`. The child then explicitly binds the copied helper to the
checkout owning the designated tests. `full-evidence-red.log` preserves the
subsequent red control for missing full-suite evidence. One partial-state log
was accidentally overwritten; its known exit and that loss are recorded in
`attempts.log`, not reconstructed.

The separately recorded pre-repair lexical scan returned exit 1 on Main's
intentional, unapproved kernel-replay axiom fixture, and reported the replay
adapter's `unsafe` token for review; `main-lexical.json` is unchanged. The
Math scan at `9bdfc54b76173b57912503bfcf901a5f9d43b09c` returned exit 0
for 1,974 Lean blobs with no listed tokens; `math-lexical.json` is the raw
report. These scans alone cannot establish axiom closure or scientific
alignment. Workspace candidate `e9b6ef598ae249120be0d7498c6d5adcc95a1cab`
returned exit 2 on its gitlink; it was not silently skipped. The Math Python
census returned exit 1 on the preserved historical documentary writer
`frontiers/source_derivative_gaussian_20261025/actual-source-finite-superlevel-components-evidence/raw/producer/history/freeze_documents.001.py`
(blob `f588617889ed04eb2c76bb1ab428fcfa5217ac4b`), whose original parser
failure and later correction are preserved in that packet. Python AST float
signals are a census, not a verdict that a bound is or is not certified.

The first synthetic workflow-guard control failed because its stub expected
21 tests while the candidate shell still expected 20; that red output remains
in `raw-failure-logs.zip` as `workflow-guards-red-normal.log`, and the workflow now includes the guard
control in its 21-test group. Its five guard removals use synthetic child
outcomes and do not claim real scanner or mathematical execution.

The candidate workflow runs the explicit controls and mutation suites in
both Python modes, retains selected child evidence and full-suite logs even
on failure, and checks source hashes. Its hosted results must be observed on
the exact head before this repair can receive current source credit. The
legacy full-suite baseline intentionally records two Lean and four A3 inner
recursion-guard skips; the exact expected skip identities are checked and
retained, while all 101 selected mutation baselines per mode must have no skips.
This is not the final G0/G1 or a full Layer A audit.
