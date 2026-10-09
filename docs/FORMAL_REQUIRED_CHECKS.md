# Required formal verification — owner-authorized enforcement

Dylan Roy explicitly authorized GitHub enforcement and direct agent coordination
on 27 September 2026. Scientific effect: NONE. This implements the existing
main issue #95 obligation; it is not a new status authority or approval identity.

## Executable requirement

| Repository | Existing required check | Dependencies |
|---|---|---|
| main | verify | landing-checks + the main core-Lean package |
| Math- | math-downstream-gates | unchanged downstream replay + the Mathlib package |

The old job's checks are preserved under a distinct name. Its original required
name belongs to an always-evaluated aggregate. Both dependencies must report
exactly success, not skipped, neutral, cancelled or a missing result. Each binds
to the tested Git commit. Formal output also binds repository, workflow run,
attempt and original receipt SHA-256. The producer validates the unchanged receipt
against the current manifest and recorded log digests. No historical receipt is
edited. Failed receipt generation publishes no success output.

The aggregate is driven on every pull request, without path filters, and on main
pushes. The local reusable workflow is resolved at the caller's commit. Both
formal and existing checks use the current tested merge commit rather than mixing
PR-head evidence with base/head integration checks. Changing a base changes the
tested merge; the existing strict rulesets handle revalidation. Manual diagnostic
runs do not substitute for PR checks. This is not merge-queue configuration.

## Per-phase execution custody

The main formal workflow still executes the real gate three times: the initial
CLI, the normal test suite's real-package control, and the optimized test suite's
real-package control. Their raw outputs are retained separately under
`formal-evidence/phases/{initial,normal,optimized}/raw`. The initial and normal
directories are moved out of the shared producer path before the next execution;
the optimized directory remains at `formal/.lake/formal-evidence` for the
unchanged required-check binder. Its snapshot is copied after the binding step,
including any binding file actually produced. The artifact's existing `receipt/`
view comes from that verified optimized snapshot, never an earlier fallback.

Each phase's `observation.json` records its native step outcome, producing
commit/repository/run/attempt, source presence and complete file-size/SHA-256
inventory. Optimized observations also record the actual binding-step outcome.
These are custody records, not new gate receipts or status authorities. Receipt
presence does not establish phase success: a suite may fail after its real gate
has written a valid receipt. Skipped phases adopt no leftover canonical files.
Partial failed output and a failed phase with no output are recorded as such.
Any evidence already present before the first execution is preserved separately
as `preexisting`; it receives no fresh execution credit.

The standard-library helper refuses duplicate destinations, linked ancestors,
symlinks, hard-linked files and special files, then verifies snapshot membership
and bytes. Capture or verification failure fails the formal job. Always-run
collection/upload preserves available evidence; job cancellation, timeout or
upload failure may prevent full retention and must not be reported as complete.
The gate writes a subprocess log only after that subprocess returns, so this
change cannot supply a log that the gate never wrote. This is sequential custody
inside an owned job workspace, not protection against a concurrent hostile writer.
Outside this workflow, the gate's existing fixed-directory behavior is unchanged.

The historical C225 initial log bytes were overwritten and remain unavailable.
New runs produce new evidence; this change does not recover those bytes. Earlier
phase records do not replace the final required receipt, independent statement
alignment, source review, or scientific acceptance.

Regression controls are `tests/test_formal_evidence_retention.py`, run normally
and with `-O` before toolchain installation. Fixtures test custody and actual
receipt binding without compiling Lean; hosted real gate/suite execution remains
a separate required check. The existing shell failure controls remain in place.

## Current and future agents

Coordinate in the current source-bound PR. The earlier discussion is in
https://github.com/d6g8k5htny-coder/main/issues/95, which is closed for issue-list
cleanup; its history is kept and its open obligations are not discharged. Before merging, reread its latest reviews, complete diff and
checks at the exact current head. Do not erase an unresolved AMEND with a peer
merge, a self-review, an unrelated green job or a stale receipt. Respect the
currently posted writer claim and send bounded review findings rather than
racing another branch. Review identity is the actual agent/provider/session and
GitHub identity, not an invented approver. A notice is not an acknowledgment.

All source-preservation, informal/formal alignment and scientific acceptance
obligations remain. This required-check bridge establishes only scoped execution
and engineering dependencies. It does not prove the full research theorem,
authenticate independent semantic review, or promote mathematical status.

## Administrative trust boundary

The pre-change main ruleset23798639 requires verify/public-intake/public-shop;
Math-24045351 requires math-downstream-gates. Both require zero GitHub approving
reviews. Main has a pre-existing repository-role PR bypass; Math- has none.
This code uses the already-required contexts and does not claim to edit settings.
The managed connector does not expose ruleset writes. Do not bypass a rule or
claim an administrator change without a successful write and independent readback.

A workflow author could maliciously replace this aggregate; repository review,
protected workflow ownership and administrative controls are still trust roots.
This patch does not claim to prevent an authorized administrator from bypassing
or changing rules. Increasing required GitHub approvals needs a real eligible
nonauthor reviewer identity; models sharing one login are not multiple approvals.
Do not create an identity deadlock or forge reviews to fill it.

## Deployment evidence

The verified deployment record for main #188 and Math- #96, with the landed
runs, receipts and original workflow artifacts, is preserved in
[audits/formal_enforcement/2026-09-27/](../audits/formal_enforcement/2026-09-27/README.md).
In short:

- Landed push runs 36358139870 (main `3592abb`) and 36359933550 (Math- `22e79e8`)
  passed.
- The never-merge probes main #190 and Math- #97 made the required aggregate fail,
  and both are closed unmerged.
- The temporary Math- default-merge coordination hold is released
  ([Math-#96 comment 5860985432](https://github.com/d6g8k5htny-coder/Math-/pull/96#issuecomment-5860985432)).

## Validation

Run python3 -B -S -m unittest discover -s tests -p test_required_formal_check.py -v
and the same command with -O. The thirty tests cover real CLI success/failure,
missing/extra dependencies, all non-success states, stale commit/run/attempt,
repository substitution, malformed identity, modified logs/manifests, duplicate
JSON keys and reusable-workflow wiring. Hosted negative and positive executions
are recorded separately; unit tests alone do not establish platform merge refusal.

Official references consulted 27 September 2026:
https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks
https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows
