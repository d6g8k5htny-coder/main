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
*Correction, 2026-10-07 (live read):* the bypass actor on main ruleset 23798639 is
the ChatGPT Codex Connector integration (1144995), bypass mode `always`; see
[OP-ACCESS-20261007](../governance/OP-ACCESS-20261007.md) for the pending owner action.
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
