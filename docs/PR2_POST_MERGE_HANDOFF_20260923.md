# PR2 post-merge handoff — 2026-09-23

Scope: operational reconciliation, not a mathematical review, original pre-merge
STATUS packet, or new scientific authority. Prepared by OpenAI / ChatGPT after
Dylan asked this session to help Claude resolve the screenshot's issues.
Snapshot: direct GitHub and Drive reads during this session on 2026-09-23.
Re-read live refs and intervening comments before acting on this dated record.

## What is already resolved

[PR #2](https://github.com/d6g8k5htny-coder/main/pull/2) is merged, not draft:

| identity | verified value |
|---|---|
| merge / observed main | `b040bf0c30f33a9de220d19692e8dbcad9a1c5aa` |
| merge time | `2026-09-23T21:27:14Z` |
| first parent: pre-port main | `f25b04bb931df2eaee302b666db014913486166b` |
| second parent: migration head | `b1335def2615888463e2d134294ef451895b8d0d` |
| merged tree | `e5529cf743b43cfb2140629b435af576c1036cef` |

The [merge object](https://api.github.com/repos/d6g8k5htny-coder/main/git/commits/b040bf0c30f33a9de220d19692e8dbcad9a1c5aa)
and live PR metadata establish this event. It was not performed by this handoff.
No repeat approval or merge of #2 is needed.

The [19:44 relayed hold](https://github.com/d6g8k5htny-coder/main/pull/2#issuecomment-5801736084)
was followed by Claude's [21:26 owner-authorization record](https://github.com/d6g8k5htny-coder/main/pull/2#issuecomment-5803214188).
That later record attributes explicit merge authorization to Dylan and says it
supersedes the relay. It expressly states that the original requested STATUS
packet was not supplied. Preserve that history: this later handoff does not
backdate a missing packet or certify that its original conditions were met.

## Remove the permission deadlock, not the evidence requirement

The [existing standing authorization](../governance/OP-AUTONOMY-20260923-v1.0.md)
is an exact mirror of Drive `1N-afVwLVTloSB79GtEK1Fhfx7vdI4_b7`:
3,003 bytes; SHA-256
`1d51a9c68940b6047618286d7f9b3c55146d2650fe9d23d363e7d6133f7f4362`;
Git blob `d6993aba540b3baf915587e42bb5a7ba3efa1d67`.
The live Research Home `180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8`
also directs every model to proceed without renewed owner-permission waits.
The directive already covered engineering integration, experiments and reviews;
copying it into main distributes that permission rather than creating it.

Keep three facts separate. Owner authorization licenses operational work.
A GitHub APPROVE event is a platform review action, subject to account identity
and repository rules. A scientific verdict or organizational-independence credit
requires its own actual evidence. None is a substitute for the other two.
GitHub [does not allow authors to approve their own PRs](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/approving-a-pull-request-with-required-reviews).
Do not manufacture another identity or submit a fictitious review to escape that
restriction. Ordinary authorized integration can use the supported merge path
when its applicable requirements are satisfied; no self-APPROVE event is needed
to explain an already recorded merge.

Claude's author-side checks remain author-side. Another session's technical
review must identify its exact object, authorship and source exposure. This
handoff awards zero independence credit and closes no scientific gate.

## Current branch chain and useful work

At this observation main contains PR2's migration, but not the later hardening
stack. The retained migration branch still points to `b1335def...`; [PR #3](https://github.com/d6g8k5htny-coder/main/pull/3)
targets that branch, with hardening head
`85108745ed4444adb838c53ae79cd603ed90f6fc`. PR27 and PR29 are merged into
hardening, not thereby into main. Do not infer current ancestry from an old
PR description's base SHA; read actual refs and compare the trees.

Keep child PRs targeting their actual integration base. Prepare a separate
scratch integration of current main with the intended hardening changes, resolve
conflicts without dropping either side's checks, and run the complete applicable
suite on that combined tree. Once ready, coordinate a main-targeting integration
PR or deliberate retarget of PR3. Retargeting every leaf is unnecessary. Merging
PR3 only into the retained old branch would not by itself advance main.

The [actual earlier combined-tree failure report](https://github.com/d6g8k5htny-coder/main/pull/3#issuecomment-5802173703)
reported 58 failures and 4 errors, despite a passing checker subset. This is a
historical report for `b1335de + 3f85e93`, not a run against today's changing tips.
It identifies exact-source `ladder.py` pins and workflow/README coverage drift.
The [follow-up ladder analysis](https://github.com/d6g8k5htny-coder/main/pull/3#issuecomment-5802870638)
is evidence to investigate, not permission to overwrite historical source seals.

Concrete repair path: retain the old candidate and its original source binding;
make a numbered successor against the newer module, replay the affected
calculations and negative controls, and record old/new identities and results.
Only then change a consumer to the tested successor. AST similarity alone is
not byte identity or a replay. Reverting only the two report lines would leave
the changed docstring, so it would not restore the old raw hash. Workflow/README
coverage reconciliation can proceed in parallel rather than waiting on the
separate source-binding repair. This path is proposed work, not a completed run.

Claude owns canonical R2 activation, lane/source-map reconciliation and H3 review
filing through [PR21 coordination](https://github.com/d6g8k5htny-coder/main/pull/21#issuecomment-5803340730).
The source-repaired R2 export is Drive `1mVGU5BZD0lR4OwzAjlhbeSbrajRUAtow`,
SHA-256 `97940217312f5fbce108d7ec56356ad89473859ade0becd1d3e3de1aa4bef9db`.
[PR31](https://github.com/d6g8k5htny-coder/main/pull/31) supplies a separate
nonactivating source resolver/preflight. Neither its preview nor this note
claims canonical migration, full stack validation or H3 acceptance is complete.

## Recovery clause: a procedure, not a disclaimer or automatic rollback

A confirmed integrity regression, false status promotion or source corruption
triggers a pause of the affected integration path, preservation of the failing
inputs/logs, and a minimal repair or revert. An unrelated open theorem, an old
screenshot or a pending review is not itself a reason to erase working code.
The preferred response is to revert the smallest identified faulty change.
Preserve unrelated research, original evidence and descendant work.

If a whole-PR2 rollback is actually necessary, first re-read current main and the
merge's ordered parents above. In a clean disposable worktree branched from the
current main, `git revert --no-commit -m 1 b040bf0c30f33a9de220d19692e8dbcad9a1c5aa`
prepares the inverse relative to pre-port main. Review the resulting deletion
scope, reconcile later dependent changes, and test before making an additive
revert commit/PR. Do not use reset or force-push to erase history. This command
has NOT been executed against the research repository by this handoff.

[Git's merge-revert documentation](https://git-scm.com/docs/git-revert)
warns that later merges do not automatically restore previously reverted
ancestry; plan the recovery/reapplication explicitly. A Git revert also cannot
undo copies already made of publicly exposed data.

The public/private discrepancy identified in Claude's merge record remains
unresolved here. No visibility, sharing, credential or protection setting was
changed or verified by this note, and no general public-release permission is
inferred from the merge. That concrete issue is separate from permission to do
non-disclosing technical work.

## Scope and provenance of this handoff

This is a dated operational record, not a second claim register. Current Drive
Work Events remain the coordination record. The PR2 hold history, merged event,
source identities and branch pointers were read through the connectors. GitHub
and Git documentation above were inspected before writing this procedure.
No complete port audit, mathematical review, successful successor replay or
independence-requiring gate closure is asserted. Local file checks and this
handoff's own PR CI must be reported separately, tied to their exact revisions.
