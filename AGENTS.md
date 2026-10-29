# Agent entry

**Current authority (2026-10-29): [CBZ-0 governance](GOVERNANCE.md).**
The owner's MASTER DIRECTIVE supersedes the procedural mandates, freezes,
family-review requirements and approval/custody gates below. They remain
historical references. Execute the CBZ-0 mission without owner questions;
preserve Layer A rigor, tests, source outcomes and platform permissions.
Read `STATUS.md` for observed baseline state and reproducible gates.

Current and future agents: read the
[complete owner mandate of 15 October](governance/OWNER_MANDATE_20261015.md)
alongside the [standing delegation](governance/OP-AUTONOMY-20260923-v2.1.md).
Before assigning or reusing reviews, read the
[25 October review-family decision](governance/OWNER_DECISION_20261025_REVIEW_FAMILIES.md):
a separate engineering reviewer may be from the author's model family;
scientific review requires a different model family from the author.
Carry these three links into task handoffs and new-agent onboarding.

Use the [current workflow](governance/OP-WORKFLOW-20260930.md) to choose the
smallest useful task, coordinate its ownership, verify it and deliver it.
As of 25 October 2026, direct new claims, reviews, handoffs and releases to
[Agent Message Board #619](https://github.com/d6g8k5htny-coder/main/issues/619),
which continues [#539](https://github.com/d6g8k5htny-coder/main/issues/539);
#539 continued [#475](https://github.com/d6g8k5htny-coder/main/issues/475), and #475 continued [#419](https://github.com/d6g8k5htny-coder/main/issues/419) after
GitHub stopped accepting comments there at 2,500.
Earlier [#539](https://github.com/d6g8k5htny-coder/main/issues/539), [#475](https://github.com/d6g8k5htny-coder/main/issues/475),
[#419](https://github.com/d6g8k5htny-coder/main/issues/419),
[#307](https://github.com/d6g8k5htny-coder/main/issues/307) and
[#229](https://github.com/d6g8k5htny-coder/main/issues/229) remain readable history;
follow an explicit successor pointer if needed. Moving the board does not
transfer, renew, or release any existing claim.
Dylan's current explicit instruction resumes the requested reader experience
and workflow improvements, including implementation, review and integration.
The [September 27 stop](governance/OWNER_STOP_20260927.md) is no longer in effect
([owner decision, 5 October](governance/OWNER_DECISION_20261005_CURSOR.md));
Cursor agents work on request and arm no permanent windows, timers or loops.

The [owner delegation](governance/OP-AUTONOMY-20260923-v2.1.md) permits agents
to improve project rules and carry out authorized work without repeated permission
requests. Read actual current files and refs, respect active peer scopes, and
record what ran and what the evidence establishes. Workflow changes cannot
rewrite historical authorship, hypotheses or outcomes.

A resumed session can start on an old branch or in a shallow clone. Judge
repository state from a freshly fetched `origin/main`, not the checkout: the
December 2025 branches have no `governance/` at all. When
`git rev-parse --is-shallow-repository` prints `true`, `git merge-base` and
`--is-ancestor` can miss shared history, so take ancestry, including for the
already-landed check, from GitHub's compare API (`merge_base_commit`) or after
`git fetch --unshallow`. Ancestry shows only that a commit landed: work carried
onto `main` in a new commit needs a content comparison. The commands are in
[Workspace and tools](docs/WORKSPACE.md).

## Choose the right entry

- [Public research home](docs/site/index.html): approachable mathematics and
  exploration for readers.
- [Workspace and tools](docs/WORKSPACE.md): repository locations, commands and
  technical records for contributors.
- [Release operations](docs/RELEASE_OPERATIONS.md): source downloads, execution
  custody and integration details when those operations are needed.
- [Formal-verification guide](docs/FORMAL_VERIFICATION.md) and
  [required-check contract](docs/FORMAL_REQUIRED_CHECKS.md): read before changing
  formal code, its evidence or enforcement. The
  [rollout record](docs/FORMAL_VERIFICATION_ROLLOUT_20260927.md) explains the two
  existing pilots; do not introduce another vocabulary or scientific register.

For changes under `docs/site` or `docs/public-math`, use the
[site verification guide](docs/site/README.md#verify-the-reader-experience).
After editing, stage new or deleted files, run
`python -B tools/site_asset_release.py` followed by
`python -B tools/site_asset_release.py --check`, and include every rewritten file
in the commit.

Required hosted checks remain required at the current tested commit. Preserve
source identities and unresolved findings. Mathematical review, kernel evidence,
statement alignment and scientific acceptance are separate; a merge or green
check cannot substitute for any missing scientific predicate. Record actual
reviewer lineage and exposure; shared accounts or same-provider reviews do not
create organizational independence.

Follow the [privacy rule](governance/OP-PRIVACY-20260927.md): keep personal or
unrelated material off GitHub and private workspace material private. Apply its
[current publication clarification](governance/OP-PRIVACY-20260927.md#current-owner-authorized-publication-scope)
rather than reopening already-authorized metadata-only holds. Do not change
visibility or sharing merely because a source is readable.
