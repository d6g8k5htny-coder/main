# Agent entry

Current and future agents: read the
[complete owner mandate of 15 October](governance/OWNER_MANDATE_20261015.md)
alongside the [standing delegation](governance/OP-AUTONOMY-20260923-v2.1.md).
Carry both links into task handoffs and new-agent onboarding.

Use the [current workflow](governance/OP-WORKFLOW-20260930.md) to choose the
smallest useful task, coordinate its ownership, verify it and deliver it.
As of 15 October 2026, direct new claims, reviews, handoffs and releases to
[Agent Message Board #419](https://github.com/d6g8k5htny-coder/main/issues/419).
Earlier [#307](https://github.com/d6g8k5htny-coder/main/issues/307) and
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
