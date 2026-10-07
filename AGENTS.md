# Agent entry

Use the [current workflow](governance/OP-WORKFLOW-20260930.md) to choose the
smallest useful task, coordinate its ownership, verify it and deliver it.
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
unrelated material off GitHub and private workspace material private. Dylan's
name and Gmail address are allowed by that rule. Do not change visibility or
sharing merely because a source is readable.
