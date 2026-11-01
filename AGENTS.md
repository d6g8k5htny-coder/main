# Agent entry

**Current authority:** [CBZ-0 governance](GOVERNANCE.md) and the owner's
[expanded Master Directive](governance/MASTER_DIRECTIVE_20261029_SWARM.txt), received 29 October 2026.
The [verbatim native owner record](https://github.com/d6g8k5htny-coder/main/issues/686#issuecomment-6511339492)
has SHA256 `83ac6552fc6406e95213a1cfd6e8636095215e11aac6e3db473cd42c22275a8b`.
Layer A remains immutable. Earlier operational freezes, family quorums,
privacy holds and approval/custody ceremonies are historical where they
conflict with this directive; actual platform permissions remain binding.

Read [STATUS.md](STATUS.md) for the observed baseline state,
[DECISIONS.md](DECISIONS.md) for decisions and handoffs, and
[CONFLICTS.md](CONFLICTS.md) for incompatible work. A current intermediate
handoff is not evidence that CBZ-0 or a verification gate has passed.

## Execute as a team

Use the fluid roles and conflict protocol in GOVERNANCE.md. Ownership is a
coordination hint, not a lock. Work-steal when blocked, coordinate for one
work cycle, then proceed with a logged source-bound judgment rather than
wait for peer or human approval. No shared `/root` alias is an exclusive
canonical-file owner. Prepare reversible edits in isolated branches and
use fresh expected-head guards when changing refs; preserve recovery paths.

The active coordination board remains [Agent Message Board #738](https://github.com/d6g8k5htny-coder/main/issues/738) (successor to [#619](https://github.com/d6g8k5htny-coder/main/issues/619)).
[#684](https://github.com/d6g8k5htny-coder/main/issues/684) is the CBZ task-local
record; [#686](https://github.com/d6g8k5htny-coder/main/issues/686) records owner
source. No board rollover or foreign release is inferred from those links.

## Verify what changed

Read current files and refs before work. A stale checkout, shallow ancestry,
green badge or resolved review thread does not establish the current tested
composition. Record exact source, actual commands, exits and limitations.
Preserve failing tests and all mathematical hypotheses and bounds. Empirical
checks, engineering CI, kernel verification, statement alignment and scientific
interpretation have distinct scopes; report only the observed property.

For formal changes, read [formal verification](docs/FORMAL_VERIFICATION.md)
and the [required-check contract](docs/FORMAL_REQUIRED_CHECKS.md).
For site changes, read the [site verification guide](docs/site/README.md#verify-the-reader-experience);
after changes under `docs/site` or `docs/public-math`, include outputs from
`python -B tools/site_asset_release.py` and its `--check` invocation.
[Workspace and tools](docs/WORKSPACE.md) provides checkout and command details.

## Historical operating records

The [15 October mandate](governance/OWNER_MANDATE_20261015.md),
[standing delegation](governance/OP-AUTONOMY-20260923-v2.1.md),
[old review-family decision](governance/OWNER_DECISION_20261025_REVIEW_FAMILIES.md)
and [prior workflow](governance/OP-WORKFLOW-20260930.md) retain historical
source, authorship and execution facts. They do not reinstate conflicting
approval or exclusive-ownership requirements. Report actual reviewer lineage
and exposure without inventing independence or human review.

Keep personal or unrelated sensitive material out of project publication.
Authorized project work still respects actual access controls and the scope
of the source being handled; readability alone is not a reason to expose it.
