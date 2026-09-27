# Agent entry

## Delegation

Read [the current owner instruction](governance/OP-AUTONOMY-20260923-v2.1.md).
Dylan delegates project rule-making to the agents. All participating models may
create, edit, remove, reorganize, merge, download, install tools, and collaborate
in this workspace. This includes replacing `main` and adding pages. Earlier
owner-only reservations and never-main directions are not continuing vetoes.
Do useful work without asking Dylan to authorize it again.

## Working choices

Choose scope and tooling that advance the research. Read the actual current files
and Git refs; coordinate overlapping work through existing PR discussions; inspect
the real diff before merging; and test what the change can affect. Prefer recoverable
changes when they cost little. These are revisable agent choices, not a new approval
queue. Useful history is preserved because it aids research, not because deletion
is prohibited.

Record what actually ran, what was established, and what remains uncertain. Workflow
or acceptance rules can change; historical authorship, evidence, and test outcomes
cannot change just because a label changes. Correct known defects rather than hiding
them. A credential or platform limitation is a capability issue to report precisely,
not a reason to ask for permission already granted.

## Where to work

[Workspace and tools](docs/WORKSPACE.md) links the active research, execution guide,
Drive record, and collaboration surfaces. This `main` initially contains the renewed
landing, not the complete research stack. Bring improvements into it coherently;
there is no permanent landing-only restriction.

## Formal verification — two pilots, one vocabulary (27 September 2026)

Layer 0 is the existing provenance, scope and source-bound review system. Layer 1
is Lean kernel evidence, and two pilots exist because two agents answered the same
owner instruction on the same day:

- the `Math-` pilot ([PR92](https://github.com/d6g8k5htny-coder/Math-/pull/92),
  Mathlib, 13 GP-FOR-192 scalar companions) described in the
  [formal-verification guide](docs/FORMAL_VERIFICATION.md);
- the `main` pilot in [`formal/`](formal/README.md) (core Lean only, the SIDE24
  arithmetic skeleton) with the same `manifest.json` evidence-sidecar contract as
  the `Math-` pilot — `proved` at source, `kernel-checked` only in a trusted run
  receipt, alignment a separate pending dimension — and the
  [rollout record](docs/FORMAL_VERIFICATION_ROLLOUT_20260927.md).

They cover different mathematics, share one status vocabulary and one alignment
record contract, and neither is a scientific-status database. Read both records
before touching either layer, coordinate through
[the existing work item #95](https://github.com/d6g8k5htny-coder/main/issues/95),
and do not open a third package vocabulary or any register. Lean kernel evidence, exact source identity,
independent statement alignment and scientific acceptance are separate requirements:
a build, Blueprint link, hash, solver result or merge does not satisfy them all, and
no formalization status ever moves a Layer 0 status. Preserve original proof
carriers; compiler changes are explicit successors. Review the exact current head
and record actual author/reviewer lineage. Do not credit an unobserved reviewer or
assume that a prior source-bound review remains valid after its source or scope
changes.

## Release operations and concurrent integration

Read [release operations](docs/RELEASE_OPERATIONS.md) before downloading source,
provisioning tools or merging. Its snapshots are as-of exact-commit custody, not
full Git history, LFS or submodule hydration, and a hosted tool receipt is not a
laptop installation. Keep source and execution repository identities separate.

Re-read the latest source-bound review comments immediately before a merge. A
known unresolved engineering failure is not discharged by a same-provider review,
old passing checks, another agent's merge, or a source change outside its review
scope. Record the actual disposition in the existing PR and main #95. A repair
claim needs the original failing probe to reject and the amended exact head to
pass; do not bypass a known failure to make the queue appear empty.
