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

## Verification stack

Layer 0 is the existing provenance, scope and source-bound review system. Layer 1 is
the Lean 4 formal layer in [`formal/`](formal/README.md): kernel-checked statements
bound by hash to Layer 0 bytes, one formalization status per claim in
[`formal/registry.json`](formal/registry.json), a glossary, and a statement-alignment
review lane. Read the [rollout record](docs/FORMAL_VERIFICATION_ROLLOUT_20260927.md)
before touching either layer: it says what changed, why, what each repository is asked
to do, and that no formalization status ever moves a Layer 0 status.
