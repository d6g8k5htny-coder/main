# Contributing

This repository is worked by human and model contributors together. The rules
below are the ones that keep the mathematics checkable; they are chosen by the
participants and may be improved by them.

## Before you start

1. Read [AGENTS.md](AGENTS.md) for the cross-model entry point and
   [CLAUDE.md](CLAUDE.md) for the working conventions.
2. Read the [research guide](docs/RESEARCH_INDEX.md) for the current state of
   each result, and [open work](docs/RESEARCH_INDEX.md#open-work) for unclaimed
   tasks.
3. **Declare the work before you begin it.** Say what you intend to change, with
   a timestamp, where other contributors will see it — the coordination board on
   the research branch, or a comment on the relevant issue. This is how several
   agents work the same repository without overwriting each other.
4. Check the current branch and exact head. A merge is not a semantic
   reconciliation, and a stale base silently changes what your diff means.

## Evidence standards

These are not style preferences. A change that breaks one of them will be sent
back regardless of how good the result looks.

- **Exact rational arithmetic wherever a bound is claimed.** Use
  `fractions.Fraction`, never floating point. If a path genuinely computes in
  floats, label it `NON-CERTIFYING` in the code *and* in its output.
- **Certified enclosures go through the interval module.** Its contract is that
  containment is unconditional and tightness is best-effort. Read its public API
  rather than reimplementing it.
- **Every checker and every claimed bound needs a negative control** — a test
  that fails when the check is weakened or an inequality flipped. Resolve module
  globals at call time and drive checkers through their command-line flags in
  tests: a default argument bound at import time once made every mutation test
  silently re-check the good input.
- **Generated files are regenerated, never hand-edited.** A defect in the source
  is repaired at the source.
- **Frozen bodies get numbered successors, never edits in place**, and nothing is
  permanently deleted.
- **Python 3.11, standard library only.** `pytest` is the single test
  dependency; guard any `mpmath` or `numpy` import and skip when absent.

## Claims and status

- **No status moves by merge.** Commits, tests, receipts and reviews cannot
  promote, close, discharge or reclassify a claim. A green run is a run.
- **Say what your evidence actually establishes.** A computational test
  establishes what it checks, and no more. A reproduced bound is not a proof,
  and a passing suite is not a review.
- **Independence is recorded separately from correctness.** A reviewer from the
  same provider as the author earns zero organizational-independence credit
  whatever the technical verdict, and different-provider identity alone does not
  establish independence. Disclose source exposure: reading the author's
  derivation before attempting to break it is the weakest form of review, and it
  must be stated.

## Pull requests

- Open as a **draft** until it is ready for review.
- Describe what changed, what you verified, and — as carefully — what you did
  **not** verify.
- Keep the diff to what the task needs. Do not widen someone else's pull request.
- Do not edit exported registers or frozen proof bodies to make a check pass.
- If you disagree with a review, say why on the thread rather than silently
  changing the work.

## Reporting a problem in the mathematics

Open an issue with the exact object — path, byte count and SHA-256 where you
have them — the precise step you believe fails, and the smallest case that shows
it. A counterexample is a first-class contribution here; several of the
program's settled results are negative.
