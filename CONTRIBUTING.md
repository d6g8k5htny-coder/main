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

## Pick a public task

Use [public tasks](https://github.com/d6g8k5htny-coder/main/issues?q=is%3Aissue+is%3Aopen+label%3Aeng-only)
or [propose a bounded task](https://github.com/d6g8k5htny-coder/main/issues/new?template=task.yml).
The starter types are `replay`, `chart`, `review-comment`, `catalog-stub`, and
`docs`. Cite the existing key, public path, full commit and digest from
[the single public inventory](docs/PUBLIC_MATHEMATICS.md). An import is a byte
copy for review, not an accepted result.

Comment `/claim` before beginning. Keep one open claim per person; a maintainer
records the claimant and assigns the issue when GitHub permits. Link your PR when
ready, or comment `/release` if you stop. These are manual coordination conventions,
not an installed claim bot. The task states are `open` → `claimed` → `results-pr`
→ `review` → `parked` or `landed-eng`. `landed-eng` means published engineering
or review material, not mathematical acceptance.

Good tasks regenerate a chart from pinned JSON, document one existing import ID,
or submit a precisely located review comment. Closing the D1 parent or accepting
SARD-G is not a public starter task. Do not edit ACCEPT/AMEND rows, `STATUS.md`,
`PROOF_INDEX`, `LANDING_CLAIMS`, `claims/`, prizes or `lemma_closed`, or merge AMEND
mathematics. Math- remains the proof vault; sandbox and trial are not upload lanes.

## Submit public results for review

1. Fork this repository and create a branch from its current `main`.
2. Add a **new** `incoming/<id>/` directory. Use a short lowercase ID containing
   letters, digits or hyphens, for example `incoming/side24-replay-yourname/`.
   Change only files in that one new directory. For a correction after landing,
   use a new ID and link the previous result; earlier submissions remain readable.
3. Include `RESULT.md`, your `output.json` and/or plots, and `IDENTITY.json`.
   In `RESULT.md`, link the task and describe the exact source commit, command or
   method, runtime, result, limitations and any source exposure. State that the
   result is for review and has scientific effect `NONE`. A numerical result or
   plot is not a theorem. Never upload credentials, personal data or private sources.
4. In `IDENTITY.json`, list every artifact except the identity file itself with
   its byte count and SHA-256. List one to eight public source files by repository,
   path, full 40-character commit and SHA-256. The checker reads those exact public
   bytes. Use the real values, not the placeholders in this schema:

   ```json
   {
     "schema": 1,
     "scientific_effect": "NONE",
     "review_status": "REVIEW_REQUIRED",
     "sources": [
       {
         "repository": "d6g8k5htny-coder/Math-",
         "path": "path/to/public/source.json",
         "commit": "REPLACE_WITH_40_LOWERCASE_HEX_CHARACTERS",
         "sha256": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS"
       }
     ],
     "artifacts": [
       {"path": "RESULT.md", "bytes": 123, "sha256": "REPLACE_WITH_REAL_SHA256"},
       {"path": "output.json", "bytes": 456, "sha256": "REPLACE_WITH_REAL_SHA256"}
     ]
   }
   ```

   Generate byte counts with `len(Path(name).read_bytes())` and digests with
   `hashlib.sha256(Path(name).read_bytes()).hexdigest()` in Python, or use
   `wc -c` and `sha256sum`. Artifact paths are relative to the new package.
   The identity file is excluded to avoid a circular self-hash; its Git blob and
   PR head are checked separately.
5. Open a draft PR **against this repository's `main`**, link the claimed task,
   and request the `results-for-review` label. A maintainer applies it if you cannot.
   Mark the PR ready when the result is ready to review. Removing the label cannot
   bypass the guard: a fork, an `incoming/` path, or that label independently selects
   the restricted lane.
6. A human or designated agent reviews the content and exact bytes. If accepted
   for publication, the PR lands only in `incoming/`. This is not automatic merge,
   a theorem verdict, or transfer into Math-. A later audit needs its own reviewed
   engineering PR; no upload changes the scientific registers.

The first intake version accepts UTF-8 `.md`, `.txt`, `.json`, `.csv`, and `.png`
files, with at most 50 files, 256 KiB per file and 2 MiB per package. Source files
must be public in `main`, `Math-`, `query-` or `Universal-Law-Workspace` and regular
files of at most 2 MiB, bound to their exact commit-tree path and blob. Scripts, notebooks,
HTML, SVG, archives, executable files, symlinks and submodules are refused here.
For code or a notebook proposal, describe it in an issue; a maintainer can sponsor
a separate engineering PR. Do not encode a refused file inside an allowed format.

`public-intake` runs trusted default-branch code with a read-only token. It reads
PR metadata and blobs through the GitHub API, checks complete path lists and source
identities, and fails on missing information or a changing PR. It never checks
out or executes submitted code. The credential scan catches a small documented
set of recognizable private-key/token patterns; **it is not a complete secret
scanner**, and a green check cannot establish that content is safe or correct.
Reviewers must inspect the content before merging. Source readback proves identity
and public availability, not the source's acceptance or the output's correctness.

Maintainer branches in this repository can still submit ordinary engineering PRs.
Adding an `incoming/` change or `results-for-review` label also restricts those PRs.
Require PR review and the `public-intake` check in repository rules before treating
this workflow as an enforced merge boundary. The workflow file alone does not
install branch protection or prevent an administrator bypass. Before merging,
the designated reviewer must verify that the actual `Public results intake` run
used the trusted `pull_request_target` workflow and reports the PR's current exact
head. A check name alone does not bind its workflow or event identity; a similarly
named check from a submitted workflow is not the trusted intake guard.

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
