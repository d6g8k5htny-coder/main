# Workspace and tools

For an introduction to the mathematics, begin at the
[public research home](site/index.html). The
[source workspace](site/workspace.html) exposes the existing technical views.
This guide is for contributors who need repository locations and execution
commands. Use the [current workflow](../governance/OP-WORKFLOW-20260930.md) for
ownership, review reuse, testing and delivery.

## Current working locations

As of 12 October 2026, new work lands through pull requests to this repository's
`main` and to [Math-](https://github.com/d6g8k5htny-coder/Math-). The
`experiments/` directory on `main` holds
[`experiments/periodic_h0/`](../experiments/periodic_h0/README.md),
[`experiments/universality/`](../experiments/universality/README.md) and
[`experiments/zeta_fold_exploration/`](../experiments/zeta_fold_exploration/README.md),
the last an exploratory packet whose scientific effect is NONE, which makes no
claim of progress toward the Riemann Hypothesis and whose independent numerical
reproduction (review slice C) is only partly done: a different-provider replay of
its 59 commands is performed (record PASS, scoped). Additional fresh-seed evidence
is reported (record applicability PASS, scoped), not independently verified.
Independent verification of that evidence remains open. The Riemann–Siegel
coefficients now cite Siegel 1932, the source actually checked, and agree with it
as translated (PASS, scoped); Edwards 1974 stays unverified background reading. No
record closes all of slice C. None of the three
reports a held-out confirmation run; each README states its own scope. Two older branches remain
readable context. The legacy [hardening research branch](https://github.com/d6g8k5htny-coder/main/tree/a01c72f1937875b94bfd78e183a6dbbdb2b04da4) (`chatgpt/drive-github-hardening-20260919`, frozen; pinned tip `a01c72f1`)
holds the earlier research software and evidence; its last merge was on
27 September 2026 (`a01c72f1`). The
[Claude migration branch](https://github.com/d6g8k5htny-coder/main/tree/claude/drive-audit-github-migration-rrglpp)
holds additional proposed repairs, including main#237 and main#238; its last merge
was on 2 October 2026 (`b6ccb5e2`). Inspect actual differences and current heads
before combining them: a small advertised change can otherwise bring along
unrelated ancestry. No branch name makes work correct or permanently unmergeable.

Mathematical candidates and their reviews also live in
[Math-](https://github.com/d6g8k5htny-coder/Math-). Use the current source-bound
PR for the object being changed; older branch and rollout records are context,
not an instruction to repeat a completed review.

Use [pull requests](https://github.com/d6g8k5htny-coder/main/pulls) and
[Actions](https://github.com/d6g8k5htny-coder/main/actions) for live collaboration and
execution results. The legacy [research execution guide](https://github.com/d6g8k5htny-coder/main/blob/a01c72f1937875b94bfd78e183a6dbbdb2b04da4/docs/RESEARCH_EXECUTION.md) (pinned at `a01c72f1`)
describes the research-side commands; read it in the checkout actually being used;
it describes the hardening branch. The experiments on `main` are described in
[experiments/periodic_h0/README.md](../experiments/periodic_h0/README.md),
[experiments/universality/README.md](../experiments/universality/README.md) and
[experiments/zeta_fold_exploration/README.md](../experiments/zeta_fold_exploration/README.md).

The [Drive Research Home](https://docs.google.com/document/d/180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8)
and [coupled research registers](https://docs.google.com/spreadsheets/d/1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no)
provide research memory and Work Events. Link an applicable Work Event claim in
the object's PR rather than maintaining a second independent ownership record.
These links require the relevant Drive
access; a public GitHub page does not make linked Drive files public. Historical
permission wording in an older mirror does not revive revoked owner restrictions.

## Where coordination happens

New general questions, obstacles and handoffs go on the
[active Agent Message Board, main #419](https://github.com/d6g8k5htny-coder/main/issues/419).
[main #307](https://github.com/d6g8k5htny-coder/main/issues/307) and
[main #229](https://github.com/d6g8k5htny-coder/main/issues/229) are historical
boards; start with the
[working guidance](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6038554880).
GitHub stopped accepting comments on #229 at 2,500. Deduplicated tasks, current
claims and handoffs stay on
[main #275](https://github.com/d6g8k5htny-coder/main/issues/275) and in the relevant
issue or pull request. Most proof review happens in
[Math- pull requests](https://github.com/d6g8k5htny-coder/Math-/pulls). Dated queue
lists in these threads are starting pointers, not live ownership counts; neither
thread is a scientific-status register, a permanent ownership register or a lock.

## Dylan's review desk

The [delegated review digest](../reviews/owner_review_20260930/README.md) puts the
recent pilot, refinement, proof-appendix and audit decisions in one short reading
path, with exact sources and the discrepancies worth checking. It is attributed
to **Dylan Roy — delegated AI review**, with OpenAI / Codex named as the actual
performer. Dylan's personal reading is pending. This is a retrospective reading
aid, not another approval queue or scientific-status register. The
[delegation rule](../governance/OP-WORKFLOW-20260930.md#delegated-owner-review)
explains how later owner responses and corrections are recorded.

## Milestone audit records

The [milestone-audit protocol](../governance/OP-MILESTONE-AUDITS-20261001.md)
explains coordinated stopping points, exact-source coverage, delta audits and
occasional full audits. Each audit links its scoped report and immutable evidence
from the relevant work discussion and this guide. The
[first stopping-point proposal](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5942391227)
requires affected work owners' acknowledgments before any freeze; it is not
a completed audit or declared freeze.

## Closure evidence

The [closure-evidence protocol](../governance/OP-CLOSURE-EVIDENCE-20261006.md)
adds blind-reconstruction and adversarial-attack review bases and derived,
never-stored evidence profiles. Its
[decision record](https://github.com/d6g8k5htny-coder/main/issues/275) explains
which proposal items were already present, adopted or declined.

## Tools and execution

All participating models have Dylan's permission to download, install, create, and
use useful tools. Choose the actual runtime for the task rather than assuming the
2025 package described real installed software. For example, a clean checkout of
the default branch, where current main-side research software lands, starts with:

```sh
git clone --filter=blob:none https://github.com/d6g8k5htny-coder/main.git research-workspace
cd research-workspace
git rev-parse HEAD
```

To inspect the older hardening software (last merge 27 September 2026), add
`--single-branch --branch chatgpt/drive-github-hardening-20260919`, then `git checkout a01c72f1937875b94bfd78e183a6dbbdb2b04da4` to read the pinned legacy tip. Current numerical work on `main` is in `experiments/periodic_h0/`, `experiments/universality/`, and `experiments/zeta_fold_exploration/`.

Read the checkout's environment files and workflows before installing dependencies.
This session prefers project-local environments, identifiable upstream sources, and
recorded versions so experiments can be replayed without disrupting another worker.
Tools do not need to be installed merely to demonstrate that permission exists.
Disclose actual capability or credential failures rather than inventing a successful
installation or routing routine permission back to Dylan.

A session's preset checkout may be shallow. If
`git rev-parse --is-shallow-repository` prints `true`, `git merge-base` and
`--is-ancestor` can fail or report no shared history. Deepen the clone, or ask
GitHub, which needs no local history:

```sh
git fetch --unshallow origin main   # about 3 s here on 19 October 2026
git merge-base --is-ancestor <sha> origin/main && echo "<sha> is in main"
gh api repos/d6g8k5htny-coder/main/compare/<sha>...main --jq .merge_base_commit.sha
```

Where `--unshallow` is refused, deepen in steps with
`git fetch --deepen=500 origin main`. Until
`git rev-parse --is-shallow-repository` prints `false`, a "yes" from
`--is-ancestor` is reliable but a "no" is not.

The commit is in `main` exactly when that merge base equals `<sha>` (status
`ahead` or `identical`). A 404 alone proves nothing: GitHub also answers 404 for
a mistyped SHA, a missing branch or a repository the caller cannot see. Treat an
unexplained 404 as inconclusive. To establish unrelated history, use a complete
clone in which both commits resolve (`git cat-file -e <sha>^{commit}`) and
`git merge-base <sha> origin/main` prints nothing.

Ancestry shows only that a commit landed. Work carried onto `main` in a new
commit, such as a refresh, has a new SHA, so compare its blobs instead:

```sh
git fetch origin <sha>   # if refused, fetch a ref that contains it: pull/<n>/head
git rev-parse <sha>:<path> origin/main:<path>   # two equal lines: same bytes
```

On 19 October 2026, main#330's head `88ff9223` was not in `main`, but its
`PROOF.md` blob was.

## Local navigation check

The landing workflow's `landing-checks` job first requires that the same run's
`architecture` job (the reusable `research-architecture.yml` workflow) succeeded
and reported the current commit, repository, run ID, run attempt and an archive
name bound to that run and attempt (added by main#320, 8 October 2026); that
binding step checks those identity strings and the format of the reported archive
ID and digests, not the archive bytes or receipt contents. The job then checks the
required-check integration controls, the delivery-archive controls (added to
this job by main#274, 7 October 2026), its own test cases, declared local
documentation links, exact custody of the two relocated historical files, and the
W8b formal and intake controls (added to this job by main#304, 8 October 2026),
which test formal-receipt binding in `tools/required_formal_check.py` and
artifact byte-count and SHA-256 identity in `tools/public_intake_check.py`. The
required-check, delivery-archive and W8b controls each run in normal and
optimized (`-O`) Python; the landing checker's own tests run once, in normal
mode. Last, the job confirms that the checked-out commit is the tested commit.
Its step summary records that research branches and mathematical claims were
not verified by this job. Its required `verify` aggregate also requires
current-commit formal execution under
[the required-check contract](FORMAL_REQUIRED_CHECKS.md).
The local navigation commands use Python's standard library:

```sh
python3 tools/workspace_landing_check.py
python3 tools/navigation_check.py
python3 -m unittest discover -s tests -p test_workspace_landing.py -v
```

The landing checker covers its declared pages and historical bytes. The navigation
checker additionally checks declared Markdown anchors; neither command above
crawls remote links or executes another branch's research. Read actual workflow
definitions for the checks applicable to a change. A local navigation pass does
not replace required hosted CI or substantive review. Changes under `docs/site` or
`docs/public-math` also need `python3 -B tools/site_asset_release.py` followed by
`--check`, and the required `public-shop` steps listed in
[the site guide](site/README.md#verify-the-reader-experience).

## Formal layer check

The `formal-verification` workflow runs `tools/formal_gate_check.py --run-lean` on the
Lean 4 package in [`formal/`](../formal/README.md): fresh build, `leanchecker`, transitive
axiom audit, executable negative controls, receipt. Locally, after installing elan (see
the formal README):

```sh
python3 tools/formal_gate_check.py               # source-only identity check, no Lean
python3 tools/formal_gate_check.py --run-lean    # kernel evidence + receipt
python3 -B -S -m unittest tests.test_formal_gate -v
```

A green run means the manifest's targets built with allowed axioms only at the receipt's
`checked_commit`, every control failed as it must, and the hashes, inventory and informal
anchors agree. That is formal evidence for exactly those targets, not mathematical
acceptance; the [guide](FORMAL_VERIFICATION.md) and the
[rollout record](FORMAL_VERIFICATION_ROLLOUT_20260927.md) explain the boundary.

[Current authority](../governance/OP-AUTONOMY-20260923-v2.1.md) ·
[Current workflow](../governance/OP-WORKFLOW-20260930.md) ·
[Privacy rule](../governance/OP-PRIVACY-20260927.md) ·
[Home](../README.md) · [Historical material](../history/2025/README.md)
