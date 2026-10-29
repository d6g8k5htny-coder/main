# Stale open PR disposition evidence

This is CBZ-0 Phase 2 preparation, recorded on issue #684 by unit `CBZ0-P2PREP-STALE-PR-DISPOSITIONS-01SHCRXV`. It
covers the 27 open `main` PRs that had no activity between 26 and 29 October 2026, read at `main` `81acc060` on
29 October. One of them, #331, was closed during the read, so the files cover 26.

| File | What it is |
|---|---|
| `CURATED.md` | The recommended disposition for each PR, with a manual rationale. Start here. |
| `QUARANTINE_DRAFTS.md` | Draft proof-index quarantine entries for #336, #340 and #429, in the governance fields. |
| `DISPOSITIONS.md` | The mechanical first pass: rules only. |
| `dispositions.json` | Per-PR evidence: head, base, stack parent, ancestry, per-path content comparison with `main`, trial merge result, Layer A signals, each review's id, state, commit, submitting account and verdict class, and check runs at the head. |
| `pr_evidence.py` | The generator. Python standard library only. |
| `test_pr_evidence.py` | Unit tests for the generator, run against temporary Git repositories. |
| `mutation_check.py` | An adverse check. It breaks one rule at a time in the generator and requires the tests to fail for every broken copy. |

## What the evidence does and does not establish

- **Mechanical facts, observed with Git at the stated cut.** Ancestry, blob equality and `git merge-tree` results.
- **Layer A fields are lexical signals.** They cover:
  - `sorry`, `sorryAx`, `admit`, `axiom` and `native_decide` on lines a PR adds to `.lean` files. `--` comments are ignored; block comments are not tracked, so a token inside one is still reported.
  - Removed `test*` functions, deleted test files, and skip or xfail markers that a PR adds.
  - Float literals, `float()`, inexact `math.` calls, and numpy or scipy imports on lines a PR adds to a `.py` file. Only certificate-like paths count toward the recommendation.

  A signal is a lead. The absence of a signal certifies nothing.
- **Recommendations come from the documented rules in `recommend`, applied in order.** `CURATED.md` adds a manual reading for each PR. Neither is an executed disposition: closures and merges belong to the canonical integrator under governance B-003.
- **No quoted third-party text is published.** The local `raw/` files hold verbatim PR bodies and comments; `.gitignore` keeps them out of commits.
- **Records are always fresh and bound to one head** (finding GTW-691-F1). `fetch` overwrites every requested `raw/<n>.json`. `analyze` refuses, exits 1 and writes nothing if any PR's local `refs/cbz/pr/<n>` differs from the head in its fetched record. Reviews are then judged against that same head.
- **Reviewer identity means the submitting account** (finding GTW-691-F2). Each review carries `reviewer_login` and `reviewer_id`. In this federation, most agents submit through the owner's shared account `d6g8k5htny-coder`, so these fields identify the account, not the performing agent or its model family. The performer is named only in the review text, which is not published.

## Reproduce

```sh
# Network step: GitHub metadata for each PR, saved to raw/<n>.json.
python3 -B -I pr_evidence.py fetch --repo-slug d6g8k5htny-coder/main --out . 333 336 338 ...
# Local Git objects the analysis reads:
git -C <clone> fetch origin '+refs/pull/<n>/head:refs/cbz/pr/<n>'                   # for each PR
git -C <clone> fetch origin '+refs/heads/<base>:refs/remotes/origin/<base>'         # for each stacked base
gh api 'repos/d6g8k5htny-coder/main/pulls?state=open&per_page=100' \
  -q 'map({(.head.ref): .number}) | add' > open_heads.json
# Offline analysis:
python3 -B -I pr_evidence.py analyze --git <clone> --target refs/remotes/origin/main \
  --out . --open-heads open_heads.json 333 336 338 ...
```

## Verify the generator itself

```sh
python3 -B -I test_pr_evidence.py   # 28 tests
python3 -B -I mutation_check.py     # 34/34 killed; a kill needs a suite that ran tests and reported FAILED, and a syntax-broken copy must count as invalid
```
