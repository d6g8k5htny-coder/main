# Stale open PR disposition evidence

**Historical evidence, not current PR state or governing instructions.** The
generated records below were reproduced from the frozen API/body collection
and exact head objects at the recorded cut. Main #482 has since been superseded
and closed. Old author staging notes and the first-pass heuristic do not impose
approval/freeze rules; current GOVERNANCE.md and the owner's CBZ directive
govern every actual disposition. This tool grants no proof, gate or merge approval.

This is CBZ-0 Phase 2 preparation, recorded on issue #684 by unit `CBZ0-P2PREP-STALE-PR-DISPOSITIONS-01SHCRXV`. It
covers the 27 open `main` PRs that had no activity between 26 and 29 October 2026, read at `main` `81acc060` on
29 October. One of them, #331, was closed during the read, so the files cover 26.

| File | What it is |
|---|---|
| `CURATED.md` | The recommended disposition for each PR, with a manual rationale. Start here. |
| `QUARANTINE_DRAFTS.md` | Draft proof-index quarantine entries for #336, #340 and #429, in the governance fields. |
| `DISPOSITIONS.md` | The mechanical first pass: rules only. |
| `dispositions.json` | Per-PR evidence: head, base, stack parent, ancestry, per-path content comparison with `main`, trial merge result, Layer A signals, review identities and the verdict class, and check runs at the head. |
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
- **No quoted third-party text is published.** The local `raw/` files hold verbatim PR bodies and comments and are not committed. `fetch` regenerates them, at the time it runs.

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
python3.11 -B -S -m unittest discover -s . -p 'test_*.py' -v
python3.11 -B -O -S -m unittest discover -s . -p 'test_*.py' -v
python3.11 -B -S mutation_check.py
python3.11 -B -O -S mutation_check.py
```

The repaired source passed 35 tests in each Python 3.11.16 mode. The adjacent
main-obligation-census tests passed 9 per mode. The mutation command covers
40 disposition-tool mutants, 6 citation-checker mutants and 3 harness mutants;
all 49 were killed in each mode. A kill requires the named assertion failure,
complete test count and no test errors/skips or launch/syntax failure.

`fetch` always refreshes raw metadata, paginates/reconciles checks, preserves
reviewer IDs/logins, and refuses a PR changed during acquisition. `analyze`
refuses a different PR number or local/API head before recommendations. Public
review identities describe the account, not model independence. Original
generated records are retained under `historical/`; current historical output
adds identities from the immutable discovery collection. Raw bodies remain
local. `repair-evidence/` preserves the observed RED/GREEN/mutation results and
input hashes. Citation regeneration remained byte-identical to its original
20-row/31-target output. Optimized Python now performs explicit checks.
