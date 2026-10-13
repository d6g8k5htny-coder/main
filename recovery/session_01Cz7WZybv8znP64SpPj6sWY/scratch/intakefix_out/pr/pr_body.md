## Change and purpose

`public-intake` stops rejecting a valid intake PR when its branch scan meets a branch with unrelated history. Scope: engineering only, in the trusted intake checker. Mathematics: none.

- **The defect:** `reachable()` lists the repository's branches and compares each one with the submitted source commit. For an orphan branch, GitHub answers that compare with 404 "No common ancestor". It answers 404 too for a branch deleted between the listing and the compare. `ancestor_of_branch()` let that `HTTPError` escape, so the whole check rejected with "HTTP Error 404: Not Found".
- **Where it bit:** `claude/recovery-01N-20261013` appeared at 16:21Z with its own root commit `d3c3799f`. Since then main#386's check fails one branch short of the branch that vouches for its source. The diagnosis is in [386/6153330269](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6153330269) and [307/6153336978](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6153336978).
- **The repair:** a 404 from that one compare request now means "this branch does not vouch", and the scan continues. Every other HTTP status still raises and rejects. No other request changes. A 404 alone is never reachability: if no branch vouches, the check still rejects with "not reachable from any branch".
- **Writer assignment:** [275/6153586211](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6153586211). Pickup: [275/6153705170](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6153705170).

## Current disposition

- **Head:** `66776948e9eab8600d9caa7b6a719f0aabfdd1da`, tree `dfdc219880021be050cd61abd7632d69277e3806`.
- **Base:** `main` at `e1712fc6bd06015b80b6dd175bfb7e91546a7ac3`. The head is one commit on it.
- **Changed paths:** exactly `tools/public_intake_check.py` (+11 −1) and `tests/test_public_intake.py` (+32). Both base blobs (`edc64d9a`, `d22cdd10`) are the same as at `f036dab9`.
- **Frozen candidate:** `git diff e1712fc6 66776948` is byte-identical to the patch frozen at `31a28da`. That patch is blob `8b758740`, 4,743 bytes, SHA-256 `7cb9ee64a99d026a2c3b5d040e05a50d8fead89f9f80519fa6ed2274c5faf18a`.
- **Integration owner:** none yet. Integration needs a distinct qualified integrator.
- **Claim:** the writer claim `WRITE-INTAKE404-01CZ-20261013T1945Z` was delivered and released at [275/6153751282](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6153751282). The author, Anthropic Claude session `session_01Cz7WZybv8znP64SpPj6sWY`, stays on this PR for findings and CI.
- **Completed review scopes:**
  - **Exact-head engineering review:** `PASS_ENGINEERING_SCOPED` at `66776948` by OpenAI/Codex `root01a103c8`, a nonauthor ([review 5509066001](https://github.com/d6g8k5htny-coder/main/pull/391#pullrequestreview-5509066001)). It reproduced the old-code control and the widening mutation in both modes.
  - **Supplemental security evidence:** `PASS_SCOPED_CODE` from the same reviewer ([6153861889](https://github.com/d6g8k5htny-coder/main/pull/391#issuecomment-6153861889)), covering 11 offline probes, all non-404 statuses from 100 to 599, and 404 at every non-compare endpoint.
  - **Independence:** organizational-independence credit is 0 for both. The owner's personal reading is pending.
  - The claim was released at [307/6153800865](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6153800865).
- **Unresolved findings:** none. **Next action:** a distinct qualified integrator. This PR stays a draft until then; the author does not mark it ready or merge it.
- **Retained stops:** no workflow, permission, setting or schedule change. No dispatch or re-run. No change to any recovery branch, to #386's index or to the site branch.

## Verification

Run locally with Python 3.11.15. These are the commands from `.github/workflows/public-intake-controls.yml`:

```
python3 -B -S -m unittest discover -s tests -p test_public_intake.py -v
python3 -B -O -S -m unittest discover -s tests -p test_public_intake.py -v
```

| tree | normal mode | optimized mode |
|---|---|---|
| base `e1712fc6`, base tests | 71 run, OK | 71 run, OK |
| base checker with the new tests (old-code control) | 74 run, 1 failure, 1 error | 74 run, 1 failure, 1 error |
| head `66776948` | 74 run, OK | 74 run, OK |
| head with the catch widened to every `HTTPError` (mutation) | 74 run, 3 failures | 74 run, 3 failures |

- **Old-code control:** the orphan-before-vouching test errors with "HTTP Error 404: Not Found". The test for an orphan with no vouching branch fails because the old checker rejects with the 404, not with "not reachable". The third new test passes on the old code, because the old checker already rejected those statuses.
- **Mutation:** with every `HTTPError` swallowed, the 403, 500 and 502 subtests fail. So the test rejects a blanket exception.
- `python3 -B -S -m py_compile tools/public_intake_check.py` passes.
- **Hosted checks:** all eight passed for PR head `66776948`.

| run | event | tested commit | jobs | result |
|---|---|---|---|---|
| [38457088558](https://github.com/d6g8k5htny-coder/main/actions/runs/38457088558) Public intake checker controls | `pull_request` | merge `9d0f7259b03e4a5437ebe782b93eb81f9ef603e0`, parents base `e1712fc6` and head `66776948` | `intake-controls` (74 tests in normal mode and 74 with `-O`) | success |
| [38457088483](https://github.com/d6g8k5htny-coder/main/actions/runs/38457088483) Public results intake | `pull_request_target` | the trusted checker from base `e1712fc6`, reading PR head `66776948` | `public-intake` (lane `maintainer-engineering`) | success |
| [38457088944](https://github.com/d6g8k5htny-coder/main/actions/runs/38457088944) public-shop | `pull_request` | merge `9d0f7259` | `public-shop`, `browser-smoke` | success |
| [38457089199](https://github.com/d6g8k5htny-coder/main/actions/runs/38457089199) workspace-landing | `pull_request` | merge `9d0f7259` | `research-architecture`, `formal-evidence`, `landing-checks`, `verify` | success |

GitHub reports the PR as mergeable, with merge state `clean`.

## Evidence and limits

- **Author and exposure:** written by Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`, which also authored the diagnosis and #386. Author review earns no qualification, and a same-provider read earns zero independence credit.
- **What this does not establish:** it does not show the trusted guard is safe in general. Local test runs are not a receipt. The change only alters how one compare status is read.
- **Behaviour change beyond the orphan case:** a source commit GitHub cannot find now ends as "not reachable from any branch" after the full branch scan, not as an immediate 404. It still rejects. The scan costs as much as it already does for any existing commit that no branch contains. As the reviewer noted, that is at most 301 compare calls: three listing pages of 100 branches, plus the default branch.
- **Merging this alone does not turn #386 green.** `public-intake` runs under `pull_request_target` and executes the checker from the base branch. Re-running #386's old job would run the old checker again. A fresh PR event after the merge picks up the fix.
- Scientific effect NONE. No claim, premise or label moves.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
