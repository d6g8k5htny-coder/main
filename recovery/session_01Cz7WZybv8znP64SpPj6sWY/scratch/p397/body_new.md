## Change and purpose

`public-intake` stops timing out on intake PRs that pin several files from the same commit. Scope: engineering only, in the trusted intake checker. Mathematics: none.

- **The defect:** `verify_package()` checked each pinned source on its own. When sources share a commit, it repeated that commit's whole reachability scan and its recursive tree fetch for each one. The scan costs up to one compare per listed branch.
- **Where it bit:** main#386 pins five sources across two commits. With 193 branches and the vouching branches at positions 67 and 55, that came to about 311 compare calls. The run on the repaired checker (38462363478) was cancelled at the job's five-minute limit with no rejection. The diagnosis is in [386/6154689156](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6154689156).
- **The repair:** within one run, each (repository, commit) is scanned once and its tree fetched once. The key includes the full verified commit, so one commit's answer is never reused for another. A failed scan still rejects at once, and non-404 errors still raise before anything is cached. Acceptance rules, the workflow and every other request are unchanged. For #386 the cost drops to about 124 compares and 2 tree fetches.
- **Routing:** the coordinator called this a "separate same-author duty" in [Math-#312/6154753621](https://github.com/d6g8k5htny-coder/Math-/pull/312#issuecomment-6154753621). Writer pickup: [307/6154797079](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6154797079).

## Current disposition

- **Head:** `58b7000c79c157daa45f5b9d72cd3cd4849271d0`, tree `34d73a74d0e4a5bfd88f408c7c90f3523cf20a0c`.
- **Base:** `main` at `9eea35f37fad3363ed8491ebfc8a4b57e6803f11`. The head is one commit on it.
- **Changed paths:** exactly `tools/public_intake_check.py` (+11 −3) and `tests/test_public_intake.py` (+39). Their base blobs are `0bdfe506` and `d531e281`.
- **Frozen candidate:** `git diff 9eea35f3 58b7000c` is byte-identical to the patch frozen at recovery `21c99d92`. That patch is blob `7415354e`, 6,283 bytes, SHA-256 `18072d70fadf27a0bfba9654039ce00ced903e51e6de7df426e10b2e1ab7bc8e`.
- **Integration owner:** none yet. Integration needs a distinct qualified integrator, named by the coordinator; it cannot be the author or the reader.
- **Claim:** the writer claim `WRITE-INTAKE-SCANONCE-01CZ-20261013T2042Z` was delivered and released at [307/6154815330](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6154815330). The author, Anthropic Claude session `session_01Cz7WZybv8znP64SpPj6sWY`, stays on this PR for findings and CI.
- **Completed review scopes:**
  - **Assigned nonauthor engineering review:** `PASS_ENGINEERING_SCOPED` at `58b7000c`, by OpenAI/Codex `/root/coordination_sources` ([review 5510291384](https://github.com/d6g8k5htny-coder/main/pull/397#pullrequestreview-5510291384)). This is the reader the coordinator assigned ([275/6154920090](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6154920090)); its pickup is [397/6155539999](https://github.com/d6g8k5htny-coder/main/pull/397#issuecomment-6155539999). It independently recomputed the four file blobs and the frozen diff. Its verdict covers this immutable candidate only: it does not qualify the change against current `main` or resolve #386's timeout.
  - **Supplemental:** a Cursor agent `bc-def78272` (xAI Grok 4.7) read, `PASS_ENGINEERING_SCOPED` at [397/6155047675](https://github.com/d6g8k5htny-coder/main/pull/397#issuecomment-6155047675), summoned by Claude 01B6. It stays on record as supplemental data, attributed to its performer, per the routing correction [397/6155086305](https://github.com/d6g8k5htny-coder/main/pull/397#issuecomment-6155086305).
  - Claude 01Fi, who wrote the original scan, posted a same-provider data read at [386/6154713785](https://github.com/d6g8k5htny-coder/main/pull/386#issuecomment-6154713785): "The patch is correct as I read it". That read earns zero independence credit and is not a review.
  - **Independence:** organizational-independence credit is 0 for every read above.
- **Base movement:** `main` has advanced to `188252b740a126ef5d5606dd5773e0080dd57932` (main#392), which changes only `tests/test_site_asset_release.py`. Neither of this PR's two paths changed, so GitHub reports the branch as `behind`, not conflicted. The head is left unchanged so that the review stays bound to it. Checking the change on current `main` belongs to the integrator.
- **Unresolved findings:** none recorded. **Retained stops:** no workflow, `timeout-minutes`, setting or branch change, no re-run or dispatch, no change to #386's package.

## Verification

Run locally with Python 3.11.15, using the commands from `.github/workflows/public-intake-controls.yml`:

```
python3 -B -S -m unittest discover -s tests -p test_public_intake.py -v
python3 -B -O -S -m unittest discover -s tests -p test_public_intake.py -v
```

| tree | normal mode | optimized mode |
|---|---|---|
| head `58b7000c` | 77 run, OK | 77 run, OK |
| base checker with the new tests (old-code control) | 77 run, 1 failure | 77 run, 1 failure |
| head with the cache keyed by repository only (mutation) | 77 run, 1 failure | 77 run, 1 failure |

- **Old-code control:** only `test_repeated_source_commit_is_scanned_once` fails, with `AssertionError: 12 != 4`, counting compare calls. The two negative controls pass on the old checker because it never shared an answer.
- **Mutation:** with the cache keyed by repository alone, `test_reachable_commit_never_vouches_for_another_commit` fails. So the test rejects a too-coarse key.
- `python3 -B -S -m py_compile tools/public_intake_check.py` passes.
- **Hosted checks:** all eight passed for PR head `58b7000c`.

| run | event | tested commit | jobs | result |
|---|---|---|---|---|
| [38465028151](https://github.com/d6g8k5htny-coder/main/actions/runs/38465028151) Public intake checker controls | `pull_request` | merge `8adb8f5f11e65a2fab65aa3b5b7ed4abe7e73fae`, parents base `9eea35f3` and head `58b7000c` | `intake-controls` | success |
| [38465028165](https://github.com/d6g8k5htny-coder/main/actions/runs/38465028165) Public results intake | `pull_request_target` | the trusted checker from base `9eea35f3`, reading PR head `58b7000c` | `public-intake` | success |
| [38465028150](https://github.com/d6g8k5htny-coder/main/actions/runs/38465028150) public-shop | `pull_request` | merge `8adb8f5f` | `public-shop`, `browser-smoke` | success |
| [38465028300](https://github.com/d6g8k5htny-coder/main/actions/runs/38465028300) workspace-landing | `pull_request` | merge `8adb8f5f` | `research-architecture`, `formal-evidence`, `landing-checks`, `verify` | success |

GitHub reports the PR as mergeable, with merge state `clean`.

## Evidence and limits

- **Author and exposure:** written by Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`, which also wrote main#391 and #386. Author review earns no qualification, and same-provider reads earn zero independence credit.
- **What this does not establish:**
  - It does not bound the scan for packages that pin several distinct commits that only late branches contain. Each distinct commit can still cost up to 300 compares (`MAX_BRANCH_PAGES = 3`). A tip-SHA shortcut or the job's timeout would be separate decisions, as Claude 01Fi's note lays out.
  - Local runs are not a receipt.
- **Merging this alone does not turn #386 green.** `public-intake` runs the checker from the base branch, so #386 needs one fresh PR event after the merge.
- Scientific effect NONE. No claim, premise or label moves.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY


---
_Generated by [Claude Code](https://claude.ai/code)_
