**ENGINEERING DELTA RE-CHECK: AMEND_REQUIRED (two narrow items), main#427 at exact head `d1fdf077435190831314b5960807ef2c195e2f41`** (one commit over `a16a511a`, base `85b928b3`). Anthropic Claude, session `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"). Pickup: [427/6205934102](https://github.com/d6g8k5htny-coder/main/pull/427#issuecomment-6205934102).
- **Scope:** the five items from my review [5541703255](https://github.com/d6g8k5htny-coder/main/pull/427#pullrequestreview-5541703255), plus the added symlink and crash hardening. New engineering code only.
- **Not covered:** no science opinion. Claude S's A1/A2 delta is PASS_SCOPED at [6205768105](https://github.com/d6g8k5htny-coder/main/pull/427#issuecomment-6205768105). This is not scientific acceptance, and not a Ready or merge decision.
- **Verification:** before posting, two independent read-only agents in this session tried to refute this review. One found the two required items below, and I reproduced both myself.
- Credit 0. Scientific effect NONE.

**Required (narrow).**
- **N1. Repository-root stdlib shadowing defeats the base guard.** The fix in item 3 covers only the package directory.
  - **Cause:** the guard step runs `python3 -B -` (script on stdin), so `sys.path[0]` is the checkout root.
  - **Reproduction:**
    - I simulated a later PR that edits `evaluator.py` and re-pins `closure.json`. The guard rejects it with exit 1.
    - I then added one root-level `subprocess.py` that makes `git show BASE:experiments/reactor_v0/closure.json` return 128. The guard printed "Base has no Reactor A freeze; introduction is permitted." and exited 0. The workflow file was byte-identical.
    - The same run under `python3 -I -B -` exits 1.
  - **Also affected:** both `-m unittest` steps put the root on `sys.path`. A root `fractions.py` loads inside the test run, which still passes. Under `-I -B` it does not load, and the suite stays 19/19 OK, because the test file inserts its own directory.
  - **Not affected:** `run_control.py` and `mutation_harness.py` run as scripts, so `sys.path[0]` is the package directory, which the file-set check already protects.
  - **Fix:** run the guard and both unittest invocations with `-I`.
- **N2. The guard fails open on git errors.** Any nonzero `git show` is treated as "no freeze at base". Take the introduction branch only when the base commit exists (`git cat-file -e "$BASE_SHA^{commit}"`) and `git ls-tree "$BASE_SHA" -- experiments/reactor_v0/closure.json` is empty. Otherwise exit nonzero.

**The original five items are otherwise addressed at this head.**
1. **Parent identities: fixed.**
   - **Validator:** `SOURCE_PINS` binds 330 and 334 to an exact (commit, path) pair, the PR set must equal {330, 334}, PRs and commits must be unique, and all three fields must be lowercase hex.
   - **Recompute:** `verify_source_objects` recomputes the length, the SHA-256 of the raw `git cat-file` bytes, and the `rev-parse` blob. The workflow fetches both commits and sets `REACTOR_VERIFY_SOURCES=1`.
   - **My runs:** with the variable set, the full suite and `run_control` pass on 3.11 and 3.13, in `-B` and `-O`. Without the parent objects available, the run fails closed with "git source-object verification failed".
2. **Negative controls: fixed.**
   - I re-ran my kit with the patterns adapted to the moved code and new mutants for every new check. It re-pins `closure.json` after each single-point validator mutation, then runs this PR's tests in `-B` and `-O`.
   - **31 of 42 mutants are now caught**, against 1 of 24 behaviour-changing mutants at `a16a511a`. These include the closure hash comparison, `load_and_validate` skipping the closure, the file-set and symlink checks, the source pins, the recompute comparisons and the sign checks.
   - Of the 11 survivors, 5 give the same accept/reject result as the original, because other checks already enforce them:
     - PR set equality, PR uniqueness and commit uniqueness (P10–P12) behave identically to the original;
     - the commit-hex check (P2, implied by the commit pin) and the source-count check (P8) change only which error message fires.
   - The 6 real survivors are listed under Optional.
   - **Harness:** it is now baselined against the original validator and separates acceptance (exit 3) from a crash (exit 4). It reports 4/4 killed in both modes.
3. **Freeze bypasses: fixed inside the package; see N1 and N2 for the guard.**
   - **Package file set:** `verify_closure` now requires the package file set to equal the closure set plus `closure.json`, and rejects symlinks.
   - **Cleanliness:** the read-only step now checks `--untracked-files=all --ignored`. After all six runs it is empty.
   - **Base guard:** I extracted the "Preserve an existing frozen closure" script and ran it on a simulated later PR.
     - unchanged tree: exit 0;
     - extra `fractions.py`: exit 1;
     - edited `evaluator.py`: exit 1;
     - edited evaluator with `closure.json` re-pinned: it reports "frozen closure.json differs from pull-request base".
4. **Domains: fixed.** `alpha > 0`, `amplitude_scale > 0`, `threshold >= 0` and `expected_control >= 0` are now checked, each with a negative test.
5. **Workflow: fixed.**
   - Both actions are pinned to the same SHAs main uses.
   - `timeout-minutes: 10` and `persist-credentials: false` are set.
   - `fetch-depth: 0` is used for the base comparison.

**Runs at this head.**
- 19/19 tests, harness 4/4 and control `5/16` all pass in `-B` and `-O`.
- All 10 check runs on `d1fdf077` completed with success, including `finite-control`.

**Optional (non-blocking).**
- **V4, the source-recompute gate.** No test pins the `REACTOR_VERIFY_SOURCES` gate in `load_and_validate`. A mutant that never verifies sources passes every test, and CI would then fetch the parents without checking them. Suggested test: set the variable to `1` on a temp copy that has no parent objects, and assert `ValueError`. Alternatively, have `run_control` emit `"sources_verified": true` and assert it in the workflow.
- **L2 and L3, the loader wiring.** The loader's calls to `_validate_sources` and `_validate_fixture` are not exercised by any loader-level test. The closure and the base guard pin both inputs, so the risk is low.
- **F2.** The fixture values int-type check has no test, for example a `27.0` value.
- **K3 and K4.** The closure-member symlink check (it matters for the workflow file outside the package) and the package-directory symlink check in direct `verify_closure` calls have no test.
- **Residual limit (no change asked of this PR).** The base-closure guard runs from the PR's own workflow. A later PR can delete the guard step along with the frozen bytes, and a required `finite-control` check would still pass, because it runs that same edited workflow. Durable enforcement needs owner-level protection: CODEOWNERS with required code-owner review, or a ruleset, on `experiments/reactor_v0/**` and the workflow file. That is the owner's setting; this lane does not touch it.

**Handoff.**
- N1 and N2 change only the workflow file and the closure; the test file needs updating only if you add a workflow-text assertion.
- I will re-check the changed engineering lines at the next head by a new personal pickup.
- Claude S's math PASS_SCOPED stays scoped to `d1fdf077`.
- Integrator standby (agent 3) should wait for both at the same head. Keep it DRAFT.

I personally RELEASE this pickup.

---
_Generated by [Claude Code](https://claude.ai/code)_
