# Task 1 implementation evidence

Author: OpenAI / Codex implementation subagent `/root/receipt_repair`. Engineering implementation and author-side checks only; no scientific review or organizational-independence credit.

Base: `4ef1867175dbe1072a54a02e95f80c4e2c70a4ff`; worktree `cbz/worktrees/main-cbz`, branch `codex/cbz-zero-20261029`.

## Change

Fresh gate receipts record the actual `GITHUB_RUN_ATTEMPT` (null when absent), and `bind_receipt` requires that field to equal the strictly validated current attempt. The live interface and CLI expose no legacy mode or fallback. Same run, source, manifest and logs cannot rescue an absent or stale attempt.

The Atlas builder now has an explicitly historical route requiring all three fixed original SHA-256 identities: archive, manifest and receipt. It validates original log bytes and returns only the original fixed 3b39/run40246431034/attempt1 binding. No environment parameter or caller-selected identity exists. The unchanged binding is compared against the unchanged archive binding by the real assembler. Tests verify arbitrary/stale/attempt-inserted legacy receipts, changed manifest/archive/logs, strict live refusal of the actual old receipt and exact reproduction of published historical evidence.

Frozen bridge-profile tests previously copied current live controls under old manifest hashes. They now read exact base gate/test bytes from explicitly literal `.txt` fixtures. Those bytes are never imported or executed; the current validator/assembler executes. Both old profiles retain their exact manifests and published evidence. No new source bridge or execution/alignment transfer is inferred for the repaired manifest.

`formal/manifest.json` changes exactly two existing control-file hashes. All other manifest metadata, proof statements and bound mathematical files remain unchanged. Historical zip, receipt and published evidence module are byte-identical to base (see `identity-verification.json`).

## TDD and mutation

- `red-binder.log`: 33 tests, 11 expected failure subcases because missing/stale/malformed receipt attempts were accepted by the original binder. New fixture's valid attempt is 2.
- `red-producer.log`: two expected failures because actual attempt 7 and local null attempt were omitted from serialization.
- `red-historical.log`: after strict live repair, historical positive assembly refused the old receipt; one transient live-source test also failed before mechanical manifest rebinding.
- `red-historical-route.log`: three route tests (five failure cases) establish the missing explicit route before implementation.
- `red-frozen-profiles.log`: both frozen profiles failed current control hashes, motivating exact historical fixtures rather than rewriting old manifests.
- `mutation.log` / `mutation-results.json`: an isolated copy removes only the new workflow attempt comparison. Three adverse regression methods produce 11 expected failure subcases (exit 1, no errors). Mutation is killed; checked-in source is untouched.

## Focused verification

Source-only gate exit 0: `SOURCE_IDENTITY_PASS`; manifest SHA-256 `49360aa6588167f2e4cd49e90ba0e8a6dfce6c47a11f1f09ce64def51e0e8590`. This is source identity only.

| Suite | Exit | Result |
|---|---:|---|
| normal-test_required_formal_check | 0 | Ran 33 tests in 0.093s; OK |
| normal-test_formal_gate | 0 | Ran 51 tests in 0.398s; OK (skipped=1) |
| normal-test_atlas_formal_evidence_build | 0 | Ran 13 tests in 0.160s; OK |
| normal-test_atlas_formal_source_bridge | 0 | Ran 20 tests in 4.467s; OK |
| normal-test_formal_workflow_shell | 0 | Ran 6 tests in 0.066s; OK |
| optimized-test_required_formal_check | 0 | Ran 33 tests in 0.090s; OK |
| optimized-test_formal_gate | 0 | Ran 51 tests in 0.408s; OK (skipped=1) |
| optimized-test_atlas_formal_evidence_build | 0 | Ran 13 tests in 0.128s; OK |
| optimized-test_atlas_formal_source_bridge | 0 | Ran 20 tests in 14.145s; OK |
| optimized-test_formal_workflow_shell | 0 | Ran 6 tests in 0.076s; OK |

Commands and exact outputs are in `focused-results.json` and individual normal/optimized logs. Formal-gate real package execution is skipped because its normal discovery cannot find an installed runnable toolchain. A separate direct runtime probe returned `error: failed to locate application`; the controller observed host executable restrictions and instructed no bypass. No local kernel PASS is claimed. Hosted required checks remain separate work.

## Exact historical fixture identities

| Literal data fixture | Bytes | SHA-256 | Git blob at base |
|---|---:|---|---|
| `tests/fixtures/atlas_formal_bridge_profiles/formal_gate_check.py.txt` | 23845 | `1125e435429f0a7391d6b076c99f1198bf862376276f6c00c91b5ea4f6a88ec5` | `263d8674c9d8270d6e113f91b78c37dbcaa36c6a` |
| `tests/fixtures/atlas_formal_bridge_profiles/test_formal_gate.py.txt` | 26219 | `2b13ad83b03d2b45f4be70ce03af597c195563a1ed0bd3983a2b7040068a3a84` | `348bb503ba736187a819577d7d84bd43d700b65d` |

## Full discovery and additional positive-fixture repair

Initial repaired discovery: `full-discovery-first.log` / `full-results-first.json`, 1551 tests in 374.135s, exit 1, one error, 24 skips. The sole error was `test_required_formal_binding_control.RequiredFormalBindingControl.test_valid_receipt_binds_to_commit_run_and_bytes`: its synthetic positive receipt lacked `workflow_run_attempt` despite environment attempt 2. `red-additional-fixture.log` reproduces that exact failure in the seven-test suite. The fixture now supplies the actual attempt string 2; all existing assertions remain. Normal and optimized seven-test suites both pass (individual logs and updated `focused-results.json`). No production logic changed for this fixture repair. Final complete discovery now passes: 1551 tests in 386.305s, exit 0, 24 skips, no failures or errors (`full-discovery.log` / `full-results.json`). The initial failed run remains preserved. All 24 exact skip names/reasons appear in `full-results.json`; skip coverage is unchanged in count from baseline. Expected negative-control diagnostic prints after the summary do not represent failed tests.

## Remaining verification / integration

Final verification is complete. Controller supplied baseline: 1541 tests, 24 skips, exit 0, `cbz/discovery/main-regression-initial.log`. The repaired suite adds 10 test methods and runs 1551 tests, with the same 24 skips. Thus final passing methods are 1527 versus baseline 1517. No failing tests or assertions were deleted or disabled. No remote writes, pushes, foreign branch changes or scientific status changes were performed by this worker. Controller owns review and hosted integration.

## Commit and review

Implementation committed as `bae3c5866656ad99e88b5839f585e47d8a1fce40` (parent `4ef1867175dbe1072a54a02e95f80c4e2c70a4ff`), exactly 12 task files. Parent-owned root `DECISIONS.md`, `cbz/` evidence and design/plan files remain untracked and were excluded. `git diff --cached --check` passed before commit; the commit content and remaining status were inspected afterward. Full task membership is recorded in `completion.json`.

Separate engineering reviewer OpenAI / Codex `/root/receipt_review` accepted the stable staged implementation with no critical, important or minor findings (`REVIEW.md`). Its review inspected code/spec, exact source/historical byte identities and recorded mutation/focused test results. It did not rerun tests or claim scientific/organizational independence. The final full discovery completed after that review; code did not change afterward.

Focused verification totals: 130 tests in each interpreter mode, with one real-Lean test skipped per mode (258 passing executions plus 2 skipped). Source-only gate passed separately. Final full discovery: 1551 run / 1527 passed / 24 skipped / 0 failed or errors. Mutation: 3 adverse methods / 11 expected assertion-failure subcases, exit 1, mutation killed. No local kernel PASS and no hosted-check result are claimed here; controller owns ordinary PR publication and actual hosted verification.
