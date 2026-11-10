# Q9006 Original Real-Record Live Freshness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task after Root accepts the design and reads the actual RED result. Steps use checkbox syntax for tracking.

**Goal:** Make the existing RealRecordTests.test_real_records_fresh_against_live_source_head execute the unchanged authenticated production CLI in a mandatory native prepared normal/optimized shard, while ordinary offline discovery retains an explicit missing-prerequisite skip.

**Architecture:** Add two fixed targeted steps to the existing live-freshness job; retain its existing companion CLI runs. The original method launches the actual CLI as a subprocess in the same interpreter mode and checks the original retained receipt. No helper import, supplied observer, report-import shortcut, provider override, anonymous fallback, record edit or new CLI is introduced.

**Tech Stack:** Existing CPython 3.11, stdlib unittest/subprocess, Git, GitHub Actions, existing production GitHub HTTPS observer.
**Spec:** Design and constraints below in this unreferenced staging artifact. No source/ref/workflow changes or execution are authorized by producing this artifact.

## Design

Frozen source: Main 984a608584281bca8db88a8a9ae7a837f6e9a6bb, tree 134218620682f9467961f6a8eeb29fac670e9457.
- tests/test_living_project_validation.py: blob 87d63d104ab29387ddb3f7236d76b2e835e205d8; original method L453-463 always skips.
- .github/workflows/node-status-live-contract.yml: blob 20c74a9cb86b885298ad4d1ebe572430955d9e85; live-freshness job L70-120 already has real github.token, read-only contents/issues/pull-requests permissions, checkout and Python.
- tools/node_status_live_check.py: blob 1e4852188f15060176b4458f58b27815f56845fd.
- tools/node_status_refresh.py: blob 5fa7c8b6647938a0aca538a7735edede65de9ed5.

Approaches: Root prescribed executing the existing CLI from the original method, which is the smallest valid approach. Importing evaluate_fresh_records would expose the injected-observer boundary; consuming a previous companion report would not execute the original method's live observation. Both alternatives are excluded.

The CLI has no workflow-name/job-name restriction. It requires GITHUB_ACTIONS=true plus exact repository, checked commit, positive native run/attempt and interpreter mode (L317-330); binds actual HEAD/tree and nine committed files before executing bound dependencies (L333-362, L427-444); requires a nonblank token; uses exactly run_live's default ContentObserver; rebinds after observation (L455-488). The observer's default API is HTTPS api.github.com with bearer authentication (refresh L79-89, L154-155). Token presence or synthetic environment values alone earn no execution credit.

The nine protected source inputs are CLI, refresh, validator, architecture/node-status.schema.json, docs/site/current.json, and all four records/node_status JSON files. The original method source is a tenth workflow-bound input; it is not added to or substituted for the CLI's nine-source contract.

## Global Constraints

- Modify only the original method/imports in tests/test_living_project_validation.py and the two targeted steps plus retention paths in .github/workflows/node-status-live-contract.yml.
- Preserve all other existing method assertions, all original source pins, all existing offline contract commands/count guards, both existing companion CLI steps, and their retained outputs.
- Do not modify CLI, refresh, validator, records, schema, status, ownership, holds or scientific dispositions.
- Preserve parent -B/-S and optimized -O; CLI child uses the same sys.executable and explicit matching -B/-S/-O. No -I/provider/auth bypass is added.
- Real GITHUB_TOKEN is inherited only from the existing step's github.token; never print it, serialize environment values, replace it, or add a token-bearing command argument.
- All original logs, child exit outcomes, JSON receipts and source snapshots live outside checkout under RUNNER_TEMP, at distinct new normal/optimized paths. No TemporaryDirectory cleanup of retained evidence.
- Missing mandatory prerequisites, timeout, transport/auth refusal, source drift, existing output, malformed receipt, stale/unknown fresh candidate, or skipped selected method must fail; none can become a passing or applicability result.
- At the frozen 984a6085/944dc555 input cut there are three declared-fresh nodes and one declared-stale eng.hard-gate record. This task edits no records. Cohort/status/source-count checks derive dynamically from the actual source-bound records; a future genuine peer update may make hard-gate fresh and must be evaluated as such. The test never promotes or fabricates a status.
- This closes only the original Q9006 method's execution gap; Q9003 owner and Q9004/Q9005 interface/data blockers and full federation G5 remain separate.

## Review Focus

1. Mandatory flag set but context/token/output absent must FAIL before any optional skip.
2. Offline invocation must skip explicitly and initiate no network request.
3. Zero child exit without the current native FRESH receipt, complete fresh cohort and nine equal source bindings must fail.
4. Normal and optimized child modes/output paths must not be interchanged or reused.
5. Source drift, timeout and nonzero child exit must retain originals before failure; neither an expected adverse result nor a previous success may overwrite current evidence.

## Interfaces

- New test-only activation variable: NODE_STATUS_LIVE_REQUIRED. Absent means offline/unprepared and permits the explicit missing-prerequisite skip; exactly "1" requests mandatory execution; any other present value is a configuration failure.
- New test-only output variable: NODE_STATUS_LIVE_OUTPUT. Mandatory mode requires an absolute, new JSON file outside checkout, with an existing non-symlink parent. The CLI remains the final no-follow/new-leaf admission authority.
- Child command: [sys.executable, "-B", optional "-O" matching sys.flags.optimize, "-S", REPO/"tools/node_status_live_check.py", "--root", REPO, "--output", output].
- Selected unittest ID in both steps: tests.test_living_project_validation.RealRecordTests.test_real_records_fresh_against_live_source_head.
- Fixed outputs: RUNNER_TEMP/q9006-real-record-normal.json and RUNNER_TEMP/q9006-real-record-optimized.json; sibling .stdout.log, .stderr.log and .exit.json retain child originals. Unittest/source/step results use distinct q9006-original-normal-* and q9006-original-optimized-* paths.
- Existing workflow artifact remains immutable run/attempt-named; append only these fixed evidence patterns to its always-retention step.

## Task 1: Observe the actual original-method RED before strengthening it

**Files:** Modify .github/workflows/node-status-live-contract.yml only for this phase. Keep original test blob 87d63d104ab29387ddb3f7236d76b2e835e205d8 unchanged.

- [ ] Add two fixed normal/optimized steps in existing live-freshness after its companion CLI steps, each with if: ${{ !cancelled() }}, real github.token, NODE_STATUS_LIVE_REQUIRED=1 and its distinct NODE_STATUS_LIVE_OUTPUT.
- [ ] Each step captures native HEAD/tree plus exact working/committed byte, blob, mode and SHA256 bindings for the CLI's nine inputs and original test source before and after its invocation, on failure as well as success. Retain both snapshots; require equal before/after and native GITHUB_SHA.
- [ ] Run exactly one selected method: python3 -B -S -m unittest tests.test_living_project_validation.RealRecordTests.test_real_records_fresh_against_live_source_head -v; optimized adds -O. Retain original unittest log, raw parent exit and guard result before returning.
- [ ] Require parent exit zero; exactly one result for the complete expected ID with disposition ok; exactly one Ran 1 test summary; one final nonempty line exactly OK. Reject skip/expected-failure/unexpected-success/failure/error, duplicate/foreign identity, extra result/footer, incomplete output, or source drift. Use explicit exceptions/exit checks, not Python bare assert.
- [ ] Append exact new evidence paths/patterns to existing always-upload. Preserve existing artifact name and companion outputs.
- [ ] After Root authorizes this implementation phase, observe a genuine native run. Expected RED: both unchanged original method logs report SKIP, strict guards FAIL, and there is no original-method CLI receipt/network-exercise credit. Separate preexisting companion CLI receipts remain separate evidence.
- [ ] Stop this implementation phase until Root independently reads the actual RED originals. No mocked/local environment is an acceptable substitute for this RED.

## Task 2: Strengthen the original method and observe actual normal/optimized GREEN

**Files:** Modify only tests/test_living_project_validation.py imports and the original method. Keep all CLI/dependency/record/schema leaves unchanged.

- [ ] Keep _require_real_records() as the mandatory data loader. Evaluate activation before optional fresh-cohort skipping: absent activation produces the explicit offline/prepared-context skip; malformed activation or mandatory mode with any missing native context/token/output fails.
- [ ] In mandatory mode require a nonempty actual fresh cohort, real native context values, nonblank inherited token and the new external output path. Do not fabricate a cohort, HEAD, timestamp, observation, receipt or identity.
- [ ] Execute the exact actual CLI command above. Stream child stdout/stderr into exclusive new external files; retain child return/timeout outcome before asserting success. Use a bounded child timeout of 420 seconds and step timeout of 8 minutes so ordinary timeout failures can retain originals before host termination. Do not retry.
- [ ] After a zero child exit, require empty stderr and exact equality between retained CLI output JSON bytes and child stdout. Parse one JSON object and require schema cbz.node-status-live-cli.v1, status FRESH, scope NODE_RECORD_FRESHNESS_ONLY, scientific_effect NONE, cbz_green false and no refusal code.
- [ ] Require identity equal to current native repository/SHA/run/attempt and actual parent optimize mode. Require equal checked commit/tree endpoints, exact nine-source roster, equal before/after source bindings and their matches to actual selected source bytes. Preserve workflow's tenth-input test-source binding.
- [ ] Require freshness.records_total equal actual records; fresh_candidates == fresh_checked == actual fresh-cohort size; exact sorted actual fresh nodes; exact actual nonfresh node/status list; nonzero source observation count matching actual declared-fresh sources. Check recorded input/schema/graph hashes against actual inputs. The unchanged CLI remains responsible for production validation, live HTTPS object/timestamp comparisons, observation multiplicities and source rechecks.
- [ ] Smoke-check offline invocation of the exact original method: explicit skip and no network. Check NODE_STATUS_LIVE_REQUIRED=1 with absent prepared prerequisites: failure, never skip, and no claimed live credit. These are negative prerequisite controls only, not native success evidence.
- [ ] Run existing tests.test_node_status_live_check and tests.test_node_status_live_cli unchanged normally and optimized; retain their existing 23/20 census guards.
- [ ] Observe genuine current native normal/optimized targeted execution. Require each original one-method log to end with plain OK, zero skips/failures/errors and matching mode. Retain distinct original CLI reports/stdout/stderr/exits, ten-input snapshots, parent outcomes and strict guard outcomes.
- [ ] Independently read actual original outcomes and receipt identities/counts before any completion claim. A source review, green aggregate badge, previous companion report or artifact metadata alone does not qualify.
- [ ] Recheck current head/tree and exact two-path diff. Integrate only after Root's separate review/authorization; no fullG5/CBZ/scientific award follows from this component.

## Self-review

All five review-focus conditions have explicit verification steps; no test/provider/transport weakening or data invention is proposed. Existing output admission remains unchanged. The fixed native shard is a legitimate component in a future full-population final run; its success does not remove Q9003-Q9005 or retroactively alter historical skip counts.

Source citations:
- https://github.com/d6g8k5htny-coder/main/blob/984a608584281bca8db88a8a9ae7a837f6e9a6bb/tests/test_living_project_validation.py#L453-L463
- https://github.com/d6g8k5htny-coder/main/blob/984a608584281bca8db88a8a9ae7a837f6e9a6bb/.github/workflows/node-status-live-contract.yml#L70-L120
- https://github.com/d6g8k5htny-coder/main/blob/984a608584281bca8db88a8a9ae7a837f6e9a6bb/tools/node_status_live_check.py#L317-L362
- https://github.com/d6g8k5htny-coder/main/blob/984a608584281bca8db88a8a9ae7a837f6e9a6bb/tools/node_status_live_check.py#L408-L489
- https://github.com/d6g8k5htny-coder/main/blob/984a608584281bca8db88a8a9ae7a837f6e9a6bb/tools/node_status_refresh.py#L79-L89

## Accepted Task 1 source staging observation — 10 November 2026

This plan was accepted for Task 1 source staging only. No native execution, publication, ref, PR, original-test strengthening or runtime claim has occurred.

Fresh staging base: Main 944dc555e38c4a2912d613276e1750976e6675d5, tree 275c6fb32e1ce5b6db05c6007146442ee3d8b991. Existing workflow remains blob 20c74a9cb86b885298ad4d1ebe572430955d9e85 and original test remains exactly blob 87d63d104ab29387ddb3f7236d76b2e835e205d8. The nine protected production/input leaves were individually read at this base and still match the earlier source cut.

Proposed Task 1 workflow leaf: 78d8269bf0884ac601bd82f9784eddfdbb174cbc. Two fixed normal/optimized steps are inserted before existing live retention, with the required step-context runner.temp bindings. The existing always-retention gains only the four fixed q9006 mode prefixes. Removing those exact additions reconstructs all 5,150 original workflow characters exactly, preserving existing companion/offline steps, 23/20 guards, actions, permissions and settings. The two inline drivers differ only in their fixed mode/optimization constants; each launches the one exact original method with matching -B/-S/-O, captures ten working/committed byte/mode/blob/hash bindings before and after, retains raw parent/strict outcomes, and prints the critical originals. Actual YAML/Python parsing and actual R0 remain unexecuted.

Ruling: use API-backed unreferenced source blobs and this observation as the staging ledger; do not create a filesystem/worktree/commit or run the plan's execution commands in this source-only phase. Root must read the staged source before any publication. A later publication includes this owned plan at docs/cbz/Q9006_ORIGINAL_METHOD_PLAN.md and a truthful DECISIONS entry; GOVERNANCE/TOOLING stay unchanged. The eventual cumulative scope is workflow, original test, owned plan and DECISIONS, preserving every nonowned leaf and peer canonical prefix at that later fresh base.

Expected first R0 remains one original-method SKIP per mode and strict guard FAIL, with no original-method live/network exercise credit. Task 2 stays blocked on Root's independent read of actual native R0 originals.

## Task 2 staged after independently read actual R0 — 10 November 2026

Root independently read native run 42757945583/attempt1, contract job 129636600300 and live job 129636872010. Actual tested source was signed merge 6d0645bd851973696ff4afdee24ccb17d192687f/tree97d08f92edaeea48a2185f4a09b63e4560b48a6a, with parents [2f878f8eb5c621dcf9231e4a1bf582b685321130,329376996218ff6576c699a61f931d2cf290fc5d]. The 86 offline outcomes passed; both separate authenticated companions were FRESH. The unchanged original method skipped once per mode in 0.001s with parent exit zero; each strict guard failed only EXACT_ONE_ID_PLAIN_OK_CONTRACT_REFUSED. Four ten-input native snapshots matched. This is the preserved original-method skip deficiency, not live credit; unreferenced observation report b12f16364e67aff2b5d3b388a7cd07138b84ab98 retains the readback summary and original skip text.

Root authorized original-method strengthening. Source staging base is Main 964390a94c08f00f119dc3b2e0be16180c1544e2/tree6677c6cf4744f5136c7a4c67bf4f3a74d869cff8; existing draft head remains 329376996218ff6576c699a61f931d2cf290fc5d. Proposed original-test leaf is 3bcd5829ea60e8e8ac58979bc7f33e4fd7838201. Only that original method and necessary hashlib/os/stat/subprocess/sys imports change: inverse replacement reconstructs all 24,970 original source characters exactly. Workflow remains exactly 78d8269bf0884ac601bd82f9784eddfdbb174cbc, with all nine production/input leaves unchanged at this stage.

The actual CLI subprocess uses same sys.executable and matching -B/-S/-O, timeout420, existing --root/--output interface, real inherited native/auth environment and no retry/override. New external output/stdout/stderr/exit evidence is retained exclusively before result assertions. The original real CLI v1 shape was read from actual companion output; checks bind actual identity/endpoints/nine file hashes, byte-identical stdout/file, empty stderr, FRESH/NONE/cbzfalse and dynamically derived current fresh/nonfresh cohorts/source multiplicities. No success fixture, fake head, status promotion, owner edit, pass print or method docstring is introduced.

Workflow-preserving negative smoke route: the mandatory path runs five bounded controls of the same nested admission guard (absent flag -> SkipTest; malformed flag, missing native context, removed token and removed output -> AssertionError). These calls never invoke any CLI/provider/network and never serialize their environment dictionaries. Their type-only results are retained separately in the owned exit metadata with scope ADMISSION_ONLY_NO_LIVE_CREDIT. This does not recursively invoke unittest or the original method. Actual offline discovery calls the same guard and retains an explicit prerequisite-absence skip; the existing navigation job can separately observe that ordinary offline behavior. These admission controls are not fresh-source or full-G5 credit.

Task 2 source is staged for Root's read only: no ref/commit/publication, parser, smoke execution or native GREEN has occurred. Future composition must keep ordered parents [fresh compatible Main, old32937699], preserve the full latest Main canonical prefix plus the exact prior Q806 decision suffix, and CAS the owned ref from old32937699 without force. No base-only push. Root source GO is required before publication; a genuine normal/optimized one-method plain-OK run and final independent source/actual review remain pending.
