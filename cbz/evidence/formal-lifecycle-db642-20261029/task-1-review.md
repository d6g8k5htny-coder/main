# Task 1 independent spec and quality review

Reviewed base `7b1657700aca5f8e9c391513c7974009688945ab` through head `d676b961bc016110fa990de549a3d07888d5af24`, using the supplied 996-line review diff as the primary source view.

**Spec Compliance: compliant.**

**Task Quality: approved.**

No actionable correctness or scope findings were identified. These verdicts cover the bounded Main formal mutation-lifecycle task and its retained local Python evidence. They do not replace the controller's whole-branch review or establish hosted CPython 3.11, Lean/kernel, federation, or scientific acceptance.

## Findings and source-line review

Severity: none; no fix requested. The following are checked implementation findings, not defect reports. Paths below are relative to `/workspace/scratch/db6428847655/cbz/worktrees/main-lifecycle/`.

| Source | Review finding |
| --- | --- |
| `tests/test_formal_compiled_inventory.py:326–331`, `:460–475`, `:495–505` | Selection is frozen before execution. Admission compares the caller's expected full IDs, the runner's frozen selection, the ordered recorded population, and the actual count. Empty/duplicate expected IDs and incorrect/incomplete populations refuse. Exact TestSuite admission also avoids silently trusting a custom suite lifecycle. |
| `tests/test_formal_compiled_inventory.py:341–411` | The actual selected method is wrapped separately from the dispatcher. Entry requires dispatcher authorization, body phase, zero fixture depth, and no previous entry. The wrapper calls the captured method synchronously and records completion in `finally`; dispatch without actual method entry cannot qualify. Setup/teardown/cleanup tracking preserves fixture phase even on exception. Closure capture of each original hook and phase uses default parameters; each test's record belongs to its own `startTest` invocation. |
| `tests/test_formal_compiled_inventory.py:349–358`, `:369–383`, `:398–404` | Nonstandard lifecycle hooks/failure exceptions and coroutine, async-generator, or generator selected bodies are refused. Non-None return values also make the record unsupported; returned coroutine/generator objects are closed. Standard synchronous production bodies retain support. |
| `tests/test_formal_compiled_inventory.py:375–378`, `:428–457`, `:507–514` | Direct failure admission requires the identical AssertionError object that escaped the wrapped method. A fixture assertion or manually supplied caught assertion is not promoted into a proven body failure. Events must be body-phase failures belonging to a recorded object, and correspond in order and identity to unittest's failure entries. All subtests generate a nonadmissible event, including successful subtests. |
| `tests/test_formal_compiled_inventory.py:478–523` | Errors, skips, expected failures, unexpected successes, wrong recorder types, unknown/missing record or event fields, unsupported/incomplete bodies, unknown events, and unmapped failures fail closed. A baseline additionally requires no failure/event and normal completion of every body. |
| `tests/test_formal_compiled_inventory.py:526–536` | The preexisting named-failure contract is retained and tightened: each expected method name must resolve uniquely within the selected full IDs, and the observed full failure IDs must equal that set without duplicates. |
| `tests/test_formal_compiled_inventory.py:420–426`, `:469–473` | Instance instrumentation is restored in the runner's `finally`, including exceptional execution. No global unittest monkey-patch or dependency was introduced. |
| `tests/test_formal_compiled_inventory.py:539–789` | The four old contract methods retain their names and negative/positive meaning while using real TestCases. Added controls cover fixtures, mixed body/fixture failures, fixture calls of the method/dispatcher, arbitrary callbacks, class/module fixtures, expected-failure mechanisms, malformed population/lifecycle data, strict subtest refusal, dormant bodies, and custom lifecycle paths. Post-execution field-corruption controls supplement real execution; they are not the sole admission evidence. |
| `tests/test_formal_compiled_inventory.py:827–842`; `tests/test_formal_source_custody.py:270–288` | Both production suites use the same recorder and strict baseline/kill admission. Baseline IDs are frozen before consumption and reused for mutants. The diff leaves the seven compiled and nine custody mutation definitions and expected failure populations unchanged. |
| `formal/manifest.json:49`, `:56` | Only the two authorized existing test-source SHA256 values change. No policy or mathematical manifest field changes. |
| `cbz/evidence/formal-lifecycle-db642-20261029/verify_classifier_mutations.py:43–123`, `:126–160`, `:171–216` | The verifier exercises actual inner TestCases, then certifies clean outer baselines and exact intended outer assertion failures through the unchanged recorder. All 21 final omissions are explicit, including coupled guard removals where appropriate. Anchors must be unique. Failures/errors/skips outside the intended outer body cannot count as kills. Output is restricted to a fresh directory outside the repository. |
| `cbz/evidence/formal-lifecycle-db642-20261029/REGISTRY.md:5–22`; `README.md:9–37` | The auxiliary verifier is explicitly registered as executable future-use verification, with identities, commands, bounded qualification, source-exposed peer assistance, and limitations. The documents neither claim exhaustive mutation adequacy nor authenticate consistently forged private result data. Local skips and runtime limitations remain explicit. |

## Concrete checks performed

1. Read the task review brief, task implementation brief, full task report, and all of the supplied review diff. Read the changed recorder/admission code with numbered source lines. No unrelated unchanged source was opened; no further focused runtime scenario was necessary to resolve a concrete code doubt.
2. Inspected `git diff --name-status` for the exact base/head pair: only the two authorized test files, manifest, and six scoped evidence/docs files changed. Inspected `git status --porcelain=v1`: clean. Recomputed the four published source hashes locally; all match the report and registry:
   - compiled tests: `1e171f45a75b88ded8dc5d758edc835841546080489b01c38b38b876c7896fea`
   - custody tests: `6a45d063643362939ae703c7ba594c160e480f61dc0d797fbabf287f19850313`
   - manifest: `77679ec1514124f7c144a113f63514202029b53361dd5e4102ab07d8670a35f7`
   - verifier: `a40a282624e67d4682b79f9b91577286c68063127ad7dea770c11c030531339f`
3. Read `preservation-final.json` and its retained capture metadata. It reports all 47/22 preexisting method names retained, 60/22 current methods, identical mutation definitions, 3,065 unchanged protected blobs/modes, and only the permitted manifest hash fields changed. The exact source diff independently agrees with the scope and preservation claims.
4. Parsed both final `self-mutations-stable-{normal,optimized}/summary.json` records. Each binds the current compiled-test hash and contains exactly 21 cases. Every case has a clean one-body baseline and one exact `SafeguardControl.test_rejects_regression` mutant failure, with entered/completed/stopped records, body-phase failure events, no errors/skips, and qualification true. Normal/optimized flags are 0/1. The superseded 16-case observations were not used.
5. Parsed both `production-stable-*.json` files. Each has 18 observations: two clean baselines, seven compiled mutants with 26 selected bodies each, and nine custody mutants with 22 each. All admissions are true; ordered recorded IDs equal selected IDs; selected populations are unique; all bodies entered/completed/stopped/supported; events are body-phase failures; error/skip/expected-failure/unexpected-success counts are zero. Recorded failure populations retain the documented original 1/3/5/1/1/6/1 and 5/2/1/2/2/1/1/3/2 sizes and named contracts.
6. Read final raw command metadata and log tails: normal affected surface is 82 tests with one skip; optimized full formal surface is 242 with two skips; both exit 0 under CPython 3.12.14. The final source-only gate records the current manifest hash, 31 targets, `SOURCE_IDENTITY_PASS`, and alignment still `PENDING_INDEPENDENT_REVIEW`. Committed whitespace/status logs contain no output and their report records success.

No completed suite, production mutant experiment, 21-case self-experiment, or unchanged mathematics was rerun. No source was modified, no nested agent was used, and no external action was taken. Only this requested review report was written.

## Remaining limits

The retained skips are unavailable local Lean checks, not passes. Local observations concern CPython 3.12.14 in normal and optimized modes; hosted CPython 3.11.16 and kernel/federation execution remain unverified here. The recorder's trusted implementation is the provenance boundary: this review does not claim authentication against arbitrary coordinated forging of its private internal structures. This limitation is documented and consistent with the bounded task.
