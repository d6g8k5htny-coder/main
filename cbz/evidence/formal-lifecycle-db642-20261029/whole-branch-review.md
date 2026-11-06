# Final whole-branch engineering review

Reviewed range: `7b1657700aca5f8e9c391513c7974009688945ab..d676b961bc016110fa990de549a3d07888d5af24`.

Verdict: **ready for the controller's source-pinned publication and serial integration process, within the documented local Python qualification boundary. No actionable engineering findings.** This is not current-head hosted CI approval, kernel/federation acceptance, or a transfer of producer-custody ownership. It is a same-provider technical review, not organizational independence or approval authority.

## Scope and review method

Read the supplied whole-branch diff, task requirements, implementation report and progress rulings, then inspected the actual recorder, classifier, new real controls, both production mutation-suite call sites, the entire executable source-omission verifier and scoped documentation. Searched Python call sites for the changed public helper signatures to check integration completeness. Inspected the local standard-library `TestCase.run` and `_Outcome.testPartExecutor` specifically to establish that `startTest` instrumentation precedes method lookup and that direct exception callbacks occur before teardown changes the recorded phase. This inspection establishes the local 3.12 behavior only.

Compared the committed changed-file list and manifest diff to the authorized scope. Independently calculated the two test-source, manifest and verifier SHA256 values and compared them with REGISTRY and retained production/self-mutation records. Read retained command metadata and normal/optimized suite summaries, preservation implementation/output, production-observer implementation and all production/self-mutation result summaries; sampled the raw callback and coroutine omission failure transcripts. Conclusions do not rest on the report's verdict alone.

No suite, original mutation matrix, 21-case experiment, source gate or mathematical check was rerun. No additional benign scenario was necessary: the remaining questions were resolved by source and retained evidence. No worktree files, commits or external services were changed by this review.

## Severity-ranked findings

- Critical: none.
- Important: none.
- Minor actionable: none.

There are no parked or deferred code findings from this review.

## Concrete strengths and contract assessment

1. **Execution provenance is attached to the selected method, not merely a dispatcher callback.** `tests/test_formal_compiled_inventory.py:341` records identity and excludes custom lifecycle hooks/failure exceptions. The selected-method wrapper at line 362 requires dispatch authorization, zero fixture depth, body phase and first entry. The dispatcher at line 398 independently refuses fixture-mediated or repeated invocation. Body completion and outcome are recorded through exception paths. The direct `addFailure` callback at line 433 requires object identity with the AssertionError that escaped that selected method. Ordinary callbacks carrying a caught assertion therefore cannot manufacture a kill.

2. **Admission composes population, lifecycle and failure constraints.** `tests/test_formal_compiled_inventory.py:478` checks errors/skips/expected-success mechanisms, exact record/event shape, a nonempty unique frozen population, ordered selected/recorded identity agreement, completed stopped supported bodies, final cleanup state, mapped body-only failure events and their association with unittest failures. `require_policy_control` at line 518 additionally requires every body to finish normally; `require_policy_mutation_kill` at line 526 retains unique selected-name resolution and the exact declared failure set. Neither aggregate counts nor method-name suffixes alone establish admission. Guards use explicit exceptions, so optimization does not erase them.

3. **The bounded execution policy is coherent with its actual callers.** The compiled caller at `tests/test_formal_compiled_inventory.py:824` and custody caller at `tests/test_formal_source_custody.py:261` construct fresh standard suites, freeze complete IDs before execution, run baseline and mutants through the same recorder, and retain baseline IDs for subsequent comparisons. Both existing mutation definitions and expected failure populations are untouched in the source diff. The call-site search found no remaining old count-based caller. Instrumentation is instance-local and restored in the runner's `finally` at line 469; it does not replace unittest globals.

4. **Refusal boundaries are implemented rather than asserted in documentation alone.** Unsupported coroutine/async-generator/generator methods are detected before dispatch; non-None return shapes also refuse. Class/module fixture failures cannot be misidentified as selected body failures; ordinary setup/teardown/cleanup failures retain incompatible phase or incomplete lifecycle state. Skips and expected/unexpected-success mechanisms refuse explicitly. Every subtest callback, including success, creates a refusing event at line 454. That strict policy preserves these callers: the selected 26/22 populations and their original mutation outcomes remain intact, and none of their original bodies was rewritten to avoid a subtest obligation. Custom runners/suites and custom failure exceptions are deliberately outside admitted execution, not silently treated as passing bodies.

5. **The added controls exercise actual unittest behavior.** The four retained contract methods now use real 26-body fixtures while retaining their earlier positive/negative meanings; thirteen added methods cover fixture-only and mixed failures, class/module fixtures, fixture-driven method/dispatcher calls, arbitrary failure callbacks, missing/corrupt populations and records, subtests, expected failures and unsupported bodies. Corruption controls start with actual executions rather than manufactured counts. The source diff retains all original 69 method names and changes no original mathematical assertion body; the only preexisting body edits are the four former fabricated controls and the two mutation-suite runner/admission call sites.

6. **The executable verifier has a defensible, explicitly limited oracle.** `cbz/evidence/formal-lifecycle-db642-20261029/verify_classifier_mutations.py:168` runs the atomic outer `SafeguardControl.test_rejects_regression` through the untouched imported recorder/classifier. At line 196 it requires a clean real baseline before applying uniquely anchored source replacements and then requires the exact real outer assertion failure. The scenarios and each source replacement were inspected together: their retained kills represent safeguards ceasing to refuse the intended inner scenario, not accepting a startup error or fixture assertion as a kill. Its records distinguish coupled removals, retain source identity and transcripts, and do not claim an independent or exhaustive certifier proof. The fresh external output-directory restriction avoids overwriting existing evidence or contaminating the repository.

7. **Scope and source binding are controlled.** The committed range contains exactly the two authorized test files, two corresponding manifest hash replacements, five scoped documents and the scoped executable verifier. No production/proof/helper/workflow/pin/canonical-authority file appears in the committed change list. The retained preservation implementation checks prior method names, AST mutation definitions, protected blobs/modes and manifest-field equality; its output records 3065 protected blobs/modes unchanged. The committed diff corroborates the narrow edit surface. The independently recalculated source hashes match the registry and actual observation records:

| Source | SHA256 |
| --- | --- |
| compiled inventory test | `1e171f45a75b88ded8dc5d758edc835841546080489b01c38b38b876c7896fea` |
| custody test | `6a45d063643362939ae703c7ba594c160e480f61dc0d797fbabf287f19850313` |
| manifest | `77679ec1514124f7c144a113f63514202029b53361dd5e4102ab07d8670a35f7` |
| executable verifier | `a40a282624e67d4682b79f9b91577286c68063127ad7dea770c11c030531339f` |

## Evidence and reproducibility assessment

Retained final-source observations agree with the scoped documentation:

| Observation | Actual retained result |
| --- | --- |
| Two affected modules, normal CPython 3.12.14 | 82 tests, 81 passes, 1 existing unavailable-Lean skip, exit 0 |
| Ten affected formal modules, optimized CPython 3.12.14 | 242 tests, 240 passes, 2 existing unavailable-Lean skips, exit 0 |
| Classifier source omissions, each mode | 21 clean outer baselines and 21 exact intended outer body-assertion kills |
| Original production mutations, each mode | Clean 26/22-body baselines; all original 7/9 mutants admitted with exact expected populations, body/failure events and no errors, skips or expected/unexpected successes |
| Source gate | SOURCE_IDENTITY_PASS for the stated manifest; alignment PENDING_INDEPENDENT_REVIEW |

The raw metadata records cwd, argv, interpreter and return code. README supplies concrete normal/optimized affected-suite, self-verifier and source-gate commands; the verifier requires fresh output directories. Registry hashes identify the checked sources. The external production observer delegates to the unchanged admission functions before recording results, rather than replacing admission with a successful summary. Historical intermediate 79/239 and 16-omission observations are clearly excluded from final qualification. The two existing Lean skips are not represented as passes.

## Behaviors considered but declined to judge

- **Hosted CPython 3.11.16 behavior and current-head CI integration:** only local CPython 3.12.14 execution is retained. Local stdlib inspection and source review cannot substitute for the hosted runtime. Current-head composition/qualification remains a subsequent integration responsibility.
- **Lean/kernel execution, mathematical validity, federation acceptance and approval of new manifest bytes by those systems:** these were neither executed nor changed by this bounded Python amendment. Historical receipts remain historical; the source gate is not kernel evidence.
- **Authentication against hostile code that mutates recorder internals, unittest globals, result lists or instrumentation during execution, or dynamically bypasses the selected-method boundary using nonstandard object behavior:** trusted local runner/test source is the stated provenance boundary. Field validation and source-omission experiments do not constitute a sandbox or cryptographic proof of arbitrary in-process data. No such behavior is required by the frozen production callers.
- **Positive support for custom lifecycle runners/suites, asynchronous/lazy bodies, non-None-returning methods and subtests:** this review judges their refusal and practical compatibility with the selected populations, not execution semantics for supporting those deliberately excluded shapes. Future support would be a new implementation obligation.
- **Exhaustive mutation adequacy or an independently implemented certifier oracle:** the 21 reviewed source-omission probes are bounded regression evidence; some intentionally remove cooperating barriers together. They do not establish every possible classifier mutation is killed.
- **Remote publication state, organizational independence, producer-custody ownership transfer or canonical integration approval:** no external actions or authority checks were part of this read-only review. Existing Root ownership and serial integration remain in force.

## Readiness and runtime limits

The code and retained evidence support the intended correction for the existing synchronous 26/22 populations and original 7/9 mutation contracts. No source correction is requested before proceeding with the controller's additive publication/integration workflow. This readiness statement is confined to the exact reviewed range and local harness evidence. It must not be promoted into all-green hosted 3.11, Lean/kernel, federation or canonical B-003 acceptance, and it does not qualify later composed source bytes automatically.
