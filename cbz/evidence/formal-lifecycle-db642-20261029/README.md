# Main formal mutation lifecycle repair

The former name/count-only kill predicate admitted real setup, teardown and cleanup assertion failures. The amended harness freezes complete selected TestCase IDs before execution, records synchronous body entry/completion and normal/assertion outcome, and admits only mapped body-phase assertions with exact expected parent IDs. Both actual mutation suites use this same runner for their baseline and mutant executions.

The recorder rejects incomplete, missing, duplicate or unknown lifecycle data; incomplete, empty, duplicate or incorrect populations; errors, skips, expected failures and unexpected successes; fixture assertions; custom runner hooks/failure exceptions; coroutine, async-generator and generator bodies; and non-None dormant return shapes. It restores instance instrumentation in `finally` and does not patch global unittest methods. Standard synchronous bodies remain supported. Every subtest, including successful, foreign-parent and fixture-phase subtests, refuses; static extraction confirms all 26/22 selected production bodies are subtest-free. Body entry requires invocation of the actual selected method under dispatcher authorization outside fixture depth. Result failure callbacks must carry the identical AssertionError object that escaped that method. Arbitrary callbacks and fixture calls of the method or dispatcher refuse. Existing production failure populations are unchanged.

The four former fabricated `MutationKillContractTests` methods retain their names and meaning with 26 real synchronous bodies: intended assertion positive (plus clean positive), errors with/without intended assertion, skip/incomplete population, and survival/unrelated/extra assertions/empty failure contract. Thirteen additional methods cover the lifecycle boundaries and unsupported shapes. All 47 compiled and 22 custody prior method names are retained; only those four fabricated-control bodies and the two mutation-suite runner/admission call sites changed among preexisting methods. All 3065 protected blobs and modes match the base, and only two existing manifest hash fields differ.

## Exact reruns

From the repository root, with CPython and no third-party dependencies:

```sh
python -B -S -m unittest tests.test_formal_compiled_inventory tests.test_formal_source_custody -v
python -B -O -S -m unittest tests.test_formal_compiled_inventory tests.test_formal_source_custody tests.test_formal_execution_context tests.test_formal_gate tests.test_formal_negative_diagnostics tests.test_formal_workflow_shell tests.test_required_formal_binding_control tests.test_required_formal_check tests.test_atlas_formal_evidence_build tests.test_atlas_formal_source_bridge -v
python -B -S cbz/evidence/formal-lifecycle-db642-20261029/verify_classifier_mutations.py --output /tmp/main-lifecycle-self-mutations-normal
python -B -O -S cbz/evidence/formal-lifecycle-db642-20261029/verify_classifier_mutations.py --output /tmp/main-lifecycle-self-mutations-optimized
python -B -S tools/formal_gate_check.py
git diff --check
```

Use fresh output directories outside the repository for the source-omission verifier. Its registered executable status and exact source identities are in [REGISTRY.md](REGISTRY.md). No Lean command is needed for this harness correction.

## Observed local results

Interpreter: CPython 3.12.14, normal and `-O`, with `-B -S`. The hosted formal workflow requests Python 3.11; these local runs do not establish its runtime. The later actual CPython 3.11.17 hosted result is source-pinned in completion/README.md.

- RED on the then-unmodified predicate: the new real setup/teardown/cleanup control produced three intended outer assertion failures (`AssertionError not raised`), exit1. Genuine intended-body and clean-control positives pass on the corrected source. Peer assistance then exposed additional fixture-call/dispatcher/callback traps in the initial uncommitted recorder: three added controls produced seven intended outer assertion failures, exit1, before their correction.
- Changed modules, normal: 82 tests, 81 passes and one unavailable-Lean skip, exit0.
- Full affected 10-module formal surface, optimized: 242 tests, 240 passes and two unavailable-Lean skips, exit0.
- Source-omission self-verification: 21 clean baselines and 21 exact intended real body assertion kills per mode, exit0. Qualification results retain body/phase/population metadata and original unittest transcripts.
- Original production source mutations: all seven compiled-policy and nine custody mutants retain exact reason-bound kills in both modes, with 26/22 selected controls respectively. Separate observation captures retain every baseline and mutant transcript, selected ID, phase, body outcome, expected/actual failure ID and result state.
- Source-only gate and whitespace check pass. Manifest SHA256 is recorded in the registry.

The two existing skips are `PinnedLeanInventoryTests.test_actual_clean_and_omitted_constants` (`pinned Lean unavailable`) and `RepositoryControls.test_real_package_executes_with_receipt` (`Lean toolchain not installed`). They are skips, not passes. No Lean process, replacement runtime, kernel rerun, hosted run or federation execution was performed. Historical component/kernel results remain historical and do not qualify these new manifest bytes.

Raw commands, interpreter versions, stdout/stderr, returncodes, source-omission JSON/logs, production observations and preservation checks are retained by the controller in the task workspace at `cbz/implementation/main-lifecycle/` for exact-source review and packaging. These include the initial green-attempt infrastructure error (StringIO was not a unittest writer), corrected before qualification. Earlier 79-test/239-test and 16-omission observations apply to the superseded uncommitted recorder and are retained as development history, not qualification of the final bytes. The registry is an additive handoff for the canonical writer, not an edit to root scientific/status/governance authorities.
