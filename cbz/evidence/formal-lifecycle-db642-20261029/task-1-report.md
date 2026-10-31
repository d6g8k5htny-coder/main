# Task 1 report: bounded Main formal mutation lifecycle correction

Status: DONE_WITH_CONCERNS (local implementation and verification complete; CPython3.11/Lean/hosted qualification remains outside this run).

Worktree: `/workspace/scratch/db6428847655/cbz/worktrees/main-lifecycle`.
Branch: `codex/cbz-formal-lifecycle-db642-20261029`.
Base: `7b1657700aca5f8e9c391513c7974009688945ab`.
Commit: `d676b961bc016110fa990de549a3d07888d5af24` (`fix(formal): bind mutation kills to executed test bodies`).

## Result and scope

The shared kill predicate no longer accepts name/count-compatible setup, teardown or cleanup assertions. Both actual production mutation suites freeze the full selected TestCase IDs, use the same real synchronous lifecycle recorder for baseline and mutants, and require complete actual-method entry/exit. A direct failure must carry the identical AssertionError object that escaped the selected method under dispatcher authorization outside fixture depth. All errors, skips, expected/unexpected successes, unsupported/custom execution shapes, unknown/missing/incomplete/duplicate records, fixture calls of the method/dispatcher, arbitrary failure callbacks, and all subtests refuse. Standard synchronous production populations remain supported.

Source edits are exactly the two authorized test files, plus their corresponding existing manifest SHA256 fields. Scoped SPEC/PLAN/DECISIONS supplied by the controller, README, REGISTRY and the executable source-omission verifier are included in the local commit. No mathematical, producer, helper, consumer, workflow, dependency/runtime pin or canonical root authority byte changed. No helpers, reviewers, nested subagents, remote posts, push or merge were used.

## Investigation, RED and correction

1. Read the exact task brief, AGENTS/routing/mandate/formal guidance and the bounded Math-composition recorder interface. The original Main predicate inspected counts and failure-name suffixes but had no lifecycle/body provenance.
2. Added a real unittest fixture regression before changing the predicate. Command:

   `python -B -S -m unittest tests.test_formal_compiled_inventory.MutationKillContractTests.test_fixture_assertions_never_qualify -v`

   `raw/red-fixtures.{json,stdout,stderr}` records exit1 and exactly three intended outer AssertionErrors (`AssertionError not raised`) for real setUp, tearDown and cleanup assertion fixtures. This was the then-current source, not the historical benign custody reproducer. The historical 22 named benign bodies remain context and were never represented as original custody assertions.
3. Added the shared recorder/strict baseline and kill admission and upgraded the four fabricated-result tests to actual TestCases. Initial green attempt `raw/green-contract-1.*` had a real instrumentation error: a raw StringIO lacked unittest's `writeln`. It was corrected by constructing the result through TextTestRunner's normal writer arguments; it is not counted as a kill or qualification. `green-contract-2/3` then passed14 methods on the intermediate source.
4. Read the controller-delivered source-pinned peer assistance Main701 comment6507242980. It exposed actual missing guards in the initial uncommitted implementation: fixture calls of the selected method/dispatcher and arbitrary result.addFailure callbacks. Added three targeted real controls before correcting them. Command:

   `python -B -S -m unittest tests.test_formal_compiled_inventory.MutationKillContractTests.test_fixture_calls_to_method_or_dispatcher_never_qualify tests.test_formal_compiled_inventory.MutationKillContractTests.test_setup_dispatcher_assertion_never_qualifies tests.test_formal_compiled_inventory.MutationKillContractTests.test_arbitrary_failure_callback_never_qualifies -v`

   `raw/red-peer-traps.*` records exit1, three test methods and seven intended outer assertion failures. The actual selected method is now wrapped; dispatch alone does not certify body entry. Fixture depth and a dispatch authorization token prevent fixture-mediated body entry, and direct failure provenance is bound to the identical escaping AssertionError. `raw/green-peer-traps.*` records17 passing contract methods.
5. Simplified to strict subtest refusal after the peer's exact static extraction confirmed all original26/22 selected bodies are subtest-free. Successful, failing, duplicate, foreign-parent and fixture subtests all refuse. No production outcome or original expectation was weakened. This peer input is source-exposed design assistance, not independent review credit.

## Original-test preservation and mapping

`preservation-final.json` and `raw/preservation-stable.*` compare the exact frozen base contracts and Git tree:

- Compiled inventory: all47 prior test method names retained;60 current methods (13 added). Custody: all22 prior names retained;22 current methods. No original methods deleted/skipped.
- The four original `MutationKillContractTests` methods retain their old assertions/negative meanings with26 actual bodies: intended assertion positive plus clean positive; test errors with/without intended assertion; skip/incomplete count; survival/unrelated/extra assertions/empty failure contract. Fake SimpleNamespace result summaries are removed.
- Only those four old control method bodies and the two production mutation-suite runner/admission call sites changed among original methods. Original assertion bodies elsewhere are unchanged.
- Original seven compiled mutations, nine custody mutations and exact expected failure populations are AST-identical to the base.
-3065 protected blobs and executable/symlink modes match the frozen base tree exactly. All other manifest fields compare equal; only the two authorized test hashes differ. The checks include theorem/axiom/target/census/codegen and runtime/dependency/protocol source policy bytes.

## Final-source commands and results

Actual interpreter: CPython3.12.14, `-B -S`; optimized runs also use `-O`. No claim is made for the hosted CPython3.11.16 runtime.

All commands ran from the worktree. `capture.py` in the external task workspace retains exact argv, cwd, interpreter, timestamp, stdout/stderr and returncode as `raw/<label>.{json,stdout,stderr}`.

| Label | Exact command after `python` | Observed result |
| --- | --- | --- |
| `affected-stable-normal` | `-B -S -m unittest tests.test_formal_compiled_inventory tests.test_formal_source_custody -v` | exit0;82 tests,81 passes,1 unavailable-Lean skip |
| `full-formal-stable-optimized` | `-B -O -S -m unittest tests.test_formal_compiled_inventory tests.test_formal_source_custody tests.test_formal_execution_context tests.test_formal_gate tests.test_formal_negative_diagnostics tests.test_formal_workflow_shell tests.test_required_formal_binding_control tests.test_required_formal_check tests.test_atlas_formal_evidence_build tests.test_atlas_formal_source_bridge -v` | exit0;242 tests,240 passes,2 unavailable-Lean skips |
| `self-mutations-stable-normal` | `-B -S cbz/evidence/formal-lifecycle-db642-20261029/verify_classifier_mutations.py --output /workspace/scratch/db6428847655/cbz/implementation/main-lifecycle/self-mutations-stable-normal` | exit0;21 clean baselines +21 exact intended body assertion kills |
| `self-mutations-stable-optimized` | `-B -O -S cbz/evidence/formal-lifecycle-db642-20261029/verify_classifier_mutations.py --output /workspace/scratch/db6428847655/cbz/implementation/main-lifecycle/self-mutations-stable-optimized` | exit0;same21/21 outcome |
| `production-stable-observed-normal` | `-B -S /workspace/scratch/db6428847655/cbz/implementation/main-lifecycle/record_production_outcomes.py /workspace/scratch/db6428847655/cbz/implementation/main-lifecycle/production-stable-normal.json` | exit0;2 clean baselines + exact7/9 original mutant admissions |
| `production-stable-observed-optimized` | `-B -O -S /workspace/scratch/db6428847655/cbz/implementation/main-lifecycle/record_production_outcomes.py /workspace/scratch/db6428847655/cbz/implementation/main-lifecycle/production-stable-optimized.json` | exit0;same exact populations and failure IDs |
| `source-gate-stable` | `-B -S tools/formal_gate_check.py` | exit0;SOURCE_IDENTITY_PASS;31 targets;alignment remains PENDING_INDEPENDENT_REVIEW |
| `preservation-stable` | `-B -S /workspace/scratch/db6428847655/cbz/implementation/main-lifecycle/verify_preservation.py` | exit0;3065 protected blobs/modes unchanged;all69 old method names and9/7 contracts retained |

`git diff --check` and `git diff --cached --check` passed before commit; `raw/committed-diff-check.*` records exit0 for `git diff 7b165770 --check`. `raw/committed-status.*` records exit0 and an empty `git status --porcelain=v1` after commit. `raw/committed-identities.*` retains exact commit metadata and changed-file summary.

Unavailable-Lean skips (not passes):

- `PinnedLeanInventoryTests.test_actual_clean_and_omitted_constants`: `pinned Lean unavailable`.
- `RepositoryControls.test_real_package_executes_with_receipt`: `Lean toolchain not installed`.

Earlier79-test/239-test runs and16-omission experiments qualify only the superseded uncommitted recorder, and are retained as development history. The fresh final82/242 and21-case observations above qualify the final source. The final source change justified repeating the bounded affected formal surface; no unrelated whole-repo, Node or Lean replay was attempted.

## Bounded certifier source-omission qualification

The scoped verifier is a registered executable future-use verification surface, not merely inert evidence. It compiles bounded source snapshots and runs actual atomic synchronous control bodies through the unmodified shared recorder/classifier. For each scenario it requires a clean complete real outer baseline, then exactly `SafeguardControl.test_rejects_regression` as the real body AssertionError under the omitted safeguard; fixture failures/errors/skips cannot qualify. The output retains the exact source SHA256, interpreter, each replacement, actual IDs/body lifecycle/event metadata and original unittest outer transcripts. Actual underlying TestCases exercise each scenario; lifecycle-field corruption scenarios alter a result only after real execution, rather than fabricating counts as sole evidence.

Final21 scenarios per mode:

- Phase admission; coupled fixture phase/provenance; arbitrary-callback provenance.
- Body entry, completion, test stop and body outcome.
- Recorded population, selected snapshot, unique population and nonempty population.
- Event parent and failure-object association.
- Foreign subtest refusal and successful subtest refusal.
- Coupled actual-method/dispatcher authorization for fixture method and dispatcher calls.
- Exact named failure contract.
- Coroutine, async-generator and generator refusal.

The three unsupported cases remove the coupled early-refusal/admission barrier; the two authorization cases remove both cooperating guards; the actual fixture case removes both phase and assertion provenance; the name case removes both name-contract guards. These are explicit bounded regression probes, not independent exhaustive mutation adequacy. Trusted runner source is the provenance boundary: completely fabricated, consistently forged internal data is not authenticated by field shape.

## Original production mutation observations

Both `production-stable-*.json` files retain each actual baseline/mutant selected ID, body entry/completion/outcome, event phase, expected failure names, observed failure IDs, errors/skips/expected/unexpected-success counts and full inner unittest transcript. The observer delegates to the unchanged production admission predicate before recording admission.

| Original mutant | Selected bodies | Expected and observed assertion failures per mode |
| --- | ---: | ---: |
| compiled `ignore_noise` |26|1|
| compiled `allow_forbidden` |26|3|
| compiled `allow_extra_constant` |26|5|
| compiled `allow_missing_module` |26|1|
| compiled `allow_duplicate_constant` |26|1|
| compiled `allow_extra_theorem` |26|6|
| compiled `allow_bad_count` |26|1|
| custody `ignore_blob_bytes` |22|5|
| custody `ignore_modes` |22|2|
| custody `accept_changed_head` |22|1|
| custody `ignore_manifest_arguments` |22|2|
| custody `ignore_loaded_sources` |22|2|
| custody `omit_untracked_inputs` |22|1|
| custody `allow_git_replacements` |22|1|
| custody `omit_initial_binding` |22|3|
| custody `omit_final_binding` |22|2|

All failures are direct, proven, body-phase assertions with exact original named populations; both baselines complete cleanly. No errors, skips, expected failures or unexpected successes occurred in the selected populations.

## Exact identities and self-review

- compiled test SHA256: `1e171f45a75b88ded8dc5d758edc835841546080489b01c38b38b876c7896fea`.
- custody test SHA256: `6a45d063643362939ae703c7ba594c160e480f61dc0d797fbabf287f19850313`.
- manifest SHA256: `77679ec1514124f7c144a113f63514202029b53361dd5e4102ab07d8670a35f7`.
- executable scoped verifier SHA256: `a40a282624e67d4682b79f9b91577286c68063127ad7dea770c11c030531339f`.

Self-review checked the full diff, instance-wrapper closure ownership/restoration, decoration preservation, direct-assertion unwind identity, fixture-depth authorization, strict subtest policy, dormant return shapes, original definitions/name contracts, selected-ID snapshots and protected source bytes. Instrumentation is local to actual TestCase instances and restored in `finally`; it does not monkey-patch global unittest methods. Only ordinary synchronous TestCases with standard lifecycle hooks and AssertionError failure exceptions are admitted; unsupported paths fail closed. Current call sites were searched: all production callers now pass frozen selected IDs, and no external tools caller depends on the removed count-only interface.

Remaining limitations/concerns: local verification is3.12.14, not hosted3.11.16; no Lean/kernel, hosted required check, federation/G0 platform execution or independent review was performed. The executable auxiliary probe is registered with exact commands/hashes for subsequent review. The parent/controller still owns independent review, raw-evidence packaging, publication and integration; no producer-custody or B-003 ownership is claimed. Historical kernel/component receipts remain historical and do not certify the new manifest bytes. No unresolved local implementation failure remains.
