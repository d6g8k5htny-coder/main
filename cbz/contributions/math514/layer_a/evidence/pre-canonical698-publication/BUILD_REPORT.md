# Layer A auditor build report

The five confirmed admission gaps are repaired. The bounded auditor's current G0 is **PASS: 70 tests and 53/53 actual checker-source mutants killed by assertions, zero mutant errors**. Focused same-provider postfix review found all five original findings and the literal import extension resolved, with no remaining admission blocker in those corrections. The contribution is ready for scoped root publication; actual repository G1 and mathematical federation closure remain separate.

The current receipt is `tools/layer_a/evidence/g0-admission-final.json`, SHA256 `8f03335ff833b706445e1a6b6d8232548ece81cb439626e87275c5a6ffc98fe7`. The runner completed with exit 0 and unchanged source inputs. Its baseline, all 53 assertion-kill logs and nine checker/test/spec/README/provenance bindings were inspected in `evidence/admission-final-binding-check.json`. Original 44 mutants remain declared. The nine new mutations weaken the reviewed reflection, language, suffix, import-tail and JSON-type checks. Syntax/import/runtime errors and timeouts do not count as kills.

## Repairs and red evidence

- Engineering Python now refuses explicit builtins/namespace loader names and defined globals/class/frame reflective attributes. A hash-bound nonmathematical review cannot waive the mapped import boundary.
- Retained source needs supported language and a registered module regardless of suffix. Mapped external modules require actual supported source analysis. Known unsupported source suffixes produce `LANGUAGE_GAP` even when labeled Lean/Python.
- Lean imports retain apostrophe, `!` and `?` identifier suffixes. Unsupported or escaped operands and trailing syntax produce `LEAN_IMPORT_GAP` after a valid prefix. Ordinary/raw string and character operands are retained as invalid operand markers rather than erased.
- A7 observation matching uses canonical JSON, preserving Boolean/integer distinctions.

The original review's red run had 13 assertion failures across 66 auditor tests (`admission-review-red.log`); those cases pass after the repair. The first changed generation passed 69 total tests and 52 mutants (`g0-admission-fix.json`). A further same-scope import-tail fixture exposed three erased literal operands (`admission-import-literal-red.log`); the final generation contains that fix and the 53rd mutant. Exact first-generation source inputs remain in `evidence/pre-import-literal-fix-source/`.

## Reused evidence and provenance

The earlier 60-test/44-mutant receipt remains unchanged at `evidence/g0-hardening.json` (SHA256 `9bd3dee4b424b17b5580e7cc26af5d161418fadb93477825f22f45be7d1c3619`), with exact source inputs in `evidence/pre-admission-fix-source/`. The older 56/39 generation remains at `evidence/g0-final.json`, with its source inputs in `evidence/pre-hardening-source/`. All initial failed/surviving runs and original permission-guard survivor evidence remain preserved. The previous 174-file publication candidate is archived with its original metadata. New defects warranted the scoped changed-generation G0 runs; no unchanged mathematical verification was repeated.

The adopted public Workspace PR29 identifier lexer remains byte-identical at head `ef5b924016c0f85ced7eb1dfb8f5eb9bd776f169`, native blob `0a113896aff4c999111193359c38906ff61e52e8`, SHA256 `eb1088a53040f304b8a27b42a99d0d71b6fedf77668b4a7a5c30f0c0b8115bb0`. Its contributor 28/39 claim is reusable inherited evidence and was not rerun. Its heuristic exclusions, Git walker and standalone verdict are unused.

## Scope, commands and registry

From the complete published contribution directory:

```sh
python -B -S mutation_test.py --output /absolute/controller-evidence/g0-admission.json
python -B -S auditor.py audit --manifest /absolute/final-scope.json --output /absolute/controller-evidence/g1-report.json
```

Use fresh output paths and preserve raw historical receipts. Audit exit codes are 0 bounded `STATIC_PASS`, 1 findings, 2 input/usage error, 3 output-write error. The root supplies/authenticates the complete final manifest, classifications, immutable inventory and module identities; no implicit quarantine exclusions exist. Tool id is `CBZ-LAYER-A-EXACT-SCOPE-v1`; runtime is CPython 3.11+ standard library with the exact included lexer. `delivery/auditor/TOOLING_REGISTRY.md` is the proposed integration payload; Root685 owns canonical STATUS/GOVERNANCE/TOOLING and publication.

The exact manifest/hash/classification and mapped-isolation contract, strict closed-program integer/Fraction AST, explicit engineering review, A6 source candidates and A7 evidence bindings remain bounded. Arbitrary SAT, float/dataflow/loader completeness, semantic certificate truth, actual inventory authenticity, historical weakening, G2/G4/G7/G8 and federation closure are not established. No actual repository G1, Lean build, Git/remote mutation or mathematical check ran in this repair task.

The pre-fix independent review and completed focused postfix source review are same-provider reviews. The postfix report is `discovery/auditor-postfix-review.md`, SHA256 `fbab0e9c9c4ce6eff06022b9e319516ab9f28d82c5274cd5e83718a60f08f86b`. It reused the completed 70/53 result and integrity evidence without rerunning tests. Any provider-distinct review requirement is a process requirement and does not erase established mathematical or mechanical evidence. Reuse completed applicable checks under unchanged relevant assumptions.
