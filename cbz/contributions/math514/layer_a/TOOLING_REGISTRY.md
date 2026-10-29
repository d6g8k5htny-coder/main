# Tooling registry payload — exact-scope Layer A auditor

Root685 owns canonical STATUS/GOVERNANCE/TOOLING integration. This contribution is a proposed bounded auditor entry, not a federation green result.

| Field | Value |
|---|---|
| Tool id | `CBZ-LAYER-A-EXACT-SCOPE-v1` |
| Contribution location | `cbz/contributions/math514/layer_a/` |
| Alternative admitted location | `tools/cbz/exact_scope/`, preserving the complete layout |
| Runtime | CPython 3.11+ standard library with exact public vendored lexer |
| Contract / schema | `SPEC.md` / `README.md` |
| Purpose | G0 auditor selftests/source mutants; bounded G1 exact manifest/hash/classification/mapped isolation, exact Python AST and A6/A7 binding |
| Current affected-scope verification | Current composed evidence: 56 distinct methods freshly exercised in affected scopes (45 lexer +8 added lifecycle +3 retained harness methods),27 unchanged auditor methods reused;19 affected scanner mutants per mode plus38 unchanged historical scanner mutants, with grader safeguard observations separately version-bound. No fresh whole83-method/57-scanner-mutant run is claimed.; see sealed fresh/reused accounting, no fresh full-suite claim |
| Historical Main#697/027 verification | 70 tests; 53/53 actual checker-source mutants killed by assertions; zero errors, at its exact recorded source |
| Current composed receipt | `evidence/composed-admission.json` |
| Lexer component receipt | `evidence/lexer698-admission.json` |
| Historical receipt | `evidence/g0-admission-final.json` |
| Historical receipt SHA256 | `8f03335ff833b706445e1a6b6d8232548ece81cb439626e87275c5a6ffc98fe7` |
| Earlier generations | 56/39 `g0-final.json`; 60/44 `g0-hardening.json`; first repair 69/52 `g0-admission-fix.json`; exact input snapshots preserved |
| Actual repository G1 | NOT_RUN; root final manifest/authenticated inventory required |
| Mathematical/scientific effect | NONE; no mathematical check or closure claimed |

Historical focused same-provider postfix review resolved its five findings and literal operands at the old source cut. That review did not cover the two later canonical-lexer integration gaps or certify successor bytes. Report: auditor-postfix-review.md. Current successor review/admission remains separately recorded.

From the complete admitted directory:

```sh
python -B -S mutation_test.py --output /absolute/controller-evidence/g0-admission.json
python -B -S auditor.py audit --manifest /absolute/final-scope.json --output /absolute/controller-evidence/g1-report.json
```

Choose fresh controller output paths. Raw receipts retain execution-environment paths; `EVIDENCE_PATH_MAP.json` maps them to portable contribution-relative files. Audit exits: 0 bounded STATIC_PASS; 1 findings; 2 input/usage; 3 output-write error. G0 exits 0 only when baseline and every declared mutant satisfy the exact assertion-kill rule; syntax/import/runtime errors and timeouts are not kills. Rebuilding/copying this package does not run G0 or any mathematical check.

The repaired rules conservatively refuse explicit reflective loader names/attributes in both Python scopes, enforce supported language/module metadata for all retained and mapped external source, reject known unsupported suffixes even when mislabeled, read plain Lean prime/!/? imports completely and refuse unsupported/escaped/literal import operands/tails, and distinguish JSON booleans/integers in A7 observations. The original 44 source mutants remain; nine new mutants cover those repairs. Exact retained finding multisets and mapped direct/transitive quarantine isolation remain enforced. A nonmathematical review cannot waive source import isolation.

The public lexer is the unchanged canonical Main#698 `tools/cbz/layer_a_scan.py`, head `38312e16f7ae22aeb433823c4a00a2cc80e23d0b`, native blob `2eee3a2200082444093935291b9389c0795641cf`; SHA256 and byte count are bound by `LEXER_PROVENANCE.json`. A minimal seven-field metadata adapter repairs two concrete old-engine misses: numeral-adjoining axiom declarations and nested quoted terms in `throwError` interpolation. Its walker/context/standalone verdict remain unused. Existing actual Lean counterexample receipts and contributor 72-test/two-mode plus 66 lexical/23 A3 mutant evidence are reused; no compiler or mathematical replay is claimed. Fresh checks cover the affected Lean integration and designated mutants; exact fresh/reused counts come only from `evidence/lexer698-admission.json`. The old 70/53 receipt is retained, not relabeled as a new full run.

Concrete separate obligations: G2 authenticates all elaborated owned declarations, including private/generated/macro-injected forms and exactly the approved imported axioms propext/Classical.choice/Quot.sound. G3/G4 actually build/kernel recheck all closed declarations/binaries. G5/G6 preserve/execute actual regression and semantic mutations. G7 authenticates dependency/runtime/fractions pins. G8 authenticates execution receipts, review/certificate truth, statement nonvacuity, classification truth and historical bounds/test retention. G10 authenticates complete final inventory and actual immutable trees. Root owns final admission/G0–G11. Arbitrary Lean SAT, float/dataflow/dynamic-loader completeness, alias expansion and semantic proof validity are not claimed.

Completed applicable verification is reusable on unchanged inputs and assumptions. New defects/changes justify affected checks; a new agent, handoff or pending provider/process review does not justify mathematical rechecking. Same-provider source review and any provider-distinct process requirement remain separate from mathematical/mechanical evidence.


A mutation kill requires a complete, externally bound selected unittest population and exact direct/body-subtest assertion provenance. Lifecycle failures, unsupported identity/population states, errors, skips, expected failures, and incomplete or inconsistent receipts are not kills. Historical body-trace evidence is version-bound and retained separately from fresh structured runner receipts.

Current composed evidence: 56 distinct methods freshly exercised in affected scopes (45 lexer +8 added lifecycle +3 retained harness methods),27 unchanged auditor methods reused;19 affected scanner mutants per mode plus38 unchanged historical scanner mutants, with grader safeguard observations separately version-bound. No fresh whole83-method/57-scanner-mutant run is claimed. See `evidence/composed-admission.json`, `evidence/existing-harness-current-receipt.json`, `evidence/lifecycle-grader-repair/LIFECYCLE_REPORT.md` and the two scoped source reviews. Three dynamic one-method source controls and10+3+2 generation-bound grader safeguard observations are separate from the83-method inventory and57 scanner definitions; the repeated duplicate-population control is not counted as another distinct production mutant.
