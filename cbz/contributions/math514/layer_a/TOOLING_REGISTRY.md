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
| Current verification | PASS: 70 tests; 53/53 actual checker-source mutants killed by assertions; zero errors; source unchanged |
| Current receipt | `evidence/g0-admission-final.json` |
| Receipt SHA256 | `8f03335ff833b706445e1a6b6d8232548ece81cb439626e87275c5a6ffc98fe7` |
| Earlier generations | 56/39 `g0-final.json`; 60/44 `g0-hardening.json`; first repair 69/52 `g0-admission-fix.json`; exact input snapshots preserved |
| Actual repository G1 | NOT_RUN; root final manifest/authenticated inventory required |
| Mathematical/scientific effect | NONE; no mathematical check or closure claimed |

Completed focused same-provider postfix review: all five findings and literal operands resolved; no remaining scoped admission blocker. Reports: auditor-postfix-review.md

From the complete admitted directory:

```sh
python -B -S mutation_test.py --output /absolute/controller-evidence/g0-admission.json
python -B -S auditor.py audit --manifest /absolute/final-scope.json --output /absolute/controller-evidence/g1-report.json
```

Choose fresh controller output paths. Raw receipts retain execution-environment paths; `EVIDENCE_PATH_MAP.json` maps them to portable contribution-relative files. Audit exits: 0 bounded STATIC_PASS; 1 findings; 2 input/usage; 3 output-write error. G0 exits 0 only when baseline and every declared mutant satisfy the exact assertion-kill rule; syntax/import/runtime errors and timeouts are not kills. Rebuilding/copying this package does not run G0 or any mathematical check.

The repaired rules conservatively refuse explicit reflective loader names/attributes in both Python scopes, enforce supported language/module metadata for all retained and mapped external source, reject known unsupported suffixes even when mislabeled, read plain Lean prime/!/? imports completely and refuse unsupported/escaped/literal import operands/tails, and distinguish JSON booleans/integers in A7 observations. The original 44 source mutants remain; nine new mutants cover those repairs. Exact retained finding multisets and mapped direct/transitive quarantine isolation remain enforced. A nonmathematical review cannot waive source import isolation.

The public lexer remains exact at PR29 head `ef5b924016c0f85ced7eb1dfb8f5eb9bd776f169`, native blob `0a113896aff4c999111193359c38906ff61e52e8`, SHA256 `eb1088a53040f304b8a27b42a99d0d71b6fedf77668b4a7a5c30f0c0b8115bb0`. Its contributor 28/39 claim is recorded inherited evidence, not rerun here. Its heuristic exclusions/Git walker/standalone verdict are unused.

Concrete separate obligations: G2 authenticates all elaborated owned declarations, including private/generated/macro-injected forms and exactly the approved imported axioms propext/Classical.choice/Quot.sound. G3/G4 actually build/kernel recheck all closed declarations/binaries. G5/G6 preserve/execute actual regression and semantic mutations. G7 authenticates dependency/runtime/fractions pins. G8 authenticates execution receipts, review/certificate truth, statement nonvacuity, classification truth and historical bounds/test retention. G10 authenticates complete final inventory and actual immutable trees. Root owns final admission/G0–G11. Arbitrary Lean SAT, float/dataflow/dynamic-loader completeness, alias expansion and semantic proof validity are not claimed.

Completed applicable verification is reusable on unchanged inputs and assumptions. New defects/changes justify affected checks; a new agent, handoff or pending provider/process review does not justify mathematical rechecking. Same-provider source review and any provider-distinct process requirement remain separate from mathematical/mechanical evidence.
