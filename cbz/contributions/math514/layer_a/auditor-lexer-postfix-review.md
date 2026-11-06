# Layer A canonical-lexer integration review

The frozen lexer component resolves the two inherited false-green cases within the bounded source-auditor contract. This review found no remaining blocker in the canonical-source adoption, seven-field adapter, or preservation of the existing manifest/import/vacuity rules. It is a component review, not final package admission: the separately identified generic mutation-grader lifecycle defect and its repair remain outside this conclusion.

This separate-agent, same-provider review read the changed production source, original/current test and mutation-definition diffs, scoped controller, red/green logs, current mutation receipts and assertion traces, provenance, and retained canonical Lean outputs. No test, mutation, compiler, proof, project build, actual-repository G1, Git operation, or remote action was executed by the reviewer. The only reviewer write is this report. Completed applicable evidence was reused under `/workspace/AGENTS.md`; neither a new reviewer nor a pending process requirement erased established mathematical evidence.

The review cut is the implementer's production freeze and sealed component receipt `tools/layer_a/evidence/lexer698-admission.json`, SHA256 `0749e207efa46545ffbb2c1ad73fb1305dbf7b73f052deeaa02fb84bd1adb3b4`. It records `G0_COMPONENT: PASS`, explicitly scoped to lexer integration with lifecycle augmentation pending. Subsequent grader changes require their own affected review; they do not require replaying unchanged lexer or mathematical checks.

| Reviewed production input | Recorded SHA256 |
|---|---|
| `tools/layer_a/canonical_lean.py` | `9ed661b1490f73c4dc3664a33ed688e937c1c7d8fecfb4d48c5349fa41823818` |
| `tools/layer_a/source_checks.py` | `931a6ee9ad92b3833c342af52d978b732faa692f2a0fd819f75e9941790c647c` |
| `tools/layer_a/auditor.py` — unchanged | `c0a32e827279dfb9164d540c1f99c2269e3c49446a3fbeae23afefb53af54710` |
| `tools/layer_a/tests/test_auditor.py` | `94532912255f92250046e0e85d2633b438245e635499d815340edfe4bb29890e` |
| `tools/layer_a/mutation_test.py` — before lifecycle repair | `e6afa7a0ef6b3e5cd9ce3afb5a4784ecf5f3163c2ed36fef4a46a517f238b721` |

These are the completed scoped receipt's production/test bindings, reused here rather than rehashed. The initial receipt also binds its historical mutation generator; the later two definition adaptations are recorded separately and must not be described as the identical initial generator.

`LEXER_PROVENANCE.json` now identifies Main#698 head `38312e16f7ae22aeb433823c4a00a2cc80e23d0b`, `tools/cbz/layer_a_scan.py`, native blob `2eee3a2200082444093935291b9389c0795641cf`, 24,721 bytes, and the canonical-source SHA256 above. Its custody receipt records an exact, unmodified copy at `evidence/public-main698-38312e16/layer_a_scan.py`. The prior ULW29 source/provenance and 70/53 receipt remain explicitly historical. Published #698 verification is labeled reused contributor evidence; this contribution did not rerun or independently recreate that work.

The adapter consumes the canonical seven-field tuples and refuses unsupported tuple shape with `LEAN_LEXICAL_GAP`. It passes an error collector to the canonical lexer and forwards distinct canonical diagnostics as lexical gaps. The initially missing error-channel propagation was caught during the in-progress review, has an expected assertion red log, and is present in the frozen source. Escaped-name normalization and source-location conversion remain unchanged. Ignoring the seventh contextual hint for hazard policy is conservative: this auditor already refuses every canonical `native` identifier. It does not inherit the canonical standalone classifier, walker, path contexts, exclusions, or default verdict.

The secondary token stream still supplies the original literal sentinels, restricted import parsing, theorem/lemma/example locations, and bounded contradiction candidates. Its implementation is unchanged. `auditor.py` is unchanged, retaining complete supplied-path/hash/classification checks, exact retained-finding counts, mapped and transitive quarantine isolation, supported-language requirements, evidence bindings, and exact vacuity-annotation coverage. The new hazard stream supplements these independent checks. No import-map, empty-premise, retained-source, or vacuity waiver was introduced.

| Inherited case | Current evidence and disposition |
|---|---|
| `def harmless := 0axiom hiddenPremise : False` followed by `bad : False` | Canonical numeric spelling stops before `axiom`; the adapter emits `LEAN_AXIOM`. The manifest regression supplies an ordinary `bad@2` conditional annotation with an explicit empty premises list and requires both that hazard and `FAIL`. Its red receipt records an actual prior `STATIC_PASS` with no findings. Restoring the old numeric swallowing recreates that false green and fails the designated test-body assertion. |
| `throwError "bad {let x := "safe"; (sorry : Nat)}"` | Canonical conservative brace-hole lexing reaches `sorry` through the nested quoted term, and the adapter emits `LEAN_PLACEHOLDER`. The manifest regression supplies a benign mapped `Lean` import and requires both the placeholder and `FAIL`. Its red receipt likewise records the old `STATIC_PASS` with no findings. Removing code-hole scanning recreates that false green and fails the designated test-body assertion. |

The retained canonical compiler evidence is applicable and was not rerun: `numeral-command-lean.log` reports `'bad' depends on axioms: [hiddenPremise]`; `macro-string-lean.log` warns that the declaration uses `sorry` and reports `[propext, sorryAx, Classical.choice, Quot.sound]`. Source/log blobs are respectively `46435e50ab67663f75ad008b77d772babbc17c66` / `dd8796143faf23302ae7253e9bab316c76187478` and `3d94783f5cb651e385662eabe17b7d5bde12a6ff` / `b5dd24a938368c170b28afb1ce8b7cd29d20099e`. Copies, immutable source URLs, and the canonical independent review are retained in `evidence/public-main698-38312e16/`.

The other new tests cover numeral/radix/scientific boundaries, benign literals, and the adapter's shape/error channels. They do not claim that each lexical fixture is a freshly compiled Lean program. Ordinary string brace data, unsupported Unicode spellings, and prime notation can still be refused conservatively. Such extra findings can require retained-count updates or additional certification; they do not establish a new false green. This review found no additional concrete false green. The updated README and specification describe the canonical engine, adapter, conservative brace-hole interpretation, exact diagnostic deduplication, fresh/reused counts, and separate lifecycle obligation consistently with this scope.

The fresh/reused accounting is appropriately scoped:

| Evidence | Applicability |
|---|---|
| 45 affected methods, normal and optimized modes | 40 original methods plus five additions; both recorded baselines exit 0 with zero assertion failures/errors. The temporary selected suite preserves complete original method bodies. All original 67 auditor method ASTs remain unchanged in the complete source. |
| 19 affected current mutation definitions per mode | 17 valid initial assertion kills are reused; only the two adapted definitions were rerun. All 19 current records in each mode report positive assertion failures and zero errors, totaling 191 assertion failures per mode. |
| 30 historical methods | 27 unaffected auditor methods plus three historical runner methods retain their recorded evidence. This is not a fresh 75-method full-suite run or evidence of the newly uncovered lifecycle requirement. |
| 38 historical mutation results | Reused for unchanged rules. Current active definitions total 57: 51 unchanged legacy definitions, two adapted legacy definitions, and four additions. This is not a fresh full 57-mutant run. |

`evidence/lexer698-initial/receipt.json` honestly retains `FAIL`, 17/19 kills, and both original redundant survivors in each mode. The production lexer/adapter/test bindings were stable during that run. Canonical #698's identifier implementation made the old excluded-letter mutation ineffective; canonical error propagation made the secondary-only malformed-character mutation redundant. The current adaptations target the effective Unicode continuation rule and the final malformed-character refusal result. Their names do not imply unchanged mutation bytes. Original definitions/sources and survivor logs remain preserved; `evidence/lexer698-adapted-mutants/receipt.json` records only the two affected reruns. This is the correct reason to perform those additional checks.

The four new actual checker-source mutations and the two adaptations have the following recorded assertion counts in each mode:

| Current mutation | Assertion failures | Test-body evidence |
|---|---:|---|
| `restore-generic-numeral-swallowing` | 6 | Manifest false green and numeral-boundary expectations lose `LEAN_AXIOM`. |
| `discard-nested-string-code-holes` | 2 | Manifest false green and direct nested-string expectation lose `LEAN_PLACEHOLDER`. |
| `discard-canonical-lexical-errors` | 1 | Canonical error-channel refusal expectation loses `LEAN_LEXICAL_GAP`. |
| `ignore-canonical-metadata-shape` | 1 | Unsupported six-field metadata refusal loses `LEAN_LEXICAL_GAP`. |
| adapted `canonical-unicode-boundary-weakened` | 1 | Original Unicode-boundary test loses its placeholder finding. |
| adapted `skip-malformed-character-refusal` | 5 | Original manifest/direct malformed-character assertions lose refusal. |

The reviewer read these six normal-mode failure traces directly. They end in named `test_*` bodies and the expected assertions, including the original two actual manifest false greens. The sealed component receipt additionally records observed test-body frames for all 38 mode-specific current kills, with no setup/teardown frame. All current records have zero errors. Those actual traces support these finite kills despite the separate generic grader flaw.

The generic `mutation_test.py` accounting at this review cut can mistake setup/teardown/cleanup assertion failures for behavioral kills. Therefore this report does **not** certify its future grading contract, infer lifecycle correctness from the three historical harness methods, or treat a generic `KILLED` label alone as sufficient. The separate lifecycle repair and new targeted harness evidence must be admitted independently. The exact observed test-body failures above remain valid evidence; the newly found process/tooling defect does not erase them.

The historical `g0-admission-final.json` remains the 70-test/53-mutant receipt with SHA256 `8f03335ff833b706445e1a6b6d8232548ece81cb439626e87275c5a6ffc98fe7`, and the earlier red, failed/surviving, 56/39, 60/44 and 69/52 generations remain generation-specific evidence. None is rewritten as a result for the new bytes.

Actual-repository G1 remains `NOT_RUN`. The authenticated final scope, isolated context/module inventories, canonical registry, serial integration, and final package/source bindings remain Root-owned. Arbitrary macro expansion, generated declarations, semantic satisfiability, execution-receipt truth, kernel checking, axiom policy, and federation G0–G11/CBZ-0 remain separate obligations. This same-provider engineering review supplies no provider-distinct or scientific-review credit and makes no mathematical promotion or closure claim.
