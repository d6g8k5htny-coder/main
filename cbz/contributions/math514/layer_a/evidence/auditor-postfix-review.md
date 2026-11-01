# Layer A focused postfix admission review

The five findings in `discovery/auditor-independent-review.md` are resolved in the final reviewed generation under the stated bounded mechanical contract. The additional ordinary-string, raw-string and character import operands are also refused. This focused source review found no remaining admission blocker in those corrections. It supports admission of this exact Layer A contribution with the limitations below; it does not declare federation CBZ-0 or mathematical closure.

The review read the affected checker branches, targeted tests, mutation definitions, final baseline log and the nine new mutation assertion traces. It reused the completed final G0 result and the author's completed source/log integrity check. No tests, mathematical checks, Lean builds, actual-repository G1, Git operations or canonical writes were performed. The only reviewer write is this report.

| Original severity | Finding | Disposition and current implementation |
|---|---|---|
| P1 | Reflective engineering builtins loader bypassed isolation | Resolved. `source_checks.py:233` includes the explicit builtins/namespace reflection names. The name, defined globals/class/frame attribute and dynamic-call checks at lines 255–268 run before the engineering-scope continuation at line 269. A hash-bound nonmathematical review therefore cannot waive these dependency findings. The original `getattr(__builtins__, '__dict__', __builtins__)['__import__']('Q.bad')` fixture is covered both directly and through the manifest auditor. |
| P1 | Retained source with an unfamiliar suffix omitted language/module and escaped checking | Resolved. `auditor.py:182` requires supported language for every retained source regardless of suffix, and for an external source with a declared module. Once language is present, lines 224–230 require the registered matching module and invoke its source checker. The `Q.txt` contract tests missing language, missing module and successful exact retention in sequence. The mapped external `fake.txt` case separately checks the related language obligation. |
| P1 | Known unsupported source suffix claimed Lean/Python and avoided a gap | Resolved. `auditor.py:184` derives the supported extension relation; lines 185–186 emit `LANGUAGE_GAP` whenever a recognized source suffix has no supported checker, regardless of the claimed language. Both `.sh` labeled Lean and `.js` labeled Python are targeted cases. |
| P1 | Lean import tail was silently discarded after an accepted module | Resolved within the bounded import parser. `source_checks.py:181` accepts the documented plain apostrophe/`!`/`?` suffixes and emits `LEAN_IMPORT_GAP` for unsupported/escaped operands or trailing syntax. The mapped `Q.Bad'` fixture now enters isolation checking. Literal sentinels emitted at lines 37–38, 60, 68 and 92 ensure ordinary/raw strings and characters remain visible as unsupported import operands instead of disappearing. |
| P2 | Status observation dictionary equality conflated Boolean and integer fields | Resolved. `auditor.py:305` compares strict parsed observations using sorted canonical JSON serialization, retaining the distinction between `true`/`1` and `false`/`0`. Exact primary schema and integer-exit checks remain. The targeted bound-observation fixture requires `STATUS_EVIDENCE` for the original Boolean/integer mismatch. |

The specifications describe these policies in `SPEC.md` and `README.md`, including the restricted import syntax, defined reflective entrypoints, retained/mapped external language obligations, unsupported source suffixes and type-sensitive A7 observation binding. There is no added general dynamic-loader analysis or semantic certificate claim.

The reused final receipt is `tools/layer_a/evidence/g0-admission-final.json`, SHA256 `8f03335ff833b706445e1a6b6d8232548ece81cb439626e87275c5a6ffc98fe7`. It records `PASS`, an exit-0 baseline of 70 tests (67 auditor tests and three harness tests), zero baseline failures/errors, 53/53 actual checker-source mutants killed, and unchanged bound inputs during the run from `2026-10-29T18:08:14.586906+00:00` through `2026-10-29T18:08:38.534945+00:00`. The corresponding final console log records `G0 PASS: 70 baseline tests; 53/53 checker-source mutants killed`.

The author's `tools/layer_a/evidence/admission-final-binding-check.json` records `PASS`, nine source bindings checked, all 53 mutant log bindings and assertion inspections, and preservation of the original 44 mutant names. Root supplied this completed integrity evidence and confirmed the nine bound inputs remain stable during packaging. This review cites that check without repeating its hashing.

The following new mutation traces were read directly. Each ran the 67 auditor tests, exited 1, reported assertion failures and reported zero errors. All targeted original and extension tests appear `ok` in the recorded final baseline.

| Added actual source mutation | Assertion failures | Affected assertion evidence |
|---|---:|---|
| `restore-reflective-builtin-loader-gap` | 3 | Original manifest false-green returns under mutation; direct builtins and `vars` reflection cases lose `PY_DYNAMIC_IMPORT`. |
| `disable-reflective-loader-attributes` | 2 | Globals and class/subclass reflection cases lose `PY_DYNAMIC_IMPORT`. |
| `truncate-primed-lean-imports` | 2 | Complete suffix extraction and the mapped primed-module isolation expectation fail. This mutation still produces an import gap; its kill establishes required extraction/isolation behavior rather than recreating a green result. |
| `silence-unsupported-lean-import-tail` | 4 | Bad syntax and all three literal operands lose `LEAN_IMPORT_GAP`. |
| `drop-literal-import-operands` | 3 | Ordinary-string, raw-string and character operands lose `LEAN_IMPORT_GAP`. |
| `skip-retained-text-language-requirement` | 1 | Unlabeled retained `Q.txt` loses `LANGUAGE_GAP`. |
| `skip-mapped-external-language-requirement` | 1 | Mapped external text module loses `LANGUAGE_GAP`. |
| `allow-unsupported-source-extension-language` | 2 | Lean-labeled shell and Python-labeled JavaScript lose `LANGUAGE_GAP`. |
| `conflate-status-json-booleans-and-integers` | 1 | Boolean/integer observation mismatch loses `STATUS_EVIDENCE`. |

These logs are under `tools/layer_a/evidence/g0-admission-final-mutation-logs/`. The actual mutation runner copies the checker and auditor tests into a temporary directory, changes one actual checker source, and distinguishes assertion kills from invalid syntax, import/runtime errors, timeouts and survival. The nine new names extend the prior 44; none were removed. No harness error is represented as one of these nine assertion kills.

The final receipt records these current source bindings, carried forward from the completed integrity check:

| Relative to `tools/layer_a/` | SHA256 |
|---|---|
| `auditor.py` | `c0a32e827279dfb9164d540c1f99c2269e3c49446a3fbeae23afefb53af54710` |
| `source_checks.py` | `9c73ecdd4fcc265165af564725738bc0a8ac4f79bba2c82b7d327fd90821fb94` |
| `canonical_lean.py` | `eb1088a53040f304b8a27b42a99d0d71b6fedf77668b4a7a5c30f0c0b8115bb0` |
| `mutation_test.py` | `6d50e45680fb031768397f77ee21f091686638cf33167e496370348a41423623` |
| `tests/test_auditor.py` | `61c22bb335490ba6b9e4b34fbdb0db0047d06e820575dc00500ebdf715847c9e` |
| `tests/test_mutation_runner.py` | `350a8f11d2d06d04da5f5270dbb81368aeac2dc6a66144237004f9cbe7eb2653` |
| `SPEC.md` | `9f9725a3d49af55370cb0f33141aa16b882021416cace7889e54be85df626e29` |
| `README.md` | `0cfe9128e9a79cff349f1063c6cdc515fdf11645e61375cc5da0d25129823888` |
| `LEXER_PROVENANCE.json` | `0f0f2516729435e9c3ef977e7aca9f8d51f6d179f77be3d194f7958945cdf4db` |

The public lexer remains the exact pinned PR29 source at head `ef5b924016c0f85ced7eb1dfb8f5eb9bd776f169`, native blob `0a113896aff4c999111193359c38906ff61e52e8`. Its recorded published verification is inherited contributor evidence. This review neither reran it nor established a competing lexer implementation.

Historical evidence remains generation-specific and reusable: `g0-final.json` records 56 tests/39 mutants, `g0-hardening.json` records 60/44, and `g0-admission-fix.json` records 69/52 before the additional literal-tail correction. Their respective bound source snapshots remain under `pre-hardening-source/`, `pre-admission-fix-source/` and `pre-import-literal-fix-source/`. The newly exposed obligations explain the later corrections; neither a handoff nor pending process review erases the completed evidence for unchanged scope.

The admission claim is finite behavioral coverage of the exact-manifest classification, byte/hash, supported source, isolation and evidence-binding rules. Recognized unsupported source extensions, defined reflective entrypoints and unsupported parsed import operands block closed source; isolated retained findings remain exact counted findings rather than clean certification. This does not prove completeness for arbitrary runtime loading, arbitrary source semantics, proposition satisfiability or generated/elaborated declarations. It does not authenticate supplied Git inventories, pins, execution receipts, external review identities or certificate truth. Actual-repository G1 is still `NOT_RUN` in this review. G2/G4/G7/G8 and other federation obligations remain separate.

This is a separate-agent source review within the same provider. It supplies no provider-distinct process credit and no federation closure declaration. Applicable mathematical verification remains reusable under the owner's workspace guidance; this review did not repeat or downgrade it.
