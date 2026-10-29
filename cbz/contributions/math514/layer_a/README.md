# CBZ exact-scope Layer A contribution

This tool contributes classification/hash/import isolation, a strict exact-arithmetic Python AST profile and A6/A7 evidence bindings. It reuses Root's unchanged canonical Main#698 identifier lexer through a bounded metadata adapter. It does **not** replace federation G2/G4 or declare mathematical closure. The authoritative mechanical contract and limitations are in [SPEC.md](SPEC.md).

## Commands

With Python 3.11 or newer, change into the complete tool directory at your chosen repository-relative destination, for example `cbz/contributions/math514/layer_a/` or `tools/cbz/exact_scope/`:

```sh
python -B -S mutation_test.py --output /absolute/controller-evidence/g0-admission.json
python -B -S auditor.py audit --manifest /absolute/final-scope.json --output /absolute/controller-evidence/g1-report.json
```

Use a fresh controller output path per run; preserved raw development receipts are immutable historical evidence. Raw receipt paths identify the local environment where they ran, not publication execution paths. Run the second command for an actual repository only after applicable current tool-admission evidence is established and the controller has prepared its final tracked-file scope. Reuse completed checks on unchanged inputs; a new clone or handoff alone does not require another G0 run. G0 runs every local selftest, then actual temporary checker-source mutants. G1 reads files only; it never runs Git, Lean, project Python, external commands or mathematical checks. Its report is deterministic for identical manifest bytes/files/tool. `audit(manifest)` is also available to isolated fixture tests; it accepts the parsed object and returns the report.

Audit exit codes: **0** bounded `STATIC_PASS`; **1** one or more blocking findings; **2** invalid/unreadable JSON or command-line usage; **3** requested report output could not be written. `STATIC_PASS` always carries outstanding G2/G3/G4/G5/G7/G8/G10 obligations. Every finding blocks the static result; `fatal_gaps` is the subset that specifically identifies missing/unsupported certification. Retained isolated findings appear separately and never mean those files are clean. CLI input errors produce JSON except argparse usage errors. G0 exit 0 means every declared mutant was killed; any surviving/invalid/crashing mutant or baseline error is nonzero. No weakening/skip option exists.

To run selftests alone (development, not a replacement for G0):

```sh
python -B -S -m unittest discover -s tests -v
```

## Manifest schema v1

JSON accepts integers only; duplicate keys, floats, NaN and Infinity are rejected. Top-level keys are **exactly**:

```json
{
  "schema_version": 1,
  "approved_axioms": ["propext", "Classical.choice", "Quot.sound"],
  "roots": [
    {
      "id": "math",
      "path": "/absolute/final-tree",
      "tree_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "tracked_files": ["Closed.lean"],
      "files": {
        "Closed.lean": {
          "sha256": "<64 lowercase hexadecimal content digest>",
          "kind": "project_lean",
          "module": "lean:Closed",
          "vacuity": {
            "t@1": {
              "status": "conditional",
              "premises": [],
              "reason": "No nonvacuity certificate claimed; elaborated statement review remains G2/G8."
            }
          }
        }
      }
    }
  ],
  "module_map": {"lean:Closed": "math:Closed.lean"}
}
```

This is a schema illustration: replace the absolute root, immutable tree id, exact inventory and actual SHA256 values. Roots are nonempty, normalized absolute directories with unique ASCII ids (`[A-Za-z0-9_-]+`) and nonoverlapping paths. A tree id is 40 or 64 lowercase hexadecimal characters; G10 authenticates its connection to the supplied inventory. Every root has a nonempty unique list of normalized relative NFC paths, no `..`, backslashes or symlink component. `files` keys equal `tracked_files` exactly. No exclusions or inferred quarantine folders exist. The controller supplies the full final tracked scope; untracked state and inventory authenticity remain G10.

Every file requires `kind` and `sha256`. Unknown fields and fields inapplicable to its kind fail. Additional fields by kind:

| kind | Fields and obligations |
|---|---|
| `project_lean` | `module`, `vacuity` required; `.lean` source |
| `math_python` | `module` required; `.py`/`.pyi` closed exact-program profile |
| `engineering_python` | `module`, `nonmath_review` required; `.py`/`.pyi`; floats only in explicit bound nonmathematical reviewed scope |
| `negative_fixture` | `language`, `module`, `expected_findings`, `fixture_purpose`; retains actual negative tests |
| `quarantined_source` | `language`, `module`, `expected_findings`, `quarantine`; exact terminal isolation |
| `external_pinned` | `pin` required; source requires supported `language`/`module` and Lean `vacuity`; optional binary `bindings` |
| `binary_proof` | `bindings` required, optional `pin`; bytes are not treated as text |
| `documentation`, `data` | optional `status_evidence: true` for JSON status surfaces; other known source/proof suffixes cannot be relabeled this way |

`language` is `lean` or `python`. Retained source always requires language and a registered module, including a `.txt` filename; a mapped external module likewise requires supported source analysis. Historical `.lean.txt`/`.py.txt` source is recognized. Known unsupported source extensions, including `.mjs` and `.mjs.txt`, fail with `LANGUAGE_GAP` even if labeled Lean/Python; they require an appropriate certifier, not arbitrary data labeling. Nonbinary files must decode strict UTF-8. Actual reads and explicit permission bits are checked, including under a privileged reader. Immutable `pin` is exactly `{ "revision": "<40 or 64 lowercase hex>", "origin": "<nonempty source>" }`. Dependencies are scanned, never automatically declared safe by pinning. Known binary object suffixes also require G2/G4 bindings when `external_pinned`.

A source module has exactly one identity, e.g. `lean:Closed.Base` or `python:certifier`, and its one `module_map` value is `rootid:relative/path`. Maps cannot point to documents or alias a differently declared module. Explicit imports resolve through this map. Closed code may not reach negative/quarantined source through any mapped import chain. Python `from fractions import Fraction [as alias]` is the sole built-in exact-math profile import; G7 must bind/authenticate that Python runtime and fractions implementation. Mapped local `fractions` modules take precedence over the stdlib exception; a mathematical source cannot depend on an engineering-only replacement. Python mapped package prefixes and `from package import submodule as alias` candidates also enter the isolation graph. Mathematical source cannot import engineering-only mapped candidates or package prefixes. Other unknown imports, relative/star imports, reflective loaders, macro expansion and unsupported mechanisms remain reported gaps. Plain Lean module identifiers including apostrophe, `!` and `?` suffixes are extracted completely. Unsupported or escaped operands and trailing syntax are explicit `LEAN_IMPORT_GAP` findings even after a valid prefix. Both Python scopes refuse explicit reflective namespace entrypoints (`__builtins__`, `vars`, `getattr`, `setattr`, `delattr`, globals/locals and defined globals/class/frame attributes); an engineering review does not waive that boundary. There is no general import/dataflow completeness claim.

## Retained source and certificates

`expected_findings` is an exact map of checker code to positive integer occurrence count, e.g. `{"LEAN_PLACEHOLDER": 1}`. `{}` is valid for a retained semantic failure that has no lexical violation; G5 still must retain and execute its actual failing test. Missing or extra scanner findings block. No retained source is counted as clean.

A quarantine object has exactly:

```json
{"id":"Q-1","boundary":"root:path and exact isolated mathematical boundary","blocking_reason":"specific unproved fact","owner":"none","maintenance":"zero"}
```

All evidence references have exactly `{"root":"id","path":"relative/file.json","sha256":"<actual file SHA256>"}`. The referenced file must itself be classified, readable, hash-bound JSON inside the manifest. Binding is mechanical; authentication and truth of its review/certificate are G8/controller work.

An engineering `nonmath_review` reference points to exactly:

```json
{"type":"nonmathematical_review","source_sha256":"<source hash>","scope":"nonmathematical_execution_only","reviewer":"identified reviewer","rationale":"explicit reviewed execution scope; no mathematical certification"}
```

This is an affirmative review claim, not a waiver for mathematical floats. The controller must substantiate it.

Lean `vacuity` keys cover exactly every lexical theorem/lemma/example as `name@line` (anonymous `example@line`). An entry has `status`, nonempty `reason`, and only the following applicable fields:

- `conditional`: explicit `premises` string list, possibly empty; no satisfiability claim.
- `vacuous`: reason naming the unsatisfiable source-shaped hypotheses; syntactic contradiction candidates require this classification.
- `nonvacuous`: `evidence` reference to `{ "type":"nonvacuity_certificate", "subject_sha256":"<source hash>", "declaration":"name@line", "witness_description":"actual witness and certificate description" }`.

Candidates are syntax observations; arithmetic interpretations, aliases and arbitrary proposition SAT are not decided. G8 must validate the semantic interpretation of annotations, and G2 must expose generated/aliased statements. Conditional status never closes a mathematical hypothesis obligation.

Binary `bindings` is exactly `{"G2":<reference>,"G4":<reference>}`. Each reference points to a passing JSON object with `type: "kernel_evidence"`, corresponding `gate`, `subject_sha256` equal to the object's content hash, `observed_status: "PASS"`, integer `exit_code: 0`, a nonempty command list and nonempty `run_id`. G2/G4 must actually supply/authenticate that result; byte/hash presence cannot create it.

## Status evidence schema

If `status_evidence` is present, its value must be exactly true; null, false, zero and all other values fail closed. A file with `status_evidence: true` contains exactly required fields `schema_version:1`, `gate:G0..G11`, `observed_status:PASS|FAIL|NOT_RUN|ERROR|INACCESSIBLE`, nonempty `run_id`, nonempty string-list `command`, `exit_code`, and nonempty `inputs` mapping `rootid:path` to actual content SHA256. The only optional field is `evidence`, a bound JSON reference. `PASS` requires integer exit 0; `FAIL` requires nonzero integer exit; other observed execution states require integer exit. The observation JSON must equal all status fields except the evidence pointer, using canonical JSON comparison that distinguishes booleans from integers. `NOT_RUN` requires null exit and no observation evidence. This checks structure and bindings, never an aspirational Markdown label or command execution authenticity.

## Published lexical reuse and evidence

`canonical_lean.py` is the unchanged public `tools/cbz/layer_a_scan.py` from Main#698, head `38312e16f7ae22aeb433823c4a00a2cc80e23d0b`, native Git blob `2eee3a2200082444093935291b9389c0795641cf`. `LEXER_PROVENANCE.json` records its exact source binding. The adapter consumes its seven-field identifier stream, character-literal recognizer and unchanged numeral spelling recognizer; path contexts, filesystem/Git walking and standalone CLI are unused. Every supplied file remains explicitly classified by the manifest. Unsupported token metadata or lexical errors cannot produce a clean result.

This replaces the older ULW29 engine after two concrete source-admission gaps: a valid `0axiom` command boundary hid an owned axiom, and nested quoted terms inside `throwError` interpolation hid an explicit `sorry`. The canonical engine supplies the numeric boundaries and conservative interpolation-hole scan; no second lexer engine is introduced. The bounded secondary metadata parser remains responsible for imports and source-shaped declaration/vacuity metadata. A conservative hole interpretation may also flag hazard-looking brace text inside an ordinary string. Arbitrary macro semantics remain G2/G4 work.

The lexer integration receipt remains `evidence/lexer698-admission.json`: its 45 affected methods and 19 scanner mutants per mode are predecessor-scope observations reused here. The predecessor 87-method composition, including four then-new lazy-body methods, is recorded separately at `evidence/lazy-body-profile/current-composed-admission.json`. Main#698's existing actual Lean counterexample sources and compiler logs are retained under `evidence/public-main698-38312e16/independent-review/` and reused without compiler replay. Its contributor 72-test/two-mode and 66 lexical plus 23 A3 mutant results are inherited source evidence, not a new run of this wrapper.

The historical Main#697 head `027505fee908693ded0a9e99eaa974ea6f419c6b` passed 70 tests and 53/53 source mutants, receipt `evidence/g0-admission-final.json` (SHA256 `8f03335ff833b706445e1a6b6d8232548ece81cb439626e87275c5a6ffc98fe7`). Those results remain valid in their recorded scope and are reused only where the relevant source/dependencies are unchanged. They do not cover the newly discovered cases or certify the changed lexer bytes. The old source/tests/provenance and receipts are preserved at `evidence/pre-canonical698-source/`; earlier b7eebb4 and hardening generations remain distinct.

G0 receipts bind the actual checker/test/spec/README source hashes and each changed checker hash/log. Invalid syntax, import/runtime errors and timeouts do not count as killed mutants. Identity-mutant and crashing-mutant tests verify the harness itself. Initial red runs and the original permission-guard survivor are retained as development evidence; the original guard-removal mutant remains in the final declared set alongside the actual skip-file mutant. The privilege-descriptor fixture kills the original by assertion. All final threshold decisions use exact integer counts, never floating point.

Owned by the root CBZ integrator after admission. This agent only writes this directory and its discovery build report. Root owns the final registry, tracked scope, source receipt verification, G0–G11 final run and any CBZ-0 declaration.

Hardening scope: `Environment.addDeclWithoutChecking` and qualified/escaped forms are blocking `LEAN_KERNEL_BYPASS` findings. Escaped `admitGoal`/`mkSorry` helper constants are refused alongside `sorryAx`. Unterminated character literals fail with `LEAN_LEXICAL_GAP`. Every exact `native` identifier is conservatively refused, so native configuration flags do not evade detection through whitespace, line breaks or nested comments; this policy can also reject harmless uses of that identifier. Comments and raw strings remain noncode; ordinary string bodies are also conservatively scanned for executable brace holes. No arbitrary alias expansion, semantic kernel trust or SAT completeness is claimed. The finite planted layouts are listed in `evidence/hardening-assessment.json`.

Canonical lexical diagnostics are deduplicated by their exact `(line, column, kind)` tuple in first-seen order and emitted after the secondary lexer findings. Distinct diagnostics remain distinct; the manifest auditor sorts its final findings deterministically.


A mutation kill requires a complete, externally bound selected unittest population and exact direct/body-subtest assertion provenance. Lifecycle failures, unsupported identity/population states, errors, skips, expected failures, and incomplete or inconsistent receipts are not kills. Historical body-trace evidence is version-bound and retained separately from fresh structured runner receipts.

The numeric-metadata predecessor composition was 94 methods (79 auditor + 15 harness), with 52 affected methods checked in both modes and 42 unaffected methods reused. Its 60 scanner definitions comprised 57 unchanged and 3 new guard mutants checked in both modes. This retained generation is `evidence/declaration-numeric/current-composed-admission.json`; see `NUMERIC_DECLARATION_FIX.md`. It was not a fresh whole-94/60 run and is not the current combined population.


## Synchronous None-return mutation-grader profile

Mutation grading supports ordinary synchronous Python unittest functions or bound methods returning exactly None. Generator, coroutine, async-generator and every wraps-linked lazy method layer are refused during pristine discovery before fixtures or bodies execute. Every non-None return is ERROR, including False, zero, empty strings and opaque iterables. Returned objects are never evaluated for truth, compared for equality, iterated or awaited. Only a fresh native coroutine in CORO_CREATED state may be closed without entering its body. Creating a suspended call object is not evidence that its deferred body executed.

The numeric-metadata predecessor composition was 94 methods (79 auditor + 15 harness), with 52 affected methods checked in both modes and 42 unaffected methods reused. Its 60 scanner definitions comprised 57 unchanged and 3 new guard mutants checked in both modes. This retained generation is `evidence/declaration-numeric/current-composed-admission.json`; see `NUMERIC_DECLARATION_FIX.md`. It was not a fresh whole-94/60 run and is not the current combined population.

The predecessor receipt remains unchanged at `evidence/composed-admission.json`: its 56 distinct affected methods (45 lexer, 8 lifecycle and 3 retained harness methods) plus 27 unaffected auditor methods cover its 83-method population under their recorded assumptions. Its 19/38 scanner-mutant composition, three one-method dynamic source controls and 10+3+2 generation-bound grader observations remain historical; repeated duplicate-population observations are not distinct production mutants. The original 70/53 and earlier failed generations also remain historical.

The predecessor lazy-body repair has four distinct new methods in each mode, eighteen fixture receipts per mode with zero lazy-execution sentinels, and three final None-profile safeguard mutations killed by assertions in both modes (six mode executions). The superseded lazy-type generation contains five safeguard themes in two modes (ten mode executions); three unchanged shape, wraps-chain and native-coroutine-disposal themes remain applicable and are reused. These sixteen generation-bound observations are not sixteen distinct production mutants. Packaging ran none of these checks again.

Predecessor lazy-body composition: `evidence/lazy-body-profile/current-composed-admission.json`. Its evidence is under `evidence/lazy-body-profile/`: `REPORT.md`, `frozen-manifest.json`, `new-guard-source-controls.json`, normal/optimized logs and fixture receipts. Scoped independent review: `evidence/lazy-body-profile/shared/independent-lazy-body-review.json` (SHA256 8e6c0de322cb2296d2ac9bf472bcb29c3fcdfaa941541539e1fc467d5a47c933); its layer_a section binds the final two runtime hashes. Static original-method shape applicability is preserved alongside it. This review and mathematical evidence do not satisfy an unrelated provider-distinct process requirement.

## Current combined generation

The combined population is 110 methods: 88 auditor methods (79 numeric-predecessor + 9 review/notebook methods) and 22 harness methods (15 predecessor + 7 new identity methods). The [review-boundary repair](REVIEW_BOUNDARY_FIX.md) cumulatively covers 19 distinct affected methods per mode: 18 applicable prior results plus the one newly checked notebook method. Its 11 affected scanner mutation definitions per mode comprise 10 applicable prior kills plus one newly checked notebook guard; 3 of those 11 are adapted legacy definitions. The 68 active scanner definitions comprise 57 unchanged, 3 adapted and 8 new definitions. The [grader identity repair](GRADER_IDENTITY_FIX.md) covers 19 distinct affected harness methods per mode by composed staged receipts and targeted refreshes (12 predecessor + 7 new); 3 unaffected predecessor harness methods retain their evidence. Identity safeguards are separate grader controls, not scanner mutations.

The completed 52-method numeric-metadata verification remains reusable: Lean/numeric functions are unchanged, and its standard declaration population uses the supported default, no-hook loading profile. Original 18-method review results, the later one-method notebook controls, and identity refreshes remain separately bound. This is not a fresh whole-110-method, whole-68-scanner-mutant, CI, Lean or mathematical run. Current final source bindings and Root's source-partition record are in `evidence/final-review-identity/current-composed-admission.json` and `SOURCE_COMPOSITION.json` beside it. Source-bound original-review, notebook and identity reviews are CLEAR for their declared scopes; they do not authenticate the final repository inventory or scientific statements. Actual repository G1 remains NOT_RUN. Kernel/build provenance, approved axioms, pins and federation admission remain independent. Prior exact-source metadata is retained under `evidence/pre-review-identity-metadata/`.

The repaired Python/manifest profile refuses augmented division and requires literal nonnegative exponents for augmented power; mathematical imports cannot reach engineering-only mapped package prefixes or submodule candidates. Presence of `status_evidence` requires its value to be exactly true. Unsupported `.mjs`, `.mjs.txt`, `.ipynb` and `.ipynb.txt` source cannot be relabeled as data or documentation. Notebook recognition blocks admission; no notebook parser or executor is introduced. See [the source-bound review and notebook controls](REVIEW_BOUNDARY_FIX.md).

## Selected-declaration identity profile

The [grader identity repair](GRADER_IDENTITY_FIX.md) freezes actual class, ordinary function and code objects before execution, including wraps-linked functions, and checks them before fixtures, at test starts, at body entry and at body completion. Module coverage comes from the actual module dictionary. Module `load_tests` hooks, callable suite factories, custom constructors/allocation/metaclasses/attribute dispatch and test-method descriptors are refused before construction or invocation. Only exact builtin staticmethod/classmethod descriptors are supported; named selectors traverse ordinary module dictionaries and static class declarations, refusing proxy/descriptor factories before lookup. Ordinary module, class and method selectors remain supported, with the existing synchronous None-return and exact assertion-provenance contracts.

Trusted Python imports still execute. This bounded trusted-runner/source-custody profile is not a sandbox against arbitrary Python or authentication of fabricated consistent JSON. The final independent identity review is CLEAR for its exact source scope. Its 14 safeguard themes per tool in each mode yield 56 generation-bound assertion-kill observations across both tools: 32 + 8 + 4 + 12 across four source generations, not a fresh final-source 56-case campaign or production scanner/replay mutations. Earlier applicable observations and targeted refreshes remain separately bound. See `evidence/grader-identity-fix/CURRENT_COMPOSED_ADMISSION.json`, its shared optimization/guard provenance and `shared/independent-review-final.json` (SHA256 f47d13cc789829966ebf60e7b5e5a9e45059e5ed5db4a5089a3b2ca7534993b4).
