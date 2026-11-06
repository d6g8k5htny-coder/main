# CBZ exact-scope Layer A contribution

This tool contributes classification/hash/import isolation, a strict exact-arithmetic Python AST profile and A6/A7 evidence bindings. It reuses the published PR29 identifier lexer. It does **not** replace federation G2/G4 or declare mathematical closure. The authoritative mechanical contract and limitations are in [SPEC.md](SPEC.md).

## Commands

With Python 3.11 or newer, change into the complete tool directory at your chosen repository-relative destination, for example `cbz/contributions/math514/layer_a/` or `tools/cbz/exact_scope/`:

```sh
python -B -S mutation_test.py --output /absolute/controller-evidence/g0-admission.json
python -B -S auditor.py audit --manifest /absolute/final-scope.json --output /absolute/controller-evidence/g1-report.json
```

Use a fresh controller output path per run; preserved raw development receipts are immutable historical evidence. Raw receipt paths identify the local environment where they ran, not publication execution paths. Run the second command for an actual repository only after the first succeeds and the controller has prepared its final tracked-file scope. G0 runs every local selftest, then actual temporary checker-source mutants. G1 reads files only; it never runs Git, Lean, project Python, external commands or mathematical checks. Its report is deterministic for identical manifest bytes/files/tool. `audit(manifest)` is also available to isolated fixture tests; it accepts the parsed object and returns the report.

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

`language` is `lean` or `python`. Retained source always requires language and a registered module, including a `.txt` filename; a mapped external module likewise requires supported source analysis. Historical `.lean.txt`/`.py.txt` source is recognized. Known unsupported source extensions fail with `LANGUAGE_GAP` even if labeled Lean/Python; they require an appropriate certifier, not arbitrary data labeling. Nonbinary files must decode strict UTF-8. Actual reads and explicit permission bits are checked, including under a privileged reader. Immutable `pin` is exactly `{ "revision": "<40 or 64 lowercase hex>", "origin": "<nonempty source>" }`. Dependencies are scanned, never automatically declared safe by pinning. Known binary object suffixes also require G2/G4 bindings when `external_pinned`.

A source module has exactly one identity, e.g. `lean:Closed.Base` or `python:certifier`, and its one `module_map` value is `rootid:relative/path`. Maps cannot point to documents or alias a differently declared module. Explicit imports resolve through this map. Closed code may not reach negative/quarantined source through any mapped import chain. Python `from fractions import Fraction [as alias]` is the sole built-in exact-math profile import; G7 must bind/authenticate that Python runtime and fractions implementation. Mapped local `fractions` modules take precedence over the stdlib exception; a mathematical source cannot depend on an engineering-only replacement. Python mapped package prefixes and `from package import submodule as alias` candidates also enter the isolation graph. Other unknown imports, relative/star imports, reflective loaders, macro expansion and unsupported mechanisms remain reported gaps. Plain Lean module identifiers including apostrophe, `!` and `?` suffixes are extracted completely. Unsupported or escaped operands and trailing syntax are explicit `LEAN_IMPORT_GAP` findings even after a valid prefix. Both Python scopes refuse explicit reflective namespace entrypoints (`__builtins__`, `vars`, `getattr`, `setattr`, `delattr`, globals/locals and defined globals/class/frame attributes); an engineering review does not waive that boundary. There is no general import/dataflow completeness claim.

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

A file with `status_evidence: true` contains exactly required fields `schema_version:1`, `gate:G0..G11`, `observed_status:PASS|FAIL|NOT_RUN|ERROR|INACCESSIBLE`, nonempty `run_id`, nonempty string-list `command`, `exit_code`, and nonempty `inputs` mapping `rootid:path` to actual content SHA256. The only optional field is `evidence`, a bound JSON reference. `PASS` requires integer exit 0; `FAIL` requires nonzero integer exit; other observed execution states require integer exit. The observation JSON must equal all status fields except the evidence pointer, using canonical JSON comparison that distinguishes booleans from integers. `NOT_RUN` requires null exit and no observation evidence. This checks structure and bindings, never an aspirational Markdown label or command execution authenticity.

## Published lexical reuse and evidence

`canonical_lean.py` is the unchanged public `tools/cbz/layer_a_scan.py` from Universal-Law-Workspace PR29, head `ef5b924016c0f85ced7eb1dfb8f5eb9bd776f169`, native Git blob `0a113896aff4c999111193359c38906ff61e52e8`. `LEXER_PROVENANCE.json` records exact source binding and inherited verification claims. We call only its `lex_tokens`; its heuristic path contexts, default exclusion behavior, filesystem/Git walking and standalone CLI are **not used**. Our manifest classifies every supplied file explicitly. Its published 28-test/39-mutant claim is recorded as inherited contributor evidence and was not rerun here. Our added integration tests cover the reported unchecked insertion, escaped helpers, malformed character literals and multiline/comment-separated native configurations. The previous b7eebb4 source and its G0-bound integration are preserved under `evidence/pre-hardening-source/`. Its earlier published tests are reusable evidence, not fresh federation gate evidence. Our integration tests and checker-source mutations exercise the adopted lexer at this boundary.

G0 receipts bind the actual checker/test/spec/README source hashes and each changed checker hash/log. Invalid syntax, import/runtime errors and timeouts do not count as killed mutants. Identity-mutant and crashing-mutant tests verify the harness itself. Initial red runs and the original permission-guard survivor are retained as development evidence; the original guard-removal mutant remains in the final declared set alongside the actual skip-file mutant. The privilege-descriptor fixture kills the original by assertion. All final threshold decisions use exact integer counts, never floating point.

Owned by the root CBZ integrator after admission. This agent only writes this directory and its discovery build report. Root owns the final registry, tracked scope, source receipt verification, G0–G11 final run and any CBZ-0 declaration.

Hardening scope: `Environment.addDeclWithoutChecking` and qualified/escaped forms are blocking `LEAN_KERNEL_BYPASS` findings. Escaped `admitGoal`/`mkSorry` helper constants are refused alongside `sorryAx`. Unterminated character literals fail with `LEAN_LEXICAL_GAP`. Every exact `native` identifier is conservatively refused, so native configuration flags do not evade detection through whitespace, line breaks or nested comments; this policy can also reject harmless uses of that identifier. Comments and ordinary/raw strings remain noncode. No arbitrary alias expansion, semantic kernel trust or SAT completeness is claimed. The finite planted layouts are listed in `evidence/hardening-assessment.json`.
