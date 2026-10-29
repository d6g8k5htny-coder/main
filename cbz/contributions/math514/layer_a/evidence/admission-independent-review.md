# Layer A focused admission review

The reviewed generation requires corrections before admission under its stated mechanical contract. Five new, narrowly scoped fixtures produced `STATIC_PASS` with no blocking findings: four source/import coverage gaps and one type-insensitive status binding. These findings concern the exact-scope auditor, not mathematical correctness.

Reviewed source generation:

- `auditor.py`: SHA256 `5d95d3c500e767305727d3f8a1c1e8944be4d67100e7703e6326de9db0264440`.
- `source_checks.py`: SHA256 `0f03cc4c182c8a10c839be30d59bec70f545e10fd7660642b672ed782c5cd786`.
- Existing G0 receipt: `tools/layer_a/evidence/g0-hardening.json`, SHA256 `9bd3dee4b424b17b5580e7cc26af5d161418fadb93477825f22f45be7d1c3619`.

The completed 60-test/44-mutant result, nine source bindings, and checked logs were reused as supplied verification evidence. They were not rerun. The earlier 56-test/39-mutant generation and its evidence remain applicable to their unchanged scope; these new cases expose uncovered obligations rather than invalidate that historical evidence. No repository mathematical verification, actual-repository G1, Lean build, Git operation, canonical write, or source edit was performed by this reviewer.

## Findings

| Severity | Location in reviewed generation | Finding |
|---|---|---|
| P1 | `source_checks.py:224`, `source_checks.py:246`, `source_checks.py:258` | Engineering source can use a direct reflective builtins importer without a dynamic-loading finding or import edge. |
| P1 | `auditor.py:172`, `auditor.py:182`, `auditor.py:222` | A retained source with an unrecognized suffix can omit language and module, so its source is never checked. |
| P1 | `auditor.py:175`, `auditor.py:182`, `auditor.py:184` | A known unsupported source extension can claim Lean/Python and avoid the required language gap. |
| P1 | `source_checks.py:169`, `source_checks.py:173`, `source_checks.py:177` | Lean import extraction silently discards a trailing unsupported identifier after one accepted module, bypassing mapped isolation. |
| P2 | `auditor.py:301`, `auditor.py:303` | Observation comparison treats Boolean fields as equal to integer schema/exit fields. |

### P1: reflective engineering loader bypasses isolation

The engineering checker accepts:

```python
getattr(__builtins__, '__dict__', __builtins__)['__import__']('Q.bad')
```

`check_python(..., mathematical=False)` returns empty findings and imports. The expression explicitly obtains the builtins importer, handling both module and dictionary forms of `__builtins__`. The current dynamic-name set omits `__builtins__`, and the AST checks for arbitrary calls/reflective attributes are skipped for engineering source.

The full fixture had a clean classified `Q/__init__.py`, a quarantined `Q/bad.py` containing `x=0.5` with exact retained finding map `{"PY_INEXACT":1}`, the engineering runner above, and a valid hash-bound nonmathematical review. All four files were read. The auditor returned `STATIC_PASS`, `findings: []`, and the retained `PY_INEXACT` finding for `Q/bad.py`; no edge connected the runner to that quarantined source.

This contradicts the documented refusal of reflective loaders and the closed-source isolation boundary. A nonmathematical execution review does not waive that boundary. Refuse the explicit reflective entrypoints conservatively or report a dependency-certification gap; no arbitrary dataflow analysis is required to reject this case.

### P1: retained source can escape source scanning

The full fixture contained one `Q.txt` file with these classification fields:

```json
{
  "kind": "quarantined_source",
  "sha256": "<actual content hash>",
  "expected_findings": {},
  "quarantine": {
    "id": "Q-1",
    "boundary": "exact isolated fixture file",
    "blocking_reason": "unproved fixture",
    "owner": "none",
    "maintenance": "zero"
  }
}
```

Its content was `theorem t : False := by sorry\n`. Language and module were absent. The auditor returned `STATIC_PASS`, empty blocking/retained findings, and classified/scanned counts of one. Here `scanned_count` records a byte read; `check_lean` was never called, so the declared exact retained finding map was never compared with source findings.

The README requires language and module for every retained source. Require a supported language and registered module for these source classifications regardless of filename suffix. The same static branch also permits a mapped `external_pinned` module without a language when its suffix is not recognized; mapped module identities should require actual supported source checking. That related external variant was identified statically, not executed as an additional fixture.

### P1: unsupported source extension can claim a supported language

The one-file fixture classified `bad.sh`, content `echo hello\n`, as `external_pinned` with `language: "lean"`, `module: "lean:Bad"`, `vacuity: {}`, a valid immutable pin, and the exact module-map entry. The result was `STATIC_PASS` with empty findings and one file read.

The suffix is recognized as source, but only known Lean/Python suffixes establish an `expected_language`. Consequently, the supported label suppresses `LANGUAGE_GAP` for known unsupported extensions. Enforce the supported extension/language relation before invoking a source checker, and report a gap for `.sh`, `.js`, and other explicitly known unsupported source extensions. This is a classification-support issue, not a request to inspect arbitrary file content or implement another language parser.

### P1: trailing import token is silently omitted

The fixture's closed file contained:

```lean
import Good Q.Bad'
```

`Good.lean` was clean project source. `Q/Bad'.lean` was mapped to `lean:Q.Bad'`, quarantined, and retained `LEAN_PLACEHOLDER: 1` exactly. `check_lean` returned `imports: ["Good"]`, empty findings, and no declarations. The full auditor returned `STATIC_PASS`, empty blocking findings, and the quarantined source's retained placeholder finding.

The import-name regular expression excludes the trailing apostrophe even though Lean identifiers can contain it. More generally, once any module is accepted, the loop terminates at an unrecognized token without reporting a gap. The unchecked tail must either be parsed completely or refused. The documented fail-closed rule requires this even for a syntax shape outside the bounded parser's support; it must not silently become an absent dependency.

### P2: observation equality is not type-sensitive

A valid status file used `schema_version: 1`, `exit_code: 0`, `observed_status: "PASS"`, valid gate/run/command/input fields, and an exact hash binding to a classified observation JSON file. That observation was identical except for `schema_version: true` and `exit_code: false`. The full auditor returned `STATIC_PASS` with empty findings.

The primary status file is type-checked, but the bound observation is compared through Python dictionary equality. In Python, `True == 1` and `False == 0`; the observation therefore passes despite differing from the required integer schema/content. Validate the observation's field types or use a type-sensitive structural/canonical-JSON comparison. This finding concerns exact schema binding only, not execution authenticity.

## Review limits and handoff

The review read SPEC, README, the current build report, actual checker sources, behavioral tests, mutation harness tests, public-lexer provenance, and the G0 receipt. The mutation runner's current kill branch requires a nonzero return, a test count, assertion failures, and zero reported errors; syntax failure, runtime/import errors, timeout, and identity survival are handled separately. No new harness error counted as a kill was found in this static inspection. This does not authenticate arbitrary future log output.

The existing discovery build report still described the earlier 56/39 lexer generation when read. Its update was already assigned to the author as pending final packaging and was not treated as an independent source defect.

The author was notified promptly of each confirmed fixture and acknowledged all five. The author will preserve the 60/44 generation, add focused failing contracts and corresponding source mutations, and correct only the affected rules. This report records the reviewed generation and original observations; subsequent corrected-source evidence must identify the changed generation and the affected checks. A new handoff alone does not require repeating unchanged verification.

This is a separate-agent source review within the same provider, not provider-distinct external review. Any such process requirement remains separate from the mathematical and mechanical evidence already completed. The contribution remains an exact-manifest/isolation complement. Arbitrary SAT/dataflow, inventory authenticity, semantic certificate truth, G2/G4/G7/G8, and federation closure remain separate obligations.
