# Auditor mutation-grader adverse controls

This isolated test-first contribution preserves the canonical #698 scanner and census source unchanged, along with their existing suites. It adds tests for abnormal exits, mixed/foreign failures, skipped or inconsistent summaries, absent expected observations, missing target names, and timestamp/size-valid poisoned Python caches for the three existing test loaders.

The old grader is expected to fail these controls. An expectation is not an observed RED: see the actual run logs after execution. The diagnostic workflow runs ordinary and optimized Python and retains both outputs. It does not certify Layer A, G0, G1, or CBZ-0. The underlying 66 lexical and 23 A3 source mutants and their original positive cases remain unchanged; no failing test is deleted or disabled.

Run locally from a clean checkout with CPython 3.11:

~~~sh
python3.11 -B -S -m unittest discover -s tests -p test_cbz_mutation_grader_adverse.py -v
python3.11 -B -O -S -m unittest discover -s tests -p test_cbz_mutation_grader_adverse.py -v
~~~

The full unmutated positive suites and each exact mutant/control must be separately qualified after the repair. Receipt observations are finite same-process controls and cannot isolate deliberately malicious code or a compromised runtime. Owner none.

## Witness diagnostic collector

The test-first file `tests/test_auditor_mutant_diagnostics.py` specifies a narrow collector that will parse only literal existing mutation registries, resolve one exact source test per named guard, record real subprocess exits and both streams, and refuse missing/ambiguous targets or changed mutation anchors. Its output is raw diagnostic evidence for independent guard review, never a mutation-kill, Layer A or CBZ certificate. The implementation is not yet present and the test-first execution is pending. After implementation, run its tests in both Python modes and record the actual result here and in GOVERNANCE.md before use.

## Observed adverse results and source-byte repair

Hosted run 40956034512, job 123736264636, tested commit aacd2f904e850673e730fccd6232148859740596. Its preserved decoded log is in `red/aacd2f904e850673-job123736264636.log`: each Python mode ran six test methods with 15 assertion failures and zero errors. Both original graders accepted malformed or unrelated failure conditions. All three original loaders executed a genuinely timestamp/size-valid poisoned bytecode cache while the checked source bytes remained legitimate. This is historical RED evidence, not a result for the later source.

The three test loaders now compile one source-byte read directly, avoiding timestamp-cache selection for those tool loads. This change does not authenticate an already-loaded interpreter, test module, imported library, or child process. Fresh execution and source-mutation tests are required to qualify the repair; no such result is implied by this edit. The grader failures remain open.
