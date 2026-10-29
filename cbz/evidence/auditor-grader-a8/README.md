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

The test-first file `tests/test_auditor_mutant_diagnostics.py` specifies a narrow collector that will parse only literal existing mutation registries, resolve one exact source test per named guard, record real subprocess exits and both streams, and refuse missing/ambiguous targets or changed mutation anchors. Its output is raw diagnostic evidence for independent guard review, never a mutation-kill, Layer A or CBZ certificate. The bounded implementation is present but unqualified: the candidate hosted run is queued; the original collector test-first run was subsequently observed failing. Run its controls in both Python modes and record actual results before relying on its output.

## Observed adverse results and source-byte repair

Hosted run 40956034512, job 123736264636, tested commit aacd2f904e850673e730fccd6232148859740596. Its preserved decoded log is in `red/aacd2f904e850673-job123736264636.log`: each Python mode ran six test methods with 15 assertion failures and zero errors. Both original graders accepted malformed or unrelated failure conditions. All three original loaders executed a genuinely timestamp/size-valid poisoned bytecode cache while the checked source bytes remained legitimate. This is historical RED evidence, not a result for the later source.

The three test loaders now compile one source-byte read directly, avoiding timestamp-cache selection for those tool loads. This change does not authenticate an already-loaded interpreter, test module, imported library, or child process. Fresh execution and source-mutation tests are required to qualify the repair; no such result is implied by this edit. The grader failures remain open.

A new test-first control constructs timestamp/size-valid poisoned bytecode for the grader child’s own test-module import. The previous source-byte loader fix does not cover that import. This new adverse control has not yet been executed; no outcome is asserted.

A second test-first control plants a timestamp-valid poisoned child test cache under an inherited `PYTHONPYCACHEPREFIX`. It exercises both graders. Execution and child isolation are pending; no RED or PASS is inferred.

## Raw witness collector candidate

`tools/cbz/auditor_mutant_diagnostics.py` reports literal registry contents, exact source hashes and real named-control and mutant subprocess outcomes, including both raw streams. It reports no mutation kill or Layer A result. The candidate verifies unmutated controls, source hashes and clean Git state at both endpoints; those observations do not prove an atomic snapshot or protect against malicious concurrent edits. Its registry parser supports only explicit literal declarations and extensions, not arbitrary dynamic Python mutation or hostile code. The interpreter, standard library and selected source are trusted.

~~~sh
python3.11 -I -B -S tests/test_auditor_mutant_diagnostics.py -v
python3.11 -I -B -O -S tests/test_auditor_mutant_diagnostics.py -v
python3.11 -I -B -S tools/cbz/auditor_mutant_diagnostics.py --root ROOT --scope both --output NEW_EXTERNAL.json
~~~

The collector has meaningful adverse controls and source mutations, but no candidate test has yet been observed. The original collector test-first run is observed RED; the candidate run remains queued. Do not treat the collector as qualified, adopt its observed failures as their own witnesses, or claim a passing gate. Independently review every guard-specific observation before modifying a grader. Owner none.

Test-first head `9d52d68086f98d20a8b7b000a1d00b9e8b0c878d`, hosted run 40956363103/job 123737390295, actually ran seven controls per Python mode and produced 15 assertion failures and zero errors in each, including the three `CBZ_BYTES_MATCH`/`CBZ_POISONED` loader exploits. Its decoded job log is retained in `red/9d52d68086f9-job123737390295.log`. This confirms the old graders' and loaders' failures at that exact head; it does not verify a successor.

Collector test-first head `b1d80c1c62134c23f948d22ee451f5de1910de9b`, hosted run 40956746044/job 123738691733, actually ran six test methods per mode with six assertion failures and zero errors in each because the specified collector did not exist. The decoded original job log is retained in `red/b1d80c1c6213-diagnostic-job123738691733.log`. This is a test-first observation only, not verification of the later implementation.
