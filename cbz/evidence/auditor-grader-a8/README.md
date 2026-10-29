# Auditor mutation-grader adverse controls

This isolated test-first contribution preserves the canonical #698 scanner and census source unchanged, along with their existing suites. It adds tests for abnormal exits, mixed/foreign failures, skipped or inconsistent summaries, absent expected observations, missing target names, and timestamp/size-valid poisoned Python caches for the three existing test loaders.

The old grader is expected to fail these controls. An expectation is not an observed RED: see the actual run logs after execution. The diagnostic workflow runs ordinary and optimized Python and retains both outputs. It does not certify Layer A, G0, G1, or CBZ-0. The underlying 66 lexical and 23 A3 source mutants and their original positive cases remain unchanged; no failing test is deleted or disabled.

Run locally from a clean checkout with CPython 3.11:

~~~sh
python3.11 -B -S -m unittest discover -s tests -p test_cbz_mutation_grader_adverse.py -v
python3.11 -B -O -S -m unittest discover -s tests -p test_cbz_mutation_grader_adverse.py -v
~~~

The full unmutated positive suites and each exact mutant/control must be separately qualified after the repair. Receipt observations are finite same-process controls and cannot isolate deliberately malicious code or a compromised runtime. Owner none.
