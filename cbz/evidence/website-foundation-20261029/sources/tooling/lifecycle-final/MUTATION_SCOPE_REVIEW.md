# Finite mutation classifier review and repair

The frozen b7 contribution's generic mutation-credit predicates admitted assertions from unittest setup, teardown and cleanup. Actual benign replacement fixtures reproduced this for both portable runners and the static auditor, normally and under optimized Python. These fixtures preserve test method names but replace their bodies; they are not reruns of the original assertions.

The recorded historical finite observations remain valid at their exact source/test hashes: all fourteen portable guard-kill logs trace to each expected test body, and all twelve original auditor mutants have direct `test_*` body assertion failures. No lifecycle frame appears. This independent log inspection does not turn the old generic classifier into a sound future grader. The old auditor selftest also omitted `-O` in its child argv: outer optimized execution must not be described as optimized mutant-child execution. Separate optimized unit-test observations retain their own scope.

The repaired synchronous classifier is `tools/cbz_mutation.py`. It instruments each actual named test method and its unittest dispatch/setup/teardown/cleanup entrypoints, matches direct body exception identity, checks exact expected identities and complete population, and refuses skipped/error/expected-failure/unexpected-success or failed lifecycle runs. A dispatcher assertion without calling the named method is rejected. A subtest failure is diagnostic and never the sole credit for a kill. Existing subtest assertions remain unchanged; direct body failures elsewhere can independently demonstrate a mutant was caught. The scanner now launches this helper in a separate process with the same optimization mode and a source-bound expected population. Both portable runners use it in process and hash it as an input.

Independent engineering review found an additional actual false-positive baseline: replacing an async test method with an ordinary wrapper stopped unittest from awaiting it. The method could contain a failing assertion while its deferred body never executed. Five real RED controls preserved this. The final helper explicitly refuses coroutine, generator and async-generator named bodies before invocation. Any non-None return from a synchronous body is refused; a returned coroutine is closed without executing it. The supported boundary is intentionally synchronous unittest bodies, not an async testing framework.

Observed final verification on the exact sources in `mutation-review/FINAL_SOURCE_IDENTITY.json`:

- 25 lifecycle controls pass normally and under `-O`; their actual child-mode controls include normal, `-O` and `-OO` helper subprocesses. No full auditor `-OO` mutation run is claimed.
- 26 reference-tool positives and all 7 original guard mutants pass in each mode.
- 18 baseline-tool positives and all 7 original guard mutants pass in each mode.
- All 10 strict source mutants against the new classifier are rejected in each mode, using the unmutated classifier to verify actual assertion-body failures across the complete 25-test population.
- The unchanged 43-test auditor population passes, and all 12 original scanner source mutants have actual direct body assertion kills in each child mode 0 and 1. Final scanner/test/helper hashes remain unchanged across each run.

Reproduction commands from the source checkout:

```sh
python -B -S tests/test_cbz_mutation_lifecycle.py
python -B -O -S tests/test_cbz_mutation_lifecycle.py
python -B -S tools/cbz_audit.py selftest --tests tests/test_cbz_audit.py
python -B -O -S tools/cbz_audit.py selftest --tests tests/test_cbz_audit.py
```

The full portable/strict-mutant evidence replay is `mutation-review/run_final_controls.py`, executed normally and with `-O`. It uses the documented `/workspace/universal-law/website-repair-20261028` source checkout and new output directories under the separate `mutation-review` evidence directory. The script and final output records are evidence recipes, not mathematical certificates. Prior RED and intermediate 18/20-test evidence is preserved separately and is not relabeled as the final 25-test run.

The finite helper does not prove arbitrary test semantics or mathematical soundness. A malicious replacement of the registered test source can still write a direct body assertion; such source/test hash changes require separate review and new evidence. Trusted, exact source identities and the original mathematical assertions remain essential. No mathematical statement, proof, bound or pre-existing test assertion was changed by this repair.
