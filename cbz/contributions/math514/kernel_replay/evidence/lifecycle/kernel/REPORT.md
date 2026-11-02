# Lifecycle-safe kernel mutation grading

The isolated patch replaces summary-only grading with a custody-bound structured unittest receipt. A mutation kill requires the exact pristine selected parent-method population, one start/finish and one authorized body entry/exit for every method, and at least one actual body AssertionError. Each subtest must reference that actual selected parent TestCase object. Body subtest assertions remain valid with explicit parent/phase/outcome events; lifecycle assertions, missing/duplicate identities, errors, skips, expected failures, synthetic/empty populations, missing metadata and inconsistent integer accounting fail closed.

The selected method itself is wrapped, preserving unittest decorators. The dispatcher authorizes body invocation, while lifecycle depth prevents calls from setup, teardown or cleanup from becoming body evidence. Direct body failures require the identical escaping AssertionError object; subtest assertions require an active selected body. Multiple failing subtests may belong to one selected parent method. An externally discovered pristine method list is bound before each mutant runs, with discovery cached per selector and no fixtures or bodies executed by --list.

Changed runtime files: run_tests.py and mutations.py. Changed harness: tests/test_mutation_harness.py. No new runtime helper module. Existing registry entries for these three files need their content hashes updated; keep kernel_replay.py, ExactReplay.lean and original production tests/bindings intact. Portable copies already include run_tests.py through the existing copytree. The patch adds an internal --list mode to run_tests.py, and classify(value, code, expected_selection=None, expected_ids=None) remains callable with the old two arguments while rejecting legacy summary-only payloads.

Fresh verification is limited to the changed harness: twelve normal methods, one separately added population-listing method, and thirteen methods with PYTHONOPTIMIZE=1 and -O (also inherited by fixture subprocesses). A subsequently identified foreign-subtest-parent control passes normally and under optimization, with the preceding thirteen controls reused because they all use matching actual parents. The current fourteen harness method IDs replace the old single method; together with the unchanged 31 wrapper and 9 real methods, the current production-suite population is 54. Benign fixture invocations inside harness methods are controls, not additional production-suite methods.

Eight targeted actual harness-source mutants were killed in the first source generation. The later dispatcher/lifecycle implementation change warranted four runner-source controls on the final runner (three affected controls refreshed and one new authorization control). The five unchanged classifier-source controls are reused. The foreign-parent followup adds one actual drop-parent-identity guard mutant, also killed. This is ten distinct safeguard mutations across thirteen executions, not thirteen distinct mutations or a fresh original 25-mutant run. Each kill has a structured guarded body-assertion receipt. kernel-delivery.json records exact method IDs, hashes, counts and applicability.

The old 19 fresh plus 6 earlier original source-mutant assertion kills remain historical evidence: production source, original wrapper/real test bodies, and mutation definitions are unchanged, and previously established failure traces are in the selected bodies. This patch does not retroactively fabricate lifecycle metadata for old receipts. It does not rerun Lean, proof replay, builds, the unchanged original tests, or original production mutants. Source custody and provider-distinct review remain separate requirements.

Fresh evidence commands (paths relative to the CBZ root):

- Normal twelve-method log: delivery/mutation-lifecycle-fix/evidence/kernel-controls-green.log.
- Separate listing control: delivery/mutation-lifecycle-fix/evidence/kernel-population-listing-green.log.
- Optimized thirteen-method command: PYTHONOPTIMIZE=1 python3 -B -S -O -m unittest discover -s delivery/mutation-lifecycle-fix/work/kernel_replay/tests -p test_mutation_harness.py -v.
- Initial source controls: delivery/mutation-lifecycle-fix/evidence/kernel-harness-mutants/summary.json.
- Changed final runner controls: delivery/mutation-lifecycle-fix/evidence/kernel-final-runner-mutants/summary.json.
- Foreign-parent followup: delivery/mutation-lifecycle-fix/evidence/kernel-foreign-subtest-{red,green,optimized-green}.log and kernel-parent-identity-mutant/summary.json.
- Final independent review: delivery/mutation-lifecycle-fix/evidence/independent-lifecycle-review.json, bound to the final kernel and layer source hashes.

The saved source-control scripts are provenance artifacts for those runs. Their default output directories already contain sealed receipts; use an isolated new output directory if adapting them for future execution. Do not overwrite completed evidence or replay unchanged checks solely because of handoff.

JSON receipts validate accounting from the custody-bound runner, not cryptographic authenticity of a completely fabricated consistent JSON document. Missing, malformed, contradictory or wrongly selected payloads are rejected; the paired runner/process/source bindings remain part of the trust boundary.
