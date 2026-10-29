# Lifecycle-safe mutation grading

The grader now requires an independently discovered exact selected parent-method population, one authorized body entry and exit per selected method, and exact direct or body-subtest assertion events. Setup, teardown, cleanup, class-fixture failures, missing/duplicate identities, unsupported or incomplete populations, skips, errors, expected failures, inconsistent/noninteger counts and foreign subtest parents remain non-green. Direct assertion callbacks require the exact exception escaping the selected method. Valid body subtests remain supported.

Runner receipts are custody-bound execution evidence. The schema checks accounting and is not cryptographic authentication of adversarially fabricated consistent JSON. Source/runner/test custody remains necessary. Legacy summary-only JSON is refused; historical receipts are not retroactively assigned new metadata.

Only `run_tests.py`, `mutations.py` and `tests/test_mutation_harness.py` changed. Production replay, the Lean driver, the31 wrapper and9 real-Lean methods, and all25 production mutation definitions are unchanged. Fourteen current harness methods replace the old single summary-only method; its source is preserved in `evidence/lifecycle/baseline/`. Current54-method evidence is a composition:14 fresh affected harness methods,31 unchanged wrapper methods and9 unchanged real-Lean methods. There was no new full54-method run or Lean/project proof replay.

Fresh normal evidence is12 methods plus a separate population-listing method plus a later foreign-parent method; optimized evidence is13 plus the foreign-parent method, with child optimization inherited. Ten distinct grader safeguard source mutations were killed across13 warranted executions:8 initial,4 after dispatch/lifecycle changes (3 refreshed/1 new), and1 foreign-parent case. Five unchanged classifier controls were reused. These are separate from the retained19 resource-wrapper and6 earlier driver/import-trust mutation cases. See `evidence/lifecycle/evidence/kernel-delivery.json`, `evidence/lifecycle/evidence/independent-lifecycle-review.json` and the source-control receipts for exact versions/counts.

Run only affected checks when changed source or a new obligation warrants them. From the complete tool directory, use a fresh external output path:

```bash
python3 -B -S run_tests.py --test test_mutation_harness.MutationHarnessTests --output /absolute/new-evidence/harness-normal.json
PYTHONOPTIMIZE=1 python3 -B -S -O run_tests.py --test test_mutation_harness.MutationHarnessTests --output /absolute/new-evidence/harness-optimized.json
```

`run_tests.py --list --test SELECTOR --output NEW.json` discovers canonical identities without running fixtures or bodies; import/loading still occurs. Existing full-suite and production mutation commands in README remain available for genuinely affected scope. Saved controllers under `evidence/lifecycle/` are version-bound execution provenance, not automatic commands against a future source revision; their original default outputs must not overwrite preserved receipts. Registered active runners and portable commands are in REGISTRY.json.

The independent same-provider engineering review is clear against final source hashes. It supplies no provider-distinct scientific-review credit and no federation G0-G11/CBZ claim. G2, complete inventory/build provenance and final canonical admission remain separate.
