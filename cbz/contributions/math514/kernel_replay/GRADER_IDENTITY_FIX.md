# Binding selected test bodies

The runner freezes the actual class, ordinary function and code objects for each
selected test before execution. It derives module coverage from the actual module
dictionary, rather than user-provided directory listings. It checks those identities
before fixtures, at test starts, at body entry and at body completion. A changed
identity remains non-green even if a later fixture restores it.

The supported population is standard unittest declarations with ordinary TestCase
construction and attribute dispatch, ordinary synchronous functions or bound methods,
and exactly None returns. Module `load_tests` hooks and callable suite factories are
refused before invocation. Custom constructors, allocation, metaclasses, attribute
dispatch and test-method descriptors are refused before instance construction.
Only exact builtin staticmethod/classmethod descriptors are admitted; subclasses
can override lookup and are refused. Special-member inspection uses static lookup.
Named selectors traverse ordinary module dictionaries and static class declarations;
proxy/descriptor factories are refused before lookup. Ordinary module, class and
individual-method selectors remain supported.

Loading trusted Python modules still executes their imports. This is a bounded
trusted-runner/source-custody contract, not a sandbox for arbitrary Python, a proof
of assertion semantics, or authentication of fabricated consistent JSON.

The actual RED controls preserve omitted second-body assertions: module hooks alias
methods or replace a class under the same name, constructors alias bodies before
snapshots, module __dir__ hides a class, and named/static-descriptor lookup replaces
a body before admission. Independent review discovered and retained the latter
boundaries before final admission. No test was removed. One existing layer-a test
now asserts the stricter complete two-declaration population and hook refusal,
rather than expecting an executed hook's selected subset.

Seven new test methods per tool cover these cases and preserve ordinary positives,
body assertions, exact named selection, existing lifecycle/subtest/error controls
and the synchronous None-return profile. The kernel has 27 affected harness methods
per mode; the auditor has 19. These are composed, scoped receipts with targeted
refreshes after an actual predicate or test change. They are not new full-suite,
Lean, project, scanner-mutation or production-replay campaigns. Applicable prior
mathematical and ordinary-body evidence remains reusable.

The new runner safeguards have 14 distinct themes per tool in each mode, across
four source generations: 8 initial themes, 2 construction/discovery themes, 1 named
selector theme and 3 exact-descriptor/static-inspection themes. Their 56 assertion-
kill observations are separate from production scanner/replay mutation cases;
reused earlier observations remain source-bound and their applicability is explicit.
Original REDs, AMEND reviews, intermediate passes and final controls are retained
under `evidence/grader-identity-fix/`. See `CURRENT_COMPOSED_ADMISSION.json`,
`OPTIMIZATION_PROVENANCE.json` and the review records there.

For a future genuinely affected scope, from the complete tool directory:

```sh
# Kernel tool
python3 -B -S run_tests.py --test test_mutation_harness.IdentityHarnessTests --output /absolute/new-evidence/identity.json
PYTHONOPTIMIZE=1 python3 -B -S -O run_tests.py --test test_mutation_harness.IdentityHarnessTests --output /absolute/new-evidence/identity-optimized.json
# Layer-a tool
python3 -B -S mutation_test.py --internal-test-runner --test-root tests --test test_mutation_runner.IdentityHarnessRules --output /absolute/new-evidence/identity.json
PYTHONOPTIMIZE=1 python3 -B -S -O mutation_test.py --internal-test-runner --test-root tests --test test_mutation_runner.IdentityHarnessRules --output /absolute/new-evidence/identity-optimized.json
```

Use only the applicable pair for its tool, and use fresh output paths. Do not rerun
completed checks merely because of a clone, handoff or review. Historical development
controllers are retained provenance with frozen-source inputs, not automatic commands
against a later source version. Canonical registration and final federation coverage
remain the integrator's responsibility. No G1 repository audit, G2/G3/G4/full CBZ
claim follows from this repair.
