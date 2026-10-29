# CBZ receipt attempt binding

Owner authority: the Crystalline Baseline Protocol supplied on 29 October 2026; canonical task [main#684](https://github.com/d6g8k5htny-coder/main/issues/684).

The existing formal producer omits workflow_run_attempt. The current-run binder consequently accepts a receipt from a different attempt of the same run and emits the consumer's attempt. This violates source-bound verification truth. Repair this at the producer and binder; preserve mathematical statements, existing tests, source hashes, and historical receipt bytes.

## Decision

Add the producer's actual GITHUB_RUN_ATTEMPT to fresh receipts and require an exact match in the live required-check binding. Missing, malformed or stale attempts must fail live binding. Local runs may retain null hosted identity and must never acquire hosted authority.

The Atlas source bridge intentionally consumes a fixed historical receipt without that field. Preserve this historical observation through a separate, explicitly historical validation route, confined to the existing exact receipt/artifact/commit/run identity. It must not be callable through the live receipt CLI or accept arbitrary legacy receipts. Never insert a fabricated attempt into original receipt bytes.

Alternatives considered: accepting absent fields preserves the defect; rewriting historical receipts destroys evidence; deleting the historical consumer loses a valid observation. A strict live boundary plus fixed historical validation is the smallest correction.

## Verification

TDD at the real binder boundary, producer serialization, and historical consumer. Require valid non-default attempt, stale same-run attempt, absent field, wrong types/zero/whitespace, and exact historical-byte preservation. Mutation-test the added safety check by removing it and observing the adverse test fail. Existing source-only gate, normal and optimized formal/control suites, historical Atlas tests, and full Python discovery must retain all assertions. A real Lean run is separate evidence; no mock or old CI can substitute.

## Scope

Main tools/formal_gate_check.py; tools/required_formal_check.py; tests/test_formal_gate.py; tests/test_required_formal_check.py; existing historical Atlas producer/tests only if needed; formal/manifest.json mechanical code hash; docs/FORMAL_REQUIRED_CHECKS.md or formal/README.md for actual contract. No theorem changes, workflow expansion, scientific promotion or another registry.

PR347 is a distinct evidence-retention proposal and does not edit these producer/binder files. PR358 overlaps the old formal manifest/test set; preserve its exact head and do not rewrite its branch. Current-main repair is against 4ef1867175dbe1072a54a02e95f80c4e2c70a4ff, with a fresh integration check before publishing.

