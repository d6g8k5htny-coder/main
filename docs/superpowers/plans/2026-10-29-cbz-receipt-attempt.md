# CBZ receipt attempt implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development. The owner explicitly authorizes autonomous implementation, verification and integration without further questions.

**Goal:** Make live formal receipts bind to their actual workflow attempt while preserving exact historical evidence.
**Architecture:** Extend the existing producer/binder contract; isolate the fixed historical consumer from live acceptance. No general audit framework.
**Tech Stack:** Existing Python standard library and Lean gate.
**Spec:** ../specs/2026-10-29-cbz-receipt-attempt-design.md

## Global Constraints

No changed proof statements, weakened bounds, deleted failing tests, fabricated hosted metadata, rewritten historical receipts, inferred green gates, or claim of complete Layer A certification.

## Review Focus

- Old receipts must fail live binding even when run ID, commit and logs match.
- Non-default workflow attempt must survive production and binding.
- Historical validation must not supply a reusable bypass to live CI.
- Old artifact bytes and published historical data must remain unchanged.
- The manifest must bind actual changed source, not bypass its digest check.

### Task 1: Repair existing receipt producer and binder

**Files:** Modify tools/formal_gate_check.py, tools/required_formal_check.py, tests/test_formal_gate.py, tests/test_required_formal_check.py, formal/manifest.json; historical Atlas producer/tests and one existing formal contract doc only as needed.

**Interfaces:** Current bind_receipt(receipt, manifest, env, git_head) remains strict and current-run only. Fresh receipt contains workflow_run_attempt from actual environment. Historical validation, if separated, accepts only the already fixed original identity and returns unchanged original binding.

- [x] Add adverse tests rejecting same-run stale and missing receipt attempts; observe failure before code.
- [x] Add producer test verifying a non-default attempt and local no-hosted-identity behavior.
- [x] Implement producer field and strict consumer match; do not default a missing attempt to 1.
- [x] Preserve existing fixed historical consumer through an explicit immutable historical route; test arbitrary/stale legacy input refusal and original bytes unchanged.
- [x] Rebind only required mechanical source hashes and document the contract.
- [x] Run source-only, relevant normal/-O suites, historical consumer tests and one full Python discovery; record failures/skips separately.
- [x] Remove the added attempt check in an isolated temporary copy and require the adverse regression to fail; retain mutation evidence.
- [x] Commit only task files; controller owns design/plan, mission governance and ledger.

### Task 2: Review and integrate

- [x] Fresh reviewer examines producer/binder/history boundary and evidence.
- [x] Resolve findings without weakening tests.
- [x] Reconcile live head, changed paths and current PR activity.
- [ ] Publish the tested source through the existing required checks; observe actual result.
- [ ] Record exact source and gate outcomes in the CBZ ledger. This task alone does not establish CBZ-0.

