# CBZ intent adversarial qualification implementation plan

> Agentic workers: execute inline using the existing Superpowers verification and
> test-first workflow. No owner approval checkpoint is required by current governance.

**Goal:** identify reproducible false-success or coverage gaps in the delivered
CBZ tooling, and supply portable regression tests to its writer.

**Architecture:** an isolated Main checkout at PR #712's exact head; standard
unittest cases exercising real Git fixture federations. Original suites and new
negative cases have separate retained evidence.

**Tech stack:** Python standard library, Git.

**Spec:** ../specs/2026-10-29-cbz-intent-adversarial-01a103c8.md

## Constraints and review focus

Layer A is immutable. No production crawler, canonical status/governance, workflow,
theorem or original test is modified. No global G7/CBZ PASS is inferred. Preserve
every failure. Required cases cover graph closure, missing declared inputs,
unresolvable revisions, newly opened repository scanning and remote-ref freshness.

## Task 1: establish the subject

- [x] Clone an isolated repository, verify AGENTS/GOVERNANCE and exact base.
- [x] Observe the Main compiled-inventory baseline: 43 cases, one skip.
- [x] Fetch and read PR #712's exact 13-file delivery.
- [x] Retain actual normal/optimized runs of its original suite and source hashes.

## Task 2: exercise the gaps

- [x] Add tests/test_cbz_drift_adversarial_01a103c8.py with the six independent
  observable cases in the spec, using the existing temporary Git fixture helper.
- [x] Run against unchanged 15e8f689 and retain failure versus error identities.
- [x] Confirm representative failures through the CLI where applicable.
- [x] Separate documented domain limitations from implementation findings.

## Task 3: deliver

- [ ] Publish exact reproductions and portable test identities to #712/#684.
- [ ] Route implementation repair to its current writer.
- [ ] Rerun the same cases on a frozen successor; never relabel old failures.
- [ ] Supply the B-003 integrator proposed G0/G5/G6/G7/G8/G11 applicability,
  leaving G2–G4 mathematical verification intact.
