# Q-9006 Live Freshness Companion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans after the parent authorizes implementation. Steps use checkbox syntax.

**Goal:** qualify one real read-only freshness companion while leaving ordinary discovery offline.
**Architecture:** one small standard-library entrypoint reuses the existing production validator, ContentObserver and refresh_record. Direct helper tests use complete synthetic external observation replies; live qualification requires the actual default HTTPS observer.
**Tech Stack:** CPython 3.11, unittest, existing node-status v0 modules, GitHub read-only HTTPS.
**Spec:** docs/plans/2026-11-09-node-status-live-check-design.md

## Global Constraints

- Tests/docs plus a read-only offline contract runner in the initial publication; no production or live workflow yet.
- Preserve the existing dormant RealRecordTests method and all history.
- Scope only Q-9006 freshness; no Q-9002--Q-9005 adoption or scientific effects.
- All four selected real records are mandatory; all declared-fresh candidates must be checked.
- Require GITHUB_TOKEN; one real default observer per mode; no fallback or retry.
- Fail on absent/invalid data, stale/unknown/error, incomplete counts or input drift.
- No supplied transcript or synthetic unit result earns live coverage.

## Review Focus

- Extra/missing records must not silently reduce the selected input/cohort.
- Source pin equality with HEAD is not the cited-object-content-v1 definition.
- Schema/graph byte drift and duplicate JSON keys must fail, not just record drift.
- Observer observations/captured heads must cover every requested source exactly.
- The later CLI must bind actual source, interpreter mode and hosted run/attempt.

### Task 1: Publish and observe the expected absence RED

Files: tests/test_node_status_live_check.py, the two design/plan documents and a read-only offline contract workflow.
No tools/node_status_live_check.py is included.

- [ ] Parent publishes these tests/docs and the offline-only normal/optimized contract workflow at an exact guarded branch head.
- [ ] Run normal: python -B -S -m unittest tests.test_node_status_live_check -v
- [ ] Run optimized: python -B -O -S -m unittest tests.test_node_status_live_check -v
- [ ] Require actual nonzero outcomes whose failure is "Q-9006 live companion implementation is absent"; retain full output and source identity. Syntax/import/environment errors are not that RED.
- [ ] Parent reviews that result before authorizing production implementation.

### Task 2: Implement only after that authorization

Create: tools/node_status_live_check.py.
Tests: tests/test_node_status_live_check.py.

Interfaces:
- LiveCheckError(ValueError).
- evaluate_fresh_records(root: Path, *, observer, identity: dict) -> dict.
- run_live(root: Path, *, identity: dict) -> dict.
- Later CLI: --root selected-root and --output new-external-report; default root is the source containing the tool. Derive real source/run/mode identity rather than accepting it from a report.

- [ ] Capture/validate all selected records, schema and graph with unchanged existing production validators.
- [ ] Reuse existing refresh_record for every declared-fresh candidate with one observer; reject invalid or incomplete evidence and retain the stale declaration separately.
- [ ] Preserve the staged controls for schema/graph drift, duplicate input keys and unexpected records. Add direct CLI actual-source identity controls before its later hosted admission; the helper's supplied unit identity is not a hosted-source attestation.
- [ ] Implement mandatory-token run_live using ContentObserver() exactly once; no injected observer/API CLI option.
- [ ] Run both exact unit commands; require all actual selected controls executed with no skips/errors/failures and source bytes preserved.
- [ ] Obtain a fresh source-exposed engineering review; no independent scientific-review credit.

### Task 3: Qualify the actual live companion separately

The later authenticated live CI change is separate from this initial offline contract runner.

- [ ] Parent combines a dedicated read-only live job with the next necessary prepared-regression workflow work; preserve source/run-attempt bindings.
- [ ] Run normal and optimized real entrypoint invocations with actual default HTTPS and a read-only GITHUB_TOKEN only in this job.
- [ ] Retain original input hashes, source/tool/commit identities, real mode/run/attempt, head/time observations, counters and failures. Do not invent successful observations if unavailable.
- [ ] Inspect authentic results before documenting an explicit suite split or changing the old dormant-test disposition. Keep Q-9002--Q-9005 unresolved and never relabel skipped discovery as 100%.

Parent ruling: use a small read-only offline runner to observe both initial
missing-companion AssertionErrors promptly and retain complete logs. This adds
no production or live HTTPS execution and does not adopt dormant coverage.
