# Reliable Source Workflows Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the inline freshness repair; the independent UI task may be delegated with its own TDD gate. Record decisions and original execution evidence in the task ledger.

**Goal:** Make operational source freshness and excerpt-to-source reuse accurate.
**Architecture:** Transient typed Git object observations feed the existing
refresher; narrow renderer callbacks feed the existing reading-list controller.
**Tech stack:** Python standard library, existing browser JavaScript, installed
optional Playwright/Chromium for bounded engineering verification.
**Spec:** docs/superpowers/specs/2026-10-29-source-workflows-design.md

## Global constraints

No record/schema/math/workflow/protection changes; pins never advance implicitly.
Unknown evidence stays unknown; excerpt hashes never become whole-file hashes.
Reuse current storage/citation and source-admission logic. Preserve original
failed runs and unique output directories. No new dependency or scientific review.

## Review focus

- Tree or blob changes and different current types must never falsely be fresh.
- Missing, truncated, duplicate-path, symlink/submodule or malformed responses must fail closed.
- Snapshot pins after last path edits must retain fresh identity and original pins.
- Unmatched/ambiguous excerpt identities must not enter saved-source storage.
- Append/history/cross-tab/removal/undo must synchronize all duplicate save controls without losing reading context.

## Task 1 — Operational object freshness (Root inline)

Files: tools/node_status_refresh.py; tests/test_node_status_refresh.py.
Interface: github_fetch returns a URL-bound typed GitSourceState for pinned
sources, existing discussion timestamps otherwise. refresh_record consumes it
strictly while retaining existing injected scalar/tuple compatibility.

- [ ] Add independent offline API-boundary regressions for unchanged snapshot pins,
  changed blob/tree, missing/malformed data, fixed-tip traversal and identity mismatch.
- [ ] Run them against unchanged product and preserve the expected failing output.
- [ ] Implement minimal fixed-commit nonrecursive Git-tree comparison and strict validation.
- [ ] Run all existing refresher tests and optimized interpreter controls.
- [ ] Compare the real unchanged public proof source report without writing records.

## Task 2 — Save a whole source from an excerpt (disjoint UI implementer)

Files: docs/site/library-discovery.mjs; research-desk.mjs initialization only;
tests/test_library_discovery.mjs or a focused companion; new bounded browser check.
Interface: source decorator receives (row, excerpt); post-render callback updates
existing save states. Desk validates exact source lookup and uses its one controller.

- [ ] Add a real-browser failure for text query → source save → list/citation.
- [ ] Implement exact admitted-source decoration after populating the source map.
- [ ] Verify duplicate/append/remove/undo/reload behavior and unmatched-source refusal,
  retaining whole-file versus excerpt identity separation at phone/desktop sizes.
- [ ] Run relevant discovery/desk/store/citation tests and existing18-case discovery checks.

## Task 3 — Frozen review, integration and delivery

- [ ] Stage new site/test files, regenerate/check canonical asset keys once inputs stabilize.
- [ ] Run one final applicable complete local verification with source-qualified originals.
- [ ] Obtain fresh nonauthor whole-branch engineering review; fix reproduced findings.
- [ ] Create/Ready one ordinary PR, qualify actual hosted gates and integrate with expected-head guard through a separate integrator.
- [ ] Verify actual published source/native bytes and fresh downloaded offline copy.
- [ ] Deliver additive source/working/evidence/HANDOFF archives and one existing-register delivery; preserve prior Drive items and personally release own scope.
