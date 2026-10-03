# Research evidence reader implementation plan

> For agentic workers: use superpowers:executing-plans. One whole-branch nonauthor review follows implementation.

**Goal:** Make recorded scope and evidence navigable without duplicating scientific metadata.

**Architecture:** Derive presentation from the existing SHA-verified graph. Keep source JSON unchanged; use the existing viewer and release generator.

**Tech stack:** Static HTML/CSS, native ES modules, Node tests, hosted Chromium.

**Spec:** ../specs/2026-10-03-evidence-reader-design.md

## Global constraints

Scientific effect NONE. No graph, status, proof, inventory or formal acceptance edits. No new dependencies. Dated snapshot scope remains visible. Library/equation redesign is separate.

## Review focus

- Missing metadata must not imply false, reviewed, accepted or independent.
- Unsafe source/review strings remain text without executable links.
- Saved invalid classifications must refuse visibly.
- Filtering must not change a selected object's meaning or hide a source limitation.
- Keyboard/browser history must retain reader focus and URL state.

### Task 1: Source-derived evidence and filters

Files: docs/site/dependency-model.mjs; tests/test_dependency_viewer.mjs.
Interfaces: evidenceRows(node) returns label/state/detail/href records; classificationOptions(index) returns recorded strings; filterNodes(index,query,classification) intersects search and exact class; readFilters(index,search) returns query/classification/error.

- [ ] Write behavior tests for absent fields, safe source/review links, no classification inference, exact filters, empty all-records view and invalid URL refusal.
- [ ] Run tests and observe missing-interface failures.
- [ ] Implement the minimal derived functions; run all viewer tests.

### Task 2: Reader interface and integration

Files: docs/site/dependencies.html/mjs/css; tests/test_dependency_viewer.mjs; tools/public_shop_browser_check.py.
Consumes Task1 functions. Keep compiled identity validation. Selection renders recorded scope/notes, evidence table, native audit disclosure and existing typed relationships. Input/select controls encode q/classification without dropping node or unrelated URL state; history restores all.

- [ ] Write startup/browser assertions for filter interaction, saved URL, missing evidence, audit disclosure and reset.
- [ ] Observe startup failures before implementation.
- [ ] Implement HTML and DOM rendering; verify viewer tests and all public-site tests.
- [ ] Regenerate supported asset URLs and check source JSON hashes unchanged.
- [ ] Publish isolated PR, obtain whole-branch review, inspect actual-head hosted evidence, integrate only after findings resolved and required checks pass.

## Decision record

Use native execution under the user's explicit modify/adapt approval. Preserve the brand and existing graph as authority. Broader roadmap is an intake record, not a claim of implemented features or mathematical closure.
