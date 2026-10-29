# Library source search implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Recover from an empty current-source search and reopen/share its query.
**Architecture:** Library-only progressive enhancement delegates to the existing
verified search. Existing reader storage and canonical data stay untouched.
**Tech Stack:** Native browser DOM/CSS/modules, Python Playwright, Node test runner.
**Spec:** `docs/superpowers/specs/2026-10-26-library-source-search.md`.

## Global constraints

No dependency, network, storage schema, scientific record, current.mjs, importer,
Atlas or workflow change. Preserve current vs dated source-cut semantics and all
existing safety refusals. Use the existing brand and release-key tooling.

## Review focus

- A literal query with Unicode, &, # or percent text must round-trip as text.
- Clearing after pagination removes only source-q and resets the existing window.
- A refused snapshot must expose no enabled enhancement controls.
- Back and storage/background updates must not move unrelated keyboard focus.
- Link wording must not promise frozen results or authenticated source custody.

### Task 1: Progressive search tools and regression

**Files:** new `docs/site/library-source-search.mjs`, small import/attach in
`research-desk.mjs`, scoped `research-desk.css`, README paragraph;
`tools/library_source_search_browser_check.py` and
`tests/test_library_source_search.mjs`.

**Interfaces:** `startLibrarySourceSearch(doc,win)` attaches once to the verified
enabled Library input/list, returns null without those prerequisites. Delegate
all actual filtering/history writes to current.mjs by dispatching input.

- [x] Write a real browser regression for absent controls/recovery. Run against
  the original immutable base; retain expected failing report.
- [x] Implement the helper and integration; native query link updates from input,
  popstate and observed list replacement. Use textContent for copy, no query HTML.
- [x] Run the focused browser matrix and Node URL/initialization checks.
- [x] Run the prescribed site suite and prior desk/Undo/Cite browser regressions;
  run asset re-key and check, stage all derived assets.
- [ ] Record exact input/output evidence and limitations; commit cleanly, obtain
  separate whole-branch engineering review and current protected integration.
- [ ] Verify actual publication and an additive offline/source delivery in Drive.
