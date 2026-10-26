# Atlas Live Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make admitted View selection and equation-search results usable with assistive technology while preserving exact source state.

**Architecture:** Keep the existing Atlas state model. A small DOM helper enables View without retaining disabled option objects, reconciles equation options by excerpt ID, and manages a final-result announcement separate from the immediate visible count.

**Tech Stack:** Existing native DOM modules, Python Playwright1.62, Chromium154 and local Orca48.1; no added production dependency.

**Spec:** `docs/superpowers/specs/2026-10-26-atlas-live-controls-design.md`

## Global Constraints

- Preserve source admission, refused bookmarks, mathematical/formal records, historical snapshots and other contributors' work.
- Source originals must have independently verified private Drive custody before replacement.
- The announcement quiet period is 500ms; actual speech is distinct from DOM assertions.
- Current controls only; no historical renderer rewrite or proof-coverage inference.

## Review Focus

- A surviving selected excerpt keeps the same ID and exact source; removing it selects the placeholder.
- Empty and rapid query changes never announce a stale result.
- Nondefault mobile/URL View survives enabling and history.
- A refused or unavailable snapshot never adopts current source pins through these controls.
- No-JS and 2D/list/3D experiences preserve their existing boundaries.

### Task 1: Source custody and baseline

- [ ] Preserve original complete site at Maina15d2fb6 in private Drive; independently download and verify all members/whole bytes/CRC.
- [ ] Run existing130 current/reader checks and the same real Orca journey before changes. Keep failed setup runs with their reason.
- [ ] Add focused browser assertions for remaining option identity, immediate visible count, settled final announcements, selection/source and View/history/refusal. Run against baseline and retain expected failures.

### Task 2: Live control repair

**Files:** new `docs/site/atlas-controls.mjs`; modify `docs/site/atlas.mjs`, `docs/site/dependencies.html`, `docs/site/atlas-layout.css`; focused driver `tools/atlas_live_controls_browser_check.py`.

- [ ] Implement the smallest helper and integration satisfying the failing assertions. Keep source decisions in the existing controller.
- [ ] Run focused browser checks, actual same-driver Orca replays, existing reader/refusal/3D/formal regressions, visuals and performance comparison.
- [ ] Stage new files, run asset release and check, run all prescribed site-generation/public-shop/formal contract checks. Commit exact implementation.

### Task 3: Reviewable release

- [ ] Add source/version-qualified evidence report and guide command; request actual separate whole-branch engineering review.
- [ ] Address material findings with reproducing checks; preserve source/preimages before replacements.
- [ ] Prepare Draft PR, exact-head preservation and separate integrator handoff. Integrate only through current protected process; verify actual deployed bytes, source observation and reader journeys.
- [ ] Preserve final evidence privately, report limitations and release only own claim.
