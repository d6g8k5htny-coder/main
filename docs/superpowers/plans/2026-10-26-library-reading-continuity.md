# Library reading continuity implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep focus and source-identity disclosures stable while reading-list data updates.

**Architecture:** Reconcile saved rows with existing entryKey(). Retain surviving control DOM, update changed source metadata and citation href, remove missing rows and make only necessary moves. Capture focus at render time and use a reachable fallback only if its control disappears or becomes disabled.

**Tech Stack:** Existing dependency-free JavaScript and Python/Playwright browser runner.

**Spec:** docs/superpowers/specs/2026-10-26-library-reading-continuity.md

## Global constraints

- Same bounded schema and literal source identities; no new storage or backend.
- No current/Atlas/math/source/admission changes; workspace change is derived reader-module key only.
- Engineering only; same-family separate reviewer allowed, scientific review needs different actual family.
- Preserve prior evidence and Drive releases; no deletion or sharing changes.

## Review focus

- Same-key metadata enrichment must update citation SHA and displayed identity.
- Removing an earlier entry must not switch focus by row index.
- Peer removal/clear must leave reachable focus, with closed-list/outside-focus preserved.
- Queued writes capture current focus at completion, not old click-time focus.
- Local-only storage failure must retain the same reading behavior without overwriting storage.

### Task 1: Reproduce, repair and qualify reading continuity

**Files:** Modify docs/site/research-desk.mjs and derived docs/site/workspace.html; create tools/research_desk_continuity_browser_check.py and scoped report.

**Interfaces:** Consumes existing startResearchDesk(), entryKey(), validated entries and native storage events. Produces unchanged public API and data schema, with stable surviving reading controls.

- [x] Write real-browser controls for peer add, all saved control roles, saved search, closed/open disclosure, earlier-row removal, same-key enrichment, focused-row removal, clear and outside/late focus.
- [x] Run against untouched Main bytes; retain expected focus/disclosure failures, outside controls and source/browser identity.
- [x] Implement keyed rows and changed metadata refresh; preserve actual-removal fallback without moving unrelated focus.
- [x] Run targeted controls, original reading-list and citation browser journeys; use actual local-only/queued-lock negative controls.
- [x] Stage additions, generate/check canonical asset keys, run prescribed public-shop tests and preserve reports/source identities.
- [ ] Commit, obtain separate engineering review, resolve actual findings, then hand off ordinary protected integration to a separate eligible integrator. Publish/archive only after actual results.
