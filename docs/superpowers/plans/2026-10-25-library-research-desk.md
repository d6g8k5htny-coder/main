# Library research desk implementation plan

**Goal:** Turn source discovery into a resumable, source-linked reading task.
**Architecture:** Add a Library-only progressive enhancement and bounded local reading-list store. Reuse currentSnapshot and existing catalog/source query contracts.
**Tech:** HTML/CSS, native modules, Node tests, Python Playwright.
**Spec:** ../specs/2026-10-25-library-research-desk-design.md
**Execution:** Implementer executes under Dylan's explicit continuing authorization; separate same-family engineering review permitted.

## Review focus
Unsafe imported identities/URLs; failed or unreadable storage; cross-tab lost updates; readiness/focus theft; current/datetagged query confusion. No scientific approval is supplied.

## Checklist
- [x] Store contract: write meaningful negative parser/identity/capacity/concurrency tests, observe RED, implement strict validated entries and atomic merge.
- [x] Library view: visible collection choice, reveal current query, explicit source/query Save, empty/ready/failure states and remove/export/import; no loader changes.
- [x] Actual browser controls: native keyboard selection, source identity roundtrip, current/datetagged queries, persistence and additive import, malformed/oversize input, unavailable storage, snapshot refusal, no-JS,320px/light/dark/print and history.
- [ ] Freeze candidate; canonical release-key rewrites; prescribed public-shop checks and focused browser/source-conservation review. Inspect screenshots.
- [ ] Separate engineering review and fix material findings; ordinary Ready/review/protected integration; actual live/package verification and additive Drive archive.

## Files and interfaces
New research-desk-store.mjs owns read/validate/merge/serialize and safe source/query URL construction; maximum50 entries and128KiB input. New research-desk.mjs owns DOM projection, storage custody, imports/downloads and currentSnapshot hookup; research-desk.css owns Library-only styles. workspace.html adds the surface outside CURRENT-SITE. New test_research_desk.mjs and research_desk_browser_check.py exercise behavior. No app.js/current.mjs/current-model.mjs, Atlas or historical data edits.

Progress/evidence remain at /workspace/research/library-research-desk-20261025, outside source. No temporary evidence is deleted.
