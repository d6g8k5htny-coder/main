# Atlas Claim Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put recorded scope, exact source and explicit formal gaps first without losing detailed evidence or navigation.

**Architecture:** A small DOM-only `createClaimSummary(data,node)` renderer consumes the controller's admitted projection. The controller owns disclosure restoration and refusal cleanup; the spatial shell no longer appends a second evidence projection.

**Tech Stack:** Existing browser modules/native HTML details, Node controller harness, Python/Playwright/Chromium. No new product dependency.

**Spec:** `docs/superpowers/specs/2026-10-25-atlas-claim-summary-design.md`

## Global Constraints

- Keep identifiers; do not invent titles or convert notes into hypotheses.
- Missing formal mapping stays visible while details are closed.
- No mathematical data, formal mapping, source acquisition, workflow/protection, historical evidence or scientific-status edits.
- Preserve2D/List/3D, source refusal, Share/reload/history and no-JS static reading.
- Claude54 owns the actual-site Orca audit; no duplicate audit or inferred speech PASS.

## Review Focus

- Missing scope or an unresolved source must stay explicit rather than borrowing nearby evidence.
- A changed/refused claim must not retain the previous claim's source, formal panel or disclosures.
- Graph/List/layout redraws must preserve native disclosure state for the same claim.
- Long scope/identifiers must wrap without clipping; mobile controls retain visible focus and readable status.
- Moving review data must preserve all literal records, dependency buttons and reading-copy links.

---

### Task1: Source-first projection and native evidence disclosures

**Files:** new `docs/site/atlas-claim-summary.mjs`; modify `atlas.mjs`, `atlas-spatial-shell.mjs`, `atlas-layout.css`; tests `tests/test_current_site.mjs`; new `tools/atlas_claim_summary_browser_check.py`.

**Interfaces:** `createClaimSummary(data,node)` consumes admitted currentSnapshot and the selected literal graph record, returns a section with `atlas-claim-summary`, existing `atlas-formal-evidence`, and stable disclosure IDs. No fetching/state writes. Controller preserves details by ID only for the same node/error identity. The existing h3 remains the selected identifier consumed by the spatial adapter.

- [ ] Write/run the real-controller regression: source text precedes continuation/review metadata; formal gap exists in the controller projection; complete literal fields are retained; same-claim layout preserves details; changed/refused claims clear them. Expected RED on current93ad order/projection.
- [ ] Record four-scheme/width baseline screenshots, source/formal offsets and panel heights using the new browser driver in explicit observation mode. Expected original source/formal order failure, not a false acceptance PASS.
- [ ] Implement the DOM renderer; reuse existing sourceForNode/sourceURL and all seven formal statements, changing only positional wording. Label notes separately from scope. Make formal/review/source-identity disclosures native and preserve state by IDs. Remove shell-owned sibling evidence insertion.
- [ ] Run Node controller and focused browser tests. Expected PASS for source order, missing data, inert strings, keyboard disclosures, layout/history/refusal and exact sources.
- [ ] Stage new files, run canonical asset release/check, commit implementation and focused tests. Expected clean keys and no source/data changes outside the intended presentation set.

### Task2: Persistent status markup and actual AT handoff

**Files:** `docs/site/dependencies.html`, `atlas-spatial-shell.mjs`, `atlas-spatial.css`; browser driver; `docs/site/README.md`; bounded AT tool only if runtime work yields a reusable method.

**Interfaces:** Keep `atlas-status`, `equation-count`, `atlas-space-status` IDs/text writers/description targets on paragraphs inside role=status divs. Hide the spatial wrapper with its contents on narrow non-browse panels.

- [ ] Add/run browser assertions for paragraph targets, live wrapper semantics, dynamic updates and mobile hiding. Expected RED on the direct-text roles before markup changes.
- [ ] Implement only these wrappers and styles; no selector-option/admission changes.
- [ ] Request exact candidate replay from the existing Claude54 lane; preserve its baseline/method/source attribution. Report actual observed output or explicit qualification limits; do not equate AX with speech.
- [ ] Re-run relevant browser checks and canonical asset release/check. Expected no state/source regression; commit the narrow accessibility delta.

### Task3: Qualification, evidence and permitted delivery

**Files:** report `docs/reports/atlas-claim-summary-20261025/**`, prescribed importer keys, README usage/check notes.

- [ ] Replay165 existing focused Node tests plus new regressions,16 actual spatial and10 refusal cases; run the site's prescribed public-shop/current/formal-contract command groups. Expected all applicable checks PASS; preserve failures and documented fixture skips/replays.
- [ ] Inspect before/after desktop/narrow light/dark screenshots and compare the same49/55 graph measurements. Preserve recorded scope/source and label missing AT/device evidence accurately.
- [ ] Freeze exact source and archive evidence; request one fresh nonauthor whole-branch engineering review, fix material findings with causal regressions, and create a reviewable PR. Author does not self-merge.
- [ ] Hand off to a separately claimed integrator after current hosted/Ready review gates; verify actual publication bytes and live reader journeys when permitted. Post source/review/test/deployment identities and release only this invocation's stage.
