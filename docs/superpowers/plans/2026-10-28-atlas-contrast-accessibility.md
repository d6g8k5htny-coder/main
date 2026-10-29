# Atlas fallback accessibility implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Keep restored 2D claims readable and correctly selectable after optional 3D failure.

**Architecture:** Reuse the existing layout-only controller callback after restoring
visibility. One production function changes; new real-browser coverage observes
geometry, pointer selection and source/history invariants.

**Tech Stack:** Existing native modules, Playwright and sandboxed Chromium/SwiftShader.

**Spec:** `docs/superpowers/specs/2026-10-28-atlas-contrast-accessibility-design.md`.

## Global constraints

- No new production dependency or formal/scientific/status/workflow change.
- Keep source refusal and recorded relationship direction/type unchanged.
- Keep raw failures, old audits and private preservation archives.
- Coordinate only the spatial-shell failure hunk; other agents retain their scopes.
- Follow current required checks and separate review/integration ownership.

## Review focus

- Module failure restores geometry just as unavailable WebGL does.
- Context loss after a real rendered update restores the current filtered page.
- Layout repair creates no extra history entry or new source/selection.
- Narrow List fallback and keyboard reach remain useful.
- A refused source never gains substituted current graph nodes.

### Task 1: Reproduce with real browser geometry and picking

**Files:** Create `tools/atlas_webgl_fallback_browser_check.py`.

- [x] Read the current shell, callback contract and independent failure evidence.
- [x] Add real geometry/pointer tests for the review-focus cases and phone/theme matrix.
- [x] Run against the untouched, source-bound Main79bd public capture; retain RED output.

### Task 2: Restore usable layout through the existing callback

**Files:** Modify `docs/site/atlas-spatial-shell.mjs` `fail()` only, plus mechanical release keys.

- [x] After display restoration, call `onChange({write:false})`; preserve existing status and disposal.
- [x] Run required asset release/check and create a separately hash-bound candidate fixture.
- [x] Run the identical new browser assertions; retain GREEN geometry and actual pointer results.
- [x] Run existing spatial, source-refusal and focused controller tests plus prescribed site checks.

### Task 3: Qualify and deliver the bounded result

**Files:** New `docs/reports/atlas-contrast-accessibility-20261028/` records.

- [x] Inspect before/after screenshots, measure observed restore latency and document limits.
- [x] Add source-bound contrast/AT evidence without relabelling old incomplete or failed cases.
- [ ] Obtain fresh nonauthor engineering review; preserve preimages and verified private custody.
- [ ] Publish one reviewable PR; integrate/deploy through current checks and separate owner when permitted.
- [ ] Verify actual deployed-source behavior and close only personally owned finite roles.

