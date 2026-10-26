# Persistent Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Let readers hear persistent source status without losing source or historical identities.

**Architecture:** Native status wrapper around existing paragraphs on current presentations. Explicit `readable_status` legacy rendering option preserves exact previous live-page restoration; frozen historical rendering remains unchanged.

**Tech Stack:** Existing Python generators, native HTML/CSS, pinned Playwright1.62/Chromium154 and installed Orca48.1; no production dependency.

**Spec:** `docs/superpowers/specs/2026-10-26-persistent-status-design.md`.

## Global Constraints

- Independently verified private custody before replacing original GitHub bytes; public neutral markers only.
- Preserve hypotheses, classifications, timestamps, source snapshots, refusal/admission requirements and historical evidence.
- Do not edit frozen renderers or source-loader/controller/653 control behavior.
- No full accessibility, old-platform speech, scientific acceptance or larger-scale claim from these bounded checks.

## Review Focus

- Previous published live pages restore byte-exactly, with no source/text/pin tamper accepted.
- Historical renderers and archived ZIPs retain their exact original bytes.
- Freshness and error text remains visible and spoken through the same paragraph writers.
- No-JS and static source reading retains exact references and scientific limits.
- Compact panels and preserved-viewer notices keep geometry/focus across mobile, desktop and themes.

### Task 1: Preservation and reproduction

- [x] Verify original site/generator/test/publication archives privately with audience preflight/readback.
- [x] Run current site/build/history baseline checks; reproduce actual persistent status speech with source-matched HTML/payload/meta.
- [x] Add focused generation/restore/admission and browser assertions; retain their intended baseline failures.

### Task 2: Bounded presentation repair

**Files:** `tools/current_site_build.py`, repository-owned live helpers/restore selection only in `tools/current_site_history.py`, `docs/site/brand.css`, `docs/site/dependencies.html`, `docs/site/atlas-layout.css`, current build/history tests, focused browser driver and required generated current/release-key outputs.

- [x] Add wrapper and explicit `readable_status=True/False` live rendering, preserving frozen functions.
- [x] Test both representations by full descriptors, exact legacy bytes, altered pins/text/digest rejection and frozen archive hashes.
- [x] Run matched actual speech, focused browser geometry/source checks and relevant existing controls/3D/refusal/formal checks.
- [x] Run standalone generation, asset release/check and all current prescribed local verification groups; commit exact implementation.

### Task 3: Review and delivery

- [x] Record source-qualified before/after screenshots, speech, performance and limitations with no private locators.
- [ ] Obtain a separate actual whole-branch engineering review; fix material findings with reproductions.
- [x] Create Draft PR, preserve exact head, hand off to a separate actual integrator for current protected process.
- [ ] Verify actual deployed bytes and reader journeys; preserve final evidence and release only own claim.
