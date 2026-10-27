# Library Undo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Recover accidental Remove/Clear without losing concurrent reading work.
**Architecture:** One tab-local recovery array, captured/consumed by the existing locked mutation coordinator; existing validated additive merge for restoration.
**Tech Stack:** Existing native ES modules/HTML; Python Playwright browser checks; no dependencies.
**Spec:** docs/superpowers/specs/2026-10-26-library-undo.md

## Global Constraints

Exact source identities, schema, storage key,50-entry and128KiB bounds unchanged. No Atlas/importer/math/status/workflow changes. Preserve originals and existing refusal/focus behavior; same-family engineering review permitted, scientific acceptance out of scope.

## Review Focus

- Peer writes between removal and Undo survive; conflicting metadata refuses atomically.
- Capacity/byte-limit failure retains recovery for a later retry.
- Queued no-op deletion never erases the previous useful recovery.
- Late completion never steals newer outside focus.
- Storage/lock failure remains explicit; tab-only Undo must not overwrite stored data.

### Task1: Reliable one-step recovery

**Files:** Modify docs/site/research-desk.mjs, docs/site/workspace.html (control/copy and generated key); create tools/research_desk_undo_browser_check.py; document in docs/site/README.md. No model/schema changes.

**Interfaces:** Existing `mutate(operation,success,refusal)` extended with a bounded intent (`remove`/`undo`); existing `mergeEntries(existing,incoming)` performs restoration; DOM control `#desk-undo`.

- [x] Write real-browser regression assertions against complete entry identities and latest persisted list, including adverse inputs above; run original source and record expected missing-recovery failures.
- [x] Add minimal mutation/recovery state and native control; capture actual removal under lock; consume only successful Undo; preserve pending buffer on refusal/no-op.
- [x] Run new browser controls and existing continuity/list/Cite tests; investigate failures from evidence.
- [x] Update README and scoped source/verification report, run asset release tool and full prescribed local verification once on frozen source.
- [ ] Commit with AI attribution, open Draft and obtain separate whole-branch engineering review; resolve material findings with reproduced tests.
- [ ] Preserve exact final source/evidence in Drive, qualify current native gates through actual distinct integrator, verify actual publication/live and downloadable extraction, deliver concise honest completion receipt.
