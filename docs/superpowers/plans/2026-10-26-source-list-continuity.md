# Current-source continuity implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.
> Root implements the product change inline; a delegated engineer owns the
> independent browser regression, followed by a fresh whole-change reviewer.

**Goal:** Preserve source evidence and keyboard position while revealing more files.

**Architecture:** Retain the existing immutable admitted snapshot and filter.
Append only a new suffix for an unchanged query; retain replacement on query
reset. Focus a newly revealed source only when the activated More button owned
focus. No loader, data or mathematics change.

**Tech stack:** Existing browser ES modules; Python Playwright/Chromium driver;
existing Python/Node site checks and release-key generator.

**Spec:** `docs/superpowers/specs/2026-10-26-source-list-continuity-design.md`.

## Global constraints

50-file batches; immutable accepted observation; no production dependencies.
Preserve literal source identities, source admission/refusal and mathematical status.
Do not delete, move, rename or reshare Drive files; delivery is additive.
Use explicit OpenAI Codex attribution for Git commits; no human review claim.

## Review focus

- New query after expansion replaces the old prefix and resets the window.
- Programmatic query changes before More cannot mix two query populations.
- A final partial batch focuses its first newly revealed named source.
- A programmatic click while another control owns focus does not steal focus.
- Retained acquisition details and save controls stay the same DOM objects.

### Task 1: Repair and qualify source browsing

**Files:** Modify `docs/site/current.mjs` results/More block only; create
`tools/source_list_continuity_browser_check.py`; add a small usage note to the
existing site guide; include all release-tool-derived importer/HTML keys.

**Interfaces:** Consume existing `filterSources`, `renderCurrentSource`, admitted
`data.sources` and annotations. Keep their interfaces and output untouched.
The browser driver consumes the actual included site or optional `--origin`
and produces report JSON/screenshots outside the repository.

- [ ] Write the focused real-browser cases for all five review-focus conditions,
  exact ordered URLs/no duplicates, clear/history and refusal controls.
- [ ] Run on unchanged c1678a5; retain actual behavioral RED separately from setup
  failures. Product write waits for actual #501 hunk reconciliation.
- [ ] Track rendered query/count in the existing result closure. Append only the
  suffix on an unchanged query; otherwise replace/reset. Capture focus before
  handling More, then focus the first new source link/row only if it owned focus.
- [ ] Run focused GREEN in desktop/phone themes; run existing Library search and
  reading-desk continuity checks. Keep original RED and every failed attempt.
- [ ] Stage additions; run `python -B tools/site_asset_release.py` and `--check`;
  run the site guide's prescribed public-shop checks and repository full
  discovery, recording unrelated failures by name rather than silently repairing
  peer-owned scientific or test work.
- [ ] Obtain a fresh separate engineering review; fix supported findings, rerun
  affected checks, commit explicit AI identity and open the ordinary PR.
- [ ] Reconcile current Main/hunks and required hosted checks; perform only the
  guarded authorized integration and verify the landed/public/offline behavior.
- [ ] Deliver an additive source/site/evidence download with actual hashes,
  changes, benefits and open limits. Release only this task's finite claims.

Self-review: every spec requirement maps to this single component task. The two
browser control-state branches are tested independently. No new shared API or
mathematical acceptance path is introduced. Existing user delegation selects
continued execution and supersedes the generic skill approval prompt.
