# Verification museum implementation plan

> For agentic workers: use the subagent-driven-development workflow for the independent projection, display, and geometry tasks; root owns integration and release.

**Goal:** Expose pinned public mathematics, its exact scoped reviews and open limits, and landed intake packets without changing scientific status.

**Architecture:** Extend the existing static Pages shop with a separate museum page and a small reproducible display projection. Reuse the existing public-math inventory and trusted incoming guard. Geometry explains source language; it is never replay evidence or an acceptance register.

**Tech stack:** Python standard library exporter and checks, vanilla ES modules/SVG, an isolated optional Three.js scene with a 2D fallback.

**Spec:** Owner's PUBLIC VERIFICATION DISPLAY + INTAKE ENGINEERING contract, 2026-09-26; W1–W12 are mapped below.

## Global constraints

- Scientific effect NONE. STATUS, proof bodies, review dispositions, and vault claims stay byte-identical.
- Math snapshot: `d6628da09384728992dcbe6e921cc28ba85aebb0`; main snapshot: `71400b94f6cb354a8cf7aba73ffede2138a64efa`.
- Existing STATUS/inventory snapshot and SIDE24 enclosure pins remain historical and explicit; query checkout stays `c88768bb11efd1f7d6bda188f13064bedec54a06`.
- Every exhibit names its object, class, repository/path, full commit and SHA-256, and states: “This canvas explains the pinned source. It is not a proof and does not change status.”
- No new repository or source inventory. No live field sampling. No outsider write or automatic packet promotion.
- One main PR, draft until its hosted checks pass. Do not touch files owned by open PRs; specifically leave `docs/NAVIGATION.json` to #125 and the draft packet in #150 to its author.

## Tasks and verification

- [x] **W1–W2: exact claim projection.** `tools/museum_data.py` regenerates `docs/site/museum.json`: all 11 reviewed-result bullets and three AMEND rows, each with immutable proof/review identities. Preserve EXACT_COUNTEREXAMPLE and review-pointer limitations. Tests reject changed source bytes, quotes and invented acceptance. `museum.html`, `museum.mjs`, and `museum.css` show claim/source/replay columns and an engineering strip.
- [x] **W3/W8: existing-shop integration.** Complete the SIDE24 visible identity contract, link the museum, and filter the original catalog by repository/path. Keep exact decimal endpoints. Replace the mutable browser ref fetch with the pinned query identity and local check instructions. Exercise real HTML IDs and filter behavior.
- [x] **W4–W7: scoped exhibits.** Show a fixture-absent D2 card; EC-014 frame with camera orbit and 2D fallback; fixed-remote and fixed-annulus diagrams with OPEN complements; source-stated P15 finite example. Test displayed domains and controls; do not invent bounds or certifying samples.
- [x] **W9: landed packets.** Project only the single packet present at the main snapshot, with RESULT identity, issue not stated, and “packet — not STATUS.” Reject draft-PR inclusion and unknown source identity.
- [x] **W10–W11: retain intake/replay.** Existing trusted `public-intake.yml`, guard, task form, manual claim protocol and SIDE24 notebook already implement these lanes. The audit found no missing write-isolation feature.
- [x] **W12/release.** Keep README and Math/query pins consistent. Run existing navigation, intake and public-shop checks plus new projection/view checks. Review the entire diff, then attach every touched file's commit and SHA-256 to the single PR. Hosted CI must pass before readiness; deploy only through the repository's permitted merge path.

## Review focus

- D5 fixed-annulus ACCEPT must never relabel the open pin-neighborhood/microdisk scope.
- Imported EC-014 self-labels are custody bytes, not adopted dispositions.
- Mutable review comments receive a pinned pointer identity, never an invented comment digest.
- Missing WebGL or unavailable source bytes leaves readable fallback/absence states, not a success badge.
- A packet is listed only from the pinned default snapshot; no result is promoted onto claim cards.

## Audit baseline

Current defaults and the live Pages site were inspected. Main's 17 open PR file lists were checked before edits. The baseline passed 85 Python tests and 16 frontend controls. Math and query need no PR. No committed lifetime sample was found, so the implementation uses the expressly allowed fixture-absent card.

## Pre-publication validation

The implementation passed 97 Python tests in normal and optimized modes; 16 existing frontend controls, four shop interaction controls, 15 museum frontend/integration controls, and eight geometry controls. Public source readback, navigation and landing checks passed. Independent source/scope and engineering reviews found no remaining critical or important findings after the source-association and bounded-fetch fixes. No scientific source/status file changed. Hosted PR checks and publication are release gates; these local checks do not assert deployment or successful GPU rendering.
