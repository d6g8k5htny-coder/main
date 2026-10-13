# Current research site implementation plan

> For agentic workers: use the Superpowers implementation and verification workflow. Standing owner authorization supplies the execution decision; root coordinates disjoint generator and UI work, followed by a fresh branch review.

**Goal:** Make the public pages automatically reflect current source commits and provide an interactive mathematical atlas.

**Architecture:** Generate an immutable public source observation, render it through shared reader navigation and a dedicated equation/graph atlas, and deploy a verified generated Pages artifact on source updates and a schedule. Preserve dated evidence.

**Tech Stack:** Python standard library; browser ES modules and SVG; existing brand CSS; GitHub Actions and Pages.

**Spec:** `docs/superpowers/specs/2026-10-13-current-research-site.md`

## Global constraints

- Scientific effect NONE; no new scientific status register or inferred evidence.
- Source commits are 40-hex Git commits, source SHA-256 values are 64-hex; sources are public `d6g8k5htny-coder/main` and `Math-` only.
- Preserve existing dated source exports and reading cuts.
- No fetched-code execution, external font/analytics/theme framework or private source ingestion.

## Review focus

- Branch movement during generation must not mix commits.
- Partial or stale source observations must never appear current.
- Hostile paths and equation strings must remain inert.
- Required and contextual edges must preserve different meanings.
- Scheduled deployment must leave the last successful site available on failure.

### Task 1: Immutable current-source generator

Files: `tools/current_site_data.py`, `tests/test_current_site_data.py`, generated `docs/site/current.json`.

Interface: payload `{schema_version:1, observed_at, scientific_effect:'NONE', repositories:[{repository,commit,...}], sources:[{repository,commit,path,blob,bytes}], graph:{identity,nodes,edges,...}, equations:[{id,text,repository,commit,path,line_start,line_end,sha256}], references:[...]}`. Root and generator author settle exact added fields in code before UI integration. `--output` writes atomically after complete validation; offline reader fixtures exercise real generation.

- [x] Write negative controls for missing/truncated/mismatched input, unsafe paths and required-edge cycles; observe failures.
- [x] Implement pinned reads and deterministic current inventory/equation export.
- [x] Verify controls and generate from public current commits, preserving exact source readback.

### Task 2: Current navigation and mathematical atlas

Files: `docs/site/current-model.mjs`, `current.mjs`, `current.css`, `atlas.mjs`; all site HTML navigation; `tests/test_current_site.mjs`.

- [x] Write model tests for validation, freshness, independent node/equation state, filtering and required/context neighborhood traversal; observe failures.
- [x] Implement hash-verified local current payload rendering, accessible graph and equation collection.
- [x] Link five primary pages and expose current observations on every linked page. Keep script-free Research's generated current section independent of frozen cuts.
- [x] Verify browser behavior and current-source refusal, desktop/mobile light/dark and keyboard.

### Task 3: Generation and publication

Files: `tools/current_site_build.py`, `.github/workflows/current-site.yml`, affected publication/control tests and site guide.

- [x] Test rendered current Research section and historical byte preservation, workflow event/permission/concurrency controls.
- [x] Implement generated build from checkout plus pinned current sources; stage and run asset-release generation/check.
- [ ] Run complete applicable public-shop checks, fresh branch review and required hosted checks.
- [ ] Integrate under current protection, verify Pages settings, execute deployment, read deployed bytes and actual interactions back.

## Execution ledger

Baseline: 64 reader-link tests and 82 dependency/navigation tests pass at main b7deb198. Default shell lacks Node; bundled Node is available. Native worktree creation reported Not a git repository in the empty task folder; isolated Git worktree created from authoritative main, preserving the existing checkout.

Task 1: complete in initial PR387 candidate ac79faced79d0e72c83b19e527631e570b217eca; 30 offline generator controls passed normal/optimized, 108/108 excerpts independently matched pinned Git source lines.
Task 2: initial candidate and subsequent review fixes implemented; current model11 controls and current Chrome10 cases pass. Nonauthor reviewer identified stale bookmark, inherited-key selection, graph-filter persistence and static mobile defects; controls reproduced them and fixes passed targeted review. Hosted legacy-reader check additionally found startup URL mutation, fixed by deriving a share URL without changing the address until an atlas interaction.
Task 3: initial publication/archive candidate is draft; remote source history restoration remains in implementation. Do not enable Pages Actions until its tests/review, final exact-candidate required checks and native deployment are verified.
Final: review note resolved - five axes remain metadata presence; review source paths now link through the pinned inventory and nested metadata renders as inert JSON. Foreground age recomputes every minute. Missing/malformed payload browser report controls observed RED then GREEN.
Final: Ruling - source discovery uses a standard-library generator with the existing static-site asset pipeline rather than adding another SSG runtime; this preserves current URLs and avoids replacing the scientific source vocabulary. Cost if wrong: later reader adapters may require another renderer, while pinned generated inputs remain reusable.
Final: Ruling - Canvas presentation is adapted to /site/dependencies.html and static /site/nodes/ rather than modifying the active /reader/ candidate. Its owner retains runtime/lease/formal source scopes; all Canvas reader/parity/source/custody requirements stay in this task. Cost if wrong: route integration needs a compatibility adapter, not invented evidence joins.

Successor verification: 681 Python tests passed with six explicit skips; the pinned-source interaction harness passed. Current model23 controls passed, and broad Node controls passed411 with two explicit skips (the fixture-dependent public-shop app was separately exercised by the pinned-source harness). Current Chrome14 cases passed, including keyboard reading, structured accessibility trees, 320/1200px text enlargement, monochrome distinction and synthetic1001-node paging. Spoken assistive-technology use is not observed. Repeated complete Docs builds agree across329files; final release/commit and hosted current checks bind the published successor.
Successor implementation: public history restoration and closed deterministic bundles retain two reviewed source snapshots, subject to bounded capacity; generated-artifact Chromium now gates upload. Current snapshot reachability/routes, omitted-frontier pagination and equation text search close the acceptance-audit findings. No semantic join or scientific evidence is inferred. Actual deployment, exact public readback and an observed automatic refresh remain required before goal completion.
