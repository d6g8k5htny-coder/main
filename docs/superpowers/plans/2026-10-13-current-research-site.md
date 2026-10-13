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

- [ ] Write negative controls for missing/truncated/mismatched input, unsafe paths and required-edge cycles; observe failures.
- [ ] Implement pinned reads and deterministic current inventory/equation export.
- [ ] Verify controls and generate from public current commits, preserving exact source readback.

### Task 2: Current navigation and mathematical atlas

Files: `docs/site/current-model.mjs`, `current.mjs`, `current.css`, `atlas.mjs`; all site HTML navigation; `tests/test_current_site.mjs`.

- [ ] Write model tests for validation, freshness, independent node/equation state, filtering and required/context neighborhood traversal; observe failures.
- [ ] Implement hash-verified local current payload rendering, accessible graph and equation collection.
- [ ] Link five primary pages and expose current observations on every linked page. Keep script-free Research's generated current section independent of frozen cuts.
- [ ] Verify browser behavior and current-source refusal, desktop/mobile light/dark and keyboard.

### Task 3: Generation and publication

Files: `tools/current_site_build.py`, `.github/workflows/current-site.yml`, affected publication/control tests and site guide.

- [ ] Test rendered current Research section and historical byte preservation, workflow event/permission/concurrency controls.
- [ ] Implement generated build from checkout plus pinned current sources; stage and run asset-release generation/check.
- [ ] Run complete applicable public-shop checks, fresh branch review and required hosted checks.
- [ ] Integrate under current protection, verify Pages settings, execute deployment, read deployed bytes and actual interactions back.

## Execution ledger

Baseline: 64 reader-link tests and 82 dependency/navigation tests pass at main b7deb198. Default shell lacks Node; bundled Node is available. Native worktree creation reported Not a git repository in the empty task folder; isolated Git worktree created from authoritative main, preserving the existing checkout.
