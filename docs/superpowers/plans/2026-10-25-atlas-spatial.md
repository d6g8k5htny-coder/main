# Atlas spatial view implementation plan

> Execute inline with Superpowers executing-plans and test-driven-development.

Goal: let readers explore the current verified graph in optional 3D while retaining canonical selection, evidence, filters and source-bound refusal.

Architecture: add a presentation adapter to atlas-shell. It observes only the existing controller's rendered list and selected heading; graph geometry uses edges from currentSnapshot after verification. Node picking invokes the canonical list button. Keep atlas-view graph/list and add atlas-space=3d and atlas-space-focus=dependencies, composed by existing shell URL hooks. No controller semantic edits, parallel graph authority or inferred joins.

Approaches: a replacement graph library would duplicate state and touch held controller scopes; a shell adapter preserves ownership and the existing page boundary. Choose the adapter. Use locally vendored MIT Three.js 0.160.0 (same release as the museum), loaded only on explicit 3D activation or a 3D share URL. Implement camera orbit/pan/zoom directly, without further dependencies. No animation/inertia, including reduced motion. Desktop side inspector; existing mobile browse/reader navigation retained.

Formal boundary: inspect actual Main and Math manifests, target declarations, receipt and alignment formats. Current graph has zero explicit artifact bindings. Report all six fields (acquisition/review/declaration+hypotheses/source revision/checked execution/alignment) as unmapped where missing; link the existing evidence guide and separately discoverable source-pinned formal packages, never claim coverage by name or CI. Scientific status remains unchanged.

Constraints: no workflow/protection changes; archive preimages before editing; no Drive deletion; preserve historical content and refusal; maximum 150 displayed nodes; no idle frame loop; GPU cleanup on exit/context loss; no render when hidden; accessible node and typed relationship lists.

## Tasks

- [x] Baseline: capture identical search→select→dependency→source/share journeys at 1200×900 and 390×844, light/dark; record load/search timings and screenshots on pinned committed dataset.
- [x] Model RED→GREEN: deterministic finite bounded coordinates, dependent→dependency edges and relationship types, focus subgraph, no input mutation or fabricated nodes. Files: atlas-spatial-model.mjs, tests/test_atlas_spatial.mjs. Run node --test tests/test_atlas_spatial.mjs.
- [x] Renderer + shell: installSpatialAtlas({root,onChange}), restore/decorateURL/dispose interface. Observe only list/inspector; ensure latest-state wins across async dependency loading. Renderer returns update/reset/zoom/pause/dispose. New files atlas-spatial-shell.mjs, atlas-spatial.mjs, atlas-spatial.css, vendor/three-0.160.0; shell hooks only. Test real WebGL/picking, no-WebGL, blocked module, context loss, history and refusal with tools/atlas_spatial_browser_check.py.
- [ ] Evidence + verification: same journeys after; performance/render count/visible ceiling; screenshots and concise limitations. Run prescribed current-site generation into disposable site and all site README checks, asset release/check, existing Atlas browser suite. Archive exact logs and record source identities, license hashes and missing mappings in report. Commit, create reviewable PR, request nonauthor review/separate integrator. Integrate only through those actual owners and controls.

Review focus: stale async imports after switching views; refused source bookmarks; mobile hidden-reader rendering; pointer drag versus node pick; GPU resource loss and reactivation. Each is a browser acceptance case.

Ruling: user explicitly directed proceeding into implementation and asking only for genuinely missing information; no additional design permission round. No subagent tool is available; request repository nonauthor review after concrete delivery.
