# Readable 3D labels Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: use superpowers:executing-plans.

Goal: read complete visible claim IDs in the existing optional 3D Atlas.
Architecture: pure screen-label packing plus measured native DOM labels in the
existing on-demand renderer. Source/state/geometry/camera remain unchanged.
Tech stack: existing native JS/CSS, Python browser tooling and pinned Three.js.
Spec: docs/superpowers/specs/2026-10-26-atlas-readable-3d-design.md.

## Global constraints

- Exact IDs/recorded directed typed edges/source refusal remain authoritative.
- No graph, controller, URL, formal evidence/validator/workflow or dependency change.
- Complete visible IDs, measured bounds, selected-first deterministic packing;
  labels need not all fit. Keep the accessible list complete and its metadata intact.
- Preserve original bytes and failed probes; source/fixture/observer identities separate.

## Review focus

- Different label heights/widths must not overlap or leave the viewport.
- Long selected IDs must wrap rather than ellipsize; no injected markup.
- Tiny/empty viewports and clipped points must not produce invalid placements.
- Dense views may hide labels but must retain node picking and complete list IDs.
- Camera/selection updates and cleanup must not add idle rendering or stale labels.

### Task 1: pure geometry and renderer labels

Files: new docs/site/atlas-spatial-labels.mjs and
tests/test_atlas_spatial_labels.mjs; modify atlas-spatial.mjs/atlas-spatial.css.
Interface: placeSpatialLabels(entries,width,height), entries
{id,x,y,z,width,height,selected}; output {id,left,top,width,height} rectangles.

- [ ] Write tests for measured rectangles/bounds/own-anchor gap, selected priority,
  deterministic input order, clipped/invalid viewports and a150-node projection;
  observe intended RED, implement minimal packing, verify GREEN.
- [ ] Add browser RED for complete long IDs and actual rectangle intersections.
  Replace only label placement, batch DOM dimension reads, use full wrapping text,
  preserve rendering/camera/graph behavior. Verify successful real rendering/picking.
- [ ] Stage new site file, run canonical asset release/check, commit all importers.

### Task 2: matched reader evidence and prescribed verification

Files: new tools/atlas_spatial_labels_browser_check.py and dated report;
existing spatial browser observer only if exact canvas-hit accounting is needed;
correct prior reportREADME byte figure to15,359. Consumes the same frozen baseline
driver/fixture and Task1 renderer; produces before/after PNGs and qualified metrics.

- [ ] Run matched phone/desktop/light/dark journeys and inspect screenshots;
  compare complete-ID readability, label bounds, picking and render/interaction cost.
- [ ] Run existing real3D/refusal/current-reader checks and all current prescribed
  local commands; source failures stay failures, no assertion weakening.
- [ ] Record original failures, exact source/fixture/driver identities and limits.

### Task 3: bounded delivery

- [ ] Fresh whole-branch engineering review, current Ready/required checks and a
  separate actual nonauthor/nonreviewer integrator; retain head/public preimages.
- [ ] Ordinary permitted expected-head integration, exact deployed-source assets
  and live journeys; no author merge/protection bypass or foreign claim release.
- [ ] Private final archive with independent canonical/member/whole-byte/CRC
  readback, final PR record and release only this author's finite claim.
