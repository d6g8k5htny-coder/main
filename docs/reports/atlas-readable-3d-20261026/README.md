# Read the complete identifier in the 3D Atlas

Visible labels now wrap complete claim IDs and use their measured dimensions.
The selected claim has priority, label boxes stay separate and inside the
viewport, and a distinct unselected projected node is kept clear for canvas
picking. Node positions, camera behavior, recorded directed/typed edges,
selection, filters, source pins and the complete accessible list are unchanged.
The label helper loads with the optional renderer; no new library is added.

[Before phone](before/initial-graph-390x844-light.png) ·
[after phone](after/initial-graph-390x844-light.png) ·
[before orbit](before/after-12-orbits-graph-1200x900-dark.png) ·
[after orbit](after/after-12-orbits-graph-1200x900-dark.png).

## What improved for a reader

The exact selected ID, math.d5-component.punctured-pin-continuum-check,
was truncated in all52 before observations (313px scroll width in a228px
client box). It is fully visible in all52 after observations. All visible
after labels have zero text overflow, host/viewport clipping or pairwise
intersections. Four matched journeys retain all49 accessible nodes and55
recorded relationships, native keyboard/camera/navigation and source links.
All four actual canvas-hit-verified picks complete before and after.

The visual review confirms the distinguishing end of a long ID is now readable
on390px and1200px layouts in both schemes. Desktop fits7–9 complete labels in
these views; phone fits4–6. Labels are still culled when space runs out. A dense
edge-on graph cannot show all49 identifiers simultaneously; orbit, dependency
focus, the inspector and accessible list remain necessary reading paths.

## Measurements

Identical49-node/55-edge fixture, unchanged observer, Linux Chromium154.0.8037.57,
Playwright1.62, sandbox enabled, SwiftShader, reduced motion, DPR1. Four contexts
and48 measured keyboard orbits per run. These are shared-environment observations,
not a statistical speedup or physical device/GPU promise. Potential parent CPU
overlap in the before run is recorded by the helper.

| Observation | Before | After |
|---|---:|---:|
| Selected ID fully readable | 0/52 | 52/52 |
| Verified canvas picks | 4/4 | 4/4 |
| Load to admitted graph, median | 320.6ms | 326.4ms |
| 3D activation, median | 274.3ms | 250.4ms |
| Key to observed frame, median | 17.9ms | 19.4ms |
| Renderer CPU, median | 3.65ms | 3.30ms |

Resource bodies before3D activation differ by−1byte; the whole journey adds
2,219bytes. Neither Three.js nor the label helper loads before activation.
Keyboard timings include driver polling; renderer CPU excludes GPU completion.
Exact per-case/frame values are in MEASUREMENTS.json and observation projections.

## Failures retained and verification

Six initial pure-packing tests and four real-browser cases first failed as
intended. The first wrapping candidate made IDs readable but covered every
visible blue node after an edge-on orbit: the matched observer correctly
reported0/4 verified picks. retained-picking-regression.json preserves that
failure; it is not a pass. A seventh test reproduced the occlusion, then the
placement correction kept a distinct unselected anchor clear and preferred
positions covering fewer nodes. All7 pure tests and146 focused Node tests pass.
The existing spatial observer now checks elementFromPoint is the actual canvas
before clicking a sphere; overlaid label-button clicks receive no GPU credit.

Final local checks:4focused label browser cases,16existing real spatial cases,
10source-refusal cases and39existing reader browser cases pass. All14 current
prescribed local groups and canonical asset release/check pass. Chrome channel
is absent locally; the39-reader replay uses only a Chromium executable launcher
substitution, with sandbox/CSP/assertions preserved. Its code is detachedbb05;
its payload is the separately identified frozen public observation.

No new whole-AT, Safari, physical touch/GPU or larger-graph qualification. Actual
a13/Claude54 accessibility work and the #637 formal amendment remain separate.
There is no new formal connection, mathematical review or scientific acceptance.
The prior formal-source report's prose byte figure is corrected to its existing
raw15,359B value; original bytes were privately archived before replacement.

## Source and reproduction

Before codec447d102; after productbb05e7c2. The matched after run used workingtree
correction bytes later committedbb05; exact module hashes were compared. It is
not relabelled as a committed execution. Payload Mainc447d102 / Math8b545105,
SHA2566ce5f31f34deaae34c94e525079378ce7cd6eca255127104182afe8e5e38e245.
This records the observed public source pins, not an endorsement of the separately
reported Math merge process. Source, fixture, driver and actual operator identities
remain distinct in SOURCE.json.

Use the byte-identical fixture/current.json with copies of each source's docs/site;
run tools/current_site_build.py --payload EXACT_JSON --site COPIED_SITE, then run
matched-observer.py --site COPIED_SITE --payload EXACT_JSON --output NEW_OUTPUT
--code-label ACTUAL_SOURCE. The output must be under the copied observer's directory.
The observer does not write the checkout or inject renderer state. The primary
browser assertions are tools/atlas_spatial_labels_browser_check.py and the existing
spatial/refusal/current-reader observers. The source code identity does not replace
the recorded served-file hashes.

Observation JSON files are explicit projections: only repeated hidden-label
geometry is elided. Every visible/selected label, measured count, inventory,
failure and timing remains. Complete original raw reports, failed probes,
baseline seal and runtime qualification are retained privately with exact hashes
in SOURCE.json. Baseline helper Nietzsche is implementation-side OpenAI/Codex,
same provider/org0; after operator is the parent author. Neither is reviewer,
human, scientific or AT approval. Separate source review/integration and actual
deployment outcomes belong to current PR records, not inferred from this report.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
