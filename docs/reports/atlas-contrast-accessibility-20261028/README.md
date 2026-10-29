# Atlas fallback recovery and accessibility evidence

When optional 3D failed, restored Graph cards could overlap by 41.734375px.
Clicking covered congruence-erratum metadata selected the next claim. Recovery
now remeasures the visible graph and keeps its roughly 24px card gaps. It also
keeps the evidence inspector, its disclosures and keyboard focus intact.
Selection, filters, source pins and browser history remain consistent.

The final product composes the separately merged Library/status/camera work
from PR682. Its only semantic production changes are the graph-only callback
in `atlas.mjs` and the recovery request in `atlas-spatial-shell.mjs`.
The other two site changes are canonical release keys. No dependency, formal
mapping, workflow, check authority or scientific status changes.

| View | Original79bd | Initial geometry repair | Final composition |
| --- | --- | --- | --- |
|390px light|[Covered metadata](before-390-light.png)|[Separated cards](after-390-light.png)|[Readable current graph](composition-390-light.png)|
|1200px dark|[Overlap](before-1200-dark.png)|[Separated cards](after-1200-dark.png)|Same layout guard, actual context-loss test|
|320px forced dark|[Overlap](before-320-forced-dark.png)|[Separated cards](after-320-forced-dark.png)|Forced-color case passes|

The six original before/after images and current 390px image were inspected.
These are local rendered observations, not a human usability study or physical
phone. Narrow 2D graphs still need panning; List is a useful alternative.

## Source, dataset and review finding

Qualified product: `2d64de7b16b09a954d30cdc0b55b63b3e89261c7`, tree
`fb453db99053bc5ab2bdb714e6f80dca79b0b196`, exact base
`eb79bc9360c041da8b0fd24e52a09c4fbe8a47fd`.
Native publication, final reviewed head and actual deployment are recorded
separately in PR683. They are not inferred from this local result.

Frozen admitted observation: Main `79bdde89a3daf7eaf5d4481961eb87433dd002d5`,
Math `8151f592918a845bf19eb8eafaf8444540a32734`; 49 indexed nodes,
55 directed edges, 108 excerpts, 1,923,440 payload bytes; SHA256
`8a3595667134e0b196ad957277a572d34be6f39a3eebf0741faed55830f2e989`.
Failure tests use a seven-card neighborhood and five filtered cards, not an
unbounded graph. [Source binding](SOURCE.json) verifies all 68 bound fixture assets against the committed product. The HTML retains the
original observation block. Original execution fields remain unchanged.

The first repair, b43b7438 / final report2b0d5c19 / native Draft7dd9f529,
passed its twelve tests but received an Important engineering HOLD (E1): a full
controller render detached a focused inspector summary during context loss.
The next Tab then jumped to the source link. That review remains a HOLD on its
original source. [Original report](phase1-report.md) and
[original binding](phase1-source.json) are preserved as an earlier phase.

The new thirteenth regression test fails on that original repair and passes on
the current composition. Recovery requests `graphOnly:true`; it reflows only
the displayed graph and keeps unrelated inspector/List DOM connected. It skips
refused sources, hidden graphs and List views. No focus restoration heuristic
or extra history entry is needed. [E1 RED](focus-e1-red.json),
[current GREEN](composition-fallback.json), [graph after the focus case](composition-focus.png).
A separate current-source nonauthor engineering review is required before Ready.

## Current composition verification

| Check | Result at product2d64 |
| --- | --- |
|Focused failure/history/source/geometry/pointer operator|13/13 pass|
|Actual successful 3D, node picking, filter then context loss|Pass in focused and existing spatial operators|
|Open inspector metadata, context loss, focus and next Tab|Exact original summary remains focused; next Tab reaches metadata PRE|
|Existing spatial operator|16/16 pass|
|Existing source-refusal/static operator|10/10 pass|
|Prescribed and retained release groups|13 groups pass, including new Root tests|
|Two complete site generations at fixed admitted input|1,721 files / 57,048,055 bytes identical; all 12 commands exit0|

[Spatial](composition-spatial.json), [refusal](composition-refusal.json),
[prescribed commands](composition-prescribed.json), [generation](composition-generation.json).
These include reduced motion, keyboard access, mobile/desktop light/dark,
320px forced colors, module failure, unavailable WebGL, context loss, sharing,
reload/Back/Forward and refusal without source substitution. Existing JS-disabled
and List paths remain qualified by the refusal/static and spatial operators.
The focused tool is a local/public release operator; no CI workflow was changed.
Native required checks, automatic review and deployment qualify separately.

The original identical twelve-case operator gave ten failures/two passes on
79bd and twelve passes on b43. Its initial 11/12 trial corrected only a refusal
word expectation (stable “refused”, rather than “unavailable”); original pins
and no-substitution guards remained. All those raw results are retained.

## Contrast and actual screen-reader evidence

The source-bound diagnostic measured subtitle contrast 10.400134:1 dark /
6.278236:1 light; its decorative pseudo-element ends 13px before the first glyph.
Normal graph text measured 15.762223:1 / 15.252424:1 and List metadata
8.586809:1 / 5.934460:1. All 6,128 character-center samples in eight panned
variants hit their owner. Four long-title axe incompletes persisted; their
engine cause is unresolved. Original pseudoContent/incomplete records remain.
These measurements do not establish full WCAG conformance. [Evidence](contrast.json).

Actual Orca48.1/Chromium154 speech and nonzero PCM qualified scope, revision/blob,
arithmetic declaration, hypothesis explanation and dependency selection with an
explicit missing mapping on original79bd. Separate Orca46.1/Chromium141 diagnostic
attempts were inconclusive for declaration reading; intervening screenshot capture
coincided with AT-tree invalidation, a protocol confound of unisolated cause.
A separately attributed parent control on b43 omitted that capture: native
source/formal disclosures opened and Orca spoke the path/revision, pairing_sum_six,
arithmetic statement and coefficient/parent-theorem limitation. Complete blob,
hypothesis and execution/alignment reading on Orca46 remain unqualified.

[Orca48](speech/orca48.json), [Orca46 control](speech/orca46.json),
[qualification](speech/qualification.json). These earlier source-specific runs
are not relabelled as AT executions of the eb79 composition. No human audition,
NVDA/JAWS/VoiceOver or physical touch qualification is claimed. Full contrast/AT
raw diagnostics and failed attempts remain retained locally; the report and
private delivery contain a curated subset, not every one of 236 screenshots.

## Observed performance and reproduction

Shared Linux x86_64, sandboxed Chromium154.0.8037.57, Playwright1.62.0, DPR1,
reduced motion, local HTTP, SwiftShader software WebGL. One activation per eight
unavailable-WebGL width/theme variants: original79bd median106.75ms
(range91.15–134.00); b43 median98.81ms (88.74–112.82); final composition
median104.16ms (95.17–116.73). Polling/module initialization and shared-host
variation are included. These establish no speedup or sustained frame-rate claim.
Final repair adds 343 decoded bytes to atlas.mjs and 195 to spatial-shell versus
eb79, no new dependency or pre-activation Three.js request. Initial179-byte
phase1 repair is superseded. Large graphs/hardware GPU/FPS remain unqualified.

```sh
python -B tools/atlas_webgl_fallback_browser_check.py --site docs/site --output browser-evidence/atlas-fallback
python -B tools/atlas_spatial_browser_check.py --site docs/site --output browser-evidence/atlas-spatial
python -B tools/atlas_refusal_browser_check.py --site docs/site --output browser-evidence/atlas-refusal
```

Use a coherent generated site and fresh output directories. Public replay uses
`--base-url` only after verifying actual deployed assets and observation pins.

## Preservation and limits

Before replacing GitHub material, original79bd four files were independently
raw/member/CRC verified privately00:32:14UTC; exact newbaseeb79 four files at
01:07:47UTC; all27 original public Draft7dd paths plus raw commit at01:17:05UTC
on29October. [Original source](original-source-custody.json),
[new base](original-base-eb79-custody.json), [old Draft](original-draft-7dd-custody.json).
Eleven archives remained retained at report freeze; no Drive deletion or sharing
change. Final delivery custody and registration are separate postdeployment work.

The historical arithmetic companion and adverse package review remain unchanged.
The verified declaration is UniversalLaw.Side24.pairing_sum_six, the closed Nat
statement pairingSum6=76; it proves no coefficient/Kac–Rice/parent theorem.
Missing mappings stay explicit. Rendering, CI and speech establish no scientific
acceptance or new alignment. Agents are actual OpenAI/Codex with disclosed source
exposure; organizational-independence credit0, scientific effect NONE.

The next highest-value qualification is a physical touch and spoken assistive-
technology user journey, including other screen readers and remaining manual
contrast states. Larger changes need their own measured reader problem.
