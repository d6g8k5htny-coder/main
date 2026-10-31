# Atlas fallback recovery and contrast qualification

When optional 3D failed, the restored Graph could cover one claim's metadata
with the next card. A real mouse click intended for the congruence-erratum
claim then selected the section9 replacement. This release restores the graph's
layout after revealing it, through the existing callback that writes no history.
It repairs reading and selection without changing a mathematical record.

The original cards were 205.734375px high but only 164px apart: three pairs
overlapped by 41.734375px. The repaired cards have about 24px of space between
them. Narrow 2D graphs still require panning; List remains a useful alternative.

| Same restored view | Before | After |
| --- | --- | --- |
|390px light|[Covered metadata](before-390-light.png)|[Readable metadata](after-390-light.png)|
|1200px dark|[Overlapping cards](before-1200-dark.png)|[Separated cards](after-1200-dark.png)|
|320px forced dark|[Covered metadata](before-320-forced-dark.png)|[Readable metadata](after-320-forced-dark.png)|

These screenshots were inspected. The before/after views use the same admitted
observation and selected node. They are local rendered evidence, not a human
usability study, physical phone or deployed-source replay.

## Source and dataset

The qualified local product is `b43b74388161f624ce7c684542232e97d90bb0b1`,
tree `f42d1100d4f129025aff01be727239e7af0379db`, based on Main
`79bdde89a3daf7eaf5d4481961eb87433dd002d5`. The PR supplies the separate native
publication, integration and deployed-source identities. They are not inferred
from this local commit or a green browser result.

The frozen public observation is Main79bd / Math
`8151f592918a845bf19eb8eafaf8444540a32734`: 49 indexed nodes, 55 directed edges
and 108 acquired excerpts. Payload 1,923,440 bytes, SHA256
`8a3595667134e0b196ad957277a572d34be6f39a3eebf0741faed55830f2e989`.
The reproduced failure uses a seven-card neighborhood; the filtered context-loss
case restores five matching cards. This does not qualify every larger graph.

[Source and execution binding](SOURCE.json) distinguishes product bytes from
observation pins. Some original execution reports retain checkout-base `9a4b90e8`
while serving the uncommitted candidate fixture. All 74 served production assets
were subsequently matched byte-for-byte to b43b7438; those raw checkout fields
have not been relabelled as executions of a later Git commit.

## Verification

| Journey or check | Original | Repaired product |
| --- | --- | --- |
|Same focused browser operator|10 failures, 2 existing passes|12/12 pass|
|Restored card geometry and real metadata mouse pick|Overlap and wrong claim|No overlaps; intended exact claim|
|Unavailable WebGL / blocked renderer module|Wrong geometry|Useful Graph fallback|
|Actual successful canvas, filter change, then context loss|Wrong geometry|Current filtered Graph restored|
|Failure Back/Forward, source pins and history length|Geometry fails on Forward|Preserved; no extra failure entry|
|List keyboard fallback / source refusal|Pass|Pass|
|Existing spatial operator, including actual canvas picks|Unchanged assertions|16/16 pass|
|Existing refusal/static operator|Unchanged assertions|10/10 pass|
|Focused controller/relationship/label Node checks|132 pass|132 pass|
|Prescribed and retained release groups|Separate earlier records|13 groups pass at b43b7438|
|Two complete offline generations|Fixed admitted input|1,698 files / 56,453,760 bytes identical|

[Original result](fallback-before.json), [repaired result](fallback-after.json),
[existing spatial](spatial.json), [refusal](refusal.json),
[prescribed commands](prescribed.json), [generation](generation.json).
The new focused browser operator is a local/public release check; no workflow or
required hosted check was changed. Native CI and deployment qualify separately.

An initial repaired run passed 11/12: the refusal test waited for the generic
word “unavailable” while the stable status explicitly said “refused”. Only that
case's expectation was corrected to require refusal; its original-pin and
no-substitution assertions remain. [That first result](trial-first-after.json)
is preserved. The final identical driver was then run on both original and
repaired bytes; no geometry or picking assertion was removed.

## Contrast findings

The independent diagnostic invocation measured the actual rendered subtitle
foreground/background and verified that the decorative pseudo-element ends
13px before its first glyph. Current Atlas subtitle contrast is 10.400134:1
dark and 6.278236:1 light. The specific pseudo-background uncertainty can be
resolved for those measured states; the original axe `pseudoContent` incomplete
remains in its raw record.

Normal graph text measures 15.762223:1 dark / 15.252424:1 light; List metadata
8.586809:1 / 5.934460:1. Across the diagnostic's eight panned baseline variants,
6,128 nonspace character-center samples hit their owning card. Four long-title
axe incompletes persisted despite those measurements; their exact engine cause
was not isolated. They are not rewritten as an axe PASS or full graph-wide
conformance. The real failure-path overlap was repaired separately.

[Bounded contrast evidence](contrast.json) records source/driver hashes and
limits. Axe4.10.3 was developer-only tooling. No dependency was added to the site.

## Actual screen-reader evidence

Orca48.1 / Chromium154 read the selected scope, source revision/blob, exact
arithmetic declaration, hypothesis explanation and dependency selection with an
explicit missing formal mapping. These are actual synthesizer submissions and
nonzero recorded PCM, with no human audition.

The separate Orca46.1 / Chromium141 diagnostic initially left declaration reading
unqualified: a full-page capture between native caret acquisition and Return
coincided with accessibility-tree invalidation. That is a protocol confound,
not an established product defect. Its failed attempts remain preserved.

A separately attributed parent control on the repaired product omitted that
intervening capture. Native source and formal disclosures both expanded, and
Orca46 spoke the source revision/path, `pairing_sum_six`, the arithmetic statement
and its explicit coefficient/parent-theorem limitation. This new result does not
retroactively turn the diagnostic's earlier failure into success. Complete blob,
hypothesis and checked-execution/alignment reading on Orca46 remain unqualified.

[Orca48 records](speech/orca48.json), [new Orca46 control](speech/orca46.json),
[speech qualification and provenance](speech/qualification.json).
Neither version establishes pronunciation/comprehension, NVDA/JAWS/VoiceOver,
physical touch/GPU or full WCAG conformance.

## Observed performance and reproduction

Environment: shared Linux x86_64, Chromium154.0.8037.57, Playwright1.62.0,
sandbox enabled, DPR1, reduced motion and SwiftShader software WebGL; local HTTP.
Across one observation for each of eight unavailable-WebGL variants, median
activation-to-fallback observation was 106.75ms before / 98.81ms after; ranges
91.15–134.00ms / 88.74–112.82ms. These include polling/module initialization,
were collected on a shared host and establish no speedup or loading benchmark.
The fix adds 179 decoded bytes to the spatial shell and no new dependency or
pre-activation Three.js request. Larger graphs and sustained hardware frame
rates remain unqualified.

With the existing browser requirements and a coherent generated site:

```sh
python -B tools/atlas_webgl_fallback_browser_check.py --site docs/site --output browser-evidence/atlas-fallback
python -B tools/atlas_spatial_browser_check.py --site docs/site --output browser-evidence/atlas-spatial
python -B tools/atlas_refusal_browser_check.py --site docs/site --output browser-evidence/atlas-refusal
```

Use fresh output directories. Public replay additionally uses the focused
operator's `--base-url`, after verifying that the public assets and source
observation match the intended deployment. Historical observations retain their
original hashes; current live data may differ.

## Preservation and remaining work

Before any GitHub source replacement, the four original Main79bd Git files,
paths, blob hashes/modes, revision and raw commit were privately archived and
independently downloaded/member/whole-byte/CRC verified at 00:32:14 UTC on
29 October. Archive 80,632 bytes, SHA256
`f1ced4e7fcbb21f90e85a91176d7dcd57314ca24789c652301e29d653a50f24c`.
[Neutral custody receipt](original-source-custody.json). All earlier archives
remain retained. No Drive item was deleted or sharing changed.

The existing historical arithmetic companion, original adverse package review,
source pins and missing coefficient/parent mappings are unchanged. Green CI,
rendering or speech establishes no mathematical alignment or scientific acceptance.
Actual author and diagnostic agents are OpenAI/Codex with stated source exposure;
organizational-independence credit 0, scientific effect NONE. A separate current
engineering review and integration record belong to the PR.

The next useful qualification is physical touch and a spoken assistive-technology
user journey, including other screen readers and remaining manual contrast
states. Further cosmetic changes have lower value than that evidence. Separate
reader-discovery/status-fetch/camera work remains owned by its other contributor.
