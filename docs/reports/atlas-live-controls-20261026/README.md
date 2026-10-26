# Atlas live controls: bounded accessibility repair

The equation search now preserves surviving native options and speaks a final result count after typing pauses. An enabled replacement View control retains the URL/viewport choice without reusing the initially disabled option objects. The visible count still updates immediately. Selecting an excerpt keeps its exact source revision and lines; filtering it out selects the placeholder. Refused bookmarks announce refusal and retain their source pins.

Implementation: `bb94e4e6345adb576f7775f61768af18888a2768`, based on Main `a15d2fb682fdb397584787143fffcf7290753ecf`. Evidence-only commits do not change the production hashes in [source.json](source.json). Deployment is a separate subsequent observation, not claimed by this local report.

The comparison freezes the **same** public Main `b7e012c23ed71b16a3a8245f0f0c3f24b8aa8065` / Math- `180d6406b0a51b268a152241a70bde6a1b39a142` observation: **49 nodes, 55 relationships, 108 excerpts**, payload **1,732,202 bytes**, SHA-256 `0c3a4a9b57f0d7214db6da2d2e6aceaeb0c643f1ba7e656655bb43b36925b9b2`. The local shells use their own production modules with the same admitted CURRENT-SITE block/payload/meta. These fixture pins are not the implementation commit. A separate fresh acquisition at fixed Maina15/Math180d and standalone generation into scratch also succeeded; it is not substituted into the comparison.

## Before and after

| Same journey | Baseline | Candidate |
| --- | --- | --- |
| Type `sigma`, 45ms between keys | 549 option additions/removals; 5 visible-count rewrites | 85 removals; 2 count rewrites; all surviving option objects retained |
| Matched Orca48 speech: final 23 results | Missed in 2/2 matched runs | Spoken in 2/2 matched runs; repeated on final implementation |
| Zero results / clear query | Spoken | Spoken once, concise, after quiet |
| View Graph→List with pointer away from popup | Spoken/selectable | Spoken/selectable; **no claim of a demonstrated new-version improvement** |
| Desktop/phone × light/dark focused regressions | 0/4: option churn/identity and missing settled region | 4/4, including source/selection, share/reload, history and refusal |
| Median local cold readiness, four observations | 403ms | 417ms |
| Decoded resource bytes before 3D, per observation | 1,974,286 | 1,976,732 (+2,446B) |

Measurements use shared Linux x86_64, Chromium154.0.8037.57, Playwright1.62, sandbox enabled, DPR1, software SwiftShader and reduced motion. Four timings per side are observations, not a speedup/regression significance claim or scale test. The explicit quiet period is 500ms. Playwright's polling observed the completed announcement after roughly 0.8s in the matched probe; that is an upper observation including polling, not precise speech latency. Real Orca output is recorded separately. Current spatial regression activation was 202–346ms across its two actual canvas cases; rendered/input behavior is otherwise unchanged.

Phone light: [before](baseline/equations-390-light.png), [after](candidate/equations-390-light.png). Desktop light: [before](baseline/equations-1200-light.png), [after](candidate/equations-1200-light.png). Both dark counterparts are in the same folders. Native input/select/buttons remain familiar, readable and within the narrow viewport. The count and explicit no-logical-join scope remain visible. This release intentionally does not add another visible settings panel. Screenshots were inspected; pixel layout is substantially unchanged. Functional checks alone do not establish usability.

## Verification and speech boundaries

The same-driver matched JSON is [baseline](baseline/controls-browser.json) / [candidate](candidate/matched-measurement.json). The final candidate driver adds an explicit refusal-announcement assertion; [candidate/controls-browser.json](candidate/controls-browser.json) passes all four layouts. The initial baseline failed the intended identity/churn/missing-region assertions. An additional refusal test first failed with a current-result announcement, then passed after the refusal guard. Two clock/lifecycle tests exercise unchanged-count requeue, rapid abandoned queries, hidden documents and page-cache transitions. The real controller/model/shell tests run the production helper, not a copied helper.

All **132 Node controls/current/reader checks**, **14 prescribed local groups**, **39 official current-reader cases**, **16 spatial cases** (including **two successful CANVAS picks**, dependency navigation, keyboard/history, context loss and initialization failure), **10 refusal/fallback cases**, and **4 formal inspector cases** passed. [Prescribed results](prescribed-results.json), [spatial](spatial.json), [refusal](refusal.json), [formal](formal.json). The official browser adapter replaces only unavailable Chrome-channel selection with `/usr/bin/chromium`; sandbox/CSP/assertions remain enabled. This is local packaged-Chromium evidence, distinct from hosted CI's packaged Chrome. Asset release/check, fresh fixed-pin data acquisition and standalone site generation also passed. Raw command output and operator adapters are retained privately.

Actual speech uses **Orca48.1, AT-SPI2 2.56.2, Chromium154, speech-dispatcher0.12.0, eSpeakNG1.52.0**, an isolated Xvfb/DBus session and real keyboard events. [Speech summary](speech/summary.json), [scenario](speech-scenario.json), version files and per-run transcript/source-response hashes preserve provenance. Strings are what Orca submitted to the synthesizer. PulseAudio recorded a null sink; **no human audition** was performed. This is implementation-side replay, not blind, organizationally independent or full WCAG qualification. The final refused-bookmark speech run speaks refusal and no current-result count.

Retained setup/iteration evidence: an initial run correctly refused a mismatched HTML/payload digest and never enabled controls; the fixture was corrected. An intermediate candidate's changing `aria-describedby` produced duplicate long/short counts; that description was removed before final replay. The early View scenario did not park the pointer and started at the last option; matched runs use the corrected parked-pointer Graph→List sequence. Matched final candidates preserve their own exact response identities; the subsequent refusal-only controller change is separately exercised by the final normal/refused runs. Original raw runs, failed assertions and interim screenshots remain in private custody.

**Still open:** Claude54's separate Orca46.1/Chromium141 qualification and the older intermittent View sensitivity finding; broader `.current-state` and archived `#load-status` persistent-reading repairs; NVDA/JAWS/VoiceOver, hardware mobile/touch and full accessibility conformance. No historical renderer or snapshot page was rewritten to obtain these results. Hidden/page-cache behavior has deterministic lifecycle coverage, not a human device qualification. Dense 3D labels still cull at the current dataset's overlaps; no larger-graph capacity claim is made.

## Evidence and preservation

No new production dependency or license obligation. Existing pinned Three.js and its license remain unchanged; the native helper follows the repository's existing code licensing. Cursor's `aadd04e4841e96b472f92abbd84dda6e9c1270af` assist and Claude54's observed failure mechanisms informed the assessment. This is a new bounded implementation; the assist's historical renderer changes and in-place ID mutation were not adopted.

Original complete site Maina15 (254 files, 10,210,906B) was privately archived and independently raw-downloaded, canonical/member/whole-byte/CRC verified **before source replacement**: [custody receipt](original-site-custody.json), archive SHA-256 `5f3a687aadb777af31f076a779e37eae834ceafbc13969d2daaa99563718a100`. Original controller test bytes were separately privately archived/readback before its harness update, ZIP SHA-256 `d8a77f1d129f2c43b6708bf0fae778203fcb4c372f5496100ffa5e00fa4a44cf`. No Drive deletion/overwrite/sharing change, peer branch mutation or historical evidence regeneration.

The existing limited SIDE24 source-document/arithmetic-companion connection and original checked receipt remain unchanged. It does **not** prove the coefficient or parent theorem. Package alignment remains REJECTED(AMEND), manifest PENDING_INDEPENDENT_REVIEW, accepted package alignment and full coefficient/parent coverage zero. Acquisition, review, kernel execution, alignment and scientific acceptance remain separate; absent mappings stay explicit. No new proof coverage or scientific disposition is asserted.

Next increment: exact-source nonauthor speech qualification on the remaining platform, then repair persistent source-observation/archive status reading without changing historical evidence identities.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
