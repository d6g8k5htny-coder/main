# Atlas reader accessibility audit — engineering scope

26 October 2026. Implementation-side observer: OpenAI/Codex, same provider as the implementer, shared account, organizational independence0. This is a scoped WCAG2.2AA engineering audit. It is not code review, integration approval, scientific approval, screen-reader qualification or a full-site conformance statement.

Baseline code: `8fb04ee5f0488eeb4652e604d3c5b8552b931b87`. The frozen baseline and final-layout candidate used the same captured fixture, SHA-256 `a3dd0addb60a2499eab33239aa926785361ae2bc4bf5427ec6e7c4613be206a1`,1,671,822bytes. The candidate is identified by retained per-file hashes; it was uncommitted when audited. Both input trees remained unchanged during observation. The intermediate layout was not relabelled as the final result.

Tools: sandboxed Chromium154.0.8037.57, Playwright1.62.0, pinned axe-core4.10.3 (MPL-2.0). Original npm tarball SHA-256 `0f2b4d7dcdf7d1219df8d1959ad68e565f51d14c3f0d88bb71cd59abeb956292`; published npm SHA-512/SHA-1 matched. No production dependency, sandbox/CSP bypass or security-policy change. Raw axe, DOM/AX captures, UI prose, source-risk notes and source-specific historical findings remain in the private evidence package.

## Findings and observed behavior

| Item | WCAG criterion | Disposition |
|---|---|---|
| A11Y-01: dark3D accessible-list secondary text |1.4.3 Contrast (Minimum)|Confirmed in baseline and original final layout:1.1097:1 at13px normal;4.5:1 required. Original failure is immutable. Separately identified postfix recheck verified9.9395:1 dark/13.2187:1 light across seven actual3D labels each; scoped contrast axe0violations/0incomplete in both. Repaired in sampled source only. |
| A11Y-N01: wrapped footer hit-area overlap |2.5.8 Target Size (Minimum)|Usability issue retained. Three raw axe flags have equivalent same-page130×44CSS controls at320px; Equivalent exception applies. Footer repair deferred. |
| A11Y-N02: generic-label and noncomputed contrast results |1.3.1/4.1.2;1.4.3|Manual-review items retained. Axe incomplete results do not establish failure or conformance. |
| New native guide/provenance |1.3.1,2.1.1,2.4.3,2.4.6,4.1.2|Enter/Space toggled and retained summary focus. Chromium AX expanded state changed false→true→false. Two targeted open-content scans had no violations/incomplete results within their limited scope. |
|320CSS reflow/spacing |1.4.10,1.4.12|No page horizontal overflow in sampled list/reader states under default spacing, WCAG text-spacing override and forced-colors emulation. Actual browser/text zoom and physical-device behavior not established. |
| Graph/list/real3D keyboard |2.1.1,2.1.2,2.4.3|Native View/list selection, inspector/source-link focus and canvas Tab exit observed. Actual WebGL rendering present. Exhaustive camera/pointer/gesture equivalence not established. |
| Status semantics/assistive technology |4.1.2,4.1.3|Persistent DOM status structure observed. Actual speech NOT_TESTED here. Previously reported persistent-reading, select-state and count-announcement AT issues remain open in their separately owned lane. |

Six full-document axe states per frozen input plus two targeted new-disclosure scans were retained. Completing probes does not mean the product passed: the original contrast failure, incomplete results, exception disposition and untested checks are explicit. No parent performance/full-screenshot suite was repeated.

The repaired-input fixture matched the original exactly and all postfix files remained unchanged across observation. Full-document scans were not repeated after the small repair; only actual3D metadata contrast was rechecked in both themes. Other original contrast incomplete cases remain unqualified.

## Criterion coverage

The accompanying sanitized criterion table accounts for all55 current A/AA success criteria. After focused repair recheck:25partial observations,1partial criterion with sampled A11Y-01 repair verified,13not-applicable only to sampled content,7NOT_TESTED and9manual/AT/further-review items. The original observed failure remains preserved at its own source. These are evidence dispositions, not55criterion-wide conformance verdicts. Removed4.1.1 is omitted; reduced-motion emulation does not count AAA2.3.3 as AA coverage.

Explicit NOT_TESTED: orientation1.3.4; actual text resize1.4.4; timing adjustment2.2.1; measured flashing threshold2.3.1; language of parts3.1.2; whole-site navigation consistency3.2.3; whole-site help consistency3.2.6. Other gaps include all non-text contrast, all focus-obscuration states, physical gesture/drag alternatives, complete error processes, other browsers and actual AT speech. No audio/video, personal-data entry, transaction or authentication process occurred in the sampled reader; that absence is not asserted site-wide.

The environment lacked the installed speech/display/AT stack needed for a real screen-reader run. Existing separately attributed evidence was reused without claiming a new replay, repair, ownership transfer or speechPASS. DOM/Chromium AX/axe checks cannot discharge those obligations.

The original private evidence seal covers141members/26,556,777bytes; manifest SHA-256 `e41471736610f1faa9a825c12ee8670c67e7d8cec576609f3325f9d0308b5f66`. The separate postfix site was frozen by its per-file manifest. Only the spatial stylesheet materially differed after release-key/trailing-whitespace normalization; this is not whole-branch review. Focused recheck04:23:44–04:23:48UTC retained its own source identities, raw outputs and disposition. The original finding remains preserved. No claimed candidate commit was fabricated; the source was uncommitted when observed. A later read-only local commit binding found parent HEAD331a11ef03904f6c86842bfa316d47a1d9d4a7ec: three consumed served assets matched exact Git blobs; authored HTML outside its one generated region matched. This is a bounded source comparison, not fresh commit execution or whole-branch review. Parent controls source repair, publication custody, separate review and integration. Observer holds no Mainlock or foreign role.

[Mandate](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OWNER_MANDATE_20261015.md) · [Delegation](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-AUTONOMY-20260923-v2.1.md) · [Review families](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OWNER_DECISION_20261025_REVIEW_FAMILIES.md).

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
