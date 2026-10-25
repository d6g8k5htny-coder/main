# Review corrections: current tested successor

Implementation/test revision: `2644777d89886c042c414d252e2c934761fb2659`; product bodies are ce81d404, with2644777d changing only the explicit-refusal test assertion. The source/evidence below supersedes f679/fc2 for the repaired behavior; original13/39-case evidence and original report are retained unchanged in the parent directory. Required hosted checks and reviews still need the final actual head/tested merge.

Author disposition of UI613/6402699952 and automated comments4372262847/4372262853/4372262857:

- Default List→3D: fixed in8ef7468d, retained and exercised in current suite.
- Failed3D Back/Forward: RED onfc2, GREEN after7e; failure cleanup no longer calls the history-writing shell callback. MissingWebGL and blockedmodule cases both traverse back and forward without adding history. Failed3D intent remains in its existing URL for retry.
- Refused formal evidence: RED onfc2, panel removed after7e. The stricter automatedP1 exposed a remaining admittedcount9 in status/ARIA; that RED is retained, and ce81 clears count/status/nodes/relationships as well as formal links on refusal.
- Node classes/source wording: spatial buttons preserve literal kind/classification and are compared against canonical labels. The evidence row now explicitly says graph-source revision; formal artifact remains unmapped.

All16 actual browser cases pass on2644777d, Chromium154/SwiftShader, **sandbox enabled**.125 focused Node controls and canonical asset release/check pass. The final full49/55 CPU draw median is 4.75ms, keyboard-to-observed-frame median 19.50ms; raw12 samples, decoded resource sizes and viewport load observations are in browser/spatial-browser.json. These have the same local/concurrent-load/software-GPU limits as the original performance report. No spokenAT/physicalSafari/mobileGPU or scientific acceptance is inferred.

The original13-case harness used Playwright’s default disabled browser sandbox; that does not relabel the original existing39-case suite, which ran sandboxed. Current16-case harness explicitly enables it. An intermediate run correctly refused the source but failed an old assertion expecting the word“unavailable”; the assertion now requires explicit“refused” plus no stale count. Product behavior was not weakened.

[Current screenshots](browser/after-1200.png) · [Narrow](browser/after-390.png) · [Full49-node graph](browser/full-dataset.png)

Reproduce one targeted regression with `python tools/atlas_spatial_browser_check.py --only admitted-to-refused-formal-history --output /tmp/refusal-check` or run the complete tool without --only. Author reports and tests are not the pending nonauthor verdict.

## UI-requested traversal extension

The same refusal regression now also checks Back→refused→Forward→readmitted, then Back→refused→Graph/List switch. The real sandboxed browser control passed on the unchanged ce81/55 product source; `refusal-roundtrip.json` records that execution from55 with the subsequently committed test-driver extension. The only successor implementation delta is five test lines plus these evidence files; no product code changed.
