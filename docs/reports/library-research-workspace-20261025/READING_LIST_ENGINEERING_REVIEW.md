# Library research desk — separate engineering review

Reviewed frozen head `cc94d56c1c455fdb0e99fa291f0aa04df7106ce9` against base `6332601a0169890ccebd67f9d34e0aaa2ac67f08`, 25 October 2026.

Reviewer: separate OpenAI Codex agent `/root/desk_engineering_review`, GPT-6 family (the runtime does not expose a more precise model variant). Authors as supplied in the dispatch: OpenAI/GPT-6 root implementer and separate OpenAI store helper. This is same-family, same-provider engineering review permitted by `governance/OWNER_DECISION_20261025_REVIEW_FAMILIES.md`; organizational independence credit is zero. No mathematical correctness, formal verification, scientific statement alignment or acceptance verdict is supplied. Reviewer did not implement this branch or post remotely.

## Verdict

**Spec compliance: substantially compliant, with two minor gaps. Code quality: acceptable with minor corrections recommended. No Critical or Important findings.** The branch preserves scope, source boundaries, static routes and old fragment destinations; keeps current and catalog query contracts separate; admits explicit source/search saves; validates imports before mutation; enforces 50-entry and 128-KiB limits; preserves unreadable storage; and uses Web Locks plus rereads for coordinated mutations. Store code has a useful pure-function boundary and adversarial coverage. DOM output uses inert text and reconstructed links. Git blob and SHA-256 identities remain distinct.

## Findings

### Minor M1 — repeated current-search navigation loses input focus

`docs/site/research-desk.mjs:115` (interaction with line 116 and `docs/site/workspace.html:39`, the focusable `#library-current` target).

When the URL already ends in `#library-current`, activating the hero “Search current sources” link focuses the search input in the click handler, then native fragment navigation focuses the introductory `#library-current` div. Because the hash has not changed, the hashchange listener does not run to restore search focus. This falls short of the design's explicit-navigation input-focus behavior; keyboard readers must navigate again to enter their search. Ordinary first navigation with a different hash works.

Reviewer reproduced on actual sandboxed Chrome 156.0.8078.47 using the packaged executable recorded in the frozen-head browser report: load `workspace.html#library-current`, await enabled `.current-search`, focus `#library-current-link`, press Enter. Twice, `document.activeElement` was `{id:'library-current', className:'library-discovery'}`. A correction should preserve native link/history behavior while ensuring the final focus destination is the input (or unavailable summary) for same-hash activation too. Add a focused assertion for this case.

### Minor M2 — source-button attachment uses a different URL spelling from the existing loaders

`docs/site/research-desk.mjs:88`, `docs/site/research-desk.mjs:99`, `docs/site/research-desk.mjs:151`, and `docs/site/research-desk-store.mjs:134`.

The desk escapes `!'()*` in path segments beyond `encodeURIComponent`; existing `current-model.mjs:12` and `core.mjs:9` source links leave these characters unescaped. Current source attachment looks up an exact href string, and catalog attachment requires exact href equality. Therefore valid sources with such filenames can be displayed by the existing loaders but receive no Save source button. This is not an unsafe-link defect; it is a valid-source compatibility gap. No such paths occur in the current 3,178-source observation examined, so there is no demonstrated presently affected current result.

A tiny reviewer Node diagnostic with admitted-shape path `docs/proof(1).md` produced loader href ending `/docs/proof(1).md` and desk href ending `/docs/proof%281%29.md`, with equality false. Compare canonical validated identities, or consistently canonicalize both sides before attachment; preserve the strict source URL reconstruction on output. Cover at least one legal punctuation path in attachment behavior.

## Evidence and coverage

Read the whole review patch, design and plan, new store/controller/CSS, HTML and navigation differences, modified link tests, new store tests and browser harness. Read root/site agent guidance and owner mandate, delegation and review-family decision. Examined the existing loader URL and rendering contracts. Ran only two tiny diagnostics for the concrete findings, plus a generated-block conservation comparison: all ten existing CURRENT-SITE blocks remained byte-identical across base/head. No loader, Atlas, source-data or historical-record changes appear in the branch diff.

Inspected frozen-head `browser-cc94d56c/report.json`: 13 cases passed, no page errors, source equals reviewed head, sandbox enabled, Chrome 156.0.8078.47. This is author-side evidence, not a reviewer rerun. Its journeys cover basic current/catalog saves, persistence/export/import, invalid import refusal, storage denial/corruption/quota/lock refusal, two-tab saves, snapshot refusal, late-readiness focus, no-JS, narrow/light/dark and primary navigation. Inspected the 320px dark screenshot directly; collection choices and reading-list entry are legible and fit the viewport. Other supplied broad Node/Python/public-shop evidence was reported by the implementer and was not independently rerun.

## Limits

No Safari/Firefox/physical-device or spoken assistive-technology run. No independent print rendering or all-screenshot audit. The two-tab harness performs sequential clicks and confirms propagation, rather than forcing overlapping lock requests; serialization correctness was assessed from code. No separate actual-browser no-Web-Locks diagnostic; the session-only path was inspected. No new load/performance benchmark, deployed-site replay, remote-source acquisition audit, or scientific verdict. No changes made to product files; this review artifact is outside the repository.
