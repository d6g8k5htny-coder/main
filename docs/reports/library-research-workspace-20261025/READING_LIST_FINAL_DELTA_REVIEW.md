# Library research desk — final targeted delta review

Reviewed frozen composed head `ea3db7e36b0eaa9c82f6541120f7dd8bd7eae61f` in `/workspace/universal-law/library-research-desk-20261025`. Scope: M1/M2 fixes in `232af26b` relative to the earlier reviewed `cc94d56c`, and the shared-parser delta from `c034` to this head recorded in `shared-parser-review.patch`. This supplements `engineering-review.md`; it is not a repeated full audit of temporal comparison.

Reviewer: separate nonauthor OpenAI Codex agent `/root/desk_engineering_review`, GPT-6 family; no independently verified finer runtime variant. Authors: OpenAI/GPT-6 root implementer and separate helpers as disclosed in the task handoff. Same-family engineering review is permitted under the 25 October owner decision. Organizational-independence credit: **0**. Scientific effect: **NONE**. No mathematical or formal-verification verdict.

## Verdict

**PASS for the declared delta. Spec compliance and code quality are acceptable. No new Critical, Important or Minor findings. Both earlier reading-desk findings are closed.**

| Change | Assessment |
|---|---|
| M1: same-fragment current-search focus | Resolved. The click handler reveals the disclosures immediately, then uses requestAnimationFrame to focus after native fragment navigation. The callback checks both the current hash and whether focus still belongs to the link or native fragment target. It therefore fixes repeated same-hash activation while preserving native navigation and avoiding overwriting focus moved to another control. Existing unavailable-summary/readiness behavior remains. The browser journey now explicitly repeats keyboard activation after reload and asserts input focus. |
| M2: valid punctuation source paths | Resolved. `sourceLinkKey` compares decoded segments in a JSON tuple, allowing equivalent percent-encoded spellings while retaining segment boundaries. Current lookup and catalog equality both use this key; output links still come from strict validated identities. URL origin, credentials, search and fragment refusal remain. Encoded slash within a segment is distinct from an actual slash separator; literal percent sequences retain the intended single decoding. Unit controls cover equivalent punctuation spellings and slash distinction, and the added catalog DOM fixture exercises visible Save and successful save for a legal punctuation filename. |
| Shared strict JSON parser and reading-list import | Acceptable. I mechanically compared the old comparison parser function with the extracted `reader-json.mjs` function: the function text is byte-identical. Its lexical maximum remains 32 MiB and its fail helper is equivalent. The comparison module imports and re-exports the same API. Reading-list parsing still checks UTF-8 size against 128 KiB first, then checks raw duplicate keys before strict schema/identity validation and atomic merge. Duplicate escaped keys are handled by the existing decoded-key scanner, and rejected duplicate storage data follows the existing preserved/unreadable-storage path. No new network or DOM dependency enters the pure store. |

The extraction's release-key changes consistently connect the new parser, comparison model/controller and reading-list store/controller. No current-model/core/source-loader or scientific-data edit is part of this declared delta. The new held-Web-Lock fixture deliberately queues both saves before release, strengthening the earlier sequential-click concurrency evidence without changing product storage behavior.

## Evidence and boundaries

Read the complete declared diffs, final relevant code, new controls and `/workspace/research/library-temporal-compare-20261025/engineering-review-fixes.md`. The latter separately clears the four temporal-comparison findings at `99650fe13c566b2a7559dcdfd8fbc0d0d3c87d75`; I did not re-review those unrelated fixes or represent that review as my own.

Inspected fresh composed author-run reports:

- `composed-browser-desk/report.json`: exact head `ea3db7e36b0eaa9c82f6541120f7dd8bd7eae61f`, `passed: true`, 13 case records, no page errors. Its updated cases include same-hash focus, legal-punctuation DOM attachment, duplicate-key import refusal, and genuinely queued two-tab saves under a held lock.
- `composed-browser-compare/report.json`: exact same head in `source.git_head`, 12 completed check descriptions, no page errors, sandboxed Chrome 156.0.8078.47. This report uses completed check descriptions rather than a `passed` field; no fabricated combined test count is inferred.

The supplied 42 passing model tests and retained RED→GREEN logs are author-side evidence; I read the changed tests but did not rerun broad suites. The only reviewer execution in this delta review was read-only inspection and the exact-function extraction comparison. Original full-review browser/platform/accessibility/performance limits continue to apply. No separate new Safari/Firefox, physical-device, spoken-AT, print, live-deployment or remote-acquisition run. No source changes, remote posts or subagents were used in this review.
