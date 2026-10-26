# Source-list continuity — separate engineering review

Reviewed exact range `336b6f8927f37e2a8d727c6bea845f874c52d8f5` → `9fdfa18296b3386abffe4ce0bb100e3477a5d781` in `/workspace/universal-law/source-list-continuity-20261026`, 26 October 2026. All 22 changed paths reviewed. The checkout, index, HEAD and refs were not modified; report/probe artifacts are outside the repository. No subagents, external posts, Drive operations or broad test campaign were used.

Reviewer: OpenAI Codex `/root/source_continuity_review`, GPT-6 family; separate from the author, same model family, source-exposed and nonblind. Author/test evidence is also attributed to OpenAI/GPT-6. This is permitted separate engineering review under `governance/OWNER_DECISION_20261025_REVIEW_FAMILIES.md`. Organizational-independence credit **0**; scientific effect **NONE**. This report neither provides scientific review nor certifies publication, source authenticity or accessibility.

Read the stored review package, specification, plan, implementation ledger, root/site AGENTS, owner mandate, standing delegation, and 25 October family decision. Applied the `superpowers:requesting-code-review` reviewer template. The implementation is judged by actual reader effects, including reasonable cases beyond the explicit examples.

## Strengths

- The product change is confined to the existing renderer/More closure (`docs/site/current.mjs:35`). Tracking the last rendered query and count is enough for the immutable observation; no framework, dependency, storage, schema or shared API is added.
- On unchanged queries, slicing from the rendered count and appending retains row, acquisition disclosure and save-button objects. Existing research-desk attachment is idempotent and observes added result children; it does not replace retained controls. New query input/history paths still replace and reset.
- The query comparison occurs before slicing (`current.mjs:37`), so a silent value assignment followed by More cannot carry a prior query's prefix or expanded limit into the replacement population.
- Focus ownership is captured before rendering (`current.mjs:45`). The new first source link receives focus only when More owned it, including when rendering hides More. The no-link fallback is present. Other focused controls in retained rows or outside the list survive unchanged-query expansion.
- All 17 other changed product files were independently compared at the exact two commits after replacing only `site-release=` 64-hex values: they are byte-identical under that normalization. The release-key checker passes. The current model, payload, metadata, library enhancer, historical/mathematical records and classifications have no changes in this range.
- The new browser driver uses real admitted payload bytes and native controls. Its independent Python URL/filter oracle checks exact ordered windows and duplicates; direct object references check DOM continuity. The RED failures are behavioral, including lost disclosure/control objects, focus left on More/BODY, and a silent-query reset returning 150 rather than 50 rows. The HTTP refusal case also preserves disabled/unavailable controls.

## Review focus — verdicts

| Requirement | Assessment |
| --- | --- |
| New query after expansion replaces the old prefix and resets the window. | Satisfied by input/popstate replacement paths and browser input/Clear/back/forward cases. |
| Programmatic query changes before More cannot mix two query populations. | Satisfied by query guard; browser case checks exact replacement and old-row disconnection. |
| A final partial batch focuses its first newly revealed named source. | Satisfied: exact source URL at index 200 receives visible focus for 200→224; name rendering is unchanged. |
| A programmatic click while another control owns focus does not steal focus. | Satisfied by ownership guard and unrelated-summary browser case. |
| Retained acquisition details and save controls stay the same DOM objects. | Satisfied by append path and object-identity/open/saved-state assertions across four viewport/theme cases. |

I also examined empty populations, recovery from a silent zero-match query, repeated synthetic clicks after the last batch, raw query-string changes, shared-module consumers, and MutationObserver interaction. A narrow reviewer-run Chromium probe on `cite.html`, `index.html`, and `dependencies.html` passed row retention, first-new-link focus, empty-query recovery and terminal deduplication, with zero page errors. Raw query changes conservatively reset even when normalized filters would match; this preserves the specified reset behavior.

## Issues

### Critical

None found.

### Important

None found.

### Minor

**M1 — Published-origin reports ambiguously label local code identity.** `tools/source_list_continuity_browser_check.py:294`–`300` selects a published-origin mode but always populates `source_sha256` and `checked_commit` from local `ROOT`. With `--origin`, the actual fetched `current.json` is correctly checked and captured in each case, but those top-level code hashes and commit do not identify served HTML/JS. A downstream reader could mistake them for the code tested remotely. This does not invalidate the exact-head local GREEN, nor independently recorded served-byte proofs. Rename/label these fields as local checkout metadata (as the existing desk driver does), and qualify served code separately if a remote-code identity claim is needed. Do not relabel the preserved RED/GREEN artifacts retroactively.

## Evidence checked

- Stored RED `qa-red-programmatic-query/report.json`: unchanged `c1678a5`, 3 passing / 6 behavioral failures, no page errors. Stored initial GREEN: 9/9, no page errors. Both use harness SHA-256 `c4625d96870026b59d8ad5fede25af940c5e4ba43817b4d31b2acf68b8625dce`.
- Exact-head `qa-final/report.json`: 9/9, no page errors, Chromium 154.0.8037.57 with sandbox; 320/390/1440 CSS-pixel widths, light/dark cases as individually recorded. I independently matched all recorded local file hashes and harness hash to the reviewed HEAD. Admitted payload: 962639 bytes, SHA-256 `560d3d4e77369835c832dbb4fd0d12f4988744045b9cbc4c7978beddb6e9d20f`, 3178 indexed sources, 224 `proof` matches.
- Stored Library report: 16/16 with zero page errors. Stored desk continuity report: 49 cases, overall pass, zero page errors; exact reviewed HEAD; timing wrapper exit 0.
- Stored site summary: all 12 required commands exit 0. Related Node log: 154 tests pass, zero failures/skips. These are inspected executor receipts, not reviewer reruns.
- Reviewer ran `git diff --check BASE HEAD` and `python -B tools/site_asset_release.py --check`; both exit 0. Canonical release result `e6501f8126b8db53a77e2d188410a672394558ce8e923290ee85bb494b87188a`.
- Reviewer additional probe is recorded in `reviewer-narrow-browser.json` (3/3 pages). An initial probe mistakenly assumed `research.html` contains a source-search input; it failed setup before testing continuity. Actual markup showed only its observation summary. This failed attempt is recorded separately in `reviewer-initial-probe-setup-failure.txt`; it is not a product failure or passing evidence.
- The earlier full Python baseline was recorded as 1435 tests, 5 failures / 2 errors / 22 skips associated with the Atlas formal-manifest guard. The fresh base includes the separately owned #670 repair. The composed full discovery was still running when inspected; its partial progress is **not** a full-suite pass. Root must retain the terminal result and ordinary required hosted-check evidence before integration.

## Declined to judge

- Mathematical correctness, statement/source alignment, hypotheses, applicability and acceptance: outside this engineering change and this same-family review's qualification; scientific effect NONE.
- Independent source authentication, acquisition truth or historical proof regeneration: unchanged source/admission contracts; local digest/DOM checks cannot establish those predicates.
- Live deployment, offline packaged bytes and remote HTML/JS identity: not supplied as completed exact-head delivery evidence for this review; M1 explains why local metadata cannot substitute.
- Spoken assistive-technology, Safari/Firefox and physical touch-device behavior: not exercised by the available Chromium evidence; no certification claim is made.
- The separately owned Atlas formal-test repair and scientific fixture validity: already in BASE, outside this 22-path delta; pending composed-suite success must be recorded independently.
- Mutation of the admitted snapshot by a future consumer: the current consumers examined read it; the architecture/spec uses a single immutable observation. A mutable-observation API or global renderer redesign is not part of this repair.
- Synchronizing the address bar after a silent programmatic input-value assignment: the existing URL contract is the input event/popstate path. This change guards More's population/window, and introduces no programmatic query-setter API.

## Assessment

**Ready to merge within the engineering boundary: Yes**, with M1 as a nonblocking report-labeling improvement. No Critical or Important defect was found in the reviewed change. The implementation satisfies the five focus conditions with direct behavioral evidence and a minimal architecture. This is an engineering readiness verdict for the exact reviewed delta, not confirmation that the still-running full suite, hosted checks, integration or delivery have completed. Root owns the disposition of M1 and every declined scope, fixes, adoption and integration; this review ends here and creates no future review seat.
