# Reader exports and navigation — separate engineering review

Candidate: `9e08f01090e6120997c9fb231ee39231557df47e`, reviewed against `99adffc8902b1fbfb7ab487020f9c0267f28f88d`. Worktree HEAD and clean status confirmed before review. The intervening Main618 composition adds two retrospective documents; the product scope remains the three declared reader features plus the existing workflow's test-command extension.

Reviewer: nonauthor OpenAI Codex agent `/root/desk_engineering_review`, GPT-6 family; no independently verified finer variant. Authors are the OpenAI root implementer and its separate PNG/citation helpers disclosed in the handoff. I previously reviewed the underlying Library store/controller and composition, but did not implement these changes. Same-family separate engineering review is permitted by the 25 October owner decision; organizational-independence credit **0**. Scientific effect **NONE**; no mathematical correctness, hypothesis applicability, statement alignment, scientific acceptance or formal-verification verdict.

## Verdict

**Substantially spec-compliant; code quality acceptable subject to one Minor correction. No Critical or Important findings.** The PNG and citation features have no identified actionable defects in the reviewed implementation. The navigation enhancement does not completely meet its explicit ambiguous-markup refusal contract.

## Minor M1 — duplicate target IDs are accepted rather than refusing the affected navigation

`docs/site/research-reading.mjs:22` through the target checks at lines 28 and 35–36.

Existing source disclosures are accepted when their own ID matches the expected string, but the code never checks that this ID uniquely resolves to that disclosure. Global questions/map/archive targets and section IDs likewise use the first `getElementById` result without uniqueness checks. If another element carries the same source ID, “Exact sources” is still installed and native navigation/reader-shell can resolve the unrelated element. This violates the spec's requirement that missing or ambiguous target markup omit the enhancement. The committed page presently has no demonstrated duplicate target, so this is a robustness/refusal-contract gap, not a claim that ordinary current reading navigation is broken.

A tiny reviewer diagnostic in sandboxed Chrome reproduced the source-target case: load Research, await the seven controls, remove generated controls, prepend an unrelated div with ID `reading-cut-20261010-details`, then call `installReadingNavigation(document)`. Result: `{installed: 7, affectedNavigation: 1, targetIsUnrelated: true}`. The affected cut should have been omitted while the other six remained useful.

Correction: resolve each target uniquely and require it to be the expected section/disclosure; reject ambiguous shared questions/map/archive targets before installing any dependent controls. Keep absent runtime disclosure IDs assignable only when the new ID is unused. Add a focused duplicate-ID refusal control alongside the existing missing-disclosure control. Do not modify historical source bodies to repair this runtime issue.

## Positive assessment by scope

- **PNG:** Explore calls the existing geometry/resource validators and captioned serializers; it does not redraw the mathematical model in the exporter. The controller uses a single timestamp and complete SVG/metadata/citation/dimension fingerprint for final recapture. Explicit control/theme revisions cancel pending work, and stale completion cannot overwrite newer state. Rasterization uses CSP-permitted data images, 2× fixed dimensions, bounded timeout and size, and cleans up image/canvas resources. Download Blob URLs are revoked after a bounded delay. PNG chunk bounds, CRCs, dimensions, terminal structure and preexisting text metadata are checked before adding UTF-8 iTXt records. Error and retry state is separate from the existing readable citation. Trust remains the validated caller SVG plus browser canvas encoding, not an independent PNG proof/image-data validator.
- **Reading navigation:** The seven explicit cut mappings avoid invented scientific associations. Native fragment navigation and the unchanged reader-shell preserve history and disclosure revelation. Missing-disclosure and idempotence checks are useful; duplicate-ID refusal needs the correction above. The maintained board619 routes are confined to the declared newer-work and Join navigation, retaining historical route labels and source-cut prose.
- **Saved source to Cite:** The helper first validates the reading-list entry, uses the existing builder's narrower syntax contract, reconstructs a same-site cite URL and carries only repository/full commit/literal path/optional actual SHA-256. Git blob is not substituted for SHA-256. Unrepresentable paths return an unavailable signal without destroying saved source/export/remove actions. The controller creates inert text/native links and leaves the existing citation authority unchanged.
- **Workflow:** The sole public-shop workflow change appends the two new test files to the existing Node command. Existing test files, triggers, job permissions, action versions and other workflow behavior are unchanged. This review does not approve any broader workflow/security change.

## Evidence inspected and personally checked

Read current root/site AGENTS, brand guide, owner review-family decision, spec/plan, all new product modules, integration/HTML/style differences, the two new Node test files, modified route guards, the three new browser harnesses and verification manifest. Existing source-reference and reader-shell contracts were read directly; relevant Explore redraw/export integration was inspected.

Personally compared all 16 product/test/workflow SHA-256 entries in committed VERIFICATION.json to the frozen checkout: all match. Personally verified all three saved browser report hashes against that manifest: all match. Reports record Research 6 groups, citation 3 groups, and PNG 11 groups with 41 downloads, all passed with empty page-error arrays. These are author-side runs bound by content hashes, not falsely relabeled reviewer reruns at the final report-binding commit. Inspected an actual downloaded curvature PNG visually: full diagram and source/citation caption are present and legible.

Independently byte-compared nine existing files against base: curvature-export, teaching-export, pin-export, palette-export, current, current-model, source-reference, research-desk-store and reader-shell; all unchanged. The Research diff changes only new asset imports and the maintained newer-work aside; dated cut bodies remain untouched. The supplied existing test guard conservation and public-shop evidence therefore remain consistent with the source diff.

The manifest records four Node and ten Python command exits as zero, including the configured Node suite, current controller, reading-list/comparison controls, canonical asset check, preserved route guards and current data/build tests. I inspected the commands and changed tests but did not duplicate those broad runs. My only behavior execution was the small duplicate-ID diagnostic above; it made in-browser fixture changes only and did not edit product source.

## Limits and next disposition

M1 remains open for this exact candidate and should be corrected and subjected to a targeted fix review. Earlier failed runs and their actual source identities remain evidence; no waiver or finding deletion is implied. Current-head automatic review, required hosted checks, protected nonauthor integration and observed publication remain separate stages.

No Safari/Firefox, physical touch, spoken assistive-technology, high-node-scale or independently rerun full-browser qualification. No scientific or authenticated-source-custody conclusion; PNG preserves recorded provenance with verification not performed. No source edits, remote comments, subagents, Ready transition, lock claim or merge performed by this reviewer. Root may relay this report with literal attribution and scope.
