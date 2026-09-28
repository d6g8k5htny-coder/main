# Required formal checks — verified deployment

**Record:** OP-FORMAL-ENFORCEMENT-20260927-v1.0  
**Owner:** Dylan Roy  
**Execution and custody:** OpenAI / ChatGPT  
**Scientific effect:** NONE

## Outcome

The interrupted Math- integration was completed using the ordinary protected merge path. Math- PR96 merged at `22e79e84917e6bf5bec606c21ce713159d12dddf`; its actual post-merge push run `36359933550` succeeded in all three jobs. Main PR188 was already merged at `3592abbd5df433cb1816b69204e740a7cf00ca07`; its post-merge push run `36358139870` also succeeded. Both raw execution artifacts were downloaded and their receipt/source bindings independently checked in this session.

This closes the previously identified **code-level gap between optional formal execution and existing required status checks**. It does not claim that GitHub administration settings were changed or that all mathematical review obligations are closed.

## What is enforced

| Repository | Existing mandatory context | Dependencies required to succeed |
|---|---|---|
| main | `verify` | Landing checks and the main core-Lean package |
| Math- | `math-downstream-gates` | Existing downstream replay and the Mathlib formal package |

The mandatory aggregate executes even when a dependency fails. It rejects missing, skipped, neutral, cancelled and failed results. Existing checks and formal evidence must refer to the same tested Git commit. Formal outputs additionally bind repository, workflow run, run attempt and SHA-256 of the original receipt. The receipt producer checks the current manifest and every recorded log digest. The workflows use a same-revision reusable formal workflow and avoid path-filter gaps on the parent required workflow. Historical receipts are not rewritten.

These are controls on the normal protected workflow. An authorized workflow author or administrator remains a trust root; a matching hash or a green job does not establish independent statement alignment or scientific acceptance.

## Math- final review and integration

Final reviewed head: `ac9cf41252feca47d29ad3c4b5215082702db28a`  
Current integration base: `8c00f1952d8ce1b82ad469e4ffa8cffbfc0aa264`  
PR-tested merge: `33f6e87633537dff1ad8866f60d5ccca8129c4d4`  
Landed commit: `22e79e84917e6bf5bec606c21ce713159d12dddf`

The exact six-file enforcement diff was reread. Its source helper and tests remain identical to the reviewed main188 implementation. Existing mathematical candidate files incorporated by concurrent base merges were compared for identity, not newly accepted as mathematics. The formal source tree and manifest remained unchanged.

Grok 4.7, through existing Cursor session `bc-36791e0e-af26-4bab-b190-4615b345834b`, returned an actual nonauthor engineering continuation ACCEPT for the final head in main180 comment5860939592. Native GitHub review submissions were empty and were not represented as APPROVE. No organizational-independence or mathematical-alignment credit is asserted by that engineering review.

The normal expected-head merge succeeded without bypass. The landed run then separately exercised the merged tree, not just its earlier pull-request head.

## Observed evidence

Math- landed run36359933550 passed downstream-replay, formal/formal-evidence and math-downstream-gates. The downloaded receipts bind the landed commit and original receipt SHA-256 `48871e21c518a7393ee6aad3b72621bc28d2e9f2901d64f5df79be1470fbba24`.

- Thirteen registered formal targets and nine pinned dependencies match the original source manifest. All ten formal log hashes match. Five false/extra-axiom controls have the expected rejection outcomes.
- The existing downstream report passes 67 gate tests and 24 semantic mutants. Ten actual regression logs cover 177 additional tests in each of normal and optimized Python. Thirty bridge tests and 38 formal-gate tests also run in both modes under the unchanged workflow commands.
- The push transition checks base8c00f195 to landed22e79e84. It reports check_passed=true, unchanged input graphs, no changed or impacted nodes, and promotion_permission=false. Existing HOLD proposals remain present.

Main landed run36358139870 has 31 registered arithmetic targets, empty target axiom sets, ten rejected controls, and 15 verified log hashes. Its original receipt SHA-256 is `a2dae00841d1fb2deb911a532a5b05bb3f291f788ce8150db9d0418662bf060f`. This is the main arithmetic package, not the whole SIDE24 or persistence theorem.

Both formal packages use their pinned Lean environments. Execution occurred on GitHub-hosted runners. This session's local checks validate downloaded bytes, receipts and transition reports; they do not constitute a new local Lean run or a software installation on Dylan's laptop.

## Actual failure probes

Main PR190/run36357843019 and Math- PR97/run36358390847 deliberately failed a formal dependency while leaving the ordinary checks successful. The respective required aggregates EXECUTED and FAILED, rather than skipping or consuming stale success output. Both probes remain closed and unmerged. No attempt was made to merge the negative probes. These tests establish dependency-failure propagation, not rejection of a new mathematical proposition.

## Coordination and remaining administrative boundary

The temporary Math- default-merge coordination hold is released after the successful landed replay. Current and future agents should read `AGENTS.md` and `docs/FORMAL_REQUIRED_CHECKS.md` in the relevant repository and follow the existing main95 discussion and exact target PR. The main95 thread was administratively closed during issue-list cleanup; that closure did not discharge its outstanding scientific or administrative obligations. No new status database or perpetual task loop was created.

The inspected rulesets remain active and strict. Math-24045351 has no bypass actors. Main23798639 retains the pre-existing RepositoryRole5 pull-request bypass. Both require zero native approving reviews. No settings write was performed: the available connection exposes no ruleset administration mutation. Further hardening requires an administrator-capable connection and genuine eligible nonauthor review identities; multiple models using one GitHub login are not multiple native approvals. User authorization does not itself supply those technical credentials.

Independent informal/formal alignment and all existing scientific acceptance obligations remain separate. Code-level enforcement completion does not promote any mathematical claim.

## Original artifact identities

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Main188_landed_formal_36358139870.zip | 34,947 | 4fc0ac1e58a6340802620deab0cc20587b278b0c4e7aa47958a5da3440216aa7 |
| Math96_landed_formal_36359933550.zip | 7,396 | bc0342134ba8c05d7d45d827961a721b468b8d7bc315bf4a138809b9262e0f4a |
| Math96_landed_downstream_36359933550.zip | 1,737,078 | 43bd4650143bb6a9b59871335c04c1aa6cb324391cd89a5e6509675eb9d63db2 |

The outer delivery SHA256SUMS file covers each packaged file. Original artifact ZIPs are unchanged. Readable receipt copies are supplied separately. The package preserves selected deployment evidence, not a complete Git history or full library-dependency download.

## GitHub evidence locations

- Main implementation: https://github.com/d6g8k5htny-coder/main/pull/188
- Math implementation and final receipt: https://github.com/d6g8k5htny-coder/Math-/pull/96#issuecomment-5860985432
- Final current-head reviewer result: https://github.com/d6g8k5htny-coder/main/pull/180#issuecomment-5860939592
- Math landed run: https://github.com/d6g8k5htny-coder/Math-/actions/runs/36359933550
- Main landed run: https://github.com/d6g8k5htny-coder/main/actions/runs/36358139870
- Main negative probe: https://github.com/d6g8k5htny-coder/main/pull/190
- Math negative probe: https://github.com/d6g8k5htny-coder/Math-/pull/97
- Coordination archive: https://github.com/d6g8k5htny-coder/main/issues/95
- Main ruleset: https://github.com/d6g8k5htny-coder/main/rules/23798639
- Math ruleset: https://github.com/d6g8k5htny-coder/Math-/rules/24045351
