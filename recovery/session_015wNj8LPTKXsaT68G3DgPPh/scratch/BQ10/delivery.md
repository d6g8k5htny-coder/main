## BQ-10 DELIVERED / RELEASE: three Museum review-parent bindings (BQ10-R1-20261007T0010Z-e105)

Dylan Roy — delegated AI evidence maintenance. Actual performer Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). I did not write `museum.json`, `PROOF_INDEX.md`, `LANDING_CLAIMS.json` or any of the reviews. Exposure is disclosed in [6027793004](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6027793004). Credit 0, scientific effect NONE.
- **Authorization:** confirmed by the owner (relayed in pickup [6029891910](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6029891910)) and by the coordinator in [6029899166](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6029899166), with window 02:52–03:50Z.
- **Reads:** native GitHub REST reads and local hashing only, done 02:52–02:54Z. No UI edit, proof reread, test run or write.

### Result: all three bindings authenticate as pointers. Two scope notes and one limit follow; no defect requiring a change.

**Common pins (verified).**
- main @ `cddbb7f6` `docs/site/museum.json`: blob `bd0f209733b14e02939bf2483c5e67bbb5e8f38c`, 59535 B.
- Each of the three `review` descriptors binds Math- `PROOF_INDEX.md` @ `d6628da09384728992dcbe6e921cc28ba85aebb0`. I fetched it there: blob `e4cb1d943eaeb2049d080fcc53a57e042aebc6e8`, 12708 B, SHA-256 `f9c4db77aeab1c61d2a0c02f88a964cbfc6be63d18619acf0e047c884e0c0249`, all equal to the descriptor.
- Each entry's `scope_quote` occurs **verbatim** in that file: d2 at line 18, d4 at line 20, p15 at line 27. Each `review_url` appears inside its quote.
- All three descriptors carry `pointer_only: true` and the notice that the linked review is not byte-frozen. That matches what I found: the GitHub API exposes no edit history, so the bodies below are the **current** bodies (read 02:53Z), not frozen review bytes.

**1. `d2-lifetime-remainder` → main#67/[5841782206](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206).**
- **Target:** an issue comment on main#67 by account `cursor[bot]`, self-declared xAI Grok 4.7, Cursor session `bc-dfbce246…`. Created 2026-09-26T01:04:21Z, edited 01:07:56Z. 5062 B, `56380aff57663fbb…`.
- **Subject bound:** it names LIFETIME_REMAINDER.md blob `247b3ecf80bfbe896948d5d489b2d5842a81c481` (17734 B, `380b7d0a…`). That equals the museum `proof` pin and the blob at `d6628da0`.
- **Slice (note 1):** this is the **R5/R6 delta only**: "R5 and R6 are ACCEPT … I did not replay R1–R4".
  - Theorem R's full acceptance at its O(1) scope composes three comments: this delta, the R1–R4 parent [5841270276](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841270276) (5825 B, `6ffb8f83…`, same blob read), and the #63 parent-interface ACCEPT [5841570965](https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841570965). All three are the xAI/Grok family.
  - PROOF_INDEX line 18 names both main#67 comments. The museum `review_url` carries only the delta. That is a faithful single pointer to the comment that completes the verdict, not a complete review set.
- **Dated context:** line 18's "main #67 remains open" was true at `d6628da0` (2026-09-26T16:46Z). main#67 was later closed `not_planned` on 2026-09-27T23:38Z, with comment 5860882360 saying the closure "changes no scientific status".

**2. `d4-fixed-remote-rn` → main#76/[5841783172](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172).** I reuse my 2026-10-05 identification of this comment and re-authenticated it here.
- **Target:** an issue comment on main#76 by `cursor[bot]` (xAI/Grok). Created 2026-09-26T01:04:29Z, edited 01:12:36Z. 7149 B, `05fa419c…`.
- **Subject bound:** it states it reviewed `frontiers/remote_window_20260924/PROOF.md` at commit `191ea7d5`, 18355 B, SHA-256 `a332bae9…`. Natively, that file is blob `b383bfcc88ec4ad497dff01fb6640e429ba24a84` at both `191ea7d5` and `d6628da0`, equal to the museum `proof` pin.
- **Slice:** seven fixed-ρ/fixed-η interfaces. No full RN or 24-jet closure.
- **Parent issue:** main#76 was closed `completed` at 01:15:55Z after the account reconciliation comment [5841861362](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841861362) (626 B, `e899b7f8…`). That comment is also cited on line 20, but the museum does not link it. It closes the review work package only.

**3. `p15-demand-one-counterexample` → whole issue [main#67](https://github.com/d6g8k5htny-coder/main/issues/67) (note 2: no verdict found at the target).**
- **Binding verified:** line 27 routes through Math- `claims/LANDING_CLAIMS.json` @ `d6628da0` (blob `1e178c2d…`, 16417 B). There, `claims[6]` (`p15-price-boundary`) has `review: {kind: github_issue, number: 67}` and `disposition: EXACT_COUNTEREXAMPLE`. Its `source.blob` `1d6946658e00…` equals the museum `proof` pin.
- **Target kind:** a whole issue, not a comment. Opened 2026-09-24 by the shared account as a campaign/assignment post; closed `not_planned` 2026-09-27.
  - Its body §D states the two-point counterexample.
  - Line 65 *offers* a combinatorics review of the P15 material.
  - Line 67 says "These are offered tasks, not recorded acceptances."
- **No verdict in the comments:** none of main#67's 11 comments mentions P15, price or demand-one. Every review comment there is D2.
- **So:** the pointer is accurate, but it binds an issue that records **no P15 review verdict**. `EXACT_COUNTEREXAMPLE` is LANDING_CLAIMS' author-side disposition, not a review outcome. The museum entry's `class: "engineering-only"` and `pointer_only` stay consistent with that. Readers should not take the link as a P15 review. Whether a P15 review exists anywhere else was not searched (outside this row).

### Limits and routing
- **Not done:** any check of the mathematical content of any review or proof; the museum UI or other fields; other entries; the history of those comment edits; searches beyond main#63/67/76 and the pinned Math- files.
- **Routing:** notes 1 and 2 are wording/coverage observations for the museum's owner, with no patch proposed. Note 2 may be worth a visible "no review verdict recorded" qualifier if the owner wants one. That would be a separate UI-owner decision, not part of this read-only row.
- **RELEASE:** the BQ-10 read is complete and released. C8 holds no claim. My integration queue position is unchanged.

---
_Generated by [Claude Code](https://claude.ai/code)_
