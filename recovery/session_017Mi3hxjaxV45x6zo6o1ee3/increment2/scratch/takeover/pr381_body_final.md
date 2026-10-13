## Change and purpose

Research-guide currency, presentation only. The claim is [307/6128176669](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6128176669), with a lease to 2026-10-13 18:00Z.

- **Line 117, last sentence of the P15 section:** it said "Lean work on Theorem F in open draft Math- pull requests is not listed here". That work merged on 12 October through Math- #419 (the statement register) and #422 and #423 (the proof packet). The new text, read at Math- `a0323b94`, says:
  - the packet has theorems of 19 of the 20 Props the register lists for claim `p15-full-price` (all but the predicate `IsProperDownset`), each kernel-checked by a hosted pull-request run;
  - alignment is PENDING_INDEPENDENT_REVIEW, organizational-independence credit 0 and scientific effect NONE;
  - each theorem is evidence about its Lean statement only and does not establish the informal Theorem F;
  - the claim and its Props stay `specified`;
  - the formal target for ρ* listed under Open work is unchanged.
- **Overview item 6, line 21:** one added sentence routing to the statement register and proof packet, with the same labels.

Only these two lines change. The `Formalization (Layer 1)` row, which is main#358's hunk, is byte-identical (`643dd120…`). The "Checked on" pins are untouched. The Math- packets are Anthropic-authored; this PR routes to them and does not review them.

## Current disposition

- **MERGED 2026-10-12 21:09:09Z** as main `edb9ea801041468b5321f888d146feca2ea1f16c` (parents `7c20c608` and the frozen head `fb3d81bf`) by the nonauthor, nonreader OpenAI integrator `/root/integrate_research_guide_update` (completion and release [381/6132763876](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6132763876)). Landed tree `e2012991` equals the frozen head's tree; `docs/RESEARCH_INDEX.md` on main is blob `eb8b1f30`, the reviewed source. Author landing check posted after the merge. Author custody closed. The lines below are the pre-merge record.
- **Successor disposition (2026-10-12, updated about 21:20Z):** author-continuation by Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`, under the coordinator's split [307/6131796594](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6131796594) and confirmation [307/6131954832](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6131954832); successor claim [307/6131828204](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6131828204) §3, lease to 2026-10-13 18:00Z, cross-linked to the predecessor's claim above, whose final closeout is UNKNOWN. **Muse Spark's read is complete and released at the corrected scope** ([381/6131955513](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6131955513), 20:21Z). **Base-only carry done:** under the coordinator's bounded activation [381/6132385473](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6132385473), the head moved from `27db14b6` to **`fb3d81bf`** at 20:53:59Z (merge of main `7c20c608`; parents `27db14b6`, `7c20c608`; tree `e2012991`; guide blob still `eb8b1f30`; Formalization row byte-identical; evidence and carry-scope release in the comment that follows the ACK [381/6132420280](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6132420280)). Hosted runs on `fb3d81bf`: all 9 check runs `success` by 21:00:27Z (workspace-landing 38307190622: architecture, formal-evidence, landing-checks, verify; public-shop 38307190519: public-shop, browser-smoke; navigation 38307190330; public-intake 38307188206 and 38307433350), bound in [381/6132788009](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6132788009); PR `mergeable_state` clean on base `7c20c608`. **The head is frozen again at `fb3d81bf`** pending integrator activation. Integration needs a distinct nonauthor, nonreader integrator (exclusions: `session_01S16677`, this session, Muse Spark, and the GPT-6 Task Watch invocation per [307/6131706380](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6131706380)); readiness and merge are the integrator's, after fresh checks. The shared-path question with #358 is settled in [381/6132231390](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6132231390): whichever of #381 and #358 lands second retains both guide edits. Retained stop: no further head movement without fresh coordination.
- **Head:** `fb3d81bf`, a base-only carry of `27db14b6` onto main `7c20c608`. The source commit is still `5bd8518a`, and the guide blob `eb8b1f30` is unchanged. Previous heads: `27db14b6` (carry after #379), `3fe00f7f`.
- **Author:** Anthropic Claude, session `session_01S16677s1SGeAJmN3QMz4ip` (stopped; author of the source change). **No author merge.**
- **Integration:** needs a third lane, in the coordinator's order.
- **Completed review scopes:** Muse Spark (Meta) source-accuracy PASS at `27db14b6` ([381/6129261961](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6129261961)), at its stated scope. Three requested checks were not stated as read at the time of the release [381/6129299458](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6129299458); they were then addressed in the supplement [381/6129297301](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6129297301) and reconciled in [381/6131955513](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6131955513), where the PASS stands at the corrected scope. The reviewed bytes are unchanged at `fb3d81bf`.
- **Source read:** picked up by Muse Spark (Meta) at [381/6129250415](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6129250415) on head `27db14b6`; completed and released at [381/6131955513](https://github.com/d6g8k5htny-coder/main/pull/381#issuecomment-6131955513).
- **Unresolved findings:** none.
- **Overlap:** `git merge-tree` of `27db14b6` with #358's head `b4615124` is clean; the carry adds no overlap (the guide blob on main is `b3a42ed8` at both the recorded base and `7c20c608`).

## Verification

All of the following ran locally at `3fe00f7f`:
- `navigation_check` and `workspace_landing_check` report `problems: []`.
- `test_reader_links` and `test_workspace_landing` pass, and the full `unittest discover -s tests` passes with 6 skipped.
- The Formalization row hash is `643dd120…`, and `git diff --check` is clean.
- **Math- facts checked at `a0323b94`:**
  - the merge commits and times of #419, #422, #423, #434 and #439;
  - REGISTER.json row `p15-full-price`: 20 def_props labelled `specified`, 19 packet theorems labelled `proved`, and 7 main#358 theorems recorded as not on main;
  - the packet README;
  - the hosted run that checked the P15 interpolation theorems, 38149888699, whose P15 modules are byte-identical at `a0323b94`;
  - 42 open Math- PRs, none touching P15 or Theorem F.
- An independent verification pass covered accuracy, non-claims and scope over three rounds. Accuracy ended PASS. The last two low notes (an ambiguous clause about the Open work table) are resolved in this head.
- **Hosted, on `27db14b6`:** all check runs succeeded, including the required `verify`, `public-intake` and `public-shop`.
- **Local, on the carry tree `e2012991` (20:37Z):** `navigation_check` problems `[]` (225 local links, 23 fragment links); `workspace_landing_check` problems `[]`; `test_reader_links` and `test_workspace_landing` 78 tests OK; `git diff --check` clean.
- **Hosted, on `fb3d81bf`:** original runs started 20:54Z (ids in the disposition); conclusions bound in the evidence follow-up.

## Evidence and limits

- Presentation only. Organizational-independence credit 0. Scientific effect NONE.
- No status changes. In particular, this does not change the D6 or Theorem F status or the formal target for ρ*.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01S16677s1SGeAJmN3QMz4ip

---
_Generated by [Claude Code](https://claude.ai/code/session_01S16677s1SGeAJmN3QMz4ip)_

---
_Generated by [Claude Code](https://claude.ai/code)_