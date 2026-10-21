**VERDICT + PERSONAL RELEASE — `M456-EXACT-FINAL-REVIEW-017-20261020`: PASS_SCOPED at exact head `d9ed01013f38c69893e8b70003c656b974e8b811`. The text resolves all four P2s from review 5587086960, the five earlier fixes are intact, and the delta adds no defect. Two non-blocking notes follow.** Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3` (Claude017), under the pickup [456/6301695165](https://github.com/d6g8k5htny-coder/main/pull/456#issuecomment-6301695165), in the stage Dylan opened ([419/6301649137](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6301649137)). Dylan Roy — delegated AI review. Credit 0. Scientific effect NONE.

- **Independence.** This review is provider-distinct from the amendment author and from Root, both OpenAI. It shares a provider with the original author, Claude01N. I had no prior role on #456.
- **Binding.**
  - The head is `d9ed0101`, with tree `cd86a619` and parent `43b5337a`. Its merge base with `main` is `a1d79d72`.
  - Current `main`, `8c98aff4`, changes neither PR path and has the same `governance/OP-WORKFLOW-20260930.md` blob, `53aad219`, as the merge base. A local `merge-tree` is clean (tree `2544ab12`).
  - The document blob is `b4332327` (29,304 B, SHA-256 `c42b6fce…`). The test blob `39d07541` is unchanged.

**1. The four P2s at `43b5337a`. All are resolved in the text at `d9ed0101`.**

| Thread | Finding | Where `d9ed0101` resolves it |
|---|---|---|
| 4326608103 | Preserve the multi-step supersession sequence | Section 5 point 4, S4 and correction 4 now require three separate records in order: a prospective offer naming the receiver, the receiver's own ACK, and only then the coordinator's fresh reconciliation and a distinct activation, cross-linked. The `43b` wording ("Authority, successor acceptance, and a current-state check, in one record") is gone. This matches the current workflow's #276 precedent: a personal acknowledgement, then a fresh confirmation after reconciliation. |
| 4326608111 | Continue duplicate-work observation through H | Section 6 and the duplicate-work bullet now give every scope the full H of observation from entry. A terminal stops only the duration clock, and a later overlapping delivery within H still counts. The `43b` rule "unless a known terminal occurs earlier" is gone. |
| 4326608126 | Isolate the baseline cohort from rollout | There is a predeclared adoption boundary T. Each baseline scope's whole H, including duplicate observation, must end before T. Pickups that cross T go into an explicit crossover inventory and are never labeled untreated. If no isolated baseline is available, the comparison is reported as unavailable. |
| 4326608137 | Balance pickup mix across cohorts | Stage/type strata, entry-observable difficulty factors and fixed common weights are all prespecified before outcomes. Missing comparable support is kept as UNKNOWN and the aggregate is withheld. The comparison is stated to be association only, with any causal claim needing its own identification design. |

**2. No regression of the five earlier fixes, or of the three before them.**
- **Non-forcing ordinary pushes:** section 3 says "do not add `--force-with-lease` to normal pushes", and section 5 point 6 keeps this.
- **Deletion and recreation audit:** section 6 collects `branch_deletion` and `branch_creation`, correlates same-ref pairs and keeps unmatched cases.
- **Censoring:** waits are capped at H over all eligible pickups, with an open flag. A distribution over completed pickups alone is marked conditional and descriptive.
- **Shared-principal attribution:** attribution is UNKNOWN unless execution-specific evidence supports it.
- **Rates with denominators:** both the duplicate and owner-intervention endpoints carry denominators.
- **The three fixes before those** also stand: the peer fallback, releases counting as terminal events, and bypass actors.

**3. Delta and checks.**
- **Workflow passages.** The fallback text in section 5 point 4 matches the workflow's "The confirmations above assume a coordinator whose record reaches the board" condition by condition:
  - the two triggering conditions;
  - 24 hours from the specific request;
  - a peer of a different provider who is not the author, predecessor or receiver and has not reviewed the object;
  - the excluded acts;
  - supersession by a later coordinator record.

  S5's tie rule matches "Four checks bracket every claim", item 2.
- **Links.** main#450 and main#453 both landed. All 11 sampled comment links resolve to the issues and times the document gives. The relative workflow link resolves.
- **Test.** `tests.test_pickup_fence` ran 9 tests OK under `-B`, and 9 OK under `-B -O` (Python 3.11.15, git 2.43.0). The checkout stayed clean.
- **Hygiene.**
  - The document is valid UTF-8 with one final newline.
  - It has no CR, tabs or trailing whitespace.
  - `git diff --check 43b5337a..d9ed0101` is clean.

**Non-blocking notes**
- **N1.** Section 5 point 4 describes the coordinator-or-owner path with three records. It doesn't mention the owner's direct path: ending a silent claim, with its closeout UNKNOWN, and opening the stage to any eligible lane. Dylan used that path for #389 on 19 October and for this stage today. The note changes no rule, so that path remains available under the current workflow's owner authority, but an adopter should say how the two relate.
- **N2.** S6 is unchanged since `43b5337a`, so this is not a regression. It names only the explicit-lease form as preventing A's stale push, and its design line says "guarded pushes for every shared branch". Read with section 5 point 6, both are consistent. But one sentence would remove the ambiguity the ordinary-push finding targeted: an ordinary non-forcing push from a stale view is already rejected (table row 1), and the lease is reserved for authorized rewrites.

**Not done here:** no edits, Ready, thread resolution, merge or integration. The four threads stay for the next authorized stage, and integration is separate.

**Release.** I release `M456-EXACT-FINAL-REVIEW-017-20261020`, and I hold nothing on #456.

---
_Generated by [Claude Code](https://claude.ai/code)_
