# main#452: pre-assessment of whether C8's review applies after a refresh

Written by Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8), the reviewer of record (PASS_SCOPED at `63cf47667f06f78644c0f1276f602c24d2c300d2`, [452/6279333662](https://github.com/d6g8k5htny-coder/main/pull/452#issuecomment-6279333662)). Read-only. Credit 0. Scientific effect NONE.

It was assessed against current `main` `b78134c75e733febc6bd2becac4459036ef8c0ad` at about 00:20Z on 20 October, before any refresh existed. It is not a verdict on any new head. After a refresh, it must be re-validated against the actual head and base. If `main` has moved again, the base delta must be assessed again.

Byte identity, review applicability and hosted checks are separate questions. Each needs its own answer.

## 1. Byte identity (mechanical; `delta_readback.sh`)

A simulated refresh, a local commit with parents `63cf4766` and `b78134c7`, gives:
- tree `7e561fef`, equal to a fresh `git merge-tree --write-tree`;
- delta paths `AGENTS.md` (+4/−1) and `docs/WORKSPACE.md` (+34/−0) only;
- a raw diff identical to the reviewed diff, `efc2fe90..63cf4766`: same paths, modes, and before and after blobs, so the patches are byte-identical;
- no delta path changed on `main` since the reviewed base.

These results establish byte equivalence only within what the script checks.

## 2. Review applicability (C8's judgment)

**Base changes.** `efc2fe90..b78134c7` adds three files under `incoming/h3-closure-restoration-20261018/` (main#446). None of them is a #452 path, and none is a dependency of the review.

**Dependencies of the review.** Each object is the same at the reviewed base and at current `main`:

| Path | Object |
|---|---|
| `governance/OP-WORKFLOW-20260930.md` | `ff423599` (lines 230–242 are cited) |
| `CONTRIBUTING.md` | `1fd09dbd` |
| `.github/workflows/public-intake.yml` | `d1b421ea` (its trigger list is used in the integrator note) |
| `.github/workflows/public-shop.yml` | `9dc24a7f` |
| `.github/workflows/navigation.yml` | `c4e15729` |
| `.github/workflows/workspace-landing.yml` | `e37310e9` |
| `tools/workspace_landing_check.py` | `07ccd323` |
| `tests/test_workspace_landing.py` | `0e57dd21` |
| `experiments/universality/spectral_gaussian_admission/PROOF.md` | `72c89647` (the #330 example) |

**Interactions with the original six checks:**
1. **The 404 statement:** this is GitHub API behaviour, which a base change does not affect.
2. **Ancestry against carried content:** the example's blob and commits are unchanged. #446 only adds intake files under `incoming/`. Nothing in the reviewed text refers to them, and nothing in it becomes false.
3. **Shallow and restricted fetches:** these do not depend on the base.
4. **The exact tested merge commit:** this does not carry forward. At the new head H′, `refs/pull/452/merge` must have parents `b78134c7` and H′, and `verify` must bind to that commit. This belongs to the hosted checks (section 3).
5. **#421:** it still has no active owner; the last comment is 6240855744, from 17 October. Its head `c2907912` composes cleanly with the simulated H′ (tree `e0654901`), and the composition differs from H′ by exactly #421's own +8/−6.
6. **Intake runs kept distinct from the Ready-triggered review:** this interaction needs the integrator's decision.
   - #452 was marked Ready at `63cf4766` (23:09:03Z). A refresh of a PR that is already Ready produces a `synchronize` `pull_request_target` intake run at H′; no `ready_for_review` event happens at H′.
   - `CONTRIBUTING.md` (lines 127–131) requires the trusted `pull_request_target` run and that it reports the PR's current exact head.
   - My original integrator note ("bind the run that the `ready_for_review` event created") was written for the draft-to-Ready transition.
   - Whether `OP-WORKFLOW`'s Ready-event binding also requires a new Ready event at H′ is for the integrator to settle, under Dylan's "applicable trusted intake evidence".
   - The same applies to the automatic review: an earlier review of another head does not cover H′.

**Landing check on the composed tree:** 16 pages, 188 local links, 2 historical files, no problems. This is the same result as at the reviewed head.

**Notes N1 and N2:** unaffected.

**Conclusion (provisional):** if byte identity holds at the actual H′ and `main` is still `b78134c7`, C8's PASS_SCOPED applies to the composition. Two conditions remain: point 4, the tested merge commit, which needs fresh hosted checks; and point 6, which the integrator settles. Anything that lands on `main` after `b78134c7` needs a fresh assessment.

## 3. Hosted checks (reported separately after a refresh)

At H′: `verify`, `public-intake` and `public-shop` (strict, ruleset 23798639), plus the PR's other checks. Each must be reported with its own event, workflow and checked commit, and none is inferred from section 1 or 2.

## Ownership and restriction

C8 integrates nothing. Before any read after a refresh, C8 confirms current ownership, takes a bounded pickup pinned to the exact head and base only if no one else owns that assessment, and otherwise coordinates.

A release by 014K or Claude017 ends that worker's reservation. It does not clear a permission denial or authorize another session to perform the denied action (Dylan, 20 October about 00:14Z, in C8's session).
