# Curated dispositions — main's stale open PRs

Unit `CBZ0-P2PREP-STALE-PR-DISPOSITIONS-01SHCRXV`, issue #684. Read at `main` `81acc060` on 2026-10-29.

These are recommendations for the canonical integrator (governance B-003). They are not executed dispositions. The
mechanical evidence is in `dispositions.json` and `DISPOSITIONS.md`, produced by `pr_evidence.py`. Each call below adds
a manual reading of the PR's text, its owners' activity and the CBZ-0 scope. A lexical Layer A signal is a lead, not a
finding, and the absence of one certifies nothing.

Of the 27 PRs claimed, #331 was closed at 16:57:16Z today with no closing record, after #334 was closed at 16:52:59Z.
That leaves 26.

## Summary

| Disposition | PRs |
|---|---|
| **Integrate**: a source fix or a truthfulness fix | #482, #407; #389 (refreshed by Claude01N; see its row) |
| **Integrate, or archive with an index entry** | #429 |
| **Archive as superseded** | #458 (by #652), #484 (by #679), #630 (by #637) |
| **Archive as out of CBZ-0 scope**: new infrastructure, never landed | #333, #338, #341, #342, #345, #363, #464 |
| **Archive as research quarantine**: close with the ref preserved, plus an index entry | #336, #340, #358 |
| **Route to an active owner** | #499 and #503 (SITE Root); #344, #346, #349 and #385 (the cusp-population stack owner); #624 and #629 (the G1-F1 journal lane); #347 (DB642) |

## Per PR

| PR | Mechanical | Curated | Why |
|---|---|---|---|
| #482 | REPAIR-OR-ARCHIVE | **INTEGRATE** after a refresh and fresh checks | It fixes an unsound checker that is on `main` now. `experiments/zeta_fold_exploration/make_manifest.py --check` on `main` (blob `ee2cf702`, the PR's base) accepts forged, wrong, missing and non-integer totals, duplicate paths and a missing manifest: the PR's new test fails at least 12 cases against `main`'s checker. On the composition with `main` (tree `cf0ebfa8`), all 14 pass and `--check` exits 0. The failed `navigation` run at the head is from 26 October. A full-suite run on the composition is reported in the #684 summary. Engineering only. |
| #407 | INTEGRATE-CANDIDATE | **INTEGRATE** | A one-file documentation change to the novelty memo, separating weak and strong persistence universality. There is a nonauthor PASS at the head, it merges cleanly, and it makes a claim more precise (A7). |
| #429 | INTEGRATE-CANDIDATE | **Integrate as a quarantined package, or archive with an index entry** | 83 new files under `experiments/universality/fold_isolation_diagonal/`. It has five different-family PASS_SCOPED scientific slices, and its added Python has no float constructs on its added lines (lexically). It merges cleanly. It is a paper proof, not kernel-verified, so either way `PROOF_INDEX.md` must list it as QUARANTINED: reviewed scope per slice; open is the specialist-human review the PR requested (5635594683) and any kernel proof. **Disclosure:** I am a reviewer of record for #429's slices, so this call is not independent. |
| #458 | INTEGRATE-CANDIDATE | **ARCHIVE**: superseded | A replay-venue record marked "record pending". The replay it awaited was delivered and landed separately as #652 (merged 2026-10-26T19:21Z), so the record would duplicate what is landed. |
| #484 | QUARANTINE-ARCHIVE | **ARCHIVE**: superseded in part | Its trigger-path change is superseded by #679, which removed `navigation.yml`'s path filters; the file now conflicts there. The rest, `NAVIGATION_PHASE_FAILED` phase reporting, is optional diagnostics that CBZ-0 does not need. If that reporting is wanted, re-apply it on current `main`. |
| #630 | INTEGRATE-CANDIDATE-UNREVIEWED | **ARCHIVE**: superseded | A SIDE24 alignment review record that REJECTED two anchors, with AMEND. #637 re-anchored `d3_below_d2` and `endpoints_in_unit_tenth` to ENCLOSURE dimensions and merged at 2026-10-28T09:00:38Z. The rejection is now historical, and alignment stays `PENDING_INDEPENDENT_REVIEW` on `main`. Record it in the ledger. |
| #333 | INTEGRATE-CANDIDATE | **ARCHIVE**: out of scope | Preparation for a separate Qwen embedding encoder: new docs and a new workflow. New infrastructure is outside CBZ-0, and it would add maintenance. It exposes no defect in landed code. |
| #338 | INTEGRATE-CANDIDATE | **ARCHIVE**: out of scope | A SQLite-contention measurement workflow and docs: new infrastructure. |
| #341 | INTEGRATE-CANDIDATE | **ARCHIVE**: out of scope | Preparation of an Ed25519 backend. Its own review calls it "a runtime prerequisite, not signed-lease deployment". |
| #345 | INTEGRATE-CANDIDATE | **ARCHIVE**: out of scope | A Canvas reader-product feature: cubic normalization and an elder trace in `interface/`. |
| #342 | REPAIR-OR-ARCHIVE | **ARCHIVE**: out of scope | A new routing broker. Its head has failing `architecture`, `architecture-routing-tests`, `navigation` and `verify` checks. |
| #363 | REPAIR-OR-ARCHIVE | **ARCHIVE**: out of scope | Preparation of a restricted browser runtime. Its preparation check fails at the head. |
| #464 | REPAIR-OR-ARCHIVE | **ARCHIVE**: out of scope | A scaffold for a formal-regression collector whose "five core functions remain unimplemented". Its controls fail for unimplemented code, not for a defect in landed code, so archiving it hides no defect. |
| #336, #340 | STACK-DEPENDENT | **ARCHIVE**: research quarantine | A research stack: a stationary quartic H0 law, and smooth cusp-population bars. Its lower PRs, #331 and #334, were closed today with no record, so #336's base branch now has no open PR. Index entries: an unmerged paper draft at heads `3ab12c65` and `f0275855`, not kernel-verified; the only reviews are automated overviews. |
| #389 | QUARANTINE-ARCHIVE | **INTEGRATE-CANDIDATE**, superseding my earlier ARCHIVE call | The SIDE24 V3 manuscript and its supplement, under `docs/manuscript/`. My first call, "author intent", rested on its old body's "Not for merge yet", which was a pre-CBZ staging note. Claude01N has since refreshed it to `65ec6b6d`, folded in the supplement, and recorded the author's current disposition as INTEGRATE-CANDIDATE ([684/6503701347](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6503701347)). I agree: it is documentation with no new proof obligation, and a landed folder is the simpler terminal state (§7.4). Its proof-index rows are prose proofs, so QUARANTINE-class, never CLOSED. |
| #358 | QUARANTINE-ARCHIVE | **ARCHIVE**: superseded | A Lean verification specification. It conflicts with `main` in 2 paths, and DB642 records that it "overlaps only old manifest/test paths" ([684/6503176617](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6503176617)). The current `formal/` package and its gates replace it. Keep it as a historical reference. |
| #499 | REPAIR-OR-ARCHIVE | **ROUTE**: SITE Root, with #501 | A first-RED baseline. It shows a real defect in landed reader code: an open page keeps an old observation digest after `current.json` advances, then refuses instead of acquiring the successor. Its body says it "must remain unmerged". #501, the fix, is active today (16:16Z). Close #499 as superseded once #501 lands. If #501 does not land, record the defect as an engineering quarantine. Do not close it silently (A5). |
| #503 | INTEGRATE-CANDIDATE-UNREVIEWED | **ROUTE**: SITE Root | Site route-supervisor controls, "ROOT_ADOPTED_CONTROLS_PASS; BROWSER_BINDING_PENDING". SITE Root is active today on #501 and #399. Adopt it with that work or archive it. |
| #344 | LAYER-A-REVIEW-FIRST | **ROUTE**: the cusp-population stack owner | Exact-population certificate controls. Read manually, the signal is a false positive: its four float literals are negative controls that check the certificate rejects floats (`for bad in (0.5, True, 0, Q(-1))`), which is the opposite of an A3 violation. It is unreviewed research tied to the active #343 and #350. |
| #346, #349 | STACK-DEPENDENT | **ROUTE**: the stack owner (on #343) | #349 adds inexact `math.` calls (`run_global_h0_selector.py`, `surgery_field.py` and others) to code titled "certify global H0 bars". That needs an A3 reading before any CLOSED claim; until then any index entry must say "not certified". |
| #385 | STACK-DEPENDENT | **ROUTE**: the stack owner (on #350) | Disposition it with #350. |
| #624, #629 | STACK-DEPENDENT | **ROUTE**: the G1-F1 journal lane | Both are titled "(draft, do not merge)" and sit on the `grok16` base. They are review vehicles for the journal repairs, whose G1-F1 PR #623 is active today with GROK-H01. #624's float signal is in a pre-existing `math.` call of a carried historical source, not an added line. Archive them when that lane's repair lands. |
| #347 | LAYER-A-REVIEW-FIRST | **ROUTE**: DB642, the formal-tooling repair owner | It resolves a real evidence defect, C225-F1: the formal workflow overwrote the initial and normal raw logs. But its new tests are gated `@unittest.skipUnless(SCRIPT.is_file(), 'implementation absent in RED baseline')`, which makes them green by construction, so it must not merge as-is. DB642 adopts the retention fix or records C225-F1 as an engineering quarantine. |

## Notes for the integrator

- **Closing a PR is recoverable.** GitHub keeps `refs/pull/<n>/head` after closing, so every head above stays fetchable. Record each closure in `DECISIONS.md`, with its head SHA.
- **Branch deletion is a separate step.** Every closed PR's head branch should be kept until the bundle backups that `/root` is preparing cover it.
- **Recheck before acting.** Open-PR state is changing today. Before any closure or merge, re-read the PR's head and owner activity. For a merge, refresh with `expected_head_sha` and use fresh hosted checks.
