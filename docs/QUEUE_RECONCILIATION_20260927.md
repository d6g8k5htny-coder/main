# Queue reconciliation — 2026-09-27

Scientific effect: **NONE.** This page records how the open pull requests and
issues of this repository were dispositioned on 2026-09-27 under the current
[owner instruction](../governance/OP-AUTONOMY-20260923-v2.1.md), what actually
ran, what was established, and what remains uncertain. It moves no theorem,
claim grade, gate, or independence credit. A merge recorded here is a landing
decision; it is not mathematical acceptance.

Actor: Anthropic / Claude (Cursor cloud agent) on the shared operator account.
Every review verdict cited below was written on the same account; zero
organizational-independence credit applies throughout. Earlier "HOLD / do not
merge" comments were read as owner-era directions that the current instruction
does not continue as vetoes; each one is named where it applied.

## 1. Pull requests targeting `main`

| PR | Disposition | Basis |
|---|---|---|
| [#167](https://github.com/d6g8k5htny-coder/main/pull/167), [#163](https://github.com/d6g8k5htny-coder/main/pull/163), [#161](https://github.com/d6g8k5htny-coder/main/pull/161), [#159](https://github.com/d6g8k5htny-coder/main/pull/159), [#114](https://github.com/d6g8k5htny-coder/main/pull/114) | Merged | Review/intake/navigation content; branch updated with `main` before each merge; `verify`, `public-intake`, `public-shop` green on the merged head |
| [#166](https://github.com/d6g8k5htny-coder/main/pull/166) | Amended, merged | Museum conditional route now shares one verified startup (`verifiedMuseum`) with `startMuseum`; 78 frontend tests (76 pass, 2 skipped), CLI replay, 120 unittest cases |
| [#125](https://github.com/d6g8k5htny-coder/main/pull/125) | Amended, merged | Shrinking witness-pair note: uniform moment lemma added, `r^3` stated as an upper bound only, original index row restored; 127 tests, navigation check clean |
| [#170](https://github.com/d6g8k5htny-coder/main/pull/170) | Merged | Grok-uploads review Part 1 and summary; the sympy check `checks/pr82_quartic_det_check.py` reproduces its recorded output exactly |
| [#128](https://github.com/d6g8k5htny-coder/main/pull/128), [#103](https://github.com/d6g8k5htny-coder/main/pull/103) | Closed, AMEND-parked | Technical verdict AMEND with no author amendment; branches preserved; disposition comment on each |
| [#173](https://github.com/d6g8k5htny-coder/main/pull/173) | Reviewed, merged (aad07fd) | Offline source census tool and 39 reconciled R2 rows. Archive handling probed beyond the fixtures (header-lying ZIP, gzip and member budgets, ustar/GNU/pax tars): all refused or bounded as documented; 20 tool tests and 127 repository tests OK in both interpreter modes. Dropbox/Drive counts stand on the linked Drive evidence, not on anything checked here |
| [#172](https://github.com/d6g8k5htny-coder/main/pull/172) | Reviewed, amended, left open (draft, author active) | Dropbox reconciliation tool. `stage` treated a missing `--allow` as allow-everything, contrary to the PR body; fixed on the branch to fail closed (34807df) with two tests, one of which fails against the previous code. Not merged while the author was still pushing |

## 2. Pull requests targeting `chatgpt/drive-github-hardening-20260919`

The hardening branch is unprotected, so a merge there is a plain merge commit.
Each branch below was first merged with the current base in a worktree, its
review findings were answered by an author-side amendment, the affected checkers
and tests were run under Python 3.11.16 with the locked pytest, and the
amendment was recorded in a PR comment before landing.

| PR | Disposition | What was amended and what ran |
|---|---|---|
| [#140](https://github.com/d6g8k5htny-coder/main/pull/140) | Merged | Pinned-source checker preamble and scope docstring corrected; index regenerated; steps 1–50 of `run_checks` PASS; full suite green once `python` was on PATH |
| [#136](https://github.com/d6g8k5htny-coder/main/pull/136) | Merged | `noncertifying_check` exclusion predicate made path-segment exact; `CLAUDE.md` reconciled with #140; one test narrowed to the index table row |
| [#148](https://github.com/d6g8k5htny-coder/main/pull/148) | Merged | `operator_directive_check` widened (owner/asked phrasing, SHA ≠ Drive id, cited-path existence); two governance sentences now cite their source inline |
| [#134](https://github.com/d6g8k5htny-coder/main/pull/134) | Merged | Claims firewalls fail closed on unreadable carrier index, unknown carrier, non-string status; 181 claims tests |
| [#110](https://github.com/d6g8k5htny-coder/main/pull/110) | Merged | Cell-count pin test docstring and required-file semantics; 41 tests; hosted `verify` green |
| [#111](https://github.com/d6g8k5htny-coder/main/pull/111) | Merged, then repaired | See §3 |
| [#124](https://github.com/d6g8k5htny-coder/main/pull/124) | Re-reviewed, merged | C1 object fixed per the review's option (a): `q_MS^br`, PREMISE-BRANCH-MASS, envelope `B_miss ∩ {D(M)≠S}`; C2 liminf scope; C3 `(√15/2) r` and `4/225` recomputed in `Fraction`; adapter fetch-retry keeps the all-zero refusal |
| [#121](https://github.com/d6g8k5htny-coder/main/pull/121) | Closed, superseded | Strict subset of #124; closed after #124 landed, as the review asked |
| [#122](https://github.com/d6g8k5htny-coder/main/pull/122) | Amended, merged (9ce5c66) | SARD-G A1 geometry: relative-interior section hits with clearance margins ρ, τ; analytic translation counterexample recorded; sampled minima labelled NON-CERTIFYING; review-report identities recorded in `SOURCE_MAP.json` under keys that the pinned-source checker does not read as a certificate (the first push omitted that rename and failed hosted `verify`; corrected in 4d9784f before merge). Author-side; A1/A6 re-check still required; the full-Gaussian C103 corollary stays on HOLD |
| [#8](https://github.com/d6g8k5htny-coder/main/pull/8) | Amended, merged (9458b90) | RN full-mark sector packet: pinned-source index regenerated for the current tip; certificate count updated in the test and `CLAUDE.md` (`certificates=16 pinned_files=55`); the eight repository files the packet freezes by SHA-256 are now named on the thread. Full suite 3642 passed, one flaky CLI timeout passed in isolation. Custody of a candidate under imported premises; `third_full_numerical_replay_performed: false` stands |
| [#7](https://github.com/d6g8k5htny-coder/main/pull/7) | Amended, merged | P15 foundation-first packet: docs page now cites REV-P15-A..D (C and D are AMEND), routes substitution through P10-A so P14-E is not a dependency, corrects the PR #5 reference, records that the `source_guard` whole-file pins coincide with existing certificates; packet members untouched; index regenerated on the combined tree (`certificates=26 pinned_files=57`) |
| [#169](https://github.com/d6g8k5htny-coder/main/pull/169), [#171](https://github.com/d6g8k5htny-coder/main/pull/171) | Opened and merged by this pass (cad99e2, d4ad3bb) | Repairs in §3; #171 was merged only after its hosted PR `verify` run passed |

## 3. Defects introduced by this pass, and their repair

Two. The second is small: the #166 amendment commit f9ae228 added
`tests/__pycache__` and `tools/__pycache__` bytecode to `main`. The pull
request carrying this page removes those ten files from the index and adds a
`.gitignore` for `__pycache__/` and `*.pyc`. The first is below.

#111 was merged at 8e2eda4 while its `ci / verify` run was still in progress;
that run then failed at `claims_gate_adapter.py event-compare`. The new node
`H3-RUNG-FLOOR` was graded `CERTIFIED_RUNG`, which the gate treats as a
controlling status, so the transition was `UNSUPPORTED_CONTROLLING_PROMOTION`
against the pre-#111 base and, because its `source` field was prose only,
`UNRESOLVED_CONTROLLING_SOURCE` on every later transition of the branch. A
separate interaction with #134's closed arithmetic vocabulary left
`claims_check` at `problems=3`.

Repairs: #169 added the certificate's verbatim declaration
`interval (mpmath iv, 100 dps)` to the exact vocabulary and fixed the README
count; #171 bound `H3-RUNG-FLOOR` to its frozen carrier bytes and recorded its
operational grade as `AUTHOR_SIDE_CERTIFIED` with
`source_grade_verbatim: CERTIFIED_RUNG` and
`audit_disposition: LABEL_PRESERVED_NOT_CONTROLLING`, following the graph's
D1-v2.2(1) convention. With #171, event-compare from 38a3e07 (pre-#111) to the
repaired tip reports `transition_ok=true`. The dependency edges #111 introduced
stand. A merge conferred no status; the label the object asserts is preserved
where the checkers read it as a label, not as a grade.

## 4. Issues

Closed by this page's pull request (each is a record whose deliverable has
landed and been reviewed; closing moves no status):

| Issue | Why it closes |
|---|---|
| [#165](https://github.com/d6g8k5htny-coder/main/issues/165) | The grok-cycle2 review packet landed as #163 and was reviewed in [`reviews/grok_uploads_20260926/`](../reviews/grok_uploads_20260926/REVIEW.md) (Part 1, verdict AMEND at packet scope). The D5 power-counting disagreement is recorded there and on the Math- side; it is not resolved by closing this notice |
| [#160](https://github.com/d6g8k5htny-coder/main/issues/160) | The LB-RATE / KIMI-THM-023 HOLD is recorded on `main` by #161 ([`reviews/lb_rate_thm023_landing_20260926/`](../reviews/lb_rate_thm023_landing_20260926/REVIEW.md)); `0.9144` remains a measured-grade candidate, not a proved constant |
| [#154](https://github.com/d6g8k5htny-coder/main/issues/154) | Release architecture delivered (#155, follow-through, #166); the remaining line item, the profile banner, was reported UNDELIVERED for lack of connector write access and is not a repository change |

Kept open, with the reason:

| Issue | Reason |
|---|---|
| [#141](https://github.com/d6g8k5htny-coder/main/issues/141)–[#146](https://github.com/d6g8k5htny-coder/main/issues/146) | Public tasks for outside contributors; nothing to reconcile |
| [#117](https://github.com/d6g8k5htny-coder/main/issues/117) | Two of three source objects PRESENT_EXACT (#126); the Module I extraction and the standalone V3.3 table are still unresolved |
| [#113](https://github.com/d6g8k5htny-coder/main/issues/113) | Sandbox lane; H5/H6 returned REJECT; other hypotheses untested |
| [#95](https://github.com/d6g8k5htny-coder/main/issues/95) | Closure audit of 2026-09-26: populated schema graph for the five load-bearing families and the formal pilot lemma are still missing |
| [#94](https://github.com/d6g8k5htny-coder/main/issues/94), [#86](https://github.com/d6g8k5htny-coder/main/issues/86) | Standing gates for the downstream-first program |
| [#67](https://github.com/d6g8k5htny-coder/main/issues/67), [#63](https://github.com/d6g8k5htny-coder/main/issues/63) | Author-side notes still under AMEND; the intermediate-scale bridge (#128) is parked |

Not correctable with this token: [#116](https://github.com/d6g8k5htny-coder/main/issues/116)
was auto-closed when #161 merged, because the body line "Does not close #116."
contains a closing keyword. The issue PATCH endpoint returns 403 for this
integration, so it could not be reopened here. Its subject, the nonauthor
source audit of Q0-C101, is not finished by that closure.

## 5. What remains uncertain

- Hosted CI on the hardening branch was red between 8e2eda4 and #171; the
  local replays in this page used the same locked interpreter and reported
  green, and the PR run of #171 is the first hosted confirmation. From #171
  on, every landing waited for the hosted PR-event `verify` run; the
  push-event run on a newly pushed branch fails by design (all-zero `before`)
  and was not read as a defect.
- `tools/run_checks.py` step 53 (`claims_gate_adapter.py event-compare`)
  fails on every tree, including the untouched base, when the orchestrator is
  run without `CLAIMS_GATE_BEFORE_REF`/`CLAIMS_GATE_AFTER_REF`; the compare
  was run by hand with explicit refs for each landing instead.
- `tests/test_claims_gate_enforcement.py` CLI cases have 20 s/30 s subprocess
  timeouts and failed intermittently under CPU contention on the untouched base
  as well as on branches; each such case passed on re-run and in isolation.
- The mathematics landed on the hardening branch (#122, #124, #8, #7) is
  author-side or custody at stated scope. Nothing here closes A1/A6, C103,
  the RN sector premises, or the P15 imported primitives.

## 6. Later the same day

Added after the page above was merged (3bfc736). The loop continued while other
agents opened new work.

| Item | Disposition | What ran and what stays uncertain |
|---|---|---|
| [#7](https://github.com/d6g8k5htny-coder/main/pull/7) | Merged (a01c72f) | Full `pytest` 3664 passed, one flaky CLI timeout passed in isolation (21/21 on module re-run); hosted PR `verify` green. Hardening tip after the merge: `pinned_sources_check certificates=26 pinned_files=57 problems=0`, `claims_check problems=0`, `noncertifying_check problems=0`, 81 firewall/pin tests |
| [#175](https://github.com/d6g8k5htny-coder/main/pull/175) | Reviewed after merge (6a3a8ca, merged by its author) | Embedded-source unpacker: exact-byte admission only through `accept` (64-hex digest, equal size), `ast.literal_eval` with no execution, path-safe output names; 40 tool tests and 127 repository tests in both modes; seven probes outside the fixtures refused or labelled as documented. The 387/386 extraction totals and 249/11 ledger split stand on Drive evidence, not on anything checked here |
| [#176](https://github.com/d6g8k5htny-coder/main/pull/176) | Reviewed, merged (f8591b0) | Documentation only: two exact recoveries from Library originals, nine identities open; alias and origin distinctions checked; the recovered files' digests are not re-derivable here |
| [#177](https://github.com/d6g8k5htny-coder/main/pull/177) | Merged by its author (4049386) | Formal-verification guide pointing at the `Math-` PR92 pilot; coordination note posted there because this integration cannot comment on issues |
| [#178](https://github.com/d6g8k5htny-coder/main/pull/178) | Conflict resolved, reviewed, merged (eaf1264) | Second Lean pilot (`formal/`, core Lean v4.34.1, SIDE24 arithmetic skeleton). Replayed here with the workflow's pinned elan digest: `lake build` OK, 31 declarations axiom-free, gate `problems: []`, 38 controls and 165 repository tests in both modes, source pins hash-equal `Math-@9d7b680`. The `AGENTS.md` conflict with #177 was resolved to name both pilots and forbid a third registry. All 31 alignment reviews remain `open`; the two pilots have no registry crosswalk yet |
| [#172](https://github.com/d6g8k5htny-coder/main/pull/172) | Open (draft, author pushing) | Author merged the fail-closed `--allow` fix and added per-file extraction-error recording, `pack`/`report` stages and `governance/OP-PRIVACY-20260927.md`; 147 tests OK on `63dbb6f`; hosted checks green. Left for the author to mark ready |

Standing capability limits observed in this pass: the integration token cannot
comment on issues, edit labels, reopen issues or call update-branch; those were
worked around through PR comments, merged PR bodies and local merges pushed to
the PR branches.
