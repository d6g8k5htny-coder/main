## DELIVERED — shard W (`incoming/grok-cycle4-20260926/**`) Wave 0 shadow record; review requested; claim held for findings

Dylan Roy — delegated AI evidence maintenance. Actual performer: Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). This completes the inventory for assignment 6019572367 / PICKUP 6019718301. Scientific effect NONE.

- **Record hash:** the record is the text between the first `---` rule and the closing `---` / footer, with trailing newlines collapsed to one. That is 13832 B, SHA-256 `2a7640982bd0f2d64cca847107a6eef76a6b07310f0968298396fc8e2a619aa6`. Check: `rec=body[body.index('\n---\n\n')+6 : body.rindex('\n---\n_Generated')].rstrip('\n')+'\n'`.
- **Result:** 12/12 objects accounted for. 5 are already compliant, 7 need additive supplements (W-S1 to W-S7), and none are blocked or out of scope. Three blocked axes (B-1 to B-3) remain visible and are not counted as finished. No assessment of the D5 or SARD-G mathematics.
- **Notable findings:**
  - (1) The only reviewer-identified native PR review, **5332391136**, binds to `05c20b67`. Six W files changed or were added after it, and the PR merged with "D5, SARD-G A1, A6 remain AMEND".
  - (2) **No recorded hosted CI ever ran W's tests.** The final-head `math-downstream-gates` log runs other folders only.
  - (3) A **main-repo** Claude review (session `01B6fUVk…`, not this one) binds to `0fab9330`. Its findings are recorded against the two files that are byte-identical since then.
  - (4) File 7 is **consumed** by `frontiers/sard_g_robust_charts_20260929/SOURCE_MAP.json`. HARPER_REVIEW_RECORD.md is explicitly not a review of it.
- **Execution:** one new labeled run: the README command (23 tests OK in both modes) and `schur_diagnostic.py` (mode-identical, `--mutate` rejects). Nothing was written to any repository.
- **Review request:** one named nonauthor review of this exact record, by a **non-Claude** lane chosen by you.
- **Status:** claim held for findings until 18:00Z. K v1.1 and U v1.1 are with their readers. Integration queue unchanged.

---

### Shard W record — `incoming/grok-cycle4-20260926/**` evidence inventory (Wave 0 shadow, proposed, not committed)

**Scope.** Math- `08f86862f859ac4804b4fd6c6477a7ed0421e37f`. W tree `d3e44122267be60ed5ff683b2c698f37768f1fba`, 12 blobs. Assignment 6019572367, PICKUP 6019718301.
- **Performer:** Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Credit 0, scientific effect NONE.
- **Exposure:** no authorship, review or integration of these paths. K cites PR87 @`0fab9330` as cited-only context.
- **Requirements applied.** Classes are judged against the requirements in force and the **proposed** v1.1 axes, with K-R1 and K-R2 distinctions applied from the start. No object or shared file was edited. **No assessment of the D5 or SARD-G mathematics is made.**

#### 1. Membership (12/12, all authenticated at base)

| # | Path (under the W tree) | Git blob | Bytes | SHA-256 (prefix) | Changed after review 5332391136 @`05c20b67`? | Class | Supplement IDs |
|---|---|---|---:|---|---|---|---|
| 1 | `PUBLIC_READING_MAP.md` | `f46294127d0ebcf1a25a843ac2bcb34d6827f629` | 2164 | `e608d427` | modified +1/−1 | needs additive supplement | W-S2, W-S5 |
| 2 | `benjamin/ALPHA_4PIN_SERIES.md` | `300e1eb0e285671f341fb803c5d30cb2be027cbe` | 469 | `bb951caa` | no (same blob) | needs additive supplement | W-S1, W-S2 |
| 3 | `benjamin/REDUCED_FRAME_4SLOT.md` | `bd558a265adf8d1b9228fb2012358489939f40c1` | 2260 | `c3d59403` | no (same blob) | needs additive supplement | W-S1, W-S2 |
| 4 | `harper/CLOSED_FORMS_FTS_FTT.md` | `55dc77175976b7dd897bd8f71e52a96bda6a7a8c` | 3867 | `27cb967a` | no (same blob) | needs additive supplement | W-S2, W-S7 |
| 5 | `harper/D5_OBSTRUCTION_LEDGER.md` | `45afad9fe283f08aee53924422fe16958bc03d68` | 17415 | `a06ab084` | modified +109/−29 | needs additive supplement (review axis blocked) | W-S2, W-S6 |
| 6 | `harper/README.md` | `6d1fb3162fced44661f3f7bdcb9156518949acbf` | 5446 | `2d49a429` | modified +18/−1 | already compliant | — |
| 7 | `harper/SARD_G_A1_RELATIVE_INTERIOR_LEMMA.md` | `0ee24f953a4402e58c97ac6d3897eb80d17513a9` | 7329 | `37020f4d` | no (same blob) | needs additive supplement (review axis blocked) | W-S1, W-S2, W-S7 |
| 8 | `harper/schur_diagnostic.py` | `8f5a5305dafffac6bfcdba60700ba6f903007081` | 8806 | `eaa5c23b` | **added** | already compliant | (W-S4 covers execution) |
| 9 | `harper/test_closed_forms.py` | `888f945cb72733161a7ef3cc59f45eef761fa9e2` | 8128 | `163d68b5` | no (same blob) | already compliant | (W-S4) |
| 10 | `harper/test_ftt_conditional_mean.py` | `eb289d99bce11838729036510b6fd83ccc49636c` | 8918 | `981eb395` | **added** | already compliant | (W-S4) |
| 11 | `harper/test_schur_diagnostic.py` | `df3db1ca991b24ce4928da71ed7422d6f2171b01` | 2590 | `5d5b2921` | **added** | already compliant | (W-S4) |
| 12 | `lucas/SARD_G_A1_APPLIED_TO_SUCCESSOR.md` | `af428e6673e25b61a1cfc1a6e4ac429e43c68d02` | 495 | `7e178256` | no (same blob) | needs additive supplement | W-S1, W-S6 |

**Counts:** 12 eligible = 5 already compliant + 7 needing additive supplements + 0 blocked + 0 out of scope. Three evidence **axes** are blocked; they are listed in §3 and not counted as finished.
- **Change relative to review 5332391136's object (W tree `f60faab2` at `05c20b67`):** 6 files are byte-identical, 3 are modified and 3 are added.
- **Alias and citer counts are separate (W-S7).** No citer adds to the 12.

**Identity checks.**
- **W tree history:** `47ee2a83` @`0fab9330`, then `f60faab2` @`05c20b67` (reviewed), then `fac0436f` @`bc85130c`, then `d3e44122` @`52c953c6`. It stays `d3e44122` at final PR head `a1841d50`, at merge commit `8c00f1952d8ce1b82ad469e4ffa8cffbfc0aa264` and at base.
- **Commits:** 8 commits touch W, all in PR #87.
  - `77d1a035`, `d20ce86a`, `f3af9b56`, `8657130b` and `db66b25a` were committed under the Dylan Roy account, 2026-09-26/27, with no session trailer.
  - `25bd0269`, `bc85130c` and `3ee95965` were committed as Cursor Agent, 2026-09-27.
- **Merge:** PR #87 merged 2026-09-27T23:41:37Z by the shared `d6g8k5htny-coder` account, from branch `incoming/harper-cycle4-d5-sard-20260926`. The integrating performer is **UNKNOWN**.

#### 2. Proposed additive supplements (Wave 1 only, after a landed contract; facts are recorded as found)

- **W-S1, provenance (labels as recorded; a name or folder does not authenticate a provider).**
  - In-file labels are lane personas: "Author-side packet: Harper cycle-4 (D)/(E)" in the ledger and lemma, and object labels `ND-REDUCED-4SLOT-20260926` and `SARD-G-A1-APPLIED-SUCCESSOR-20260926`.
  - `ALPHA_4PIN_SERIES.md` has no author line. It asserts "Lucas independent (b,k)-split grid agrees", and no native record of that grid was found.
  - Provider: "Grok" appears only in the folder name `grok-cycle4` and in team note 5851042294 ("Grok + Harper + Benjamin + Lucas"). Navigation row 39 records "xAI / Grok team lane persona". The main review RM16 (see W-S2(d)) independently notes "no author/provider line; the only identifier is the path token 'grok'". **The provider is not authenticated, and the sessions are UNKNOWN.**
- **W-S2, review record (scope-bounded).** Searched: PR #87's reviews, review comments and issue comments.
  - (a) Four `copilot-pull-request-reviewer[bot]` reviews, all COMMENTED with "Changes recommended" (5328259658, 5328269288 and 5328289349 @`8657130b`; 5330637208 @`0fab9330`), plus 11 inline threads. **All 11 threads are marked resolved, with the resolver UNKNOWN.** These are automated and not a mathematical approval.
  - (b) **The one reviewer-identified native review is 5332391136**: "Source-bound engineering/scoped algebra review — OpenAI/ChatGPT", @`05c20b67` only, posted under the shared account, with no independence credit claimed.
    - It verified the denominator-order repair "at algebra scope".
    - Its disposition is "**AMEND remains for the obstruction ledger**".
    - It says: "Retain D5/A1/A6 parent AMENDs… do not silently carry this review to another head."
  - (c) Cursor comments 5860423129, 5860488212 and 5860575976 state that the review is **not carried forward** to `bc85130c` or `52c953c6`, and that "D5, SARD-G A1, A6 remain AMEND". A fresh review was requested. **No later review was found in the searched records**, and the PR merged with the review binding only the predecessor.
  - (d) **Main-repo review records** (outside PR #87), searched by `git grep` of main `cddbb7f6`. These are `reviews/grok_uploads_20260926/` REVIEW.md `a1d82ac9`, PART1 `e8b9783f`, PART2 `7bec0898` and PART3 `291623ff`, from commits `8d0061e` and `544fafb`, 2026-09-27.
    - **Reviewer:** "Anthropic / Claude", session `01B6fUVk…` (not this session). It states it is same-account, source-exposed and earns zero independence credit.
    - **Cutoff:** 2026-09-27T00:45Z. It assessed PR #87 at **`0fab9330`** (W tree `47ee2a83`, 9 files).
    - **Verdict:** "AMEND; **do not merge as is** (D5 ledger blocker)", with findings including P2 D1 (blocker) and D2 (major) on the ledger.
    - **Binding:** every finding binds to the `0fab9330` bytes. Only files 2 and 12 are byte-identical between `0fab9330` and base. For those two, the recorded findings apply to identical bytes: A1 (minor, PLAUSIBLE) on file 2; RM16 (minor), RM10 (nit, PLAUSIBLE), L1 (minor) and L2 (nit, PLAUSIBLE) on file 12. They are recorded as is and not re-assessed here.
    - **Other files** changed after `0fab9330`. Whether later revisions address the findings is **not assessed** here. 5332391136 and commit `bc85130c` ("Withdraw two D5 ledger assertions") are the recorded later actions.
  - Per-file effect:
    - Files 2, 3, 4, 7 and 12 are byte-identical to what 5332391136 read. Its stated scope (denominator repair; ledger AMEND) applies to them as stated and is not widened.
    - Files 1 and 5 were modified, and 8, 10 and 11 were added, after it. **No review in the searched records covers their final bytes.**
    - `harper/README.md` (file 6) records this binding correctly (lines 41–43), which is why it is classed already compliant.
- **W-S3, Git identity labeling.** Commit IDs in the packet (`05c20b67`, `8657130`, `a1fc9581`) are commits, not blobs. W-S6 gives the blob identities where a file is consumed.
- **W-S4, recorded versus observed execution.**
  - **Recorded hosted CI:** check-runs on PR heads `05c20b67`, `52c953c6` and `a1841d50` show only `math-downstream-gates` (runs 36352390785, 36356687019 and 36359404934, all success).
  - **That job does not run W's tests.** The log of job 108733444011 (run 36359404934) shows that it checked out the synthetic merge `0a25452b` (= `refs/pull/87/merge`, "Merge a1841d50 into 5c249928"). It then ran only the downstream-gate suite and five fixed regression folders: `coefficients/side24_v1` and four `frontiers/*`. No workflow at base references `incoming/grok-cycle4`. **So no recorded hosted run executed W's test or diagnostic files.**
  - **Author-side or recorded-by-others:** review 5332391136 reports running "the 12 closed-form tests in both normal/-O" @`05c20b67`. `test_closed_forms.py` is unchanged since. Cursor comments report local runs, which are not re-verified here.
  - **New observation (C8, 2026-10-06, local CPython 3.11.15, clean base worktree, worktree unchanged afterwards):**
    - The README's own command, `python -B -S -m unittest discover -s incoming/grok-cycle4-20260926/harper -p 'test_*.py' -v`, ran normally and under `-O`. Result: **23 tests OK** in each mode (12 in `test_closed_forms`, 5 in `test_ftt_conditional_mean`, 6 in `test_schur_diagnostic`).
    - `schur_diagnostic.py` exits 0. Its stdout is byte-identical across modes (1203 B, SHA-256 `b19bf67e…`).
    - `--mutate` exits 1 with "mutation detected, as required".
    - The printed values equal those quoted in 5332391136, for example Var(f_tt(M))/r^4 = 0.1663336110 at r = 0.1. That is consistency only. It is not a uniform result, and not historical runtime evidence.
  - All four Python files import only standard-library modules (`fractions`, `decimal`, `math`, `itertools`, `unittest`, `sys`, `os`) and the local `schur_diagnostic`.
- **W-S5, where the commands come from, and stale pointers.**
  - The command is in `harper/README.md` line 20 and in the `schur_diagnostic.py` docstring (prose only). No machine replay record and no workflow exist.
  - W's own `PUBLIC_READING_MAP.md` line 11 names "Branch … @ `8657130`", which is a historical, mutable pointer. The supplement records the final W tree beside it.
- **W-S6, consumed external sources.**
  - The ledger's "preserved predecessor" pin, `05c20b67` / SHA-256 `ba0f5f2e48ea9704…`, matches the predecessor ledger (verified).
  - `lucas/…` consumes `research/q0/sard_g/SARD_G_EXECUTION_REPAIR_20260925.md` @ `a1fc9581`. That is a **main** commit, where the path resolves to blob `8f86e0c9ef91fde473e2344a3e461c31f63f8529` (17285 B). The packet names the commit only, and the supplement records repository plus blob.
- **W-S7, citers, aliases and consumers.** Searched by `git grep` of Math- `08f86862` outside `incoming/` and of main `cddbb7f6`, for `grok-cycle4`, `harper-cycle4`, `#87` links and the file names.
  - **Consumer:** `frontiers/sard_g_robust_charts_20260929/SOURCE_MAP.json` entry `RI` binds file 7 by blob `0ee24f95` (= base) at commit `0b0b2819`, with role "applied in P1–P4".
  - **Review-record alias, not a review of file 7:** `reviews/sard_closure_integration_20260929/HARPER_REVIEW_RECORD.md` is an OpenAI editorial summary of Harper's review of a *different* source (`frontiers/sard_g_robust_charts_20260929/PROOF.md`). It states that it "corroborates that prior input's source-level application, **not independent truth** of Harper's own RI proposal".
  - **Cited only:**
    - `certificates/bf_six_pin_scalar_20260926/PROOF.md:103` cites file 4 (see K-S6);
    - `PROOF_INDEX.md:83` and navigation row 39 point to file 6;
    - root `PUBLIC_READING_MAP.md` lines 36–72 hold alias link tables to W files pinned @`05c20b67` and @`52c953c6`;
    - main `reviews/grok_uploads_20260926/` (4 review files plus `checks/d5_detHS_scaling.py`) are review records, already counted in W-S2(d) and not as extra evidence.

#### 3. Blocked axes (outside evidence maintenance; visible, not finished)

- **B-1:** any review covering the final bytes of files 1, 5, 8, 10 and 11. None was found in the searched records.
- **B-2:** the recorded open AMENDs ("D5, SARD-G A1, A6 remain AMEND", 5332391136 and 5860575976). They are mathematical, not resolvable by maintenance, and not changed here.
- **B-3:** hosted CI execution of W's tests. None is recorded, and no workflow covers `incoming/`.
- **Routed, not edited:** whether `incoming/` packets should get a dedicated workflow is a question for the workflow and required-check owners, through the coordinator.

#### 4. Validation and limits

- **Checked:**
  - 12/12 coverage;
  - full blob, tree and SHA-256 identities;
  - the W tree across five PR heads, the merge and base;
  - per-file change against the reviewed head;
  - all native PR #87 review objects, threads and issue comments;
  - the check-runs and executed checkout of the final-head run (from its job log);
  - the external predecessor and `a1fc9581` pins;
  - all Math- citers;
  - agreement with navigation row 39;
  - main-repo citers and review records.
- **UNKNOWN stays UNKNOWN:** author sessions and provider, thread resolver, integrator. No review, independence, reproduction or acceptance credit is added. Folder names authenticate nothing.
- **Not done:** no mathematical assessment of D5, SARD-G or the closed forms, and no assessment of whether the `0fab9330` findings were addressed. Issue threads beyond PR #87, Drive and Dropbox were not searched.
- **Delta queue:** empty. Main is still `08f86862`, with no post-cut change to these paths.

---
_Generated by [Claude Code](https://claude.ai/code)_
