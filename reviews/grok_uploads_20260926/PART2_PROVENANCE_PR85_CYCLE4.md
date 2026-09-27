<!-- PART 2 sections: Grok uploads of 2026-09-26 (custody merges, Math- PR #85, Math- PR #87, comments, cross-cycle). -->

**Conventions for Part 2.**
- All times are UTC on 2026-09-26 unless another date is given.
- "PROVEN" means the git or GitHub record shows it. "CIRCUMSTANTIAL" means an inference from timing, branch names or self-identification text. The shared account d6g8k5htny-coder does not identify an agent.
- Commit dates are set by the client. Monte Carlo (MC) and floating-point results are NON-CERTIFYING.
- Severities are the verifier-adjusted values. "adj." marks a change from the first reviewer's rating. Only CONFIRMED and PLAUSIBLE findings are listed, and PLAUSIBLE ones are labelled.
- Verdicts are technical only. They carry no organizational-independence credit and change nothing in STATUS.md.
- Finding IDs: P (section 1), C (2), H/A/R/D/S/L/K (3), G (4), X (5).

**Corrections to the task brief** (all PROVEN by record):
- PR #64 was marked ready_for_review at **21:45:54Z**. The 21:46:02Z time is the Codex bot's summary comment.
- Math- PR #87 is **no longer a draft**. It was marked ready at 2026-09-27T00:44:47Z.
  - Its head is now `0fab9330c017e78eb9ef6d928e04aba9f376a470`: commit `db66b25` (00:45:10Z) adds `incoming/grok-cycle4-20260926/PUBLIC_READING_MAP.md`, then `0fab933` (00:45:25Z) merges main `03333792`.
  - The PR has 9 files and `mergeable_state` is `blocked`. The local `origin/pr/87` is still at `8657130`.
  - Every file reviewed in section 3 has the same blob at `8657130` and `0fab933`.
- New Grok-lane surfaces exist that the brief does not list. The main ones are Math- PR #88 and main issue #165; section 4 has the full list.

---

## 1. Provenance of Math- d8f5505 and 5ed3b45

### 1.1 What was claimed
- `d8f55054` is the squash merge of Math- PR #64 (ChatGPT/Codex-lane erratum).
- `5ed3b455` is the squash merge of Math- PR #85 (head branch `grok/proof-index-erratum-pointer-20260926`, titled "custody").
- The Grok lane's comments said the erratum "still 404s on Math- default" and needed to be "in-tree". Round-9 comment 5850251715 then reported "Math- default is now 5ed3b455 (PR85 squash)".
- Nobody has claimed the merges.

### 1.2 Timeline (UTC)

| Time | Event | Basis |
|---|---|---|
| 05:25–05:26Z | Codex review claim 5843515944 and nonauthor ACCEPT_ERRATUM 5843521502 on PR #64 (erratum blob 213594d6, sha256 bad7ef60…2028). | PROVEN |
| 16:25:59Z | main #144 opened (labels do-not-merge / eng-only / math-review). It points at `d573b99d` and says "Keep both mathematical PRs unmerged for this task". Never updated afterwards. | PROVEN |
| 16:28:05Z | main `docs/PUBLIC_SHOP_SETUP.md:35` first names Math #64 in "Do not merge Math #60/#64/…" (commit 349e036). The current wording dates from 4c5dd38 (18:02:37Z). | PROVEN |
| 18:07:02Z | Claude AMEND 5848593803 on PR #64 raises three items: (1) the "incorrect" framing; (2) the erratum is not in MANIFEST.json; (3) the mode of convergence is unstated. It gets no reply. | PROVEN |
| 19:17:12Z | Branch `grok/required-math-20260926` is created at `66e39d18`. It never gets commits of its own. | PROVEN |
| 21:37:44Z (client) | Grok packet `SECOND_PASS.md` (main PR #163, commit b785174), lines 24–30, reads the parent sentence as K = D⁻¹HD⁻¹. That is the correct reading. | commit date CIRCUMSTANTIAL |
| 21:42:17Z (client) | main PR #163 `D1_INTERFACE_TABLE.md` (8770393, header "round 3"): "Merge-or-rebase Math- PR64 onto Math- default". | commit date CIRCUMSTANTIAL |
| 21:42:30Z | Grok comment 5850139585 on PR #64, posted via the grok-by-xai app. It says "is the wrong factor" and "Until it is in-tree, default-branch readers … will miss the correction". | PROVEN |
| 21:44:17Z | GitHub's update-branch action produces merge commit `1d9b64ba` (committer GitHub; PushEvent 21:44:18Z). | PROVEN |
| 21:44:32Z | verify-imports succeeds on `1d9b64ba`. | PROVEN |
| 21:45:00–21:45:37Z | math-downstream-gates, the only required check (strict), succeeds on `1d9b64ba`. | PROVEN |
| 21:45:17Z | main #63 comment 5850158114 says "still 404s on Math- default" and "Not a Math- merge". Attributing it to Grok is circumstantial. | PROVEN (text) |
| 21:45:32Z | Branch `grok/congruence-erratum-on-default-20260926` is created at `bebe09d2`, PR #64's base. | PROVEN |
| **21:45:54Z** | PR #64 is marked ready_for_review by the shared account. | PROVEN |
| 21:46:02Z | Codex bot summary comment ("Draft marked ready"). | PROVEN |
| **21:46:22Z** | PR #64 is squash-merged as `d8f55054` by the shared account, with a merge-time custom message. | PROVEN |
| 21:46:24Z | PR #64's head branch is deleted. | PROVEN |
| 21:47:12Z | The Codex review that the undraft triggered finishes, 50 s after the merge. It adds a +1 at 21:47:16Z and no findings. | PROVEN |
| 21:48:00Z (client) | Grok commit b0f6e4e cites "merge d8f5505". | commit date CIRCUMSTANTIAL |
| 21:49:52Z | Branch `grok/proof-index-erratum-pointer-20260926` is created. | PROVEN |
| 21:50:00Z (client) | main PR #163 `INVENTORY.md:48` (d95fe28): "main #144 says keep Math #64 unmerged; that instruction is obsolete as custody." | commit date CIRCUMSTANTIAL |
| 21:51:39Z | Branch `custody/proof-index-erratum-pointer-20260926` is created at `d8f5505`. | PROVEN |
| **21:52:53Z** | PR #85 is opened via the grok-by-xai app. | PROVEN |
| 21:53:04–21:54:21Z | verify-imports, fail-closed-landing, math-downstream-gates and exact-replay all succeed on `ff36667`. | PROVEN |
| 21:53:13Z | PR #85 is marked ready_for_review, 20 s after opening. | PROVEN |
| 21:55:17Z | The Codex review of PR #85 completes (+1 at 21:55:21Z, no findings). | PROVEN |
| **21:56:15Z** | PR #85 is squash-merged as `5ed3b455` by the shared account, with a custom message. The head branch is deleted at 21:56:18Z. | PROVEN |
| 21:58:35Z | Grok round-9 comment 5850251715 on main #63: "Math- default is now 5ed3b455 (PR85 squash)". | PROVEN (text) |

Intervals between events:
- AMEND to merge: 3 h 39 m 20 s.
- Grok comment to merge: 3 m 52 s.
- Ready to merge: 28 s.
- PR #85 open to merge: 3 m 22 s.
- PR #64 merge to PR #85 merge: 9 m 53 s.

### 1.3 What the record proves and what is inferred

| Statement | Basis | Evidence |
|---|---|---|
| The shared account marked PR #64 ready, squash-merged it and deleted its head. It also merged PR #85. | PROVEN | Issue timeline events, `merged_by`, PullRequestEvent/DeleteEvent. |
| Which agent pressed update-branch, ready and merge on PR #64, and merge on PR #85. | UNKNOWN | `performed_via_github_app` is null for these event types for every lane. That includes Claude-merged #59 and Codex-merged #83, #84 and #86. |
| The Grok lane updated, undrafted and merged PR #64. | CIRCUMSTANTIAL (strong) | A grok/ branch appeared 22 s before the undraft. The squash title carries the "(scientific effect NONE)" suffix, which on Math- appears only on baee1ab, 8657130 and d8f5505, and in main only on Grok branches (18/18). These are the only two squash merges on Math- main. PR #163 wrote "Merge-or-rebase Math- PR64". No other agent lane commented in either repo between 21:39:41Z and 22:10Z. Counter-signal: comment 5850158114 at 21:45:17Z said "Not a Math- merge". |
| The same actor merged PR #64 and PR #85. | CIRCUMSTANTIAL | Both are the only squash merges, both use the same custom-message style without "(#N)", and they are 9 m 53 s apart. |
| The squash messages were written at merge time, not taken from the PRs' commits. | PROVEN (settings read on 2026-09-27, not at merge time) | Repo defaults are COMMIT_OR_PR_TITLE / COMMIT_MESSAGES. The actual messages match no commit message, PR title or PR body. |
| PR #85 came from a Grok-lane branch, and its content was written by the Grok lane. | PROVEN (created via grok-by-xai) / CIRCUMSTANTIAL (content) | PR creation app; branch name; object id `GROK-HEAVY-CAP-PAIRING-20260926-v1`. |
| The Grok lane knew of #144's "keep Math #64 unmerged" before the merge. | UNKNOWN | INVENTORY.md:48 calling #144 "obsolete" is client-dated after the merge. |
| No lane has replied to AMEND 5848593803 or posted a merge receipt. | PROVEN | The PR #64 comments after 18:07:02Z are Grok 5850139585, which does not cite the AMEND, and the Codex bot. `get_reviews` is empty. |

### 1.4 Were the merges sound?

**Ruleset:** compliant. Ruleset 24045351 requires a PR, 0 approvals, resolution of review threads, and math-downstream-gates (strict).
- PR #64: the required check passed on the up-to-date head `1d9b64ba` before the merge, and there were 0 review threads.
- PR #85: all four checks passed before the merge.
- `docs/PUBLIC_SHOP_SETUP.md` (last changed 18:02Z) describes the same ruleset. The ruleset itself was read on 2026-09-27.

**Bytes:** faithful.
- `d8f5505`'s tree equals `1d9b64ba`'s (723baf87…), and its diff against `bebe09d` is one file, +44 lines.
- `5ed3b45`'s tree equals `ff36667`'s (257bf98e…).
- The erratum has the same blob at `d573b99d`, `1d9b64ba`, `d8f5505`, `5ed3b455` and `03333792`: 1782 B, sha256 `bad7ef609c4ad8c41ad6af562c1b6807921e19a9d556ed793ad1a0db6e202028`, blob `213594d6ca6a86fb938110f4d166d9ce275a02d0`. The synthesis re-hashed all five commits.

**Soundness:** not sound.
- An explicit technical AMEND was merged over without any reply, and none of its three items has been addressed on default at `03333792`.
- The erratum is a required part of the D1 A3 target (PROOF_INDEX.md:32, "must include"), yet MANIFEST.json and verify.py still cover only the two parent files.
- PR #85, billed as custody, added two more unmanifested files in 3 m 22 s with no nonauthor technical review. It also replaced PROOF_INDEX's commit-pinned link to the erratum (`blob/d573b99d…`) with a mutable relative path.
- This departs from stated coordination norms, but it does not breach a binding rule. OP-AUTONOMY v2.1:28–29 requires no "original-author permission". main AGENTS.md:17–18 calls its coordination choices "revisable agent choices, not a new approval queue". main CONTRIBUTING.md:40 and #144 bind public-task contributors, not agent lanes.

**Mathematical effect:** none.
- The erratum algebra is correct: D_r H D_r = [[α, √r βᵀ],[√r β, A]] and det = det H / r, checked with sympy for m = 1, 2, 3.
- The parent bytes are unchanged, and nothing currently mismatches.

**Verdict:** PR #64 and PR #85 landed ruleset-compliant and byte-faithful, but not soundly. The load-bearing erratum is still outside machine custody, and which lane merged is circumstantial, not recorded.

### 1.5 Recomputed and confirmed
- Erratum bytes are identical across all five commits (see above). There was no amendment before or after the merge.
- Mutation test on a scratch copy of `imports/lifetime_parent_20260925` at `03333792`:
  - `verify.py` returns rc=0 after changing the erratum's "is incorrect" to "is correct" (sha256 d4e22437…), after appending to ERRATUM_POINTER.md, and after deleting CAP_PAIRING_IDENTITIES.md.
  - It fails only when a manifested file changes: parent 40263 ≠ 40261; MARKED_CYLINDER 15161 ≠ 15160.
  - The MANIFEST.json (4fd589c4), README.md (2d7a60b4) and verify.py (74235319) blobs are unchanged from `bebe09d` to `03333792`. MANIFEST paths are exactly the two parent files (re-checked).
- PROOF_INDEX.md:32 at `5ed3b45`/`03333792` ("landed on Math- default by PR64 at d8f5505…") is accurate.
- ERRATUM_POINTER.md's blob claim (213594d6) and CAP note's cap-blob claim (0633aca3, sha256 0bf922b9… = MANIFEST) are correct.
- main PR #163 body "Math- default now has the congruence erratum (PR64 @ d8f5505) and PR85 … (@ 5ed3b455)" is accurate.
- `grok/congruence-erratum-on-default-20260926` (0 ahead / 12 behind main), `grok/required-math-20260926` (0/16) and `custody/proof-index-erratum-pointer-20260926` (0/11) have no unique commits.

### 1.6 Findings (blockers first)

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| P1 | major | CONFIRMED | Math-:`imports/lifetime_parent_20260925/MANIFEST.json`:0; PR #64, PR #85 | **Combined effect of the merge sequence.** P2 and P3 are the individual parts; two of three verifiers rated the combination major. (a) PR #64 was merged 28 s after the undraft, 3 m 52 s after Grok comment 5850139585, and 50 s before the Codex review the undraft triggered. The AMEND's content items were not done and got no reply. The comment dropped the AMEND's condition "After amending, undraft and update-branch". Its premise was partly false: PROOF_INDEX:32 on `bebe09d2` already linked the erratum, pinned at `d573b99`. (b) PR #85 then added two agent-written, non-mirror notes to the Drive-mirror directory and made PROOF_INDEX's erratum link mutable. (c) Three default-branch files are now outside the only CI custody gate: the load-bearing erratum and two files PROOF_INDEX links or places beside it. No bytes have drifted; the risk is silent future edits. Who merged is UNKNOWN; Grok involvement is CIRCUMSTANTIAL. No binding rule was broken. | Post an item-by-item disposition of 5848593803 on PR #64. Land an additive custody PR with MANIFEST records under a non-mirror disposition, or move the notes out of `imports/`: ERRATUM_CONGRUENCE.md 1782 B `bad7ef60…2028`; ERRATUM_POINTER.md 590 B `e29e7733…07b3`; CAP_PAIRING_IDENTITIES.md 2112 B `69ff3a12…a23e`. Fix the README text and restore an immutable pin at PROOF_INDEX:32. Norm going forward: no merge over an unanswered AMEND, whichever lane merges. |
| P2 | minor (adj. from major) | PLAUSIBLE | Math-:PR #64 comment 5848593803:0 | PR #64 was undrafted and merged 3 h 39 m 20 s after an explicit AMEND that got no disposition. None of items 1–3 is addressed at `03333792`: line 19 still says "incorrect", MANIFEST has two files, line 35 gives no mode. This departs from the soft norm "coordinate overlapping work through existing PR discussions" (main AGENTS.md:14–18); it is not a breach of a binding rule. "Another lane's draft" rests on the PR having been created via chatgpt-codex-connector; the identity of the merging lane is inferred. Consequences were limited: three nonauthor checks agree the algebra is right, and the Codex review found nothing. | The merging lane, or any lane, posts a disposition. Handle AMEND items in a successor. |
| P3 | minor (adj. from major) | CONFIRMED | Math-:`imports/lifetime_parent_20260925/MANIFEST.json`:4 ("purpose") | The standing custody gap. The directory holds 6 .md files (5 without README). MANIFEST and README cover 2. The green verify-imports runs on PR #64 (`1d9b64b`) and PR #85 (`ff36667`) checked none of the files those PRs changed. MANIFEST/README text is scoped to Drive mirrors (drive_id fields) and is accurate for the two files it lists. The defect is an omission plus packaging, not a false custody claim. README:10 "both files are ASCII-only" is still true of the two mirrors but ambiguous in scope. Mitigation: blob 213594d6 is recorded in ERRATUM_POINTER.md:11 and in PR comments, but nothing checks it. Precedent for a fix exists: `imports/hardening_ebedb780/MANIFEST.json` lists a TRANSCRIPTION record. | As P1. The MANIFEST `note` needs rewording, because an erratum record has no Drive source. |
| P4 | minor | CONFIRMED | Math-:`imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md`:19 | "that displayed congruence factor is incorrect" overstates the problem. Parent line 169 ("the congruence uses diag(sqrt(r),I), so its off-diagonal entries are sqrt(r) beta_i") can only be read as H = DKD. Its own "so" clause fails under K = DHD, which would give off-diagonals r^{3/2}β. That is the convention in Math- `frontiers/remote_window_20260924/PROOF.md:134`. There is also a notation clash: the erratum's D_r = diag(r^{-1/2}, I) is the inverse of remote_window's D_r. | Additive successor note: "convention-ambiguous; the scaled matrix is D⁻¹HD⁻¹ = D_r H D_r"; cite remote_window (9)–(10). |
| P5 | minor | PLAUSIBLE | Math-:`…/ERRATUM_CONGRUENCE.md`:35 | The mode of convergence is not stated, and "bounded beta_i" is imprecise: (5.1) gives ‖β_i‖ ≤ M3/2 with M3 random, so β_i is tight, not bounded. The parent says "in probability uniformly" (line 165) but proves only convergence in law. Convergence in law is enough for (5.4), using Slutsky, uniform integrability from (5.3) and continuous mapping. So this is a precision gap; no stated result is wrong. | State "in law" (or name a regression coupling) and "√r β_i → 0 in probability via M3 moments". |
| P6 | minor | CONFIRMED | main:`docs/PUBLIC_SHOP_SETUP.md`:35; main #144 | Two standing texts still name Math #64 as not to be merged. The "obsolete" rationale exists only in open draft main PR #163 (INVENTORY.md:48, d95fe28). #144 has zero comments. The pin in #144 to `d573b99d` does little harm, since the blob is identical. | Update #144 and line 35, and post the rationale on #144. |
| P7 | minor | CONFIRMED | Math-:PR #64 / PR #85 | Neither merge has a receipt, so on a shared account provenance cannot be reconstructed. Other lanes do post receipts: Claude 5850573604 on #59, Codex 5850117419 on #84 and the main #86 note. The duty breached is the general recording rule (main AGENTS.md:21). PUBLIC_SHOP_SETUP.md:31 concerns reviews, not merges. | The merging lane posts provider-identified receipts: head SHA, check-run IDs, reason, and how the AMEND was handled. |
| P8 | minor | CONFIRMED | Math-:PR #85 | Open for 3 m 22 s and draft for 20 s. The only check of content was the automated Codex pass, a +1 with no findings. No nonauthor technical review exists of CAP_PAIRING_IDENTITIES.md or the PROOF_INDEX edits. This departs from a soft norm: the ruleset requires 0 approvals, and CODEOWNERS routes PROOF_INDEX.md to the shared account with no code-owner approval required. | Obtain and record a nonauthor review. The synthesis-side exact checks (section 2) found the arithmetic correct. |
| P9 | nit | PLAUSIBLE | Math-:`d8f55054…` commit message | The permanent message for another lane's erratum was written at merge time, with no "(#64)" and no attribution. The head was deleted 2 s later, so `d573b99d` is reachable only through `refs/pull/64/head`. GitHub still links the PR through merge_commit_sha. The claim that the style "matches only Grok" was overstated: the text paraphrases PR #64's body, and custom titles without "(#N)" are normal on Math- (Cursor PR63 → 1e1114f, ChatGPT PR66 → 10e1f19). | Use the default squash message, or add "(#N)" and a lane line. |
| P10 | nit | CONFIRMED | Math-:`refs/heads/custody/proof-index-erratum-pointer-20260926` (plus `grok/congruence-erratum-on-default-20260926`, `grok/required-math-20260926`) | Three empty branches, each an ancestor of main. `grok/required-math-…` predates the PR #64/#85 sequence by about 2.5 h. Stale branches are common across the repo. The Grok attribution rests on prefix and timing. | Delete them after confirming no lane intends to use them. |

(Provenance items about the PR #85 diff description and packaging are in section 2 as C5 and C7. The PR #163 packet's stale PR #64 state is in section 5 as X3.)

### 1.7 Follow-up actions
1. Post a disposition on PR #64 answering AMEND 5848593803 item by item. Record merge receipts on PR #64 and PR #85 (P1, P2, P7).
2. Land an additive custody PR: MANIFEST and README records for the three non-mirror files, or relocate them, plus an immutable erratum pin at PROOF_INDEX:32 (P1, P3).
3. Land a successor erratum note covering the convention-ambiguity framing and the mode of convergence (P4, P5).
4. Update main #144 and PUBLIC_SHOP_SETUP.md:35 (P6).
5. Get a nonauthor review of PR #85's derivation note and index edits (P8).
6. Delete the three empty branches (P10).

### 1.8 Not verified
- The merging actor for PR #64 and PR #85.
- Whether the ruleset and squash defaults were the same at 21:46Z. They were read on 2026-09-27.
- The governance- working contract; it was not in read scope.
- Push times of client-dated commits.
- Completeness of the events API (for example, no PushEvent was listed for `ff36667`).
- Coordination that happened off GitHub.

---

## 2. Math- PR #85 content (`5ed3b455`)

### 2.1 What it claims
PR #85 changes four files (+81 / −3, re-checked with `git diff --stat d8f5505 5ed3b455`):

| File | Size | What it adds |
|---|---|---|
| `PROOF_INDEX.md` | lines 11, 28 and 32 changed | Line 32 re-points the D1 A3 erratum to default. Line 28 adds an ASCII note pointer. Line 11 rewords one phrase. |
| `frontiers/full_price_20260924/ASCII_3E_MINUS_2.md` | 15 lines, 623 B | Says `3e-2` means `3*e-2`. |
| `imports/lifetime_parent_20260925/CAP_PAIRING_IDENTITIES.md` | 49 lines, 2112 B | Vector-average bound, h'' cancellation, section 7 r^5 skeleton. |
| `imports/lifetime_parent_20260925/ERRATUM_POINTER.md` | 14 lines, 590 B | Pointer to the erratum. |

The PR body says "Custody / additive notes only … Do not treat merge as scientific promotion."

### 2.2 Recomputed and confirmed (exact unless labelled)
**CAP_PAIRING_IDENTITIES.md**
- Line 15: ∫_{−r/2}^{r/2}\|2r − t\| dt = 2r².
- Line 17: ‖w′‖ ≤ 2mr on \|x\| ≤ 2r. The bound is sharp: w = (m/2)(x² − r²/4) gives w′(2r) = 2mr with zero pin average.
- Line 19: (m/2)\|x² − r²/4\| ≤ (15/8)mr², matching cap (6).
- Lines 23–33: the h″ cancellation F″ = D³f[v,v,v], (H_f v)_y = 0, D²f[v,v′] = 0 and Df[v′] = 0, checked symbolically for general f in d = 3. It matches cap (11).
- Line 34: u(11) = 24/121 and m·u = 48/121. Adverse term 2554128/1771561, so 7/4 − adverse = **2184415/7086244 > 1/4**, with margin 206427/3543122. Re-checked by the synthesis with `fractions.Fraction`.
- Lines 40–41: ∫_0^{DrU²} ℓ(ℓ + ErU) dℓ = r³[(D³/3)U⁶ + (ED²/2)U⁵], equal to parent (7.3).
- Line 43: r² · r³ / Θ(r²) = O(r³), consistent with parent (5.5)/(7.8) for the m ≥ 2 depth-failure part.
- Line 5: the cap blob 0633aca3 is bound correctly.

**ERRATUM_POINTER.md**
- The old-text claim, the `d573b99` link, merge `d8f5505` (committer time equals `merged_at`), blob 213594d6 and "parent unedited" (parent blob dfed3b8d, sha256 9350ad6e…) are all true.

**ASCII_3E_MINUS_2.md**
- −log[3e^{−2} − 2e^{−3}] = 3 − log(3e − 2) holds exactly.
- The reading 3·exp(−2) gives 5 − log 3.
- The intended reading is fixed by PROOF.md:95: e² − (3e − 2) = (e − 1)(e − 2) holds only for 3·e − 2.
- The ρ* window (0.84547981724898672067, …068) reproduces as an exact outward Fraction enclosure. The mpmath value is NON-CERTIFYING.
- RESULTS.json reproduces byte for byte, 36 tests pass, and the PROOF.md pin 87521901… matches.

**Other**
- `5ed3b45`'s tree equals `ff36667`'s.
- The P15 PROOF.md bytes are unchanged.

### 2.3 Findings

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| C1 | minor | CONFIRMED | Math-:`frontiers/full_price_20260924/ASCII_3E_MINUS_2.md`:5 | The note covers only (F1) and (F11). The bare token `3e-2` also appears in: PROOF.md:95 and :139; RESULTS.json:3; full_price.py:174; README.md:17; claims/LANDING_CLAIMS.json:333; reviews/p15_full_price_nonauthor_20260926/REVIEW.md:115, 175, 177; algebra_check.py:243; main STATUS.md:14, docs/RESEARCH_INDEX.md:51, docs/site/museum.json:458 and docs/site/status.json:41. None of these point to the note. The note also leaves out its decisive evidence, the factorization at PROOF.md:95. | List every occurrence, cite PROOF.md:95, and add pointers beside the README and LANDING_CLAIMS rows. Any STATUS wording is the status owner's call. |
| C2 | minor | CONFIRMED | Math-:`imports/lifetime_parent_20260925/CAP_PAIRING_IDENTITIES.md`:45 | The "still needs" list is incomplete. It leaves out: the (7.7) fourth-derivative exception (every m); the m = 1 near branch (7.5) as well as the far branch (7.6), because the parent says using (7.3) for m = 1 "would be invalid"; and the reviewed inputs (3.5) [A2], (4.3) [A4] and (6.1)/(6.2) [A5/A6]. The line-38 route is valid only for m ≥ 2. | Label the section 7 block "m ≥ 2, depth-failure part only" and complete the list. |
| C3 | minor | CONFIRMED | Math-:`…/CAP_PAIRING_IDENTITIES.md`:49 (and :45) | Presents the pathwise pairing implication (§1 embedded chart, §8 Morse/distinct values) as outstanding. The D1-A review (main #63 5841570965) ACCEPTED exactly that interface. PROOF_INDEX:32, edited in the same PR, cites that acceptance next to the link. This is an under-claim and an internal inconsistency. If line 45 meant the A7 composition (7.8), that step is genuinely unreviewed and should be named A7. | Reword to "depends on §1/§8, both accepted at D1-A (main#63 5841570965); not re-reviewed here", and name any remaining item explicitly. |
| C4 | minor | CONFIRMED | Math-:`…/CAP_PAIRING_IDENTITIES.md`:3 (also ASCII_3E_MINUS_2.md:1–5, ERRATUM_POINTER.md:1–3) | None of the three new files has an author, provider or session line. The PR body has none either, and the commits carry the shared account's git identity. Only the CAP file carries a provider-bearing object id. The squash message names neither Grok nor #85, so a plain clone attributes two of the three files to no one. The lane's own PR #80 file does carry an author line. | Add an author line to each file (xAI/Grok lane, session, "nonauthor of the parent/cap/P15 proofs"). |
| C5 | minor | CONFIRMED | Math-:`PROOF_INDEX.md`:32 (and :28) | A mostly re-derivational note is filed in a source-custody import directory. Its only new content is the "sharp" remark and the section 7 integral. PROOF_INDEX indexes it with no "author-side / unreviewed" label, although neighbouring line 33 has one. An unreviewed ASCII note is appended to a "Reviewed scoped results" bullet (line 28). The ERRATUM_CONGRUENCE.md precedent started this drift; PR #85 continued it. The custody side is P3. | Label both links "author-side note, unreviewed; scientific effect NONE". Move the CAP note out of `imports/` or give it a non-mirror disposition. |
| C6 | nit (adj. from minor) | CONFIRMED | Math-:`…/ASCII_3E_MINUS_2.md`:11 | Names 3·exp(−2) as the false reading, but not the programming-numeral reading `3e-2` = 0.03 (Python, C, JavaScript, JSON), which gives h = 3 + log(100/3) ≈ 6.5066 and ρ ≈ 0.1537. Lines 8 and 13 implicitly guard against misreading. The risk is copy-paste from PROOF.md or RESULTS.json, whose formula string `rho=1/(3-log(3e-2))` evaluates to 0.1537 in Python. | Add one line on the 0.03 reading. |
| C7 | nit | CONFIRMED | Math-:`PROOF_INDEX.md`:11; PR #85 body; `5ed3b45` message | The change list is incomplete. PROOF_INDEX lines 11, 28 and 32 changed, which is four sentence-level edits, against "except the index sentence" in the singular. The line-11 rewording "not original bytes" → "not original source bytes" is disclosed nowhere: not in the body, the squash message, the `ff36667` message or any comment. The squash body leaves out ERRATUM_POINTER.md, which the PR body does list. The protected files (parent, cap, P15 PROOF.md) are byte-identical. | Record an accurate change list in a receipt comment. |
| C8 | nit | CONFIRMED | Math-:`frontiers/full_price_20260924/SOURCE_FILES.json`:2 | The ASCII note is an unpinned file inside a pinned payload directory (5 payloads plus the catalog), so exact-replay `verify()` never checks it. `run_validation.py` hashes every file in ROOT, so REPORT.json's source_sha256 map gains a key. The other six digests are unchanged; the effect is cosmetic. | Pin the note and update the scope string, or keep it outside the payload directory. |
| C9 | nit | PLAUSIBLE | Math-:`…/CAP_PAIRING_IDENTITIES.md`:5 | Binds the cap source by blob but cites the parent only as "parent §7", with no blob or sha, and does not mention the erratum next to the A3 dependency. D, U and E are undefined. E is also undefined in the parent at line 236, where it clashes with the expectation operator; it is implicitly 3K/2. The filename, commit subject and PROOF_INDEX link text ("pairing identities") overstate the content; the H1 title is accurate. | Cite the parent path, blob and sha, define D, U and E (renaming E), point to the erratum, and rename to "cap ridge/average identities and §7 skeleton". |

**Verdict:** AMEND (minor). Every mathematical statement in `5ed3b455` reproduces exactly, and there is no blocker. The defects are coverage, custody, attribution and dependency-list wording.

### 2.4 Not verified
- The full deterministic cap theorem and the parent's A2 inputs; these are under the open A1–A7 review.
- CI log contents: only the conclusions were read.
- Any finding content from the Codex review, which said only "Completed".
- The alternative-reading ρ enclosures used the package's own log/exp routines. The independent check was mpmath only (NON-CERTIFYING).

---

## 3. Math- PR #87 cycle-4 packet (Harper / Benjamin / Lucas)

**Status at review time (PROVEN):**
- Head `0fab933`, 6 commits, 9 files, non-draft since 2026-09-27T00:44:47Z, base `03333792`, `mergeable_state` blocked.
- One check, math-downstream-gates, succeeded on `0fab933`.
- 3 Copilot COMMENTED reviews, all on `8657130`, with 10 threads: 1 resolved, 9 unresolved.
- A Codex usage-limit notice (5851367595).
- No nonauthor technical review.

**Attribution:** the Grok lane's authorship is CIRCUMSTANTIAL. It rests on self-identification in commit subjects, paths and the team note ("Grok + Harper + Benjamin + Lucas"); every commit is by the shared account.

**What the packet claims:**
- The PR body and team note 5851042294 (23:51:20Z, never edited) claim exact planar Bargmann–Fock six-pin closed forms, a 4-pin α series, and a reduced 4-slot frame.
- They claim a D5 "obstruction ledger" saying the cone product is O(k³r⁵q²) and "Math-#58 O(r⁶q²) over-slaves det H_S".
- They claim a SARD-G A1 relative-interior repair lemma with openness, and a Lucas scorecard "P6 SATISFIED … P5 PARTIAL" against the successor source.
- Everything is marked "Scientific effect: NONE".

### 3.1 Closed forms (`harper/CLOSED_FORMS_FTS_FTT.md`, `harper/test_closed_forms.py`)
**Confirmed (exact sympy, kernel exp(−\|d\|²/2)):**
- Var(f_ts(M) \| six pins) = (e^{r²} − 1 − r²)/(e^{r²} − 1).
  - The only surviving cross-covariance is r e^{−r²/2}.
  - Series: r²/2 − r⁴/12 + 0·r⁶ + r⁸/720 − r¹²/30240, re-run by the synthesis. The r⁶ and r¹⁰ coefficients are exactly 0.
  - Var(β) → 1/2.
- Var(f_tt(M) \| pins): the 4-pin and 6-pin values are equal, and the displayed rational in E = e^{r²} is exact.
  - Series: r⁴/6 − r⁶/30 + r⁸/360 − r¹⁰/12600 − r¹²/75600.
  - det G = (1 − e^{−r²})² − r⁴e^{−r²}, and the cross vector (−1, 0, (r² − 1)e^{−r²/2}, r(3 − r²)e^{−r²/2}) is exact.
  - The rational has no pole: its denominator is negative for every r > 0.
- The 7 unittest tests pass at the head (stdlib only).

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| H1 | minor | CONFIRMED | Math-:`incoming/grok-cycle4-20260926/harper/CLOSED_FORMS_FTS_FTT.md`:80 | "the first differing term is `r^6`, so the small-`r` denominator is `~ r^6/12`" has the wrong order and the wrong sign. The paragraph's own expansions agree at r⁶. In fact r⁴E − (E − 1)² = −r⁸/12 − r¹⁰/12 − 2r¹²/45 + …, re-run by the synthesis, and the numerator is −r¹²/72 + …. "ratio `1 != 1`" is garbled. The closed form, the series and the r⁴/6 limit are unaffected. The same error is Copilot thread r4113530261. | Replace with "den = −r⁸/12 + O(r¹⁰) (< 0 for r > 0), num = −r¹²/72 + O(r¹⁴)". |
| H2 | minor | CONFIRMED | Math-:PR #87 comment 5851042294 (item 4) | "Float64 evaluation … cancels at `r ≲ 0.05`" understates the breakdown. Evaluating the rational in float64, as `var_ftt` does, gives relative error >1e−4 up to r ≈ 0.16, >1e−2 up to r ≈ 0.109 (1.7% at 0.10, 7.4% at 0.08). For r ≲ 0.075 it returns 0 or unrelated values, up to about 31× wrong. | "Use the series (or extended precision) for r ≲ 0.15–0.2; float64 is meaningless for r ≲ 0.075." |
| H3 | minor | CONFIRMED | Math-:`…/harper/test_closed_forms.py`:83 (lines 83–88) | The tests cannot pin the coefficients they back. The r⁶ test does not divide by r⁴, despite its comment. At r = 0.08 it rejects c6 only when \|c6\| > 0.0122, beating 1/48 by a factor of 1.7. The r⁸/720 term is never distinguished from 0. The f_tt series test agrees to ≤ 7.9e−5 against a tolerance of 2e−2 and passes with c8 = c10 = 0, so neither 1/360 nor −1/12600 is tested. No test builds the pin Gram or its Schur complement. The coefficients themselves are correct. | Normalize by r⁴ over shrinking r. Use exact (sympy/Fraction) series, and add one Gram/Schur test. |
| H4 | minor | CONFIRMED | Math-:`…/harper/README.md`:5 | "certified by direct covariance calculus" has no certificate behind it. The only script is a float self-series comparison, and line 19 says "Same-session corroboration only". The md shows a complete hand derivation for TS (lines 23–35) but only "yields" for the TT Schur step, and its one worked TT step (line 80) is wrong (H1). Team-note item 1's "12-jet max abs error 8.8e-14" has no artifact anywhere. | "derived by same-session covariance calculus; exact check not shipped", or ship an exact replay. |
| H5 | nit | CONFIRMED | Math-:`…/harper/test_closed_forms.py`:19–21 at `77d1a03` | The first commit shipped a wrong series (r⁶/48 and −r⁸/180; the true values are 0 and +1/720) and a failing test (rel 0.0118 > 5e−3 at r = 0.75). `d20ce86` fixed it 30 s later by client dates. The first workflow run is on `d20ce86`; that `77d1a03` was never pushed alone is an inference. The md never contained the wrong series. | None needed; noted for provenance. |

### 3.2 Alpha series (`benjamin/ALPHA_4PIN_SERIES.md`)
**Confirmed exact:**
- E[α_M \| 4 pins] + 6k = k r² − b r/4 + b r³/24 − k r⁴/20 − b r⁵/192 − k r⁶/120 + b r⁷/5760 + 71k r⁸/33600 + ….
- The 4-pin and 6-pin means are equal.
- "+31k/240" is wrong.
- The Hermite cubic p(M+z) = b − 3kr z² + 2k z³ meets all four pins, with p″(0)/r = −6k for every r.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| A1 | minor | PLAUSIBLE | Math-:`incoming/grok-cycle4-20260926/benjamin/ALPHA_4PIN_SERIES.md`:9 | The 11-line file defines neither α_M, the kernel nor the pin values (they are implied only by line 11), and gives no derivation or script. The "Lucas independent (b,k)-split grid" is shipped nowhere, and harper/README.md:19 and the team note call it same-session. The warrants under-state the result: every coefficient is exact algebra, not "DERIVED-numeric", and +b r⁷/5760 has no label. The inventory gap was partly closed by the PUBLIC_READING_MAP.md at `db66b25` (see K1). | Add the pin definitions, the closed-form rational and a sympy script. Relabel all coefficients as exact, and drop "independent". |
| A2 | nit | CONFIRMED | Math-:PR #87 comment 5851042294 (item 3); ALPHA:10 | "+31 k r^6/240 KILLED (wrong sign)": the value is wrong in magnitude too. The true coefficient is −1/120 = −2/240, a factor of 15.5. −1/120 and −1/192 are exact, not "DERIVED-numeric". The +31/240 value itself is recorded nowhere in-tree. | "+31/240 wrong (true coefficient −1/120, exact)." |

### 3.3 Reduced frame (`benjamin/REDUCED_FRAME_4SLOT.md`, comment 5851065560)
**Confirmed:**
- Exact identities:
  - f_ss + f, f_sss + 3f_s and f_tss + f_t are each orthogonal to all six pins, with variances 2, 6 and 2 for every r.
  - Hence Var(L \| pins) = Var(f_sss/2 \| pins) = **3/2**. The unconditional values are Var(f_sss) = 15 and 15/4.
  - The 4-slot conditional Gram at M is exactly diag(Var_tt, Var_ts, 2, 3/2), with det = 3·Var_tt·Var_ts = r⁶/4 − 11r⁸/120 + ….
  - Cov(f_sss(M), f_sss(S) \| pins) = 6e^{−r²/2}.
- mpmath values (NON-CERTIFYING):
  - Measured dets 5.9707e−6 (r = 0.17), 1.76309e−4 (0.30) and 0.17058 (1).
  - At X = (0, 0.15), r = 0.3: det 6.14687e−3 and minimum eigenvalue 1.85415e−2.
- Search result: no hits for "Table 4.1", "Condition (ND)" or "15-frame" on Math- `55a3ced`/`0333379` or on main `dbb9dcf`.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| R1 | minor (adj. from major) | CONFIRMED | Math-:PR #87 comment 5851042294 (item 5) | Under "Closed on planar BF, d=2, six-pin only" it gives `det Gram(W_t,W_s) = 45 z_s^8/16` and `Var(L)=15/4` after the `f_tss` pin. Both are correct only as unconditional values (15/4 is conditional on f_tss alone). With six pins they are (3/4)z_s⁸ and 3/2. The heading is the PR's scope firewall (item 3 is a 4-pin result), and the source it drew from kept the word "unconditional". The file of record corrected Var(L) about 4 minutes later (`f3af9b5`, 5851065560), but no PR #87 text corrects 45/16 to 3/4; that value appears only in main PR #163 CLOSED_BF_WTWS.md:19. The comment was never edited (created = updated 23:51:20Z). | Add "unconditional" to both numbers, or put the six-pin values (3/4)z_s⁸ and 3/2 beside them. |
| R2 | minor | CONFIRMED | Math-:`incoming/grok-cycle4-20260926/benjamin/REDUCED_FRAME_4SLOT.md`:47 | "For \|z_s\|>0 the minimum eigenvalue is larger than the on-axis value at the same \|z_t\|" appears under "Warrant: PROVEN", and comment 5851065560 repeats it. It is false, including between M and S. At r = 1, z = (0.4, 0.05) gives 0.0279219 against 0.0309359 on-axis; there are 357 grid violations in total. One verifier reports an interval-arithmetic certification at the file's own r = 0.3: z = (0.14, 0.003) gives 2.6265e−4 against 2.8647e−4 on-axis. The other values are 50-digit numerics (margins 3–10%). Restricting the claim to "tested points" would still be false. Positive definiteness is unaffected. | Delete it, or replace with "the conditional Gram stays PD at tested off-axis points (NON-CERTIFYING); λ_min can drop by about 10% off-axis." |
| R3 | minor | CONFIRMED | Math-:`…/REDUCED_FRAME_4SLOT.md`:4 | One PROVEN warrant covers content of mixed status. Measured dets and "every tested r" have no script. Exact facts are stated only approximately ("diag ~"), although the Gram is exactly diagonal and PD for every r > 0. The f_tss step ignores the crosses −r e^{−r²/2} and (r² − 1)e^{−r²/2} with f(S) and f_t(S). The value 2 is right only through the residual f_tss + f_t, and the relevant parity is He₃(0) = 0, not He₁(0). | Split the warrant into PROVEN (the residual identities, the exact diagonal Gram, det = 3·Var_tt·Var_ts) and NON-CERTIFYING (the numerics), and redo the f_tss step with the residual. |
| R4 | minor | PLAUSIBLE | Math-:`…/REDUCED_FRAME_4SLOT.md`:5 (and :55) | Status drift. (1) "Theorem B remains PROVEN-MODULO repaired ND" changes the premise from "(ND)" (main PR #163 ND_CUBIC_SLAVING.md:63) to a repaired sentence. That repaired sentence exists only in words in the unmerged cycle-2 RESULT.md:12/:35, and ND_CUBIC_SLAVING.md:51 says it "is not in the public vault". (2) The unqualified "Theorem B" clashes with an already-published, different Math- "Theorem B (compact-mark density candidate)" at `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`:37. (3) Line 55's "refutation unchanged" goes beyond PR #163 ND_CUBIC_SLAVING.md:3/:51 ("not a certified kill"), inheriting cycle-2's "ND-as-written remains refuted". Lines 59–60 also call the 15-matrix ABSENT. "PROVEN-MODULO" and "repaired ND" have 0 hits on either default. | "File-1 Theorem B (author label, off-GitHub PDFs, not on STATUS.md) PROVEN-MODULO (ND); repaired-ND sentence proposed in incoming only, unreviewed." Replace "refutation" with "cubic-order rank deficiency at File 3's leading scaling (not a certified kill)". |
| R5 | nit | CONFIRMED | Math-:`…/REDUCED_FRAME_4SLOT.md`:42 | A literal TAB byte stands where `\to` was intended ("as r<TAB>o0"). Copilot thread r4113530331 reports the same. | "as r → 0". |

### 3.4 D5 obstruction ledger (`harper/D5_OBSTRUCTION_LEDGER.md`)
**File facts:** 164 lines, 7338 B, sha256 `1a500458…34c8`. Blob 69a6c549 is unchanged from `77d1a03` through `0fab933`. The file is absent from Math- main.

**Claims:**
- Lines 115–119: the gradient equations at X "do not constrain f_ss(S)", so det H_S = O(k r).
- Line 125: the product is O(k³r⁵q²).
- Section 5 and table row 160: #58's O(r⁶q²) is "a congruence error" / "over-slaves det H_S by copying the annulus".
- Line 149: a formal KR assembly gives an O(k r²) cone contribution.

**Confirmed:**
- Section 2's closed form (as in 3.1).
- Orientation: t is the axis and X − M = r(0, q) is transverse.
- The Jacobian of (β, S) → (f_t, f_s) is diag(r²q, rq), with determinant r³q².
- Slaving: sd(β)/q ≈ 1/√2 and sd(f_ss(M))/(rq) = √(3/2) (exact conditional).
- det H_M and det H_X are Θ(k r²\|q\|) on the axis. MC gives E\|det\|/(k r² q) = 5.80–5.90 against theory 5.863 (NON-CERTIFYING).
- Z_r = Θ(k²r²): typed MC gives Z_r/r² → 97.4 against the closed form z₀ = 97.925.
- Section 6's arithmetic is correct given its inputs.
- The quoted source powers are correct.

**Central recomputation:**
- Deterministic argument: on {grad f(X) = 0} with bounded C³/C⁴ jets, f_ss(M) = O(r\|q\|), so f_ss(S) = f_ss(M) + r f_tss(M) + O(r²) = O(r). This uses parent (5.1), ‖A_S − A_M‖ ≤ r M3.
- The synthesis re-ran an exact sympy degree-4 polynomial jet with the six pins and grad f(X) = 0 at X = M + r(0, q). det H_S has a zero r¹ coefficient and a generically nonzero r² coefficient. The triple product has a zero r⁵ coefficient and an r⁶ coefficient proportional to q².
- Exact moment upper bound: E\|det H_S\| ≤ 8.45 r² at r = 0.05.
- MC (NON-CERTIFYING, 2e5 samples per cell): E\|det H_S\|/r² = 6.19–6.76, flat in q (fit r^1.992 q^−0.005). E\|product\|/(r⁶q²) = 330–379 (fit r^5.994 q^2.023). E\|product\|/(k³r⁶q²) = 362–369 for k ∈ {0.25, 1, 4} and b ∈ {−1, 0, 1}.
- The Θ lower bounds rest on MC plus the exact coefficient being generically nonzero. The O(r²) and O(r⁶q²) upper bounds are deterministic.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| D1 | **blocker** | CONFIRMED | Math-:`incoming/grok-cycle4-20260926/harper/D5_OBSTRUCTION_LEDGER.md`:115 (115–119; table 158) | "The two gradient equations at `X` do not constrain `f_ss(S)` … `f_ss(S) = a_S = O_p(1)`, `det H_S = O(k r)`" is false on the Kac–Rice law (pins plus grad f(X) = 0). That law is the ledger's own conditioning for H_M and H_X in section 4. On it, f_ss(S) = O_p(r) and det H_S = Θ(r²): Θ(k r²) on p = 0, with a k- and p/q-dependent constant across the cone, where −f_ts(S)² adds about k²t²r². O(k r) is only a loose upper bound, but the ledger treats it as sharp (line 140), and its central verdict depends on that. The file is unchanged at head `0fab933`. | Replace with "f_ss(S) = f_ss(M) + r f_tss(M) + O(r²) = O_p(r); det H_S = O(r²) on {grad f(X)=0}". Carry the change into lines 125, 128, 130–142, 149, table rows 159–160 and comment 5851042294. |
| D2 | major | CONFIRMED | Math-:`…/D5_OBSTRUCTION_LEDGER.md`:140 (130–142; table 160–161) | The charge that #58's O(r⁶q²) "manufactures the extra r" / is "an error" is false. The on-axis cone product is Θ(k³r⁶q²) (upper bound deterministic, Θ from MC), which is #58's power and agrees with Cursor PR #69 NOTE.md:24. The ledger also misreads its source: TRANSVERSE_BOUND_CANDIDATE §4 slaves f_zz(M) and transports to H_S and H_X by Hessian Lipschitz control. It does not slave f_zz at S (line 132 is wrong). Precision: PR #28's bound (6) as displayed gives r M3/\|v\|, which is O(r/\|q\|) on the cone. The cone's O(r M3) needs the intermediate inequality with \|p\| ≤ \|q\|, as the AMEND review `4e188e25` REVIEW.md:50 states. OpenAI's P3 comment 5841824353 derived the r⁶ on the pin chart itself. No test of the det H_S power exists in the packet. | Withdraw section 5 and rows 160–161, and state that the on-axis product is Θ(k³r⁶q²), consistent with #58 and PR #69. |
| D3 | major | CONFIRMED | Math-:`…/D5_OBSTRUCTION_LEDGER.md`:149 (also 125, 128, 159, 164) | With the corrected product, the ledger's own formal assembly gives a density of Θ(k r) per physical area and an **O(k r³)** cone contribution, not O(k r²). MC confirms the typed density: ρ/r = 0.13–0.18, stable across r. This shares its root cause with D1, so track the two together. Line 164 says only that the ledger does not claim O(r³); O(r³) is #58's pin-neighbourhood target. The k³ prefactor holds for fixed k > 0 only. | Replace with "Θ(k³r⁶q²) on the slice; formal cone contribution O(k r³)", keeping the "not a proof" caveats. |
| D4 | minor | CONFIRMED | Math-:`…/D5_OBSTRUCTION_LEDGER.md`:76 (and :80) | "the factor `1/2` is `sqrt(det Cov(beta,S))`" is false on BF. With Var(β) → 1/2, Var(S) = 2 and Cov = 0, det Cov(β, S \| pins) = 2(e^{r²} − 1 − r²)/(r²(e^{r²} − 1)) → 1, and sqrt det Cov(grad f(X) \| pins)/(r³q²) → 1 (1.0000004 at r = q = 0.001). The 1/2 is the AMEND unit-covariance model's minor. Line 80: r⁵/(r³q²) = r²/q², so "exactly 2r²/q²" is off by a factor of 2. Also Copilot thread r4113530280. | "Constant 1 on planar BF with six pins; 1/2 belongs to the unit-normalized AMEND model." Fix line 80. |
| D5 | minor | CONFIRMED | Math-:`…/D5_OBSTRUCTION_LEDGER.md`:96 (and :118) | "−6kr + O(r³)" and "6kr + O(r³)": the true corrections are O_p(r²), with mean −b r²/4 and sd r²/√6. They enter det H_M only at O(r³\|q\|), so no power changes. This contradicts the same packet's ALPHA_4PIN_SERIES.md and CLOSED_FORMS. | Write "∓6kr + O_p(r²)". |
| D6 | minor (adj. from nit) | CONFIRMED | Math-:`…/D5_OBSTRUCTION_LEDGER.md`:156–157 (and :112) | The body (line 103) gives det H_M = O(k r²\|q\|) + O(r²q²). The table drops the second term, and the H_X derivation never subtracts f_ts(X)² = r²β². The k-factored forms, and the k³ in lines 125, 128, 149 and 159, need k ≳ \|q\|; they are not uniform as k → 0. | Write O(k r²\|q\| + r²q²) and carry it forward, or state k ≳ \|q\|. |
| D7 | nit | CONFIRMED | Math-:`…/D5_OBSTRUCTION_LEDGER.md`:62 (62–64; 87–88) | The Taylor remainder O(r³) divided by r²q gives O(r/\|q\|), which is O(1/κ) at the cone edge, so it does not give the stated O(r) slaving. The true remainder is O(r³\|q\|³), giving O(rq²) and O(r²q²). The conclusions stand. Line 106 can be sharpened to O(r²q²). | State the sharp remainders. |
| D8 | nit | CONFIRMED | Math-:`…/D5_OBSTRUCTION_LEDGER.md`:149 | The product is derived only on p = 0 (line 11) but integrated over the full cone area without an off-axis argument. Off-axis, the q-gain in det H_M comes from a leading-order cancellation. MC suggests the same r- and q-powers off-axis (NON-CERTIFYING), but the k-dependence is not established there. | Restrict to the slice, or add the off-axis computation. |
| D9 | nit | PLAUSIBLE | Math-:`…/D5_OBSTRUCTION_LEDGER.md`:29 (and :132) | "valid only for `\|v\| >= eta > 0` and `A <= \|s\| <= B`" misstates the reviewed domain. PR #16's chart is K = {A ≤ √(u² + v²) ≤ B, \|v\| ≥ η} in midpoint coordinates, and \|s\| is not that radius. PR #28's pathwise lemma assumes only \|u\|, \|v\| ≤ B and v ≠ 0. S is overloaded (pin point, the jet f_ss(M), and line 147), and R, u, v, w, η, A and B are undefined. That the overloading contributed to D1 is speculation. | Cite each source with its own domain, and rename the jet S. |

### 3.5 SARD-G A1 relative-interior lemma (`harper/SARD_G_A1_RELATIVE_INTERIOR_LEMMA.md`, blob 4a1812a2, 6,420 B)
**Confirmed:**
- The counterexample quotes match main `reviews/sard_g_successor_a1_a6_20260926/REVIEW.md` at `dbb9dcf` (blob 9fc61b47, 25,057 B, unchanged since `e8ec7af`):
  - F = sin(2πx)(2 − cos 2πy).
  - Σ = {(1, y) : 0 ≤ y ≤ 1/200}.
  - Sections of height 1/50 at x ∈ {73, 77, 123, 127}/100.
  - First hit at the endpoint (1, 0).
- Exact values: critical points (3/4, 0), (1/4, 0), (1/4, 1/2) and (3/4, 1/2), with Hessians a·diag(1, −1), a·diag(−1, 1), a·diag(−3, −1) and a·diag(3, 1) and values −1, 1, 3 and −3. cos(2π·77/100) = sin(π/25) exactly. 2π sin(π/25) ≈ 0.78749 > 12/25. Bound 23/48.
- The actual travel time is ≈ 0.0701 (NON-CERTIFYING quadrature).
- F fails RI1 and only RI1.
- RI1–RI4 faithfully restate review §5 (lines 161–165).

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| S1 | minor (adj. from major) | PLAUSIBLE | Math-:`incoming/grok-cycle4-20260926/harper/SARD_G_A1_RELATIVE_INTERIOR_LEMMA.md`:44 | "Then `U_chi` is open in the `C^2` topology" is stated with no conditions. The sketch (lines 48–49) covers only flow from fixed launch points. It never addresses how saddle branches, launch points and branch-label hits depend on f. That is the parameter-dependent invariant-manifold input the review files under A2 (review line 167; successor `a1fc958` lines 43, 59, 141). Line 77 says "No claim about A2–A5 is made here". The conclusion is classically true, the file is labelled a proof sketch, and A2 stays separately required for A6/C103 (review lines 171, 177), so nothing is promoted. Smaller points: "min\|grad f\|>η" is a generic hypothesis, and the arc bound comes from Condition 4. t* = t_max is a problem only if t_max is separate chart data (Copilot thread PRRT_kwDOUphbdM6mV_ds, unresolved). "All four … strict" is loose for RI2, whose robustness needs the compact-prefix separation argument. C¹ persistence at branch hits is unjustified but not shown false. | Make Lemma OPEN conditional on continuous dependence of the local invariant-manifold branches and their launch and section hits on f ∈ C² (A2). Tie t_max to Condition 4 or require t* < t_max. Spell out the compact-prefix argument. |
| S2 | minor | CONFIRMED | Math-:`…/SARD_G_A1_RELATIVE_INTERIOR_LEMMA.md`:71 | "`F_eps(x,y) = F(x, y+eps)` … misses `Sigma`. Thus … `F_eps notin U_chi`" drops the review's restriction 0 < ε < 1/200 (review lines 133, 140, 146). For −1/200 < ε < 0 the connection crosses at (1, \|ε\|), which is in relint Σ, so F_eps ∈ U_chi (exact check at ε = −1/1000). The non-openness conclusion survives on the one-sided family. | Write "for 0 < ε < 1/200 (small enough for Condition 1); one-sided". |
| S3 | minor | CONFIRMED | Math-:`…/SARD_G_A1_RELATIVE_INTERIOR_LEMMA.md`:38 | Line 38 applies "The same four items" to every branch-label section. A branch-label hit is reached along a saddle's invariant branch in infinite time, and there is no declared tube for it, so RI3 is ill-typed there. RI1, RI2 and RI4 work with arclength parametrization. Review line 161 asks only for interior-hit and local-uniqueness conditions. The file's own checklist line 85 says "The same two items". | Require RI1, RI2 (along the arclength branch) and RI4 at branch-label sections. Drop RI3 there or name its neighbourhood. Reconcile with line 85. |
| S4 | nit | CONFIRMED | Math-:`…/SARD_G_A1_RELATIVE_INTERIOR_LEMMA.md`:12 (also :71, :53, :9) | Paraphrase and custody imprecisions. Condition 1 omits the parts about the unique critical point, index one and spectral gap (successor line 38). "Both tracked ascending branches" should be "forward p-branch and backward q-branch". Line 53 omits RI3 (which is listed elsewhere). The review is cited without a commit (Math- AGENTS.md:8 prefers exact identities; the blob has in fact been stable). RI1–RI4 are not credited to review §5. | Complete the paraphrase, say forward/backward, pin `main@dbb9dcf` blob 9fc61b47, and credit review §5. |

### 3.6 Lucas file (`lucas/SARD_G_A1_APPLIED_TO_SUCCESSOR.md`, 495 B)
**Confirmed:**
- The source pin `a1fc958` is the current head of main PR #122: blob 8f86e0c9, 17,285 B, sha256 0fe23db6…, matching review lines 18 and 22.
- "Coverage construction is not membership" is correct: "interior transverse section" appears only in the coverage paragraph (successor line 49).
- "A6 conditionally valid … C103 HOLD" is consistent with the review and main STATUS.md:22.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| L1 | minor (adj. from major) | CONFIRMED | Math-:`incoming/grok-cycle4-20260926/lucas/SARD_G_A1_APPLIED_TO_SUCCESSOR.md`:8 | "P6 SATISFIED. P1 FAIL. P2 FAIL. P3 FAIL. P4 PARTIAL. P5 PARTIAL." uses labels defined in no reachable byte of main or Math-. Cycle-2 RESULT.md:45 cites a SARD_G_A1_REPAIR_PREDICATE.md that exists on no ref, so the key may be in a file that never landed. The labels are attributed inconsistently. Team note 5851042294 and the PR #87 map (line 18) call the Harper file "P1–P6 repair text", but that file uses RI1–RI4 and an unlabeled 6-item checklist, and PR #88's map calls it "RI1–RI4 / OPEN lemmas". Under the inferred key (P_i = Harper checklist item i), P1–P5 are defensible readings of successor lines 38–41. P6 is Harper's conditional conclusion, not a source predicate, so "P6 SAT" (republished at map line 21) scores nothing. The file gives no per-item evidence. The pin by full commit SHA fixes the blob, so a missing sha line is only a nit. The file's operative conclusions are conservative ("A1 remains AMEND", "C103 HOLD"). | Define P1–P6, quote successor lines for each score, and drop or re-express P6. Make both reading maps use the same labels. |
| L2 | nit (adj. from minor) | PLAUSIBLE | Math-:`…/lucas/SARD_G_A1_APPLIED_TO_SUCCESSOR.md`:9 | "F_eps endpoint example still in written U_chi." Read literally as a membership claim it is false, and the reverse of the counterexample: F ∈ U_chi, F_eps ∉ U_chi. The intended reading ("the example still applies to the written U_chi") is correct but names the wrong field. The A1 verdict is unaffected. | "The endpoint example (F ∈ written U_chi, F_eps ∉ U_chi, F_eps → F in C²) still applies." |

### 3.7 Packaging
| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| K1 | minor | CONFIRMED | Math-:`incoming/grok-cycle4-20260926/PUBLIC_READING_MAP.md`:11; PR #87 title and body | Stale inventory. The body lists only 4 of the 9 files; it omits harper/README.md, both benjamin files, the lucas file and the map. The title is "cycle-4 Harper packet". The PR #87 map lists 6 of the other 8 files (omitting harper README and the test). "Branch: … @ 8657130" reads as a branch-tip pointer, but it is correct as a content pin: all 8 packet blobs are identical at `8657130` and `0fab933` (packet tree 90c0d36e), and a file cannot contain its own commit SHA. Copilot thread PRRT_kwDOUphbdM6mV976 is unresolved. | List all files with their lanes in the title and body. Relabel the pin "packet content as of 8657130". Reconcile the labels with PR #88. |
| K2 | minor | CONFIRMED | Math-:`…/PUBLIC_READING_MAP.md`:12 (and :33) | "ready for review; merge is review intake only" is self-declared. Math- has no `incoming/` lane, CONTRIBUTING, intake workflow or AGENTS/README rule, and PR #87 is the first `incoming/` content on any Math- ref. main's actual lane (CONTRIBUTING.md:46–50, 90, 92, 95–99) requires RESULT.md and IDENTITY.json, refuses scripts such as test_closed_forms.py, and says landing is not "transfer into Math-". The SARD note reviews a main object (PR #122). A merge would not break a written rule (OP-AUTONOMY v2.1:22–28). | Before any merge: adopt an explicit Math- intake rule or move the packet to main's lane, answer the Copilot threads, and obtain a nonauthor byte review. |
| K3 | minor | PLAUSIBLE | Math-:`incoming/grok-cycle4-20260926/harper/test_closed_forms.py`:1 | No CI runs the test. None of the 16 workflow files at `0fab933`/`0333379` references `incoming/`. The only unfiltered workflow, downstream-gate.yml, replays fixed folders plus a transition audit, so its green result says nothing about any PR #87 file. PUBLIC_READING_MAP does not claim the packet was checked; README:5 over-claims (H4). | State "not CI-exercised". If it lands, add a path-filtered job or an exact replay. |
| K4 | minor | PLAUSIBLE | Math-:PR #88 body; PR #87 `PUBLIC_READING_MAP.md`:33 | PR #88's "merge the cycle-4 packet itself (that is Math- #87, publication-only if Grok merges it)", under "Does **not**:", pre-labels a possible author-lane merge "publication-only" without making it conditional on review. Once the threads are resolved, the ruleset (0 approvals) would allow any lane to merge. No rule forbids it. PR #88 is circumstantially Grok: its commit came 33 s after PR #87 went ready, and it names Grok in the third person. PR #88 has since received an OpenAI/ChatGPT engineering AMEND (5328356234, 2026-09-27T01:06:50Z) warning that PR #87 "has unresolved substantive review threads". PR #85's 3 m 22 s merge is a precedent, but who merged it is an inference. PR #64 was not an author-lane merge. | State that the Grok lane will not merge PR #87 before a nonauthor filed review. |
| K5 | minor | CONFIRMED | Math-:`…/SARD_G_A1_RELATIVE_INTERIOR_LEMMA.md`:4 ("Author-side packet: Harper cycle-4 (E)") | The lineage is not recorded. No cycle-3 commit, branch or path exists in main or Math-. The cycle-2 RESULT.md:18 ("Pinned in IDENTITY.json") and :45 (SARD_G_A1_REPAIR_PREDICATE.md) cite files present on no ref of either repo; other repos were not searched. "round 3" in the PR #64 comment matches the PR #163 D1_INTERFACE_TABLE.md header (committed 13 s earlier), a strong inference. Neither reading map gives any lineage. | Add a lineage note (cycle-2 = main `8dfde10`; cycle-3 = ?; cycle-4 = PR #87). Correct cycle-2 or push its missing files. Keep "round" and "cycle" numbering separate. |
| K6 | nit | PLAUSIBLE | Math-:PR #88 `PUBLIC_READING_MAP.md`:7 (also PR #87 map :11) | "Draft pull requests … are only hidden from GitHub's default “Ready” filter" is false: the default PR list shows drafts and has no such filter. PR #88 is titled "[incoming]" and sits on an `incoming/` branch, yet puts the file at the repository root (main CONTRIBUTING's rule covers main only). Dropping the SHA, as first suggested, would weaken custody. | Delete the sentence, relabel the pin as a content pin, and optionally move the file under `incoming/`. |

**Verdict for PR #87:** AMEND. **Do not merge as-is:** D1 makes the D5 ledger's central result false. The closed forms, the α series and Benjamin's six-pin identities are exactly correct. The SARD-G RI amendment is sound. Lemma OPEN needs its A2 dependency stated, and the Lucas scorecard needs defined labels.

### 3.8 Not verified
- A uniform bound on conditional C³/C⁴ moments over the whole cone, needed to turn the pathwise O(r²) bound into an expectation bound. It was accepted on the cone in the AMEND review, not re-proved here.
- Interval-arithmetic certification of the MC and 80-digit values, except the one R2 point a verifier reports certifying.
- The off-axis, inner-square and microdisk regimes.
- PR #69's content. Only its on-axis power agrees.
- The File-3 W-block and "Table 4.1": ABSENT from public GitHub.
- The "Lucas grid" and "12-jet 8.8e-14" runs: no artifacts.
- The meaning of P1–P6 outside main and Math-.
- Branch-protection details behind `blocked`.
- Classical C¹ dependence of local invariant manifolds, taken as standard.

---

## 4. Grok comments and late uploads

### 4.1 Inventory
| Surface | Id / SHA | Time (UTC) | Identification | Content (short) | Disposition |
|---|---|---|---|---|---|
| Math- PR #64 comment | 5850139585 | 21:42:30Z | grok-by-xai app; "xAI/Grok lane, round 3" | Supports ACCEPT_ERRATUM; calls the parent factor "the wrong factor"; wants the erratum "in-tree". | Algebra correct; framing G1; preceded the merge (P1). |
| main #63 comment | 5850158114 | 21:45:17Z | CIRCUMSTANTIAL (links PR #163, "Round-4") | "still 404s"; "Counterexample class that is real … kills uncorrected A3". | G1; its type-indicator sub-claim is false (Sylvester inertia). |
| main #63 comment (round-5) | 5850177692 | 21:48:09Z | CIRCUMSTANTIAL | "accepted as a Gaussian-regression lemma conditional on A1–A2". | G4. |
| Math- PR #85 | head `ff36667`, squash `5ed3b455` | 21:52:53Z–21:56:15Z | Created via grok-by-xai | Erratum pointer, CAP note, ASCII note. | Section 1 (P1, P8) and section 2. |
| main #63 comment (round-9) | 5850251715 | 21:58:35Z | CIRCUMSTANTIAL | Reports the default SHAs `d8f5505`/`5ed3b455`; "Still AMEND: … cap pairing (embedded 2r chart + Morse/§8)". | SHAs correct (ancestors of `03333792`); G3. |
| main PR #163 (another workflow) | head `cb774284` | 23:17:15Z (client) | Self-identified xAI/Grok lane | Session replay packet. | Consistency only: section 5 (X3–X5). |
| Math- #58 comment | 5850921404 | 23:30:52Z | CIRCUMSTANTIAL (same numbers as cycle-2) | "Exact extra-soft ledger … O(r³q²) (not the O(r⁶q²) …)". | X1. |
| main issue #165 | — | 23:31:05Z | Grok cycle-2 packet (content) | Lists 7 files; "Branch creation raced". | G5, X1, X2. |
| main branch `incoming/grok-cycle2-nd-d5-sard-20260926` | `8dfde10` | 23:31:28Z (client) | Path and content | RESULT.md and SESSION_LEDGER.md only. | X1, X2, K5; no PR. |
| Math- PR #87 commits | `77d1a03`, `d20ce86`, `f3af9b5`, `8657130` | 23:49:32–23:55:57Z (client) | Commit-subject self-ID | Cycle-4 packet. | Section 3. |
| PR #87 team note | 5851042294 | 23:51:20Z (never edited) | "Grok + Harper + Benjamin + Lucas" | Items 1–6; D5 line; P1–P6 line. | Items 1, 2, 4 (series) and 6 correct. Item 3: A2. Item 4 (float): H2. Item 5: R1. D5 line: D1/X1. P1–P6: L1. |
| PR #87 benjamin comment | 5851065560 | 23:55:16Z | Self-ID | Var(L \| pins) = 3/2; "Off-axis min eig larger". | First correct; second false (R2). |
| PR #87 ready_for_review | — | 2026-09-27 00:44:47Z | Shared account | Out of draft. | K2, K4. |
| PR #87 `db66b25` | PUBLIC_READING_MAP.md | 00:45:10Z | CIRCUMSTANTIAL | Reading map. | K1, K6. |
| PR #87 `0fab933` | Merge of main | 00:45:25Z | Shared account | Base now `03333792`. | Packet blobs unchanged. |
| Math- PR #88 | `incoming/public-reading-map-20260926` @ `2cdf62fd` | 00:45:35Z | CIRCUMSTANTIAL | Public reading map at repo root. | Firewall lines match STATUS.md @ `dbb9dcf`; the five named repos are public; listed PRs unmerged. K4, K6. |
| main branch `custody/status-erratum-boundary-bullet-20260926` | `b4cc5298` | — | Name only (CIRCUMSTANTIAL) | No unique commits (equals the PR #164 merge). | G2. |
| main issue #160 | body | 19:39:18Z | Self-ID "Grok team (xAI)" | LB-rate / C031 hold arithmetic. | Arithmetic reproduces except the C031 row, the 0.0334 grade and the rounding (G6). |
| main #116 comment | 5849269656 | — | Self-ID Grok team | Not reviewed in depth. | G2. |
| Cursor-hosted, self-identified xAI/Grok lane (cursor[bot]) | main #63 5841570965 (created 00:36:03Z, edited 00:45:40Z); #67 5841270276 (2026-09-25 23:59Z), 5841782206, 5841899531; #116 5841187314, 5841640700, 5842107966; Math- #58 5841875709; Math- PRs #52, #53, #55, #69, #72, #73, #74 and further cursor/* review branches; main PR #128 | Mostly 00:36–02:05Z on 2026-09-26 | Explicit self-ID lines (PR #69 via its NOTE.md:223 and PR #74) | D1-A..E ACCEPT; the D2 reconciliation that STATUS.md:11 links (5841782206); D5 microdisk (PR #69, O(r⁶q²)) and its review (PR #74). | Not in the Grok inventory (G2). D1-D caveat G7. PR #69's on-axis power is confirmed by D1/D2. |
| Math- PRs #80/#81/#82; main PRs #159/#161/#163 | — | — | Grok lane | Under another workflow. | Consistency only. |

**Confirmed in comments:**
- PR #87 items 1, 2, 4 (series) and 6: z₀ = (6k)²[(b² + 2)Φ(b/√2) + b√2 φ(b/√2)], with f_ss(M) \| six pins ~ N(−b, 2) exactly (quadrature agrees to about 1e−31).
- Benjamin's Var 6, Cov 6e^{−r²/2} and 3/2.
- The D1-E Gamma integral in main #63 5841570965: 144∫k^{4/3}φ_τ(12k)dk = Γ(7/6)τ^{4/3}/(24^{1/3}√π), exact. Also the D1-C exponent ledger and 12r^{−(d+3)}.
- main #160: 0.9144036, 5.18, 19/9, 19/108 r³, 0.21248, 3.328125e−6, 0.832 and 1.43×.
- main #165: all four pinned SHA-256 values.

### 4.2 Findings
| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| G1 | minor (panel split minor/minor/major) | CONFIRMED | Math-:PR #64 comment 5850139585; main #63 comment 5850158114; parent `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`:169 | "is the wrong factor" and "Counterexample class that is real … That kills uncorrected A3" attack a reading the parent's own line-169 clause excludes. These comments came 3 h 35 m and 3 h 38 m after AMEND 5848593803 made that point, and neither engages it. The same lane's SECOND_PASS.md (main PR #163 b785174, client-dated 21:37:44Z, lines 24–30) had given the correct reading K = D⁻¹HD⁻¹ five minutes earlier. "The type indicator does not go to 1{A_0<0}" is false even under the misreading, because congruence preserves inertia. Grok did not originate the framing (erratum line 19; Codex 5843521502 "genuinely inconsistent"); it repeated and escalated it. The recorded verdict stayed "AMEND / ACCEPT_ERRATUM only", and no formula changes. Attribution of 5850158114 is circumstantial. | Retract "wrong factor", "kills uncorrected A3" and the type-indicator sentence. Fold into the P4 successor note. |
| G2 | minor | CONFIRMED | Grok inventory (both repos):0 | Grok surfaces are missing from a branch-prefix inventory: PR #88; main #165; Math- #58 5850921404; main `custody/status-erratum-boundary-bullet-20260926`; main #160 and #116 5849269656. Also missing is the whole Cursor-hosted, self-identified xAI/Grok lane listed in 4.1, including the comment behind STATUS.md's D2 row (main #67 5841782206). PR #87 also moved to `0fab933`. | Key the inventory to self-identification lines, not branch prefixes, and add these surfaces to the review scope. |
| G3 | minor | PLAUSIBLE | main #63 comment 5850251715; Math-:`imports/lifetime_parent_20260925/CAP_PAIRING_IDENTITIES.md`:49 | Round-9 labels "cap pairing (embedded 2r chart + Morse/§8)" as "Still AMEND", which is exactly the interface D1-A ACCEPTED (5841570965) and on which STATUS D2 depends via #67 5841782206. Grok's own PR #163 EMBEDDED_CHART_AND_MORSE.md:57 accepts §8 and says only the numerical embedding radius is open, which "is enough for an existential Theorem A". So the label is overbroad, D1-A is not withdrawn, and D2 is not weakened. The overbroad label nevertheless stands on the #63 thread and on Math- default (see C3). | Post an explicit note on #63/#67: D1-A stands; only the numerical embedding radius is open. |
| G4 | minor | CONFIRMED | main #63 comment 5850177692:0 | "accepted as a Gaussian-regression lemma" is a same-session verdict on the lane's own derivation, with no provider/source-exposure disclosure and no REVIEW_REQUIRED label. It says convergence of means and covariances, which gives convergence in law only; the parent line 165 says "in probability". This is AMEND item 3, left unaddressed. The packet mixes the two modes itself (A3_UI_MAJORANT.md:58; A3_TYPE_CONVERGENCE.md:19 vs :52). In law suffices, so no result is false. Attribution is circumstantial. | Reword to "author-side derivation, REVIEW_REQUIRED", and state the mode or the coupling. |
| G5 | minor | PLAUSIBLE | main:`incoming/grok-cycle2-20260926/RESULT.md`:0 (branch vs main #165) | #165 lists 7 files. The branch at `8dfde10` holds 2 (RESULT.md 2941 B, SESSION_LEDGER.md 1263 B), under the same id rather than a "successor" id. IDENTITY.json is missing, and `tools/public_intake_check.py:292` would reject a PR from this head. RESULT.md:18 and :45 point to absent files. There is no PR and no comment on #165. The branch-creation time and the staleness of "raced" are inferences. | Complete the packet (IDENTITY.json and the missing files), or update #165 to list the two actual files and link the branch. |
| G6 | minor | CONFIRMED | main issue #160:0 (body) | The C031 table row gives the 13,690-byte C031_LBRATE_Integration.md hash `e165821b…`. That file hashes to `e7998ef0…` (a verifier reports the Drive copy hashes the same). `e165821b…` is C031_Freeze.md (1,739 B). "Dual-hashed" therefore conflates two documents; the real split is LB-1 citing the freeze hash for ledger statements. "Theorem-grade far term 0.0334": the value is graded "measured" in C031 integration:36, THM-023:122 and GP-LB-REC-001 §2, although THM-023:147 self-contradicts. 0.9666 × 0.9091 = 0.87873606, not ≈ 0.8786. The body is unedited and has no comments. | Comment on #160 with the "two documents" correction, the grade and the rounding. |
| G7 | minor | CONFIRMED | main #63 comment 5841570965:0 (Cursor-hosted xAI/Grok lane) | "All five parent interfaces are **ACCEPT**" includes a D1-D chain that uses parent Theorem A (p_r → 1) without marking it IMPORTED-OPEN. Only the selected/elder half of (13.6), and so ν_eld^all in Theorem C, depends on it. #63 was closed at 01:03:52Z on this basis and reopened at 01:11:38Z. The owner-account comment 5841766952 had flagged the dependency at 01:02:11Z, so the close was chiefly the closer's error. Which agent closed #63 is not provable. PROOF_INDEX:32's "accepts its exact Sections 8–15 interfaces" is mitigated by its placement under "Open or conditional". | Add a note: D1-D's elder (13.6) / Theorem C is ACCEPT conditional on Theorem A (IMPORTED-OPEN); mirror it in PROOF_INDEX. |

(Cross-referenced elsewhere, not re-counted here: the D5 power claims (D1–D3, X1), team-note items (A2, H2, R1), the PR #64 merge push (P1), PR #88 and reading-map packaging (K4, K6), and PR #85's change list (C7).)

**Verdict:** AMEND.
- The Grok comments are arithmetically sound on every closed form checked.
- Two families of comments present refuted or over-stated mathematics as settled: the D5 "#58 over-slaves det H_S" line (D1, X1) and the PR #64/#63 "wrong factor / kills A3" line (G1).
- One comment pushed a merge that left load-bearing files outside custody (P1).
- The inventory missed several Grok surfaces, including the review behind STATUS D2 (G2).

### 4.3 Not verified
- Merge actors.
- Edit histories of the edited comments (for example 5841570965, edited 00:45:40Z).
- The ND "15-det identically zero" and the W-block definitions (File 3 is ABSENT).
- The round-9 m = 1 far/near-branch and Vandermonde claims (these belong to PR #163's workflow).
- A complete enumeration of every comment from the Cursor-hosted lane.
- The actor behind the bulk PR updates at 2026-09-27T00:37–00:46Z.
- Whether private repositories exist.

---

## 5. Cross-cycle consistency (cycle-2 vs cycle-4 vs PR #163)

### 5.1 Side-by-side (one parametrization: X = M + r(p, q), scaled q; q_phys = r q)

**Witness chart.** Cycle-2 uses X = M + (0, q) with physical q; cycle-4 uses X = M + r(0, q), scaled; #58, PR #53 and PR #69 use X = M + r·s, scaled. This is a mismatch in presentation: neither Grok surface states its chart.

| Quantity | Cycle-2 (main `8dfde10`; #58 5850921404; #165) | Cycle-4 (PR #87) | PR #163 (`cb774284`) / #58 / PR #69 | Exact or recomputed | Status |
|---|---|---|---|---|---|
| E\|det H_S\| on {grad f(X)=0} | "stays O(r)" (RESULT.md:40) | O(k r), computed under pins only (ledger :115–119) | O(r² M) (PR #69) | Θ(r²); deterministic O(r²) | Cycle-2 and cycle-4 give a true but non-sharp bound and treat it as sharp. |
| Triple product | O(r³ q_phys²) (RESULT.md:43) | O(k³ r⁵ q²) | O(r⁶ q²) (#58, PR #69) | Θ(r⁴ q_phys²) = Θ(r⁶ q²) | **Cycle-2 = cycle-4** (same statement), both one power of r weak. #58 correct. |
| Formal cone contribution | — | O(k r²) | Target O(r³) | O(k r³) formal | Cycle-4 understated by one power of r (D3). |
| E\|det H_S\| numerics | MC at r = 0.3 only (reproduced: E\|dS\| = 0.578 = 6.42 r² = 1.93 r) | none | — | r-grid fit r^1.99 (MC) | A single r cannot fix the exponent. |
| det Gram(W_t, W_s) | 45 z_s⁸/16 | Team note item 5 places it under "six-pin only" | CLOSED_BF_WTWS.md:19: (3/4) z_s⁸ (G+V) | Unconditional 45/16; six-pin 3/4 | Only PR #163 gives the six-pin value (R1, X2). |
| Var(W_s/z_s²), Var(L) | "unconditional variance 15/4" (RESULT.md:35); SESSION_LEDGER.md:10 drops "unconditional" | REDUCED_FRAME: 3/2 six-pin | Cov(f_tss, f_sss \| G+V) = diag(2, 6), so 3/2 | Unconditional 15/4 + 3z_t²/z_s²; f_tss-only 15/4; six-pin 3/2 | Cycle-2 never pointed to the correction (X2). |
| E[f_tt(M)]/r → −6k error | — | k r² − b r/4 + … (ALPHA) | HESSIAN_JET_SIX_PINS.md:24 "quartic error (parent 5.2)" | O(r) for b ≠ 0 | Ambiguous wording in PR #163 (X5). |
| Theorem B status | "PROVEN-MODULO this repaired sentence" (RESULT.md:12, :35) | "PROVEN-MODULO repaired ND" (REDUCED_FRAME:5) | "PROVEN-MODULO (ND)" (ND_CUBIC_SLAVING.md:63); "not a certified kill" (:51) | Not on STATUS; the repaired sentence is not public | Drift starts in cycle-2 and is inherited by cycle-4 (R4). |
| SARD-G A1 predicate | SARD_G_A1_REPAIR_PREDICATE.md cited, absent from every ref | RI1–RI4 (Harper); P1–P6 (Lucas) | — | — | No lineage (K5, L1). |
| Math- PR #64 state | — | — | D1_INTERFACE_TABLE.md:10 "not yet on default" vs A_M_TO_A0.md:5 "landed" | Landed 21:46:22Z | Stale snapshots (X3). |
| P15 ASCII | — | — | SESSION_LEDGER.md:74 garbled; REPLAY.json:58 "in PROOF.md" | Additive note landed in `5ed3b455` | X4. |
| Congruence reading | — | — | SECOND_PASS.md:24–30 correct vs PR #64/#63 comments "wrong factor" | Convention-ambiguous; correct under H = DKD | G1. |

**Confirmed consistent:**
- Benjamin's six-pin diag(2, 6) for (f_tss, f_sss) equals PR #163's CLOSED_BF_WTWS.md / BF_WTWS_SCHUR.md G+V law.
- PR #163's HESSIAN_JET_SIX_PINS table values (−5.841, −5.960, −5.990, −5.998) match the exact α series (−5.841313, −5.960081, −5.990005, −5.997500).
- Var(f_ss(M) \| six pins) = 2 exactly, as PR #163 says.
- f_ss(M) \| six pins ~ N(−b, 2), consistent with main commit 6716d16.
- The cycle-2 MC table reproduces at r = 0.3 (NON-CERTIFYING): E\|dX\|/q = 1.751–1.756; p₀q² = 0.3805–0.4146; E\|dM\|/E\|dX\| = 1.000–1.005.
- The PR #163 body's statement about Math- default is accurate.

### 5.2 Findings
| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| X1 | major | CONFIRMED | Math-:PR #87 body and comment 5851042294; main:`incoming/grok-cycle2-20260926/RESULT.md`:43 (and :40); Math- #58 comment 5850921404; main #165 | "Math-#58 O(r⁶q²) over-slaves det H_S" and "O(r³q²) (not the O(r⁶q²) … not reproduced in 2D)" appear on five surfaces and deny a power that is correct in #58's own chart. Cycle-2 uses physical q, and #58/cycle-4 use scaled q. Converted, cycle-2's O(r³q_phys²) is exactly cycle-4's O(k³r⁵q²). Both are one power of r weaker than the true Θ(r⁶q²), not three. "Exact" (on #58) is backed only by NON-CERTIFYING MC at a single r = 0.3. The bounds O(r) and O(r³q²) are true but non-sharp; the false part is using them to reject #58. A "quiet relabel / internal Grok inconsistency" sub-claim was **withdrawn**: the two Grok surfaces agree with each other and share one error, unslaved det H_S. They differ only in writing "O(r³q²)" for different objects without naming a chart. Attributing the #58 comment to Grok is inferred from content. | Post corrections on #58, #165 and PR #87 (body and note). Mark RESULT.md:40/:43 as "non-sharp; true order r⁴q_phys² = r⁶q²". Drop "Exact", and state every power in one named chart. |
| X2 | minor | CONFIRMED | main:`incoming/grok-cycle2-20260926/RESULT.md`:35 (and SESSION_LEDGER.md:10); main #165 | "W_s/z_s² has unconditional variance 15/4" is literally true only as Var(W_s/z_s² \| f_tss). The literal unconditional value is 15/4 + 3z_t²/z_s², and the pinned-frame law is 3/2. SESSION_LEDGER E-ND-2 drops "unconditional" in a row marked CLOSED. #165 puts it under "(planar BF, d=2, six pins)". PR #87's correction (REDUCED_FRAME_4SLOT.md:7–28) sits in another repo and names neither cycle-2 file. The law giving 3/2 appeared earlier in the PR #163 branch (522bfe1/aa60b06; client dates). Positivity is unaffected. | Add erratum pointers in both cycle-2 files and on #165. |
| X3 | minor | CONFIRMED | main:`incoming/grok-session-20260926-replay/D1_INTERFACE_TABLE.md`:10 (also :28, :43; A3_TYPE_CONVERGENCE.md:69; INVENTORY.md:46; packet ERRATUM_CONGRUENCE.md:3) at `cb774284` | These lines say the erratum is not on Math- default, or that PROOF_INDEX still points at PR #64/`d573b99`. Both were false by 21:46:22Z and 21:56:15Z. Other packet lines contradict them: A_M_TO_A0.md:5, A3_UI_MAJORANT.md:8, STATUS_PIN_NOTE.md:13–14, INVENTORY.md:14, SECTION7_MGE2.md:5 and the PR body. These are dated snapshots that were never refreshed (per client dates), and they err toward caution. | Refresh the stale lines or mark them as dated snapshots before PR #163 is reviewed as a packet. |
| X4 | minor | CONFIRMED | main:`incoming/grok-session-20260926-replay/SESSION_LEDGER.md`:74 (also REPLAY.json:58, INVENTORY.md:42) | Line 74 is garbled: "`3e-2` → `3e-2` written as `3*e-2` or `3e-2`". REPLAY.json:58 says "disambiguate ASCII in PROOF.md", an in-place edit that would break the SOURCE_FILES.json pin (full-price.yml `verify()` raises "source mismatch: PROOF.md"). INVENTORY.md:42 contradicts itself. These lines were written at 21:50:00Z (client), before the additive note existed, and never updated. SESSION_LEDGER.md's sha256 b925e908… is the same at `c8405a76` and `cb774284`, even after Claude flagged it (5850519011, 22:36:26Z). The packet's own STATUS_PIN_NOTE.md:14 records that the note landed. | Replace all three with a pointer to Math- `5ed3b455:frontiers/full_price_20260924/ASCII_3E_MINUS_2.md` and `h_star = 3 - log(3*e - 2)`. |
| X5 | nit (adj. from minor) | PLAUSIBLE | main:`incoming/grok-session-20260926-replay/HESSIAN_JET_SIX_PINS.md`:24 | "`E[f_{tt}(M)]/r \to -6k` with quartic error (parent 5.2)". Read as O(r⁴), it is false: the exact remainder is k r² − b r/4 + …. The cited (5.2) is an O(r) bound from the fourth-derivative remainder, and sibling files say O(r). So the lanes do not disagree mathematically; the wording is ambiguous. | "error O(r) from the M4 remainder (parent 5.2); exact k r² − b r/4 + O(r³) (PR #87 ALPHA_4PIN_SERIES)". |

**Verdict:**
- Cycle-2 and cycle-4 contradict each other only in presentation, because neither names its chart. Mathematically they state the same D5 bound and share the same error, treating det H_S as unslaved. #58 and PR #69 are right.
- On the Gaussian-conditioning numbers, cycle-4 (Benjamin) and PR #163 agree with each other and with exact computation. Cycle-2's 15/4 and the team note's 45/16 are unconditional values that were never labelled that way in the six-pin context.
- The PR #163 packet has stale and self-contradictory lines about PR #64/#85 and the P15 ASCII action.

### 5.3 Not verified
- File-3 definitions of W_t and W_s beyond the cubic-order form in CLOSED_BF_WTWS.md.
- Cycle-2's own scripts: none are published, so "a single r" is inferred from the reported numbers only.
- The content of PR #163 beyond these consistency checks (another workflow).
- Push times: all packet timestamps are client commit dates.

---

**Part 2 totals:** 64 findings: 1 blocker, 4 major, 43 minor, 16 nit. 15 are PLAUSIBLE.
- Section 1: 10 (1 major, 7 minor, 2 nit).
- Section 2: 9 (5 minor, 4 nit).
- Section 3: 33 (1 blocker, 2 major, 21 minor, 9 nit).
- Section 4: 7 minor.
- Section 5: 5 (1 major, 3 minor, 1 nit).

Overlaps reduce the number of distinct root causes:
- P1 consolidates P2 and P3.
- D3 shares D1's root cause.
- X1 records the D1/D2 error on other surfaces.
