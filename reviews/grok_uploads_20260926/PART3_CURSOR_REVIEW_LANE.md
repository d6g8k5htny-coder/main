# PART 3: the "Grok 4.7 via Cursor" nonauthor-review lane (2026-09-25 to 2026-09-26)

**Scope.** This part covers every object in the two repositories whose own text self-identifies as "Provider | xAI", "Model | Grok 4.7 (`grok-4.7-high-fast`)", or xAI/Grok via Cursor. That means review files, open review PRs, lane-authored candidates the lane later reviewed, and issue comments. The objects are grouped into 11 units (U1–U11).

**Repositories and pins.** Math- main is `03333792b0fde32ac6b43649ba502d2fe638b5a6`; main default is `dbb9dcf64d23ecb45267ce2779f69f17b2fa6265`.

**Method.** Each unit was re-derived independently, and every finding was then checked by separate verifier passes. The severities below are the verifier-adjusted ones. Where the verifier passes disagreed, the split is noted.

**Evidence labels.**
- Exact arithmetic (sympy / Fraction) is certifying for the identity it checks.
- Floating-point, quadrature and Monte Carlo results are labelled **NON-CERTIFYING**.

**Status labels.** **CONFIRMED** means reproduced from bytes. **PLAUSIBLE** means supported, with a stated caveat.

**Disclaimer.** These verdicts are technical only. They carry zero organizational-independence credit and change nothing in either repository.

**Authorship basis.** All agents share one GitHub account, `d6g8k5htny-coder`. Git metadata for every lane commit shows only `Cursor Agent <cursoragent@cursor.com>` with the owner as co-author. So "Grok" below means *self-identified* Grok, and no provider identity is externally attested. The coordinator says so explicitly in issue-24 comment 5839816464 ("this coordinator does not independently attest provider provenance") and in PR31 comment 5840168397 ("no coordinator attestation of model/provider identity").

**Totals (verifier-adjusted).**

| | Count |
|---|---|
| Findings retained | 101 (76 CONFIRMED, 25 PLAUSIBLE; one dropped: U1-F5) |
| Blocker | 0 |
| Major | 7 |
| Minor | 51 |
| Nit | 43 |

---

## 1. Overview

| # | Object | Grok self-ID (quoted) | Landed | Downstream reliance | Verdict |
|---|---|---|---|---|---|
| U1 | Math- `reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md` (review of the OpenAI fixed-transverse candidate, R1–R5) | REVIEW.md:29–30 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (run id `grok-4.7-high-fast`) \|”; :31 Cursor run `bc-e46e875d-…` | commit c360f46 (Math- PR #26); blob 4003ff13, unchanged to Math- main | PROOF_INDEX:25; main `docs/site/museum.json` card `d5-fixed-transverse` (ACCEPT-scoped); CI replay `.github/workflows/contact-small-gap.yml`. Not cited by STATUS, LANDING_CLAIMS or GRAPH. | SOUND at stated scope; minor defects |
| U2 | Math- `reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md` (eight interfaces, fixed-annulus height window) | REVIEW.md:27–28 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (`grok-4.7-high-fast`) \|”; :29 run `bc-3524e567-…`. The script has no self-ID. | commit 27d4af0 (Math- PR #42); blob 1f153966 | LANDING_CLAIMS `rn-fixed-annulus-window` REVIEWED_SCOPED; GRAPH `math.rn-fixed-annulus-window` (review_provider “xAI/Grok via Cursor”); PROOF_INDEX:22, :45; museum `d5-height-window-annulus`. **No main STATUS row.** | SOUND at stated scope; minor defects |
| U3 | Math- `reviews/pr25_contact_kernel_20260925/REVIEW.md` (Sections B–C, R1–R4) | REVIEW.md:27–28 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (run id `grok-4.7-high-fast`) \|”; :29 run `bc-e46e875d-…` (same run as U1) | commit 8373c3a (Math- PR #31, merged 30925ca); blob ddf6859 | PROOF_INDEX:39 (under “Open or conditional”); contact-kernel-tail note. Not STATUS, LANDING or GRAPH. | SOUND; minor defects |
| U4 | Math- `reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md` (R1/R2 ACCEPT, R3 AMEND_REQUIRED) | REVIEW.md:28–29 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (run id `grok-4.7-high-fast`) \|”; :30 run `bc-89801c5e-…` | commit 1896281 (Math- PR #32); blob fa8dbe8; on main via 53320cb | PROOF_INDEX:39; bound by CUMULATIVE_TRANSFER_CORRECTION.md and the U7 D2 review. Not STATUS, LANDING or GRAPH. | SOUND at stated scope; minor defects |
| U5 | Math- `reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md` (R1–R7, all-height fixed annulus) | REVIEW.md:25–26 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (run id `grok-4.7-high-fast`) \|”; :27 run `bc-872dce35-…` | commit bebb77a (Math- PR #44, merged 52fca50); blob 559d7224 | PROOF_INDEX:21, :45; museum `d5-all-height-annulus` (ACCEPT-scoped); imported boundary in later pin notes. Not STATUS, LANDING or GRAPH. | SOUND at stated scope; minor defects |
| U6a | Math- `reviews/replacement_20260925_pr19_pr21/REVIEW.md` (M1–M7, S1–S4, (2)–(8), (1)) | REVIEW.md:25 “\| Reviewer model \| Grok 4.7, xAI (`grok-4.7-high-fast`) \|”; :26 session `bc-ff620630-…`. The scripts have no self-ID. | commit e34077f (Math- PR #27, merged 610b873); blob f1868bd9 | PROOF_INDEX:24, :44; museum `d5-inner-belt-density` | SOUND; minor defects |
| U6b | Math- `reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md` (S6–S21, (S3)–(S5)) | TWO_SCALE_REVIEW.md:17 “\| Reviewer model \| Grok 4.7, xAI (`grok-4.7-high-fast`) \|”; :18 same session | commit bc7d754 (Math- PR #34, merged 810bc16); blob 89882516 | PROOF_INDEX:23; LANDING_CLAIMS `rn-fixed-annulus-window` dependency `two-scale-s6-s21`; GRAPH indirectly, through U2's import of (S4); museum `d5-two-scale` | SOUND; minor defects |
| U7a | Math- `reviews/d2_cumulative_correction_20260925/REVIEW.md` (equation (D2) cumulative correction) | REVIEW.md:17 “\| Reviewer model \| Grok 4.7, xAI (`grok-4.7-high-fast`) \|”; :18 session `bc-ff620630-…` | commit 9a740b5 (Math- PR #43, merged 7a6d58f); blob 3ab51b01 | PROOF_INDEX:26; museum `cumulative-transfer-correction`. **Not** the basis of the STATUS D2 row (see U11). | SOUND |
| U7b | Math- `reviews/p15_full_price_nonauthor_20260926/REVIEW.md` (P15 Theorem F) | REVIEW.md:30–31 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (`grok-4.7-high-fast`) \|”; :32 run `bc-3902d28e-…`. The PR #63 body has no provider line. | commit cab2db5 (Math- PR #63); blob 07db19f8; integrated 1e1114f5 | main STATUS:14 D6 (via main #74 comment 5842112010); LANDING_CLAIMS `p15-full-price` REVIEWED_SCOPED; PROOF_INDEX:28. The GRAPH D6 node is stale. | SOUND; downstream wording defects |
| U8a | Math- PR #52 `reviews/pr25_d2_correction_confirm_20260926/REVIEW.md` | REVIEW.md:26–27 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (run id `grok-4.7-high-fast`) \|”; :28 run `bc-44e86e32-…` | Open draft, head f5e8594; not integrated | None. PROOF_INDEX:26 cites U7a instead. | SOUND; redundant, adds no independence |
| U8b | Math- PR #55 `reviews/d5_pin_neighborhood_20260926/REVIEW.md` (review of the Grok-authored PR53 note) | REVIEW.md:17 “\| Reviewer model \| Grok 4.7, xAI (`grok-4.7-high-fast`) \|”; :19 “… Same provider, different session. Organizational independence is not awarded. \|” | Open draft, head 4e188e25 | PROOF_INDEX:44 and main STATUS:21 cite it **as an AMEND review**; museum pointer | **AMEND.** The steep-strip ACCEPT's stated derivation diverges (major). The inner-disk AMEND is correct. |
| U9a | Math- PR #53 `reviews/pin_neighborhood_recon_20260926/NOTE.md` (lane-**authored**) | NOTE.md:132–133 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (`grok-4.7-high-fast`) \|”; :134 run `bc-3524e567-…` | Open draft, head 9a6f8a66 | PROOF_INDEX:44 (reconnaissance with an AMEND review); STATUS:21 AMEND | AMEND: headline r^2 log(1/r) not proved as argued |
| U9b | Math- PR #69 `reviews/d5_pin_microdisk_20260926/NOTE.md` (lane-**authored**) | NOTE.md:223–224 same two lines; :225 run `bc-23df1f0a-…`; :230 “Author-side note. Organizational independence is not awarded.” | Open draft, head ae45d351 | None (STATUS:21 stays AMEND) | Scalings survive exact recomputation; the continuum transfer is asserted, not proved |
| U9c | Math- PR #74 `reviews/pr69_pin_microdisk_nonauthor_20260926/REVIEW.md` | REVIEW.md:33–34 same two lines; :35 run `bc-28cc6d13-…`; :39 “Provider independence is **not awarded** either: the candidate and this review are both xAI Grok sessions.” | Open draft, head eb8bf7a5 | None | **AMEND.** It gives six ACCEPTs, but the load-bearing microdisk steps are not established (2 major). |
| U9d | Math- PR #87 `incoming/grok-cycle4-20260926/harper/D5_OBSTRUCTION_LEDGER.md` (related; outside the Cursor lane) | **ABSENT.** The file says only “Author-side packet: Harper cycle-4 (D)”. Grok attribution rests on the directory name and PR87 comment 5851042294 (“Team integration note (Grok + Harper + Benjamin + Lucas)”). Commits are by the owner, not Cursor Agent. | Open, head 0fab933 | None | Wrongly labels U9b's r^6 q^2 as an “error” (minor) |
| U10a | Math- PR #72 `reviews/pr125_shrinking_witness_nonauthor_20260926/REVIEW.md` (review of main PR #125) | REVIEW.md:32–33 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (`grok-4.7-high-fast`) \|”; :34 run `bc-4428875b-…`. The script has no self-ID. | Open draft, head cdb6bdd4 | None (main `docs/site/observations.json` lists it as an open PR only) | SOUND: 5 ACCEPT plus a correct AMEND; minor defects |
| U10b | Math- PR #73 `reviews/pr36_kernel_tail_nonauthor_20260926/REVIEW.md` (Q1–Q4) | REVIEW.md:27–28 “\| Provider \| xAI \|” / “\| Model \| Grok 4.7 (run id `grok-4.7-high-fast`) \|”; :29 run `bc-be016086-…` | Open draft, head 5913b004 | None | SOUND; minor defects |
| U11a | main issue #63 comment 5841570965 (D1-A–E) | “Provider Cursor, model Grok 4.7 (`grok-4.7-high-fast`), session `bc-ce4bf0bc-…`”. It names Cursor as the provider; there is **no “xAI” line**. | Mutable comment, edited 00:36:03→00:45:40Z | STATUS:20 (D1, AMEND/open: “accepted the load-bearing §8–§15”); PROOF_INDEX:32; the STATUS D2 chain through U11c | Arithmetic correct. D1-A accepts the cap step by citation only (major, via U11c). D1-D's Theorem C conclusion is conditional. |
| U11b | main issue #67 comment 5841270276 (D2 R1–R4) | “**Reviewer.** xAI, Grok 4.7 (`grok-4.7-high-fast`), Cursor cloud session `bc-81faa729-…`.” | Mutable comment, edited 23:59:05→00:07:25Z | STATUS:11 D2 ACCEPT; PROOF_INDEX:18 | Arithmetic correct. Parent §§2–3 imports are marked “Read and used” rather than IMPORTED-OPEN. |
| U11c | main issue #67 comment 5841782206 (D2 R5/R6 delta) | “**Reviewer.** xAI, Grok 4.7 (`grok-4.7-high-fast`), Cursor cloud session `bc-dfbce246-…`.” plus “Same provider family as those two reviews.” | Mutable comment, edited 01:04:21→01:07:56Z | **The link cited by STATUS:11 D2 ACCEPT**; PROOF_INDEX:18 | Closes the earlier IMPORTED-OPEN cap import through U11a's citation (major) |
| U11d | main issue #67 comment 5841899531; main PR #128 `notes/intermediate_scale_20260926/PROOF.md` (lane-**authored**) | The comment has **no self-ID** in its body (footer link to bc-235357c8 only). The PR body says “**Reviewer.** xAI, Grok 4.7 (`grok-4.7-high-fast`)…”; PROOF.md:4 says “**Author of this note:** xAI / Grok 4.7”. | PR #128 open draft, head 674f88cd; not merged | None (GRAPH `math.rn-region.intermediate-r-to-rho` OPEN_ACTIVE; PROOF_INDEX:45 “NO COMPLETE PROOF YET”) | AMEND: the §5 step is false for small L in rotated frames (PLAUSIBLE major) |
| U11e | main issue #116 comments 5841187314, 5841640700, 5842107966 | “This audit is Grok 4.7 in a Cursor cloud agent…”, “This review is Grok 4.7 in this Cursor cloud run…”, “This note is from Grok 4.7 in this Cursor cloud run…”. None has a provider line. | Mutable comments | Consumed by main PR #124 | Fail-closed audits, substantively right. One BLOCKED verdict over-generalises (minor). |
| U11f | Math- issue #58 comment 5841875709 | **No self-ID** in the body. The footer run bc-23df1f0a matches the U9b provenance. | Mutable comment | None (STATUS:21 AMEND) | Over-claims “closed” (minor) |

---

## 2. Per-unit results

Finding IDs are `U<unit>-F<n>`, keeping the numbers from the unit-level review. Each table lists blocker first, then major, minor and nit. There are no blockers anywhere in Part 3.

### U1: PR16 fixed-transverse review

**What it claims.** An ACCEPT of R1–R5 for the OpenAI candidate `TRANSVERSE_BOUND_CANDIDATE.md` @32b80ee: E(count under Q_r^W) = O(k r^3 area(E)) on the fixed-transverse chart. The scope is d=2, |v| ≥ η > 0, A>1, B<∞, compact marks, with qualitative C and r_*. It states: “Organizational independence is **not awarded**.”

**What was independently re-verified.**
- **Custody.**
  - Candidate: blob 024d927, 9902 B, sha256 f64c9542…6584e36. Identical at 32b80ee, at the Math- main import 6e4085a, and at the museum pin d6628da.
  - Review: blob 4003ff13, 12510 B, sha256 20d64514…. Identical at c360f46, main and d6628da, and equal to the CI pin.
  - Script: sha256 9f3580d4…, matching REVIEW.md:165.
- **Script re-run.** Exit 0. The U_4 higher-jet printout is {2: {5: 1/40}, 4: {7: 1/4480}}.
- **A step the review did not check: general degree-7 jet with the six pins solved (sympy).**
  - fxx(0) = −c40 r^2/24, fxz(0) = −c31 r^2/24, fxxx(0) = 12k − c50 r^2/40, fx(0) = −3kr^2/2, fz(0) = −c21 r^2/8.
  - fxx(M)/r = −6k + c40 r/12 and fxz(M) = −c21 r/2.
  - The J1, J2, J3 limits match. The limiting (a,c,d) minor is v^6/24 and the 3×4 map has rank 3.
- **Exact L=∞ kernel.**
  - det Cov(U_0) = 12.
  - Cov((a,q,c,d) | U_0) = diag(2,2,2,6) and E[a | U_0] = −b.
  - det Cov(J_0) = v^8(3(4u^2−1)^2 + 64u^2v^2 + 16v^4)/384 > 0.
- **Periodized kernel at L=4 (NON-CERTIFYING).**
  - Z_r/r^2 = 61.48, 63.25, 63.59, 63.77±0.18 at r = 0.2 … 0.02, against the limit 63.843.
  - det Cov(J_r | pins) = 4.18 → 4.86, against 4.866.
  - cond(Cov U_r) ≈ 285.
- **R3 three-Hessian suppression (NON-CERTIFYING sup).** On exact-rational random families the ratios stay in 0.16–1.5 as r goes from 1e-1 to 1e-3.
- **Ledger.** r^2 · kr^3 · r^-6 · r^6 = kr^5, and dividing by Z_r ≥ z_* r^2 gives kr^3.

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U1-F2 | minor | CONFIRMED | Math- `reviews/pr16_fixed_transverse_nonauthor_20260925/algebra_check.py:297` | REVIEW.md:165 says “All printed checks passed, including … the `U_4` contact term, the two `det/r` expansions, and the exponent identity `2+3−6+6=5`”. Several of those checks do not depend on the data. Line 297 `target4 = F(6) * (F(0) - 2 * (F(-1)))` never reads the cubic, and `b, kr3 = F(1), F(1)` is dead code. Lines 334–335 `signed = (-6) * 6` print OK unconditionally: a mutant `(-6)*7` still prints “ALL ALGEBRAIC CHECKS PASSED”. Lines 337–341 are bare integer arithmetic. The “limit minor” at line 238 is actually the finite-r minor, and no r=0 minor is ever formed. The 1/40 coefficient is printed, not asserted. CI replays this script as evidence. All the identities themselves are true. | In a successor script, compute the U_r target from the pins, assert the det/r products and the 1/40 coefficient, and form the r=0 minor separately. |
| U1-F3 | minor | CONFIRMED | Math- `PROOF_INDEX.md:25` | The line “D5 fixed-transverse chart: proof […] (byte-for-byte import …); review […]” sits under “Reviewed scoped results” with no scope. REVIEW.md:16 (“Exact fixed-transverse dimension-two chart only: `A>1`, `B<∞`, `\|v\|≥η>0`, compact marks, qualitative `C` and `r_*`”) is not carried over. main `museum.json:302–345` copies line 25 as the scope_quote of an “ACCEPT-scoped” card with status_quote null. For this card the footer “Review wording is quoted at its declared scope.” (museum.mjs:166) is therefore false. PROOF_INDEX lines 22 and 23 have the same gap (see U2-F9, U6-F6). This is not an over-claim, and there is no STATUS or LANDING row. | Carry the review's scope row in a source-bound navigation refresh, then regenerate the museum. |
| U1-F1 | nit (reviewer: minor) | PLAUSIBLE | Math- `reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md:99` | “…has determinant exactly `v^6/24`”. The Jacobian in the monomial directions is exact for *every* field. The imprecise step is the next one, “the pushforward law of J_r has covariance bounded below”. That treats J_r as a function of (a,q,c,d) alone, which is exact only for the cubic. The floor really comes from the r→0 limit, which the same paragraph states. The reviewer's later concession (PR25 contact-kernel REVIEW.md:40–44; PR26 comments 5839875151, 5839954322) concedes too much. Issue-24 comment 5839526890 repeats the unqualified sentence. PROOF_INDEX:25 carries no forward pointer. | Add an additive note: the result is an exact derivative identity in the monomial directions; J_r is not a function of (a,q,c,d) alone; the covariance floor comes from the limit. Do not edit the frozen blob. |
| U1-F4 | nit (reviewer: minor) | PLAUSIBLE | Math- PR #16 comment 5841312565 | “…independently reviewed in merged review PR26…” leaves out the review's own qualifier, REVIEW.md:35 “Organizational independence is **not awarded**”, and the coordinator's 5839816464. It is not a flat contradiction: REVIEW.md:37 asserts provider-level separation. “Subsumed in scope” is accurate. Nothing cites this comment. | Write “nonauthor technical review, self-identified xAI Grok, zero organizational-independence credit”. |
| U1-F6 | nit | CONFIRMED | Math- `reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md:20` | “\| State \| ACTIVE on publication of this file. …\|” and :23 “…HTTP 403 … This file is the claim and the result.” The claim and the result are in the same commit (c360f46, 21:09:36Z), so the stale-claim window had no effect. Meanwhile cursor[bot] comment 5839526890 on issue 24 exists (created 20:58:40Z, edited 21:10:07Z) and itself says “Issue 24 could not receive the comment (HTTP 403)”. | Record that the platform comment exists and that only the agent's own write was refused. |
| U1-F7 | nit | PLAUSIBLE | same file:131 | “Under `Q_r`, section 2 supplies uniform `L^2` bounds…”. The uniform-regression and moment inputs are accepted by restating the candidate. The review runs no computation on the actual K_L covariance floors or the Z_r limit; it checks only the cubic fixture plus one general-jet U_4 check. The checks above found no error. | Record at least one general-jet or actual-kernel check. |
| U1-F8 | nit | PLAUSIBLE | same file:30 | “\| Model \| Grok 4.7 (run id `grok-4.7-high-fast`) \|” labels a model slug as a run id; the run id bc-e46e875d is on line 31. The same label appears in pr25_contact_kernel:28, pr25_typed_transfer:29 and pr28_annulus_bridge:26. The PR26 body and comment 5839526890 place it correctly. | Relabel as “model slug” in successor records. |

**Not verified:**
- Provider identity.
- governance- PR #3/#4 and meta-framework PR #6 comments (outside the allowed read scope).
- Certifying uniform positivity for the periodized kernel over all frames, marks and L.
- The fine print of the R5 weighted Kac–Rice step (the cited theorem was not re-read).

### U2: PR22 fixed-annulus review

**What it claims.** ACCEPT on all eight interfaces of the OpenAI `FIXED_ANNULUS_CANDIDATE.md` @2804dc1d, which uses the two-scale addendum @b2e1652. The bound is E N_j(rE0) ≤ C k r^3 area(E0), stated on the fixed annulus 1<A0<B0<∞, d=2, fixed L, compact positive marks, with a height window of length k r^3. It is explicitly not all-height. “Organizational independence is not awarded.”

**What was independently re-verified.**
- **Custody.**
  - Candidate: blob 081abc13, 17646 B, sha256 1fd9fe71…552b.
  - Addendum: blob 89cae3a9, 15902 B, sha256 079f9399…7e4d.
  - Review: blob 1f153966, 12741 B, sha256 29a0c6d0…127d, identical on main.
  - Script: sha256 e7141a60…fc45, matching line 59. PR22 was merged at the reviewed head 2804dc1.
- **Generic degree-8 pinned field (sympy).**
  - All (A6) remainders start at r^4, except f_xxx − 12k, which starts at r^2.
  - (A7) det H_M/r + 6kS, det H_S/r − 6kS and f_xx(M)/r + 6k all start at r^1.
  - (A10) row remainders are O(r) as identities.
  - Minors: (S,C,D) = t^6/24, (S,T,C) = t^4(u^2−1/4)/8, (S,T,D) = ut^5/12, (T,C,D) = 0.
  - det(MM^T) − t^12/576 = t^8(64t^2u^2 + 9(4u^2−1)^2)/9216 ≥ 0.
  - The Jacobian is exactly r^-6.
- **Exact arithmetic.** 1 − 12q = 1/2; 18 + 72 + 6 = 96; −6 + 6 + 3 − 2 = 1; 168/12 = 14; (n/e)^n ≤ n! for n = 48 and n = 168.
- **Bargmann–Fock proxy at 80 digits (NON-CERTIFYING).** ‖Cov − MGM^T‖/r is stable in r. λ_min/t^12 is bounded below; it actually scales like t^8 at \|u\| = 1.6. E J_r1 → 6k(u^2−1/4).
- **Three-Hessian bound (NON-CERTIFYING).** Over 20000 random cubics the ratio stays ≤ 13.4.
- **Inner strip.** (A25) equals (S4) and its hypotheses hold. bc7d754 (21:45:27Z) predates 27d4af0 (22:01:28Z).
- **Later errata.** The PR64 erratum and the main #63 reopening concern D1 §5. They do not touch this candidate, which computes det H/r directly.

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U2-F1 | minor | CONFIRMED | Math- `reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md:51` | “`algebra_check.py` … checks the pin target, the degree-three contact expansion, the minor `t^6/24`, the triangular Jacobian `r^{-6}`, and the exponent budgets.” In fact each is checked at a single rational point (r=1/3, u=5/2, t=−2/5), with the low jets pre-substituted by their (A6) values. The Jacobian check (script:130) is the scalar tautology (1/r^2)(1/r)(1/r^3) == r^-6. Line 188 (“finite rational script”) partly discloses this. The identities hold symbolically. | Describe these as single-point instance checks, or replace them with symbolic identity checks. |
| U2-F2 | minor | CONFIRMED | same file:152 | “The cited Kac–Rice theorem is used only as this integral representation.” The weighted, height-disintegrated representation (A21) is the candidate's own review item R6. The review accepts it without naming arXiv:2304.07424 Thm 7.1/2.2 or checking its hypotheses. It also does not bind to the Stecconi Thm 29 check in TWO_SCALE_REVIEW row S19. (A21) is correct (the weight is nonnegative and lower semicontinuous), and LANDING_CLAIMS already marks the framework IMPORTED_FRAMEWORK. | State (A21) as an import, and name the theorem and its hypotheses, or bind it to S19. |
| U2-F6 | minor | CONFIRMED | Math- `frontiers/downstream_gate_20260925/GRAPH.json:301` | “For this scope only, the reviewed analytic route bypasses CH-LIFT / Piece-2 / OBL-H5-JETMOD.” The review never assessed those predicates: REVIEW.md:3 and :188 say it “does not … flip a scientific flag”. The node was added in Math- PR #51, from the OpenAI-lane branch `chatgpt/d5-graph-regional-update-20260925`, merged with zero reviews. `SELECTOR_REGION.json` still has covered_region_ids = [fixed-remote], which test_hard_gate.py:192 hard-codes. The “reviewed” wording spreads to PROOF_INDEX:47 (“outside reviewed regional bypasses”) and to open PR #65 AUDIT.md:67. Gate impact is nil, because all four edges are `required: false`. | Mark the bypass as a separate author-side implication that needs its own review, and reconcile SELECTOR_REGION. |
| U2-F3 | nit | CONFIRMED | REVIEW.md:148 | “The sites `M`, `S`, and `X` stay at least `r(A0-1/2)` apart.” Read pairwise, this is false for A0 > 3/2, since \|M−S\| = r. Only qualitative distinctness is used. | Write “X is at least r(A0−1/2) from each pin; the pins are r apart.” |
| U2-F4 | nit | CONFIRMED | REVIEW.md:178 | “The boundary circle has expected count zero…”. The dividing set is the lines \|t\| = r^{1/24}, not a circle. The candidate (line 222) correctly says “Boundary lines”. | Correct the wording or drop the sentence. |
| U2-F5 | nit | CONFIRMED | REVIEW.md:33 | “This run is not the PR16 session … and not the two-scale session …” is accurate by run id but leaves out the lineage. 27d4af0 has the single parent c360f46 (the PR16 review commit), on branch `cursor/pr16-transverse-r1r5-review-b00c`. The whole acceptance chain is xAI Grok (bc-3524e567, bc-ff620630, bc-e46e875d). | Add a lineage line. |
| U2-F7 | nit | CONFIRMED | Math- `claims/LANDING_CLAIMS.json:195` | “"reviewed_subject_commit": "bc7d754…"” holds the review's own commit, while lines 176 and 188 hold the subjects (2804dc1, b2e1652). The PR22 review object pins only 760340e / blob 1f15396 and has no pointer to 27d4af0. All hashes are correct; the mislabel comes from the rename in 07d7478. | Rename the field or add `review_commit`. |
| U2-F8 | nit | CONFIRMED | GRAPH.json:273 vs Math- PR #51 body | The PR body says “`REVIEWED_SCOPED` classification”. The value that landed is `AUTHOR_SIDE_CANDIDATE` plus review_disposition ACCEPT (after cc3a991). The landed value is the more conservative one. | None needed; readers should rely on the landed value. |
| U2-F9 | nit | CONFIRMED | Math- `PROOF_INDEX.md:22` | The line is a bare pointer with no scope. main museum `d5-height-window-annulus` quotes it verbatim with status_quote null. LANDING_CLAIMS and Math- README:13 do carry the scope. Lines 23 and 25 are equally bare. | Add scope text to lines 22, 23 and 25, then re-pin the museum. |

**Not verified:**
- Provider identity.
- The content of arXiv:2304.07424 and arXiv:2103.10853 (offline).
- Uniform-in-frame bounds for the periodic K_L (only the Bargmann–Fock proxy was checked).
- The S6–S21 derivation (covered in U6).
- Whether the historical CH-LIFT, Piece-2 and JETMOD predicates are logically bypassed.

**Observation.** Open PR #87 D5_OBSTRUCTION_LEDGER.md:29 misdescribes this bound as “valid only for \|v\| >= eta > 0”, conflating it with U1.

### U3: PR25 contact-kernel review

**What it claims.** ACCEPT on R1–R4 (the six-pin cubic contact kernel (C1)–(C5) on the fixed chart) in the OpenAI `reviews/collision_mechanism_20260925/NOTE.md` @ad35e46. It excludes Section A “as a separate geometry theorem”, Section D, the tails, annulus synthesis and η→0.

**What was independently re-verified.**
- **Custody.** NOTE: blob 3ee30829, 16243 B, sha256 530dd3ef…c9d37e. Review: blob ddf6859, 9969 B, sha256 d729cfa8…, equal to the entry in main's public-math inventory. Script: sha256 51b208df…, matching line 139.
- **Script re-run.** Exit 0 with 17 OK lines.
- **Structure of (B1).** The general cubic under the six pins has rank 6, and the free jets correspond to A/2, q/2, c/2, d/6.
- **(B2)–(B4) exact.** The sample point u=2, v=1, k=1, θ=1/2, q=−30 gives w=−1, A=−3, c=75, d=−363/2 and determinants 18, −18, −18.
- **(B5) exact** at 199 rational θ. The interval length is 2(1+√θ−√(1−θ)) > 0.
- **Generic quintic with exact finite-r pins.**
  - J_r at r=0 equals (C4).
  - The finite-r (a,c,d) Jacobian is [[0,v^2/2,0],[v,ruv,rv^2/2],[0,0,−v^3/12]], with determinant v^6/24.
  - a_r/r → A, and H/r → B_M, B_S, B_X.
  - dJ/d(fx,fz,f) = r^-6, and the exponent count gives r^3.
- **Normalizer.** z0 = 36k^2 E[a^2 1{a<0}], the same as in U1.
- **(NON-CERTIFYING)** The coarea identity ratio is 1.000000000000000000000000000030, and Λ_1 ≈ 1.25e-24 > 0 at one point.
- **(A4) majorant.** Re-derived by hand. On 20000 random witnesses (NON-CERTIFYING) the maximum ratios are 0.40, 0.32, 0.43 and 4.7e-6.
- **Bargmann–Fock diagnostic.** Mean (−b,0,0,0) and covariance diag(2,2,2,6), exact.
- **Later errata.** None targets Sections B–C.

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U3-F1 | minor | CONFIRMED | Math- `reviews/pr25_contact_kernel_20260925/REVIEW.md:139` | “The run checks the six pins, the solved jet (B2), the three cleared determinant identities, the sample point `18,-18,-18`, and the exponent count `2+3-6+6-2=3`.” The script checks only 4 of the 6 pins: ∂_tP(±1/2,0) is never evaluated (script:266–270). The exponent check is `if (3 + 2 - 6 + 6 - 2) != 3:` (script:289). The script also prints “OK contact minor v^6/24” (script:281–282, `minor = F(1, 24)`) and “OK det B_X negativity gap” without computing either, and J0 appears only in comments. All of these facts are true (sympy). | Compute the minor, Jacobian and J0 symbolically, evaluate ∂_tP, and reword line 139. |
| U3-F2 | minor | CONFIRMED | same file:117 | R4 accepts the marked Kac–Rice representation of (C1) on bookkeeping alone. The source, NOTE.md:160, says: “This use of marked Kac-Rice needs separate analytic review, not just the finite algebra tests.” The review does not acknowledge this. The hypotheses do appear to hold, and the same lane's PR16 R5 covers the identical representation but is not cited. The only downstream exposure is PROOF_INDEX:39, which is in the open/conditional section. | Name the theorem and check its hypotheses, or cite PR16 R5 together with the J_r change of variables. |
| U3-F4 | minor | CONFIRMED | same file:109 | “`\|det B_X\| ≥ (9k^2/v^2)·4δ(1-δ) ≥ c(δ,η,k_min)>0`”. Since 9k^2/v^2 decreases in \|v\|, a lower bound needs the outer bound \|v\| ≤ B. The correct constant is 36 k_min^2 δ(1−δ)/B^2. The bound does hold on the note's compact chart K. | Replace c(δ,η,k_min) with c(δ,B,k_min). |
| U3-F3 | nit (reviewer: minor) | PLAUSIBLE | same file:105 | “The pathwise majorant (A4) supplies …”, yet line 18 excludes Section A and no citation is given. The same run had already accepted the qualitative bound 24 minutes earlier, in PR16 R3 (c360f46), and G1 comment 5841036818 later accepted the explicit constants. The mathematics is sound (re-derived). | Cite PR16 R3 and G1 as the imported basis. |
| U3-F5 | nit | CONFIRMED | same file:16 | The scope row, “on the fixed chart `\|v\|≥η>0`”, drops A0 ≤ \|(u,v)\| ≤ B, and R3/R4 use B. Open PR46 `closed-lemmas/L-TYPED-CUBIC.md:7` copies the shortened form. | State the full chart K. |
| U3-F6 | nit | CONFIRMED | same file:55 | The converse (that every pinned cubic is (B1)) is not checked. The reason given, “every transverse term carries a positive power of `t`”, does not cover the P_t pins; the (s^2−1/4) factor does. Both statements are true. | Give the rank argument and cite the factor. |
| U3-F7 | nit | CONFIRMED | same file:57 | “The transverse derivative fixes `c=…`”. It is the axial derivative ∂_s that fixes c. | Replace “transverse” with “axial”. |
| U3-F9 | nit | CONFIRMED | same file:32 | “…The reviewer is a different provider from the OpenAI author.” This rests on self-declaration only (owner readback 5840168397). The same applies to typed-transfer REVIEW.md:36. | Write “self-identified xAI Grok; provider identity unattested”. |
| U3-F8 | nit | PLAUSIBLE | same file:42 | “It is not an exact finite-`r` determinant identity for every field.” This is ambiguous. The identity is exact in the monomial directions for every field, while the regression-coordinate minor is v^6/24 + O(r^2). The conclusion is right. | Call it a derivative identity, and state covariance positivity separately. |
| U3-F10 | nit | PLAUSIBLE | Math- PR #46 (open) `closed-lemmas/L-CONTACT-KERNEL.md:21` | The scope line leaves out Section A and rewrites the exclusion as “Section D tails beyond D1”, which implies D1 falls under this acceptance. The card also lacks the review's “does not change `lemma_closed`” disclaimer. The PR is unmerged. | Copy the review's exclusions verbatim. |

**Not verified:**
- Uniform-in-r regression bounds for the periodized covariance.
- A formal proof of the Kac–Rice hypotheses.
- Explicit constants for Λ_1.
- Provider identity.

### U4: PR25 typed-transfer review

**What it claims.** On the same NOTE @ad35e46:
- **R1 ACCEPT:** (B4) and (B5).
- **R2 ACCEPT:** the typed kernel with factor 24k/(z0\|v\|^6).
- **R3 AMEND_REQUIRED:** the cumulative (D2) statement must keep the hypothesis h/(κr^m) → 1.

**What was independently re-verified.**
- **Custody.** NOTE blob 3ee30829. README feb62c5c. exact_checks e51a5bda, with 24 author tests. Review: blob fa8dbe8, sha256 468b47b8…. Script: sha256 f645c6f8…, matching :213. The same blobs are at main, 596e809 and d06a562.
- **Script re-run.** 12 tests OK.
- **B1–B5.**
  - B1 has rank 6 and nullity 4, with free monomials t^2, t^3, st^2, s^2t.
  - B2–B4 residuals are 0, and the sample gives dets 18, −18, −18.
  - B5 checked on an exact 39×801 grid: 0 mismatches.
- **Kernel quantities.**
  - J0 derived from a general quartic Taylor field, with minor v^6/24.
  - Jacobian r^-6.
  - Pin target (b−kr^3/2, −kr^2, 0, 12k, 0, 0).
  - Slice factor v^6/24 recovered by an independent change of variables.
- **Monte Carlo (NON-CERTIFYING).** 300000 cubics: 89 hits against 91.9 expected.
- **Bargmann–Fock diagnostic** (the review did not check it). Mean (−b,0,0,0), covariance diag(2,2,2,6).
- **R3.**
  - h′ = r^2[3 − cos(1/r) + 4r sin(1/r)].
  - Density limits 1/2 and 1/4.
  - ν = 2(ℓ^{-1/2} − ℓ^{-1/3}).
  - The negative control h = 2κr^m gives N^3 = 1/512 against 1/128, a ratio of 2^{-2/3}.
  - The author correction d06a562 implements all four amendment bullets.

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U4-F1 | minor | CONFIRMED | Math- `reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md:182` | “At `ℓ=1/64` the value is `8`, while a pure multiple of `ℓ^{-1/3}` does not reproduce that value.” This is false: 2·ℓ^{-1/3} = 8 at ℓ = 1/64. The script (371–374) compares 8 only against 4. The phase conclusion itself is right, via the asymptotics. | Replace with: ν ℓ^{1/3} = 2(ℓ^{-1/6} − 1) → ∞. |
| U4-F2 | minor | CONFIRMED | same file:138 | “…the symbols `1`, `ξ`, `ξ^2`, `ξ^3`, `η`, `ξη` in the pin frame are linearly independent polynomials. The conditional covariance of `(a,q,c,d)` given `U_0` is therefore positive definite”. This does not follow. Positive definiteness needs all ten monomials of degree ≤ 3 to be independent on the dual lattice. The conclusion is true. | State the ten-monomial criterion. |
| U4-F3 | minor | PLAUSIBLE | same file:140, :150 | “…once the witness conditioning has put every Hessian at order `r`.” This relies on Section A (A4), as NOTE.md:139 shows, but line 17 excludes Section A and the dependency is not disclosed. As a result, PROOF_INDEX:39 “Section A is outside both acceptances” hides the dependency; for U3, which uses (A4) explicitly, that sentence is simply wrong. A1–A4 were accepted later, in PR25 comment 5841036818 (xAI/Grok session bc-06c651fb, 23:27Z), after this review (21:40:59Z). | Disclose the (A4) dependency and have PROOF_INDEX:39 note it. |
| U4-F4 | minor | CONFIRMED | same file:213 | The Computation record lists as “passing identities” the minor `v^6/24`, `2+3-6+6-2=3`, `2(8-4)=8`, and others. These are hard-coded (script 230–245, 276–297, 325–332, 354–374). The B5 test only checks θ(1−θ) > 0, and the script never opens NOTE.md. Line 48 partly disclaims this. The mathematics is true. | Relabel these as arithmetic reminders, or replace them with real checks. |
| U4-F5 | minor | CONFIRMED | Math- issue #29 comment 5839903956 | “R1 and R2 were rederived in exact arithmetic.” The R2 continuum limits were argued analytically, not computed exactly. This contradicts REVIEW.md:48 (“A passing run checks those finite identities only”). No downstream surface cites the comment. | Split R1/R2 finite identities (exact) from the R2 continuum limits (analytic). |
| U4-F8 | minor | PLAUSIBLE | Math- `reviews/collision_mechanism_20260925/NOTE.md:185` (on main) | The unamended (D2) paragraph was kept byte-identical, which is good for custody. But README.md (blob d5b0a45) has no pointer to CUMULATIVE_TRANSFER_CORRECTION.md, and the README and the correction still say “Review requested” / “Separate reviewer confirmation requested”. The correction file does sit in the same directory, and PROOF_INDEX:26 and :39 route readers to it. | Add an additive pointer (for example an ERRATA file); keep NOTE.md unchanged. |
| U4-F6 | nit | PLAUSIBLE | same file:104 | “Both sides of that comparison are positive” has an ambiguous referent. | Write “Both √θ+√(1−θ) and 1 are positive.” |
| U4-F7 | nit | CONFIRMED | same file:22 | “Issue-comment writes are not available to this agent. This file is the claim and the result.” Yet cursor[bot] 5839903956 on issue 29, from the same session, carries the dispositions. | Say the wrapper's comment carries a summary and REVIEW.md is authoritative. |
| U4-F9 | nit | CONFIRMED | same file:29 | The “run id `grok-4.7-high-fast`” mislabel again. It appears in 4 of 9 lane files. | Relabel as “model id”. |

**Not verified:**
- The analytic continuum steps of R2 for the finite-L field.
- Positive definiteness at any specific L.
- The PR16 pin transform (not read).
- Provider identity.

### U5: PR28 annulus-bridge review

**What it claims.** ACCEPT on R1–R7 of the OpenAI `rn_annulus_bridge_20260925/PROOF.md` @dedc69e1: the all-height fixed-annulus bound (2) at d=2, fixed L, compact positive-gap marks, 1<A<B<∞. The review is cross-provider, and organizational independence is not awarded.

**What was independently re-verified.**
- **Custody.**
  - PROOF.md: blob 6f317515, 16948 B, sha256 d55e2c03….
  - All 7 proof-folder blobs are identical on main.
  - Review: blob 559d7224 (sha256 69ea7a47…, 15925 B). Script: blob bc6b55e9. Both identical from bebb77a to main.
- **Test runs.** The review's script passes 9/9. The author's unittest, which the review did not run, passes 21/21.
- **Hermite jet and pinned target (exact).**
  - U_r = (s0 − a^2d1/2, (3d0 − s1)/2, d1, 3(s1 − d0)/a^2).
  - v_r = (b − kr^3/2, −3kr^2/2, 0, 12k, 0, 0).
- **A step the review did not check:** U_r → U_0 at rate O(r^2) on a degree-6 pinned field.
- **Remainder and floor.** The Y1/Y2 remainder envelopes are confirmed. The Cauchy–Binet floor minimum is 1/2.
- **R4 coefficient.** Its exact supremum is B^2 + 3B/2 + 3/8 + B√(2B^2 + B + 1/4), which is below 3B^2 + 2B + 1 for all B > 0. The squared difference is 2B^4 + B^3 + 5B^2/2 + 5B/8 + 25/64.
- **Off-axis limits.** J1(r=0) = 6k(u^2−1/4) + f_xxz uv + f_xzz v^2/2 and J2(r=0) = f_zz v. The minor is −v^3/2.
- **Endpoint limits.** det H_M/r → −6k f_zz(0) and det H_S/r → 6k f_zz(0).
- **Ledgers.** r^-13 · s^14 → r. exp(−c0/s^2) ≤ 7! c0^{-7} s^14. The gap is (A−1)(3A+1)/4.
- **(NON-CERTIFYING)**
  - Schur eigenvalues stay in fixed bands, e.g. [0.65, 2.9] at u = 1.2.
  - δ\|τ − m_Y\| → 6(u^2−1/4)k, with values 7.14, 22.5, 12.
  - Lemma (6): the maximum ratio over 4000 cubics is 0.320.
- **Timeline.** “TAKE … NOW” at 22:06:32Z. Review commit at 22:17:49Z. PR44 left draft at 22:22:01Z and merged at 22:22:07Z. PR28 merged at 22:23:14Z.

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U5-F1 | minor | CONFIRMED | Math- `reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md:3` vs :74 | “It is not permission to promote the candidate, merge it, or treat (2) as a closed prerequisite.” versus “These accept the stated fixed-annulus upper bound and the seven interfaces that produce it.” The author's PROOF.md:245 makes exactly this separate review the gate for closed-prerequisite use. PR28 was merged 67 s after PR44, even though its body says “Keep DRAFT”. Later notes import it as “accepted by merged PR44”, within the technical scope. This clause appears only in this file among the lane's reviews. | State one posture: a scoped technical ACCEPT, usable as a scoped input. |
| U5-F2 | minor | PLAUSIBLE | same file:56 | “Sections 3–8 were rederived from …” overstates two steps. The Schur floor (line 108) is stated without a perturbation inequality. R6 names no Kac–Rice theorem (line 142, “framework”), though it does check the main hypotheses; the non-local mark and almost-sure nondegeneracy are not treated. The conclusions are correct. | Say which steps were rederived and which were checked at source level, and name the theorem. |
| U5-F4 | minor | PLAUSIBLE | main `docs/site/museum.json:147` (dbb9dcf) | The card `d5-all-height-annulus` has “class”: “ACCEPT-scoped” and status_quote null, while no STATUS, LANDING or GRAPH entry cites this review on any of the 147 Math- refs or on main. The museum is a “Selected display projection only”, with scientific_status_authority false; its class is copied from PROOF_INDEX “Reviewed scoped results”, and the null status_quote is hard-coded. The real inconsistency is that STATUS shows no D5 reviewed-scoped result at all. | Add scoped STATUS and manifest rows, or label the card reviewed-not-registered. |
| U5-F3 | nit | CONFIRMED | same dir `algebra_check.py:309` | `minor = Q(0) * Q(0) - v * (v * v / 2)` is hard-coded. The crossover exponents (331–343) are literals, the check at 351–353 uses four samples, and the rank floor is sampled. The review discloses “checks those identities only”. | Derive these symbolically. |
| U5-F5 | nit | PLAUSIBLE | REVIEW.md:18 | “\| State \| ACTIVE on publication of this file. …\|” is never closed. Issue-29 comment 5840317793 was created at 22:06:36Z and overwritten at 22:18:36Z. PR44 really was a draft until 22:22:01Z, so its “draft pull request” wording was accurate at the time. The same template line appears across the lane. | Use separate, immutable claim and verdict comments, and give the file a terminal state. |
| U5-F6 | nit | PLAUSIBLE | REVIEW.md:48 | “That audit is source-exposed and same-provider. … none of its MATCH sentences was used…”. The review was not blind, and the non-use claim cannot be checked. Most of the shared framings come from the source PROOF.md. Only “The rare conditional mean is retained.” (:114) is traceable to audit 5840044226. | Write “non-blind; exposed to a same-scope MATCH audit”. |
| U5-F7 | nit | PLAUSIBLE | Math- `frontiers/rn_annulus_bridge_20260925/PROOF.md:38` (source) | “three noncollinear zero gradients force all three Hessians to be small” oversells (6). The review scoped the lemma correctly (lines 116, 120, 124, 148). This is a source-wording nit only. | Reword the source. |

**Not verified:**
- A certifying Schur floor for the torus kernel.
- The hypotheses of the Kac–Rice theorem in arXiv:2304.07424.
- Torus distinct-site nondegeneracy.
- External Cursor and Drive artifacts.

### U6: PR19/PR21 replacement review and two-scale review (session bc-ff620630)

**What they claim.**
- **REVIEW.md** accepts:
  - M1–M7 and S1–S4 of the finite-r C6 supplement (PR19 @e93eade);
  - (2)–(8) and corollary (1) of the inner-axial density proof (PR21 @b420099).

  Line 114 lists as an open, imported gap: “Conditional moments of a random derivative ceiling M after all six pins.”
- **TWO_SCALE_REVIEW.md** accepts S6–S21 and (S3)–(S5) of `TWO_SCALE_ADDENDUM.md` @b2e1652f, on δ ≤ δ0 and A ≤ \|u\| ≤ B.

**What was independently re-verified.**
- **Custody.**

  | File | Bytes / blob | sha256 |
  |---|---|---|
  | C6 | 8054 B | 8260e567… |
  | REPAIR | blob 40c7ff79 | cc70ee30… |
  | PROOF | 10858 B | 8b9376a6… |
  | Addendum | 15902 B | 079f9399… |
  | REVIEW.md | blob f1868bd9, 15615 B | 7ff9e4a0… |
  | TWO_SCALE_REVIEW | blob 89882516, 10073 B | 04cf4c98… |

  All are identical at the merges, 760340e, d6628da, 1e1114f5 and main.
- **Shipped script output.** “cases=360 M1=1, M2=1, M3=2/3, M4=40/121, M5=1/14, M6=1, M7=1, S1=19902/113659, S2=540/4381, S3=6510/14443, S4=103680/841357”, identical to REVIEW.md:102. The author suites pass 21, 15 and 13 tests.
- **Constants, exact.**
  - 1/384, 1/1920, 3/80, 121/15360, and 7/960 = 3/640 + 1/384.
  - C_x = 7/960 + 27B/320 + 3B^2/160 + 4B^3/3.
  - C_y = 49/384 + 27B/640 + 2B^2.
  - C_ax = 1/384 + 27B/640 + B^3/6.
  - C_h = 121/15360 + 19B/1920 + 81B^2/1280 + B^3/160 + 2B^4/3.
- **Adversarial scan (NON-CERTIFYING).** 72 degree-7..9 pinned polynomials, all ratios ≤ 1.
- **Sharp fixtures.** The quartic defect is 30r at u = 2. The pure x^5 fixture pins to x^5 + 3x^3/2 − 23x/16 + 3/2, with defect 120/384.
- **PR21 on a symbolic degree-9 field.**
  - The shear holds for generic data.
  - The r^0 term of Y_r is (αQ + uwT, βT + wS).
  - U_r − U_0 = O(r^2).
  - The minor is u(u^2−1/4)^2/12 and the Jacobian is r^5.
- **Two-scale.**
  - Y1 remainder monomials satisfy i ≥ 2, i+j ≥ 4; Y2 satisfy i ≥ 1, i+j ≥ 3.
  - The Jacobian is r^3δ^2.
  - S17: det H/r = ∓6kA_r + O(r).
  - Exponents −3−7γ, −10, −13/2, and D_N^2 = c/(2(N+4)).
- **(NON-CERTIFYING)** At L = 4 the least eigenvalue is 0.30177, and the Schur eigenvalues are 2.2214 and 9.5435, matching the review. The two-scale Schur minimum over the grid is 0.360.
- **Manifest validator.** `landing_claims_check.py` passes (claim_count 10).

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U6-F1 | minor | CONFIRMED | Math- `reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md:37` (S17), :38 (S18) | “The transverse pins force `f_{xz}` at both pins to be `O(r)` in `Lᵖ`, because `f_{xxz}(0)` has bounded moments.” / “Markov removes an `O(rᵖ)` probability…”. Both need conditional L^p control, under Q_r, of fixed-point remainders. Addendum:164 only asserts it, and the same session's REVIEW.md:114 lists random-ceiling moments as open. The ACCEPT survives, because \|E_Q R\| ≤ sd(R)(v_r^T Cov(U_r)^{-1} v_r)^{1/2} = O(r^m), but that bridge is not written. Open PR46 marks L-TWO-SCALE closed and L-RANDOM-M-CEILING open without recording any dependency between them. This ground supports the LANDING_CLAIMS fixed-annulus entry. | Add a note that S16–S18 use only fixed-point linear remainders, and that this does not discharge the random-ceiling gap. |
| U6-F4 | minor | CONFIRMED | Math- `reviews/replacement_20260925_pr19_pr21/REVIEW.md:29` | “\| Independence claimed \| Provider-level nonauthor status relative to the OpenAI author lane. …\|” rests on self-declaration. None of the three files from session bc-ff620630 (this one, TWO_SCALE_REVIEW, and the U7a D2 review) states zero organizational credit, while pr16:35, pr22:33, pr25_contact_kernel:32, pr25_typed_transfer:34, pr28:31 and p15:36 all do. Owner note: “provenance is reviewer-declared, not independently attested” (issue 23 comment 5839813890). No downstream surface over-claims. | Add “Organizational independence is not awarded.” |
| U6-F2 | nit | CONFIRMED | same file:99 | “Pin specialization (3), the determinant-one shear, the minor (6), and the Jacobian `r^5`”. In check_interfaces.py, line 230 is `assert spec[0] == s0 - r * r * 0 / 8`, with V2 a literal 0. Line 236 is `assert Q(2, 7) ** 5 == Q(2, 7) ** 3 * Q(2, 7) ** 2`. The shear's determinant is never computed. The mathematics is true. | Call these spot checks, or replace them with generic-data checks. |
| U6-F3 | nit | CONFIRMED | same file:61, :102 | The sharp fixture is actually x^5 − z^2 with b = 2, and all 22 scan fixtures silently add −z^2 (script:243, :254). Re-running without the −z^2 gives the same maxima. | Disclose the fixtures exactly. |
| U6-F5 | nit | CONFIRMED | TWO_SCALE_REVIEW.md:54 | The Stecconi Thm 29 check is recorded against an unversioned ar5iv rendering, while addendum:181 pins “arXiv:2103.10853v1, Theorem 29, printed p.20”. | Record the version and a hash of what was fetched. |
| U6-F6 | nit | CONFIRMED | Math- `PROOF_INDEX.md:23` | No scope qualifier; the review's exclusions are at :11, :25 and :60. The museum `d5-two-scale` card quotes the line verbatim. | Append the scope sentence. |
| U6-F7 | nit | CONFIRMED | same dir `check_two_scale.py:27` | The ledger asserts are bare integers, the circle bound is sampled at 5 points, and the S10–S12 remainder claim has no check in the reviewer's script. | Replace with symbolic checks, or label as reminders. |

**Not verified:**
- Stecconi Thm 29 (egress blocked).
- Analytic L^p bounds for the non-polynomial periodized field.
- A certifying Schur floor over O(2).
- Provider identity.

### U7: D2 cumulative-correction review and P15 full-price review

**What they claim.**
- **U7a** accepts the author's CUMULATIVE_TRANSFER_CORRECTION.md @d06a562 (blob 044ac5fd, 3272 B, sha256 83f65339…), a correction to equation (D2) of NOTE §D.
- **U7b** accepts Theorem F of `full_price_20260924/PROOF.md` @53320cbc (blob 582180e4, 11352 B, sha256 87521901…) in six slices, with sharp ρ* = 1/(3−log(3·e−2)).

**What was independently re-verified.**
- **U7a (D2 correction).**
  - Custody is exact, and the script prints “correction arithmetic holds”.
  - ℓ = 1/4, N = 1/8, N^3 = 1/512 against (ℓ^{2/3}/2)^3 = 1/128; the ratio cubed is 1/4.
  - The domination identity simplifies to 0 (sympy).
  - A step the review did not check: consistency with NOTE §D3, N_local ~ (3/2)C ℓ^{2/3}.
  - A nontrivial instance (NON-CERTIFYING) gives 0.75442, 0.75031, 0.750016, converging to 0.75.
- **U7b (P15) custody and runs.**
  - Review: blob 07db19f8 (15569 B, sha256 691ea0db), identical at cab2db5, 1e1114f5 and main.
  - Script: sha256 68c66cbe, matching :58; it passes 7 certificate lines.
  - The author suite passes 36 tests and 7 mutants, and RESULTS.json is byte-identical on regeneration.
- **U7b (P15) mathematics.**
  - H_(3,1) = 3 − log(3·e−2), exact.
  - ρ* = 0.845479817248986720678… (NON-CERTIFYING; the 20-decimal window is certified by the review's exact script, which was re-run).
  - Misreadings of the ASCII form give different values: 0.03 → 0.15369, and 3e^{-2} → 0.25632.
  - Rational certificates: 31967/11760, 11/23520, 26081/933120, 197/32.
  - F10 is exact for a = 0..24, and the augmentation identity is exact for n < 15.
  - The minimum hazard equals h* (NON-CERTIFYING).
  - MILP on 6 instances (NON-CERTIFYING): max(covercost − min(1, ρ*·hazard)) = 0.0. In the demand-one cases the cost is 1, above the right-hand sides 0.431 and 0.246.

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U7-F2 | minor | CONFIRMED | main issue #74 comment 5842112010 (the link at main STATUS.md:14) | “merged Math PR63 records a cross-provider nonauthor ACCEPT … The p=1 strictness sentence is corrected to interior 1/2<p<1”. This drops REVIEW.md:36 “Organizational independence is not awarded”. It also calls the sentence “corrected” when PROOF.md is byte-unchanged (blob 582180e4): REVIEW.md:109 says the endpoint “does not require an amendment”. | Say “xAI/Grok nonauthor technical ACCEPT (provider-separated; organizational independence not awarded); PROOF.md:81–85 is one endpoint too wide and needs no amendment”. |
| U7-F3 | minor | CONFIRMED | main `STATUS.md:14`; Math- `claims/LANDING_CLAIMS.json:333`; U7b REVIEW.md:115 | “sharp `rho*=1/(3-log(3e-2))`”. The ASCII `3e-2` is ambiguous: as a Python literal it is 0.03, which gives 0.15369. ASCII_3E_MINUS_2.md (Math- 5ed3b45) is linked only from PROOF_INDEX:28. The same form appears in Math- README:17, RESULTS.json:3, full_price.py:174, and main RESEARCH_INDEX:51, museum.json:458 and status.json:41. | Write `3*e-2`, or link the note from these surfaces. |
| U7-F4 | minor | CONFIRMED | Math- `frontiers/downstream_gate_20260925/GRAPH.json:215` | The node `math.p15-full-price` still has “AUTHOR_SIDE_CANDIDATE”, fingerprint “full-price-20260924”, and notes “…; review open”, with no review_* fields. LANDING says REVIEWED_SCOPED and STATUS says ACCEPT. The graph was last changed at 1a8fdcf, before 1e1114f and 8101de4. PROOF_INDEX:28 says the review “leaves the landing disposition untouched”, but 8101de4 flipped it. This is an under-claim. | Reconcile the D6 node with review_* fields, following the D5 node's convention. |
| U7-F5 | minor | CONFIRMED | main issue #67 comment 5841270276 (and 5841782206; PROOF_INDEX:18) | “\| R1 coupling \| §2 Fourier summability; §3 uniform positive-definiteness of the contact covariance on a fixed band \| Read and used. This is not an acceptance of Theorems A–C. \|”. The owner had asked for IMPORTED-OPEN labels (5836435928, 5841269622). §§2–3 were later reopened as A1 (5841766952), and 5850251715 says “Still AMEND: A2/(3.5) as reviewed inputs”. STATUS D2 and PROOF_INDEX:18 do not mention this input. No mathematical defect was found in it. | List the parent §2/§3 inputs as consumed, and either accept them or mark them IMPORTED-OPEN. |
| U7-F1 | minor | PLAUSIBLE | Math- `reviews/d2_cumulative_correction_20260925/REVIEW.md:1`, :3 | “# Review of the cumulative transfer correction (D2)”. Here (D2) is an equation label in NOTE §D, and it collides with program layer D2 (Theorem R). The review's “density theorem D1” likewise collides with program D1. PR #43 supports no STATUS row. The line-3 bc7d754 sentence is a session-continuity disclaimer. No surface actually conflates the two D2s today. | Add a disambiguation line, and explain the line-3 sentence. |
| U7-F6 | nit | CONFIRMED | same file:19 | “\| Provider \| xAI. The correction is an OpenAI-lane note. This is not an OpenAI session. \|”. There is no zero-organizational-credit statement and no account-wrapper row, and the file does not say that the PR32 amendment it is bound to (bc-89801c5e) came from the same provider. The merge commit 7a6d58f does carry the disclaimer. | Add those three facts. |
| U7-F7 | nit | CONFIRMED | Math- `reviews/p15_full_price_nonauthor_20260926/REVIEW.md:109` | The F10 endpoint remark is unattributed. It was pre-supplied in the Math- issue #57 body, first raised in PR18 commit d87637d and main #74 comment 5838299792, and already checked by lane session bc-075f842d in PR18. The review never claims it as new. | Attribute it. |
| U7-F8 | nit | CONFIRMED | same file:91 | “Equality holds in (F7) at `p_i=1-e^{-1}` and `c_i=1`.” (F7) contains no c; the c_i = 1 condition belongs to (F8). The wording is copied from PROOF.md:73. | Fix both files. |

**Not verified:**
- Provenance.
- The brute-force P15 checks, which are NON-CERTIFYING.
- The original P15-B Drive source.
- CI logs.

### U8: open PR #52 and PR #55

**What they claim.**
- **PR52 (run bc-44e86e32)** is a second ACCEPT of the same D2 correction blob. It calls itself a “cross-provider technical review” and says organizational independence is not awarded.
- **PR55 (session bc-ff620630)** reviews the lane-authored PR53 note @9a6f8a66, with six ACCEPTs and one AMEND (the inner disk, and with it the summed r^2 log(1/r) bound).

**What was independently re-verified.**
- **PR52 custody.** Blob 044ac5fd, 3272 B, sha256 83f65339…. The script passes.
- **PR52 identities (symbolic).**
  - The overlap constant 1/(mβ) = 1/(α+1).
  - h = 2κr^m gives exactly 2^{-β}.
  - The oscillatory density limits are 1/2 and 1/4, against the D1 target 1/3.
- **PR52: a step the review did not check.** 1 + r sin(1/r) ≥ 3/4 on (0, 1/4), so c0 = 3/4 is legal. Dominated convergence (NON-CERTIFYING): N/ℓ^{2/3} = 0.782 … 1.425 as ℓ goes from 1e-3 to 1e-15, creeping toward 3/2.
- **PR55 custody.** NOTE: blob c396e64b, 7828 B, sha256 f6efc9c2…. Script: blob f683d8e5. PR28: blob 6f317515.
- **PR55 algebra.**
  - Degree-4 pinned jets; drift 6kp(p−1).
  - Minor sum q^4((p−1/2)^2 + q^2/4).
  - 5184/(2p−1)^2, with minimum 2304 at p = −1/4.
- **PR55 inner disk.** det Cov at p = 0 is r^6(q^4+q^6)/4, so √det ~ r^3 q^2/2, and det(JJ^T) = Qt^4/4 for every P. The AMEND is confirmed.
- **PR55 cone (exact contact).** det H_M/r^2 ~ −6.5q, det H_X/r^2 ~ +6.5q, det H_S/r^2 ~ −19. The product is O(r^6 q^2), so the cone's log(1/r) is an artefact of loose bounds.
- **PR55 steep strip.** Taking the review's stated factors literally, the integral is r^3(1/(6r^6) − 2048/3) ~ r^-3. Quadrature (NON-CERTIFYING) gives 4.797e7, 4.797e10, 4.797e13 at r = 1e-2, 1e-3, 1e-4.

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U8-F1 | **major** (verifier passes split 2 major : 1 minor) | CONFIRMED | Math- PR #55 `reviews/d5_pin_neighborhood_20260926/REVIEW.md:72` | “The cone Jacobian `r^3 q^2`, the three-Hessian majorant `O(r^6 M_3^6/q^6)`, and a conditional moment `(O(\|p\|/\|q\|))^6` produce, after division by `Z_r`, an integral `O(r^3 log(1/r))`.” Those factors multiply to r^3 p^6 q^-14 e^{-cp^2/q^2}. After q = pt this integrates to about r^-3/(6κ^6), which diverges. The source (NOTE.md:100) offers only an analogy, so this steep-strip ACCEPT rests entirely on the reviewer's own arithmetic. The conclusion can be recovered: f_zz(S) = f_zz(M) + O(rM3) gives r^3 log(1/r), and the review's own line-50 bound gives r^2/κ. A later comment, 5848720985, “reproduces” the claim only by dropping one q^-6 factor. PR55 is an open draft, and downstream cites it only as “an AMEND review”, so no status rests on this ACCEPT. | Replace the majorant with O(r^2 M3^2 \|p\|/\|q\|) per determinant, show the substitution, and state that the source supplies no derivation for this strip. |
| U8-F2 | minor | CONFIRMED | same file:44 | “Large fixed `κ` and `κ r≤\|q\|≤δ` make `λ_min(Cov Y)≥c>0`”, yet line 60 integrates up to \|q\| = 1/4. The gap can be filled from L L^T = diag((p−1/2)^2 + q^2/4, 1), which gives σ_min^2 ≥ 1/16. | State the uniform-rank bound. |
| U8-F3 | minor | CONFIRMED | same file:94 | “`check_pin_chart.py` checks the minor identity, the transverse minor `1/2`, … the cone exponent `-2-3+5=0`, … the cone product factor `1/κ`, and the inner-square density ratio `2 r^2/q^2`.” Four of these are tautologies (script lines 91, 46, 81–83, 68–72). The load-bearing √det Cov = r^3 q^2/2 appears only in a comment (62–64). | Compute the covariance from the jet map, and label the rest as illustrations. |
| U8-F4 | minor | CONFIRMED | same file:28 (row), :50 | “\| Determinant-product power on that cone \| **ACCEPT** \|” accepts the non-sharp log ledger. It does not acknowledge the two PR53 comments posted before its commit (01:14:54Z): 5841805160 at 01:07:47Z (O(r^6 q^2), the log is removable) and 5841824353 at 01:10:37Z (P5 BLOCKED). By the review's own majorant, the cone is O(r^2/κ) with no log at all. | Record that the row accepts a majorant only, and cite P5. |
| U8-F9 | minor | CONFIRMED | Math- PR #52 `reviews/pr25_d2_correction_confirm_20260926/REVIEW.md:42` | “A prior xAI file … was visible in the working tree. It was not used as the derivation or the verdict.” That prior file is an ACCEPT of the same blob by the same provider, already landed (9a740b5, merged 7a6d58f) and cited at PROOF_INDEX:26. PR52 never says it adds zero independence. The owner requested the re-review (5841652870). Nothing double-counts the two ACCEPTs yet. | Mark PR52 as a confirmation only, or close it as superseded. |
| U8-F5 | nit | CONFIRMED | PR55 REVIEW.md:40 | This accepts PR53 NOTE.md:43–44 “… + O(r^2)”, which omits r q(U(3p^2−1) + 3Vpq + Wq^2)/6 = O(r\|q\|). That is harmless for the accepted bounds, but it matters for any inner-square repair built from the displayed formula. | Note the remainder is O(r\|q\|) + O(r^2). |
| U8-F6 | nit | CONFIRMED | PR55 REVIEW.md:38 | “supported at two or three distinct points”: it is always exactly three. The PR28 §3 positivity result is imported but not listed in the line-7 “Used only for” scope. | Correct the count and the import list. |
| U8-F7 | nit | CONFIRMED | PR55 REVIEW.md:84 | “The pathwise majorant `\|det H_S\|=O(r^2 M_3^2/\|q\|)` is also not `O(r^5)` uniformly…” compares a single factor with the scale of the whole product. | Restate it for the product. |
| U8-F10 | nit | CONFIRMED | PR52 REVIEW.md:124 | “the normalized density equals `1/2` where `cos=1` and `1/4` where `cos=-1`” leaves the quantity and its target (1/3) unnamed. | Name both. |
| U8-F8 | nit | PLAUSIBLE | PR55 title | “Nonauthor review of the PR53 pin-neighborhood chart”. This is Grok reviewing Grok. The repo uses “nonauthor” in a session sense (main docs/PUBLIC_SHOP_SETUP.md:31; Math- SECURITY.md:15), and the body discloses the shared provider. | Adopt a repository-wide same-provider title convention. |

**Not verified:**
- Uniform conditional-moment statements.
- Whether the full punctured-disk count is O(r^3).
- CI runs.

### U9: lane-authored PR #53 and PR #69, same-provider review PR #74, and related PR #87

**What they claim.**
- **PR53 (author, run bc-3524e567):** “the ledger is integrable”, with summed bound C r^2 log(1/r).
- **PR69 (author, run bc-23df1f0a):** the two-point divided-difference microdisk “closes the inner-disk gap”, with a cone product O(r^6 q^2), a pin disk O(r^3) and a microdisk O(r^5).
- **PR74 (reviewer, run bc-28cc6d13):** six ACCEPTs and “No interface is AMEND or COUNTEREXAMPLE.”

**What was independently re-verified.**
- **Custody.** PR69 NOTE: blob 6d6ec7f5, 11120 B, sha256 6ba24ef3; PR69 script: blob a55e1355, 11607 B. PR53 NOTE: blob c396e64b. PR74 script: sha256 52a840db. The PR69 input chart is the PR53 head.
- **Script robustness.** All three scripts print `ok`. With a deliberately broken assert, they still print `ok` under `python -O`.
- **Degree-6 pinned jet.** f_xx(M) = −6kr + r^2 f_xxxx/12 + r^3 f_xxxxx/30, and f_xz(M) = −r f_xxz/2 − r^2 f_xxxz/6 − r^3 f_xxxxz/24.
- **Microdisk determinants.** The r^0–r^2 terms of det H_M vanish. The r^3 term is 3kβ(D + 3Cτ − 24kτ^3), and the r^4 term also carries β. The r^2 term of det H_S is 6k(C − 12kτ^2). So the product is O(r^8β^2).
- **Cone numerators.** Divisible by q at every order in r. On the axis the product is −54k^3 C D^2 r^6 q^2.
- **Constants.**
  - 576Φ − ψ^2 = 143β^4 + 2r^2α^2β^2.
  - mST = −β^2 r^5(2αr−1)/2 and mTQ = −α^2 r^7(αr−1)^2(2αr−1)/24.
  - Checksum (−769/36, 3031/180, −3433/36).
  - Quartic ratios 5184k^2/r^2 and 5184k^2/(37r^2); 5184·4/9 = 2304.
  - The PR74 floor 9/131072 is conservative: a NON-CERTIFYING grid minimum is 1.373e-4.
- **Fifth-order tail.** The f_xxxxx coefficient in G_x is α r^2(αr−1)(5α^2r^2 + 5αr − 4)/120.
- **Band integral.** ∫_{\|β\|≤r\|α\|} dβ/(β^2 + r^2α^2) = π/(2r\|α\|).
- **Planar Bargmann–Fock exact-conditioning Monte Carlo (NON-CERTIFYING).**
  - On the axis, E\|prod\|/(r^6q^2) = 377/366/365 at r = 0.02 and 378/366/365 at r = 0.01, with E\|dS\|/r^2 ≈ 6.75.
  - In the microdisk, E\|prod\|/(r^8β^2) ≈ 3597–4090.
  - On the quartic band at β = 0, E\|prod\|/α^2 ≈ 9.4e-4 → 8.1e-4.
- **Independence of the PR74 script.** Its contact() is PR69's contact_scaled() renamed; only check_gram_factor is new.

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U9-F1 | **major** (split 2 major : 1 minor) | CONFIRMED | Math- PR #74 `reviews/pr69_pin_microdisk_nonauthor_20260926/REVIEW.md:153` | “The soft-factor count at `q=rβ` gives the product moment `O(r^8 β^2)`.” The review proved the soft-factor count only on \|q\| ≥ Kr (line 141, “because r≤\|q\|/K”). Read literally at q = rβ it gives O(r^8(1+\|β\|)^2), and the transverse-square integral then diverges logarithmically. β-divisibility at every order in r is never derived or checked; PR69 NOTE:143 only asserts it. The conclusion is true: det(H)h = −adj(H)v with \|v\| ≤ M3\|h\|^2/2, plus ‖H‖ = O(r). There is no downstream use: PR74 is an open draft, STATUS:21 is AMEND, and PROOF_INDEX:44 says “NO COMPLETE PROOF YET”. | Use the adjugate identity, and display the r^3 coefficient. |
| U9-F2 | **major** (split 2 major : 1 minor) | CONFIRMED | PR74 REVIEW.md:157 (also PR69 NOTE.md:165, :167) | “A degree-6 polynomial in the shifted quartic jet produces only a power of `1/r`. Multiplied by the area of the band, the contribution is `O(r^{-C}) exp(-c/r^2)`…”. The density C/(r^5ψ), with ψ ≍ r^2α^2, is unbounded as α → 0, and the Gaussian cost does not decay in α. Without a determinant factor proportional to α^2, the band integral diverges like log(1/ε). The axial band implicitly needs O(r^8β^2)(α/β)^N (a weaker point). The later repair comment 5848849115 repeats the gap. The conclusion is true, via \|h\|^2 repulsion. | Write the band integrand with E[\|dets\| \| G=0] ≤ C r^{-C'} r^4 α^2, and show the α^2/ψ cancellation. |
| U9-F4 | **major** (split 2 major : 1 minor) | CONFIRMED | Math- PR #53 `reviews/pin_neighborhood_recon_20260926/NOTE.md:7` (also :102, :86) | “Off that point the normalized gradient has an explicit rank and Jacobian ledger, and the ledger is integrable. No further base blow-up is required to state a count lemma.” The normalized-gradient density is ≍ 1/ψ on \|α\| ≲ \|β\|. With the note's uniform O(r^5) determinant count, the inner-square integral therefore diverges logarithmically, so the headline r^2 log(1/r) is unproved as argued (not shown false). The line-86 mechanism is also wrong: det H_S = Θ(r^2). algebra_check.py:201 and :203 are tautologies. Downstream already records this as AMEND/BLOCKED, but the note itself was never amended. | Retitle Conclusion (A) as conditional, and correct the H_S mechanism. |
| U9-F3 | minor (reviewer: major) | CONFIRMED | Math- PR #87 `incoming/grok-cycle4-20260926/harper/D5_OBSTRUCTION_LEDGER.md:160` (also :115–119, :140–142) | “\| #58 product `O(r^6 q^2)` \| extra `r` \| error \| over-slaves `det H_S` by copying the annulus \|” and line 118 “f_ss(S) = a_S = O_p(1)”. Both are false on the conditioned event: f_ss(S) = r(C − Dq/2) + O(r^2), and det H_S = r^2(6kC − 3kDq − C^2q^2/4) + O(r^3). Issue #58 comment 5850921404's “O(r^3 q^2)” uses physical q; the correct order there is r^4 q_phys^2, and its Monte Carlo was run at a single r. The file has **no Grok 4.7 self-ID**. PR87 is unmerged and says “Scientific effect NONE”, and it errs in the conservative direction. | Withdraw the “error” row, and cite PR69/PR74 and comments 5848591450 and 5848849115. |
| U9-F5 | minor | CONFIRMED | Math- PR #69 `reviews/d5_pin_microdisk_20260926/NOTE.md:108` (repeated at PR74 REVIEW.md:87) | “The fifth-derivative remainder in `G_x` is `O(r ψ^2)` on this square”. This is false: the pin-solution tails are O(r^2(\|α\|+\|β\|)), about αr^2/30 on the axis. The correct bound is O(r·√ψ), and the frame conclusion survives it. | Restate the bound, and run the Weyl argument with ‖E‖ ≤ C r h. |
| U9-F6 | minor | CONFIRMED | PR74 REVIEW.md:120 | “The jet floor `λ` multiplies this Gram determinant by `λ^2` once the conditional covariance of `(S,T,Q)` is at least `λ I`.” This does not follow: the other jets and the remainders are correlated with (S,T,Q). A valid argument needs s_min(B) ≥ c·h (from the determinant plus a trace bound) and Weyl's inequality. The conclusion is true. | Use the singular-value/Weyl argument. |
| U9-F7 | minor | CONFIRMED | PR74 `algebra_check.py:2` | “Independent finite checks for the PR69 pin-microdisk review.” In fact contact(), pins(), ev() and the ledger check are the author's code re-typed, with the same grids and checksum and 4 of 5 mutants. The author's solve_st and pin checks were dropped. REVIEW.md:49 says “expanded from the six pins”, but no shipped artifact does this. | Derive inside the review script, and drop the word “independent”. |
| U9-F8 | minor | CONFIRMED | PR69 NOTE.md:217 | “Six semantic mutants of those coefficients are refused.” Three of them exercise no code (script:291–303: `if Q(-1, 2) == Q(-1)`, `if -2 - 3 + 5 + 2 == 3`, `if Q(1, 2) + Q(1, 2) < 1`). The bare asserts in all three scripts are skipped under `-O`. | Mutate the real functions, and use explicit `raise`. |
| U9-F12 | minor | CONFIRMED | PR69 NOTE.md:205 (also :26, :202; PR74 REVIEW.md:45, :183) | “Continuing either chart out to distance `A-1/2>1/2` lands in the separated-site regime PR28/PR44 already estimate.” This holds only on the outward ray: M + (A−1/2, 0) has ρ = A − 1. The imported region is also misstated as “ρ≥A>1”, whereas PR28 covers only A ≤ ρ ≤ B, and ρ > B is never mentioned. | State A ≤ ρ ≤ B, and delete or qualify the sentence. |
| U9-F9 | minor | PLAUSIBLE | PR69 NOTE.md:7; Math- issue #58 comment 5841875709 | “The nested microdisk closes the inner-disk gap left by the pin chart. No deeper blow-up is required.” The continuum transfer and the tilt estimates are asserted, not proved. Later records are split: PR74 (same provider) gives ACCEPT, while 5848591450, 5848849115 and 5848850610 give AMEND. | Word it as a conditional author-side majorant with listed obligations. |
| U9-F10 | minor | PLAUSIBLE | PR69 NOTE.md:17; PR74 REVIEW.md:43 | The q-soft cancellation, O(r^6q^2) and the divided-difference frame were first stated in PR53 comments 5841805160 and 5841824353. PR60 (01:22:30Z) has the same frame and floor (81/589824, giving 9/131072). PR69 credits this only indirectly, through the issue #58 task. PR74 omits PR60, PR55 and P5. | Credit these sources. |
| U9-F11 | minor | PLAUSIBLE | PR74 REVIEW.md:72 | “No interface is AMEND or COUNTEREXAMPLE.” After the freeze: comment 5848455418 gave AMEND on the PR74 record (though its mathematical dispositions survive), 5848591450 gave AMEND on PR69, 5848849115 and 5848850610 say PR69 “remains AMEND”, and 5 automated review threads are unresolved. There is no successor. No status surface depends on PR74. | Supersede PR74 with an AMEND record, or close it. |
| U9-F13 | nit | CONFIRMED | PR74 REVIEW.md:1 | “# Nonauthor review — pin-microdisk divided-difference candidate”. This is a same-provider review, and its title and path cannot be told apart from the lane's cross-provider reviews; only line 39 discloses the difference. | Use a same-provider title. |

**Not verified:**
- The K_L covariance floor and moments (only Bargmann–Fock was checked).
- The block-Gaussian tilt.
- The symmetric S-chart.
- Kac–Rice validity for the tilted count.

### U10: open PR #72 and PR #73

**What they claim.**
- **PR72 (run bc-4428875b)** reviews main PR #125 @00c9d08a (the shrinking-witness m=2 bound). It gives five ACCEPTs, plus an AMEND on the transition from the fixed-η bound. It does not claim provider separation, because the PR125 model is undeclared.
- **PR73 (run bc-be016086)** reviews Math- PR #36 @ef312fed (kernel tails, self-identified “**Author:** OpenAI / ChatGPT”). It accepts Q1–Q4 at limiting-kernel scope.

**What was independently re-verified.**
- **PR72 script and custody.** Script sha256 2ffe3a18… matches line 58, and the script passes. Note: blob d29d00f2 (12367 B). Test: blob 8112f93b. PROOF.md: 18355 B.
- **PR72 identities.**
  - det[[I,0],[I,δI]] = δ^d for d = 1..6.
  - The gap is −μδ^3/12 − νδ^4/24 − ρδ^5/80.
  - \|det H\| ≤ ‖Hu‖ ‖H‖_F^{d−1}: 0 violations in 599 exact cases.
  - Bargmann–Fock: Var f111 = 15, Cov = −3, residual 6.
  - 1/24 − (51/250)^2 = 19/375000.
- **PR72: steps the review did not check.**
  - Gap variance δ^6/24 + δ^8/96 + O(δ^10).
  - Pair density p·δ^2 → √3/(12π^2) = 0.0146245.
  - E\|det H_x det H_y\|/δ^2 → 4.
  - Ledger (3/2)κ^{-2/3}w^{5/3} − w^2/(κR) for general κ and R.
- **PR73 script and custody.** 12 tests pass; the script sha256 is 5c439b68…, but REVIEW.md does not pin it. Note: blob 26890787 (10695 B). Main now carries blob 4d03f110 (10870 B).
- **PR73: steps the review did not check.** (1.1) and all three Hessian determinants derived from PR25 (B1).
- **PR73 identities.**
  - J = 27392/315 in both integration orders, and 2916J = 8875008/35.
  - Exact 199×1601 grid: 0 mismatches.
  - Counterexample: R = 0, P = 255214387799/8303765625.
  - Bargmann–Fock: a \| even ~ N(−b, 2), Ω = diag(2,2,6), p_a0·C_odd = e^{−b^2/4}/(16√3π^2).
  - C_L(0) = 184896√3/(35π^2) = 927.0867058 (the decimal is NON-CERTIFYING).
  - Recurrence (5.1) and tail primitive (2.3).

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U10-F1 | minor | CONFIRMED | Math- PR #72 `reviews/pr125_shrinking_witness_nonauthor_20260926/algebra_check.py:1`; REVIEW.md:46 | “Independent certificates for the main PR125 shrinking-witness review.” / “Author unit tests were not used as evidence.” Five of the six check groups are the author's test with light edits: det(), jacobian_matrix, axial_jets, the parameter sets (2/7, 3, −5, 11), the 2×2 samples, and `15 - ((-3) ** 2)`. REVIEW.md:103 cites “the three `2×2` samples in the note”, but those exist only in the test. | Call it a re-run of the author's locks, or derive from first principles. |
| U10-F3 | minor | CONFIRMED | PR72 REVIEW.md:161 | “Letting the lower cutoff `η` tend to zero does not move the pairs with `δ` of order `δ_*` onto that scale.” Under the review's own index-blind ledger, the pair constant above the kink grows like 1/η: ∫_η^R w^2/(κs^2) ds = w^2(R−η)/(Rηκ). So the bound is not uniformly O(r^6) for any η → 0. This is heuristic; an index-aware ledger could change it. | Add one sentence saying so. |
| U10-F5 | minor | CONFIRMED | PR72 REVIEW.md:42; PR73 REVIEW.md:18 | Neither record has been updated since the later comments: PR72's 5848502620 has items 2 and 4 marked required, and PR73's 5848453668 has items 1–5. PR73's state is still “ACTIVE”, although its claim lapsed at 2026-09-26T03:47:50Z. math-downstream-gates has never run on either head. PR72's source exposure excludes PR125's non-additive RESEARCH_INDEX edit, even though line 13 names the whole PR as the object. | Answer or fold in the items, and give PR73 a terminal state. |
| U10-F6 | minor | CONFIRMED | Math- PR #73 `reviews/pr36_kernel_tail_nonauthor_20260926/REVIEW.md:21` | “The same note bytes are on `main` at `1e1114f5…`.” This is stale since 718029c (PR #59) changed line 11 to BLOCKED_ABSENT. The review says it read the note “in full”, yet it never flagged the absent `TRANSVERSE_CONTACT_ASYMPTOTIC.md`; issue #56 was already open (01:16:30Z) before the claim (01:47:50Z). The gap is not load-bearing for Q1–Q4. | Record the one-line difference and the absent exposition. |
| U10-F7 | minor | PLAUSIBLE | PR73 REVIEW.md:50 | “None of its calculations was used as an input.” Yet the Q2 parity paragraph has similarity ratio 0.69 to the earlier xAI ACCEPT 5841035707, with near-verbatim sentences. This is a second ACCEPT by the same provider with source exposure, and the two records number Q1–Q4 differently. PR73 does add real derivations of its own. | Count PR73 and 5841035707 as one xAI-provider confirmation, and map the Q numbers to interface names. |
| U10-F2 | nit | CONFIRMED | PR72 `algebra_check.py:151–152` | κ = R = 1 are fixed, so the κ^{-2/3} and 1/R dependence is never exercised, yet REVIEW.md:165 says “certified”. Line 131 hard-codes 15 and −3. | Loop over κ and R, and derive the moments. |
| U10-F4 | nit | CONFIRMED | PR72 REVIEW.md:180 | “Issue 76 being closed…” is ambiguous: Math- #76 is a gate-rename PR, and the intended object is main issue #76 (comment 5841783172). | Write “main issue 76”. |
| U10-F8 | nit | CONFIRMED | PR73 REVIEW.md:123 | “The same bound does not extend to `u<-1`.” One point does not refute an integral bound. The claim is nonetheless true by a tube argument: on the tube P ≥ 1.32 and Q_E ≤ C/v^4, so the integral is ≳ \|v\|^3 exp(−Ck^2/(2v^4)). | Add the tube bound. |
| U10-F9 | nit | PLAUSIBLE | PR73 `algebra_check.py:396` (docstring :8) | The floating-point tests are not marked NON-CERTIFYING in the script. The coefficient test restates integer arithmetic. REVIEW.md pins no script digest. Q3 lacks the compactness sentence for a uniform λ and m(E). | Label, derive, pin, and add the sentence. |

**Not verified:**
- Provider identity, and the authoring model of PR125.
- PR25's finite-r (C1)–(C5) convergence.
- Ω(E) and m(E) for the periodized covariance.
- The uniform C^q-moment lemma.
- CI.

### U11: main issue comments (#63, #67, #116), main PR #128, Math- issue #58

**What they claim.**
- **main #63, comment 5841570965 (session bc-ce4bf0bc).** ACCEPT on D1-A–E of the parent `UNIFORM_MATRIX_CAP_AND_LIFETIME.md` @703e947 (blob dfed3b8d, 40261 B, sha256 9350ad6e…). It adds: “Accepting D1-A–E removes the parent-import block on #67 (R5)/(R6).”
- **main #67, comment 5841270276 (session bc-81faa729).** ACCEPT on D2 R1–R4. It marks the cap import IMPORTED-OPEN.
- **main #67, comment 5841782206 (session bc-dfbce246).** The R5/R6 delta ACCEPT: “No successor-specific gap turned up.”
- **main #116.** The C101/C103 audits, including the C1 counterexample and “PREMIS-Z-LOWER is BLOCKED”.
- **main PR #128 (lane-authored).** The frame-uniform envelope ρ ≤ C k r^3 \|x\|^{-6}.
- **Math- issue #58, comment 5841875709.** “closed the inner-disk gap”.

**What was independently re-verified.**
- **Custody.**

  | Object | Blob | Bytes | sha256 |
  |---|---|---|---|
  | Parent | dfed3b8d | 40261 B | 9350ad6e…84bc7 |
  | LIFETIME_REMAINDER.md | 247b3ecf | 17734 B | 380b7d0a… (only commit 8a13ee9) |
  | Cap file | 0633aca3 | 15160 B | 0bf922b9… |
  | PR128 PROOF.md | 59f637fd | 14340 B | b15b4395…1f90b |

  PR128's tests pass 6/6 in both normal and `-O` mode.
- **D1-C.** \|det T_r\| = 12 r^{−(d+3)} for d = 2..5. The target is (b − kr^3/2, −kr^2, 0, 12k). The ledger sums to 1.
- **D1-E.** The Gamma-integral ratio is exactly 1 (0.181455968 at τ = 1). The amplitude exponent is −2/3.
- **D1-D.** Coercivity holds: min_b = k^2(574 − r^6)/4 ≥ 0 for r ≤ 1.
- **PR64 congruence.** diag(√r, I) H diag(√r, I) has off-diagonal entries r^{3/2}β; the D^{-1} convention gives √r·β.
- **D2 R1.** U3 = f''' + (r^2/40) f^{(5)} + (r^4/4480) f^{(7)}.
- **D2 R4–R6.**
  - rδ^2 = kδ^3.
  - Missing mass 3ℓ^{7/3}/(7r0^7).
  - η^{7/3} = ℓ^{7/18} and δ(η) = ℓ^{1/9}.
  - ∫_0^t cℓ^{−1/3} = (3/2)c t^{2/3}.
- **Cap rational constants.** u = 24/121, mu = 48/121, F'' > 2184415/7086244 > 1/4, drop 9r^3/32, radial 4400/121. The ridge and face logic was not fully re-derived.
- **#116.**
  - C101 blob 7c170c31 (3957 B) names no lemma path.
  - All 9 source-map blobs resolve.
  - f_yy \| pins ~ N(−6/5, 2) exactly, and z0 = 3.2309785352870 (NON-CERTIFYING).
- **PR128 jet identities.** Gram det = z^8(9x^4 + 4x^2z^2 + z^4)/576. The quartic minus (x^2+z^2)^2 equals 8x^4 + 2x^2z^2. The axial minor is x^10/5760.
- **PR128 §5 (NON-CERTIFYING periodized regression).** At L = 2, θ = 15°: τ = 2.305, γ = −0.333, zero ray at \|x1\|/s ≈ 0.925. At L = 3, θ = 30°: zero ray at 0.681.
- **Math-#58 model (exact-rational deterministic jet model).** prod/(r^6q^2) ≈ 66.4–71.3, stable for r = 1e-2..1e-4.

| ID | Sev. | Status | File:line | Problem (quote; evidence) | Fix |
|---|---|---|---|---|---|
| U11-F2 | **major** (3 major) | CONFIRMED | main issue #63 comment 5841570965 (D1-A), carried by main #67 comment 5841782206 | D1-A: “On \(G_r\), MARKED-CYLINDER-CAP lines 23–28 and 157–163 give maximin level \(s\) and identify \(S\)”. Delta: “No successor-specific gap turned up.” Cap §§2–4 were accepted by citing only the theorem statement and the §5 conclusion. The cap was an AUTHOR_SIDE_CANDIDATE (review_issue 63) with no filed nonauthor review. Earlier lane reviews had left it unaccepted: PR112 comment 5839550742 says “does not accept Theorem A or the cap theorem”. The first D2 review (5841270276) marked it IMPORTED-OPEN, and the delta closed it. STATUS:11 and PROOF_INDEX:18 rest on this without naming it. Exact checks of the cap constants pass, so no mathematical defect was found. The owner's 5841830743 did permit #67 to consume D1-A. | Mark the cap as IMPORTED-OPEN, or bind it to a filed review of cap §§2–4, and cite the dependency in the STATUS D2 row. |
| U11-F3 | **major** (3 major) | CONFIRMED | main `STATUS.md:11` vs Math- `claims/LANDING_CLAIMS.json:112–113` vs `GRAPH.json:125–134` | STATUS lists “**D2 — unrestricted lifetime remainder**” under “ACCEPT — scoped”. LANDING says “"disposition": "FAIL_CLOSED"” with reason “theorem remains fail-closed because parent/global elder/Kac-Rice interfaces are still analytically open on #63”. Math- README:10 says “parent interfaces remain under review”. main RESEARCH_INDEX:24 says “Author-side O(1) remainder”. The GRAPH D2 node (line 128, AUTHOR_SIDE_CANDIDATE) has no review_* fields, unlike the D5 node. LANDING's stated reason is stale for Kac–Rice and elder, but the disposition itself is defensible, because R2 imports §§2–3 (A1, which was reopened). Commit 8101de4 (01:29Z) flipped D6 but left lifetime alone. | Run a source-bound reconciliation that names the remaining IMPORTED-OPEN items, or narrow the STATUS row. |
| U11-F7 | **major** (split 2 major : 1 minor) | PLAUSIBLE | main PR #128 `notes/intermediate_scale_20260926/PROOF.md:115` (also :109, :133) | “On `\|x_1\|≥s/2` one has `m≥ c k_- s^2` for small `s`”. The formula at line 109, “m = 6 k x_1^2 + O(s^3)”, omits the conditional means of f_xxz and f_xzz. Those are nonzero in rotated frames at small L, so line 115 is false at L = 2 (15°) and L = 3 (30°) under “Fix L>0 … for every frame”. This is a proof gap, not a counterexample: σ is bounded away from 0 on those rays, and det Σ ≥ c σ^8 s^12 still gives s^-6. The finite-r transfer at line 133 (“move by `o(1)`”) is unsupported, as comment 5842951887 notes. The unit-level cross-term sub-claim (line 95) was refuted by the verifiers and is dropped. Comments 5842951887 and 5850654979 (which already reports the rotated-frame mean) were never answered. 5841899531 announces “the between-pin intensity satisfies `ρ ≤ C k r^3 \|x\|^{-6}`” without caveat. No downstream use. | Restate as a candidate under AMEND, split on \|m\| ≥ c s^2 or bound (τ, γ) uniformly, and prove a scale-relative Schur transfer. |
| U11-F1 | minor (reviewer: major) | PLAUSIBLE | main issue #63 comment 5841570965 | “Pointwise, \(A_r\to A_0\) and Theorem A on a compact neighborhood … gives \(p_r\to 1\). Dominated convergence yields (13.6) … which is Theorem C.” Theorem A (§§2–7) and (10.3) A_r → A_0 (§§3, 5) are neither derived nor marked IMPORTED-OPEN. The #67 dependency sentence is accurate, however: R5/R6 bypass Theorem A through (R10)–(R12). STATUS:20 and PROOF_INDEX:32 already sit in AMEND/open sections. Theorem C also follows from the accepted D2 Theorem R. | Mark (10.3), (13.6) and Theorem C as conditional, or obtain them from D2. |
| U11-F4 | minor | CONFIRMED | main issue #63 comment 5841570965 | “\| D1-B §9 \| **ACCEPT** \| 298–308 \|” accepts the unrepaired text. It does not mention the lane's own earlier AMEND (PR112 comment 5839550742) or the repair REPAIR.md (694b7ff / fe9b9ce4), which was C1–C6 accepted in Math PR37 comment 5841032512. PROOF_INDEX:33 still says “nonauthor re-review required”; that line was added (6e4085a) after the PR37 ACCEPT. | Bind D1-B to REPAIR.md, and fix PROOF_INDEX:33. |
| U11-F5 | minor | CONFIRMED | main issue #67 comment 5841270276 (repeated in 5841782206) | “…both loss pieces are `O(ℓ^{7/18})` inside the scaled density.” The inner piece is C[ℓ^{7/18} + r0 ℓ^{1/3}] = Θ(ℓ^{1/3}). The conclusions (R18) and Theorem R are unaffected. | Split the statement into the k ≥ η piece, O(ℓ^{7/18}), and the a ≤ k ≤ η piece, O(ℓ^{1/3}). |
| U11-F6 | minor | CONFIRMED | main issue #63 comment 5841570965 | “Provider Cursor, model Grok 4.7 … This is a different provider from that author.” It names Cursor, not xAI. None of the three D2-chain comments states zero organizational credit; only 5841782206 says “Same provider family”. STATUS:11 discloses neither, and in fact no STATUS row does. | Normalize to “xAI Grok 4.7 via Cursor; zero organizational-independence credit”. |
| U11-F8 | minor | PLAUSIBLE | main PR #128 body; PROOF.md:4–5 | The body says “**Reviewer.** xAI, Grok 4.7 …”, while PROOF.md:4 says “**Author of this note:** xAI / Grok 4.7” and :5 “additive nonauthor derivation”. Comment 5841899531 has no self-ID. The PR was never merged. Its tests sit outside `tests/`, so CI does not run them, and two guards are vacuous: the floor test passes with got = 0, and the Gram test passes with 576 changed to 580. | Relabel the author role, add a self-ID, and move the note under Math- with exact guards. |
| U11-F9 | minor | CONFIRMED | main issue #116 comment 5842107966 | “**PREMIS-Z-LOWER is BLOCKED.** The typed maximum–saddle six-pin normalizer used by C103 has no proved bound `Z_r ≥ c r^2` on the whole interval”. D4 remote_window (11), accepted at #76 comment 5841783172 (STATUS:13 “endpoint normalizer”), and parent (5.4)–(5.5) N5/N6, accepted at PR112 comment 5841352633, both supply the r↓0 limit. PR128 §2, from the same lane, imports it 13 minutes later. The error is in the fail-closed direction. | Qualify the verdict to the hardening-tree sources, and cite D4 (11). |
| U11-F10 | minor | CONFIRMED | Math- issue #58 comment 5841875709 | “I picked up issue 58 and closed the inner-disk gap in a new note. … No deeper blow-up is required.” This duplicates U9-F9. The contradiction with the owner-account comment 5850921404 was never reconciled; the exact model supports the scaled r^6 q^2. | Say “symbol-level ledger; continuum transfer AMEND”, and add a self-ID. |
| U11-F11 | minor | CONFIRMED | main `STATUS.md:11` | The accepting reviews are mutable issue comments, each edited after posting (23:59:05→00:07:25, 00:36:03→00:45:40, 01:04:21→01:07:56). Math- `reviews/` has no byte-bound record of them. The proof link is `blob/main`, not pinned to 247b3ecf, though today it resolves to the same bytes. All edits were finished before any downstream citation. The same pattern holds for rows 11–14. | File byte-bound records with sha256, and pin the link. |
| U11-F12 | minor | CONFIRMED | Math- `PROOF_INDEX.md:32` | “[D1-A–E review] accepts its exact Sections 8–15 interfaces.” §§11–12 were not reviewed. The lane's PR112 §3, §4 and §5 N1–N6 ACCEPTs (5841317882, 5841289285, 5841352633), on the closed, never-merged PR #112, are cited nowhere. The §5 N4 comment “exactly sqrt(r) beta_i, as written” silently used the correct convention without flagging the defect in the literal line 169. | Write “Sections 8–10, 13–15”, and record or exclude the PR112 verdicts, binding N4 to the erratum. |

**Not verified:**
- The Kac–Rice hypotheses in arXiv:2304.07424v3 (egress blocked).
- The full cap proof.
- The §8 mesh lemma and measurability.
- The PR128 upper bound.
- The pre-edit text of the edited comments.
- A complete enumeration of the lane's cursor[bot] comments.
- The D3 and D4 STATUS reviews (main #65 comment 5841269490 and #76 comment 5841783172), which were not among Part 3's verified units.

### Cross-cutting patterns (from the findings above; no new findings)

- **Script evidence is over-described.** Shipped “checks” are often tautological, sampled at one point, or copied from the author: U1-F2, U2-F1, U3-F1, U4-F4, U5-F3, U6-F2, U6-F7, U8-F3, U9-F7, U9-F8, U10-F1. None of these hides a false identity.
- **Independence wording is inconsistent.** “Organizational independence is not awarded” is present in most lane files but missing from all three bc-ff620630 files and from the main-issue comments. One comment names “Provider Cursor”. See U6-F4, U7-F6, U11-F6, U3-F9, U1-F4 and U7-F2 (the downstream “cross-provider” wording). “Run id” is used for the model slug in 4 of 9 files (U1-F8, U4-F9).
- **Claim protocol has no effect.** The claim and the result land in one commit, and the state line “ACTIVE” is never closed: U1-F6, U5-F5, U10-F5.
- **Later AMENDs go unanswered.** No lane object was amended after other lanes' AMENDs: U9-F11, U10-F5, U11-F7, U8-F4.
- **Scope-free index lines.** PROOF_INDEX:22, :23 and :25 feed museum “ACCEPT-scoped” cards that carry no scope text: U1-F3, U2-F9, U6-F6.

---

## 3. Does any STATUS.md / LANDING_CLAIMS / GRAPH.json acceptance rest on an unsupported Grok verdict?

**Short answer.** No acceptance row rests on a Grok verdict that is *wrong on its own scope*. There is no blocker.

One STATUS acceptance, **D2**, rests on a Grok verdict chain that accepted a load-bearing import (MARKED-CYLINDER-CAP §§2–4) by citation only. It also consumed parent §§2–3 without labelling them. That is a **major under-support, not a demonstrated error**. Separately, the machine registers disagree with STATUS on D2 (and lag on D6).

The task premise that a main STATUS row D5-fixed-annulus rests on these reviews is **inaccurate**: main STATUS.md @dbb9dcf has no D5 ACCEPT row.

| Surface / row | Grok basis | Rests on an unsupported Grok verdict? | Notes (finding IDs) |
|---|---|---|---|
| main STATUS.md:11 **D2** “ACCEPT — scoped” (Theorem R) | main #67 comments 5841270276 (bc-81faa729) and 5841782206 (bc-dfbce246, the linked review), building on main #63 comment 5841570965 (bc-ce4bf0bc). **Not** Math- PR #43 (U7a). | **Partly: under-supported, not shown wrong.** The pairing step A_r(1−p_r) ≤ B_r rests on D1-A's citation-only acceptance of the unreviewed cap theorem, which the delta relabelled from IMPORTED-OPEN (U11-F2, major). Parent §§2–3 are consumed as “Read and used” (U7-F5, minor). The row contradicts LANDING FAIL_CLOSED and the GRAPH node's missing review metadata (U11-F3, major). The review record is mutable comments and the proof link is unpinned (U11-F11). The arithmetic of R1–R6 re-derives exactly, the cap constants check, and D2 bypasses Theorem A, so it is **not a blocker**. | U11-F2, U11-F3, U7-F5, U11-F5, U11-F6, U11-F11 |
| main STATUS.md:12 **D3** SIDE24 | main #65 comment 5841269490 (+ reconciliation 5841779222) | **Not determined in Part 3.** That review was not among Part 3's verified units. (LANDING_CLAIMS has `side24-coefficient` HOLD_WITH_DOMAIN; this is noted only as the register state, not as a finding.) | none |
| main STATUS.md:13 **D4** fixed-remote RN | main #76 comment 5841783172 (+ 5841861362) | **Not determined in Part 3**, for the same reason. Incidentally: the lane's own later #116 comment wrongly calls the D4 endpoint normalizer unproved, in the fail-closed direction (U11-F9). PR72 and PROOF_INDEX:20 correctly keep D4's fixed-η exclusion. (LANDING `rn-fixed-remote-window` HOLD_WITH_DOMAIN is the register state only.) | U11-F9 |
| main STATUS.md **D5 fixed-annulus** | none | **No such row exists** at dbb9dcf. The ACCEPT table lists only D2, D3, D4, D6. | premise inaccurate (U2, U5 downstream) |
| main STATUS.md:21 **D5 pin neighborhoods / microdisk** (AMEND/open) | cites PROOF_INDEX + Math- issue #58. PROOF_INDEX:44 cites the U9a PR53 note and **the U8b PR55 review as an AMEND review** | **No.** It is not an acceptance, and it correctly keeps the microdisk incomplete. The defective PR55 steep-strip ACCEPT (U8-F1) and the PR74 ACCEPTs (U9-F1, U9-F2) do not reach it. | U8-F1, U9-F1, U9-F2, U9-F4 |
| main STATUS.md:14 **D6** P15 full price | Math- `reviews/p15_full_price_nonauthor_20260926/REVIEW.md` (cab2db5), via main #74 comment 5842112010 | **No.** The Grok ACCEPT is supported at its stated scope and custody is exact. The defects are wording only: the reconciliation drops the zero-organizational-credit caveat and says “corrected” when the bytes are unchanged (U7-F2), and the ASCII `3e-2` is ambiguous (U7-F3). | U7-F2, U7-F3 |
| main STATUS.md:20 **D1** (AMEND/open) | main #63 comment 5841570965 | **No acceptance row.** “Later review accepted the load-bearing §8–§15 interfaces” overstates the scope: §§11–12 were not reviewed (U11-F12), and D1-D's Theorem C step is conditional (U11-F1). | U11-F1, U11-F12, U11-F4 |
| LANDING_CLAIMS `rn-fixed-annulus-window` REVIEWED_SCOPED | U2 (PR22 review) + U6b (TWO_SCALE_REVIEW, dependency `two-scale-s6-s21`) | **No.** Both verdicts are supported at the stated scope. Minor: the height-disintegrated Kac–Rice (A21) is imported rather than reviewed (U2-F2); the S17–S18 conditional-moment ground is incomplete but repairable (U6-F1); a field-label nit (U2-F7). | U2-F2, U6-F1, U2-F7 |
| LANDING_CLAIMS `p15-full-price` REVIEWED_SCOPED | U7b | **No.** Supported; the ASCII `3e-2` form is ambiguous (U7-F3). The dependency narrowing to (P1), (P2) and Theorem P2 matches the review's Source-exposure paragraph. | U7-F3 |
| LANDING_CLAIMS `lifetime-remainder` FAIL_CLOSED | none | **No:** it does not rest on Grok. It is the conservative side of the D2 inconsistency (U11-F3). | U11-F3 |
| LANDING_CLAIMS, other entries | none | **No.** No entry cites U1, U3, U4, U5, U8, U9 or U10. | none |
| GRAPH `math.rn-fixed-annulus-window` (AUTHOR_SIDE_CANDIDATE; review_disposition ACCEPT; review_provider “xAI/Grok via Cursor”) | U2 | **No.** Supported. | none |
| GRAPH `regional.fixed-annulus.high-jet-route` SUPERSEDED_NONBLOCKING | none. It is a PR #51 author-side inference that uses U2's wording | **Not a Grok verdict.** The review never assessed CH-LIFT, Piece-2 or JETMOD. The inference is unreviewed, but its edges are `required: false`. | U2-F6 |
| GRAPH `math.p15-full-price` | none recorded | **No** (an under-claim): the node still says “review open”. | U7-F4 |
| GRAPH `math.lifetime-remainder` AUTHOR_SIDE_CANDIDATE | none recorded | **No**, but it is inconsistent with STATUS D2 (U11-F3). | U11-F3 |
| GRAPH `math.rn-region.intermediate-r-to-rho`, `math.rn-region.witness-collision` (OPEN_ACTIVE) | none | **No.** PR128 (U11d) and PR72 (U10a) are correctly not consumed. | U11-F7, U10-F3 |
| PROOF_INDEX:18 (D2) | same chain as STATUS D2 | Same answer as STATUS D2: under-supported, major, not a blocker. | U11-F2, U11-F3, U7-F5 |
| PROOF_INDEX:21 (PR28), :22 (PR22), :23 (two-scale), :24 (inner-belt density), :25 (PR16), :26 (D2 correction), :28 (P15) | U5, U2, U6b, U6a, U1, U7a, U7b | **No.** Every verdict is supported at its stated scope. Lines 22, 23 and 25 carry no scope text into the museum's “ACCEPT-scoped” cards (U1-F3, U2-F9, U6-F6). Line 26 has the D2 name collision (U7-F1). Line 21 has contradictory permission wording in its source review (U5-F1). | U1-F3, U2-F9, U6-F6, U7-F1, U5-F1 |
| PROOF_INDEX:32 (D1-A–E), :33 (§9 repair) | U11a | **Open section, no acceptance.** “Sections 8–15” is over-scoped (U11-F12), and line 33 is stale (U11-F4). | U11-F12, U11-F4 |
| PROOF_INDEX:39 (Sections B–C) | U3, U4 | **No** (open/conditional section). “Section A is outside both acceptances” hides R3's and R2's reliance on (A4) (U3-F3, U4-F3). | U3-F3, U4-F3 |
| main museum “ACCEPT-scoped” cards (`d5-fixed-transverse`, `d5-height-window-annulus`, `d5-two-scale`, `d5-all-height-annulus`, `d5-inner-belt-density`, `cumulative-transfer-correction`, `d6-p15-full-price`, `d2-lifetime-remainder`) | U1, U2, U6, U5, U7, U11 | **No unsupported verdict**, but several D5 cards show no scope and have a null status_quote (U1-F3, U2-F9, U6-F6). The museum is marked `scientific_status_authority: false` (U5-F4). | U1-F3, U2-F9, U6-F6, U5-F4 |
