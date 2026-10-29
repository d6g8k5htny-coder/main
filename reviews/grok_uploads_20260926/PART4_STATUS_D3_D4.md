## PART 4 — Cursor-run acceptance reviews cited by STATUS.md rows D3 and D4

**Scope.** This part covers the two cursor[bot] acceptance reviews that main `STATUS.md` at `dbb9dcf64d23ecb45267ce2779f69f17b2fa6265` cites under "ACCEPT — scoped":

- **D3**, line 12: main#65 comment 5841269490, with owner reconciliation 5841779222.
- **D4**, line 13: main#76 comment 5841783172, with owner reconciliation 5841861362.

**Conventions.**

- Every GitHub account in this project is the one shared account `d6g8k5htny-coder`, so authorship rests only on self-identification lines and the cursor.com run-id footers.
- Model names appear here only inside quoted self-IDs.
- The verdicts below are technical only. They give zero credit for organizational independence and change nothing in any repository.
- "Exact" means sympy, `Fraction` or closed-form algebra, and it certifies. Float, mpmath image sums and Monte Carlo (MC) results are marked NON-CERTIFYING.
- Severities have been adjusted by the verifiers. PLAUSIBLE findings are labelled.
- Findings are listed blocker, then major, then minor, then nit. **Neither review has any blocker or major finding.**

**Synthesis spot-checks run this pass.** I re-read the bodies of main#65 comments 5841269490, 5841768057 and 5841779222, main#76 comments 5841782672, 5841783172 and 5841861362, main#116 comments 5842107528 and 5842107966, and Math-#40 comment 5840221357 through GitHub MCP (read-only). Every quote below matches those bodies. I also re-checked the following in local repos:

- the line numbers of STATUS.md@dbb9dcf6 (D3 row = 12, D4 row = 13, D1 row = 20, "Reading rule" = 54–56);
- `sha256(e329fba1:coefficients/side24_v1/PROOF.md)` = `c06daccc…917769`, the same at Math- `origin/main` 03333792;
- `191ea7d5:frontiers/remote_window_20260924/PROOF.md` is 18355 B, sha256 `a332bae9…20cb7`, blob `b383bfcc`, the same at 03333792.

---

### D3 — SIDE24 coefficient calculation (main#65 comment 5841269490)

#### Self-ID basis

- Comment 5841269490 is authored by cursor[bot]. It was created 2026-09-25T23:58:59Z, 4 s after the @cursor request 5841269044 at 23:58:55Z, and last edited 2026-09-26T00:09:32Z.
- Quoted self-ID: *"Provider/model/session: Cursor cloud agent, Grok 4.7 (`grok-4.7-high-fast`), run `bc-be5dc8d9-8a69-4ca9-983b-af79d710c1c1`"*. The cursor.com footer carries the same run id.
- Independence statement, quoted: *"This is a same-session nonauthor check of the stated expression, and parent issue #63 is still open."* and *"No Vault99 reading, no source edits, no child agents. Local interpreter: CPython 3.12.3."*
- Package author (PROOF.md line 3): *"Author: OpenAI / ChatGPT."* So "nonauthor" is accurate at the provider level.
- The model identity is self-reported. Only the bot author and the run footer corroborate it. The review claims no organizational independence, and none exists, because the account is shared.

#### What it claims

C1–C6 are all **VERIFIED**, with the disposition **COEFFICIENT-CALC-REVIEWED / PARENT-IMPORTED-OPEN**:

| Item | Claim |
|---|---|
| C1 | D1 = 4/3 |
| C2 | D2 = 29/6 − √6, with the negative-definite cone kept |
| C3 | Angular/all-direction reduction: the parent (15.2) scalar identity, and formula (1) |
| C4 | SIDE24 image/periodization bound, E = 21175738586478·10⁻¹²⁵ with 60E < 10⁻¹⁰⁸ |
| C5 | Covariance/Schur sandwich at ε = 10⁻¹⁰⁸, with 32ε < 10⁻¹⁰⁶ |
| C6 | Outward Γ/Stirling arithmetic, plus the published 20-digit endpoints for c₂,₂₄ and c₃,₂₄ |

The review also gives four file digests and an explicit "Still parent-open" list:

- the contact Jacobian 12 r^−(d+3) and the limits ±6κ that produce (6κ)²;
- ordered max/saddle Kac–Rice and the elder identification;
- Theorem C;
- finite-radius constants, RN/24-jet and P15.

#### Independently re-verified (specific numbers)

**C1.** From symbolic derivatives of K∞ = e^−|z|²/2 (sympy):
- Cov(V) = diag(3,1[,1]), Cov(G) = I, Var t = 15, Cov(t,G) = (−3,0,…), τ² = 6.
- Given V = 0, A has covariance [[8/3]] for d = 2 and [[8/3,0,2/3],[0,1,0],[2/3,0,8/3]] for d = 3.
- **D1 = 4/3 exactly.**

**C2.** From the derived Cov(s,x,y) = diag(5/3,1,1), the exact cone integral gives **D2 = 29/6 − √6** (residual 0). Also exact:
- ∫₀^a (a−z)² e^−z/2 dz/2 = a² − 4a + 8 − 8e^−a/2;
- E e^−s²/2 = √6/4 = √(3/8);
- half the untruncated moment is 29/6.

A 4·10⁶-sample MC (NON-CERTIFYING) gives D1 ≈ 1.3327 ± 0.0015 and D2 ≈ 2.3842 ± 0.0077, against exact values 1.33333 and 2.38384.

**C3.**
- The ratio 144∫₀^∞ k^{4/3} φ_τ(12k) dk ÷ [Γ(7/6) τ^{4/3} / (24^{1/3}√π)] is exactly 1.
- The assembled c_ref over formula (1) is exactly 1 for d = 2 and d = 3.
- 6^{2/3}/24^{1/3} = (3/2)^{1/3} exactly.
- Hermite-interpolation algebra: with f′(±r/2) = 0 and f(−r/2) − f(r/2) = kr³, we get f_xx(M)/r = −6k, f_xx(S)/r = +6k, f_xxx = 12k, and 4(6k)² k^{−2/3} = 144 k^{4/3}.

**C4.**
- The coefficients of 76(1+y)⁶ − P_q(1+y) are nonnegative for every q ≤ 6. The coefficient sums are 1, 1, 2, 4, 10, 26, 76, and P_q(0) ≤ 15.
- The Taylor sum for e^{288/125} is 10.0141590846 > 10, and e^−288 = 10^−125.0768.
- E = 2.1176·10⁻¹¹², and 60E = 1.27·10⁻¹¹⁰ < 10⁻¹⁰⁸.

**C5.**
- min eig C_ref = 8 − √58 = 0.384227 > 1/3.
- The Hessian block eigenvalues are 2 and d+2.
- The Schur-complement transfer works by the infimum argument.

**C6.**
- mpmath at 60 digits gives Γ(7/6) = 0.927719333630039200708349482534621…, which lies inside the code's interval (width 1.445·10⁻³¹).
- c₂,ref = 0.0734069193060342710301359629…: 1.36·10⁻²² above the published lower endpoint and 9.86·10⁻²¹ below the upper.
- c₃,ref = 0.0417759318405983433429366654…: margins 2.94·10⁻²¹ and 7.06·10⁻²¹.
- B₂…B₂₂ match sympy.

**Package.**
- The 30 shipped tests pass. `coefficient.py` regenerates `ENCLOSURE.json` exactly (JSON-normalized diff is empty).
- (15.1)/(15.2) are transcribed correctly from the parent at Math- 03333792 (sha256 `9350ad6e…`, 40261 B). σ is ordinary surface area.

#### Load-bearing steps the review did not check (checked here)

| Step | Method | Result |
|---|---|---|
| Density-ratio step (3)→(4): (1±ε)^⌈a⌉/(1∓ε)^⌈b⌉ with a = m+2/3+n/2, b = d+n/2 | Exact `Fraction`, ε = 10⁻¹⁰⁸ | (a,b) = (13/6, 5/2) for d=2 and (25/6, 9/2) for d=3. Upper ratio < 1+28ε, lower > 1−13ε. Holds. |
| Lattice shell count behind the 1458e⁻²⁸⁸ bound | Exact: (2j+1)³ − (2j−1)³ = 24j² + 2 ≤ 27j³ | Holds. The actual sums Σ\|n\|⁶e^−288\|n\|² are 4e⁻²⁸⁸ (d=2) and 6e⁻²⁸⁸ (d=3). |
| The (S−1)·D^qφ(0) normalizer term in K₂₄ = N/S | Exact | S − 1 = 4e⁻²⁸⁸ or 6e⁻²⁸⁸, far below the allowance. |
| Periodic covariance in genuinely rotated frames | mpmath at 180 digits, images \|n\|∞ ≤ 2, 3 random orthonormal frames per d (NON-CERTIFYING) | max\|C₂₄ − C_ref\| ≤ 3.1·10⁻¹¹⁷. Relative eigenvalue deviation ≤ 5.2·10⁻¹¹⁸, which is ≤ 5.2·10⁻¹⁰ ε. |

#### Custody

- The four digests and byte counts in the review are exact:
  - `PROOF.md`: 10272 B, `c06daccc…`
  - `coefficient.py`: 8073 B, `03ae6d0f…`
  - `ENCLOSURE.json`: 1090 B, `72b6cd92…`
  - `test_coefficient.py`: 4762 B, `81689b53…`
- The bytes did not change later. The last commit to touch the directory is `d83af85` (2026-09-24). Every remote ref that contains `coefficients/side24_v1` has the identical tree `224848cd`, including Math- main 03333792.
- The museum pin `d6628da0` and the LANDING_CLAIMS pin `760340e9` carry the same PROOF blob `44b66f04`.

#### Findings (blockers first; none are blockers or majors)

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| D3-F3 | minor | PLAUSIBLE | main#65 comment 5841768057 : 0 | The owner asked for a fresh distinct-lane review at 01:02:20Z, then closed #65 94 s later on the earlier review. The request was never answered or withdrawn. | Record on #65 / PROOF_INDEX that 5841768057 was superseded without a response. Also record the second, uncited Cursor review (Math-#40 5840221357). |
| D3-F5 | minor | CONFIRMED | main STATUS.md@dbb9dcf6 : 12 | The row drops the owner-prescribed label COEFFICIENT-CALC-REVIEWED / PARENT-IMPORTED-OPEN and the review's parent-open list. In particular it omits that reading (15.2) as the lifetime constant goes through parent (10.3)/§5, which is reopened A3. | Keep the label. Add to limits: "identification of Eq. 15.2 as the lifetime constant depends on D1 (parent §5/A3, AMEND)". |
| D3-F1 | nit | PLAUSIBLE | main#65 comment 5841779222 : 0 | "independently verifies C1-C6" overstates a self-described "same-session nonauthor check". C1, C2 and C5 follow the author's own route. | Describe it as a nonauthor, different-provider, single-run re-derivation with no organizational independence. |
| D3-F2 | nit | PLAUSIBLE | main#65 comment 5841779222 : 0 | "Parent review #63 has now completed at its exact scope" was reversed 7 min 40 s later (#63 5841830743) and never amended. | Add an additive note on #65. |
| D3-F4 | nit | CONFIRMED | main#65 comment 5841269490 : 0 (C4) | The "worst ratio … about 0.0135" is the majorant ratio 0.013502, not the ratio of ∏\|He\| (0.012817 at \|y\| = 24; supremum 1/76 = 0.013158). The general-\|y\| argument is not stated. | Give the monotone majorant argument and label 0.0135 as the majorant ratio at \|y\| = 24. |
| D3-F6 | nit | CONFIRMED | main STATUS.md@dbb9dcf6 : 12; docs/site/status.json : 30 | The package link uses the moving `Math-/tree/main/…`. | Pin it to `/tree/e329fba1…/coefficients/side24_v1`. |
| D3-F7 | nit | CONFIRMED | main#65 comment 5841269490 : 0 ("Digests and commands") | The command block cannot run as written: it has no `cd Math-` and no `cd coefficients/side24_v1`. | Add both `cd` steps, or use `python3 -m unittest discover -s coefficients/side24_v1`. |
| D3-F8 | nit | CONFIRMED | Math- claims/LANDING_CLAIMS.json : 26 | `"statement_heading": "Theorem"`, but PROOF.md has no such heading. The checker only tests that the key exists. | Point it to "Scope and exact parent", or make the checker verify the heading. |
| D3-F9 | nit | CONFIRMED | Math- frontiers/downstream_gate_20260925/GRAPH.json : 143 | The note "parent unreviewed" is stale (it errs conservative). | Update the note and keep the fail-closed classification. |
| D3-F10 | nit | CONFIRMED | main#65 comment 5841269490 : 0 | "same-session" is never defined. Another repo record uses it for the opposite sense. | Use explicit wording such as "single-run, different-provider, not replicated". |
| D3-F11 | nit | CONFIRMED | main#65 comment 5841269490 : 0 (C3 and parent-open list) | Uses κ where the parent uses k. As written, "4·(6κ)²=144" is dimensionally wrong on its own line. | Write "4(6k)² k^{−2/3} = 144 k^{4/3}" and "−6k at M, +6k at S". |

##### Finding details

**D3-F3 (minor, PLAUSIBLE, process)**
- Quote (5841768057): *"FRESH D3 COEFFICIENT REVIEW REQUEST — … @cursor or another distinct numerical-analysis lane: review immutable `coefficients/side24_v1/PROOF.md` and `coefficient.py` at current Math main. … Recompute at least one expression independently rather than accepting printed digits."*
- Timeline: request at 01:02:20Z, reconciliation 5841779222 at 01:03:54Z, #65 closed at 01:04:08Z. #65 has 6 comments and none comes after 5841779222. The reconciliation neither cites nor retracts the request.
- Verifier correction: the claim that "STATUS rests on one single-session review" is true only of what STATUS cites. A second cursor[bot] review of the same bytes exists: Math-#40 comment 5840221357 (created 21:56:54Z, edited 22:08:26Z, run `bc-98c9f02f-…`), with "C1 through C5 are **VERIFIED** on commit `e329fba1…`".
  - That comment has no provider/model line, so its independence from run bc-be5dc8d9 cannot be established. Both are Cursor-lane runs on the shared account.
  - Its scope is narrower on the parent reduction: *"The absolute reduction of parent equation (15.2) to formula (1) remains a parent input; this review does not accept that theorem."* By contrast, 5841269490 says *"The local algebra from those displayed factors to (15.2) and formula (1) is part of the verified coefficient calculation above."*
- Severity reasoning: the earlier review did recompute Γ(7/6) (by a separate Stirling shift of 48), so the substance of the request was arguably met. This stays a process defect.

**D3-F5 (minor, CONFIRMED, consistency)**
- STATUS.md:12 accepted scope: *"The coefficient expression for SIDE24 in dimensions 2 and 3, including cone moments, all-direction periodization comparison, covariance/Schur transfer, and outward special-function arithmetic"*. Limits: *"Coefficient arithmetic acceptance is separate from any imported parent theorem and from a finite-radius error band"*.
- What is right: the row stays within arithmetic scope and carries a generic disclaimer. The reading rule at lines 54–56 also applies.
- What is missing:
  - the owner-prescribed disposition label from 5836436516 (*"disposition should be COEFFICIENT-CALC-REVIEWED / PARENT-IMPORTED-OPEN"*);
  - the review's "Still parent-open" list;
  - any cross-reference from the D1 row (line 20) to D3.
- Refinements:
  - The constant 144 sits in the step just before (15.2), in c_{d,L} = 144∫…, and comes from (13.6) 4·z₀·k^{−2/3} with z₀ = (6k)²E[…].
  - The PR64 erratum does **not** change the value (6k)². ERRATUM_CONGRUENCE.md line 40 says no formula in (5.3)/(5.4)/(5.5) changes. It affects the justification of (5.4) only.
  - So what remains open is the *identification* of (15.2) as the Theorem-C lifetime constant, via (10.3) "By Sections 3 and 5, A_r→A_0", which is the reopened A3. The arithmetic is not in question.
- Other surfaces are inconsistent with STATUS in the conservative direction:
  - LANDING_CLAIMS has HOLD_WITH_DOMAIN, with *"persistence interpretation remains conditional on open analytic review #63"*.
  - GRAPH.json has AUTHOR_SIDE_CANDIDATE.
  - STATUS and museum.json show the same object as ACCEPT-scoped without the label.

**D3-F1 (nit, PLAUSIBLE, over-claim)**
- Quote (5841779222): *"comment5841269490 independently verifies C1-C6 and explicitly separated coefficient arithmetic from parent acceptance."*
- The review follows PROOF.md's own route for C1, C2 and C5:
  - C1: the same 8/3;
  - C2: the same s±x split, the same a²−4a+8−8e^{−a/2} identity, and the same E s⁴ = 25/3 and E e^{−s²/2} = √(3/8);
  - C5: the same 2/3 minor and 7/9 determinant.
- It does add separate work in three places: C3 re-checks the parent scalar integral, C4 enumerates 84 multi-indices, and C6 uses a separate Stirling shift by 48 with its own log/exp.
- The adverb "independently" appears only in 5841779222. PROOF_INDEX.md:19, the museum scope_quote and STATUS all say just "verifies the coefficient calculation". All six verdicts reproduce, so this is a wording problem only.

**D3-F2 (nit, PLAUSIBLE, consistency)**
- Quote (5841779222, 01:03:54Z): *"Parent review #63 has now completed at its exact scope. Closing #65 as the coefficient-review work package."*
- Its premise was the #63 closure 5841778773 (01:03:50Z). That was withdrawn by #63 5841830743 (01:11:34Z, *"FAIL-CLOSED RECONCILIATION — reopening #63 narrowly"*), and erratum 5841947606 (Math- PR64) followed.
- 5841779222 was never edited (updated_at equals created_at), and it is the last comment on #65.
- Verifier corrections:
  - The D1 row that records the reopening is STATUS.md line 20, not 15.
  - The STATUS half of the original fix already exists: the D3 limits text, the reading rule at lines 54–56, and PROOF_INDEX:19's *"Interpretation as the parent lifetime law retains the parent's separate dependency and review boundaries."*
  - No downstream surface repeats the stale sentence. The only fix needed is an additive note on #65.

**D3-F4 (nit, CONFIRMED, math)**
- Quote (C4): *"For every multi-index of order at most 6 and every \|y\|≥24, ∏_i\|He_{α_i}(y_i)\|≤76\|y\|^6. The check covered all 84 multi-indices in d≤3; the worst ratio to 76·24^6 is about 0.0135, on the pure sixth derivative."*
- Exact values:
  - He₆(24)/(76·24⁶) = 62050747/4841275392 ≈ 0.012817;
  - the supremum over \|y\| ≥ 24 is 1/76 ≈ 0.0131579, approached only as \|y\| → ∞;
  - the majorant He*₆(24)/(76·24⁶) = 65368517/4841275392 ≈ 0.013502, where He*₆(t) = t⁶+15t⁴+45t²+15.
- He*₆(t)/t⁶ decreases in t, so the majorant with the nonnegative-coefficient step (76(1+y)⁶ − He*_q(1+y) ≥ 0 coefficient-wise for q = 0…6) proves the bound for all \|y\| ≥ 1.
- PROOF.md lines 88–95 contain exactly this argument. The C4 VERIFIED verdict stands; only the label on the number is wrong.
- Evidence: `work4/d3-side24/image_check.py`.

**D3-F6 (nit, CONFIRMED, custody)**
- Quote (STATUS.md:12): *"[Proof/package](https://github.com/d6g8k5htny-coder/Math-/tree/main/coefficients/side24_v1)"*.
- The tree is currently `224848cd` at e329fba1, d6628da0 and 03333792, so today this is a latent risk only.
- The same unpinned pattern affects the D2, D4 and D6 rows (see D4-F7).

**D3-F7 (nit, CONFIRMED, packaging)**
- The block is `git clone …` / `git checkout --detach e329fba1…` / `sha256sum coefficients/side24_v1/*` / `python3 -m unittest test_coefficient.py`.
- Run literally, all three steps after the clone fail:
  - checkout exits 128 with "not a git repository";
  - sha256sum exits 1 with "No such file or directory";
  - unittest exits 1 with `ModuleNotFoundError`.
- `python3 -m unittest coefficients/side24_v1/test_coefficient.py` from the repo root also fails, because the test does `import coefficient as c`.
- Two forms work: running from inside the directory, or `unittest discover -s coefficients/side24_v1`. Both give "Ran 30 tests … OK".
- The prose line ("… in that directory, 30 runs, 0 failures") is accurate.

**D3-F8 (nit, CONFIRMED, packaging)**
- `"statement_heading": "Theorem"`, while PROOF.md (blob `44b66f04`) has no heading or text "Theorem". The bounds are stated under "## Scope and exact parent" (lines 6–30).
- `tools/landing_claims_check.py` lines 24–27 only check that the key is present.
- The problem is systemic. Four other entries have statement_heading values that occur nowhere in their files: rn-count-interface, p15-price-boundary, p15-price-budget-restricted, downstream-hard-gate. The claim is HOLD_WITH_DOMAIN, so this is a broken navigation pointer, not an over-claim.

**D3-F9 (nit, CONFIRMED, consistency)**
- Quote: *"classification": "AUTHOR_SIDE_CANDIDATE" … "notes": "Evaluates #63 Eq15.2; parent unreviewed"*.
- The note was accurate when written (2f4baff9) and at the file's last edit (1a8fdcfc, 00:31:48Z). It went stale after #63 5841570965 (00:36:03Z) accepted §§8–15.
- The classification remains correct as a fail-closed value, because Theorem A (§§2–7) is reopened.

**D3-F10 (nit, CONFIRMED, process)**
- The "same-session" phrase is never defined.
- The facts are consistent only with "a single run by a different provider, not replicated". The comment never says so outright.
- The ambiguity is concrete: Math- `imports/upper2d_stage_e_20260926/REVIEW.md` line 3 uses "same session" to mean the same session *and provider* as the author.

**D3-F11 (nit, CONFIRMED, consistency)**
- Quote: *"\(4\cdot(6\kappa)^2=144\) and \(k^{-2/3}\cdot k^2=k^{4/3}\)"* and *"limiting longitudinal eigenvalues \(\pm6\kappa\)"*.
- κ appears nowhere in the parent or the package.
- Replacing κ with k alone would still leave "4·(6k)² = 144", which is false as written because it equals 144k². The arithmetic the review intends is exact.

#### Downstream surfaces (Q3)

These stay within the arithmetic-only scope:

- STATUS.md:12 and status.json:30, with the gaps in D3-F5 and D3-F6;
- Math- PROOF_INDEX.md:19;
- LANDING_CLAIMS side24-coefficient, HOLD_WITH_DOMAIN (conservative);
- museum.json claims[1], pinned at d6628da0;
- the notebook README.

GRAPH.json:135–143 is stale in the conservative direction (D3-F9). None of these surfaces carries the review's "still parent-open" list in full. Only the generic STATUS disclaimer and the PROOF_INDEX sentence cover it.

#### Later contradicting records (Q6)

- **PR64 congruence erratum.** D_r = diag(r^{−1/2}, I), so det(D_r H D_r) = det H / r → ±6k det A₀. This is consistent with the (6k)² factor and changes no D3 input.
- **main#63 reopening** (5841830743). This makes the #65 reconciliation sentence stale (D3-F2). It does not touch the arithmetic.
- **main#116 comment 5842107966.** This concerns the C103 six-pin normalizer, not D3's arithmetic, so it does not contradict D3. Its structure mirrors parent (5.4), which D3 already lists as parent-open.

#### Not verified

- How the reviewer obtained the parent bytes (40261 B, `9350ad6e…`) before the in-tree import b6282ec (00:21:13Z). The comment says Drive, and Drive was not checked.
- The GitHub edit history of 5841269490 (the placeholder at 23:58:59Z versus the final body at 00:09:32Z).
- The model identity behind run bc-be5dc8d9. The Cursor run page was not opened.
- The reviewer's unpublished shift-48 Stirling code and its 84-index script. Only their stated outputs were compared.
- The DLMF 5.11(ii) remainder rule, which was taken as standard.
- The parent-open interfaces themselves: the contact Jacobian, (5.4)/A3, Kac–Rice, elder identification, and Theorems A and C.
- The 180-digit rotated-frame comparison and the MC results are NON-CERTIFYING.

#### Does STATUS.md row D3 rest on an unsupported Grok verdict?

**No.** Each C1–C6 VERIFIED verdict re-derives exactly against the unchanged e329fba1 bytes, and the published 20-digit endpoints for c₂,₂₄ and c₃,₂₄ are confirmed. The row's accepted scope, which covers coefficient arithmetic only, is supported.

The remaining defects are wording, consistency and packaging:
- the row omits the prescribed PARENT-IMPORTED-OPEN label and the parent-open list (D3-F5);
- "independently" in the reconciliation (D3-F1);
- the stale "#63 completed" sentence (D3-F2);
- the unanswered fresh-review request (D3-F3);
- unpinned links and nits.

None of these makes the ACCEPT wrong on its own scope.

---

### D4 — fixed-remote RN count theorem (main#76 comment 5841783172)

#### Self-ID basis

- Comment 5841783172 is authored by cursor[bot]. It was created 2026-09-26T01:04:29Z, 4 s after the owner's @cursor request 5841782672 (01:04:25Z), and last edited 01:12:36Z.
- **The body has no provider, model, session or independence line.** Its only provenance is the cursor.com footer for run `bc-593a8246-91d3-4d99-90bc-7448dff875dc`.
- The attribution "xAI/Grok" comes only from owner reconciliation 5841861362: *"the fresh xAI/Grok review on this issue returns ACCEPT on all seven …"*.
- The run id appears in no other record. git grep over all refs of main and Math-, a commit-message grep, and PR search all return nothing. The D3 run id is likewise absent from the repos, so the missing self-ID *line* is what sets D4 apart.
- Every verdict-returning Cursor review that night self-identifies. Examples:
  - #63 5841570965: *"Provider Cursor, model Grok 4.7 (grok-4.7-high-fast), session bc-ce4bf0bc…"*;
  - #116 5842107966: *"This note is from Grok 4.7 in this Cursor cloud run"*.
- By concurrent default the attribution is plausible, but it cannot be verified. The review gives zero organizational-independence credit.

#### What it claims

Seven verdicts on PROOF.md at 191ea7d5 (18355 B, sha256 `a332bae9…`), at fixed ρ and fixed η only:

| # | Interface | Verdict |
|---|---|---|
| 1 | Conditional covariance after pins (Jacobian 12 r^{−(d+3)}, not re-inserted) | ACCEPT |
| 2 | Three determinant factors, O(k r⁵) numerator, H = D_r K D_r, det(D_r)² = r | ACCEPT |
| 3 | Original endpoint normalizer, "Z_r/r²=z_0+O(r) with inf z_0>0" | ACCEPT |
| 4 | Contact kernel and O(k r⁴ \|E\|) remainder | ACCEPT |
| 5 | Fixed-separation factorial moments, P ≤ E[T]/m! = O(r^{3m}) | ACCEPT |
| 6 | Selector/height-window mapping | "ACCEPT for the implication in §7 only" |
| 7 | Uncovered complement | "ACCEPT as stated in §7" |

The review also gives a region crosswalk and a "citation correction" for arXiv:2304.07424v3.

#### Independently re-verified (specific numbers)

**Pin Jacobian** (sympy, d = 2…5):
- \|det\| = 12/r^{d+3} exactly.
- The target is v_r = (b − kr³/2, −kr², 0, 12k, 0, …).
- Row-3 coefficients are [12/r³, 6/r², −12/r³, 6/r²].

**Axial row.** Row 3 = f‴ + (r²/40) f⁽⁵⁾ + (r⁴/4480) f⁽⁷⁾. The r²/40 coefficient is confirmed.

**Hermite values** (sympy): α_M = −6k, α_S = +6k, f_xxx = 12k exactly.

**Congruence** (sympy, m = 1, 2, 3):
- With D = diag(√r, I): D K D = H, det(D)² = r, det H = r det K.
- The wrong-way product diag(√r, I) H diag(√r, I) has a zero first row at r = 0. That is the #63 "D_old" defect, and D4 does not use it.

**Contact normalizer (planar d = 2 proxy).**
- Exact contact law: A₀ \| U₀ ~ N(−b, 2).
- z₀ = (6k)² E[Q² 1{Q<0}] = 43/25·(1+erf(3/5)) + 6e^{−9/25}/(5√π) = 3.23097853528700 at k = 1/6, b = 6/5.
- MC (NON-CERTIFYING): Z_r/r² = 2.812, 3.118, 3.206, 3.217 at r = 0.4, 0.2, 0.1, 0.05, so the convergence is O(r).

**Exponent ledger.** Numerator r^{2+3m}, mean r^{3m}, single-witness k r⁵, remainder k r⁴, and m! conversion are all confirmed.

**Package.**
- The 28 shipped tests pass in normal and `-O` modes.
- The `remote_window.py` output equals RESULTS.json byte-for-byte.
- `run_validation.py` detects all 7 mutants in both modes.
- The museum replay command runs 28 tests OK.

#### Load-bearing steps the review did not check (checked here)

| Step | Method | Result |
|---|---|---|
| U_r − U₀ = O(r²) for **every** row (the review checked only row 3) | sympy Taylor series | Row 0 = f + r²/8 f″; row 1 = f′ + r²/24 f‴; row 2 = f″ + r²/24 f⁗; transverse rows c₀ + r²/8 c₂ and c₁ + r²/24 c₃. All O(r²). |
| Hermite remainder (6) | sympy, pinned perturbation (s² − h²)² w(s) | Shift is 2r·w(∓r/2) = O(r M₄). Row 3 of any pinned profile is exactly 12k. |
| Block identity det K_s = α det A − s βᵀ adj(A) β, and bound (9) with singular A and index crossings | sympy identity for m = 1, 2, 3; exact `Fraction` grid through `remote_window.block_bound` | Residual 0. 15947 exact cases (1635 index-changing), 0 violations. |
| Filtered-determinant Lipschitz bound (8) | 20000 float random symmetric pairs (NON-CERTIFYING) | Max lhs/rhs = 1.0 (equality at n = 1). No violation. |
| Distinct-jet nondegeneracy of (U₀, Y_x, A₀, H_x) | Fourier-series covariance on T²_L, L = 24, 4, 3; 13 functionals; 7 frames × 9 points (NON-CERTIFYING) | Minimum eigenvalue positive in every case: 2.2e−6 (L=24, ρ=1), 0.37 (ρ=5.9), 9e−7 (L=4), 3.5e−10 (L=3). Positive but possibly tiny, consistent with "no numerical C or r_*". |

#### Custody

- PROOF.md at 191ea7d5 is 18355 B, sha256 `a332bae9…20cb7`, blob `b383bfcc`.
- The blob is identical at 593adcaa, 760340e9, d6628da0, 58f7936d and Math- main 03333792. The only commit that touches the directory is 191ea7d5, which is an ancestor of Math- main.
- All 7 SOURCE_FILES.json hashes match.
- The museum pin (d6628da0), the LANDING pin (760340e9) and the sources-01 pin (1e1114f5) all carry `b383bfcc`.
- **The reviewed bytes never changed.**

#### Findings (blockers first; none are blockers or majors)

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| D4-F1 | minor (verifier-adjusted from major) | PLAUSIBLE | main#116 comment 5842107966 : 0 | #116 calls PREMIS-Z-LOWER BLOCKED and lists a "missing carrier". C103's normalizer is the d=2, L=24, k=1/6, b=6/5 case of D4's Z_r, which interface 3 ACCEPTs. No record links D4 (10)–(11) to #116, to the #63 A3 packet, or to STATUS D1. | Add a cross-reference: cite D4 (10)–(11) for r ≤ r₁ = min(r_*, z₀/(2C)) plus C102's fixed-δ bound, or say why D4 does not transfer. |
| D4-F2 | minor | CONFIRMED | main#76 comment 5841783172 : 0 ("one citation correction") | The Kac–Rice renumbering (Theorem 6.1 / §7.1) conflicts with four earlier repo records that read the v3 PDF, which place Crofton's formula at Theorem 6.1. The review gives no URL or quotation. | Withdraw the correction or verify it against the v3 PDF with quoted headings. Do not retarget PROOF §8. |
| D4-F3 | minor | CONFIRMED | main#76 comment 5841861362 : 0 | The review has no provider/model/session/independence line. "xAI/Grok" rests only on the owner. #76 required independent-reviewer status for acceptance. | Record it as: Cursor run bc-593a8246, model not self-declared, owner attributes xAI/Grok, zero organizational-independence credit. |
| D4-F4 | minor | PLAUSIBLE | Math- PROOF_INDEX.md : 20; main#76 comment 5841861362 : 0; museum.json : 141 | "ACCEPT on (all) the seven … interfaces" drops interface 6's "for the implication in §7 only" and the explicit non-mapping of historical/legacy selectors. | Word it as "ACCEPT on 1–5; 6 only as the conditional §7 implication (no historical/legacy selector shown to imply (1)); 7 confirms the open complement". |
| D4-F5 | minor | CONFIRMED | main STATUS.md@dbb9dcf6 : 13 (also status.json : 37, PROOF_INDEX.md : 20, Math- README.md : 12) | The limits omit compact marks with k_− > 0, fixed d and L, no numerical C or r_*, no probability lower bound or Poisson law, and no elder selection. | Add those limits. |
| D4-F6 | minor | PLAUSIBLE | Math- claims/LANDING_CLAIMS.json : 162/164; GRAPH.json : 155–164 | These still show HOLD_WITH_DOMAIN, with the review pointer on issue #76 as a whole. GRAPH lacks review_disposition and review_provider fields. The front door says ACCEPT-scoped, and README:20 names LANDING_CLAIMS as the governing surface. | Align the dispositions, pinned to 5841783172 with the D4-F4/F5 qualifiers, or add a note explaining why they stay at HOLD. Do the same for D3. |
| D4-F9 | minor | PLAUSIBLE | main#76 comment 5841783172 : 0 | Interfaces 1–5 restate the proof rather than derive it: coupling (5), remainder (6), the other U_r rows, the L^p/UI steps in (10)–(11), and bound (8). The review was finalized about 8 min after the request and closed 3 min 19 s later. | Mark interfaces 1–4 as accepted at proof-sketch level, or attach derivations. |
| D4-F7 | nit | CONFIRMED | main STATUS.md@dbb9dcf6 : 13; docs/site/status.json : 36 | The proof link uses the moving `Math-/blob/main/…`. | Pin it to 191ea7d5 or d6628da0 (same blob) and state the sha256. Do the same for the D2, D3 and D6 rows. |
| D4-F8 | nit | PLAUSIBLE | Math- frontiers/remote_window_20260924/PROOF.md : 134 | "D_r" is the inverse of the PR64 erratum's D_r, and #63 calls diag(√r, I) "D_old". A reader could wrongly think D4 inherits the §5 defect. | Add one reconciliation line: D2's and D4's D_r = the erratum's D_r⁻¹, so neither note carries the parent §5 line-169 defect. |

##### Finding details

**D4-F1 (minor, PLAUSIBLE, consistency). Verifier-adjusted.** Of three verifier passes, one kept this at major and two lowered it to minor. The adjusted severity is minor, because no ACCEPT is wrong, each conflicting record is defensible on its declared scope, and every conflicting record errs on the conservative side.

- Quote (#116 5842107966): *"**PREMIS-Z-LOWER is BLOCKED.** The typed maximum–saddle six-pin normalizer used by C103 has no proved bound `Z_r ≥ c r^2` on the whole interval `0 < r ≤ 0.025`. … Neither fact controls `r ↓ 0`. The missing carrier is a continuous extension of the pin-adjusted endpoint law … through `r = 0`, with longitudinal limits `-1` and `+1`, a common transverse limit `Q_L` of strictly positive conditional variance, positive mass on the open cone `{Q_L < 0}`, fold-type identification `|Δ_M Δ_S| → Q_L^2` on that cone, and uniform integrability."*
- D4 review, interface 3: *"`Z_r/r^2=z_0+O(r)` with `inf z_0>0`"*.
- The identity of the two normalizers holds. C103 §1 uses f(M) = b, f(S) = b − r³/6, ∇f = 0 at both points, and the typed determinant-pair weight. That is D4's Q_r^W with k = 1/6 (so ±6k = ±1) and b = 6/5.
- Item by item, D4 supplies #116's list for r ≤ r_*:
  - (5): coupling and uniform moments;
  - (6): α → ∓6k;
  - (7)–(9): A_i → A₀ and the filtered-determinant bounds;
  - (10): W_r/r² → w₀ in every L^p, which gives uniform integrability;
  - line 141: full density on the negative cone;
  - (11): the limit itself.
- D4 reaches the same limit by another route. Bounds (8)/(9) replace type-indicator convergence, so every item is either supplied or not needed.
- The planar z₀ = 3.23098 matches C098's "3.230979-class" value.
- Verifier corrections:
  - (a) The records are not "the same lane". D4 is run bc-593a8246 with no model line. #116 is run bc-21b57885 (self-ID quoted above). The #63 A3 packet 5850158114 was posted from the owner account, and its packet self-labels as "xAI / Grok lane". The quote *"D2/D3/D4/D6 scoped ACCEPT unchanged"* is at CLOSED_BF_WTWS.md:53 (commit aa60b06), which is later than the 63ce3e4 packet that 5850158114 cites.
  - (b) The #116 BLOCKED answers an owner prompt limited to *"search the current bound sources"* (5842107528). D4 is outside those sources, so the verdict is right on its own terms. The only inaccurate text is the unscoped sentence "has no proved bound … on the whole interval". That sentence is repeated in PR #121 comment 5842152602.
  - (c) The #63 A3 AMEND is a verdict on the parent's own §5 text, which has a real congruence defect and a different UI route. It is not a verdict on D4.
  - (d) STATUS.md:20 (D1) never names the normalizer. It would stay AMEND in any case, because A1, A2, A4–A7 and the cap implication are open. So there is no STATUS-level contradiction, only implicit tension.
  - (e) Cross-references do exist in the other direction. Two later main-repo notes consume D4's normalizer as a lower bound:
    - `docs/rn_d5_shrinking_witness_20260926.md:79` (commit 00c9d08, 01:27Z): *"For small `r`, `Z_r ≥ c r^2`."*
    - `notes/intermediate_scale_20260926/PROOF.md:46–50` (commit 674f88c, 02:01Z, self-identified in that file as "Grok 4.7", Cursor session bc-235357c8).
    What is missing is a link from D4 to #116, to the A3 packet, or to STATUS D1.
  - (f) The split point is r₁ = min(r_*, z₀/(2C)), not r_*.
  - (g) The review was created 43 min before #116 (01:04:29Z versus 01:47:57Z). "About 35 min" is measured from its last edit.
  - (h) N(−b, 2) is exact only for the planar kernel. The torus corrections are O(e^{−L²/2}).
- Other records with the same gap, which should get the cross-reference:
  - PR #121 5842152602;
  - Math- `reviews/pr124_c1_c2_nonauthor_20260926/REVIEW.md` (C2);
  - the C103 repair-branch source map (PREMISE_Z_LOWER "UNRESOLVED_AT_ZERO");
  - PR #121 5848715982.

**D4-F2 (minor, CONFIRMED, over-claim)**
- Quote: *"The operative Kac–Rice input is Theorem 6.1, Remark 8, and §7.1 of arXiv:2304.07424v3 … In that HTML, Theorem 7.1 is the i.i.d. sum model, and the critical-point discussion is §7.1 rather than §8.1. … A later bibliographic edit can retarget the sentence in §8"*.
- Earlier records that read v3 disagree:
  - Math- `reviews/d1_section9_borel_repair_20260925/REPAIR.md:13–17` (commit 62dce51, 21:51Z): *"Theorem 6.1: Crofton's formula … Theorem 7.1: Expected integral on the level set"*;
  - #63 5841570965 (00:36Z), which read the v3 PDF;
  - remote_window RECONNAISSANCE.md:5 and ANNULUS_RECON.md:5;
  - SUPERSESSION.md (commit 3c82293, 01:35Z, on an unmerged cursor branch): *"The v3 PDF, not the ar5iv HTML, is the numbering source … The ar5iv HTML renumbers these statements"*.
- The review's numbering matches the ar5iv rendering used in an earlier review (main PR112 5839550742). A v4 of the paper (22 Apr 2025) may account for the renumbering.
- PROOF §8 line 215 ("Theorem 7.1 and Section 8.1") is correct on the v3-PDF reading. The retarget the review invites would point the proof at Crofton's formula.
- Impact: PROOF.md is unchanged and no record adopted the correction. The count does not depend on the citation, so the damage is to the review's credibility on source-reading.
- Not verified directly: arxiv.org is blocked by the egress proxy (see "Not verified").

**D4-F3 (minor, CONFIRMED, process)**
- Quote (5841861362): *"the fresh xAI/Grok review on this issue returns ACCEPT on all seven fixed-rho/fixed-eta interfaces"*.
- Nothing in the repos adds a provider name. STATUS:13, PROOF_INDEX:20, museum.json and status.json give none, and the GRAPH D4 node has no review_provider field.
- The trigger 5841782672 did not ask for disclosure. The #76 assignment 5836886717 did say *"Independent analytic reviewer preferred. If no independent reviewer picks this up, an author-side audit may only produce falsifiers/crosswalks, not acceptance."* So the ACCEPT row depends on a nonauthor status that the review never states.
- Circumstantial support: the #67 run bc-dfbce246 was triggered 8 s earlier by a comparable @cursor request, and it self-identifies as *"xAI, Grok 4.7 (grok-4.7-high-fast)"*. Two non-review Cursor deliveries also lack a self-ID: #76 5841862224 and #67 5841899531.

**D4-F4 (minor, PLAUSIBLE, over-claim)**
- Quote (PROOF_INDEX.md:20): *"record ACCEPT on the seven fixed-rho/fixed-eta interfaces"*. Reconciliation: *"returns ACCEPT on all seven … selector/window mapping, and the explicit uncovered complement"*.
- The review's own wording: *"**Selector and height-window mapping — ACCEPT for the implication in §7 only.** … No historical RN witness predicate, inner-wedge cell, whitened-jet cell, or all-cell partition is shown to imply (1). Those objects stay outside this acceptance."*
- The original target 5836886717 item 6 was *"exact selector/height-window correspondence to the historical RN witness cells where valid"*, and that was not established.
- Verifier corrections:
  - The trigger 5841782672 had already narrowed item 6 to "selector/height-window mapping".
  - The interface-7 part is weak, because both surfaces state that the regions stay open.
  - STATUS.md:13, the museum status_quote and class, and GRAPH are unaffected.

**D4-F5 (minor, CONFIRMED, over-claim)**
- Quote (STATUS.md:13): *"Does not cover shrinking `rho` or `eta`, pin neighborhoods, intermediate scales, all remote heights, or global RN/24-jet closure"*.
- The source relies on restrictions that the row leaves out:
  - PROOF line 17: compact B = [b₋, b₊] and K = [k₋, k₊] with k₋ > 0, and fixed d and L;
  - line 141: *"Compactness, k_->0, and the endpoint-only version of (10) give"* inf z₀ > 0;
  - line 53: no constant is uniform as marks become unbounded or d, L vary, and there is no numerical C or r_*;
  - lines 51 and 197: no probability lower bound and no Poisson law;
  - the review's own crosswalk row: *"Probability lower bound, Poisson law, numerical `C` or `r_*`, elder selection | Not conclusions of this argument"*.
- z₀ ∝ k² → 0 as k → 0, and the downstream lifetime integral runs over every k > 0.
- museum.json embeds the full §1 text, so it does carry these restrictions. No record on file actually transfers D4 outside its mark range.

**D4-F6 (minor, PLAUSIBLE, consistency)**
- The front door says ACCEPT-scoped: STATUS:13 and museum.json claims[2].
- Math- main still says HOLD:
  - LANDING_CLAIMS.json:164 has `"disposition": "HOLD_WITH_DOMAIN"`, and line 162 points at issue #76 as a whole;
  - the GRAPH node `math.rn-fixed-remote-window` has no review_disposition or review_provider.
- README.md:20 says each row *"is governed by its … review object in claims/LANDING_CLAIMS.json"*, and PROOF_INDEX.md:51 requires a source-bound reconciliation for any change. So the governing surface still says HOLD.
- Verifier corrections:
  - D6's explicit update landed in LANDING_CLAIMS only. The GRAPH node carrying "Source-bound xAI/Grok technical ACCEPT" is the D5 fixed-annulus node.
  - D3 has the same drift.
  - The GRAPH convention keeps classification AUTHOR_SIDE_CANDIDATE even for reviewed nodes.
- The error runs in the conservative direction.

**D4-F9 (minor, PLAUSIBLE, process)**
- Quote: *"I recomputed the pin Jacobian, the target `v_r`, the axial coefficient `r^2/40` on `s^5`, and the Hermite values … The local exact suite, 28 tests, passed under Python 3. Those tests check finite algebra. They do not evaluate `Λ_j`."*
- Timeline: request at 01:04:25Z, placeholder 4 s later, final edit at 01:12:36Z, reconciliation at 01:15:52Z, #76 closed at 01:15:55Z.
- Verifier corrections:
  - The Hermite values α_M and α_S are not covered by any shipped test, and the review also attempted a (disputed, D4-F2) citation check. So "only finite algebra already covered by tests" is too strong.
  - (9) *is* exercised by test_index_crossing_bound, test_singular_index_bound and test_matrix_grid (108 exact checks). Bound (8) has no test.
- My own checks found every unshown step correct, so no verdict changes. The issue is the depth of the review.

**D4-F7 (nit, CONFIRMED, custody)**
- Quote: *"[Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/remote_window_20260924/PROOF.md)"*.
- The file has a single commit in its history, so nothing has drifted yet.
- STATUS.md@dbb9dcf6 has 6 `blob/main` or `tree/main` Math- links and only one pinned Math- link (58f7936d).

**D4-F8 (nit, PLAUSIBLE, consistency)**
- PROOF.md:134: *"With D_r=diag(sqrt(r),I), H_i=D_r K_i D_r. Congruence preserves inertia and det(D_r)^2=r."*
- The review writes `H=D_r K D_r` and `det(D_r)^2=r` without defining D_r. However, det(D_r)² = r is consistent only with diag(√r, I), so the review cannot be read as the defective D_old H D_old.
- D2's LIFETIME_REMAINDER.md lines 106–108 use the same correct convention.

#### Downstream surfaces (Q3)

- STATUS.md:13 lists only interfaces 1–5. Its limits match the review's crosswalk for ρ, η, pins, intermediate scales, heights and global RN, but omit the items in D4-F5.
- PROOF_INDEX.md:20 and museum.json:141 over-summarize interfaces 6 and 7 (D4-F4).
- LANDING_CLAIMS and GRAPH lag in the conservative direction (D4-F6).
- The later PR #125 (D5 shrinking witness) uses the fixed-η result only as its outer boundary, which is consistent with (15)/(16).

#### Later contradicting records (Q6)

- **PR64 erratum.** It is consistent with D4 (D4-F8). D4 uses the correct congruence direction and does not carry the parent §5 displayed-factor defect.
- **main#116 5842107966, PR #121 5842152602, and the #63 A3 packet 5850158114.** These treat the same normalizer as BLOCKED or AMEND without citing D4. This is unreconciled and conservative (D4-F1). It does not contradict D4's mathematics.
- **main#63 reopening.** It concerns parent §§2–7 and does not reach D4's own derivation.

#### Not verified

- The theorem and section numbering of arXiv:2304.07424v3. arxiv.org, ar5iv and mirrors were blocked by the egress proxy, so D4-F2 rests on repo records.
- A full-rigor, line-by-line proof of coupling (5) and of the Kac–Rice hypotheses for the height-disintegrated count, with explicit constants. These were checked at proof-sketch level.
- The model behind run bc-593a8246.
- Z_r/r² at L = 24 for the periodized field. Only planar BF was simulated. All MC and float results are NON-CERTIFYING.
- Whether C102 plus D4 (11) closes PREMIS-Z-LOWER on all of (0, 0.025]. C102 was not read, and D4 gives no numerical r_*.
- Strict positivity of Λ_j for every index j, beyond the full-support argument.
- Owner actions after dbb9dcf6 beyond those cited (for example #63 rounds 5–9 and PR #163).

#### Does STATUS.md row D4 rest on an unsupported Grok verdict?

**No, on the row's stated scope.** The five interfaces the row lists (covariance, three determinant factors, endpoint normalizer, contact kernel, fixed-separation factorial moments) are supported by the unchanged 191ea7d5 bytes at proof-sketch rigor. Every named identity re-derives exactly, and several steps the review did not check also hold. So no ACCEPT is wrong.

The row does have three weaknesses:
- It rests on a review with **no self-identification**. The provider label is owner-supplied (D4-F3).
- Its limits omit k₋ > 0, fixed d and L, and the absence of numerical constants (D4-F5).
- The same endpoint normalizer is shown as BLOCKED or AMEND in #116 and the #63 A3 packet with no cross-reference (D4-F1, minor, PLAUSIBLE).
