<!-- PART 1 sections: Grok uploads on main (PR #163 replay packet, PR #159, PR #161, cycle-2 branch, issue #165) and Grok Math- drafts (PRs #80, #81, #82), plus the Files 1-6 public-mirror check and the Math- PR #87 reading maps and 5ed3b45 default files. -->

# PART 1 — Grok uploads on main (PRs #163, #159, #161, cycle-2 branch) and Grok Math- drafts (PRs #80, #81, #82)

**Conventions for Part 1.**
- **Status.** CONFIRMED means the finding was reproduced from the record or by exact computation, and at least one independent verifier pass agreed. PLAUSIBLE means the core is supported, but a verifier pass disagreed in part or part of the evidence is inference; PLAUSIBLE findings are labelled. Only CONFIRMED and PLAUSIBLE findings are listed. DROPPED and UNVERIFIED findings are not results (§11 lists UNVERIFIED leads separately).
- **Severity** is the verifier-adjusted value. "adj." marks a change from the first reviewer's rating, and a verifier split is shown as "panel 2 major : 1 minor". Blocker: must be fixed before the surface can land, or needs an erratum if it already landed. Major: a stated result or status is wrong in a way a reader would rely on. Minor: a local error, a stale pointer, or an over-claim that changes no status. Nit: wording.
- **Arithmetic.** Exact arithmetic (sympy, Fraction) certifies the identity it checks. Floating point (float64, or mpmath without interval enclosure), quadrature and Monte Carlo (MC) are NON-CERTIFYING and are labelled.
- **No credit.** Verdicts are technical only. They carry zero organizational-independence credit, because all lanes share one GitHub account and several of the reviews involved are same-provider. They change nothing in STATUS.md, PROOF_INDEX.md or LANDING_CLAIMS.
- **Times** are UTC. Commit dates are set by the client; "server-side" marks GitHub event times.
- **"Q0 n"** is line n of the public OCR mirror blob `ed0f51f1` (Q0_MASTER.md), described in §8.
- **Pins.** Findings are assessed at the pinned heads in §0.1. What changed after the pins is recorded in §0.2 and in each section. It was not re-reviewed in depth and does not change the counts.
- **Deduplication.** A finding that duplicates a Part 2 or Part 3 finding appears under "Covered elsewhere (not re-counted)" with the covering ID. A finding reported by two Part 1 units is counted once: in the unit whose primary scope it is, or otherwise under the ID with more verifier passes (at the lower severity on a tie). The other ID is listed as merged. IDs keep the original unit numbering, so a gap in a sequence is a merged, covered or dropped finding.
- **ID prefixes:** DC, BC, BT, PK (§1, the four PR #163 sub-areas); HY (§2); LB (§3); CN, CS (§4); CK (§5); MD (§6); XC (§7); FM (§8); IS (§9); RM (§10).
- **Paths.** In §1 a bare file name is under `incoming/grok-session-20260926-replay/` on main. In §4 and §9 it is under `incoming/grok-cycle2-20260926/` on main. Math- paths carry the prefix "Math-:".

## 0. Surfaces, pins and post-pin state

### 0.1 Pinned heads

| Surface | Ref | Pinned object | Notes |
|---|---|---|---|
| main PR #163 | `incoming/grok-session-20260926-replay` | `cb774284b3f53a7539fe556cbb18a175542ea9d5` | Draft at pin; 34 added files; intake packet. |
| main PR #159 | `grok/drive-lane-map-20260926` | `39161a6d2a0c6a073055236d314066e4ac22eec7` | Draft at pin; README.md, GAP_LEDGER.md, SOURCE.json. |
| main PR #161 | `review/lb-rate-thm023-hold-20260926` | `43b3ced366fa30a4169f454f7a48802367b45758` | `reviews/lb_rate_thm023_landing_20260926/REVIEW.md`. |
| main cycle-2 branch | `incoming/grok-cycle2-nd-d5-sard-20260926` | `8dfde10b6f206d4dd9c46669e54b1940fcf662f4` | No PR. RESULT.md (2,941 B, sha256 a93bbfc1…) and SESSION_LEDGER.md (1,263 B). |
| main issue #165 | — | body created 2026-09-26T23:31:05Z | Never edited, no comments, no linked PR. |
| main default | `main` | `dbb9dcf64d23ecb45267ce2779f69f17b2fa6265` | Reference tree for the intake gate. |
| Math- default | `main` | `03333792b0fde32ac6b43649ba502d2fe638b5a6` (earlier `55a3cedde916e454d410bad5c4f62c6f8b882c22`) | |
| Math- PR #80 | `grok/contact-kernel-substitute-20260926` | `a79007eba6795994ab02a55e2a03f6b53b2b2e67` | SUBSTITUTE.md (159 lines, 7,531 B, sha256 75e1181a…); base 66e39d1. |
| Math- PR #81 | `grok/drive-hole-ledger-20260926` | `b28cf2526532ad318f082955fb419f09dbb5c61e` | LEDGER.md (45 lines). |
| Math- PR #82 | `grok/d5-microdisk-20260926` | `b190a4de0a0f0c5b500d84039e924fc9c5344431` | NOTE.md (4,844 B, sha256 56451eee…), QUARTIC.md (2,031 B, sha256 54547b2d…). |
| Math- PR #87 | `incoming/harper-cycle4-d5-sard-20260926` | `0fab9330c017e78eb9ef6d928e04aba9f376a470` | §10 only: SARD-G half, reading map. |
| Math- PR #88 | — | `2cdf62fdc98f150a0e351da7d746901d87321e5b` | §10 only: reading map. |
| Public mirror | main history (`80e7d0d`, ancestor of default); ~90 branch tips, e.g. `chatgpt/drive-github-hardening-20260919` @ `38a3e070`; also `7caac254` | blob `ed0f51f1e838a4119b4bb7e77c13c416318529f4` | Q0_MASTER.md, 1,831,987 B, sha256 3112fb61…22cc. Removed from the default tip by `f35eef1` (2026-09-23). |

### 0.2 Post-pin state (read on 2026-09-27; not re-reviewed in depth)
- **main PR #163 merged** as `46c0f69` (17:50:09Z). Just before that, commit `857904a` (17:41:09Z, git author "Cursor Agent") added RESULT.md and IDENTITY.json and a SESSION_LEDGER supersession note. The note says the files predate 21:46:22Z, INVENTORY.md:46 is stale, and A3_UI_MAJORANT is conditional on A4. The same commit fixed the garbled SESSION_LEDGER `3e-2` line. Public intake then passed.
  - The packet diff from `cb774284` to the merged head touches only those three files. So PK1 is resolved, and Part 2 X4 is fixed on SESSION_LEDGER.md only.
  - Every other §1 finding persists in the landed intake packet on main default. Intake is add-only and immutable, so fixes now need a successor packet or an erratum.
- **main PR #159 merged** as `7cf3bdb` (17:52:05Z), after `f21e222` (17:43:08Z).
  - `f21e222` added RESULT.md and IDENTITY.json. It corrected the erratum row, the "Owner pre-approved" and "independently confirmed" wording, removed the google-drive replica pointer and dropped step 3. So HY1 and HY10 are resolved.
  - README.md:42 ("Recovery still OPEN"), :50 (step 1), :51 (the recipe) and the GAP_LEDGER draft-#59, CL_ANTHROPIC, JETMOD and "Doc" rows persist (HY2, HY3, HY7, HY4, HY5, HY11).
- **main PR #161 merged** as `dc58979` (17:51:20Z), after `1c557c5` (17:45:42Z, "apply AMEND items 1-4").
  - `1c557c5` fixes LB1, LB2, LB4, LB5, LB6 and LB15.
  - **LB3 was realized:** issue #116 was closed at 17:51:22Z by cursor[bot], with state_reason `completed` and closed_by_pull_requests [#161].
  - LB9, LB12 and LB17 persist; after the edit they sit at REVIEW.md:30, :76 and :92.
- **Math- PR #82 merged** at 2026-09-27T17:35:56Z (by cursor[bot]). Its head `b1b8a76` is a merge of Math- main (git author "Cursor Agent") on top of `b190a4d`. So the NOTE.md and QUARTIC.md defects (MD1–MD4, MD7–MD9) are now on Math- default. This is inferred from the commit list; the landed bytes were not re-hashed. Math- default is now `3b2ac59`.
- **Unchanged at the pinned heads:** Math- PR #80 and #81 (open drafts), Math- PR #87 (`0fab933`), PR #88 (`2cdf62f`), the cycle-2 branch (`8dfde10`, still no PR) and the empty branches in §2. main default is now `4cc4587` (18:07:56Z).
- **Unknown operator.** The shared account does not show who operated the post-pin "Cursor Agent" commits.

---

## 1. main PR #163 — replay packet (`incoming/grok-session-20260926-replay` @ `cb774284`)

**What the packet claims.**
- A 34-file intake packet with "Scientific effect: NONE". It replays and navigates the D1 lifetime chain, the Bargmann–Fock (BF) cubic and Condition (ND) analysis, the BF transverse/D5 material, and the custody of the July-5 Files 1–6 stack.
- It proposes no STATUS change.
- Its custody files state that the Files 1–6 / O1 / GCJA / C006 / Gate Framework stack is "ABSENT from public GitHub".

**Verdict:** AMEND at the pin. The intake blocker was resolved post-pin and the packet landed; the other findings are still unaddressed in the landed packet.

### 1.1 D1 chain (DC)

**What it claims.**
- D1_INTERFACE_TABLE maps the lifetime parent's A1–A7 interfaces to the #63 dispositions.
- A3_TYPE_CONVERGENCE presents the D_old congruence as a realized type-convergence counterexample.
- A3_UI_MAJORANT "ACCEPT[s] the structure".
- A2_RESIDUAL_GAP (the only file new since the prior review) names a "single missing estimate", λ_min(Cov_Q(U_r ⊕ vec A_r)) ≥ c_*.
- SECOND_PASS, A_M_TO_A0, EMBEDDED_CHART_AND_MORSE and STATUS_PIN_NOTE complete the chain.

**Recomputed and CONFIRMED (exact):**
- The congruence and determinant identities for m = 1, 2, 3: the (5.3) block-determinant identity, D_r H D_r with det(H)/r, and D_old H D_old with r det H.
- det T_ax = 12/r^4, det T_tr = 1/r, and |det T_r| = 12 r^-(d+3), which matches the Math- parent at line 90.
- The target v_r, the (7.3) integral, the (7.4) power count and the far-branch split.
- The canceled-pivot bound (6.1), on 600 exact cases for m = 2, 3.
- The cap rationals, the axial cubic and the embedding radius.

**Prior review follow-through** (review 5850519011 at c8405a76). Ten of the eleven D1 files are byte-identical to what that review saw. At the pin none of these was addressed:
- item 1 (intake);
- item 2 (erratum custody);
- item 3 (A3_UI conditional);
- INVENTORY:46;
- the SESSION_LEDGER:74 garble.

After the pin, `857904a` resolved item 1 and added ledger notes for A3_UI_MAJORANT and INVENTORY:46. The file lines themselves are unchanged.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| DC3 | minor (panel 1 major : 2 minor) | CONFIRMED | A3_TYPE_CONVERGENCE.md:39 (also :56, :60) | "Counterexample" / "Realized" is false. The type indicator is invariant under congruence; only one proof route of type convergence fails. Parent line 169 is consistent with H = D_old K D_old, which is the SECOND_PASS reading. Same root cause as Part 2 G1; this is the file surface. | "The D_old display breaks one proof route; the type indicator is congruence-invariant." |
| DC5 | minor | CONFIRMED | A2_RESIDUAL_GAP.md:41 | λ_min(Cov_Q(U_r ⊕ vec A_r)) ≥ c_* is identically false: U_r is pinned under Q, so the covariance is singular. | State the floor for the unconditional covariance of (U_r, vech A_r) (m(m+1)/2 entries), or for Cov(A_r \| U_r). |
| DC6 | minor | CONFIRMED | STATUS_PIN_NOTE.md:9 (also :1, :5) | 58f7936d is the STATUS reading checkout, not the 2,138-object inventory pin. The inventory pins are Math- 1e1114f5 (×54) and main 7caac254 (×2084). | Name the inventory pins. |
| DC7 | minor | CONFIRMED | SECOND_PASS.md:24 | Reads the parent sentence as K = D^-1 H D^-1, which is consistent. D1_INTERFACE_TABLE:19/:27, A3_TYPE_CONVERGENCE:39/:56/:60, A3_UI_MAJORANT:54 and the erratum copy (line 21) read the same sentence as an error. The two readings are never reconciled (SECOND_PASS b785174 at 16:37 predates 8770393). | Add one reconciliation line and adopt the consistent reading. |
| DC8 | minor | CONFIRMED | A2_RESIDUAL_GAP.md:43 (also :1, :36, :49) | "Single missing estimate" is overstated. Theorem A is existential, and parent lines 103–109 get existence by compactness. The label should be packet A2, not "#63-A1". | "One route to a quantitative A2 needs …". |
| DC10 | minor | CONFIRMED | D1_INTERFACE_TABLE.md:18 | Attributes the 12 in \|det T_r\| to f_xxx = 12k. The 12 is structural and independent of k. T_r is written out in parent (3.1), lines 76–86; A2_RESIDUAL_GAP:19–27 rebuilds it; line 35 is stale. | Cite parent (3.1). |
| DC11 | minor | CONFIRMED | EMBEDDED_CHART_AND_MORSE.md:56 | ACCEPTs the cap Euclidean theorem without re-deriving §§2–5. There is no per-file independence line; the blanket statement at SESSION_LEDGER:88 is weaker. | Scope the ACCEPT to what was re-derived, and add a zero-credit line. |
| DC13 | minor | CONFIRMED | SECOND_PASS.md:50 | Lists "global elder convention" as a D1 blocker, against STATUS:20, #63 comment 5841830743 and EMBEDDED_CHART:57. Repeated at REPLAY.json:45, SECTION7_MGE2:49 and SECTION7_FAR_BRANCH:35. Related: Part 2 G3. | Drop it. |
| DC2 | minor | PLAUSIBLE | D1_INTERFACE_TABLE.md:13 | The A-numbering differs from #63 comment 5841766952 in 6 of 7 rows. SECTION7_MGE2:1 follows the table's own scheme. No false ACCEPT follows from it. | Add a crosswalk column. |
| DC4 | minor | PLAUSIBLE | A3_UI_MAJORANT.md:70 | "ACCEPT the structure … complete" drops the conditions at line 58 ("Granted A2 moments and A_i → A0"). It is contradicted later by A3_BF_D2_CONTACT:36–38 (BT8) and BF_TRANSVERSE_EXACT:26. It is scoped to the structure, so it is not literally unconditional. It also misuses "Z_r/r^2 → … in L^1". The D2 bound (R10) \|Z_r/r^2 − z_0\| ≤ C r(k+r)P^N (LIFETIME_REMAINDER:133–134), which would supply the normalizer part, is not cited. Prior item 3; post-pin only a ledger note. | "Structure accepted conditional on A2 moments and A_i → A0 (A4)"; cite D2 (R10). |
| DC9 | minor | PLAUSIBLE | A_M_TO_A0.md:31 (also :56) | "C^2 continuity of the kernel" is the wrong hypothesis. The step needs C^4 (mean-square continuity of D^2 f), cross-covariances of order 5, and order 6 for U0*; C^2 stationarity gives only mean-square C^1. The conclusion holds because K_L is real-analytic. | State the needed order. |
| DC12 | nit | CONFIRMED | EMBEDDED_CHART_AND_MORSE.md:14 | The "Verbatim" quote carries unmarked bold. "Morse/distinct" appears at cap lines 28, 161 and 181. Grok's point, that Morse is not a hypothesis of the deterministic theorem, holds. | Mark the emphasis as added. |
| DC14 | nit | CONFIRMED | A_M_TO_A0.md:35 | "Lattice anisotropy ruled out" is wrong. Jet rank rules out degeneracy, but the torus is anisotropic (parent line 107; the file's own line 57). | "Degeneracy ruled out by jet rank." |
| DC15 | nit | CONFIRMED | A3_UI_MAJORANT.md:52 | The range 0 < r ≤ 1 should be 0 < r ≤ min(r0, 1): for L ≤ 1, r = L makes M and S collide. "Z_r/r^2 in L^1" misuses the notation, since Z_r/r^2 is a deterministic number. | Correct the range and wording. |
| DC16 | nit | CONFIRMED | D1_INTERFACE_TABLE.md:22 | The A6 cell says it "mentions wrong D", but §6 (parent lines 184–211) never mentions the congruence; only §5 does, at line 169. | Point to §5 line 169. |

**Covered elsewhere (not re-counted):** DC1 → Part 2 X3, the stale "unmerged/404" erratum lines. The unit adds that the packet copy lost two trailing spaces and that the branch was the real PR64 head.

**Not verified:**
- The PR64→d8f5505 and PR85→5ed3b455 associations, which rest on Math- PROOF_INDEX:32 and ERRATUM_POINTER (provenance is Part 2 §1).
- Math- PR64's draft state when the table was written.
- Whether `chatgpt/lifetime-parent-congruence-erratum-20260926` ever existed.
- The File-4 convention behind A2_RESIDUAL_GAP:34 (§8 FM18 covers it).

### 1.2 BF cubic / Condition (ND) (BC)

**What it claims.** FILE3_CUBIC_LEFTOVER, ND_14FRAME_AND_GCJA_C2, ND_CUBIC_SLAVING, CONDITION_ND_EXTRACT, FINITE_R_WTWS_SCHUR, BF_WTWS_SCHUR, CLOSED_BF_WTWS and CUBIC_B4_AND_C5_DIAGNOSTIC analyse File 3's Condition (ND) at cubic order for planar BF:
- The Euler relation and W_t = (z_s^2/2)H^-_ss make the literal 15-frame singular.
- A reduced or 14-frame is proposed.
- A "finite-r closure" is computed.
- (ND) is "still OPEN".

**Recomputed and CONFIRMED** (exact sympy, C(u) = exp(−|u|^2/2), unless marked):
- Cov(f_tss, f_sss) = diag(3, 15) unconditionally.
- Var R = z_s^4(5z_s^2 + 9z_t^2)/12.
- The Euler relation 3W_0 = z_t W_t + z_s W_s.
- Unconditional det Gram(W_t, W_s) = 45 z_s^8/16.
- Cov(f_tss, f_sss | G+V) = diag(2, 6), unchanged when f_tts or f_ss is added.
- The Gram given G+V has det (3/4) z_s^8.
- det G+V = 12, with λ_min = 8 − √58 = 0.38423.
- 105, 15, and the 5×5 det 7962624.
- The axis form diag(2/3, 1/6) z_t^6 with det z_t^12/9 (one-point).
- 15-jet: rank 15, det 47775744, λ_min 0.20234 (NON-CERTIFYING).
- B1–B5 and the witness (u = 2, v = 1, k = 1, θ = 1/2), which gives q = −30, A = −3, c = 75, d = −363/2 and dets 18, −18, −18.
- C5: mean (−b, 0, 0, 0), covariance diag(2, 2, 2, 6).
- On V5: Var(f_sss | V5) = 6 and Var(f_ttss | V5) = 4.
- The FINITE_R table 0.87084, 0.75847, 0.75053, 0.750128, 0.750033, 0.7500043, and 0.0029297 at z = (1, 0.5) (mpmath, NON-CERTIFYING).
- The GCJA quotes match Q0 29987 and 29980–81.
- CONDITION_ND_EXTRACT is a faithful paraphrase of Q0 31195–31300.

**Prior review follow-through:** item 1 (intake) was not addressed at the pin; it was resolved after the pin.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| BC5 | **major** (panel 2 major : 1 minor) | CONFIRMED | CONDITION_ND_EXTRACT.md:1 (also CLAUDE_FILES_1_6_MAP.md:57, O1_PART_B_AND_GATES.md:51, D5_REPLAY_AND_TYPE.md:24, D5_MISSING_INEQUALITY.md:38, MUSEUM_TRIAGE.md:59) | "(still OPEN)" for Condition (ND) as written.<br>- det Σ2(0) = 0 identically, by the Euler relation plus W_t = (z_s^2/2)H^-_ss (FILE3_CUBIC_LEFTOVER:20–21).<br>- The stack's own kill log records this as K4. MAP:48 mislabels K4 as an "instrument registry" (PK9).<br>- Grok's cycle-2 RESULT.md:35 says "remains refuted".<br>- So "Theorem B PROVEN-MODULO (ND)" is modulo a condition that is false as written.<br>Scope: the column "Status in File 3" accurately transcribes Q0 31211 "OPEN"; the defect is the present tense. | "(ND) as written: OPEN at the File 3 date; refuted in the same stack (K4; S3 §5; GCJA C2) and replaced by (ND′) (GCJA C4)." Merged: PK8, FM2 (status part). |
| BC2 | minor | CONFIRMED | ND_14FRAME_AND_GCJA_C2.md:24 (also :21–22, :32; FILE3_CUBIC_LEFTOVER.md:36–39) | The leftover variances "after H^-" are given as the unconditional 15/4 z_s^4 and 5/12 z_s^6.<br>- The Gaussian residuals are 3 z_s^4 and z_s^6/3 given H^- alone.<br>- They are (3/2) z_s^4 and z_s^6/6 given the pair frame, since Var(f_sss \| pair) = 6 (GCJA C1).<br>- Line 32 sets the 6 aside.<br>FILE3:33–39 and cycle-2 RESULT:35 are literally correct given f_tss alone, because Cov(f_sss, f_tss) = 0. SESSION_LEDGER:10 is unqualified. Positivity is unaffected. Related: Part 2 X2 (cycle-2 surface). | Quote the pair-conditional values. |
| BC3 | minor | CONFIRMED | FINITE_R_WTWS_SCHUR.md:33 | The "finite-r closure" is a one-point 8-jet at the saddle; pair separation is absent.<br>- Under the full pair frame, det Gram(W_t, W_s \| pair) ~ (3/8) z_s^8 r^2 → 0.<br>- The nondegenerate object is (r^-1 W_t, W_s), with limit diag(z_s^4/4, 3z_s^4/2) and det (3/8) z_s^8.<br>(3/4) z_s^8 is the limit of a genuine two-point object in which only G+V is conditioned and the Hessians are left outside, so "closure" is overstated rather than wrong. | Scope it as one-point, and give the renormalized pair statement. |
| BC6 | minor | CONFIRMED | CLOSED_BF_WTWS.md:41 | The axis frame "candidate stratified frame" is exact only as a one-point G+V object with f_tts pinned. Under pair data W_t^ax → 0 and Var W_s^ax → z_t^4/24 (the surviving f_tttts), so det → 0. No two-point pair σ-algebra reproduces it. | Label it one-point. |
| BC7 | minor | CONFIRMED | CONDITION_ND_EXTRACT.md:5 (also FILE3_CUBIC_LEFTOVER.md:5, ND_CUBIC_SLAVING.md:55) | Cites only the "attached File 3", although the public mirror carries it (Q0 31195–31262). Commit 3414a15 calls the extract "verbatim", but it is a paraphrase. | Cite the mirror lines and say "paraphrase". |
| BC8 | minor | CONFIRMED | ND_CUBIC_SLAVING.md:49 (also :51) | "Overcounts by 1", and a 14-frame (W_t, W_s) with H^- in the frame, where W_t = (z_s^2/2)H^-_ss is singular. This contradicts FILE3_CUBIC_LEFTOVER:23 ("rank drop 2") and ND_14FRAME:38, and is never amended.<br>The pair block has 12 rows, not 14; File 3 itself says 14 (Q0 30919, 30949). File 3's Lemma 4.1 list (Q0 30936–30938) gives only 9 distinct functionals, so the literal 15-frame has rank 10 (FM4). | Mark lines 49–51 superseded. |
| BC9 | minor | CONFIRMED | ND_14FRAME_AND_GCJA_C2.md:26 (also FILE3_CUBIC_LEFTOVER.md:51, ND_CUBIC_SLAVING.md:45) | The remainder "O(r^2)" is wrong off-axis under the pair-conditioned Schur complement. It is O(r), through the unpinned T = (r/2) f_ttss: W_t ≈ r(z_s^2/4)T, W_s carries r(z_t z_s/2)T, and W_0 carries r(z_t z_s^2/4)T. O(r^2) holds on the axis, and CLOSED_BF_WTWS:29 is under an axis heading, so it is correct. FM9 has an O(r^2) statement for quartic content under File 3's Taylor normalization; that is a different object. | "O(r) off-axis, O(r^2) on the axis." |
| BC11 | minor | CONFIRMED | BF_WTWS_SCHUR.md:22 | The eigenvalues are not z_s^2 times a quadratic: λ− ≈ (3/8) z_s^6/z_t^2 and λ+ ≈ 2 z_s^2 z_t^2. (3/4) z_s^8 needs f_ss and f_tts pinned with f_tss free, not "Hessians outside". The W_0 clause is conditional. | Correct the eigenvalue statement and the conditioning. |
| BC4 | minor | PLAUSIBLE | ND_14FRAME_AND_GCJA_C2.md:38 (also :40; FILE3_CUBIC_LEFTOVER.md:43–49; ND_CUBIC_SLAVING.md:45, :51) | The "drop W_t" repair loses information.<br>- At (6.1) normalization W_t/r^4 → 0, but its remainder is O(r), not O(r^2): sd (z_s^2/2) r^5, through T = resid(f_ttss \| V5) with Var 4.<br>- Renormalized by r^5, W_t is an independent nondegenerate coordinate; this is (ND′): diag(z_s^4/4, 3z_s^4/2).<br>- Line 40 attributes the off-axis σ = (2,2) stratum to the axis. On the axis σ = (6,0) and Var(f_t \| pair) ~ z_t^4 r^14/80.<br>"Not in public vault" is literally true (vault = Math- per CONTRIBUTING:41). The defect is the missing cross-reference to (ND′), which O1_PART_B_AND_GATES:35 names. | Cross-reference (ND′) and GCJA C4. |
| BC12 | minor | PLAUSIBLE | FINITE_R_WTWS_SCHUR.md:29 | λ_min ≈ 1e-14 is the float64 floor. The true value is 2.4752e-21 at r = 0.1 and scales like r^14 (mpmath, NON-CERTIFYING). The raw det ≈ 1e-25 at line 27 is real (3.2285e-25). | Report det_W = 0.7500043 and drop the λ_min figure. |
| BC14 | minor | PLAUSIBLE | INVENTORY.md:0 (also SESSION_LEDGER.md, REPLAY.json, PR body) | None of the 31 content files added after 16:50 is indexed in INVENTORY, SESSION_LEDGER or REPLAY.json.<br>- Near-duplicates are unmarked: overcount 1 vs 2; 14- vs 13-frame; BF_WTWS_SCHUR and CLOSED_BF_WTWS 30 s apart; BF_D2_CONTACT_A0 and A3_BF_D2_CONTACT 6 s apart.<br>- 3015c3b dropped an object id.<br>- The PR body "New this round: C006 …" is stale.<br>The "15 vs 6" pair is not a contradiction; each value is right for its conditioning. | Index the files and mark supersessions. Merged: BT17. |
| BC15 | nit (FM8 verifier: minor; tie → nit) | CONFIRMED | FILE3_CUBIC_LEFTOVER.md:49 | "14-frame already contemplated by File 3 §6.5": File 3 has no §6.5. The 14-frame is in the Lemma 6.5 proof in §6.4 and in §7.3, and it drops only W_0, on the Appendix-B curves. Dropping the whole W-block leaves a 12-frame that File 3 never contemplates. Separately, CUBIC_B4_AND_C5_DIAGNOSTIC.md:7 cites NOTE.md without a SHA and swaps (s,t) without flagging it (blob 3ee3082 is stable). | Cite "§6.4 (Lemma 6.5 proof) / §7.3" and say the axis needs a 12-frame or a renormalized stratum. Merged: FM8. |

**Covered elsewhere (not re-counted):**
- BC1 → PK1 (the same intake rejection).
- BC10 → IS2 (Remark 7.2 integrability).
- BC13 (cycle-2 labels) → CS5, Part 2 X2 and CS1.

**Not verified:**
- Whether the OCR mirror equals the PDFs Grok says were attached. Page references such as "p.18" cannot be checked.
- CLOSED_BF_WTWS:23's "five points", which are not listed.
- ND_14FRAME:13's grid, which is unspecified (the identity is exactly 0 symbolically).
- The Ladgham–Lachièze-Rey citation at ND_CUBIC_SLAVING:57, which was not checked against the journal.
- All two-point numerics are mpmath NON-CERTIFYING.

### 1.3 BF transverse / D5 (BT)

**What it claims.**
- BF_TRANSVERSE_EXACT: f_ss(M) | six pins ~ N(−b, 2), exactly for every r.
- BF_D2_CONTACT_A0: A0 ~ N(0, 2), z0 = 36k^2.
- HESSIAN_JET_SIX_PINS: a table of E[f_tt]/r, Var f_tt and Var f_ts, and the order of β.
- D5_REPLAY_AND_TYPE: "MC tracks P(A<0)".
- D5_MISSING_INEQUALITY: states (D5-μ) and restates Math- PR82's det H_M ≡ 0.
- A3_BF_D2_CONTACT, GV_SPECTRUM_AND_P3 and CUBIC_B4 cover the contact moment, the G+V spectrum and periodization sizes.

**Recomputed and CONFIRMED** (exact unless marked):
- f_ss(M) | six pins ~ N(−b, 2) for every r: g = f_ss + f is orthogonal to all six pins and has Var 2.
- f_ss(S) ~ N(−(b − kr^3), 2).
- Cov(f_ss(M), f_ss(S) | pins) = 2e^{−r^2/2}.
- E[f_tt(M)]/r = −6k + kr^2 − br/4 + br^3/24 − kr^4/20 + ….
- Var f_tt = r^4/6 − r^6/30 + r^8/360 and Var f_ts = r^2/2 − r^4/12 + ….
- The table entries E[f_tt]/r = −5.841312759, −5.960080528, −5.990005008, −5.997500313 and Var f_ts = 0.0778676, 0.0198667, 0.00499167, 0.00124948.
- The G+V characteristic polynomial (λ−1)^2(λ^2 − 16λ + 6)(λ^2 − 4λ + 2), with det 12.
- diag(24, 6).
- The closed form for E[A^2 1{A<0}]: b = 0 gives 1, b = −3 gives 0.0080263, b = 1 gives 2.7201411.
- The File-5 Jacobian 6^{2/3}/3 = 2·6^{−1/3} = 1.100642416.
- The PR53 algebra_check (sha256 12ef5313…) prints ok, and Math- PR82 NOTE §2's expansions reproduce.
- No file promotes D5 (STATUS D5 AMEND; PROOF_INDEX:44; Math- #58).

**Prior review follow-through:**
- Item 1: not addressed at the pin; resolved after it.
- Item 2: not addressed.
- Item 3 (A3_UI): not addressed, and now contradicted by A3_BF_D2_CONTACT:38 (BT8).
- C3: partly addressed. The BF-only law A0 ~ N(−b, 2) with P(A0<0) = Φ(b/√2) > 0 now appears in A3_BF_D2_CONTACT.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| BT1 | **major** | CONFIRMED | BF_D2_CONTACT_A0.md:8 (also :10, :12) | A0 ~ N(0,2) and z0 = 36k^2 drop the pin value b. The correct law is A0 ~ N(−b, 2), with z0 = 36k^2[(b^2+2)Φ(b/√2) + √2 b φ(b/√2)]. The stated value is off by a factor 124.59 at b = −3 and about 11 at b = 3. It contradicts A3_BF_D2_CONTACT:22–32 (committed 6 s earlier), BF_TRANSVERSE_EXACT:13, D5_REPLAY:13, HESSIAN_JET:8 and :31, CUBIC_B4:26, and the collision note C5 at line 164. | Correct the law and z0, or withdraw the file. |
| BT4 | minor | CONFIRMED | D5_MISSING_INEQUALITY.md:9–13 (also :34; CLOSED_BF_WTWS.md:49) | Restates Math- PR82's leading-order "det H_M ≡ 0" as an exact polynomial identity on the constrained stratum. Under the exact witness equations det H_M ≠ 0 at order r^3 (MD1). Line 34 misdescribes the mechanism: PR82 §4's obstruction is the Q = 0 axis, r^2∫dQ/Q^2. PR82 is cited unpinned. | "Vanishes at leading order only; exact r^3 coefficient 3k(−24kP^3 + 3cPQ^2 + dQ^3)/Q^2." Do not import QUARTIC's λ formula (MD2). Merged: MD6 (PR #163 part). |
| BT5 | minor | CONFIRMED | HESSIAN_JET_SIX_PINS.md:20–21 | Var(f_tt \| pins) is given as 1.70e-5 and 1.00e-6. The exact values are 1.66334e-5 (r = 0.1) and 1.04115e-6 (r = 0.05), from r^4/6 − r^6/30 + r^8/360. The float64-cancellation explanation is not borne out; the cause is unknown, and no script is published. | Replace with the exact series values. |
| BT6 | minor | CONFIRMED | HESSIAN_JET_SIX_PINS.md:31 | β = O_p(r) is wrong for β = f_ts/r, which is the form the formula needs: β ~ N(0, 1/2 − r^2/12 + …) is O_p(1). r β^2 = O_p(r) still holds. | "β = O_p(1); rβ^2 = O_p(r)." |
| BT7 | minor | CONFIRMED | D5_REPLAY_AND_TYPE.md:15 | "MC tracks P(A<0) within ~0.005 at r = 0.1" comes with no script, seed, N or event definition. For the joint parent indicator the gap is ≈ 0.016 at b = 0 (the leading term arccos(e^{−r^2/2})/(2π) = 0.015902), ≈ 0.0127 at b = ±1, and O(r) (0.00813 at r = 0.05). Only the single-endpoint indicator is within 0.005 (MC, NON-CERTIFYING). | Define the event, ship the script, and state the O(r) gap. |
| BT8 | minor | CONFIRMED | A3_BF_D2_CONTACT.md:38 | Names uniform integrability of det(A_r)^2 1{A_r<0} as the A3 obstruction, but that follows at once from (3.5). The real open object is uniform integrability and type convergence of W_r/r^2 via (5.3), which needs the §4/(4.1) moments. This conflicts with A3_UI_MAJORANT:70 (DC4). | Name the W_r/r^2 object. |
| BT9 | minor | CONFIRMED | GV_SPECTRUM_AND_P3.md:30–31 (also A3_BF_D2_CONTACT.md:42) | The periodization sizes are kernel-level, not jet-level.<br>- L = 24: G+V ~3e-117, not 8.4e-126; even the kernel term 4e^{−288} is 3.35e-125.<br>- L = 2π: jets are 2e-7 to 2e-4, not 1e-9. The kernel term with the four nearest images is 1.07e-8; GV_SPECTRUM:31 quotes the single image, 2.7e-9.<br>- L = 2: O(1) (0.54 at kernel level, and G+V is near singular), not O(1e-1).<br>BF_D2_CONTACT_A0:14's derivative-aware ≤ 1e-113 is correct; SIDE24 uses 1458(76·24^6 + 15)/10^125 ≈ 2.1e-112. | Quote jet-level bounds. Merged: FM14 (nit; the kernel values 1.07e-8 and 0.54 were confirmed independently). |
| BT10 | minor | CONFIRMED | HESSIAN_JET_SIX_PINS.md:39 (also D5_REPLAY_AND_TYPE.md:20, CLOSED_BF_WTWS.md:53, ND_CUBIC_SLAVING.md:63, PAIRING_IMPLICATION_GAP.md:45) | "Still PROVEN-MODULO (ND)" narrows the File-1 label of record. For Theorem A the label is Lemma P, Lemma I and Sublemma R0, and Theorem B adds Proposition B2 (MAP:16, :25). A GitHub D1 AMEND cannot enter a File-1 label (MAP:54: no transfer). | Quote the label of record. Merged: FM11 (with CS3 and Part 2 R4). |
| BT11 | nit | CONFIRMED | HESSIAN_JET_SIX_PINS.md:33 | "File 5 (5.4)" collides with "parent (5.4)", and the formula switches from ℓ to an undefined u. | Disambiguate. |
| BT12 | nit | CONFIRMED | HESSIAN_JET_SIX_PINS.md:10 (also BF_TRANSVERSE_EXACT.md:20) | The proof sketch covers t-derivatives only. The f_s pins need (∂_ss + 1)C_2 = d_s^2 C_2 to vanish to second order, which holds because C_2 is even. | Add the s-pin step. |
| BT13 | nit | CONFIRMED | BF_TRANSVERSE_EXACT.md:5 | "Exactly for every r > 0" holds only on R^2. On T_L^2 the law is N(−(1−ε_L)b, 2 + δ_L − ε_L^2). The corrections are ~1e-122 / 5.5e-120 at L = 24, 2.1e-7 / 7.5e-6 at L = 2π, and O(1) at L = 2 (mean coefficient −0.14, variance 1.36). | Say "on R^2". |
| BT14 | nit | PLAUSIBLE | A3_BF_D2_CONTACT.md:51 | "Independent match of Lucas": Lucas is a same-lane teammate (REPLAY.json:4). No published Lucas artifact supports it; the one Lucas file, in Math- PR87, is about SARD-G and postdates the pin. | "Same-session corroboration." Merged: XC13. |
| BT15 | nit | CONFIRMED | D5_REPLAY_AND_TYPE.md:7 | The replay from 9a6f8a66 is the unmerged PR53 head, and PR55 (4e188e25) AMENDs its inner disk. 2304 is the pure-Q value only. Two of the checks are hard-coded constants. | State those limits. |
| BT16 | nit | CONFIRMED | D5_MISSING_INEQUALITY.md:17 (also :9, :19, :27, :29) | The pins are undefined. D(S_0) is undefined and should be D(C_*)∖{0}. φ_∇(0) should be p_{∇f(X) \| pins}(0). The thin-tube PROOF review is still open (PROOF_INDEX:35). S is overloaded. | Define the terms. |

**Covered elsewhere (not re-counted):** BT2 → MD2; BT3 → MD3; BT17 → BC14.

**Not verified:**
- The Lucas match, and Grok's checks at r ∈ {0.17, 0.3, 1.0, 2.5}: no scripts were published. The identity was proved symbolically instead.
- The D5_REPLAY MC itself.
- The SIDE24 package internals.
- The File-4/File-5 conventions; §8 checks them against the mirror.

### 1.4 Custody and packaging (PK)

**What it claims.**
- INVENTORY, SESSION_LEDGER, REPLAY.json and STATUS_PIN_NOTE index the packet.
- CLAUDE_FILES_1_6_MAP, C006_ARM1C_V2_INGEST and O1_PART_B_AND_GATES map the July-5 stack, which they call "ABSENT from public GitHub" with "zero hits".
- MUSEUM_TRIAGE lists "public, reviewed objects".
- PAIRING_IMPLICATION_GAP states the cap-pairing gap.

**Recomputed and CONFIRMED:**
- Every intake rule except RESULT/IDENTITY passes: 34 files, all added; one package; .md/.json only; mode 100644; no protected names; UTF-8; strict JSON (3,287 B); no credential hits.
- merge-tree is clean.
- With a synthesized RESULT.md and IDENTITY.json (35 artifact rows, 3 Math- 5ed3b455 sources) the offline checker passes, and its 60 tests pass.
- INVENTORY lines 5–16 match STATUS. The cap rationals check.
- PAIRING_IMPLICATION_GAP lines 9–20 are verbatim from cap line 28. §8 Morse plus distinct values suffices, with no Morse–Smale needed, and r < L/(4√2) is sharp.
- MAP lines 11–26 match File 1.
- All C006 numbers match Q0 34264–34384: S 0.861 / M 0.637; P-1c-1 0.643 [0.464, 0.784]; C-1 +0.292 [0.141, 0.429]; P-1c-2 2.212 (1.271); 4.36e-3 vs 8.62e-3 (ratio 1.977); N = 1195; RMSE 1.1e-15.
- SIDE24 30/30 and P15 36/36 at Math- 55a3ced.
- Lemma R3.2 is exact for m = 1..3.
- D2 = 29/6 − √6.
- The museum 2025 formulas are present.

**Prior review follow-through** (5850519011): items 1–3 were not addressed at the pin. Seventeen files were added after that review, and none of the earlier 17 was modified. After the pin, item 1 was resolved by `857904a`.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| PK1 | **blocker** | CONFIRMED | packet:0 | Public intake rejects the packet with "RESULT.md and IDENTITY.json required" (tools/public_intake_check.py:292; job 108507023652). An offline replay gives the same result. The fix is mechanical, and an offline positive control passes. **Resolved post-pin** by `857904a`. | Add RESULT.md and IDENTITY.json. Merged: BC1. |
| PK3 | **major** (panel 2 major : 1 minor) | CONFIRMED | C006_ARM1C_V2_INGEST.md:9 | "This cycle is ABSENT from public GitHub" is false.<br>- The OCR text is in main history at Q0 34264–34391 (blob ed0f51f1, at 7caac254), catalogued on main default in docs/public-math/sources-05.json:772–777.<br>- A byte-exact PDF (82,688 B, sha256 390c7157…) is inside drive/deltas/2026-09-19/DG-MIGRATION-20260919/packs/fb7c6055….zip (objects.json:3364; inventory.jsonl:3449).<br>- The mirror text comes from a different carrier (624,596 B, sha256 9989196b…).<br>- "Zero hits" may be literally true of code search, which indexes the default branch only and has a size limit.<br>- The rewrite b6b7ea3 dropped "pointer, not byte custody".<br>- A later public record, q0_machine.json (blob 13c48e95 at 7caac254/80e7d0d), records "C006-Arm1c-DA + Arm1c-v3" with T_fold M = 0.9862 self-labelled "SURVIVES". MAP:40 "Does not confirm" and O1:31 "not confirmed" omit it (from XC2). | Cite the mirror lines and the zip member; restore "pointer, not byte custody"; mention the later record as a self-label. |
| PK4 | **major** | CONFIRMED | CLAUDE_FILES_1_6_MAP.md:5 | "Zero hits … ABSENT from public GitHub. Drive/chat attachments only" is false.<br>- OCR text of File 1 (×2), 3, 4, 5 (×2), 6, GCJA and O1 is in blob ed0f51f1, with BEGIN SOURCE markers at Q0 29812, 30060, 30733, 31333, 32698, 33124, 33505, 33833 and 34720.<br>- The blob is in default history at 80e7d0d (removed from the tip by f35eef1 on 2026-09-23) and live at about 90 branch tips.<br>- The default tree catalogues it (docs/public-math/sources-05.json:772 and sources-05.md:105, added 2026-09-25, before the MAP commit).<br>As a code-search report "zero hits" holds for 5 of the 7 terms. "Pinning Lemma" hits Math- EC-014/proof.md.export.txt:120 (b39e9ff at 01:54Z, about 20 h before MAP commit 08befe6 at 22:12Z), and "near-diagonal" hits generic files. File 2 has no OCR block of its own. The error is the step from "search found nothing" to "ABSENT". | "Present in public git history as OCR text (image layer dropped): blob ed0f51f1 …, not on the default tip (removed by f35eef1)"; cite Q0 lines per File. Merged: FM1 (major, 3 passes). |
| PK2 | minor | CONFIRMED | PR #163 state:0 | Draft, and 5 commits behind main (#164 README, #157 action-pin bumps) under live ruleset 23798639 (strict checks); merge-tree clean. The body's "New this round: C006" is stale: C006 landed in 9a5502f at 17:03, and 20 later files are undescribed. Keeping it draft was right while checks failed; ready was needed because drafts cannot merge. Post-pin: merged. | Update the body; refresh the branch. |
| PK5 | minor | CONFIRMED | O1_PART_B_AND_GATES.md:43 | "Zero hits" for the Gate Framework is false: there are 2 default hits (sources-05.json:244 and sources-05.md:39, from 23e2866 on 2026-09-25). Master v1.2 is at 7caac254 as a standalone md, a 20-page PDF and Q0 7437–8551. For 4 of the 5 terms a code search does return 0. | Cite the catalog rows. |
| PK6 | minor | CONFIRMED | O1_PART_B_AND_GATES.md:13 (also :17) | \|det H_S\| = κrμ(1 + O(r)), with μ = −f_ss(x_S), is false near μ = 0. In fact \|det H_S\| = f_tt μ + f_ts^2 with f_ts = rν_S, and an exact counterexample gives ratio 101 at r = 1/100 (κ = c = 1, μ = r^2). The corrected curvature is μ̃_S = μ + rν_S^2/κ (Q0 13156). The line copies O1 B.3 (Q0 30247–30250), which the Flat-Saddle lemma superseded. Line 17's "μ ≍ r^2" holds for μ̃; the raw \|μ\| is ≍ r. | Use μ̃_S. |
| PK7 | minor | CONFIRMED | O1_PART_B_AND_GATES.md:19 (also :17–23, :37; CLAUDE_FILES_1_6_MAP.md:36, :59; MUSEUM_TRIAGE.md:51) | E[N_inner] ≍ r^4 was superseded by R3b: an O(r^5) upper bound, with the lower bound conjectural (Q0 13107–13121). The packet's own C006:35 and :47 record R3b. The conclusions survive, since Theorem A needs only E[N_inner] → 0. | "O1: ≍ r^4, superseded by R3b to O(r^5) (upper bound)." Merged: FM6. |
| PK9 | minor | CONFIRMED | CLAUDE_FILES_1_6_MAP.md:48 | Calls K4/R3/R3b "a later instrument registry". Arm1c §8 separates "this loop's additions" (instrument kills) from "Standing registry unchanged: K1–K4, R3, R3b". K4, R3 and R3b are kill-log entries continuing File 1 §9: K4 is the full-frame (ND) failure, R3 the super-exponential inner claim, R3b the r^4 → r^5 correction. Contradicts MAP:36 and C006:47. The (ND)-OPEN consequence is counted in BC5. | Describe them as kill-log entries. Merged: FM2 (label part). |
| PK11 | minor | PLAUSIBLE | REPLAY.json:45 (also SESSION_LEDGER.md:55, D1_INTERFACE_TABLE.md:38) | imports_open_parent ["main#63 marked Kac-Rice", "elder convention", "full normalizer Z", "Theorem A selection chain"] misstates what D2 depends on:<br>- the selection chain is not consumed (PROOF_INDEX:18);<br>- KR §9 and elder §8 are ACCEPTED;<br>- the definition of Z is imported, but not the A3 limit or floor (R10).<br>MAP:55 "does not consume D1" errs the other way. | List the actual imports. Merged: XC7. |
| PK12 | minor | CONFIRMED | MUSEUM_TRIAGE.md:51 | The "public, reviewed objects" include the File-1/5 architecture, the O1 r^4 retraction and the Arm1c C-2 abort. These have no review record, and r^4 is superseded. The line also contradicts MAP:5 and C006:9 ("ABSENT"). Fold lock, D2 and D3 are defensible. | "Public in git history (Q0 mirror), not reviewed", except D2/D3. Merged: FM15. |
| PK13 | minor | CONFIRMED | MUSEUM_TRIAGE.md:32 (also :24) | "No helicity_barrier.py" and "lives only under history/2025" are wrong. Branch `claude/physics-multiscale-reconstruction-a1G1I` @ 468da8da (2025-12-16) has src/helicity_barrier.py (blob dd40b08b, 19,126 B), src/gauge_theory.py and data/ace_solar_wind/helicity_barrier_data.json. README.original and body blobs are on at least 10 branches. The census was limited to the default branch without saying so; the conclusion stands. | State the census scope. |
| PK18 | minor | CONFIRMED | CLAUDE_FILES_1_6_MAP.md:7 | "Two File-1 copies and two File-5 copies are the same text" is false. The support copies carry File 6 patches: P6.1 in File 1 Scope Remark 3.4 ("analytically CLOSED modulo … Proposition A6"), and P6.2 in File 5. The copies are File 1 at Q0 33505–33828 (sha prefix 5a5bf6c7a4da) vs 34720–35036 (7361ce3e56b1), and File 5 at 32698–33119 (f52f0995e449) vs 33833–34259 (359d842fbcb6). The Theorem A/B text is identical, so the labels are unaffected. | Cite the patched copies as current. Merged: FM7. |
| PK14 | minor | PLAUSIBLE | INVENTORY.md:14, :57 (also CLAUDE_FILES_1_6_MAP.md:56) | Still lists cap-pairing Morse / distinct values / embedding as open, while EMBEDDED_CHART_AND_MORSE:50–58 and PAIRING_IMPLICATION_GAP:28, :43 discharge them, with no supersession line. | Add a supersession line. Merged: XC9. |
| PK15 | minor | PLAUSIBLE | INVENTORY.md:0 (commit history) | Five wholesale rewrites came 6–32 s after the adds (936bf56→d95fe28, 9a5502f→b6b7ea3, c781a5e→08befe6, a88c4d9→92ac0d8, db19bd6→3015c3b), which implies at least two writers. The rewrites dropped caveats: "D4 Not independently replayed", "SARD-G Untouched", C006 "pointer, not byte custody", and object ids. The git author of every commit is the owner name, with no lane attribution. | Record writers and restore the dropped caveats. Merged: XC11 (see §7 note). |
| PK16 | minor | PLAUSIBLE | REPLAY.json:11 (also :48; SESSION_LEDGER.md:21–23, :57–59) | The replays name no Math- commit, and P15 has no package path. REPLAY.json was written once (8ce22a9), covers only the ledger and has no hashes; lines 58 and 70 are stale. D5_REPLAY:7 does pin 9a6f8a66. The tests have not changed since 2026-09-24, so the counts hold. | Pin the replay commit per entry. |
| PK17 | nit (FM12 verifier: minor; tie → nit) | CONFIRMED | O1_PART_B_AND_GATES.md:45 | "Gates 1–26" / "26-gate sheet": the v1.2 reconciled master (Q0 35116–35921) defines 16 gates, plus a SCHEMA preflight, 7 domain-specific gates and 2 candidates. "26" is probably the TOC number of the Gate 16 heading. | "Gates 1–16 (v1.2)". Merged: FM12. |
| PK19 | nit | CONFIRMED | CLAUDE_FILES_1_6_MAP.md:13 | An escaped arrow renders literally inside a code block. MUSEUM_TRIAGE:11's README quote drops "multi-model mathematical". | Fix the escape and the quote. Merged: XC15 (part). |
| PK22 | nit | CONFIRMED | PAIRING_IMPLICATION_GAP.md:45 | The File-1 Theorem A label "Lemma I / ND / R0" drops Lemma P and double-counts. The File 1 label is at Q0 33595–33597; File 5's composite label differs. | Quote the File 1 label. |
| PK23 | nit | CONFIRMED | PAIRING_IMPLICATION_GAP.md:5 | 0633aca3 is a Math- blob id, unlabelled beside a sha256 and with no repo or path. EMBEDDED_CHART:5 does label it "cap blob". | Label it. |
| PK21 | nit | PLAUSIBLE | REPLAY.json:30 (also SESSION_LEDGER.md:36) | IEEE remark: the ledger's "~1e-17" is right, since naive float evaluation gives 7e-18 to 3.9e-17 outside, above the upper wall. Only the JSON key "1e-16" is loose; the first reviewer's proposed fix would have been wrong. | Rename the key. |

**Covered elsewhere (not re-counted):**
- PK8 → BC5.
- PK10 → Part 2 X3 (INVENTORY.md:46). Reword rather than delete: main pins d6628da and 58f7936d still carry the old pointer.
- PK20 → Part 2 X4. INVENTORY.md:42 is one more instance of the garble.

**Not verified:**
- Byte identity of the attached PDFs and the mirror. For C006, the inventory sha256 390c7157… and the mirror header original-sha256 9989196b1d24 cannot be reconciled without Drive.
- Grok's actual code-search queries, the Drive title search, and the "local skills" statement.
- Grok's numpy trial script.

---

## 2. main PR #159 and branch hygiene (HY) (`grok/drive-lane-map-20260926` @ `39161a6`)

**What it claims.**
- A navigation-only Drive-to-GitHub map with "Scientific effect: NONE": README.md (3,795 B, sha256 d0061505…), GAP_LEDGER.md (4,020 B, ce191cc1…) and SOURCE.json (1,664 B, 84674637…).
- It lists nine Drive lanes and the missing carriers (TRANSVERSE_CONTACT_ASYMPTOTIC and others).
- It gives a recipe for landing byte custody.

**Recomputed and CONFIRMED:**
- The packet hashes above.
- The scientific-effect firewall is respected.
- All 9 lane folder IDs map to lane names in the public mirror metadata. The R17 titles and IDs, the vault ID, and the EC-014 reading copy, AGENT13, #63 capsule and CL-RNU-003 IDs all check.
- None of the 8 named missing objects exists as a path on any ref, except the erratum.
- The RN carrier roles and the Math- #56/#58 titles check. PR53 is unmerged.
- The google-drive replica pointer resolves.
- GAP_LEDGER:27 was correct when written.

**Prior review follow-through** (5850544205, same head):
- Item 1 was not addressed at the pin; it was resolved after it.
- Item 2 (stale erratum row) was not addressed at the pin: d573b99d is reachable only through a PR ref and would fail reachable(). It was corrected after the pin.
- Item 3 was mistaken, because the pointer resolves (HY10).
- "Independently confirmed" and "Owner pre-approved" were not addressed at the pin; both were corrected after it.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| HY1 | **blocker** | CONFIRMED | SOURCE.json:0 (packet) | Public intake rejects the packet (checks 108464041400 and 108464015704; tools/public_intake_check.py:292). SOURCE.json holds Drive IDs only, with no bytes or sha. The gate parses SOURCE.json (lines 280–288) but never uses it as identity. A fix whose IDENTITY lists 4 artifacts, with Math- 55a3ced sources (KERNEL_TAILS 10,870 B, 5faa1834…; GRAPH.json 15,375 B, 065e6a75…), passes a local non-CI simulation. An admin bypass exists, so "cannot merge" means under the policy. **Resolved post-pin** (`f21e222`). | Add RESULT.md and IDENTITY.json. |
| HY3 | minor | CONFIRMED | README.md:50 | Step 1, "Keep searching Drive", contradicts the terminal condition at GAP_LEDGER:48–50. Persists after the pin. | Drop step 1 or state the stop condition. |
| HY5 | minor | CONFIRMED | GAP_LEDGER.md:21–23 | The JETMOD strings are hunt-coined descriptors of missing mathematical objects, not titles. "bridge_phi_det_Sigma_gg…" occurs on no other ref, and "phi_bridge_detA_to_detgg" occurs only as a receipt stem (the exact_missing_object field of inventable_*_receipt.json). STATUS_JETMOD.md is on the non-default branch `chatgpt/c101-source-bound-successor-20260925` @ 1ae02b9, at lines 104–105 and 130. | Mark them as descriptors. |
| HY7 | minor | CONFIRMED | README.md:51 | The recipe "land byte custody only with SOURCE.json" would fail intake. Intake needs RESULT/IDENTITY; sources from the pillar repos (main, Math-, query-, ULW) on reachable branches; and extensions .md/.txt/.json/.csv/.png of at most 262,144 B. Persists. | Replace it with the intake rules. |
| HY2 | minor | PLAUSIBLE | README.md:42 (also :50; GAP_LEDGER.md:20; PR body) | The TRANSVERSE row "Recovery still OPEN" went stale when Math- #59 merged at 55a3ced (22:43:04Z) and closed #56 as BLOCKED_ABSENT. It was accurate when written (head at 18:44:48Z). Math- PROOF_INDEX:40 and main docs/PUBLIC_MATHEMATICS.md:65 also still say open. Persists. | "BLOCKED_ABSENT (Math- #59, 55a3ced)". |
| HY4 | minor | PLAUSIBLE | GAP_LEDGER.md:25 | "CL_ANTHROPIC_BUNDLE … RN×3 SoT" is presented as a GitHub need without a source. Math- GRAPH.json @55a3ced lines 372–375 mark that edge required:false (historical_bundle_identity), and Grok's own Math- LEDGER.md:21 says "SYM-Fw-jet receipts". README:36 does attribute it to the lane index. | Cite the GRAPH edge and mark it optional. |
| HY8 | nit | CONFIRMED | SOURCE.json:6 | observed_at 18:40:00Z is later than commit 0bcae50 (18:39:17Z) and PR creation (18:39:25Z). | Correct the timestamp. |
| HY10 | nit | CONFIRMED | README.md:46 | The google-drive replica pointer resolves: d6g8k5htny-coder/google-drive ea55c6e, ENCLOSURE.json (1,090 B, blob 57af39a0, sha256 72b6cd92…), identical to the Math- file. It lacks owner and commit, and google-drive is not an intake pillar. The prior review's item 3 was wrong. Removed post-pin. | Pin owner/commit if kept. |
| HY11 | nit | CONFIRMED | GAP_LEDGER.md:25 | The "1aCa-QG9… Doc" is an uploaded markdown file, CL-RNU-003_PIECE1_RUN_PIECE2_CHARACTERISED_2026-09-17.md (7,407 B, sha256 59b8f002…), according to metadata on 94 non-default branches. | Name the file. |
| HY12 | nit | CONFIRMED | main branch `grok/drive-missing-source-hunt-20260926` @ 6200dfd1 | The branch sits on the PR #158 merge, with no commits of its own and no PR. The hunt is probably recorded in PR #159's GAP_LEDGER (inference). Part 2 P10 covers the Math- empty branches. | Delete it or record its purpose. Merged: XC20 (part). |
| HY9 | nit | PLAUSIBLE | SOURCE.json:30 | Omits the browse status of lanes 04/05/06 (README:21–23), so it covers 6 of 9 roots. github_holes_checked covers 1 of the 8 GAP_LEDGER objects, and 13 of 23 Drive IDs are missing. | Complete the manifest. |

**Covered elsewhere (not re-counted):**
- HY6 → CK3. The defect is PR #80's wording, not PR #159's.
- HY13 → Part 2 P10 (Math- `grok/congruence-erratum-on-default-20260926` and `grok/required-math-20260926` have no commits of their own).

**Not verified:**
- The Drive-side claims: exact-title search results, what the lane index says, and the AGENT13 note.
- Drive IDs that appear on no public ref (1OMe4YG…, 18NM5B_15Jt1…, 1uWTMEgtaJzA…, 1fusoqyVo7sd…, 1mDfZd3ST56d…).
- Math- #56 comments 5841875191 and 5842401492, which were not visible on the fetched page.
- What the empty branches were meant to hold.

---

## 3. main PR #161 — LB-rate THM-023 HOLD review (LB) (`review/lb-rate-thm023-hold-20260926` @ `43b3ced`)

**What it claims.** REVIEW.md recommends HOLD on landing KIMI-THM-023. It reports:
- a "certified subregion ≈ 1.30e-5" at "3.9× budget";
- C031 as "dual-hashed";
- 0.9666 × 0.9091 = 0.87853686;
- a "theorem-grade far term";
- 2.1 "under the corrected reading";
- a Drive landing at 19:30Z.

The PR body says "Does not close #116".

**Recomputed and CONFIRMED:**
- The §1 hashes and bytes: KIMI-THM-023 14,936 B 61e14810…; AUD-023 5,448 B df6565ac…; AUD-024 3,369 B 309aefb8…; MANIFEST 1,199 B 73e20364….
- The body hashes fedcc4b6…, 23377bd4… and 0633402b… (bytes-before-marker convention). The 61f2b702 and a7decb6f bodies match.
- 0.9666 × 0.946 = 0.9144036 exactly (2286009/2500000).
- WP 664/3125 r^3 = 0.21248 r^3; 57/11; 19/9.
- The finite-r coefficients 0.9143978516 (r = 0.025) and 0.9143576126 (0.05), and the threshold r^3 = 18/1839497.
- DEF-WP-CS-01, from the bytes (verify_wp line 261, sha da424a76…; LB-1 line 294).
- LB-1's above-b closure uses the square-root form (rho_bar, lines 197–238), which answers the AUD-064 §7.6 gate positively.
- The HOLD disposition is correct.
- CI is 3/3 green on 43b3ced; merge-tree is clean; STATUS is untouched.

**Prior review follow-through.** The head did not change after prior review 5850534745. None of its 5 items, and neither automated thread (P1 at line 82, P2 at line 66), was addressed at the pin. After the pin, `1c557c5` applied items 1–4. Item 5, the closing keyword, was not applied, and it was then realized.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| LB1 | **major** | CONFIRMED | REVIEW.md:82 (also :80) | "Certified subregion ≈ 1.30e-5 … 3.9× budget": the figures are LB-1's own wp_rho output (the linear min(Cantelli, P_W), certificate line 294), which AUD-064 §7.5 decertified. The −y argmax is a phantom (≤ ~1e-21; recomputed ~7e-21).<br>What survives:<br>- I_true = 4.7602e-6, derived on a grid. That is 0.3047 r^3, or 1.43029× (190408/133125) the budget 0.213 r^3 = 3.328125e-6.<br>- The true hot spot is (0, 0.5875), ρ ≈ 1.07e-4.<br>The two defects are independent in substance, but line 82's numbers are contaminated. Showing the budget is exceeded needs a truth value or a certified lower bound; I_true is not theorem-grade. The 1.33e-4 comes from a transcript, not LB-1 §9 (1.313e-4). Prior item 1 and thread P1. Fixed post-pin. | Use I_true = 4.7602e-6 (1.43×) at derived grade. |
| LB3 | **major** | CONFIRMED | PR #161 body:0 | "- Does not close #116." contains a closing keyword, so #116 listed #161 under closed_by_pull_requests and the merge would close it. Keyword links cannot be removed from the sidebar, so prior item 5's sidebar advice was wrong. **Realized post-pin:** #116 was closed at 17:51:22Z, state_reason completed. | Reopen #116 and record why; avoid closing keywords in negations. |
| LB2 | minor | CONFIRMED | REVIEW.md:26 | "Dual-hashed" conflates two files. e165821b is C031_Freeze.md (1,739 B, blob f61db50c), which is printed as "Freeze:" in the Integration header. C031_LBRATE_Integration.md (13,690 B, blob ec2087e6) is e7998ef0. LB-1 and DER-027c cite C031 by the freeze hash for ledger content. Related: Part 2 G6 (the same conflation on issue #160). Fixed post-pin. | Give each file its own hash. |
| LB4 | minor | CONFIRMED | REVIEW.md:66 | 0.9666 × 0.9091 is 0.87873606 (43936803/50000000), not 0.87853686 (off by −1.992e-4). #160's "≈0.8786" should read 0.8787; the #116 comment's "≈0.879" is right. Prior item 3. Fixed post-pin. | Correct the product. |
| LB6 | minor | CONFIRMED | REVIEW.md:66 | "Theorem-grade far term": sup p̄ = 0.0334 is graded "measured" (THM-023:122, C031:36, GP-LB-REC-001:77). Only the DER-009 reduction is theorem-grade, and THM-023:147 contradicts its own line 122. Prior item 4 and thread P2. Fixed post-pin. | "Measured far term; theorem-grade reduction (DER-009)". |
| LB8 | minor | CONFIRMED | REVIEW.md:108 | The §5 AUD-064 summary omits "Limit constant 0.9144036: value stands at measured grade; remainder order: OPEN" (AUD-064 §7.7 lines 233–236, verdict 392–393; DER-025 §6; GP-LB-REC-001:65). The review's own departure in §3 (0.909) is not marked as a disagreement. | Quote the AUD-064 line and mark the departure. |
| LB9 | minor | CONFIRMED | REVIEW.md:28 | "Drive landing of these exact bytes: 2026-09-26T19:30Z": the same three hashes were uploaded and SHA-read-back on 2026-08-04 (GP-LB-REC-002, blob 6c2a69a6; Drive IDs 1gD9hXfm…, 1rIXNLWo…, 1UGKmNYH…). 09-26 is at most a re-landing, and it cannot be verified. README925.txt comes with no bytes or hash; #160's 2,768 B equals the size of the export README.txt (sha 04aa9dc4…), which carries a "proved floor" banner. Persists (now line 30). | Cite GP-LB-REC-002 as the first landing. |
| LB10 | minor | CONFIRMED | REVIEW.md:25 | The identity split mislabels: 61f2b702 is the body of the outer Unicode v1.0 AUD-023 carrier (whole file 0dfab4f0, 5,498 B), and the normalized landed file is also v1.0 (body 23377bd4), so the split is by carrier family, not version. AUD-024's orphan citation 40596829 is omitted; it is UNRECOVERABLE and was re-pointed to ba3974c0 by KIMI-DATA-028/AUD-064. AUD-064 seals THM-023 v1.0 as outer ba3974c0; fedcc4b6 is right for the normalized file. | Relabel the split and add the orphan citation. |
| LB12 | minor | CONFIRMED | REVIEW.md:74 | "2.1 stands only under the corrected reading (9 − 2.25)" is not the only reading.<br>- 18.75/6.75 = 25/9 is the plain area ratio.<br>- Area(0.5–3) gives 1881/875 = 2.1497.<br>- AUD-023's alternative is 1.82.<br>GP-LB-ERR-001 labels it PROVISIONAL AUTHORING ERRATUM / SOURCE-GEOMETRY CONFIRMATION PENDING. Also 3.4e-6 ≠ 2.1·(ℓ/2) = 2.734375e-6. #160 says "CLOSED under corrected reading". Persists (now line 76). | List the readings and keep PROVISIONAL. |
| LB13 | minor | CONFIRMED | PR #161 body:0 | The ticked box "[x] exact rational arithmetic or NON-CERTIFYING" is contradicted by LB4 and LB5, and no script ships. Mitigation: CONTRIBUTING:127 limits the rule to "wherever a bound is claimed". | Untick or ship the arithmetic. |
| LB14 | minor | CONFIRMED | commit 43b3ced subject | "… and STATUS AMEND row": the commit touches only REVIEW.md. Prior item 5. | Correct the description. |
| LB11 | minor | PLAUSIBLE | REVIEW.md:14 (also :28, :66, :118, :119) | No commit, blob or path pointers ("prior read"), while other main reviews pin blobs. AO48-AUD-064 is mirrored at 7caac254 but not indexed in sources-*.json. The hashes do match public blobs (AUD-064: blob 14a942de, 25,860 B, sha256 406406fb…). | Pin the blobs. |
| LB5 | nit | CONFIRMED | REVIEW.md:64 | "0.1759\overline{16}" should be 19/108 = 0.1759259… (0.17\overline{592}). THM-023:64 prints 0.1759 correctly. Fixed post-pin. | Correct the repetend. |
| LB15 | nit | CONFIRMED | REVIEW.md:4 (also PR body, #160, commit message) | "Same-provider technical pass": the packet's authors are other lanes (Kimi, AO48, GP), so this is not same-provider. Zero credit rests on source exposure and the single operator (CONTRIBUTING:152–155). Fixed post-pin. | State the actual ground. |
| LB16 | nit | CONFIRMED | issue #160 body (last line) | Targets reviews/lb_rate_thm023_hold_20260926/; the file landed under reviews/lb_rate_thm023_landing_20260926/. | Correct the path. |
| LB17 | nit | CONFIRMED | REVIEW.md:90 | "≈0.91440" shows the same digits as 0.9144 (the exact value is 0.9143978516). The threshold r = 0.0213890325… is written "0.02139…", an ellipsis after rounding up. GP-LB-REC-001:73 prints it correctly. Persists (now line 92). | Print the digits correctly. |
| LB7 | nit | PLAUSIBLE | REVIEW.md:66 (also :90) | The band 0.9091 ± 0.038 ± 0.009 is not carried through: 0.9666 × [0.8621, 0.9561] = [0.83330586, 0.92416626], which contains 0.9144036. The review's grade/scope claim is still right. The C031 rung sequence (C*(0.05) = 0.9728 > C*(0.025) = 0.946, O(r)) extrapolates to 0.9192 / 0.90929, below 0.946 (NON-CERTIFYING), so "not excluded either" is too generous. | Carry the band. |
| LB18 | nit | PLAUSIBLE | REVIEW.md:121, :129 | The scope "3D Side-24" is too narrow. Line 8's unqualified "SIDE24 coefficient status" is the right firewall, since D3 covers c_2,24 and c_3,24 on T^2_24. "D0–D7 3D lifetime" is inaccurate, because D3 and D5 include d = 2. | "D3 SIDE24 coefficient status (c_2,24, c_3,24)". |

**Covered elsewhere:** no exact duplicate. Related: Part 2 G6 (the issue #160 body).

**Not verified:**
- The Drive-only claims (the 19:30Z landing, the Drive root, README925.txt).
- #160's claim that a Drive copy of the 13,690-B file hashes to e165821b.
- C*_∞ = 0.9091 ± 0.038 ± 0.009 (the C027 quadrature was not recomputed).
- The DER-025 truth-engine values: only the CS-form spot values were recomputed, NON-CERTIFYING.
- Full re-execution of the LB-1/2/3 certificates.

---

## 4. main cycle-2 branch (`incoming/grok-cycle2-nd-d5-sard-20260926` @ `8dfde10`)

**What it claims** (RESULT.md and SESSION_LEDGER.md; no PR). Sources are said to be "Pinned in IDENTITY.json".
- **Result 1**, a "repaired ND sentence":
  - W_s/z_s^2 has unconditional variance 15/4.
  - det Gram = 45/16 z_s^8.
  - "ND-as-written remains refuted".
  - On the axis the W-block must be dropped, or the KR prefactor log-diverges.
  - "File-1 Theorem B stays PROVEN-MODULO this repaired sentence".
- **Result 2**, the D5 transverse cone at (r, b, k) = (0.3, 1, 1):
  - c ≈ 1.75, with p0 q^2 stable.
  - The rate is "O(r^3 q^2), not the O(r^6 q^2) of #58".
  - "E\|det H_S\| stays O(r)".
  - The on-axis 9-pin Gram is singular.
- **Result 3**, the SARD-G A1 repair predicate, is said to be "written in SARD_G_A1_REPAIR_PREDICATE.md".

**Verdict:** AMEND, not landable as it stands.

### 4.1 D5 numerics (CN)

**Recomputed and CONFIRMED** (NON-CERTIFYING throughout). Planar BF was rebuilt independently with physical q, X = M + (0, q):
- The HESSIAN_JET table reproduces.
- At (0.3, 1, 1), E\|det H_X\|/q = 1.7420, 1.7515, 1.7537, 1.7542, 1.7543 for q = 0.2 down to 0.0125.
- E\|det H_M\|/E\|det H_X\| → 1.
- p0 q^2 = 0.3598 … 0.4160; the leading term e^{−b^2/4}/(2π r q^2) gives 0.4132.
- p0 E\|det H_X\| q = 0.6268 … 0.7298, tending to the r-independent k(3/π)^{3/2} e^{−b^2/4} = 0.7267.
- c ≈ 5.86 k r, a dependence the packet does not state.
- Z_r/r^2 = 91.4, 95.0, 97.1, 97.8 → 97.9.
- The weighted density ρ_W/r stays bounded (0.12–0.18).
- The cited source paths exist, and the #58 quote is correctly attributed.
- Status discipline is correct (D5 stays AMEND).

**Prior review follow-through:**
- #163 item 1 (IDENTITY.json) was not addressed: cycle 2 repeats the omission.
- #163 item 2 was addressed: the sources cite 55a3ced.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| CN6 | minor | CONFIRMED | RESULT.md:43 (also SESSION_LEDGER.md:11) | "On-axis 9-pin Gram remains singular": the 9-functional set is never defined. Every natural reading tested is exactly positive definite at finite r for X ≠ M, S (λ_min 5.4e-12 … 8.3e-24 as X → M). float64 matrix_rank returns 8 for \|p\| ≤ 0.03, which is the likely source. The Gram is singular only at the collisions X = M (rank 6) and X = S. The r → 0 rescaled limit (PR53 NOTE:76) vanishes on q = 0. | Define the set; "PD at finite r; ill-conditioned; the rescaled limit degenerates at q = 0". |
| CN5 | minor | PLAUSIBLE | RESULT.md:13 | The task is a "D5 transverse-cone KR majorant", but Result 2 is a single ray (p = 0) at a single point (0.3, 1, 1), with no W_r/Z_r and no inequality. It reads r-exponents off one r: 104 r^3 q^2 fits as well as 347 r^4 q^2 at r = 0.3, where r^2 = 0.09 > r/4. "Not an expected-count lemma" mitigates this. The angular Mahalanobis distance behaves like 0.5 + 71 tan^2 θ as \|s\| → 0. | Scan r, and state no exponent from a single r. |
| CN8 | minor | PLAUSIBLE | RESULT.md:53 (also :31) | Says "numerical tables are NON-CERTIFYING", but ships no tables, scripts, q-sequence, MC n or seed. M, S, b, k, the pins, the "transverse cone" and "9-pin" are undefined, only r = 0.3 is given, and the "0.41" appears in no file. There is no output.json and no task link (CONTRIBUTING:50–51). Intake refuses scripts (lines 98–101), so a missing script is not by itself a violation. | Ship output.json with definitions and the grid. Merged: CS9. |
| CN4 | nit | PLAUSIBLE | RESULT.md:43 | "Extra soft factor cancels the 1/q singularity" is correct for the named integrand (line 42 shows the q) but ambiguous. #58 says M and X each gain a factor, while the unweighted computation shows only X. The weighted W_r/Z_r density is bounded (≈ C(b,k) r, with C from ≈ 0.03 at b = 2 to 1.6 at b = −1) but is never evaluated. D5_MISSING_INEQUALITY belongs to PR #163, not to this packet. | Say which factor and which integrand. |

**Covered elsewhere (not re-counted):**
- → Part 2 X1:
  - CN1: the headline "O(r^3 q^2), not #58's O(r^6 q^2)". The joint product is ≈ 365 k^3 r^4 q_phys^2 = 365 k^3 r^6 q_s^2 (r-scan 331, 360, 367, 368; k^3 law). The 365 constant holds on the axis only (off-axis 1.2e3 … 4e5).
  - CN2: "E\|det H_S\| stays O(r)". The sharp value is ≈ (12/√π) k r^2 = 6.770 k r^2 for q ≪ r, Θ(r(r+q)) in general; O(r) is the six-pins-only law.
  - CN3: physical q vs scaled q.
  - CN9: the SESSION_LEDGER E-D5-2 row locks the error in.
- CN7 → CS1 and Part 2 G5.

**Not verified:**
- Grok's own code, q-sequence, sample sizes and any other r values. The reconstructed setup matches the reported figures, but it cannot be confirmed to be identical to Grok's computation.
- The intended "9-pin" set.
- The leading constants 6k√(3/π), 12/√π, (3/π)^{3/2} and e^{−b^2/4}/(2π) are heuristic asymptotics that match the numerics to 3 digits; they are not proved.

### 4.2 ND / SARD / packaging (CS)

**Recomputed and CONFIRMED** (exact):
- The branch adds 2 files (75 insertions).
- The named sources exist: ERRATUM (1,782 B, blob 213594d6); PROOF_INDEX (12,954 B, blob da0b1d7c, sha256 7e5c9f64…); STATUS (6,881 B, blob 52688102, sha256 9c3e2114…); the successor REVIEW (25,057 B, blob 9fc61b47, sha256 894d27c0…).
- The package id is valid, and the landing and shop checks pass offline.
- 45/16 z_s^8 is the unconditional value. Cov(f_tss, f_sss | six pins) = diag(2, 6) for every r, which gives (3/4) z_s^8.
- The rank is ≤ 1 after f_tss (0 on the axis). The Euler relation gives det 0. Var R checks.
- Cycle 4 (Math- PR87) has Var(f_sss | six) = 6 and Var(f_tss | six) = 2.
- The program board matches STATUS, and the A1 AMEND is consistent with PR #122 head a1fc9581.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| CS1 | **blocker** (latent: no PR yet) | CONFIRMED | RESULT.md:18 | "Pinned in IDENTITY.json", but no IDENTITY.json exists. tools/public_intake_check.py:292 would reject any PR from this branch. An offline positive control, with an IDENTITY built from the four sources RESULT.md names, passes (verified_sources 4). The sources are named by full SHA and path in prose; only the sha256 manifest is missing. The branch was committed 55 min after the #163 review comment that asked for IDENTITY.json. Related: Part 2 G5 (the #165 custody framing). | Add IDENTITY.json before any PR. Landed intake is immutable. Merged: CN7, XC10 (part). |
| CS3 | **major** | CONFIRMED | RESULT.md:35 (also SESSION_LEDGER.md:10, :18) | "File-1 Theorem B stays PROVEN-MODULO this repaired sentence":<br>- The sentence is never stated, and nothing shows that Lemma I (some β > 0) survives.<br>- Given the pair frame, the inner-gradient Gram is diag(0, 3z_s^4/2). The reduced frame therefore keeps only one KR constraint, and its density is not integrable at the axis (IS1, IS2).<br>- "Stays" hides a change of hypothesis: the Euler collapse is an open-set collapse.<br>- PR #163 was more careful ("candidate (not a theorem)", "not a STATUS row"). The dependency drop is inherited from PR #163's shorthand.<br>- File 5 §8 self-labels Lemma P and R0 as DISCHARGED, so the items dropped from the label are B2, Lemma G/A′ and B0–B1.<br>- RESULT:49 "certified 13×13 File-3 matrix" credits a Grok frame to File 3. File 3's Table 4.1 has 14 pair rows (Q0 30919, 30949, 30953), is not in the carrier, and the literal ladder is degenerate (FM4). PR #163's "PROVEN-MODULO (ND)" comes from File 5's label of record (Q0 32724, 33093) (from XC6).<br>No STATUS effect. Related: Part 2 R4 (the PR87 surface). | "Theorem B: PROVEN-MODULO (File 5 label) {(ND), refuted as written; (ND′) replacement per GCJA C4} ∪ {Prop. B2 finite computation} ∪ {File 1 §6 executed obligations}; no Grok-authored sentence enters the label." Merged: FM11 (part), XC6, IS6 (part). |
| CS2 | minor (panel 1 major : 2 minor) | CONFIRMED | RESULT.md:45 | SARD_G_A1_REPAIR_PREDICATE.md exists on no ref of any account repo. The prescription it describes (relative-interior first hit plus an open tube) is already in the successor review §5, lines 157–171 (esp. 161 and 165). RESULT.md:25 lists that review as a source but does not attribute the prescription to it. SESSION_LEDGER:12 (E-SARD-2) says the file was written. The later Math- cycle-4 files (77d1a03, 8657130) postdate 8dfde10 and differ. Related: Part 2 K5 and G5. | Ship the file or cite the review §5. |
| CS10 | minor | CONFIRMED | SESSION_LEDGER.md:10 | E-ND-2 "CLOSED as repaired sentence" is a self-closure inside a REVIEW_REQUIRED packet (RESULT:4; SESSION_LEDGER:14, :22 "Not CORE-CLOSED"; CONTRIBUTING:147). The sentence was never written (#165's ND_REDUCED_FRAME.md does not exist), and the Effect column swaps in an author-side Theorem B label. | "PROPOSED"; Effect: none. |
| CS6 | minor | PLAUSIBLE | RESULT.md:35 | "ND-as-written remains refuted": "remains" silently upgrades a kill that cycle 1 declined ("Does not accept or kill", "not a certified kill"; D5_MISSING_INEQUALITY:38 "still OPEN"). Line 35 cites only positive repaired-frame quantities; the Euler / det-0 argument is only in SESSION_LEDGER:10. The mathematics is right (K4; BC5). | Cite the refutation itself (the Euler relation; K4 in the public mirror, Q0) instead of a Drive-only source. |
| CS8 | minor | PLAUSIBLE | RESULT.md:27 (also :35) | Result 1 restates PR #163 without naming or pinning it: FILE3_CUBIC_LEFTOVER:29, 39, 43, 47, 49; ND_14FRAME:15, 23, 24; ND_CUBIC_SLAVING. (cb774284 is pinnable.) "Additive continuation" is acknowledged but not named. It relabels PR #163's "candidate" as a "repaired sentence" / "CLOSED". The log-KR claim is not in PR #163. | Cite PR #163 @ cb774284 and keep "candidate". |
| CS5 | nit | PLAUSIBLE | RESULT.md:35 | det Gram 45 z_s^8/16 is unlabelled, although the Method (line 31) says six-pin conditioning. The six-pin / G+V value at cubic order is (3/4) z_s^8; the finite-r Schur value depends on r (0.8708 at r = 0.4 → 3/4). The refutation follows from linear relations, not from a determinant. PR #163 labels both values correctly, so the two packets do not disagree. | Label it "unconditional". Merged: BC13. |
| CS11 | nit | PLAUSIBLE | SESSION_LEDGER.md:6 | The "-2" suffix is a cycle number, but no cycle-1 E-ledger exists (PR #163 uses no E-IDs). E-ND-2 went stale 23 m 33 s later with the Math- f3af9b5 cycle-4 amendment, and nothing cross-references it. | Cross-reference the amendment. |
| CS12 | nit | PLAUSIBLE | RESULT.md:0 | No provider line, only the package id. RESULT:27 already states author-side status and zero credit, and PUBLIC_SHOP_SETUP:31 applies to nonauthor reviews, so it does not bind here. Precedent for adding one: side24-identity-replay RESULT:11; PR #163 SESSION_LEDGER:1, :88. | Add an author/provider line. |

**Covered elsewhere (not re-counted):**
- CS4 → Part 2 X2 (15/4 used as a certificate).
- CS7 (log singularity) → IS1.
- CS9 → CN8.

**Not verified:**
- Whether a repaired reduced frame would be enough for Lemma I: there is no written sentence and no KR integrand.
- The live intake result, because there is no PR.
- The unit took the File 3 statements from PR #163's transcription; §8 later checked them against the public mirror.

---

## 5. Math- PR #80 — contact-kernel substitute (CK) (`grok/contact-kernel-substitute-20260926` @ `a79007e`)

**What it claims.** SUBSTITUTE.md is a "contact-kernel substitute" for the absent TRANSVERSE_CONTACT_ASYMPTOTIC.md. It covers:
- the six-pin cubic family and the (A, c, d) solve;
- the type window and the type mass J;
- the contact prefactor.

It says it "may be cited in its place", calls itself "machine-checked" by verify_substitute.py, and the commit title says "verification-grade".

**Verdict:** AMEND. Accept it only as a navigation-only reconstruction.

**Recomputed and CONFIRMED** (exact):
- The six-pin cubic family has rank 6, with 4 free jets {t^2, s^2 t, s t^2, t^3}.
- The (A, c, d) solve has det v^6/24.
- B_M, B_S and B_X, and the three determinant identities.
- P_X = (w+1)^2 − 4θw.
- The window I_θ, of length 2(1 + √θ − √(1−θ)).
- The sample w = −1, A = −3, c = 75, d = −363/2, with dets 18, −18, −18.
- The R identity (3.1); degrees 6 and 3.
- 77248/945, 704/135, and J = 27392/315 in both integration orders.
- Area 2; 104976; 2916; 2916J = 8875008/35.
- The prefactor 104976 k^8/(z0 \|v\|^13).
- NOTE blob 3ee30829 is correct.
- merge-tree is clean, and the transition audit passes.

**Prior review follow-through:** the prior PR80 review's items 1–3 and the automated reproducibility flags were not addressed at a79007e.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| CK2 | minor (panel 1 major : 2 minor) | CONFIRMED | Math-:SUBSTITUTE.md:6 (also :10, :111, :157) | "May be cited in its place" / "instead of" moves citations for the cubic algebra from a reviewed and tested carrier (PR25 REVIEW R1 ACCEPT plus algebra_check.py) to an unreviewed, untested note.<br>- The type mass J (KERNEL_TAILS §4) is tested but not reviewed on default; a nonauthor Q1 ACCEPT exists only off-default, at 5913b004. J is KERNEL_TAILS's own content.<br>- Line 109 paraphrases KERNEL_TAILS lines 8–12 ("expanded attached exposition") without attribution.<br>- The off-default audit e14d411 says "not a substitute". | Navigation-only reconstruction; cite the carriers. |
| CK5 | minor | CONFIRMED | Math-:SUBSTITUTE.md:109 (also :14, :21, :158) | Cites the PR25 ACCEPT without the reviewer's identity and without its "Organizational independence is not awarded" (REVIEW lines 32 and 135). The PR25 reviewer is the same provider as this note's author (line 4). PROOF_INDEX:39 is also unqualified, which is the repo convention. | Add the qualifier. |
| CK1 | minor (panel 1 major : 2 minor) | PLAUSIBLE | Math-:SUBSTITUTE.md:153 | The companion verify_substitute.py is on no ref of any account repo. So "machine-checked" (lines 24, 74), "SymPy, this session" (81, 101), the §5 PASS table (135–151) and the commit title's "verification-grade" cannot be checked. All 13 rows do reproduce from existing stdlib Math- scripts: reviews/pr25_contact_kernel_20260925/algebra_check.py (sha256 51b208df…, 17 OK) and reviews/contact_kernel_tail_20260925/test_contact_tools.py (24 tests OK). Line 145 mentions "prior PR25 script" without a path. | Commit the script, or cite the existing ones. Merged: XC10 (part). |
| CK3 | minor | PLAUSIBLE | Math-:SUBSTITUTE.md:10 | In tension with Math- #56 item 4 and with PR #159's GAP_LEDGER do-not-substitute column. This is a conflict of stance, not a direct contradiction: item 4 guards against a misidentified carrier, and the note denies being that carrier. LEDGER b28cf25:11 "Substitute emitted" is consistent with its own rule (line 5). | Drop the substitute role. Merged: HY6. |
| CK6 | minor | PLAUSIBLE | Math-:SUBSTITUTE.md:113 (also :111, :115) | The display E_{Q_r^W} N_j(rE) = r^3 ∫Λ_j + o(r^3), with Λ_0 = Λ_2 = 0, leaves N_j, Q_r^W, W_r, Z_r, F_j and E undefined. A height window is needed: there is an exact counterexample at θ = 2 (u = 2, v = 1, k = 1, w = 1: q = −18, A = −6, c = 27, d = −57/2; X is a local minimum with det 36 and trace 51/2). This is a packaging/category over-claim, not a math error: lines 109 and 158 point to NOTE §B–C, and lines 38 and 117 restrict to 0 < θ < 1. | Define the terms and state the window. |
| CK8 | minor | PLAUSIBLE | Math-:SUBSTITUTE.md:130 | "allcell_fdz_enclosures.py" should be .json (GRAPH.json lines 43, 367). Some names are truncated: CL_ANTHROPIC_BUNDLE, and "cancelled_detgg" and "StationBox" are fragments of one name, so 6 items are listed for 5 carriers. The JETMOD names are not GRAPH nodes. | Use the GRAPH names. Merged: XC17 (part). |
| CK9 | nit | CONFIRMED | Math-:SUBSTITUTE.md:91 (also :94) | Factor labels: the contact Jacobian is 24/\|v\|^6 (v^6/24 is the minor), and the height element is k. The product 104976 k^8/(z0 \|v\|^13) is right. "height/contact" is copied from KERNEL_TAILS:68; line 94 is a new error. | Relabel. |
| CK10 | nit | CONFIRMED | Math-:SUBSTITUTE.md:125, :127 | The combined "OPEN / BLOCKED_ABSENT" label mixes open mathematics (η → 0, the axis chart) with absent carriers. Author-side limiting-kernel axis bounds exist (KERNEL_TAILS §2 (2.1); frontiers/contact_kernel_tail_20260925/NOTE.md §§3–4). | Split the label. |
| CK4 | nit | PLAUSIBLE | Math-:SUBSTITUTE.md:158 (also :14) | "PR25 REVIEW R1–R4 ACCEPT at ad35e46": the object pin is correct (ad35e46, NOTE blob 3ee30829), but the record pin is missing (REVIEW.md blob ddf68599, recorded 8373c3a7, merged 30925ca). KERNEL_TAILS is unpinned; its two versions (10,695 B before #59, 10,870 B after) differ only at line 11. | Pin the record and KERNEL_TAILS. |
| CK7 | nit | PLAUSIBLE | Math-:SUBSTITUTE.md:22 | The branch base 66e39d1 predates #59 (718029c), so the in-tree KERNEL_TAILS:11 is pre-#59 text, and the file never mentions #56 or #59. This is not an authoring error: a79007e (19:02:52Z) predates #59 (22:43:04Z). A merge would carry #59's line. | Merge main. |
| CK11 | nit | PLAUSIBLE | Math-:SUBSTITUTE.md:4 (also :29, :72, :76) | P is overloaded (the cubic, and P_M, P_S, P_X). The file gives no PR number of its own and is not linked from PROOF_INDEX. "Q[k,q,u,v,θ,w]" should be Q(k,u,v,θ,w), since the entries carry 1/v^2. | Rename; link the file. |

**Not verified:**
- The contents of TRANSVERSE_CONTACT_ASYMPTOTIC.md: no carrier exists.
- Grok's "this session" runs.
- Hosted CI on PR80: the Math- API was excluded, so only merge-tree and the transition audit were simulated locally.
- The exact wording of Math- #56 and of the prior PR80 reviews, which were read through a summarizer.
- The analytic content of PR25 R2–R4.

---

## 6. Math- PRs #81 and #82 — D5 microdisk and drive-hole ledger (MD)

**What PR82 claims** (`reviews/d5_microdisk_20260926/`, NOTE.md and QUARTIC.md at `b190a4d`; commits a6cee5a and b190a4d):
- On the six-pin cubic in the microdisk chart X = M + r^2(P, Q), "det H_M = 0 identically".
- A det H_X coefficient.
- "Rate boundary — no O(r^3) lemma".
- QUARTIC.md lifts the cubic with λ(x^2 − r^2/4)^2 to det H_M = −12λkP^2 r^3/Q^2, with "next step: law of λ".

**What PR81 claims** (LEDGER.md at `b28cf25`): a drive-hole carrier ledger. It covers the PR80 substitute, CHART_SIDE_JETMOD_PLAN, allcell_fdz_enclosures, the PR53 rows and the microdisk.

**Verdict:**
- PR82: REJECT as stated. It was merged post-pin, so the defects now need an erratum on Math- default.
- PR81: AMEND.

**Recomputed and CONFIRMED** (exact):
- NOTE §2's nested expansions.
- DΦ/D(P,Q) at r = 0 equals B_M; H_M = rB_M and H_S = rB_S.
- The unconstrained det H_M = r^2 det B_M, and the triple product is O(r^6).
- The leading solve q = −12kP/Q, A − c/2 = −6kP^2/Q^2, unique for Q ≠ 0.
- det H_S = 6kr^2(−12kP^2 + cQ^2)/Q^2, the same under the exact solve.
- ∂(f_x, f_z)/∂(P, Q) = r^4 det H_X.
- The microdisk lies inside the PR53 disk.
- C6_REMAINDERS §4's quartic row 4λru(u^2 − 1/4).
- The QUARTIC pins, f_x and H_M displays.
- At P = 0 the exact product is −54k^3 c d^2 Q^2 r^8 = −54k^3 c d^2 q^2 r^6, which is O(r^6 q^2), #58's suggestion.
- Cycle-2's "E\|det H_M\| tracks E\|det H_X\|" agrees with det H_M + det H_X = O(r^4).
- In LEDGER: the §2 GRAPH carriers and the PR80 numbers are exact, the ERRATUM blob is identical, the PROOF_INDEX statuses match, and no forbidden filename was minted.

**Prior review follow-through:**
- None of the prior PR82 reviews (three review lanes) was addressed: the branch had not moved at the pin.
- The prior PR81 review was not addressed. Its finding 1 was partly wrong (see MD11); its findings 2 and 3 are confirmed.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| MD2 | **blocker** | CONFIRMED | Math-:reviews/d5_microdisk_20260926/QUARTIC.md:29 (also :7, :31, :40) | det H_M = −12λkP^2 r^3/Q^2 mixes orders.<br>- Solving the witness equations exactly, λ cancels at order r^3: q₁ = (−12kP^2 + 4λP + cQ^2)/Q and A₁ = −(−4λP^2 + cPQ^2 + dQ^3)/(2Q^2).<br>- The r^3 coefficient is the cubic's own 3k(−24kP^3 + 3cPQ^2 + dQ^3)/Q^2, and λ first enters at r^4.<br>Exact counterexamples, (k, P, Q, c, d, λ):<br>- (1, 1, 1, 0, 24, 5): det H_M = −144r^4 − 360r^5, where QUARTIC predicts −60r^3.<br>- (1, 1/2, 1, 0, 0, 7): −9r^3 + 12r^4, where QUARTIC predicts −21r^3.<br>- λ = 0 gives −9r^3 where the premise predicts 0.<br>The expressions are rational in Q, not polynomial, and "next step: law of λ" is wrong. **Landed on Math- default post-pin** (PR #82 merge), so this now needs an erratum rather than a merge block. | Replace with the exact-solve coefficient and withdraw the λ program. Merged: BT2 (major on the PR #163 panel). |
| MD1 | **major** (adj. from blocker) | CONFIRMED | Math-:…/NOTE.md:44 (also :14 "exact on that cubic", :46, :54, :78; commit a6cee5a) | "det H_M = 0 identically" holds only for the leading solve ("Leading witness Φ = 0", line 38).<br>- Under the exact constraint f_x(X) = f_z(X) = 0 the r^2 coefficient is 0, but the r^3 coefficient 3k(−24kP^3 + 3cPQ^2 + dQ^3)/Q^2 is not. It equals minus the exact r^3 coefficient of det H_X: M and X pair as a fold.<br>- Line 54's "r^4 row which this cubic does not contain" is false; lines 21–22 show that row.<br>- Exact rational certificate, k = 1, P = 1/2, Q = 1, c = d = 0, r = 1/1000: q = −1999/333, A = −3996001/2664000, det H_M = −3996001/443556000000000 ≈ −9.009e-9.<br>- The exact solve is q = −(12kP − r(12kP^2 + cQ^2))/(Q(1 − 2Pr)). At k = P = Q = 1, c = d = 0 it gives det H_M = −72r^3(1−r)^2/(1−2r)^2. | "Vanishes at leading order; exact r^3 coefficient 3k(−24kP^3 + 3cPQ^2 + dQ^3)/Q^2." |
| MD3 | **major** (panel 2 major : 1 minor) | CONFIRMED | Math-:…/NOTE.md:52 | The det H_X r^3 coefficient is the leading-solve value, exactly twice the exact one (the exact value is 72k^2P^3/Q^2 − 9ckP − 3dkQ). Certificate: det H_X = 9.0045e-9, against the formula's 1.8e-8. This hides the fold relation det H_M = −det H_X + O(r^4). §4 does not use the coefficient. | Replace it with the exact coefficient. Merged: BT3 (minor on the PR #163 panel). |
| MD4 | **major** | CONFIRMED | Math-:…/NOTE.md:64 (also :56 heading "Rate boundary — no O(r^3) lemma", :60–68, :72; commit title) | Presents the divergence of a crude majorant as an obstruction.<br>- Its power count r^-6·r^4·r^-2 = r^-4 uses the unconstrained product. The constrained orders give r^4, and the exact Jacobian ∂(f_x, f_z)/∂(a, q) = r^5 Q^2(1 − 2rP)/2 gives r^5, which is PR69's microdisk O(r^5).<br>- The divergence comes from dropping the Gaussian factor of the solved jet q = −12kP/Q. With that factor only a log remains, and with the determinants included the integral is finite (≈ 33.5, NON-CERTIFYING; an earlier 33.6736 was a quadrature artifact).<br>- "All-height S = O(1)" is not a separate regime, since S = −6krP^2/Q^2 + O(r^2).<br>- Line 72 calls it "the obstruction". | "A crude majorant diverges; this is not an obstruction." Drop "no O(r^3) lemma". |
| MD5 | **major** | PLAUSIBLE | Math-:…/NOTE.md:68 | Neither cites nor rebuts PR69 (ae45d351: O(r^5) microdisk, O(r^3) pin disk; filed about 17.5 h earlier) or PR74 (eb8bf7a5: six same-provider ACCEPTs). The soft factor is not the point of disagreement, since PR69's "vanish at q = 0" is PR82's leading statement; the rate is. PR69 and PR74 are unmerged and same-provider, so the omission hides an unaccepted candidate, not an accepted result. | Cite PR69/PR74 and state the disagreement on the rate. |
| MD7 | minor | CONFIRMED | Math-:…/NOTE.md:76–78 (also QUARTIC.md:31) | The "(this session)" exact claims come with no script, test or mutant. Math- #58 item 5 requires exact algebra tests and mutants, and PR53/PR69 do ship algebra_check.py. The ring claims are wrong: the expressions are rational in Q. | Ship an exact check. |
| MD8 | minor | CONFIRMED | Math-:…/QUARTIC.md:15 (also :4; NOTE.md:14, :29) | "NOTE §B" points to the NOTE.md in the same directory, which has sections 1–6. The real source is reviews/collision_mechanism_20260925/NOTE.md §B, lines 52–69 (blob 3ee30829), and it is never named. PR53 is unpinned in NOTE; its SHA appears only via #58. | Name and pin the source. |
| MD11 | minor | CONFIRMED | Math-:LEDGER.md:22 | OBL-H5-JETMOD is a predicate, not a carrier, yet it sits in the carrier table.<br>- CHART_SIDE_JETMOD_PLAN.md exists in main history: at 7caac254, under drive/mirrors/2026-09-16 — HOLD_NOT_FOR_SUBMISSION/ (blob 9446daa2, 11,663 B, sha256 0ccadff3…). Its line 95 supports the row.<br>- It was added in c1b3b22, an ancestor of main, and removed by f35eef1. sources-12.md:12, 71 and 130 name it.<br>The prior PR81 review was wrong that the path and "G12-band" exist nowhere; that is true only of the default tree. | Move the predicate out of the table and cite the historical path. |
| MD13 | minor | CONFIRMED | Math-:LEDGER.md:20 (also :5, :13, :15) | Frames allcell_fdz_enclosures.json as a historical file whose bytes were lost. GAP_LEDGER:26 and NODE_ENV_RESCOV:70 (main eeebb28) call it a hunt-coined name. | "Hunt-coined name; no historical carrier." Merged: XC17 (part). |
| MD10 | minor | PLAUSIBLE | Math-:LEDGER.md:11 | Lists PR80 under "Substitute emitted" without saying it is a draft, unmerged, has no review file and lacks its companion script. The line-5 rule is in fact met by existing carriers (PR25 R1 ACCEPT; KERNEL_TAILS lines 179–183 with contact_tools.py and test_contact_tools.py). The row was written before #59 merged, so it is stale rather than wrong. GAP_LEDGER does not list the filename under do-not-substitute. | Qualify the PR80 row. |
| MD12 | minor | PLAUSIBLE | Math-:LEDGER.md:33–34 | The PR53 row gives neither the SHA nor the unmerged status (PROOF_INDEX:44 has both). The microdisk row omits PR69/PR74. Omitting PR82 is not a fault: it came 5 min later and agrees the rate "needs new lemma". | Add pins and PR69/PR74. |
| MD9 | nit | CONFIRMED | Math-:…/NOTE.md:12 | S and A are overloaded. NOTE, QUARTIC and LEDGER carry no author/provider line; only branch names identify the lane, and all commits come from the shared account. No Math- rule requires the line. | Rename; add an author line. |
| MD14 | nit | CONFIRMED | Math-:LEDGER.md:35 | "Bounded by" vs PROOF_INDEX:45's "bounded on opposite sides by". Line 31 gives a bare filename, and line 37's "ANNULUS_BRIDGE" is ambiguous. | Align the wording and give paths. |

**Covered elsewhere (not re-counted):** MD6, a cross-file item: PR #163 turns PR82's leading statement into an exact identity, and cycle 2 does something similar.
- PR #163 part → BT4.
- The math → MD1.
- Cycle-2 :43 → Part 2 X1.

SESSION_LEDGER:81 in PR #163 cites PR82 accurately.

**Not verified:**
- The Drive-side claims in LEDGER.
- The cycle-2 MC numbers (these are handled in §4).
- The Q-axis integrability evidence, which is NON-CERTIFYING: it uses a leading-order contact model, not the periodized regression.
- PR69's bounds (not re-reviewed).
- The texts of the prior PR81/PR82 reviews, and CI on those PRs.

---

## 7. Cross-cutting consistency (XC)

**What was checked.** Every Grok artifact in scope was read at its pinned head, and cross-file and cross-surface claims were checked against the repositories.

**Recomputed and CONFIRMED:**
- No Grok commit touches STATUS.md, PROOF_INDEX.md, LANDING_CLAIMS, or any file outside its own packet directory.
- The STATUS labels as quoted match STATUS lines 11–22.
- The erratum trees match the PR heads: d8f5505 has the tree of PR64 head 1d9b64b (723baf87), and 5ed3b455 the tree of PR85 head ff36667.
- The PR refs match: pr/80 = a79007e, pr/81 = b28cf25, pr/82 = b190a4d, pr/53 = 9a6f8a6.
- The SIDE24 constants c2 = 0.0734069193060342710301… and c3 = 0.0417759318405983433429… lie inside the published ENCLOSURE (NON-CERTIFYING).
- The File-1 labels match Q0 34810–34834.
- SIDE24 passes 30/30 (1.05 s) and P15 passes 36/36 (3.42 s).

**Prior review follow-through:** none of the 15 items in the three prior reviews (#163, #159, #161) was addressed at the pins, except PR #159's item 3, which was mistaken.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| XC12 | minor | CONFIRMED | CLOSED_BF_WTWS.md:1 (title) and the surfaces listed | The lane uses closure and certification words on its own unreviewed results:<br>- A2_RESIDUAL_GAP:17 "Certified this session (CAS)".<br>- ND_14FRAME:6 "Certified this session", covering a float grid check at line 13.<br>- HESSIAN_JET:6 "Already closed last round"; CUBIC_B4's title "(verified)".<br>- Cycle-2 SESSION_LEDGER:10 "CLOSED".<br>- Math- LEDGER:5 and a79007e "verification-grade"; #160 "CLOSED under corrected reading".<br>PR #163's float/MC tables (D5_REPLAY:15, FINITE_R:19–25, HESSIAN_JET:16–21) lack the NON-CERTIFYING token (README:84; CONTRIBUTING:129; the PR template covers bounds only). FINITE_R:31, "What this closes", draws a conclusion from a float table. HESSIAN_JET:12 and FINITE_R:17 do say "numerical", and the identities themselves are correct. | Use "candidate" / "author-side"; add NON-CERTIFYING to every float/MC table. |
| XC16 | minor | CONFIRMED | REPLAY.json:31 (also SESSION_LEDGER.md:84) | "HOLD_WITH_DOMAIN; parent main#63 Eq 15.2 unreviewed": Eq (15.2) was ACCEPTed under D1-E (#63 comment 5841570965; the reopen 5841830743 keeps D1-A–E). The actual HOLD reason (LANDING_CLAIMS side24-coefficient) is that the persistence interpretation depends on the open #63 Theorem A §§2–7 chain, via D1-D p_r → 1. | State the actual HOLD reason. |

**Merged into other sections (not re-counted):**

| XC ID | Counted as | What XC adds |
|---|---|---|
| XC1 | DC6 | — |
| XC2 | PK4, PK3, PK5 | The q0_machine.json "SURVIVES" record, given in PK3. |
| XC3 | PK7 | — |
| XC4 | BT1 | — |
| XC5 | BC8 | — |
| XC6 | CS3, CS10 | The 13×13 / Table 4.1 mislabel, and the fix: cite K4 rather than drop "refuted". |
| XC7 | PK11 | — |
| XC8 | DC4, BT8 | The uncited D2 (R10) bound. |
| XC9 | PK14 | — |
| XC10 | CS1, CK1 | — |
| XC11 | PK15 | See the note below. |
| XC13 | BT14 | — |
| XC14 | PK9 | The public (ND′), against ND_CUBIC_SLAVING:51; see BC4. |
| XC15 | PK12, PK19 | — |
| XC17 | CK8, MD13 | — |
| XC18 | BT9 | — |
| XC19 | LB1, LB4 | Also Part 2 G6. |
| XC20 | HY12, PK2 | Also Part 2 P10. |

**Note on XC11 / PK15.** The cross-cutting unit and its verifier said that a sentence dropped from INVENTORY at 936bf56 ("PR82 vanishing-det claim is false on the written matrix") was itself wrong, because the constrained det H_M ≡ 0. That argument rests on PR82's leading-order solve. Under the exact constraint, det H_M ≠ 0 at r^3 (MD1: three exact passes and a rational certificate), so the dropped sentence was closer to right. PK15 is counted without relying on this point.

**Not verified:**
- The Drive-only identities.
- Whether GitHub records Math- PR64/PR85 as merged. Provenance is Part 2 §1.

---

## 8. Files 1–6: public-mirror verification (FM)

**What Grok's surfaces claim about the stack.**
- In PR #163:
  - The stack is "ABSENT from public GitHub".
  - K4/R3/R3b are "a later instrument registry".
  - The (ND) route is "not a certified kill".
  - A rank-drop count and a 13-frame candidate.
  - "File 3 §6.5", and appendices "never drafted".
  - What "File 2 says".
  - Table 4.1 blocks certification.
  - A sentence attributed to GCJA.
  - The κ convention.
- In cycle 2: a 13×13 "repaired sentence" on 15/4, and a narrowed Theorem-B label.

**What the mirror is.**
- Blob `ed0f51f1` at `drive/mirrors/10_AXIOMATIC_CORE_SPINE/00.2_CANONICAL_MASTER/00_CANONICAL_MASTERS_AND_SOURCE/Q0_MASTER.md` (1,831,987 B, sha256 3112fb61…22cc). It is OCR text with the image layer dropped.
- It is in main default history at 80e7d0d; f35eef1 removed it from the tip on 2026-09-23. It is still live at about 90 branch tips of the public main repo.
- Contents:
  - File 1 (Q0 33505, 34720), File 3 (30733), File 4 (31333), File 5 (32698, 33833), File 6 (33124).
  - GCJA (29812), O1 (30060), S3 (30305), the Flat-Saddle lemma (29666), C006 (34264) and the Gate Framework (35116).
- File 2 was never generated (backstory, Q0 29596).

**Recomputed and CONFIRMED against the mirror:**
- File-4 P3: c_30 = 12ℓ/r^3 + O(r^2) = 2κ, with c_12 and c_03 free.
- File-5 (5.4) and its (1/3)·6^{2/3} prefactor.
- The File-1 Theorem A/B labels, the torus T_L^2, and the kill registry K1–K3.
- File 3's (4.2), (6.1a/b), §7.3 and Remark 7.2, and the absence of Table 4.1.
- The O1 Part B mechanism, the C006 numbers and the GCJA quotes.
- The refutation of (ND) as written holds, and it is the stack's own kill entry K4.
- S3's V5-graded frame (Q0 30322–30326): the 15-frame has rank 13, the 13-frame det is 2488320 z_s^4 > 0 off-axis, and Var(W_s | V5) = 3z_s^4/2. GCJA C4 gives Var Ξ = 6, Var T = 4 and det G12 = 1/6480.

**Prior review follow-through.** Earlier review units treated these files as Drive-only and left their File 3/File 5 checks unverified. This section closes those gaps, and the earlier items should be re-graded against the mirror.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| FM3 | minor | CONFIRMED | ND_CUBIC_SLAVING.md:51 (also FILE3_CUBIC_LEFTOVER.md:55) | "Not a certified kill of the route … Remark 7.2 allows integrable vanishing" misreads File 3's fork.<br>- The Euler relation 3W_0 − z_tW_t − z_sW_s = 0 holds exactly in the r → 0 limit, for any kernel. So Σ2(0) has a null vector at every z, and det Σ2(0) ≡ 0 on D(Z_0).<br>- Remark 7.2 covers only vanishing on a positive-codimension set, and it excludes collapse on an open set.<br>- File 3 uses its 14-frame only on the zero-area Appendix-B curves.<br>- §7.3(iv) says a degenerate finding falsifies the β = 4 route, and S3 §5 records exactly that as K4.<br>The fallback at line 51 (keep (W_t, W_s) off {z_s = 0}) is also singular at every z when H^-_ss is kept. Related: BC5, IS2. | "(ND) as written fails identically in z. This is File 3's open-set-collapse prong (K4). The β = 4 route survives only through (ND′) (GCJA C4)." |
| FM4 | minor (panel 1 major : 2 minor) | CONFIRMED | FILE3_CUBIC_LEFTOVER.md:45 (also :43–49; cycle-2 RESULT.md:12, :35, :49; SESSION_LEDGER.md:10) | The total counts assume a nondegenerate pair block. Those counts are "Listed 15 overcounts by 2", the 13-frame candidate, and cycle 2's "repaired sentence / 13×13".<br>- File 3's displayed ladder (4.2) / Lemma 4.1 does not give such a block. Its 12 pair rows converge to only 9 distinct functionals: G_t^- and H^+_tt → f_tt; G_s^- and H^+_ts → f_ts; V^-_corr and H^-_tt → f_ttt.<br>- So the 15-frame limit has rank 10, and the 13-frame (W_0 and W_t dropped) also has rank 10, with determinant identically 0.<br>- File 3's "14 functionals, orders 0..3, pairwise distinct" is impossible, since only 10 such monomials exist.<br>- A nondegenerate 13-frame needs a pair frame that contains f_tss but not f_sss, such as S3's V5 frame.<br>The finding was anchored at line 23, but line 23 ("Rank drop 2 at cubic order") is correct as the W-block's own contribution. Related: BC8. | State the frame. Replace the "13×13 certificate" with Var(W_s \| V5) = 3z_s^4/2 > 0 off-axis. |
| FM9 | minor | PLAUSIBLE | ND_CUBIC_SLAVING.md:45 | Rereads File 3's Lemma 6.1 instead of flagging it.<br>- Lemma 6.1 says W_0(0) is a cubic-plus-quartic z-polynomial that contains all fourth-order directions at every z, with W_i(0) its z-gradient.<br>- File 3 never ties that quartic content to an axis 14-frame. Its 14-frame drops W_0 on the exceptional curves.<br>- Grok's scaling point is right: at the (6.1a/b) normalizations the quartic content is O(r^2) → 0.<br>So Lemma 6.1 as printed is inconsistent with (6.1a/b). The same inconsistency recurs in Lemma 6.1's covariance clause, Remark 6.2, the Lemma 6.5 proof, §7.3(iii) and the (H1++) rationale. On the axis W_t(0) and W_s(0) also vanish, so the 14-frame is still degenerate there. BC9's O(r) concerns a different object (the pair-conditioned Schur remainder). | Flag the File 3 inconsistency; an axis stratum needs a new normalization. |
| FM13 | minor | CONFIRMED | CLAUDE_FILES_1_6_MAP.md:28 (also :56; PAIRING_IMPLICATION_GAP.md:30–32) | "File 2: a.s. Morse is literature; Morse–Smale is OPEN" is the content of the R0 literature-verification sweep (Q0 32178–32473), not of a File 2 manuscript. File 2 was never generated (Q0 29596).<br>- File 5 calls R0 "discharged by File 2" (32725–32726, 32968, 33097), yet also lists "Sublemma R0 (File 2)" as an input (32714–32715).<br>- The shorthand comes from the stack itself: File 3 calls the sweep the "File 2 sweep" (30747, 31361, 31425).<br>- PAIRING_IMPLICATION_GAP:5 is worded correctly. | Cite the R0 sweep, and note that File 5's "discharged" rests on a file that was never generated. |
| FM16 | minor | CONFIRMED | ND_14FRAME_AND_GCJA_C2.md:46 (also ND_CUBIC_SLAVING.md:61; cycle-2 RESULT.md:35, :49; Math- PR87 REDUCED_FRAME_4SLOT.md:60) | "Without Table 4.1 the pair 12-block map 10-jet → Ξ_1(0) is not unique, so a numerical 14×14 cannot be certified":<br>- The pair block's limit spans V5 = {t^m s^k : m + 2k ≤ 5}. That is 12 monomials, including f_tttt, f_ttttt and f_ttts but not f_sss, so it is not a map from the 10-jet.<br>- The positivity, rank and strata of Σ2(0) do not depend on the frame (S3 Step A). File 3 itself calls Table 4.1 "pure bookkeeping".<br>Qualification: without Table 4.1, File 3's own matrix entries cannot be pinned down. S3's "certified" is itself an author-side label. | Cite S3 Step A and V5, and keep "Table 4.1 ABSENT" as a custody fact only. |
| FM17 | minor | CONFIRMED | ND_14FRAME_AND_GCJA_C2.md:42 | Credits GCJA with "the leading form Σ_∞(z) is not the zero matrix", which GCJA does not say. GCJA's (ND′) (C4) is G_{U\|V} ≻ 0, with a strictly positive gradient-block diagonal off {z_s = 0}.<br>- "Not the zero matrix" is too weak for a KR density bound. It even admits GCJA C2's rank-1 form, whose det is ≡ 0, which the note itself discusses at line 30.<br>- The Remark 7.2 part comes from File 3, not GCJA, and it is unavailable here (IS2).<br>Nit is defensible, since the file says "Scientific effect: NONE". | Quote C4 directly. |
| FM10 | nit | CONFIRMED | FILE3_CUBIC_LEFTOVER.md:51 (also CLOSED_BF_WTWS.md:41, BF_WTWS_SCHUR.md:34) | "Appendix A/B were never drafted (File 3 p.18)": the carrier (Q0 30733–31328) shows only that the appendices are absent and deferred (31320–31326). Elsewhere File 3 contradicts itself: it says they "ship with it" (31317–31319), that Table 4.1 is "included in … appendix A verbatim" (30949–30950), and that Appendix A was "verified line by line" (31122, 31130–31131). The OCR mirror has no page numbers, so "p.18" cannot be checked. | "Not in the File 3 carrier (deferred; Q0 31320–31326)." |
| FM18 | nit | CONFIRMED | GV_SPECTRUM_AND_P3.md:23 | "c_30 = 12ℓ/r^3 + O(r^2) = 2κ under File-1 ℓ = κr^3/6" is correct in the File 1 §5.1 / File 4 V4.1 convention κ = c_30/2. But File 1 (0.1) defines κ = \|f_vvv(x_0)\| = \|c_30\|, under which c_30 = κ, so the clash is inside File 1.<br>- File 4 V4.1 repeats the literal definition (Q0 31667–31668) before deriving c_30 = 2κ.<br>- File 5 (1.3) adopts κ = \|c_30\|/2 (32732–32733).<br>- A2_RESIDUAL_GAP:34 already implies κ = c_30/2. | Name the convention. |

**Covered elsewhere (not re-counted):**
- FM1 (major) → PK4.
- FM2 (major) → BC5 (the (ND) status) and PK9 (the label).
- FM5 → BC2 and Part 2 X2. Cycle 2 closes E-ND-2 on 15/4; Math- PR87 REDUCED_FRAME_4SLOT:7–28 later amends it to 3/2.
- FM6 → PK7. FM7 → PK18. FM8 → BC15.
- FM11 → BT10 and CS3 (also Part 2 R4).
- FM12 → PK17.
- FM14 → BT9.
- FM15 → PK12.

**Not verified:**
- Byte identity between Grok's "user-attached PDFs" and the mirror sources. The mirror records only 12-hex original-sha256 prefixes (e.g. File 3 76c9e98e95c1, C006 9989196b1d24).
- Page references.
- Whether Appendix A/B were ever drafted off GitHub.
- The content of File 2.
- OCR fidelity of displayed equations: some File 4 display lines, e.g. 31455 and 31462, lose operators.
- How GitHub code search indexes.

---

## 9. main issue #165 and the axis singularity (IS)

**What issue #165 claims.** The body was created 2026-09-26T23:31:05Z and has never been edited; it has no comments and no linked PR.
- Under a "six pins" header: det Gram = 45/16 z_s^8, and W_s/z_s^2 has "unconditional variance 15/4".
- "On-axis the whole cubic W-block must be dropped or the KR prefactor log-diverges".
- Theorem B is "PROVEN-MODULO the repaired ND sentence".
- O(r^3 q^2) against #58.
- Under "Weighted vs unweighted rates (teammate measures, corroboration only)": "weighted (I r^4)/E[W] = κ r^2 with κ(c=0.75, b=0) ≈ 1.95", "unweighted … O(r^3) or O(r^3 log(1/η))", and "Both can be true".
- Seven packet files are "prepared", and "branch creation raced with another agent".

**Recomputed and CONFIRMED:**
- The four pinned sha256 values in #165 match.
- The Math-#58 comment 5850921404 exists (23:30:52Z, never edited). An earlier premise that it was missing was a summarizer artifact; the raw page has 3 IssueComment nodes.
- Exact sympy, planar BF, given G+V, with W evaluated at 0:
  - The W_s density given f_tss is (√3/(3√π)) z_s^-2.
  - The joint (W_t, W_s) density is √3/(3π) z_s^-4, or 2√5/(15π) z_s^-4 unconditionally.
  - The log-type profile 1/(√π\|z_s\|√(3z_s^2 + 4z_t^2)) appears only when the f_t(y) = 0 constraint is dropped and f_tss is averaged. It is exactly the average of the z_s^-2 density over f_tss ~ N(0, 2), and over a disk it diverges like log^2.
- The finite-r Mahalanobis distance of ∇f(y) = 0 tends to 72k^2 z_t^2/z_s^2 as r → 0. At the axis it levels off at O(k^2/r^2): 2.07e4 at r = 0.1 and 8.47e4 at r = 0.05 (NON-CERTIFYING). So the true Palm integrand carries exp(−36k^2 z_t^2/z_s^2) and is exponentially suppressed near the axis.

**Prior review follow-through:** the issue has never been amended. Nothing in it reflects the team's later correction in Math- PR87 (f3af9b5, 23:55:01Z).

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| IS1 | **major** | CONFIRMED | main:incoming/grok-cycle2-20260926/RESULT.md:35 (also SESSION_LEDGER.md:10; #165 body line 11) | "On-axis the whole cubic W-block must be dropped; keeping raw W_s produces a logarithmic KR singularity" is wrong on the exponent and on the cure.<br>- In every KR-consistent frame the prefactor is a power law: \|z_s\|^-2 for W_s with f_tss slaved (the packet's own frame), and \|z_s\|^-4 for the joint (W_t, W_s).<br>- The constraint δ(W_t) = (2/z_s^2) δ(f_tss) brings the total back to z_s^-4, and rescaling to W_s/z_s^2 does not help.<br>- A log profile arises only for the marginal of W_s with the f_t(y) = 0 constraint dropped, which is not a KR prefactor.<br>- Both power laws are non-integrable over D(Z_0), which contains the whole axis segment. Dropping the block on the axis removes only a null set.<br>So E-ND-2's "CLOSED as repaired sentence" is unsupported. | "At leading cubic order the reduced prefactor is ∝ \|z_s\|^-2 (joint ∝ \|z_s\|^-4), non-integrable over D(Z_0); no repaired-ND sentence of this form restores File-3 (7.2) or Remark 7.2." Compute the inner-zone integrand at Palm values. Merged: CS7. |
| IS2 | **major** (panel 2 major : 1 minor) | CONFIRMED | ND_14FRAME_AND_GCJA_C2.md:42 (also ND_CUBIC_SLAVING.md:51, FILE3_CUBIC_LEFTOVER.md:55; PR #163) | The appeal to "an integrable singularity on a curve if needed (File-3 Remark 7.2)" is inconsistent with the file's own line 38 (Var ≍ z_s^4). Every reading gives a non-integrable bound:<br>- The reduced 1-D bound (2π)^{-1/2} Var^{-1/2} is ∝ \|z_s\|^-2, with Var = (3/2) z_s^4 under the pins or (15/4) z_s^4 unconditionally. It is not integrable over D(Z_0) and diverges along the whole segment z_s = 0 (like 2Z_0/ε).<br>- The 2×2 reading (Hessians outside, det (3/4) z_s^8) gives \|z_s\|^-4.<br>- The finite-r block (sd_t ≈ (z_s^2/2) r^5) gives \|z_s\|^-4 r^-9.<br>Remark 7.2's proviso "PROVIDED the singularity is integrable over D(Z_0)" therefore fails. The literal 15-frame is the separate open-set-collapse prong (K4; FM3). ND_CUBIC_SLAVING:51's restatement on (W_t, W_s) contradicts ND_14FRAME:20–23, where W_t → 0 after the H^- pin and the rank is ≤ 1. File 3 §7.5 (Q0 31247) writes (2π)^{-3/2} det Cov(W \| pair)^{-1/2} for the 3-coordinate block; the 1-D form is its specialization. | "The reduced leading form degenerates like z_s^4 on the fold axis; the density bound is not integrable over D(Z_0), so Remark 7.2 does not apply." Mark ND_CUBIC_SLAVING:51 superseded. Merged: BC10 (minor on the PR #163 panel; this is the IS unit's primary question). |
| IS3 | **major** | CONFIRMED | main issue #165 body:25 (also :24, :26) | "Weighted (I r^4)/E[W] = κ r^2 with κ(c=0.75, b=0) ≈ 1.95" and "Both can be true" fail under the program's weight W_r = F_2(H_M)F_1(H_S), E[W] = Z_r (Math- thin-tube PROOF.md (T18)/(T19)). All numbers are NON-CERTIFYING:<br>- The weighted count over the scaled transverse cone \|p\| ≤ \|q\| ≤ 0.75 in X = M + r(p,q) is ≈ 0.23 r^3, for r in 0.025–0.4.<br>- Index-free counts are also r^3 (≈ 0.41 r^3, normalized by Z).<br>- The plain E_Q count is ≈ 0.43 r.<br>- There is no η-dependence: the integrand is ∝ q as q → 0.<br>- "Both can be true" fails: the weighted integrand is pointwise ≤ the index-free one, and 0.41 r^3 < 1.95 r^2 for r < 4.7.<br>One verifier (the minority, rated minor) reproduced κ ≈ 1.95–1.97 with an r^2 law under a different reading: a line integral along the on-axis slice p = 0, \|q\| ≤ 0.75, summing the M- and S-rays. There κ depends on c and b (≈ 1.15 at (0.5, 0), ≈ 0.58 at (0.75, 1)). So the constant may describe that 1-D slice, but it is not a transverse-cone rate, and I, c and η are undefined either way. The r^2 rate matches the formal assembly in Math- PR87 D5_OBSTRUCTION_LEDGER §6 (Part 2 D3). | Withdraw the line, or define I, W, c and η and publish the script; state the cone count as O(r^3). |
| IS4 | minor | CONFIRMED | main issue #165 body:23 | "Teammate measures, corroboration only": κ ≈ 1.95, "I r^4", "E[W]", "O(r^3 log(1/η))" and the η fence appear in no committed file on any ref of main or Math-, and I, c, κ and η are never defined. W and E[W] can be partly decoded from Math- and from PR #163's D5_MISSING_INEQUALITY (D5-μ), and an unweighted O(r^3) cone count appears in a Math-#58 comment. The meaning of κ, I, c and η cannot be recovered. | Commit the script and output with definitions, or remove the line. |

**Covered elsewhere (not re-counted):**
- **IS5 → Part 2 G5.** Five of the seven listed files exist nowhere; no PR is paired; and the #58 comment points to a packet that never landed.
  - Timing correction from server-side events: the branch was created at 23:29:53Z, before the #58 comment (23:30:52Z) and the issue (23:31:05Z). The two-file commit was pushed at 23:31:29Z.
  - So the "raced" wording is unsupported and stale, but it is not disproved.
- **IS6 → Part 2 X1** (the product power), **X2** (15/4), and **CS3** (with Part 2 R4) for "PROVEN-MODULO". Refinements:
  - 45/16 is the unconditional value at z_t = 0. The six-pin determinant depends on r and is not a monomial for z_t ≠ 0 (1.6186 z_s^8 at r = 0.3, z = (0.5, 1)).
  - Harper's O(k^3 r^5 q_s^2) is the same power as Grok's r^3 q_phys^2 after converting charts, not a third power.
  - The leading constant is 648/√π ≈ 365.6.
- **IS7 → Part 2 G2 / §4.1**, the empty branch `custody/status-erratum-boundary-bullet-20260926`.
  - The proposed bullet does exist, in PR #163's STATUS_PIN_NOTE.md (aa293368, 21:58:31Z).
  - Math- has a twin empty branch, `custody/proof-index-erratum-pointer-20260926`.

**Not verified:**
- The definitions behind κ.
- Whether the five missing files exist off GitHub.
- Who created the custody branch.
- File 3 Appendix A/B (the exact Π_pinned stencils).
- The finite-r suppression, which was checked only at b = 0, k = 1, r ∈ {0.1, 0.05}.

---

## 10. Math- PR #87 reading maps and SARD-G half; Math- 5ed3b45 default files (RM)

**Scope.**
- In Math- PR #87 (`0fab933`), under `incoming/grok-cycle4-20260926/`: harper/SARD_G_A1_RELATIVE_INTERIOR_LEMMA.md, lucas/SARD_G_A1_APPLIED_TO_SUCCESSOR.md and PUBLIC_READING_MAP.md.
- PR #88's PUBLIC_READING_MAP.md (`2cdf62f`).
- The Math- default files from 5ed3b45: imports/lifetime_parent_20260925/CAP_PAIRING_IDENTITIES.md, frontiers/full_price_20260924/ASCII_3E_MINUS_2.md and ERRATUM_POINTER.md.

Part 2 §3 (S, L, K) and §2 (C) already cover most of this. Only non-duplicates are counted here.

**Verdict:**
- AMEND for the SARD-G lemma, the lucas scores and both maps.
- The 5ed3b45 default files are mathematically correct; there are minor custody notes only.

**Recomputed and CONFIRMED:**
- The witness F in the written U_χ fails RI1 at (1, 0), and F_ε misses Σ for 0 < ε < 1/200.
- The exact critical points, with Hessians a·diag(1,−1), a·diag(−1,1), a·diag(−3,−1), a·diag(3,1) and values −1, 1, 3, −3.
- The minimum arc speed 2π sin(π/25) ≈ 0.7875 > 12/25, and the travel bound 23/48.
- lucas's P1–P3 FAIL and P4/P5 PARTIAL are defensible if P_i is read as item i.
- The path at a1fc9581 is exact.
- All PR numbers and branches match the refs: #53, #60, #69, #80–82, #87 @0fab933, #88 @2cdf62f.
- All 10 account repos are Public (checked 2026-09-27).
- PR87's three-dot diff is 9 files under incoming/, and PR88's status lines match STATUS lines 11–22.
- Tests and gates:
  - test_closed_forms is stdlib-only and passes 7/7, also under -O.
  - A local replay of the downstream gate passes on both heads (67 tests, 24 mutations).
  - run_validation passes 36 tests with 7 mutants.
  - verify.py passes, but it checks only 2 MANIFEST files.
- The CAP_PAIRING identities are exact: 2r^2; ‖w′‖ ≤ 2mr (sharp); 15/8; the h″ cancellation for vector h; 24/121; 2184415/7086244 > 1/4; (7.3); O(r^3).
- The ASCII identities check: 5 − log 3 with ρ ≈ 0.25632; ρ* = 0.84547981724898672067874…; e^2 − (3e − 2) = (e − 1)(e − 2).
- ERRATUM_POINTER is consistent.

| ID | Severity | Status | File:line | Problem | Fix |
|---|---|---|---|---|---|
| RM3 | minor | CONFIRMED | Math-:incoming/grok-cycle4-20260926/harper/SARD_G_A1_RELATIVE_INTERIOR_LEMMA.md:81 | The checklist "can close A1" covers openness only, but A1 is openness, countability and coverage (review line 11; source line 140). The coverage re-check for the strengthened predicate is missing (source §3 line 49; review line 169 argues that coverage holds). Countability is trivially unaffected. | Add the coverage step. |
| RM16 | minor | CONFIRMED | Math-:…/lucas/SARD_G_A1_APPLIED_TO_SUCCESSOR.md:3 (and all nine packet files) | No author/provider line; the only identifier is the path token "grok", and commits come from the shared account. The lucas file scores another lane's source like a review, but gives no reviewer identity, source exposure or zero-independence statement (cf. successor review line 33; governance- REVIEW_TOPOLOGY.md:26 requires one). The "Lucas independent" wording is covered by Part 2 A1. | Add the identity and a zero-credit line. |
| RM12 | minor | PLAUSIBLE | Math- PR #88 PUBLIC_READING_MAP.md:67 | Says "publication of pointers only", but carries unquoted self-label descriptors ("Exact", O(r^3 q^2), O(k^3 r^5 q^2)), against PROOF_INDEX:13's "Source self-labels are quoted, not adopted". The values do match the files, and firewall lines 24 and 64 exist.<br>- All 8 packet links use the mutable branch name (compare AGENTS:8; the PROOF_INDEX:3, :44 precedent).<br>- Lines 65–66 restate status (they match STATUS).<br>- It is a second map (STATUS:36 "two human maps"), and its heading is false once PR87 merges.<br>Related: Part 2 K4, K6. | SHA-pin the links, and quote or drop the descriptors. |
| RM14 | minor | PLAUSIBLE | Math- PR #87 PUBLIC_READING_MAP.md:7 (also :9, :23, :33) | The premise "packet lives on a branch" is false after a merge, and there are two overlapping maps committed 10 s apart (db66b25, 2cdf62f). Line 18's "P1–P6 repair text" is the wrong descriptor, since the lemma uses RI1–RI4. Using @8657130 as a content pin is fine. Related: Part 2 L1, K1. | Fix the descriptor and merge the maps. |
| RM19 | nit | CONFIRMED | Math-:imports/lifetime_parent_20260925/CAP_PAIRING_IDENTITIES.md:28 (default) | The derivation omits the Df[v″] = ∇_y f · h‴ term, which vanishes on the ridge. The conclusion F″ = D^3 f[v,v,v] is correct (sympy, generic quartic, vector h). | Add the vanishing term. |
| RM10 | nit | PLAUSIBLE | Math-:…/lucas/SARD_G_A1_APPLIED_TO_SUCCESSOR.md:10 | "A6 conditionally valid once charts are open" drops D_χ ∈ C^1 (A2), dD_χ[h_j] ≠ 0 (A3–A4) and the A5 decomposition (review lines 177, 12, 14). STATUS:22 uses the same open-chart framing, and "conditionally" does signal that conditions exist. | List the conditions. |
| RM22 | nit | PLAUSIBLE | Math- PR #88 PUBLIC_READING_MAP.md:0 | No author/provider self-ID. Grok authorship is plausible: timestamps 19:45:10 / 19:45:20 / 19:45:25 (−05:00), the same pin, and the "scientific effect NONE" formula; the PR body names Grok in the third person. Related: Part 2 K4. | Add an author line. |

**Covered elsewhere (dropped duplicates, with Part 2 IDs):**
- RM1 → S1 (Lemma OPEN stated without the A2 conditions). RM1 also claims that A2 was ACCEPTed by a Cursor-hosted review (main PR #122 comment 5841682597) and that later reviews keep that disposition, which would sit uneasily with S1's framing. That claim was not verified here.
- RM2 → S1 (RI2 is an infimum, not "strict"; C^1 → C^2; η).
- RM4 → S4 (the restatement is credited only as "source of the AMEND"; 5-gram overlap 1.7–8%, i.e. paraphrase; the Part 1 panel rated it minor).
- RM5 → S3. RM6 → S2 + S4. RM7 → L2.
- RM8 → L1 (adds that PR #88 line 38 also uses P1–P6). RM9 → L1.
- RM11 → K1/K6 (adds the mutable branch-name links in PR #88 lines 32–38 and 40, and PR #87 line 23).
- RM13 → K6.
- RM15 → K2 + K3 (3 of 28 default package directories are already unindexed; test_closed_forms is the only unreferenced test_*.py).
- RM17 → S4.
- RM18 → Part 2 C2.
- RM20 → C4 + C5 + C9 + P3 (README stale since d8f5505; PROOF_INDEX qualifiers inconsistent).
- RM21 → C1 + C4 + C6 (ERRATUM_POINTER is referenced from nowhere).

**Not verified:**
- A2 itself, the parametric local invariant-manifold theorem for C^2 fields on T_L^2. It is standard, but the program has no reviewed source for it.
- The origin of the P1–P6 labels (the cycle-2 predicate file does not exist).
- Repository visibility at commit time.
- Hosted CI on PR87/PR88; a local replay was run instead.
- The GitHub Pages site.

---

## 11. Units that duplicate Parts 2 and 3 (not separately verified)

Three critic-gap units reviewed objects that Parts 2 and 3 had already verified:
- `gap-math-pr87-cycle4-d5-bf-numerics`: the Math- PR #87 cycle-4 closed forms, reduced frame, alpha series and D5 ledger. Part 2 §3 covers them.
- `gap-grok47-cursor-author-math`: Cursor-hosted lane author notes (Math- PR53 and PR69, main PR #128) and cross-surface D5 rates. Part 3 U9/U11 and Part 1 §6 cover them.
- `gap-grok47-cursor-reviews-incl-landed-p15`: Cursor-hosted review records (the landed P15 review, PR55, PR74, PR52, PR73, PR71, PR68, PR46). Part 3 U7/U8/U9/U10 cover them.

Their findings were not separately verified in Part 1, and none is counted. The items below raise issues that are not covered in Part 2 or Part 3. Each is an **UNVERIFIED lead**, not a result.

**From `gap-math-pr87-cycle4-d5-bf-numerics`:**
- UNVERIFIED lead: Math- PR87 benjamin/REDUCED_FRAME_4SLOT.md:49. The on-axis minimum eigenvalue is given with no r and understated as "can be 1e-5". The unit reports 9.0e-7 at r = 0.3 and 3.0e-8 at r = 0.17, scaling like r^6, so positivity holds at each r but not uniformly (NON-CERTIFYING). Part 2 R2 is a different claim, at line 47.
- UNVERIFIED lead: Math- PR87 harper/D5_OBSTRUCTION_LEDGER.md:115. The ledger cites none of the surfaces that contradict its §4: PR #163 HESSIAN_JET_SIX_PINS:27–29 (Corr = e^{−r^2/2}), Math- PR82 NOTE.md:48 (constrained det H_S = O(r^2)) and Math- PR69. Part 2 D1/D2 cover the mathematics, not the missing citations.
- UNVERIFIED lead: Math- PR87 benjamin/REDUCED_FRAME_4SLOT.md:7. The 15/4 → 3/2 "amendment" names no file, line or commit that it amends, and the files it would amend were never edited. Part 2 R1/R4 cover the values and the status.
- Its F9, the Var(f_tt) table at PR #163 HESSIAN_JET:20, is verified in Part 1 as BT5.

**From `gap-grok47-cursor-author-math`:**
- UNVERIFIED lead: main PR #128 notes/intermediate_scale_20260926/PROOF.md:8. Integrated over r/δ0 ≤ \|x\| ≤ s0, the s^-6 envelope gives an expected-count bound (π/2)Cδ0^4 k/r, which diverges as r → 0. At the seam \|x\| = Br it exceeds the reviewed annulus bound by r^-4. So even if proved, it would not bridge the annulus and the remote window.
- UNVERIFIED lead: PR #128 PROOF.md:185. A ρ-independent C k r^3 target on all of 0 < r ≤ δ0\|x\| would force the fixed-annulus intensity to be O(r^3). Planar BF numerics at fixed B = 4 suggest an r-exponent of about 1.35 (NON-CERTIFYING).
- UNVERIFIED lead: PR #128 test_intermediate_scale.py:92. At radius 0.05 both floors are below the additive slack 1e-18, so the guard cannot fail.
- UNVERIFIED lead: PR #128 PROOF.md:179. The "counterexample" to a uniform floor is a routine consequence of the pinned gradient (λ_max(Σ) ≤ Cs^2), and neither import claims a uniform floor.
- UNVERIFIED lead: PR #128 PROOF.md:10. Both imported reviews (the annulus REVIEW.md and the #76 acceptance comment) are same-provider as the note's author, which is not disclosed. Part 3 U11-F8 covers the author/reviewer label, not this.
- UNVERIFIED lead: PR #128 docs/RESEARCH_INDEX.md:38. The navigation text presents the majorant without an author-side, unreviewed or AMEND label.
- UNVERIFIED lead: PR #128 notes/intermediate_scale_20260926/README.md:7. No log supports the HTTP 403 delivery claim, and same-night lane commits did reach Math- branches. The notes/ delivery path is outside the Math- proof home.
- UNVERIFIED lead: Math- PR69 reviews/d5_pin_microdisk_20260926/NOTE.md:72. The Gram factorization r^10 Φ is only the leading order in rα of the full degree-4-jet determinant; the lower bound survives for \|rα\| ≤ 1/4.
- UNVERIFIED lead: the Math- PR69 and PR53 heads. Neither contains Math- main's renamed protected job "math-downstream-gates" (95c733e), so neither can report the required context without a rebase.
- UNVERIFIED lead: Math- PR53 reviews/pin_neighborhood_recon_20260926/NOTE.md:96. With the T-noise included, the strip inequality is false as written: the ratio is minimized at p = −1/4, where it equals 518400/5409 ≈ 95.84. The exp(−c/r^2) conclusion survives.
- UNVERIFIED lead: Math- PR53 algebra_check.py:180. The transverse minor is a literal, not derived, and the bare asserts pass vacuously under python -O. Part 3 U9-F4 covers lines 201 and 203.

**From `gap-grok47-cursor-reviews-incl-landed-p15`:**
- UNVERIFIED lead: Math- PR74 reviews/pr69_pin_microdisk_nonauthor_20260926/REVIEW.md:163. The outer-cone and steep-strip ACCEPTs rely on PR53's cone Jacobian and covariance floor. The review's input list omits these, and it never cites PR55 (4e188e25), the only review record that ACCEPTs that interface.
- UNVERIFIED lead: PR74 algebra_check.py:4. It says "certify" for an 80-point, 4-r spot check of a polynomial identity of degree up to 20. contact() hard-codes the note's T and σ, so "the witness equations solve to the note's formulae" is untested (both identities are in fact true). Part 3 U9-F7 covers script independence.
- UNVERIFIED lead: PR74 REVIEW.md:147. The first q-derivative vanishes on the whole hypersurface D = 24kt^3 − 3Ct, not only at t = C = D = 0. The upper bound is unaffected.
- UNVERIFIED lead: Math- PR71 reviews/pr124_c1_c2_nonauthor_20260926/REVIEW.md:10–11. The pinned tip a380dcfb and repair parent a14aa4f2 are gone from every ref after a rebase. The equivalent is c688df7, with the same blobs; PR #124 is now at 079615e.
- UNVERIFIED lead: PR71 REVIEW.md:33. A "Nonauthor re-review" that does not say whether the provider of PR124's repair author is known or different.
- UNVERIFIED lead: Math- PR68 reviews/d1_section9_borel_repair_20260925/SUPERSESSION.md:62. The SUPERSEDED_NONBLOCKING disposition conflicts with Math- PROOF_INDEX:33 at 55a3ced, which now indexes REPAIR.md as an active amendment. PR68's base lacks the objects it adjudicates.
- UNVERIFIED lead: Math- PR46 open-lemmas/L-THIN-TUBE-REVIEW.md:11. The thin-tube pin moved from 760340e (on Math- main) to 0a9f508 (PR branch only); the blob, 338d92d1, is the same.
- UNVERIFIED lead: Math- PR46 reviews/map_audit_20260926/AUDIT.md:11. The Grok row has no provider/version line and a relative auditor link to a nonexistent path. It sits under line 3's "independent map audits", although all six commits share one author.

**Items in these units that Parts 1–3 already cover:**
- Reviews unit:
  - F1, F3 = Part 3 U7-F2, U7-F3.
  - F2 = Part 3 U7 (PROOF_INDEX:28 vs 8101de4).
  - F7 = Part 3 U8-F4 / U9-F4.
  - F12 = U8-F10.
  - F13 = U8-F9 + U10-F5.
- Author-math unit:
  - F6 = U11-F8; F9 = U9-F9; F10 = U9-F8; F14 = U9-F4; F15 = U8-F5.
  - F17–F20 = §6 MD1, MD2, MD4, MD5 and §1 BT4.
  - F21 = Part 3 U9-F3 / Part 2 D1.
  - F22 = Part 2 X1.

---

## 12. Totals

Counts are at the pinned heads; post-pin resolutions are listed separately below the table.

| Section | Surface | Blocker | Major | Minor | Nit | Total | of which PLAUSIBLE |
|---|---|---|---|---|---|---|---|
| 1.1 DC | PR #163 D1 chain | 0 | 0 | 11 | 4 | 15 | 3 |
| 1.2 BC | PR #163 BF cubic / ND | 0 | 1 | 10 | 1 | 12 | 3 |
| 1.3 BT | PR #163 BF transverse / D5 | 0 | 1 | 7 | 6 | 14 | 1 |
| 1.4 PK | PR #163 custody and packaging | 1 | 2 | 12 | 5 | 20 | 5 |
| **1** | **PR #163 subtotal** | **1** | **4** | **40** | **16** | **61** | **12** |
| 2 HY | PR #159 and branch hygiene | 1 | 0 | 5 | 5 | 11 | 3 |
| 3 LB | PR #161 | 0 | 2 | 10 | 6 | 18 | 3 |
| 4.1 CN | cycle-2 D5 numerics | 0 | 0 | 3 | 1 | 4 | 3 |
| 4.2 CS | cycle-2 ND / SARD / packaging | 1 | 1 | 4 | 3 | 9 | 5 |
| 5 CK | Math- PR #80 | 0 | 0 | 6 | 5 | 11 | 7 |
| 6 MD | Math- PRs #81/#82 | 1 | 4 | 6 | 2 | 13 | 3 |
| 7 XC | cross-cutting (unique only) | 0 | 0 | 2 | 0 | 2 | 0 |
| 8 FM | Files 1–6 mirror (unique only) | 0 | 0 | 6 | 2 | 8 | 1 |
| 9 IS | issue #165 and axis singularity | 0 | 3 | 1 | 0 | 4 | 0 |
| 10 RM | PR #87 maps / default files (after dedup) | 0 | 0 | 4 | 3 | 7 | 4 |
| **All** | | **4** | **14** | **87** | **43** | **148** | **41** |

**Blockers:**
- PK1 and HY1: intake. Both resolved post-pin.
- CS1: intake, latent (no PR).
- MD2: QUARTIC λ formula. Landed post-pin.

**Majors:** BC5, BT1, PK3, PK4, LB1, LB3, CS3, MD1, MD3, MD4, MD5, IS1, IS2, IS3.

**Post-pin status** (§0.2; not re-counted):
- Resolved: PK1, HY1, HY10, LB1, LB2, LB4, LB5, LB6, LB15.
- Realized: LB3 (#116 auto-closed).
- Landed on Math- default: MD1–MD4, MD7–MD9 (PR #82 merge; bytes not re-hashed).
- Persisting in landed immutable intake on main: all other §1 findings, and HY2–HY5, HY7, HY11, LB9, LB12, LB17.

**Overlaps that reduce the number of distinct root causes:**
- The "ABSENT from public GitHub" premise (PK3, PK4, PK5) feeds PK12 and BC7.
- The mislabelled K4 (PK9) keeps (ND) presented as OPEN (BC5), which FM3 and CS6 address from the File 3 and cycle-2 sides.
- The leading-order solve behind MD1 also drives BT4 and MD3.
- IS1 and IS2 share the non-integrable axis density that undercuts CS3 and CS10.
