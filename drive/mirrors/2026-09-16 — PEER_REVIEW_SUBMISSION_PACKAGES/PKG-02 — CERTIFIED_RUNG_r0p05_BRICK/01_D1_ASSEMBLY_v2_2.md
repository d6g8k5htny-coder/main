# D1_ASSEMBLY v2.2 — Stage-E round-2 repair re-issuance (final conditional theorem)

**Artifact:** D1-ASM-20260915-v2.2
**Campaign:** U2D-UPPER, Stage D→E (assembly; round-2 repair re-issuance).
**Agent:** D1 (assembly).
**Historical (preserved untouched):** v1.0 body 006b8a7d…; v1.1 body 634338b4…;
v1.2 body a7e1958c…; v2.0 body 86882dca… (gate d1_falsify.py frozen);
**v2.1 body bcc177fea4ee06f9d25c56f207ed4b6e7577eff96bfc030202928813c0688189
(gate d1_falsify_v2.py frozen)**. This v2.2 is the current strongest form and
supersedes all prior status lines. The round-2 carrier was hash-verified FROM
BYTES before consumption.

BEGIN_FROZEN_BODY

## 0. Round-2 carrier ledger (verified at consume time)

| repair | artifact | pin | verify |
|---|---|---|---|
| V1 repair (κ at the floor) | D3_percolation/D3_REMOTE_AMENDMENT_v2.md | body 6796deea4bfd5e3df19fd561e6bc34efe2531263db9db8d015d0553c3a3d1302 | MATCH (raw marker segment, 4,553 B) |
| (historical, superseded) | D3_percolation/D3_REMOTE_AMENDMENT.md | body 63d91cdd6364725597504067a8ffa2378da0e89fe0e87afc81828dd78b8fec24 | re-verified, historical |
| (all other carriers) | the v2.1 ledger (D1_ASSEMBLY_v2_1.md §0): B1/C1/C2/D2/D3/H3/H4/H5 closures; R1 H4JC-R1 42ee88da…; R2 rung floor 6347275d…; R4 freeze/tightening/errata/v3 totals; DIR 8821c8fa… | unchanged | re-verified by gate v3 |

Re-review dispositions consumed: H4JC-R1 PASS-WITHSTOOD (counterexample
instance); numerics PASS for the v2.1 gate/chain (two BRANCH_dir receipts
defects — owner's errata in flight, non-blocking; the κ-probe sensitivity
flag, which V1 repairs at source); scope FAIL with two violations V1 (κ_far
at the QMC probe assembles 0.6773 > 0.66 at the certified floor; the G.7
consumption dropped from the register; "zero MC" false) and V2 (B4.loc's dam
line must be a validity premise). V1 is repaired at source by the v2
amendment above; V2 is repaired in §2's register.

## 1. THEOREM D1 v2.2

Setting and partition as frozen (v2.0 §0): on Reg ∩ TYP,
1 − q(r, 6/5) = P_r(A) + P_r(B1) + P_r(B2) + P_r(B4). Joint carrier:
**H4JC-R1 (event-level)** — PASS-WITHSTOOD at the round-2 re-review; the
retracted pointwise form appears nowhere below.

**THEOREM D1 v2.2(1) — CERTIFIED RUNG, r = 0.05.**

    1 − q(0.05, 6/5) ≤ Ĩ_hi + C_RN(0.05)·√Q_{0.05}(B1.dir)
                       + P_{0.05}(B2) + P_{0.05}(B4),

with:

- **E_w(0.05) ≤ Ĩ_hi = 8.1272827e-2 = 650.1827·(0.05)³** (both displays
  rounded UP at the last displayed digit; exact recomputation
  650.1826140…), the floor-consistent certified snapshot:
      chart part = I_hi(v3) − 19.55·(0.05)³ = 628.2548·(0.05)³ (rounded UP;
                 UNCHANGED from v2.1: I_hi(v3) = 8.0975589252e-2 =
                 647.8048·(0.05)³ rounded UP, version-pinned 8d7028e4…)
      remote part = B_remote·(0.05)³ = 21.9279·(0.05)³ = 2.7409875e-3,
  where **B_remote = 17.6804 + 2.5282637·(1 + κ_far), κ_far ≤ 0.68
  CERTIFIED** — floor-consistent: I_ann = Num_ann/Z_lo exact-kernel with
  Num_ann = 1.714838453e-5 (consistency-ck'd against the frozen 17.0182368
  at the old probe denominator); the per-piece Z-scaling audit is verbatim
  in the v2 amendment (κ_cross 1/Z, κ_pair 1/Z, κ_y Z-free, R_pg/R_wm
  Z-free, I_ann 1/Z), assembled at Z_lo = 7.7592917375327855e-3 (R2's
  certified floor): κ_cross 0.64990178 + κ_pair 0.0025176 + κ_y 0.0243696
  = 0.677284905 → round-UP 0.68. **"Zero Monte-Carlo content" is now TRUE:**
  every denominator in the bracket is R2's interval-enclosed certified floor
  (H3_RUNG_FLOOR 6347275d…, pinned by D3 with a drift ck); the QMC probe is
  diagnostic-only, consistency-ck'd inside the R2 interval. The v2.1
  "zero MC" claim (made while the κ pieces and I_ann sat at the QMC probe
  denominator) is recorded as a FALSE CLAIM REPAIRED (scope V1); the v2.1
  value 21.2153 and the mixed form 21.2658 are both SUPERSEDED-BY-FLOOR
  (reconciliation, per the v2 amendment's table: 19.5465 κ=0 point display,
  rejected; 20.9 frozen THM display, superseded; 21.2153 v1 QMC-denominator;
  21.2658 mixed denominators; 21.9279 v2 certified floor-consistent).
  Full remote-class statement: P_r(A.rem) ≤ 1.284 + 21.9279 = 23.2119·r³
  (I_hole consumes C1's chart envelope, as frozen). Ratio to C1's evidence
  point value: ≈ 507× (honest width, stated).
- **C_RN(0.05) ≤ 3.46** (rounded UP from the certified 3.4591), denominator
  the same certified rung floor (R2; +92.12% margin; stretch rungs
  0.045/0.055 same margins).
- **Named hypotheses of (1), unchanged** — (1) is unconditional apart from:
  (a) **H5-RIM and H5-AXIS v1+v2+v3** (named-lemma grade; v3 frozen in full
      in the tightening amendment);
  (b) **D3-LEMMA-RN-UNIF(r = 0.05)** — the zone-uniformity rung part,
      precisely stated and NOT closed (rigidity-decoupling lemma for τ(y)
      uniform over {d ≥ 5}; certified interval-box Riemann sum for the
      annulus crude-spine integral). Closed at the rung within that item:
      all station κ pieces (now floor-certified), the exact far-zone main
      term (576−25π)J(ℓ), monotone-decay exact-kernel evidence.
- **G.7-scope normalizer at the rung: DISCHARGED** — every 1/Z_r in the
  consumed objects divides R2's certified floor Z_lo; recorded in the CLOSED
  register (§3) with this mechanism. (The existential two-sided normalizer
  H3 stands separately, closed since v1.1; the continuum band remains
  unexecuted, routed into OBL-D1-PROMOTE.)

**THEOREM D1 v2.2(1′) — the rung with the dir remainder priced (sharper,
unchanged from v2.1).** B1.dir ⊆ NearSwap(d_max) ∪ FarRoute(d_max),
d_max = 4r–5r:

    1 − q(0.05, 6/5) ≤ Ĩ_hi + P_{0.05}(NearSwap) + P_{0.05}(FarRoute)
                       + P_{0.05}(B2) + P_{0.05}(B4),

with NearSwap CERTIFIED SUPER-ALGEBRAIC (barrier margin Θ(1/r): 29.40σ →
58.79σ per halving at d = 1r; 7.39σ·(0.05/r) at d = 4r; sd(f_yyy | pins) =
2.4495 = Θ(1); no single-slice C⁰ dam; the C¹ tube route displayed
certified-dead per H4-SD) and FarRoute handed to PERC-DECAY with the exact
certified boundary. Independent display (EVIDENCE): q(0.05) = 0.99982,
1 − q = 1.77e-4 = 1.42·(0.05)³ (exact continuous-field MC), never a premise.

**THEOREM D1 v2.2(2) — ALL-SMALL-R FORM, CONDITIONAL (round-2 register).**
*Assume the named VALIDITY premises:*

- **OBL-D1-PROMOTE** (uniform certified chart envelope on (0, r₀']; with
  promotion-lane sub-obligations OBL-H5-JETMOD / OBL-H5-ZBAND /
  OBL-H5-REMOTE-THRESHOLD; progressing per the lead's status line: r-law cks
  green, κ = 1/8 modulus band, rungs executing);
- **D3-LEMMA-RN-UNIF** (rung part precisely stated in (1)(b); uniform-in-r
  part as frozen; foundations);
- **PERC-DECAY** — every far lane's o(r³) pricing: B1.dir-far (FarRoute,
  handed off with the exact certified boundary), B2-far, B4.rem;
- **OBL-B1-BRANCH(loop|B1)** — the re-pointed sharp loop factor (constant-
  level; E[N_loop] ≥ P(A) side consequence carried);
- **THE B4.loc DAM-LINE TUBE CERTIFICATE** — moved from REFINEMENT to
  VALIDITY premises per the scope ruling V2: with no raw counted-class
  envelope for B4.loc anywhere in the carriers (grep-verified across
  C2/D2/D3/B1/H4 by the scope instance), the dam-line sup-tail certificate
  is the ONLY route to any O(r³) bound on that summand. Exact content: the
  uniform Gaussian sup-tail over the pair-level cut net with the RN
  prefactor (C2's B4 lane: per-cut P ≤ (√E[W²]/Z_r)·Q(sup_ζ f ≤ s)^{1/2},
  the certified barrier-margin ladder κ = Θ(1/r) being kernel-grade already);
  **and, carried as part of the open item's exact content: the assembly's
  identification of the pair-level cut-net with D2's 9-pin M-ward-tube item
  (ii) (OBL-D2-AO-SHARP(ii) ≡ C2's "OBL-B1-BRANCH (tube item)") is itself
  ASSERTED, NOT ESTABLISHED — the identification must be established or the
  item stated and owned independently of D2's tube lane.**

*Then there exist r₀ > 0 — EXISTENTIAL (H3's convergence has no certified
modulus; R2's floors are pointwise at rational rungs; the interval-r band is
unexecuted, routed into OBL-D1-PROMOTE) — and a finite C with*

    1 − q(r, 6/5) ≤ C · r³     for all 0 < r ≤ r₀,

*the dir-near term super-algebraic (CERTIFIED), B2-corridor and B4.loc
super-algebraic via their dam/profile lines (B4.loc's line now a validity
premise above), the far lanes o(r³) under PERC-DECAY, and the uniform
envelope constant from OBL-D1-PROMOTE as C's leading content.*

**Refinement register (round 2):** OBL-D2-AO-SHARP (i, iii, v) ONLY — with
AO ≤ 1 exact these sharpen constants, not validity; item (ii) has moved to
the validity premises above. B2-corridor's kill remains refinement-grade
(its raw corridor envelope is Θ(r³)-or-better by its power ledger, verified
in v2.1 and unchallenged at round 2).

## 2. Display discipline and recorded nits (round 2)

Round-UP display discipline stands (v2.1 §2, repaired there): consumed
coefficient displays 650.1827 (exact 650.1826140…), 628.2548, 647.8048,
21.9279 (exact bracket 21.92788306…), 23.2119, 731.4311, 711.8811, C_RN
ladder 3.4382/3.4591/3.4644/3.4657.

Recorded nits/errata (display-only; computations unaffected):
- **N-v3-1 (scope's nit):** v2.1's inline display "I_hi(v3) = 8.0975589252e-2"
  truncates the exact json value DOWN by 1.3e-13 (immaterial; recorded; the
  round-UP form of the coefficient, 647.8048, is the governing display).
- **N-v3-2:** the coefficient display 647.8048 is the round-UP form; the
  mission-level 647.8047 (round-down) appears only in the historical H5
  errata document — recorded, no conflict.
- **N-D3V2-1 (found at assembly-gate construction, recorded for D3's owner):**
  the v2 amendment's floor-assembled κ display (0.677284905, matching the
  ratio-scaled v1 pieces to 1e-5) sits ~5e-4 ABOVE the assembly recomputed
  from its own displayed Z-free numerators (0.67678902 from N_cross =
  5.04277749e-3, N_pair = 1.953515362e-5, κ_y = 0.0243696); likewise the
  probe diagnostic (0.652843363 vs 0.65235611). Display inconsistency only:
  the certified cap κ_far ≤ 0.68 covers BOTH readings (gate cks
  C-kappa-cap-covers-both), the breach datum (> 0.66) holds under both, and
  the bracket prices at the cap — the consumed 21.9279·r³ is unaffected.
- **E-H3-2:** the h3 CR5f mislabel (flagged at the round-2 numerics review)
  is recorded as an erratum for H3's next amendment.
- **E-BDIR-1/2:** the two BRANCH_dir receipts defects flagged by the round-2
  numerics instance — owner's errata in flight, non-blocking; recorded here.
- Prior recorded errata stand: E-H3-1 (c₀ prose +1.4e-40), E-C1-1
  (a₆ − 15 = −3.1e-117), E-H5-1/2/3 (A≡B docstring; stale records; freeze-time
  totals unrecoverable ≤ 4.6e-12).

## 3. Obligation register v2.2

**CLOSED:** OBL-B1-REG; H3 existential form (c_Z = c₀/2 ≥ 1.615489267643502474,
r₀ existential); H3 rung floor (R2); **G.7-scope normalizer AT THE RUNG (R2:
every consumed 1/Z_r divides the certified floor Z_lo — discharged per the
v2 amendment §3)**; H2 at the O(r³) grade; H4 except the re-pointed loop
factor; H4-JC as H4JC-R1 (PASS-WITHSTOOD at round 2); OBL-B1-BRANCH(dir)
(NearSwap certified super-algebraic; FarRoute → PERC-DECAY); **the remote
bracket, floor-consistent κ-inclusive (v2 amendment: B_remote = 21.9279·r³,
genuinely Monte-Carlo-free)** modulo D3-LEMMA-RN-UNIF; the H5 rung interval
(v3 totals, version-pinned, downward-only).

**OPEN — VALIDITY premises of Theorem (2):** OBL-D1-PROMOTE (H5 lane + D1;
sub-obligations OBL-H5-JETMOD / OBL-H5-ZBAND / OBL-H5-REMOTE-THRESHOLD);
D3-LEMMA-RN-UNIF (rung + uniform parts; foundations); PERC-DECAY
(B1.dir-far, B2-far, B4.rem; percolation lane); OBL-B1-BRANCH(loop|B1)
(branch-control lane; constant-level); **the B4.loc dam-line tube
certificate** (uniform Gaussian sup-tail over the pair-level cut net + RN;
INCLUDING the open identification of the pair-level cut-net with D2's 9-pin
M-ward-tube item (ii), asserted-not-established — scope ruling V2;
branch-control/foundations lanes); H5-RIM / H5-AXIS production paths
(named lemmas inside the rung certificate; shell_hi_fi, factored-det).

**REFINEMENT register:** OBL-D2-AO-SHARP (i, iii, v) only.

## 4. CHANGES — round-2 rows appended to the v2.1 table

The v2.1 CHANGES table (D1_ASSEMBLY_v2_1.md §4, incl. the G1–G6 mapping
recorded in D1_V2_1_RECEIPTS.txt) stands in full. Round-2 additions:

| finding | content | v2.2 disposition |
|---|---|---|
| re-review H4JC-R1 (counterexample instance) | event-level joint carrier attacked | **PASS-WITHSTOOD** — no change |
| re-review numerics: two BRANCH_dir receipts defects | receipts-level defects in the DIR carrier | **RECORDED** (E-BDIR-1/2); owner's errata in flight; non-blocking — the consumed theorem content (margins, split boundary) is untouched |
| re-review numerics: κ-probe sensitivity flag | κ pieces evaluated at the QMC probe | **REPAIRED AT SOURCE by the V1 repair** (v2 amendment; floor-consistent) |
| scope V1 | κ_far assembles 0.6773 > 0.66 at the certified floor; G.7 consumption dropped from the register; "zero MC" false | **REPAIRED:** consumption moved to the v2 amendment (κ_far ≤ 0.68 floor-certified; B_remote = 21.9279·r³; genuinely MC-free against R2's interval-enclosed floor); the v2.1 false claim recorded as repaired; G.7-at-the-rung recorded CLOSED with its mechanism (§3); gate v3 discriminates below 21.9279 |
| scope V2 | B4.loc's dam line must be a VALIDITY premise (no raw counted-class envelope exists) | **REPAIRED:** moved to Theorem (2)'s validity premises with exact content, including the asserted-not-established cut-net/tube identification (§1); gate v3 cks its presence |
| re-review numerics: h3 CR5f mislabel | label defect in H3's records | **RECORDED** as erratum E-H3-2 (H3's next amendment) |
| scope nit | v2.1's I_hi(v3) inline display truncates down 1.3e-13 | **RECORDED** (N-v3-1; immaterial; the round-UP coefficient display governs) |
| mission/numerics nit | 647.8047 round-down in the historical errata doc vs the round-UP 647.8048 | **RECORDED** (N-v3-2; no conflict) |

## 5. EXECUTIVE-STATE BLOCK (one page; for the consolidated return package)

**Strongest theorem established exactly (rung; unconditional apart from the
two named items):** Theorem D1 v2.2(1) —
1 − q(0.05, 6/5) ≤ 8.1272827e-2 = 650.1827·(0.05)³ + C_RN(0.05)·√Q(B1.dir)
+ P(B2) + P(B4), C_RN ≤ 3.46; named items: H5-RIM/H5-AXIS v1+v2+v3 (named
lemmas), D3-LEMMA-RN-UNIF(r = 0.05). Sharper priced form (1′) with the dir
remainder split into certified-super-algebraic NearSwap + named FarRoute.

**Strongest conditional theorem:** Theorem D1 v2.2(2) —
1 − q(r, 6/5) ≤ C·r³ for all 0 < r ≤ r₀ (r₀ existential), conditional on
five named validity premises.

**Named open obligations (one line each):**
- OBL-D1-PROMOTE — uniform certified chart envelope on (0, r₀'] (the
  all-small-r promotion; sub-obligations JETMOD/ZBAND/REMOTE-THRESHOLD).
- D3-LEMMA-RN-UNIF — zone uniformity of the RN brackets (rung part:
  rigidity-decoupling + certified annulus Riemann sum; uniform part frozen).
- PERC-DECAY — subcritical level-s⁺ cluster decay feeding every far lane.
- OBL-B1-BRANCH(loop|B1) — sharp loop fraction given B1 (constant-level).
- B4.loc dam-line — uniform Gaussian sup-tail over the pair-level cut net +
  RN; incl. establishing (or independently stating) the cut-net ≡ D2
  9-pin-tube identification.
- H5-RIM / H5-AXIS production paths — shell_hi_fi / factored-det
  machine-certification of the rung certificate's two named-lemma regions.
- (Refinement: OBL-D2-AO-SHARP (i, iii, v) — sharp constants only.)

**Failed attacks that materially increased confidence (breaks → repairs):**
the H4-JC pointwise disjoint-carrier falsity (Stage-E break → R1 event-level
repair, OLD violations 132/175 pinned fail-closed, PASS-WITHSTOOD at round
2); the rung-floor gap (→ R2 certified interval, +92.12%); the remote κ
under-pricing (→ R3; then round-2 V1 floor breach 0.6773 > 0.66 → v2
amendment, 21.9279·r³ genuinely MC-free); the "unconditional" over-claim
(→ named-hypothesis restatement); PERC-DECAY unnamed (→ named validity
premise); rounding-polarity defects (→ round-UP display discipline,
gate-ck'd); H5-AXIS v3 unfrozen (→ frozen tightening amendment); B4.loc
misplaced as refinement (→ validity premises, scope V2); the H5 I_lo
rung-pollution (→ rung-scoped merger, v3 totals, version discipline); the
v2.1 gate's body-only citation ck (exposed by my own mutation test →
whole-file ck); plus the campaign-era breaks honored throughout: the C021
retraction (no historical constant consumed), the MU-5 3D-index amendment,
the poison-cell receipt th15_dl0.066 (6.47e7 vacuous record preserved,
recertified 2.17e-5).

**CANNOT-VERIFY (recorded separately, none blocking):** the freeze-time
exact H5 totals (unrecoverable; ≤ 4.6e-12 worst-case on the last displayed
digit; version discipline prevents recurrence); the h3 CR5f label content
(owner's amendment pending); the two BRANCH_dir receipts defects (owner's
errata in flight); the mission-level 647.8047/647.8048 display pair
(reconciled as round-down/round-up of the same value).

**Running tasks (states per the lead's status line):** H5 promotion rungs —
EXECUTING (r-law cks green; κ = 1/8 modulus band; rung certificates in
flight); W3 LOWER driver — RUNNING; W8 Phase-2 — RUNNING; BRANCH rung025 —
RUNNING; H3 band assessment — RUNNING (the interval-r enclosure extending
R2's pointwise floors to a continuum band; unexecuted, feeds OBL-D1-PROMOTE).

## 6. Gate v3 (d1_falsify_v3.py — operative; v2 gate frozen historical)

All v2 cks retained (with the consumed values updated), plus:
(i) the bracket ck consumes ≥ 21.92788306 with display 21.9279 and REJECTS
    anything below (extending the D3 engine's operative edge [21.27, 22.5]
    into the assembly gate: 19.5465/20.9/21.2153/21.2658-class values all
    rejected);
(ii) a κ-denominator ck: the gate recomputes the κ pieces from their Z-free
    numerators at Z_lo and at a trial denominator ABOVE the floor; the
    assembled bracket must be strictly smaller at the higher denominator,
    and the consumed bracket must dominate every above-floor evaluation —
    any consumption priced at a denominator above the certified floor fails
    closed (the assembly-level mirror of D3's MUT-V2-3);
(iii) the B4.loc validity-register ck: the dam-line premise (with the
    cut-net identification caveat) must be present in Theorem (2)'s premise
    list (removal kills the gate);
(iv) receipts nits cks as recorded in §2.
Both modes byte-identical; mutation self-tests: retracted-string injection,
upward-drift totals injection, QMC-denominator (21.2153) substitution,
B4.loc-premise removal — each killed; restored state PASS byte-identical.

END_FROZEN_BODY

## Freeze record (outside frozen body)

Extraction rule (corpus convention): lines strictly between the unique exact lines
BEGIN_FROZEN_BODY / END_FROZEN_BODY of this file; normalize line endings to LF;
strip leading/trailing blank lines; retain exactly one terminal LF.

- Frozen body bytes: 18311
- Frozen body SHA-256: 490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6
- Operative gate: d1_falsify_v3.py (92 cks, PASS both modes, byte-identical;
  PASS digest d800849ed5966b4d583a10cf72525e89350d9a53780b45a807a21ccbadf91085;
  mutation self-tests: retracted-string → FAIL; upward-drift v4 totals → FAIL;
  QMC-denominator 21.2153 substitution → FAIL F-remote-display; B4.loc-premise
  removal → FAIL F-B4loc-premise-bullet; restored state PASS byte-identical).
  Transcripts v3_t_normal.txt ≡ v3_t_opt.txt.
- Receipts: D1_V2_2_RECEIPTS.txt.
- Historical (preserved untouched, re-verified by gate block A): v1.0 006b8a7d…,
  v1.1 634338b4…, v1.2 a7e1958c…, v2.0 86882dca…, v2.1 bcc177fe…; gates
  d1_falsify.py / d1_falsify_v2.py frozen historical.
