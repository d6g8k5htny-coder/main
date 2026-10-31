# D1_ASSEMBLY v2.3 — round-3 issuance (register note applied; two-premise conditional theorem) — DRAFT

**Artifact:** D1-ASM-20260915-v2.3-DRAFT
**Campaign:** U2D-UPPER, Stage E (assembly; round-3 issuance folding the D1 v2.2 REGISTER NOTE).
**Agent:** Claude (Anthropic family) acting for the assembly role while Kimi is dark (2026-09-15 → 09-30).
**Status:** PROPOSED. This draft changes NOTHING the register note (body c1d5e95d…) did not already
adjudicate, plus three record corrections certified in CL-ERR-001 and one consumption certified in
H5_ZBAND_CONSUMPTION. Every carrier below was hash-verified FROM BYTES before consumption (gate
`d1_falsify_v4.py`, a strict superset of gate v3).
**Historical (preserved untouched):** v1.0 body 006b8a7d…; v1.1 634338b4…; v1.2 a7e1958c…; v2.0 86882dca…
(gate d1_falsify.py); v2.1 bcc177fe… (gate d1_falsify_v2.py); **v2.2 body
490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6 (gate d1_falsify_v3.py, digest
d800849e…, re-executed in the Anthropic environment 2026-09-15: identical).** v2.2 remains the last
Kimi-issued form; this v2.3 becomes the current strongest form only upon operator promotion.

BEGIN_FROZEN_BODY

## 0. Round-3 carrier ledger (verified at consume time; rule ids per CL-REG-001)

| consumed for | artifact | pin | rule |
|---|---|---|---|
| all register actions | D1_assembly/D1_ASSEMBLY_v2_2_REGISTER_NOTE.md | body c1d5e95d97e019574d6b6b09e6e606fd6e0cbc1fe3aa2e0a54953de6e4f2c470 | marker_strip_LF |
| B4 closure (loc + rem) | B4LOC_damline/B4LOC_DAMLINE.md; b4loc_driver.py; b4loc_falsifier.py; t_normal.txt ≡ t_opt.txt | body 0d5c1b3284374f3a2f7630b7f603a71c77f5ec9417fe4579aaac03becd33c05e; bf3b0225…; d59ad790…; 11baaccc… | marker_raw; whole |
| far-lane restatement | D3_percolation/PERC_DECAY.md; d3_perc_decay.py; d3_falsifier.py | body 5137a811e6be72ed27e0458c75537bdcfabb8d8e7148b211d8c7b9a1584df400; ee68ac75…; abbb40d2… | marker_raw; whole |
| normalizer band (lo) | H3_closure/H3_BAND_FLOOR.md; h3_band_floor.py; band_normal.txt ≡ band_O.txt | body 281477c39412840bc9ca56a256884f3ddd5f0ae4283f90a017162676771382a7; a907eeed…; dcb3c8e6… | excl_bodyhash_line; whole |
| normalizer band (hi) | H3_closure/H3_BAND_CEIL.md; h3_band_ceil.py; ceil_normal.txt ≡ ceil_O.txt | body cfe8a3a49e3281dfdc4ef5d925ae622b4f7aeea3d0462baae61b87dae699ede2; b97c5428…; 26d08534… | excl_bodyhash_line; whole |
| OBL-H5-ZBAND discharge | H5_ZBAND_CONSUMPTION_2026-09-15.md; h5_zband_consume.py; zb_normal.txt ≡ zb_O.txt | 9445bdd3…; f5965c43…; a570dade… (digest cbe8603f…) | whole |
| rung ladder (custody) | H5_closure/H5_RUNG2_2026-09-15.md; H5_RUNG3_2026-09-15.md; h5_totals_r0.025_v2.json; h5_totals_r0.035355_v2.json | bodies 91a34d83…, f4c3414f…; f7697bcf…; 808d6901… | sepbody; whole |
| (all other carriers) | the v2.2 ledger (§0 of v2.2) and the v2.1 ledger it incorporates | unchanged | re-verified by gate v4 |

## 1. THEOREM D1 v2.3

Setting and partition as frozen (v2.0 §0): on Reg ∩ TYP,
1 − q(r, 6/5) = P_r(A) + P_r(B1) + P_r(B2) + P_r(B4). Joint carrier: **H4JC-R1 (event-level)**,
PASS-WITHSTOOD at round 2; the retracted pointwise form appears nowhere below.

**THEOREM D1 v2.3(1) — CERTIFIED RUNG, r = 0.05 (unchanged from v2.2 except the recorded exact bracket).**

    1 − q(0.05, 6/5) ≤ Ĩ_hi + C_RN(0.05)·√Q_{0.05}(B1.dir) + P_{0.05}(B2) + P_{0.05}(B4),

with **E_w(0.05) ≤ Ĩ_hi = 8.1272827e-2 = 650.1827·(0.05)³** (round-UP displays; exact recomputation
650.1826140…): chart part = I_hi(v3) − 19.55·(0.05)³ = 628.2548·(0.05)³ (I_hi(v3) = 8.0975589252e-2 =
647.8048·(0.05)³, version-pinned 8d7028e4…); remote part = B_remote·(0.05)³ = 21.9279·(0.05)³ =
2.7409875e-3, **B_remote = 17.6804 + 2.5282637·(1 + κ_far), κ_far ≤ 0.68 CERTIFIED, exact bracket
21.927883016 (round-UP 21.9279)** — the v2.2 string "exact bracket 21.92788306" is RECORDED AS A
DISPLAY DEFECT (E-D3V2-2: I_far written 4.247483056 for 2.5282637 × 1.68 = 4.247483016; +4.4e-8,
conservative polarity; bound unaffected). Every denominator in the bracket is R2's certified floor
Z_lo = 7.7592917375327855e-3; zero Monte-Carlo content. Full remote-class statement:
P_r(A.rem) ≤ 1.284 + 21.9279 = 23.2119·r³. Honest width vs C1's evidence point value: ≈ 507×.
**C_RN(0.05) ≤ 3.46.** The C_RN·√Q(B1.dir) term is retained as a VALID BUT REDUNDANT bound (see (1′)).

- **Named hypotheses of (1), unchanged:** (a) H5-RIM and H5-AXIS v1+v2+v3 (named-lemma grade);
  (b) **D3-LEMMA-RN-UNIF(r = 0.05)** — precisely stated, NOT closed (rigidity-decoupling lemma for τ(y)
  uniform over {d ≥ 5}; certified interval-box Riemann sum for the annulus crude-spine integral); executable
  work order: CL-OBL-001 §2.
- **G.7-scope normalizer at the rung: DISCHARGED** (R2). **On the band: the two-sided normalizer is now
  bracketed uniformly**, 2.30659559567154 ≤ Z_r/r² ≤ 3.74767948915996 on (0, 0.05] (H3 band floor +
  ceiling, consumed by H5_ZBAND_CONSUMPTION).

**THEOREM D1 v2.3(1′) — THE RUNG WITH THE DIR REMAINDER PRICED (sharpened per register note §4).**
With B1.dir ⊆ NearSwap(d_max) ∪ FarRoute(d_max), d_max = 4r–5r, and **FarRoute(d_max) ⊆ {N_w ≥ 1}
CERTIFIED (PERC-DECAY §3(1))**, the events A, B1_wit and FarRoute are pairwise disjoint subsets of
{N_w ≥ 1} and are covered by ONE payment of E_w:

    1 − q(0.05, 6/5) ≤ Ĩ_hi + P_{0.05}(NearSwap) + P_{0.05}(B2) + P_{0.05}(B4),

NearSwap CERTIFIED SUPER-ALGEBRAIC (barrier margin Θ(1/r): 29.40σ → 58.79σ per halving at d = 1r;
sd(f_yyy | pins) = 2.4495), **P_{0.05}(B4) ≤ 1.23e-9 CERTIFIED (B4LOC-R1; consumed at round-UP of its
3-significant-figure nearest-rounded display 1.22e-9 — E-B4LOC-1)**, and the FarRoute term
dissolved as a free-rider of E_w. Consequently the C_RN·√Q(B1.dir) term of (1) prices nothing beyond
NearSwap plus the free-rider. Independent display (EVIDENCE, never a premise): 1 − q(0.05) = 1.42·(0.05)³.

**THEOREM D1 v2.3(2) — ALL-SMALL-R FORM, CONDITIONAL (round-3 register: TWO premises).**
*Assume the named VALIDITY premises:*

- **OBL-D1-PROMOTE** — uniform certified chart envelope on (0, r₀']: the r-scaled rung family
  647.8048 / 661.4712 / 664.3979 at r = 0.05 / 0.035355 / 0.025 (+2.11%, +0.44%; decelerating; C_unif
  candidate 731.4311 dominates with 10.1% margin; falsifier not tripped; rungs 0.0177 / 0.0125 banked
  mid-flight); **now also carrying the uniform-band extension of the three certified pieces that close
  B4, B2-far and FarRoute at the rung (B4LOC-R1, PERC-DECAY §3(1)–(2))** — each is r-scaled and proved on
  its ladder, and each must be certified on the band with r as an interval. Sub-obligations:
  OBL-H5-JETMOD (display only); OBL-H5-REMOTE-THRESHOLD (rides with D3-LEMMA-RN-UNIF);
  **OBL-H5-ZBAND DISCHARGED at consumption grade** (band floor + ceiling consumed).
- **D3-LEMMA-RN-UNIF** — rung part precisely stated in (1)(b); uniform-in-r part as frozen; foundations;
  the only premise that did not move on 2026-09-15.

*Then there exist r₀ > 0 — EXISTENTIAL (r₀ is bounded by the promotion ladder's reach, not by a certified
modulus) — and a finite C with*

    1 − q(r, 6/5) ≤ C · r³     for all 0 < r ≤ r₀,

*the dir-near term super-algebraic (CERTIFIED), B4 super-algebraic at grade exp(−c_eff/r²) with
c_eff ≥ 0.0513 increasing (CERTIFIED at the ladder; band extension inside OBL-D1-PROMOTE), B2-corridor
super-algebraic via its profile line (refinement-grade kill), the far lanes absorbed at Θ(r³) with
certified constants — B1.dir-far D3-part ≤ 23.2119·r³, B2-far ≤ 17.6802·r³ — inside the E_w accounting,
and the uniform envelope constant from OBL-D1-PROMOTE as C's leading content.*

**Refinement register (round 3):** OBL-D2-AO-SHARP (i, iii, v); **OBL-B1-BRANCH(loop|B1) — DEMOTED from
validity to refinement** (constant-level only; E[N_loop] ≥ P(A) side consequence carried); PD-CONN
(named input, upgrade-only: it certifies constants, never order; three missing pieces named in PERC_DECAY
§4); the B2-far free-rider pinning check (N_β* ≤ N_w against C2's exact definition; register note §4.4).

## 2. Display discipline and recorded nits (round 3)

Round-UP display discipline stands. Consumed coefficient displays: 650.1827 (exact 650.1826140…),
628.2548, 647.8048, 21.9279 (**exact bracket 21.927883016**), 23.2119, 17.6802 (B2-far), 731.4311,
711.8811, C_RN ladder 3.4382/3.4591/3.4644/3.4657; band normalizer 2.30659559567154 / 3.74767948915996.

New recorded errata (display-only; computations unaffected):
- **E-D3V2-2:** I_far display 4.247483056 → exact 4.247483016; bracket 21.92788306 → 21.927883016.
  Gate v4 consumes the exact value with exact-equality recompute (tolerance 0).
- **N-v3-2 RESOLVED:** I_hi(v3)/r³ = 647.804714016; 647.8047 is the round-DOWN of that single value;
  only 647.8048 is consumable. CANNOT-VERIFY item 4 of RETURN_06 closes.
- **E-B4LOC-1:** B4LOC-R1's per-rung displays (1.22e-9 / 1.52e-45 / 1.03e-197) are `mpmath.nstr(·, 3)`
  NEAREST-rounded; the driver's o(r³) gate uses the exact value, so the certificate is sound, but under the
  corpus's round-UP display discipline an upper bound must be consumed at 1.23e-9 (≤ 0.41% above any value
  that displays as 1.22e-9). Recorded for B4LOC's next amendment (print round-UP, or emit 4 digits).
- **E-LPW-29:** the float-rendered 29-digit LPW v2 display retracted by H1 F-C29 must not be cited as
  "exact"; the current LPW constant is v4's (2.2291086664054236617851686509e-10, r₀ = 1/2278031360);
  gate v4 forbids the retracted token in this body.
- Prior recorded nits stand: N-v3-1, N-D3V2-1, E-H3-1/2, E-C1-1, E-H5-1/2/3, E-BDIR-1/2 (owner's errata
  landed 2026-09-15, 7547e76a…); the v2.1 "zero MC" claim remains recorded as a FALSE CLAIM REPAIRED (scope V1).

## 3. Obligation register v2.3

**CLOSED:** everything closed in v2.2 §3; **B4.loc dam-line certificate (B4LOC-R1, whole-B4, grade
exp(−c_eff/r²)); B4.rem (adjudicated: a wrap route at level s is by definition not B4; D3's remote part
was a pricing route, not an event gap); the cut-net ≡ 9-pin-tube identification (PROVEN FALSE on four
axes — law / tube / level / scaling; the 9-pin tube is a proof template only; no asserted identification
survives anywhere in the B4 lane); PERC-DECAY in its restated form (far lanes Θ(r³) with certified
constants; validity content absorbed into the E_w accounting); the dir remainder at the rung (free-rider);
OBL-H5-ZBAND (consumption grade)**.

**OPEN — VALIDITY premises of Theorem (2):** OBL-D1-PROMOTE (H5 lane + D1; sub-obligations JETMOD /
REMOTE-THRESHOLD; plus the uniform-band extension of B4LOC-R1 / B2-far / FarRoute); D3-LEMMA-RN-UNIF
(rung + uniform parts; foundations; work order CL-OBL-001 §2). H5-RIM / H5-AXIS production paths remain
named lemmas inside the rung certificate.

**REFINEMENT register:** OBL-D2-AO-SHARP (i, iii, v); OBL-B1-BRANCH(loop|B1); PD-CONN; B2-far
free-rider pinning.

**ADVERSARIAL LEDGER (round 3 additions):** the o(r³) far-lane reading (v2.0–v2.2 prose) — REFUTED at
evidence grade by PERC-DECAY's certified corridor-alive display; the whitened/box pipeline for interval-r —
infeasible (H3 band floor §2); the naive p̄-only normalizer envelope — not viable (H3 band ceil §3);
v3 F-LEVER4 — a band floor is the wrong direction for the Palm denominator (refused, then closed by the
ceiling); the rung-2 stitch-drift alarm — resolved at rung 3 (rung 1's refine-3 advantage).

## 4. CHANGES — round-3 rows appended to the v2.2 table

| finding | content | v2.3 disposition |
|---|---|---|
| register note §1 (B4LOC-R1) | B4 ⊆ E1 ∪ E2 exact; Borell–TIS sup-tails; P_r(B4) ≤ 1.22e-9 / 1.52e-45 / 1.03e-197 (nearest-rounded displays; consumed at round-UP 1.23e-9); exp(−c_eff/r²) | **CLOSED** — validity premise 5 discharged at a stronger grade than required; identification caveat RESOLVED NEGATIVELY; B4.rem covered |
| register note §3 (PERC-DECAY) | o(r³) far-lane reading NOT reachable; Θ(r³) restatement with certified constants; PD-CONN named | **REFUTED-AS-PHRASED → RESTATED → ABSORBED**; premise 3 removed; PD-CONN to refinement (upgrade-only) |
| register note §4 (dir accounting) | FarRoute ⊆ {N_w ≥ 1} certified; A / B1_wit / FarRoute pairwise disjoint; one E_w payment | **STRENGTHENED** Theorem (1′); C_RN·√Q(B1.dir) in (1) retained as redundant |
| register note (loop factor) | constant-level only | OBL-B1-BRANCH(loop\|B1) **DEMOTED** to refinement; premise 4 removed |
| H3 band floor + ceiling; ZBAND consumption | Z_r/r² ∈ [2.3066, 3.7477] uniform on (0, 0.05] | **OBL-H5-ZBAND DISCHARGED** (consumption grade); normalizer sub-part of OBL-D1-PROMOTE discharged |
| H5 rungs 2–3 | 661.4712 / 664.3979; +2.11% / +0.44% | recorded as OBL-D1-PROMOTE progress; drift alarm resolved; **driver code unpinned (CL-PIN-001) — custody only** |
| CL-ERR-001 E1 | I_far / exact-bracket display slip +4.4e-8, conservative | **RECORDED** E-D3V2-2; gate v4 exact recompute |
| CL-ERR-001 E2 | retracted 29-digit LPW display live in RETURN_06 | **RECORDED** E-LPW-29; gate v4 forbids the token |
| CL-REG-001 | six extraction rules live; one claim unreproducible | **RECORDED**; rule ids used in §0 |
| RETURN_06 CANNOT-VERIFY 4 | 647.8047 / 647.8048 | **RESOLVED** (N-v3-2) |

## 5. EXECUTIVE-STATE BLOCK (one page; for the consolidated return package)

**Strongest theorem established exactly (rung; unconditional apart from the two named items):** Theorem
D1 v2.3(1) — 1 − q(0.05, 6/5) ≤ 8.1272827e-2 = 650.1827·(0.05)³ + C_RN(0.05)·√Q(B1.dir) + P(B2) + P(B4),
C_RN ≤ 3.46, P(B4) ≤ 1.23e-9; named items: H5-RIM/H5-AXIS v1+v2+v3 (named lemmas), D3-LEMMA-RN-UNIF(r = 0.05).
Sharper priced form (1′): 1 − q(0.05, 6/5) ≤ 8.1272827e-2 + P(NearSwap) + P(B2) + P(B4).

**Strongest conditional theorem:** Theorem D1 v2.3(2) — 1 − q(r, 6/5) ≤ C·r³ for all 0 < r ≤ r₀
(r₀ existential), conditional on **TWO** named validity premises: OBL-D1-PROMOTE, D3-LEMMA-RN-UNIF.

**Named open obligations (one line each):**
- OBL-D1-PROMOTE — uniform certified chart envelope on (0, r₀'] (three rungs certified, two mid-flight);
  plus the uniform-band extension of B4LOC-R1 / B2-far / FarRoute; sub-obligations JETMOD, REMOTE-THRESHOLD.
- D3-LEMMA-RN-UNIF — zone uniformity of the RN brackets (rung part: rigidity-decoupling + certified annulus
  Riemann sum; uniform part frozen); executable six-step work order filed.
- H5-RIM / H5-AXIS production paths — machine-certification of the rung certificate's two named-lemma regions.
- (Refinement: OBL-D2-AO-SHARP (i, iii, v); OBL-B1-BRANCH(loop|B1); PD-CONN (i)–(iii); B2-far pinning.)

**Failed attacks that materially increased confidence (round 3):** the o(r³) far-lane over-claim (→ Θ(r³)
restatement with certified constants); the asserted cut-net/tube identification (→ proven false, certificate
built directly, three orders of grade gained); the wrong-direction normalizer carrier (→ F-LEVER4 refused,
ceiling built, consumed exactly); the interval-r whitened pipeline (→ killed, Wick/Laurent rebuilt); the
rung-2 drift alarm (→ resolved at rung 3); the exact-bracket display slip (→ E-D3V2-2, exact recompute);
plus every round-1/2 break honored in v2.2 §5.

**Running tasks at issuance (Kimi dark until 2026-09-30; states as of the 2026-09-15 13:11 snapshot):**
H5 rungs 0.0177 / 0.0125 — FROZEN mid-flight (125 + 6 / 42 + 2 banked). W3 LOWER driver — DEAD (last write
00:30; A 44/1920, B 46/2688; pre-registered INCONCLUSIVE). W8 Phase-2 — PAUSED (4/4 hardest-first cells
ledgered at depth 8; X_0 sup route resolved). BRANCH rung025 — LANDED. H3 band assessment — COMPLETE
(floor + ceiling consumed). LPW — v4 frozen. D3-LEMMA-RN-UNIF — no lane executing; work order filed.

END_FROZEN_BODY

---

Body hash of this document: computed by `d1_falsify_v4.py` at issuance and recorded in `D1_V2_3_RECEIPTS.txt`
(rule marker_strip_LF, per CL-REG-001). Gate v4 digest recorded there.
