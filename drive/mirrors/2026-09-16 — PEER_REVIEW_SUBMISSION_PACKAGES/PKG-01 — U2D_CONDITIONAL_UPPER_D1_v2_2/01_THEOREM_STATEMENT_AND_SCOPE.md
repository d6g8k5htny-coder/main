# 01 — Theorem Statement and Scope (D1 v2.2)

**Status:** CONDITIONAL / certified-rung. This is **NOT** an unconditional closed all-small-r theorem.

**Authority:** D1_ASSEMBLY_v2_2.md (D1-ASM-20260915-v2.2). Controlling with CURRENT_STATE_DELTA_2026-09-15.
**Body pin (frozen):** SHA-256 `490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6`
**Whole-file assembly:** SHA-256 `7ca114f0b38680d8bb987c097de10f3faf884ae3b05c3ca47215af5df081c174`

## Setting
On Reg ∩ TYP,
`1 − q(r, 6/5) = P_r(A) + P_r(B1) + P_r(B2) + P_r(B4)`.
Joint carrier: **H4JC-R1 (event-level)** — PASS-WITHSTOOD at round-2 re-review.

## Theorem D1 v2.2(1) — CERTIFIED RUNG, r = 0.05
```
1 − q(0.05, 6/5) ≤ Ĩ_hi + C_RN(0.05)·√Q_{0.05}(B1.dir)
                   + P_{0.05}(B2) + P_{0.05}(B4)
```
with Ĩ_hi = 8.1272827e-2 = 650.1827·(0.05)³ (round-UP), C_RN(0.05) ≤ 3.46 (round-UP).

**Named hypotheses of (1) only:**
- (a) H5-RIM and H5-AXIS (v1+v2+v3; named-lemma grade)
- (b) D3-LEMMA-RN-UNIF(r = 0.05) — zone-uniformity rung part; **NOT closed**

Evidence display (never a premise): q(0.05) = 0.99982.

## Theorem D1 v2.2(2) — ALL-SMALL-R FORM, CONDITIONAL
Assume the five named **VALIDITY premises** below. Then ∃ r₀ > 0 (existential; no certified modulus) and finite C with
```
1 − q(r, 6/5) ≤ C · r³    for all 0 < r ≤ r₀.
```

### Five open validity premises (do NOT invent closures)
1. **OBL-D1-PROMOTE** — uniform certified chart envelope on (0, r₀']; sub-obligations OBL-H5-JETMOD / OBL-H5-ZBAND / OBL-H5-REMOTE-THRESHOLD.
2. **D3-LEMMA-RN-UNIF** — rung part as in (1)(b); uniform-in-r part as frozen.
3. **PERC-DECAY** — far-lane o(r³) pricing (B1.dir-far / FarRoute, B2-far, B4.rem).
4. **OBL-B1-BRANCH(loop|B1)** — re-pointed sharp loop factor (constant-level).
5. **B4.loc dam-line tube certificate** — uniform Gaussian sup-tail over pair-level cut net with RN prefactor; **including** asserted-not-established identification of cut-net with D2's 9-pin M-ward-tube item (ii).

## Scope firewall
- No closed two-sided 2D law at ratified grade.
- Do **not** compose AO48 3D ratified upper with this 2D lower/upper package (ERRATA 2026-09-13).
- Packaging does not discharge open premises.
