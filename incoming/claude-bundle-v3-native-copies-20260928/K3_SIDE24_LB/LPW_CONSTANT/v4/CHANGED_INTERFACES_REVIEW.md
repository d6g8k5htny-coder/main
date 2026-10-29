# LPW_CONSTANT v4 — CHANGED-INTERFACES REVIEW (for independent review)

v4 (successor of frozen v3, script 4a015174…9b02a8c, report body bf58dc82…)
changes exactly ONE interface relative to the v3 certificate. Everything else
(levers 1+2, the moment chain, the amplitude majorant, the invariants, the
mutation discipline) is carried unchanged and re-derived by the v4 program.

## The changed interface — the normalizer carrier (C_Z)

- **Old (v3):** C_Z = 4B₃ = 8328000000000, v1's crude uniform bound
  Z_r ≤ 4B₃·r²; v3 recorded F-LEVER4: no certified uniform upper envelope of
  Z_r/r² on (0, r₀] existed (H3 band = one-sided floor; H3 rung = pointwise at
  0.05), verdict AMEND REQUIRED upstream.
- **New (v4):** C_Z = U = 3.66282864761194, the H3_BAND_CEIL certified uniform
  ceiling E[G_r] = Z_r/r² ≤ U on the cell (0, 0.0025]. Carrier set (all
  sha256-verified from bytes at run time): H3_BAND_CEIL.md full
  18e109bc2e7a186c307f689f940ad476d69fc543e828081e0e706de2c578800d, body
  cfe8a3a49e3281dfdc4ef5d925ae622b4f7aeea3d0462baae61b87dae699ede2
  (body-excl-hash-line rule reproduced); h3_band_ceil.py b97c5428…; transcripts
  ceil_normal/ceil_O 26d08534… (byte-identical); RECEIPT_h3.json ee58ef80….
  H3's own mutation record includes cap_raised failing closed at its U ≤ 4
  consumption ck.

### Direction discipline (the load-bearing logic)

1. The Palm ratio gives 1 − q(r, 6/5) ≥ E[W_r·1_E]/Z_r. Lower-bounding this
   ratio requires an UPPER bound on the denominator: Z_r ≤ C_Z·r². A floor on
   Z_r would be the WRONG direction (it upper-bounds the ratio).
2. H3_BAND_CEIL certifies exactly the needed direction: E[G_r] = Z_r/r² ≤ U
   uniformly on (0, 0.0025], transcript line "CC6 (0,0.0025] CERTIFIED:
   E[G_r] ≤ 3.66282864761194".
3. Domain: v4's r₀ = 1/2278031360 ≤ 1/400, so (0, r₀] ⊆ (0, 0.0025]; the
   ceiling covers the entire certified range (exact Fraction ck).
4. Consumption: EXACT — U = 366282864761194/10¹⁴ as a Fraction; never rounded
   down (the sanctioned round-up to 3.6629 was available but not needed).
5. Consistency (all ck'd in-program): band floor nests below
   (2.30659559567154 < U); pointwise rung hi sits above (U < 4.58745858936);
   H3 consumption cap respected (U ≤ 4); U strictly stronger than the retired
   cap (U ≤ 4B₃).
6. Effect: c = 16·W1·W2·m·δ⁴/U = 2.2291086664054236617851686509e-10, exactly
   c_v3 × (4B₃/U) (×2.273652633308e12); the falsifier checks this factor as an
   exact rational identity.

### Mutation guards (fail-closed, both modes)

- `normalizer_cap_down`: consuming the band floor 2.30659559567154 as the cap
  (wrong direction — a floor is not a ceiling) → FAIL.
- `ceiling_consumed_above_U` (NEW, v4): consuming an inflated value (3.7) or a
  different cell's ceiling (3.66435028046302, cell 2) → FAIL (exact cell-1
  consumption enforced; modeled on H3's cap_raised).
- `denominator_change`: any change to the pinned U → FAIL.

### Verdict

**PASS WITHIN SCOPE.** The new carrier certifies precisely the direction the
Palm denominator requires, on a cell containing the whole certified range, at
v2 discipline (fail-closed, both-modes byte-identical, mutation-tested
including a consumption guard); consumption is exact and consistency-checked
against both the band floor and the rung; v3's F-LEVER4 is discharged. The
reviewer may wish to spot-verify: (i) the body-hash extraction rule against the
H3 MANIFEST line; (ii) the transcript token "CERTIFIED: E[G_r] ≤
3.66282864761194" on the (0, 0.0025] cell; (iii) the exact identity
c_v4/c_v3 = 4B₃/U in v4_falsify.py.

## Unchanged interfaces (spot-verified by re-derivation in v4)

Density floor m = 9/10000 (lever 1); δ = 14587/2621440 with ε′ = 57/640,
clearance 1/3840, weight floor (597/128)·(2596849/204800) (lever 2);
B₃ = 2082000000000 (consistency anchor), K = 8690 → r₀ = 1/2278031360;
S₃/S₄ library enclosures; amplitude majorant E[R⁴] = 8, E[R] ≤ √2; published ≤
exact, power-of-ten floor, rounding checker; 17 mutation families + unknown
trap, all exit 1 in both modes; clean runs exit 0 byte-identical.
