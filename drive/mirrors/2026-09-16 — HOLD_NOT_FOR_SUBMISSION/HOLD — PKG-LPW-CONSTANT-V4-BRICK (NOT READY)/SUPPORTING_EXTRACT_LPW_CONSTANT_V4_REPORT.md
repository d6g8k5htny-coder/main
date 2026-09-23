# SUPPORTING EXTRACT — LPW_CONSTANT_V4_REPORT.md
# Provenance: local gap-closure extract (09152026OKComputer_Project_Gap_Closure.zip)
# Role in PKG-LPW-CONSTANT-V4-BRICK: evidence that EXACT Fractions lower brick text exists.
# Does NOT by itself clear READY — CL-STATE-001 remains PROPOSED pending operator promotion.

---

# LPW_CONSTANT v4 — SUCCESSOR CERTIFICATE REPORT (levers 1+2+4 discharged)

**Statement certified.** For the six-pin law (f(M) = b = 6/5, f(S) = b − r³/6,
∇f(M) = ∇f(S) = 0, M a local max, S a saddle) of the 2D torus Gaussian field,

  **1 − q(r, 6/5) ≥ c·r³  for all 0 < r ≤ r₀,**
  **c = 16·(597/128)·(2596849/204800)·(9/10000)·(14587/2621440)⁴ / 3.66282864761194
     = 394827584594841472652359498125 / 1771235249968883322980290175117427736576
     = 2.2291086664054236617851686509e-10  (published downward-safe 2.22e-10; power-of-ten floor 1e-10),**
  **r₀ = 1/2278031360 = 1/(2¹⁸·8690) ≈ 4.3898e-10.**

v3 (script 4a015174…9b02a8c, report body bf58dc82…) and all prior carriers stay
frozen untouched; v4 is a NEW successor. Every number below is produced by
`lpw_constant_v4.py` (python / python −O byte-identical program output; receipts
separate).

## CHANGES from v3 — exactly one changed interface: the normalizer carrier

| | v3 | v4 |
|---|---|---|
| normalizer cap C_Z | 4B₃ = 8328000000000 (v1 crude uniform bound; F-LEVER4 flagged insufficiency) | **U = 3.66282864761194** (H3_BAND_CEIL certified uniform ceiling on (0, 0.0025]) |
| c | 9.8040863135804911886149013570e-23 | 2.2291086664054236617851686509e-10 (×2.273652633308e12) |
| r₀, δ, m, W, ε′, K | 1/2278031360, 14587/2621440, 9/10000, 59.13997, 57/640, 8690 | unchanged |

**Direction-discipline display (the load-bearing logic of the change).** The
Palm ratio gives 1 − q(r, 6/5) ≥ E[W_r·1_E]/Z_r; lower-bounding this ratio
requires an UPPER envelope of the denominator: Z_r ≤ C_Z·r². H3_BAND_CEIL.md
(body hash cfe8a3a49e3281dfdc4ef5d925ae622b4f7aeea3d0462baae61b87dae699ede2)
certifies E[G_r] = Z_r/r² ≤ 3.66282864761194 for **every** r ∈ (0, 0.0025]
(cell 1), with the transcript line "CC6 (0,0.0025] CERTIFIED: E[G_r] ≤
3.66282864761194". v4's r₀ = 1/2278031360 ≤ 1/400, so (0, r₀] ⊆ (0, 0.0025]:
the ceiling applies uniformly on the whole certified range. Consumption is
EXACT (U = 366282864761194/10¹⁴ as a Fraction — never rounded down; the
sanctioned round-up to 3.6629 was not needed). Consistency, all ck'd: the
certified band floor nests below (2.30659559567154 < U), the pointwise rung hi
sits above (U < 4.58745858936), H3's consumption cap holds (U ≤ 4, never
widened), and U is strictly stronger than the retired crude cap (U ≤ 4B₃).
Mutation guards: `normalizer_cap_down` (consuming the band floor 2.3066 — wrong
direction — fails), `ceiling_consumed_above_U` (consuming an inflated 3.7 or
the cell-2 value 3.66435028046302 — wrong value/wrong cell — fails), both
fail-closed in both modes.

## Carried from v3 (unchanged; re-derived by this program)

Lever 1 (density floor m = 9/10000 from the exact quadratic-form minimum over
the 16 box vertices; directed floor ≈ 9.1044e-4), lever 2 (δ = 14587/2621440
with the ε′ = 57/640 split; clearance margin exactly 1/3840; weight floor
W1·W2 = (597/128)·(2596849/204800)), B₃ = 2082000000000 (now a consistency
anchor only), K = 8690 (sets r₀), amplitude majorant (E[R⁴] = 8 exact,
E[R] ≤ √2), S₃/S₄ from the certified library (554.3913563 / 2041.8327332,
F-S4 on record), endpoint Gram/Schur/mean/eigenfloor re-derivations, the v1
Poisson defect 1 − a₂ = 9.65254179896e-123 hook, Fourier conventions and the
factor-of-two adversarial test.

## Carriers consumed (sha256 verified from bytes at run time)

v1 script/receipt/report/transcript/review; v2 script/report; H1 hardening
receipt; v3 script/receipt/report/errata (predecessor record, frozen);
H3 rung 6347275d86c5…, H3 band 7a40e267… (body 281477c3…), **H3 ceiling
18e109bc2e7a… (body cfe8a3a4…ede2), ceiling script b97c5428…, ceiling
transcript 26d08534…, ceiling receipt ee58ef80…**. Full values in RECEIPT.json.

## Certificate discipline

Exact rational arithmetic for the final coefficient; published decimal ≤ exact
fraction (invariant a); power-of-ten floor by exact integer tests (invariant b);
general rounding-direction checker over the display ledger (invariant c);
17 mutation families (v3's 16 + ceiling_consumed_above_U), each fail-closed
with nonzero exit in both modes; unknown mutations rejected fail-closed;
CANNOT-VERIFY (exit 2) separate from FAIL (exit 1); no bare asserts;
deterministic; both modes byte-identical program output (runner
labels/receipts separate).

Body hash of this document: recorded in MANIFEST.sha256 (SHA256 over the document
body excluding the hash line itself).
