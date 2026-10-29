# LPW_CONSTANT v3 — SUCCESSOR CERTIFICATE REPORT (levers 1+2+4)

**Statement certified.** For the six-pin law (f(M) = b = 6/5, f(S) = b − r³/6,
∇f(M) = ∇f(S) = 0, M a local max, S a saddle) of the 2D torus Gaussian field,

  **1 − q(r, 6/5) ≥ c·r³  for all 0 < r ≤ r₀,**
  **c = 4·(597/128)·(2596849/204800)·(9/10000)·(14587/2621440)⁴ / 2082000000000
     = 210574711783915452081258399 / 2147825968160134776949980528640000000000000000000
     = 9.8040863135804911886149013570e-23  (published downward-safe 9.80e-23; power-of-ten floor 1e-23),**
  **r₀ = 1/2278031360 = 1/(2¹⁸·8690) ≈ 4.3898e-10.**

v2 (sha256 ca654e39…c130e125) and its reviewed theorem stay frozen untouched; v3
is a NEW successor. Every number below is produced by `lpw_constant_v3.py`
(python / python −O byte-identical program output; receipts separate).

## CHANGES table from v2

| Lever | v2 | v3 | certified effect | re-verification record |
|---|---|---|---|---|
| 1 — density floor | m ≥ 1e-21 (boxed triangle formula over JBOX) | m = 9/10000 (exact quadratic-form minimum of the conditional Gaussian density over the small box, max at the 16 vertices) | ×9.1044e17 (directed floor ≈ 9.1044e-4; consumed with > 1% margin; consistency: below the RB center density 1.2832254e-3) | endpoint ten-jet Gram re-derived from the moment dictionary; Schur identity Cov(J\|U₀) = diag(a₄−a₂², a₂(a₄−a₂²)/4, ·, (a₆−a₄²/a₂)/36) verified to 1e-50; μ₀ = (−a₂b,0,0,0) exact; eigenfloor 0.126553449667289 re-derived (≥ 31/250); log-det modulus det(S_r) ≤ det(S₀)(1+x+x²), x = 2L1R/λ; mean correction ε_μ = M − a₂b = 0.0225758 consumed from v1's frozen certified M; S_r⁻¹ perturbation L1R/(λ·λ₀) |
| 2 — δ widening | δ = 1/1024, r₀ = 1/(256·8690), budget 16δ + 8Kr₀ = 3/64, weight floor 65 | δ = 14587/2621440, r₀ = 1/2278031360, budget ε′ = 57/640 with clearance margin exactly 1/3840 > 0, weight floor 597/128 · 2596849/204800 = 1550318853/26214400 ≈ 59.13997 | δ⁴ ×1054.15, weight ×0.90985 → net ×959.96 | δ = 1/256 FAILS: 16/256 + 1/32 = 3/32 > 343/3840 (recorded; reversion applied). At ε′ = 57/640 every margin re-verified UNIFORMLY over the widened box (not cited from v2): typing signs (−(1−ε′) < 0, −(10−ε′) < 0), determinant floors W1 = 6−15ε′, W2 = 14−15ε′+2ε′² (exact Fractions), path clearance 1/6 − 99/1280 − ε′ = 1/3840 > 0, endpoint 9/16 − ε′ > 0, basis C²-norm sum 4+4+2+6 = 16 re-derived on D |
| 4 — two-sided normalizer | C_Z = 4B₃ = 8328000000000 (v1 crude uniform cap) | C_Z = 4B₃ (UNCHANGED on the certified line); H3 carriers consumed as consistency anchors | none certified; conditional gain ×1.815384e12 → c ≈ 1.78e-10 (ESTIMATE, non-load-bearing) | **F-LEVER4 (flagged insufficiency):** H3_RUNG_FLOOR certifies Z_{0.05}/0.05² ∈ [3.10371669501, 4.58745858936] POINTWISE at r = 0.05 > r₀; H3_BAND_FLOOR certifies the one-sided FLOOR E[G_r] ≥ 2.30659559567154 uniformly on (0, 0.05] ⊃ (0, r₀] (r₀ ∈ cell (0, 0.0025]). Neither is a uniform UPPER envelope of Z_r/r² on (0, r₀], which is what the Palm denominator requires. Consistency checks pass (band floor ≤ rung lo < rung hi ≪ 4B₃). Follow-up: a certified uniform upper envelope on (0, 0.0025] would realize the conditional gain; mutation family `normalizer_cap_down` fail-closes any premature consumption |

Combined certified gain over v2: c_v3/c_v2 ≈ 8.63e20 (levers 1+2). The work-order
target estimate ~8.4e-11 assumed lever 4's uniform cap; the certified value is
whatever the build proves: **9.80e-23** (levers 1+2), with the lever-4 conditional
estimate 1.78e-10 documented but NOT load-bearing.

## Carriers consumed (sha256 verified from bytes at run time)

v1 script e258322c…, v1 receipt 6332afe0…, v1 report 702788dc…, v1 transcript
d0cb9259…, v1 lead review 940057a5…; v2 script ca654e39…, v2 report a469c456…;
H1 hardening receipt 49e23661… (S₃ ≤ 554.3913563, S₄ ≤ 2041.8327332 amended
enclosures, findings F-S4/F-C29); H3 rung 6347275d…, H3 band 7a40e267… (body
hash 281477c3…, body-excl-hash-line rule verified). The v1 Poisson defect
1 − a₂ = 9.65254179896e-123 > 0 is consumed from the v1 receipt (strict
positivity hook); moments a₂, a₄, a₆ are independently recomputed with certified
geometric tails (interval width < 1e-38).

## Findings carried / new

- F-S4 (carried): S₄ upper enclosure amended to 2041.8327332; non-load-bearing.
- F-C29 (carried): exact expansions by integer division; published ≤ exact.
- F-DELTA1256 (new): δ = 1/256 fails the path clearance at v2's split; certified
  reversion = re-optimized split ε′ = 57/640, δ = 14587/2621440, margin 1/3840.
- F-LEVER4 (new, flagged): no certified uniform upper envelope of Z_r/r² on
  (0, r₀] exists in the package; C_Z = 4B₃ retained; conditional gain recorded.

## Certificate discipline

Exact rational arithmetic for the final coefficient; published decimal ≤ exact
fraction (invariant a); power-of-ten floor by exact integer tests (invariant b);
general rounding-direction checker over the display ledger (invariant c);
amplitude majorant E[R⁴] = 8, E[R] ≤ √2 (the v2 repair) enforced as the exact
multiplier; 16 mutation families (v2's twelve + density_floor_up, delta_up,
weight_floor_up, normalizer_cap_down), each fail-closed with nonzero exit in
both modes; unknown mutations rejected fail-closed; CANNOT-VERIFY (exit 2)
separate from FAIL (exit 1); no bare asserts; deterministic; both modes
byte-identical program output (runner labels/receipts separate).

Body hash of this document: recorded in MANIFEST.sha256 (SHA256 over the document
body excluding the hash line itself).
