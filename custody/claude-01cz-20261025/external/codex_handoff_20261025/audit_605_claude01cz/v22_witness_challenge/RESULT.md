# v22 witness/axial applicability challenge (Claude 01Cz, 25 October 2026)

Pickup: main#605 comment 6400684672. Scientific effect: NONE. No status changes.

## Inputs, verified by hash before reading

C318's eight members, extracted from main#605 comments 6400186679, 6400186900 and 6400187065:

| Member | Bytes | SHA-256 |
|---|---:|---|
| S01 WITNESS_COLLISION_CANDIDATE.md | 17,811 | 197e15499582bdcd8556d55fd35ed0707c103b84eef2c5014cd5ccf37ad63bc3 |
| S02 AXIAL_EXTENSION.md | 16,585 | 7affe4ee13dac7c7d2cb8ff918a11c849e4114ecaed77c5e35db7b96ebbdedea |
| S03 WITNESS_COLLISION_REVIEW.md | 10,925 | 78d26ea7db2a… |
| S04 AXIAL_EXTENSION_REVIEW.md | 14,664 | e170a64602bb… |
| S05 CONFLUENT_COVARIANCE_FLOOR.md | 6,437 | 96b6e9768948… |
| S06 GAUSSIAN_DERIVATIONS.md | 27,912 | 946333fcb798… |
| S07 AXIAL_SOURCE_LEDGER.json | 2,655 | 9325c189fdf2… |
| S08 AXIAL_REPAIR_CHANGELOG.json | 338 | 9763e9ad2a0b… |

Comparison sources:
- `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` (sha256 9350ad6e…, Math- 9f3fb691), the D1 parent;
- the canonical dataset records D4-COLLISION, D5-*, D5d-* and D5-ANN-R5 (396,561 bytes, sha256 4cfae309…).

## 1. Algebra re-derived (isolated sympy 1.14.0; scripts and outputs beside this file)

All 14 checks are True.

| Check | Item | Result |
|---|---|---|
| S01 (8) | signed row Jacobian `det(T→S) = −12/(r¹² ε² τ ν)` | recomputed from the ten raw observations in the order (4) |
| S01 (10) | cubic limits; the a_ε formula | match the divided Taylor limits |
| S01 (11) | cubic minor = τνvc/4 | match |
| S01 (14) | targets of A1, M1, N, G | match |
| S01 (18)–(19) | K_H cubic part vanishes at v = 0; target −3kq(1−ε²q²)/τ² | match |
| S01 (21)–(22) | K_R cubic part = νd·J21; target formula | match |
| S02 (4) | axis-safe identity for G | match |
| S02 (8) | [Q_H]₁₃ = qτ/6 + vL₁₃ and [Q_R]₁₃ = vν²/6 | computed from a full degree-5 scalar jet, not from the stated limits |
| S02 (9)–(10) | F_R affine in ε; ∂_εF_R formula; zero at d = 0 exactly on εγ = −2 | match |
| S02 §4 | radial minor τ²v/4; horizontal minor τνc/2; (13) | match |

By hand I also checked:
- the r and ε power ledgers in S01 (28)–(35) and S02 (16), (20)–(25);
- the radial-gap arithmetic in S01 (22)–(23) and S02 (12) (ratio ≤ 0.2424);
- the horizontal-strip bound |[Q_H]₁₃| > h/24;
- the dyadic shell sums.

I found no failing step. This agrees with S03 and S04, and does not repeat their review of the axial wording repair.

## 2. What composes cleanly (positive applicability)

- **Law.** D1's image-sum covariance K_L(z) = Σ exp(−|z+Ln|²/2)/Σ exp(−|Ln|²/2) is the same law as the packet's Fourier form with a_n ∝ exp(−ω²|n|²/2), by the identification in S06 (line 26). Both are normalised to variance one.
- **Pins and targets.** The packet's six pin rows and targets (b − kr³/2, −3kr²/2, 0, 12k, 0, 0) are the coordinates D1 §13 (13.1) uses.
- **Normaliser.** D1's typed weight W_r = |det H_M det H_S|·1{H_M<0, index = d−1} satisfies the packet's hypothesis 0 ≤ W_r ≤ |det H(a) det H(c)|. D1 (5.5) gives 0 < z_* ≤ Z_r/r² on compact mark windows, and in d = 2 so does D5-ANN-R5. So the packet's "divide by a same-weight floor ⇒ O(r³)" step is available, in d = 2, on compact windows B×K, from already-recorded sources.
- **Count identity.** For distinct witness sites, S01 supplies a nonsingular joint density for ε > 0 (row determinant (8) and finite-jet positivity), and the diagonal is spatially null. The two-witness factorial Kac–Rice/Borel-mark interface (the analogue of REPAIR §9 for one extra mark) remains to be written. I saw no obstruction to it.

## 3. Applicability findings

**V22-DIM-01 (INFO, scope).** The packet is planar: n ∈ Z², R ∈ O(2), scalar transverse coordinate τ. It therefore feeds the planar D5 records, which state d = 2. It does not apply to:
- the every-dimension records D5d-*;
- D1's general-d theorems;
- D1's stated SIDE24 application (d = 3, L = 24).

A d = 3 use would need a new witness geometry: a two-dimensional transverse plane, so more rows and a different cubic minor.

**V22-BAND-01 (LOW, the smallest remaining regional gap).** The packet bounds witness–witness collisions only for cluster centres with |ξ| ≤ B (physical distance ≤ Br from the pair midpoint) and δ-separated from the pins. Its constants depend on B, with no stated growth rate. D4-COLLISION covers near-collisions only outside a fixed physical pin disk of radius ρ, in every fixed d. The band B < |ξ| < ρ/r is covered by neither. In that band the intermediate-shell records D5-I and D5d-I give first moments only, and D5d-M expressly claims no second-factorial upper bound. A regional or global second-factorial (multiple-witness) bound therefore needs either:
- B-growth tracking in this packet, enough to sum over dyadic shells up to ρ/r; or
- a separate intermediate-band collision estimate.

Clusters approaching a pin (δ → 0) are also open. S05's floor allows a pin–witness short edge, but the packet gives no estimate for that case.

**V22-K-01 (LOW, reuse of v21 OBL-K-01).** Every target gap in the packet is linear in k:
- S01 (19), (22);
- S02 (9), (19), the last via the target k(3/2 − 6s²).

So the only energy floor used is E ≥ c·k²/(H² + r²). Tracking the packet's own route:
- the off-axis shell sum Σ h⁻² exp(−ck²/h²) grows like k⁻²;
- the thin-set step needs roughly r ≪ k⁴;
- the axial sum Σ h⁻²⁴ exp(−ck²/h²) grows faster still.

This is a statement about the stated proof route, not a counterexample. Other energy contributions might rescue small k, but the packet does not use them. Consequence: the packet supports compact-window compositions such as D1 Theorem B or c_{B,K}. It cannot directly supply the k → 0 domination that an all-mark lifetime pushforward needs, such as D1 §13 (13.5) with cutoff k ≥ ℓ/r0³ (record D1-KR-DOM). Large k and unbounded b look harmless in the unnormalised product π_r·numerator, since the Gaussian factors decay, but they are not proved here either.

## 4. Verdict

**PASS_SCOPED** for the packet's algebra at its declared scope: d = 2, fixed L, compact b, k ∈ [k₋, k₊], |ξ| ≤ B, δ-separation from the pins.

The smallest unresolved applicability question for a regional planar collision bound is V22-BAND-01, the intermediate band B < |ξ| < ρ/r. V22-K-01 and V22-DIM-01 bound how far the result transfers. Nothing is accepted on the canonical record, and no status changes.
