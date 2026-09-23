# CORRECTION ADDENDUM to LEAD_SYMBOLIC_WP.md (red-team finding B2, 2026-08-05)

The frozen LEAD_SYMBOLIC_WP.md remains unchanged (transcripts are immutable). This addendum corrects
one display and every value that consumed it.

## The error

§3(A) displayed E[det²H] = M(00,22) − 2M(01,12) + M(11,11) with M(ij,kl) = C_{ij}C_{kl} + C_{ik}C_{jl} +
C_{il}C_{jk} and C = Σ_H + ννᵀ. That display is FALSE for nonzero means ν: Wick's theorem applies to
centered variables; the correct raw-fourth-moment expansion is

    E[Z_i Z_j Z_k Z_l] = ν_iν_jν_kν_l + ν_iν_jΣ_{kl} + ν_iν_kΣ_{jl} + ν_iν_lΣ_{jk}
                        + ν_jν_kΣ_{il} + ν_jν_lΣ_{ik} + ν_kν_lΣ_{ij}
                        + Σ_{ij}Σ_{kl} + Σ_{ik}Σ_{jl} + Σ_{il}Σ_{jk},

and the displayed formula overcounts by exactly 2·ν_iν_jν_kν_l per M-term, totaling exactly
2(det ννᵀ)² for E[det²H] (verified by counterexample: overcount = 2(ν_0ν_2 − ν_1²)² to 1e-9 against
both the correct expansion and Monte Carlo). The error is in the CONSERVATIVE direction everywhere it
was consumed (every D2-dependent bound is larger, hence remains a valid bound; margins shrink slightly).

## Corrected canonical values (r = 0.025, witness A = (−0.04, −0.58))

The red team reproduced the inflation factor exactly: D2_buggy/D2_correct = 1.34488, so the printed
absolutes scale by √1.34488 = 1.159689:
- defective (dropped-root) form: 1.31310e-4  → correct 1.13229e-4
- CS-valid upper bound:          2.30247e-2  → correct 1.98542e-2
- ratio (the D1 defect factor): 175.35 — UNCHANGED; the D1 quarantine is fully intact.

## Blast radius (checked)

- verify_wp_witness_v1.py / KIMI-DER-025: the printed D2 values (e.g., 0.63865194 at A) and everything
  consuming D2 (wp_exact/wp_peak/I_cs — already quarantined as invalid bounds from D1) are inflated;
  conservative direction. The two-sided truth I_true ≈ 4.7602e-6 is UNAFFECTED (direct Gauss–Hermite
  quadrature of the 3D conditional law, no Isserlis display). W4's numbers are likewise unaffected.
- LB-1's rho_CS/rho_sad (above-b channel): its D2 comes from the same display form — inflated; the
  bounds remain valid (conservative); the six-rung E/tol chain's 22+ orders of margin absorb the factor
  with no closure impact. Register note filed.
- LB-2 (mean-ridge): UNAFFECTED (its D2F/D3F are deterministic derivative bounds, not Gaussian moments).
- W2's Isserlis usage: red-team re-verified — HOLDS (its W2-9 form is correct as used).
- W6's modulus: its E[det²] was MC-validated (the red team confirms); the B1 break is unrelated.

## The corrected formula (for any future consumption)

E[det²H] = E4(0,0,2,2) − 2·E4(0,1,1,2) + E4(1,1,1,1), with E4 the raw fourth moment above.
Equivalently: center first, then E[det²] = Var-chain on the centered law + 2μ_det·E[det_c] + μ_det²
with μ_det = E[det H]. Any use of my §3(A) display must be replaced by this form.
