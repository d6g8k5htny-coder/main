# H3 RUNG FLOOR — certified Z_{0.05} (STAGE E REPAIR R2)

**Date:** 2026-09-13. **Verifier:** H3-closure line. Companion to the frozen
`H3_CLOSURE.md` (untouched; this file adds the rung-level certificate that
H3's §5(b) honesty boundary had left unexecuted).

**RESULT (certified, interval-enclosed, fail-closed):**

    Z_{0.05} ∈ [ 7.7592917375327855e-3 , 1.1468646473404396e-2 ]

with `lo = 7.7592917375327855e-3 > 4.0387231691087558e-3 = c_Z·(0.05)²`
(the c_Z of record, c_Z = c_0/2 ≥ 1.615489267643502474048123284509081575382
from the H3 closure) — **certified margin +92.12%** (ck CR7d). The MC
diagnostic (labeled, non-load-bearing) gives Z_{0.05}/r² = 3.22596 ± 0.00365,
i.e. Z_{0.05} ≈ 8.065e-3, inside the certified interval.

This discharges the R2 finding: H5's `Z_lo = max(c_Z r², …)` at r = 0.05 and
H4-RN's `C_RN(0.05) ≤ 3.46` denominator now rest on a **certified** rung
evaluation, not on the existential-r₀ theorem.

## Method (no Monte Carlo in the certificate)

`Z_{0.05} = r²·E[G]`, `G = |D_M D_S|·1{type}`, `D = αq − rβ²`, under the exact
six-pin law Q_{0.05} (corrected pin basis V_r with the trapezoidal correction,
values (6/5, 0,0,0,0, −1/6)) on the exact periodized side-24 2D Bargmann–Fock
field. The conditional law of the soft variables
`Y = (q_M, q_S, α_M, α_S, β_M, β_S)` is exactly Gaussian N(μ, Σ); everything
below is interval arithmetic (mpmath iv, 100 dps) descending from the
H2-certified `cov_exact.py` iv path (certified spectral tails carried).

1. **CR0** Dependency integrity: sha256 of `cov_exact.py`, `pin_transform.py`
   against the H2 MANIFEST of record.
2. **CR1** Exact-torus moment discipline (mutation guard):
   `0 < 1−a₂ = 9.6525…e-123 < 1e-100` (Poisson identity),
   `0 < a₄−3 < 1e-100` (summandwise-positive Hermite series). Never planar.
3. **CR2** iv covariances of the six pin functionals V_r (exact rational
   coefficients at r = 1/20: 20, −20, 8000, −200) and the six soft
   functionals; entry interval widths ≤ 1.5e-90.
4. **CR3** Neumann-certified pin inverse: candidate B by Gauss–Jordan in mp
   arithmetic, then `ρ = ‖I − B·G‖_F ≤ 1.8e-89 < 1/2` (iv) ⇒ certified
   inverse with error ≤ 4.7e-88. Interval-enclosed regression:
   μ = (−1.2, −1.19998, −1.014577, +0.984590, 0, 0),
   sd = (1.41421, 1.41421, 0.020407, 0.020407, 0.70696, 0.70696).
5. **CR4** Interval LDLᵀ whitening Σ = LDLᵀ (all pivots certified positive,
   enclosure residual < 1e-50): `Y = μ + TU`, U iid N(0,1). Since μ, T are
   interval enclosures of the true values, every subsequent interval
   evaluation covers the true law.
6. **CR5** Safe box B with exact rational edges,
   `[−7,−8,−6,−8,−12/5,−8] × [18/25,2,6,8,12/5,8]`: interval evaluation
   certifies the type event everywhere on B:
   `q_M ≤ −0.1818`, `q_S ≤ −0.0417`, `α_M ≤ −0.8921`,
   `α_S ≥ +0.8585`, `D_M ≥ +0.01822`, `D_S ≤ −0.04630`
   (corner cross-check CR5e0 against native iv multiplication). Hence on B:
   M max ∧ S saddle, and `|D_MD_S| = −D_MD_S` exactly. Zero region
   Z1 = {U₁ ≥ 43/50}: `q_M ≥ +0.0162 > 0` certified ⇒ G = 0 on Z1.
7. **CR6–CR7** Exact box-truncated polynomial expectation:
   `E[∏U_i^{e_i}·1_B] = ∏∫_{lo_i}^{hi_i} u^{e_i}φ(u)du` by the recurrence
   with Φ from an in-program certified power series (geometric-tail
   certificate, ratio < 0.9). With the 210-monomial expansion of −D_MD_S:
   `E[−D_MD_S·1_B] = 3.103716695013114…` (interval width 6.1e-83).
   Upper side: `E[(D_MD_S)²] = 31.2272…` (3003 monomials, exact full
   Gaussian moments), `P(B^c∖Z1) ≤ 0.07050…`, Cauchy–Schwarz gives hi.
8. **CR8** MC diagnostic (labeled, fixed seed 424242): 3.22596 ± 0.00365 —
   inside the certified interval.

## Remainder ledger (every approximation and its status)

| Ingredient | Remainder/error | Status |
|---|---|---|
| spectral lattice sums (cov_exact iv) | certified tails inside intervals (widths ≤ 1.5e-90) | certified |
| pin inverse | Neumann certificate, error ≤ 4.7e-88 | certified |
| regression μ, Σ | interval enclosures (widths ≤ ~1e-83) | certified |
| LDLᵀ factor T | enclosure residual < 1e-50; pivots > 0 | certified |
| Φ on box edges | power series + geometric tail bound (ratio < 0.9), widths ~1e-80 | certified |
| box expectation | interval polynomial eval, width 6.1e-83 | certified |
| hi remainder | Cauchy–Schwarz `√(E[(DMDS)²])·√P(B^c∖Z1)` (bound, not estimate) | certified bound |
| MC table | diagnostic only | not load-bearing |

The only places the method loses against the true value are (i) the safe box
excludes ~42% of the type-event mass (lo is 3.7% below the MC value), and
(ii) the Cauchy–Schwarz hi is loose. Neither affects the floor.

## Mutation suite (fail-closed, as the H3 pattern)

- `MUTATION=planar_moments` → caught at CR1a (exit 1): `1−a₂ = 0` violates
  the strict exact-torus discipline.
- `MUTATION=wrong_window` (pin value −1/12 instead of −1/6) → caught at
  CR5e (exit 1): α_M halves, the safe box no longer certifies `D_M > 0`.

## Stretch: neighboring certified rungs (pointwise)

The same script certifies any exact-rational rung (`RUNG=p/q` env). Same box,
same machinery:

| rung | certified lo for Z_r | c_Z·r² | margin |
|---|---|---|---|
| 9/200 = 0.045 | 6.2865611483300006e-3 | 3.2713657669e-3 | +92.2% |
| **1/20 = 0.05** | **7.7592917375327855e-3** | **4.0387231691e-3** | **+92.1%** |
| 11/200 = 0.055 | 9.3860471051561076e-3 | 4.8868550346e-3 | +92.1% |

(At 0.045 the transcript's CR7d checks the *stronger* 0.05-rung bound
4.0387e-3 as well — it passes with +55.7% to spare.)

**Honest scope note:** these are pointwise certifications at rational rungs.
They do NOT certify the continuum `(0, 0.05]` (that is the explicit-modulus
extension of H3's r₀, which this rung result does not attempt). A natural
continuation exists — enclose r itself as an interval through the same
pipeline (the covariance entries are explicit analytic functions of r; the
box margins have ~1e-2 slack at 0.05), optionally with a few sub-boxes — but
it is **not executed here**; OBL-D1-PROMOTE should request it explicitly if
it needs a certified band rather than rung values.

## Reproduction and receipts

```
cd /mnt/agents/output/K3_SIDE24_LB/UPPER2D/H3_closure
python3 h3_rung_floor.py  > rung_normal.txt 2>&1   # exit 0, ALL_CHECKS_PASS
python3 -O h3_rung_floor.py > rung_O.txt 2>&1      # byte-identical
MUTATION=planar_moments python3 h3_rung_floor.py   # exit 1 (caught CR1a)
MUTATION=wrong_window   python3 h3_rung_floor.py   # exit 1 (caught CR5e)
RUNG=9/200  python3 h3_rung_floor.py               # stretch, exit 0
RUNG=11/200 python3 h3_rung_floor.py               # stretch, exit 0
```

Receipts: `rung_normal.txt`, `rung_O.txt`, `rung_mut_planar.txt`,
`rung_mut_window.txt`, `rung_045.txt`, `rung_055.txt`; hashes in
`MANIFEST.sha256`; summary receipt `RECEIPT_h3.json` (regenerated).
