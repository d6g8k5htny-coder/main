# W4 INDEPENDENT WP IMPLEMENTATION — SIDE24/q0 lower-rate repair (workstream W4)

**Status: FROZEN (pre-comparison).** All results below were produced and frozen
before any consumption of KIMI-line WP conclusions (KIMI-DER-025,
verify_wp_witness_v1.py, THM-023 drafts). Method: wrapped-kernel real-space
representation + direct multivariate Gaussian integration on transformed
coordinates with closed-form innermost integrals + Gauss–Legendre/Tanh–Sinh
quadrature + IS Monte Carlo cross-check. **No spectral-lattice/Gauss–Hermite
stack anywhere.**

## 1. Model and pipeline (from scratch)

- Kernel: `K(x,y) = k1(x1−y1) k1(x2−y2)`, `k1(s) = θ(s)/θ(0)`,
  `θ(s) = Σ_n exp(−(s+24n)²/2)` (real-space wrapped/theta form).
  Derivatives via probabilists' Hermite polynomials, exact.
- Kernel verification (`transcripts/kernel_verify.txt`):
  * wrapped form vs the spectral-lattice closed form agree to **2.3e-61**
    at 7 test points (spectral tail certified < 1e-50);
  * certified |n|≥2 wrap remainder < 3e-255 for m ≤ 6, |s| ≤ 13;
  * production mode keeps only n = 0 with certified |n|≥1 remainder < 1e-40
    (|s| ≤ 4 enforced fail-closed); standard BF moments exact (Var f = 1,
    Var ∂i = 1, Var ∂ii = 3, Cov(f,∂ii) = −1, ...).
- Conditioning: joint 15-dim Gaussian (6-jet at y + 9 pins), Schur complement
  in mpmath dps=60; every matrix inverse residual-checked (fail-closed ck();
  no bare asserts; byte-identical output under `python3 -O`).
- Pins: M = (−0.0125, 0), S = (0.0125, 0), Y = (−0.0315, 0.006);
  f(M) = 1.2, f(S) = 1.2 − ℓ, f(Y) = μ_t, gradients 0;
  ℓ = r³/6 = 2.6041666666666666667e-6 (r = 0.025).
- **μ_t = E[f(Y) | (f,∇f)(M), (f,∇f)(S), ∇f(Y)=0]
  = 1.199998698656492779037679** (dps=60; conditional variance of f(Y) given
  the other 8 conditions: 3.04e-14). μ_t ∈ (b−ℓ, b) = (1.1999973958333…, 1.2).
- Estimand per y: `rho(y) = p_{∇f̃(y)}(0) · E_win(y)`,
  `E_win = E[|det H| 1{det H<0} 1{b−ℓ<f̃<b} | ∇f̃(y)=0]`.
  E_win evaluated by nested quadrature on whitened/transformed coordinates:
  u = f outer (window), z = H12 middle, w = (H11−m1)/s1 inner with the
  innermost integral in closed form (`K(v) = vΦ(v)+φ(v)` branch-stable).
  Truncations at ±10σ with certified Gaussian tails (< 1e-22 per point).

## 2. Pointwise validation of E_win (probe y = (0, 0.5672), near the peak)

| method | E_win | P(det<0 ∩ window) |
|---|---|---|
| primary GL nested (24/48/64) | 2.020901410980e-05 | 3.218939917996e-05 |
| same, orders doubled (to 64/128/160) | identical to 1e-15 | identical to 1e-15 |
| independent (X,Y)-outer decomposition, Z closed form (`w4_altcheck.py`) | 2.020901410980311e-05 | 3.218939917991254e-05 |
| mpmath Tanh–Sinh, independent nodes/precision, dps=25, h=0.05 (`w4_refcheck.py`) | 2.0209014070765537809e-05 | 3.2189399118384995439e-05 |
| IS Monte Carlo, N = 2e6, seed 20260805 (`w4_mc.py`) | 2.174e-05 ± 2.4e-06 (2sd) | 3.490e-05 ± 3.8e-06 (2sd) |

The two deterministic float64 decompositions agree to 11–13 digits; the
arbitrary-precision Tanh–Sinh run agrees to 9 significant digits
(2e-9 relative, its own node-resolution floor); MC agrees within 1.4 sd.
(An early mpmath TS run at coarse step h = 0.09 was found to
under-resolve the φ peak by 2e-4 and was superseded; the incident is recorded
in `transcripts/refcheck_peak.txt`.)

Same battery at P2 = (0.1254, 0.7114) (`transcripts/altcheck_p2.txt`,
`refcheck_p2.txt`, `mc_p2.txt`):

| method | E_win | P(det<0 ∩ window) |
|---|---|---|
| primary GL (converged 1e-15 across orders) | 3.6418143078307e-05 | 4.33181483193e-05 |
| (X,Y)-outer alt | 3.6418142485358e-05 | 4.3318148319832e-05 |
| mpmath TS dps=25 h=0.06 | 3.6418143632584e-05 | 4.3318149114031e-05 |
| IS-MC N=2e6 | 4.116e-05 ± 7.4e-06 (2sd) | 5.082e-05 ± 8.9e-06 (2sd) |

All three deterministic methods agree to ≤ 1.5e-8 relative at P2 (the alt's
closed-form tail branches lose a few digits to cancellation in the far-tail
regime; the primary's K(v) = vΦ(v)+φ(v) ≥ 0 formulation is cancellation-free;
the mpmath TS has its own node floor). Assigned pointwise quadrature
accuracy: ± 1e-8 relative — negligible against the spatial-assembly terms.

## 3. Dominant region (found independently, not assumed)

The integrand is concentrated in a **narrow angular wedge around the
+Y-transverse direction (θ ≈ 90° from the pin cluster, i.e., y ≈ (0, +0.57)),
at radii r ≈ 0.41–1.55**, peaking at:

- **ρ_max = 1.0729e-4 at y = (−0.0025, 0.580)** (θ = 90.25°, r = 0.580,
  slightly toward the Y side off the transverse axis; plateau
  ρ ≈ 1.05–1.07e-4 over |x| ≲ 0.01, |r − 0.58| ≲ 0.03).

Structure:
- angular profile at r = 0.567: ρ(90°) = 1.05e-4; ρ(80°) = 6.6e-6;
  ρ(70°) = 1.6e-9; ρ ≤ 2e-12 outside [65°, 115°];
- radial profile on θ = 90°: 1.8e-6 @ 0.41; 1.05e-4 @ 0.567; 2.6e-5 @ 0.83;
  5.3e-7 @ 1.50 — a long decaying tail to the Z boundary;
- secondary arcs at θ ∈ [140°, 200°] and [220°, 300°], r ∈ [0.8, 1.55],
  ρ ≲ 2e-7 (included);
- **pin collars are dead**: exact evaluations on rings r = 0.002…0.05 about
  the cluster give ρ ≤ 1.1e-171; the local collar exponent κ(θ) ≥ 466 at all
  three pins (e^{-466} regime) — the 9-pin cluster rigidifies H so much
  (Var(δH|pins) ~ 6e-8…1e-4) that ∇f̃(y) = Ĥ·d + δH·d has no near-pin zeros
  beyond the pins themselves, and the window/saddle condition is killed too;
- **strong up/down asymmetry**: at the mirror point (0, −0.567),
  E_win = 1.8e-20 (vs 2.0e-5 up). The Y pin (arch side, y2 = +0.006) makes
  the conditional f–H correlation flip the saddle-in-window probability:
  conditioning f into the window shifts E[H|f=u,∇f=0] by β(u−m_f) with
  β ≈ (−1.8, −4.9, −34.8) at the peak, driving det H < 0 a.s. on the up side
  and det H > 0 a.s. on the down side;
- KIMI witness point (−0.04, −0.58): **exact ρ = 7.2e-21** — the witness is
  NOT in the dominant region (corrected-CS value there: 5.4e-10).

## 4. I_WP(0.025) — point ESTIMATE (not an enclosure)

**I_WP(0.025) = 4.7569e-6  (self-assessed accuracy ± 2e-8, i.e. ± 0.4 %;
this is an ESTIMATE, not an enclosure).**

Assembly (see `transcripts/assembly.txt`, Z = union of 1.5-disks about the
three pins; polar integration about the cluster centroid with the
Z-boundary sliver correction):

| piece | value |
|---|---|
| main wedge, fine grid (θ∈[45°,135°], r∈[0.10,1.55], 0.25°×0.01) | 4.689702198e-6 |
| arc A (θ∈[140°,200°], r∈[0.80,1.54], 0.5°×0.02) | 1.7150131e-8 |
| arc B (θ∈[220°,300°], r∈[0.80,1.54], 0.5°×0.02) | 7.4825655e-8 |
| remainder (medium grid 5°×0.025 outside fine domains) | 1.6324777e-8 |
| Z-boundary sliver subtraction | −4.1085103e-8 |
| inner disk r ≤ 0.05 | < 1e-173 (exact evals) |
| **total** | **4.7569177e-6** |

Accuracy budget (all terms absolute):
- kernel representation: < 1e-40 (CERTIFIED wrap tails; spectral-vs-wrapped
  agreement 2.3e-61) — negligible;
- conditioning linear algebra (mpmath dps=60, inverse-residual-checked):
  < 1e-35 — negligible;
- pointwise E_win quadrature: converged to 1e-15 under order doubling;
  cross-method agreement 1.4e-13 rel (GL vs (X,Y)-outer) and 2e-9 rel
  (vs mpmath TS); certified truncation tails < 1e-22/pt → assigned
  ± 1e-8 relative → ± 5e-14;
- spatial quadrature, main wedge: |fine − mid| (common domain) = 1.17e-11;
  |finer − fine| on core subdomain = 4.4e-11 (1.0e-5 rel) → ± 5e-11;
- arcs: same convergence class → ± 1e-12;
- remainder domain: 1.63e-8 with assigned ± 50 % → ± 8e-9;
- Z-boundary sliver: 4.11e-8 with assigned ± 20 % → ± 8e-9;
- inner disk: < 1e-173.
- **total assigned: ± 2e-8** (dominated by remainder + sliver).

Consistency checks: the medium 5° grid over the whole annulus alone gives
I = 4.8012e-6; minus sliver = 4.76e-6, agreeing with the assembled value to
3e-9. Cross-check of the wedge integral on the coarse medium-grid
restriction vs the fine grid: 4.689666e-6 vs 4.689702e-6 (7.7e-9 rel).

**Outer shell r ∈ [1.55, 2.50] (excluded from Z): I = 1.5866e-6** — about
25 % of the in-Z mass lies just outside the rigidity-zone boundary; reported
for context (cf. the C030 transition-zone count ~1.9e-6).

**Freeze/comparison disclosure.** The intake file
`K3_SIDE24_LB/00_READ_FIRST.md` mentioned an existing estimate
"I_true ≈ 4.76e-6". That number was never read during computation and no
KIMI-line script, intermediate value, kernel code, or grid geometry was
consulted; every number above was produced by the W4 scripts from the model
definition alone and frozen here before comparison.

## 5. Probe conditional laws (deliverable 4, 25 digits)

File `probes_laws.txt`: full μ6, Σ6 (law of (f, ∇f, H) | 9 pins), p_grad(0),
and (m4, S4) (law of (f, H) | pins, ∇f(y)=0) at:
- P1 = (0.0, 0.5672) (essentially the peak),
- P2 = (0.1254, 0.7114) (θ = 80°, mid-wedge).

## 6. Corrected Cauchy–Schwarz bound (with the square root on the probability)

Pointwise valid bound
`ρ(y) ≤ ρ_CS(y) = p_grad(0)·√(E[det H²|∇f̃=0])·√(P(det<0 ∩ window|∇f̃=0))`
evaluated on all grids (column rho_CS in every maps/*.tsv).

**Integrated corrected-CS bound: I_CS = 8.5743e-4** (pieces: wedge
8.5441e-4, arcs 1.1585e-6, remainder 1.7287e-6, sliver −1.87e-8).
Ratio I_CS / I_WP ≈ 180. Pointwise:
- at the peak (0, 0.5672): rho_CS = 6.319e-3 vs rho = 1.053e-4 (ratio 60);
- at the argmax (−0.0025, 0.580): rho_CS = 5.283e-3 vs rho = 1.073e-4;
- at KIMI's witness (−0.04, −0.58): rho_CS = 5.354e-10 vs exact
  rho = 7.163e-21 — the witness sits far outside the dominant region;
  no pointwise statement there bears on I_WP.

## 7. Estimate vs enclosure statement

CERTIFIED (enclosure-grade):
- kernel wrap tails (< 1e-40 production; spectral-vs-wrapped agreement 2e-61);
- conditioning residuals (< 1e-35);
- per-point quadrature truncation tails (< 1e-22·moment bound);
- inner disk r ≤ 0.05 kill (exact node evaluations ≤ 1.1e-171; collar κ ≥ 466);
- pointwise inequality ρ ≤ p_grad·√(E[det²]) used for cheap bounds;
- μ_t to 25 digits.

ESTIMATE (not enclosed):
- I_WP(0.025): spatial quadrature error estimated by refinement
  (mid/fine/finer), NOT interval-enclosed;
- the remainder-domain (coarse-grid) contribution: small share, ~50%
  relative uncertainty assigned;
- Z-boundary sliver correction: interpolation-based;
- MC: statistical only.

## 8. Reproduction

Scripts: w4_kernel.py, w4_condlaw.py, w4_rho.py, w4_refcheck.py,
w4_altcheck.py, w4_mc.py, w4_map.py, w4_integrate.py, w4_probes.py,
w4_assemble.py. Transcripts in transcripts/. Maps in maps/.
All programs: fail-closed ck(), no bare asserts, deterministic,
byte-identical output under python3 and python3 -O.
