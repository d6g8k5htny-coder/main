# H3 BAND FLOOR — certified uniform normalizer floor on the whole rung band

**Campaign:** U2D-UPPER (Stage E / OBL-D1-PROMOTE, normalizer side)
**Scope:** periodized side-24 2D Bargmann–Fock field, corrected pin basis (trapezoid), soft six-pack
Y = (q_M, q_S, α_M, α_S, β_M, β_S), D = αq − rβ².
**Deliverable of:** `h3_band_floor.py` (this directory), executed 2026-09-15, both interpreter modes.

---

## 1. Certified statement

For **every** r in the open band **(0, 0.05]**,

  **E[G_r] ≥ 2.30659559567154  >  c_Z = 1.615489267643502474048123284509081575382**

and therefore the normalizer satisfies

  **Z_r = r²·E[G_r] ≥ c_Z·r²  uniformly on (0, 0.05]**,   worst certified margin **+42.78%**

(attained on the rung-adjacent cell (0.04, 0.05]). This discharges the normalizer side of
OBL-D1-PROMOTE: the rung certificate `H3_RUNG_FLOOR.md` covered the point r = 0.05;
this certificate covers **all** r ∈ (0, 0.05] in one pass, with r as an interval through
the pipeline. Per-cell certified floors (interval lower endpoints, 15 digits):

| sub-interval       | certified E[G_r] ≥     | margin vs c_Z |
|--------------------|------------------------|---------------|
| (0, 0.0025]        | 2.58288222715895       | +59.88%       |
| (0.0025, 0.005]    | 2.58174158877406       | +59.81%       |
| (0.005, 0.01]      | 2.54243388604045       | +57.38%       |
| (0.01, 0.02]       | 2.45593199230367       | +52.02%       |
| (0.02, 0.03]       | 2.44022641168325       | +51.05%       |
| (0.03, 0.04]       | 2.41792545239159       | +49.67%       |
| (0.04, 0.05]       | 2.30659559567154       | +42.78%       |

The bound is an interval enclosure: the true E[G_r] lies in [lo, hi] on each cell
(hi's are emitted in the transcript; only lo's are load-bearing).

## 2. Why the whitened/box pipeline cannot do interval-r (recorded negative result)

Carrying r as an interval through the Stage-E whitened pipeline fails structurally:
(i) the LDL whitened pivots D₂ ~ c·r² ≈ 4.2e-4 and D₃ ~ 2.1e-7 are destroyed by any
interval enclosure of the Gram (widths ≫ pivots at any practical sub-interval size);
(ii) naive interval-coefficient evaluation of the divided-difference pin functionals
gives catastrophic cancellation (measured Gram-entry width 3.3e7 on [0.049, 0.05]);
(iii) crude derivative/Lipschitz bounds are λ²-amplified (‖G⁻¹‖ ~ 25.6).
The band certificate therefore uses a **no-whitening Wick architecture** plus an
**exact Taylor-series device** for the conditional law (below); no factorization,
no whitening, no Monte Carlo in the certificate.

## 3. Method (four certified blocks + cross-checks)

**CB1 — exact finite-torus moments.** a_{2m} (m ≤ 40) by the Poisson–Hermite identity
(interval Hermite recurrence, |n| ≤ 3 lattice + certified geometric tail); certified
separated deviations 0 < 1 − a₂ = 9.6525417989599e-123 ±, 0 < a₄ − 3 < 1e-100
(summandwise-positive Poisson path). Planar moments are forbidden fail-closed (mutation
guard CB1a).

**CB2 — exact Laurent-series Gram algebra.** All entries of the pin Gram G, soft Gram
GY and cross Gram GYV as Laurent series in r (window [z⁻⁸, z⁶²]) with interval
coefficients built from the certified moments via the kernel Taylor series
K1⁽ⁿ⁾(z) = Σ_j (−1)^{(n+j)/2} a_{n+j} z^j/j! and ce.cov's exact sign convention
(no s-flip; verified against cov_exact source). The corrected-basis poles cancel
**exactly**: G regular; GYV regular below z⁻¹; GY below z⁻² (genuine soft poles
z⁻¹, z⁻² isolated — the soft Gram is singular at r = 0 by construction; only the
conditional law is regular).

**CB3 — exact series division.** H = G⁻¹ and W = G⁻¹z_V as Taylor series by the
recurrences H_j = −G₀⁻¹Σ_{i≤j}G_iH_{j−i}, and μ(r) = GYV·W,
Σ(r) = GY − GYV·H·GYVᵀ as series to order J = 60, **including the pole-index terms**
GYV₋₁·W_{j+1} etc. The genuine poles are annihilated by the conditioning: the z⁻¹
coefficient of μ and the z⁻², z⁻¹ coefficients of Σ are ck'd to vanish (< 1e-50).
Limit identities at r = 0 match the independent blow-up analysis: μ_q(0) = −6/5,
μ_αM(0) = −1, μ_αS(0) = +1, Var(α_M)(0) = 0, Var(q)(0) = 2 (wrong-window mutation
guard at the α-identity). Rotated law (q_M, δ = q_S − q_M, …) built at **series level**
so the var_δ = Σ₀₀ − 2Σ₀₁ + Σ₁₁ cancellation is exact in the coefficients.

**CB4 — certified tails.** Literal triangle (literal-curse) entry maxima on the circle
|z| = ρ = 0.11: M_G = 2.96e6, M_GY = 3.44e2, M_GYV = 1.95e4 (valid sup bounds of the
analytic kernel expressions — Gram entries are entire/pole-cancelled; Laurent-coefficient
Cauchy bounds |g_j| ≤ M(ρ)/ρ^j hold for all j). Recursion-consistent R-majorants for
W, H with R = 30.4545, θ = 36·‖G₀⁻¹‖(S_G(R) + T_G(R)) = 0.485 ≤ 1/2 — used **only** on
the first cell (R·0.0025 = 0.076 < 1): series tails there Σ-tail 4.68e-62, μ-tail 6.39e-66.
On the six cells with r₁ > 0 the truncation error is certified by the Neumann identity
G⁻¹ − G⁻¹_partial = G⁻¹(G − G_partial)G⁻¹_partial with ‖G⁻¹‖ Neumann-certified per cell
(contraction ρ < 1/2 ck'd) and the Gram Cauchy tail (~1e-21 at the rung) — the
pole-cancelling heavy part is the exact series, the correction is a norm bound; the
literal-curse constant is crushed by (r/ρ)^63. Correction widths T_μ, T_Σ < 1e-2 ck'd
per cell (actual ≪).

**CB5–CB6 — band evaluation.** On each cell, with the enclosed (μ(r), Σ(r)):

  E[G_r] ≥ E[p·1_{q_M∈[−3.9,−0.4]}] − E[p̄·1_{|δ|>0.3}] − E[p̄·1_{|β_M|>2}]
           − E[p̄·1_{α_M>−0.8}] − E[p̄·1_{α_S<0.8}],

p = −D_MD_S, p̄ = ((α_M²q_M²+1)/2 + rβ_M²)((α_S²q_S²+1)/2 + rβ_S²) ≥ |p| (|x| ≤ (x²+1)/2).
Type margins (law-independent, r-uniform): D_M ≥ 0.8·0.4 − 0.05·4 = 0.12 > 0,
D_S ≤ 0.8·(−0.1) < 0, q_S ≤ −0.1 — ck'd per cell, so the box certifies the type event.
Pieces: main piece by conditional-Wick polynomial algebra (conditional expectation as
polynomial in the pinned coordinate; 1D truncated-normal moments by the exact recurrence;
certified Φ: power series |x| ≤ 6 with geometric-tail certificate (ratio < 0.9, k ≥ 21,
straddles split by monotonicity), Gordon–Mills φ(x)(1/x − 1/x³) < Q(x) < φ(x)/x for
|x| > 6). The rare-event pieces (δ, α_M, α_S) use the **exact Gaussian tail** (these
variables are exactly Gaussian) with Cauchy–Schwarz: E[p̄1_A] ≤ √(E[p̄²]·P(A)),
P(A) = Q(t₀) certified by the same Φ machinery — at the rung t₀ = 9.7 (α) and 3.63 (δ);
below r = 0.03 all three are < 1e-2 and rapidly → 0 as r → 0, which is what makes the
uniform bound reach r = 0. The β_M piece (2.8σ, not rare) uses the exact truncated
integral everywhere.

**CB7 — independent cross-checks.** At r = 0.05 the Neumann-enclosed (μ, Σ) contain the
independent direct point law (ce.cov + mp Gauss solve, no series machinery) — containment
ck'd entrywise. MC diagnostics (numpy, fixed seed 90210, 4e5 samples, **labeled
non-load-bearing**): E[G_0.05] = 3.21876 ± 0.00709, E[G_0.01] = 3.23339 ± 0.00709, both
≥ the certified floor (consistent; the certificate does not use them).

**CB8 — discipline.** Fail-closed ck → SystemExit (no bare asserts); both modes
(`python3` and `python3 -O`) byte-identical transcripts; mutation suite:
MUTATION=planar_moments exits 1 at CB1a (exact-torus moment guard),
MUTATION=wrong_window exits 1 at CB3 (μ_αM(0) = −1 window identity).

## 4. Remainder ledger

| item | value |
|---|---|
| series order kept exactly | J = 60 (W, H to J+2 for pole-index terms) |
| moments | a_{2m}, m ≤ 40, exact finite torus (never planar) |
| majorant circle | ρ = 0.11; literal entry maxima 2.96e6 / 3.44e2 / 1.95e4 |
| R-majorant (W, H) | R = 30.4545, θ = 0.485; used only on (0, 0.0025] (Rr = 0.076) |
| series tails on (0, 0.0025] | Σ-tail 4.68e-62, μ-tail 6.39e-66 |
| cells 2–7 correction | Neumann+Cauchy, T_μ, T_Σ < 1e-2 ck'd (actual ≪; Gram tail ~1e-21 at rung) |
| G₀ inverse residual | 6.07e-98 |
| Φ machinery | series |x| ≤ 6 (certified tail), Gordon–Mills |x| > 6, straddle split |
| sub-intervals | 7 cells: (0,.0025], (.0025,.005], (.005,.01], (.01,.02], (.02,.03], (.03,.04], (.04,.05] |
| worst cell | (0.04, 0.05]: lo 2.30659559567154, margin +42.78% |
| MC diagnostic | consistent, explicitly non-load-bearing |

## 5. Recorded erratum (for the next amendment; no re-issue of frozen files)

Per the Stage-E numerics re-review of `h3_rung_floor.py` (verified fully otherwise:
both modes byte-identical, mutations caught, dps=150 spot-check identical to all 17
digits): **CR5f computes `aS_rng.b·qS_rng.b` rather than the sharp sup
`aS_rng.a·qS_rng.b`** — the ck remains SOUND (D_S < 0 follows from CR5b/CR5d:
α_S > 0 > q_S ⇒ α_S·q_S < 0), so it is a mislabeled intermediate only. Action for the
next amendment: correct the label or the formula; **do not touch the frozen
`H3_RUNG_FLOOR.md` (6347275d…)**. This band certificate is independent of that label
(its D_S < 0 margin is the law-independent 0.8·(−0.1) argument of §3/CB6).

## 6. Honest scope and grade

- The certificate covers **r ∈ (0, 0.05]** only — the H3 rung band. It says nothing
  about r > 0.05 (not needed for the rung route) and does not replace the frozen
  point certificates (H3_CLOSURE.md, H3_RUNG_FLOOR.md), which remain untouched.
- The statement is the normalizer floor used by D1 v2.1's C_RN and the D3 v2 κ_far
  denominator, now uniformly in r: Z_r ≥ c_Z·r² for all r ∈ (0, 0.05].
- The dominant losses (honest accounting): the β_M truncated piece (~0.08–0.25 per
  cell, exact integral), the δ Chernoff piece at the top cell (0.088), interval
  enclosure widths of lo (~0.01–0.65 total spread; see transcript [lo, hi]).
  The true E[G_r] ≈ 3.2 (MC) is nearly flat on the band; the certified floor's
  dip to 2.31 at the top cell is enclosure loss, not physics.
- The reconnaissance complex scan (min |det G(z)| ≥ 5.3e4 on |z| ≤ 0.11) motivated the
  design but is **not** cited by the certificate; all tails are literal-bound certified.

**Grade: A** for the stated scope (uniform certified normalizer floor on (0, 0.05],
fail-closed, mutation-tested, both-modes byte-identical, MC-labeled).

## 7. Artifacts and hashes

- `h3_band_floor.py` — the certificate program (sha256 a907eeedd767b0b97461df8e237cdca130a607b5dede50ad5ba4b05c957d22a5)
- `band_normal.txt` / `band_O.txt` — transcripts, **byte-identical** (sha256 dcb3c8e6152876021c8b116903c9a7c1ac6be7417a9b4042b89c00b911fe565d)
- `band_out.txt` — hashed transcript body (sha256 617b2d07fcaef23313427e68020182d0b27cad78032b399336a1b925db489f54, matches the in-transcript self-hash line)
- `band_mut_planar.txt`, `band_mut_window.txt` — mutation runs (both exit 1 at the designed guards)
- MANIFEST.sha256 / RECEIPT_h3.json — regenerated to include the above; frozen entries unchanged.

Body hash of this document: recorded in MANIFEST.sha256 (document body excluding the hash line itself).
