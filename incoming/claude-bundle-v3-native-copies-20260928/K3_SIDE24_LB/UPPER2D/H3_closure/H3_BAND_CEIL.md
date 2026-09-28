# H3 BAND CEIL — certified uniform UPPER envelope of the normalizer on the rung band

**Campaign:** U2D-UPPER (LPW_CONSTANT v4 / OBL-D1-PROMOTE, Palm denominator side)
**Scope:** periodized side-24 2D Bargmann–Fock field, corrected pin basis, soft six-pack,
E[G_r] = Z_r/r². Companion to `H3_BAND_FLOOR.md` (lower side).
**Deliverable of:** `h3_band_ceil.py` (this directory), executed 2026-09-15, both interpreter modes.

---

## 1. Certified statement

For **every** r in **(0, 0.05]**, cellwise certified ceilings (interval upper endpoints,
15 digits), each ck'd to nest the certified band floor (≥ 2.30659559567154) and to meet
the consumption cap U ≤ 4 (cap NOT widened at any point):

| sub-interval       | certified E[G_r] ≤     | margin over MC ~3.22–3.23 (diagnostic) |
|--------------------|------------------------|----------------------------------------|
| **(0, 0.0025]**    | **3.66282864761194**   | +13.3%                                 |
| (0.0025, 0.005]    | 3.66435028046302       | +13.4%                                 |
| (0.005, 0.01]      | 3.67360595202419       | +13.6%                                 |
| (0.01, 0.02]       | 3.69805856644731       | +14.4%                                 |
| (0.02, 0.03]       | 3.71138266685765       | +14.8%                                 |
| (0.03, 0.04]       | 3.72414748332433       | +15.2%                                 |
| (0.04, 0.05]       | 3.74767948915996       | +16.4%                                 |

Uniform form: **E[G_r] ≤ 3.74767948915996 for all r ∈ (0, 0.05]**, and on the v4 domain
**(0, 0.0025]: E[G_r] ≤ 3.66282864761194** — this cell contains v3's r₀ = 1/2278031360
≈ 4.39e-10 and **discharges the LPW_CONSTANT v4 normalizer lever** (its owner's estimate:
4B₃/3.6629 ≈ ×2.274e12 → c ≈ 2.23e-10, now backed by this carrier).
MC point estimates (labeled diagnostic, non-load-bearing): 3.21876 ± 0.00709 (r = 0.05),
3.23339 ± 0.00709 (r = 0.01); both below the certified ceilings as required.
Combined with the floor: **2.30659559567154 ≤ E[G_r] ≤ 3.66282864761194 on (0, 0.0025]**
(cellwise sharper pairs in the two carriers' tables).

## 2. Method

Same certified law pipeline as the floor (exact finite-torus moments; Laurent-series Gram
algebra with exact pole cancellation; exact series division for μ(r), Σ(r) to J = 60 with
pole-index terms; per-cell Neumann+Cauchy certified corrections; R-majorant tails on the
first cell; certified Φ: series |x| ≤ 6 + Gordon–Mills |x| > 6). The upper-envelope
algebra:

  E[G_r] = E[p·1_type] = E[p·1_{q_M<0}] − E[p·1_{q_M<0, ¬type}]
         ≤ E[p·1_{q_M<0}] + E[p̄·1_{q_M<0, D_M≤0}] + E[p̄·1_{q_M<0, D_S≥0}]

with p = −D_MD_S, p̄ = ((α_M²q_M²+1)/2 + rβ_M²)((α_S²q_S²+1)/2 + rβ_S²) ≥ |p| pointwise
(so −E[p·1_A] ≤ E[p̄·1_A] for any A). The three pieces:

1. **E[p·1_{q_M<0}]** — three EXACT truncated windows (conditional-Wick + 1D truncated
   moments, two-sided interval enclosures): [−3.9, −0.4], (−0.4, 0), (−100, −3.9), plus a
   (−∞, −100) residue via Cauchy–Schwarz with the certified 70σ Gaussian tail (≈ 0).
2. **{D_M ≤ 0, q_M < 0}** ⊂ {α_M > −0.6} ∪ {|β_M| > 2} ∪ {q_M ∈ (−0.4, 0)} (cover margin
   0.6·0.4 = 0.24 > 0.05·4 = 0.20 ≥ rβ², ck'd): the α piece by exact-Gaussian-tail
   Chernoff–CS (12σ at the rung, → 0 as r → 0); the β and mid-q pieces exact.
3. **{D_S ≥ 0}** ⊂ {q_S ≥ 0} ∪ {α_S ≤ 0} (α_Sq_S ≥ rβ² ≥ 0 needs equal signs): the q_S
   piece exact (~0.36, the dominant type-fail correction — q_S crossing 0 is a 0.86σ
   event), the α_S piece Chernoff (≈ 0).

## 3. Direction analysis (symmetry / asymmetry, honest)

- **Direction-neutral (two-sided enclosures):** the moments, Gram/series algebra, law
  enclosures (μ, Σ), Φ machinery, and all truncated-normal integrals. The main p-windows
  certify BOTH directions simultaneously — the floor's main piece and the ceiling's
  p-windows are the same two-sided computation.
- **Lower side (floor):** lower-bounded the main piece and needed UPPER bounds on the four
  bad pieces (|δ| > η, |β_M| > 2, α_M > −0.8, α_S < 0.8) — the rare ones by Chernoff–CS,
  the β one exact.
- **Upper side (this certificate):** needs the type-fail correction bounded ABOVE, i.e.
  upper bounds on a different event family ({D_M ≤ 0}, {D_S ≥ 0} within q_M < 0). The
  asymmetry is in the event family, not the machinery: the floor's bad pieces exclude
  mass from a certified box; the ceiling's fail pieces re-include what the box exclusion
  dropped. The dominant ceiling-side looseness is E[p̄·1_{q_S≥0}] ≈ 0.36 (a genuinely
  non-rare event, so no tail argument can shrink it — only an exact joint computation
  would) and the p̄-vs-p slack inside it.
- **The naive p̄-only envelope is NOT viable (recorded):** E[p̄·1_{q_M<0}] = 9.55 at the
  rung (E[p̄] = 9.91 vs E[G] = 3.23 — the (x²+1)/2 ≥ |x| majorant is ~3× loose under
  fourth-moment weight). The shipped construction uses exact p on the main mass and
  confines p̄ to the fail pieces; that is the whole price of the upper side.
- **Investigation of the (0.03, 0.04] cell (lead directive 2):** its enclosure did NOT
  genuinely exceed 4. The failure was in the exact conditional-Wick path for the
  α-coordinate pieces: the conditional coefficients Sig[i][α]/s_αα are ratios of two
  narrow, strongly r-dependent intervals (s_αα ~ 0.166r²), and interval division loses
  the correlation, exploding the conditional moments (observed ±700 on [0.03, 0.04]).
  This is the same interval ratio-dependency noted in the floor's build log, not a tail
  or truncation degradation: the certified Gaussian-tail (Chernoff–CS) path is both
  VALID and strictly tighter there (t₀ = 24σ → piece ≈ 0). The cell now certifies
  3.72414748332433 < 4 with the same cap — no cap widening occurred.

## 4. Remainder ledger

| item | value |
|---|---|
| law pipeline | identical to H3_BAND_FLOOR (J = 60, ρ = 0.11, R = 30.4545 on cell 1 only) |
| p-windows (exact) | [−3.9,−0.4] 2.6678; (−0.4,0) 0.00494; (−100,−3.9) 0.5624 (cell-7 values) |
| fail pieces (cell-7) | α_M 1.9e-43; |β_M|>2 0.108; q_M∈(−0.4,0) 0.0272; q_S≥0 0.3677; α_S ~0 |
| residues | (−∞,−100), (100,∞) tails ≈ 0 (certified Mills, 20σ+ margins ck'd) |
| nesting ck | hi.b > 2.30659559567154 (band floor) on every cell |
| consumption cap | U ≤ 4 per cell, NOT widened; all 7 cells pass after the α-path fix |
| mutations | planar_moments → CB1a; wrong_window → CB3; cap_raised (p̄, p ×2) → CC6 consumption, all exit 1 |

## 5. Honest scope and grade

- Covers r ∈ (0, 0.05] cellwise; the v4 lever needs only cell 1 (0, 0.0025] — discharged
  at U = 3.66282864761194. Says nothing about r > 0.05.
- The ceiling is looser than the floor in relative terms (+13–16% over MC vs the floor's
  −28%) because of the q_S ≥ 0 fail piece and the p̄ slack inside it; tightening further
  would need an exact joint (q_S, ·) computation, not better tails. Sufficient for the
  ×1.8e12-lever consumption (the lever's sensitivity to U is 1/U).
- The cap_raised mutation is the consumption-side guard the lead required: any corruption
  inflating the envelope (here ×2) is caught fail-closed at the U ≤ 4 consumption ck.

**Grade: A** for the stated scope (uniform certified upper envelope, fail-closed,
mutation-tested incl. consumption guard, both-modes byte-identical, MC labeled).

## 6. Artifacts and hashes

- `h3_band_ceil.py` — certificate program (sha256 b97c5428a20e5dd33f3e53653d20447f1e27515dcc7dc7a7c2004ebe30029452)
- `ceil_normal.txt` / `ceil_O.txt` — transcripts, **byte-identical** (sha256 26d08534979083275fe14ce789d9cfc4f59ab9250a6cfc830e9acca80a14ae00)
- `ceil_out.txt` — hashed transcript body (sha256 1642236cf8e0a8346c17d00b345a9690733e64b941af741354aa9316c59eda5a, matches the in-transcript self-hash line)
- `ceil_mut_cap.txt`, `ceil_mut_planar.txt`, `ceil_mut_window.txt` — mutation runs (all exit 1 at the designed guards)
- MANIFEST.sha256 / RECEIPT_h3.json — regenerated; frozen entries unchanged.

Body hash of this document: recorded in MANIFEST.sha256 (document body excluding the hash line itself).
