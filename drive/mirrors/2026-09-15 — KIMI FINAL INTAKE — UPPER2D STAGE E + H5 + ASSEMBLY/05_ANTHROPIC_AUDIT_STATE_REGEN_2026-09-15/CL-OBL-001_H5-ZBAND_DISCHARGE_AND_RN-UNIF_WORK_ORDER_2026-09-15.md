# CL-OBL-001-v1.0 — OBL-H5-ZBAND DISCHARGE PROPOSAL + D3-LEMMA-RN-UNIF WORK ORDER, 2026-09-15

AUTHOR: Claude (Anthropic) · CREATED: 2026-09-15 · CLASS: OBL — obligation register action
STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE until the owning lanes consume/execute.
FALSIFICATION: §1 — any r ∈ (0, 0.05] at which Z_r/r² lies outside [floor, ceiling] as tabulated; §2 — a
  counterexample station with τ-uniformity violation (D3's own falsifier clause).
NOTE 2026-09-16: §1 has since been EXECUTED (H5_ZBAND/ — certificate + carrier + freeze); §2 and Appendix A are
  SUPERSEDED by CL-RNU-001 (an engine for the rung part already exists in the tree; see RN_UNIF_2026-09-16/).
  Retained as the record of the reasoning that led there.

---

## 1. OBL-H5-ZBAND — the required carriers now exist on both sides; discharge is a consumption step

**Obligation as frozen (H5_PROMOTE.md §, OBLIGATION_LEDGER §1):** "Z_r window bounds over bands (point-rung
windows banked; lo rides H3's c_Z·r² theorem-grade uniform bound; the hi side needs the band version of the
LPW bracket)."

**How the window is consumed (verified in code, `h5_run.py` / `h5_kernel.py`):**
- `ctx.Z_lo = max(C_Z_H3·r², Z_own.a)` feeds every **upper** bound (`rho_hi`, `coarse_fine_hi`, I_hi) via 1/Z_lo.
- `ctx.Z_hi = Z_own.b` feeds only the **lower** side (`plo = pg.a·II.a/Z_hi·area`, I_lo) — i.e. the containment
  display, not the theorem-relevant I_hi.
So the lo side is load-bearing for Theorem (1)/(2); the hi side is load-bearing only for the C1 containment ck.

**Carriers now available (frozen, both-mode byte-identical, in H3's MANIFEST, hashes verified from bytes):**

| side | carrier | statement | body hash |
|---|---|---|---|
| lo | `H3_BAND_FLOOR.md` (h3_band_floor.py a907eeed…) | E[G_r] = Z_r/r² ≥ 2.30659559567154 ∀ r ∈ (0, 0.05]; per-cell floors below | 281477c39412… (excl_bodyhash_line) |
| hi | `H3_BAND_CEIL.md` (h3_band_ceil.py b97c5428…) | E[G_r] ≤ 3.74767948915996 ∀ r ∈ (0, 0.05]; per-cell ceilings below | cfe8a3a49e32… (excl_bodyhash_line) |

**Rung-band mapping (H3 cells → H5 promotion bands; take min of floors / max of ceilings over covering cells):**

| H5 band [r_{k+1}, r_k] | covering H3 cells | certified Z_r/r² floor | certified Z_r/r² ceiling |
|---|---|---|---|
| [0.035355, 0.05] | (0.03,0.04] ∪ (0.04,0.05] | 2.30659559567154 | 3.74767948915996 |
| [0.025, 0.035355] | (0.02,0.03] ∪ (0.03,0.04] | 2.41792545239159 | 3.72414748332433 |
| [0.0177, 0.025] | (0.01,0.02] ∪ (0.02,0.03] | 2.44022641168325 | 3.71138266685765 |
| [0.0125, 0.0177] | (0.01,0.02] | 2.45593199230367 | 3.69805856644731 |
| (0, 0.0125] | (0,0.0025] ∪ … ∪ (0.01,0.02] | 2.45593199230367 | 3.69805856644731 |

Note the band floor 2.3066 is **stronger** than the c_Z = 1.6155 the H5 code currently takes as its H3 input;
consuming it tightens 1/Z_lo by a factor 0.70 wherever the H3 floor (not the own-bracket) binds.

**Proposed register action:** OBL-H5-ZBAND → **DISCHARGEABLE**; state → CLOSED at the next H5 rung issuance
upon consumption (two sha256 pins + the table above inside the rung certificate; a mutation `zband_floor_as_
ceiling` that must fail). Until consumed: OPEN-BY-BOOKKEEPING, not by missing mathematics.
**EXECUTED 2026-09-15 (second pass):** `h5_zband_consume.py`, digest cbe8603f…, 3/3 mutations fail closed.

---

## 2. D3-LEMMA-RN-UNIF — work order (the one validity premise that did not move on 2026-09-15)
*(superseded 2026-09-16 by CL-RNU-001; kept as record)*

**Statement as frozen (D3_PERCOLATION.md body 8e7fef6b… §5(ii); OBLIGATION_LEDGER §2):** the sup over each
zone of the bracketed factors in D3 §3.1 is attained on the displayed probe net up to the displayed smoothness
margins. The factors are exact-kernel, smooth in y with e^{−d²/2} tails; the score-IBP step through the
non-smooth typing indicators uses the 1-Lipschitz negative-part representation (same grade as B1's Lemma R).
Rung part (r = 0.05): rigidity-decoupling lemma for τ(y) uniform over {d ≥ 5}; a certified interval-box Riemann
sum for the annulus crude-spine integral. Uniform-in-r part: as frozen.

**Why it is now on the critical path alone:** it is one of exactly two remaining validity premises of Theorem (2)
and a named hypothesis of Theorem (1) itself; it also gates OBL-H5-REMOTE-THRESHOLD. No lane is executing it.

**Smallest certificate that discharges the rung part (proposal):**

1. **Inputs (all frozen):** the §3.1 bracketed factors as functions of y on each zone; `cov_exact.py`
   (f08c1c5f…) kernel derivatives through order ≤ 8 with certified spectral tails; the probe-net coordinates and
   displayed margins from D3's tables; Z_lo = 7.7592917375327855e-3 (R2, H3_RUNG_FLOOR 6347275d…).
2. **Modulus step:** for each factor F and each zone, a certified bound sup_zone |∇_y F| ≤ L_F from the
   exact-kernel derivative bounds (the same covariance-factorization + spectral-moment machinery B4LOC uses:
   Cov(∂^α f(x), ∂^β f(y)) factors into 1-D K1 values), so that sup_zone F ≤ max_net F + L_F·h_net.
3. **Net step:** exhibit h_net (probe-net mesh) and check `L_F·h_net ≤ displayed smoothness margin` per factor
   per zone — a finite table, fail-closed.
4. **Indicator step:** the typing indicators are replaced by their 1-Lipschitz negative-part smoothings; show
   the smoothing error is absorbed by the same margin (B1 Lemma R grade, cite it rather than re-prove).
5. **Riemann-sum step (annulus crude-spine integral):** interval-box sum with certified per-box sup of the
   integrand from steps 2–3; total must reproduce the consumed 17.6804·r³ (I_ann at Z_lo) with the same
   polarity discipline (upper bound, round-UP display).
6. **Uniform-in-r part:** run steps 2–5 with r as an interval over the H3 cells — the no-whitening Wick
   architecture of H3_BAND_FLOOR already carries r as an interval through Gram algebra; reuse it. This is the
   piece that also discharges OBL-H5-REMOTE-THRESHOLD (r-scaled threshold d ≥ 2r per rung).

**Deliverable form:** `d3_rn_unif_certificate.py` — fail-closed, deterministic, MC-free, both modes byte-
identical, mutation suite (net coarsened ×2 must fail; L_F halved must fail; Z_lo raised must fail), FREEZE
record with a rule-id per CL-REG-001. Executable by any family; does not require Kimi.

**Falsifier (from D3's own clause):** a counterexample station with τ-uniformity violation kills the lemma.

---

## 3. Register summary

| item | before | proposed | 2026-09-16 |
|---|---|---|---|
| OBL-H5-ZBAND | OPEN (hi side needs band LPW bracket) | DISCHARGEABLE — both carriers exist; CLOSED on consumption | **DISCHARGED at consumption grade (H5_ZBAND/)** |
| D3-LEMMA-RN-UNIF | OPEN, no owner executing | OPEN with a six-step executable work order; owner: foundations or any family | **engine located (Kimi, unrun); blocker root-caused; fix validated (CL-RNU-001)** |

---

## Appendix A — D3-LEMMA-RN-UNIF input inventory (extracted from D3_PERCOLATION.md body 8e7fef6b… §3.1, verbatim objects)

Under the 9-pin law (six pins + ∇f(y) = 0, f(y) = v), with H_y = μ_y + B(H_pair − μ_pair) + η, η ⊥ H_pair,
B = Δᵀ Σ_pair⁻¹, Δ = Cov(H_pair, H_y | 9 pins), the exact RN factorization is

    ρ^{P_r}(y) = ρ⁰(y) · [p_grad/p_grad⁰](y) · [wm/wm⁰](y) · [m_y^{yv}/m_sad](y) · [Z_r^{yv}/Z_r] · (1 ± κ_cross),

    κ_cross = B_cross/(Z_r m_sad),  B_cross = √3 (E W_r⁴)^{1/4} (E q²)^{1/4} √τ(y),  q = ‖H_y‖ + ‖H_y⁰‖,
    τ(y) = tr(Δᵀ Σ_pair⁻¹ Δ)   (pair→y variance transfer).

**The five bracketed factors RN-UNIF must bound uniformly per zone:**

| # | factor | ingredients | y-dependence enters through |
|---|---|---|---|
| F1 | [p_grad/p_grad⁰](y) | conditional gradient density at y under 9-pin vs 7-pin law | Σ(∇f(y) \| pins), μ(∇f(y) \| pins) — 2×2 Schur complements of the Gram |
| F2 | [wm/wm⁰](y) | window-mass ratio | conditional (f(y), jets) law; same Gram |
| F3 | [m_y^{yv}/m_sad](y) | typed y-moment vs unconditioned saddle moment | E[(−det H_y)₊ \| 9 pins] — Wick moments E[det^k], k ≤ 8 (Stein recursion) |
| F4 | [Z_r^{yv}/Z_r] | pair typed moment under its 9-pin marginal vs unconditioned | pair Gram conditioned additionally on (∇f(y)=0, f(y)=v) |
| F5 | (1 ± κ_cross) | cross term | τ(y) (displayed 1.08 at d = 1.5 axis → 3.5e-4 at d = 5 axis, 2.0e-6 transverse), E W_r⁴, E q² |

Every entry of the 9-pin Gram is an explicit Bargmann–Fock kernel derivative K^{(α,β)}(x, y) = ∂^α_x ∂^β_y
e^{−|x−y|²/2}: a Gaussian times a polynomial in the coordinates, hence C^∞ in y with explicit, certifiably
bounded derivatives on any compact zone. Therefore each F_i is a smooth function of y on each zone whose
gradient can be enclosed by interval arithmetic on (i) the kernel-derivative polynomials, (ii) the Schur
complements (bounded below by the certified conditioning-shrinks-variance residuals, cf. B4LOC-R1 §, dps 100),
and (iii) the Stein-recursion Wick moments (polynomial in Gram entries).

**In the tree's engine (found 2026-09-16) these are exactly κ_pair (F4 via χ²), κ_y (F3 via w2), κ_cross (F5 via τ),
R_pg (F1) and R_wm (F2), assembled as κ_far = R_pg·R_wm·((1+κ_pair)(1+κ_y)+κ_cross) − 1; see CL-RNU-001.**
