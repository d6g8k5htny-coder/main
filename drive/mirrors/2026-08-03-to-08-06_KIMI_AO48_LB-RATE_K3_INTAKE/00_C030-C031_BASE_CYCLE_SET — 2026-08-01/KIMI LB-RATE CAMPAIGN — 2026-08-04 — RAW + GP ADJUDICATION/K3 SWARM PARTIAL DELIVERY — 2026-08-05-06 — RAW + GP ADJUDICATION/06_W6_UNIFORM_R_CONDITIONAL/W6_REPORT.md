# W6 REPORT — uniform-in-r analysis of the WP channel (SIDE24/q0, K3 swarm)

Workstream W6 (uniform-in-r analyst). Deliverable: the r-dependence structure of the
WP channel as (a) an exact r→0 limiting theory (limit jet + limiting conditional law
+ residual variance laws) and (b) a WP-min modulus `E_WP(r) = 4.2·r^{3/2}` in the
§5.6 spec of W10_REPORT (validity proved; rung-grid certified; coverage completed by
two named, quantified premises of C-3 type).

All certified claims are backed by fail-closed certificates (`w6_cert.py`, `w6_wp.py`;
ck()/SystemExit(1); no bare asserts; no randomness; byte-identical under `python` and
`python -O`; transcripts in `transcripts_*.txt`).

## Headline results

1. **WP-min modulus (W10 §5.6 spec), exact-integrand form.** E⁰_r[N_ws(B₃ ∖ collars)]
   = ∫_zone ρ_exact ≤ ∫_{B₃} ρ_exact = I_WP(r) ≤ **E_WP(r) = 3.5e-3·r^{3/2}** on
   (0, r₀ = 0.05]. Validity **proved** (Kac–Rice equality for the frozen G1
   integrand + zone ⊆ B₃ with ρ ≥ 0 — no Cauchy–Schwarz loss); explicit; vanishing
   (α = 3/2 > 0); rung-grid certified (11 rungs 0.05 → 0.00625, scaled coefficient
   C_I = I_WP/r^{3/2} ≤ 3.202e-3, **sup at r₀**; the small-r fullzone values
   0.003125/0.0015625/0.00078125 give C_I = 1.087/0.471/0.085 e-3, monotone
   decreasing); and **E_WP(0.05) = 3.5e-3·0.05^{3/2} = 3.91e-5 ≲ 0.06** (measured
   tier ✓, margin ×1533). Interval coverage is completed by the named, quantified
   monotonicity premise **P-mono** of §5 (d log I_WP/d log r ≥ 3/2; measured
   discrete log-slopes 1.66–2.37, asymptote → 3). **The earlier global-CS form
   (4.2·r^{3/2}) was REFUTED and is superseded**: the CS bound has a spurious
   near-cluster singularity (§5) — the exact integrand is the correct, tighter
   modulus object.
2. **Limit constant c confirmed independently.** c := lim (μ_t − b)/ℓ =
   **−3476069/6953125** = −0.4999290247191…, reproduced by a fully independent
   W6 pipeline (own kernel, own conditioning) to 1.5e-13 (5-point Richardson in
   r² at dps 160, rungs to 9.8e-5). The campaign's C026 rational is exact.
3. **Limit jet value f_xxx(0) → +2; the trap value −4 is refuted.** The naive
   combination 24(f(S)−f(M))/r³ → −4 ignores that the gradient pins at M,S force
   the midpoint drift f_x(0) = −r²/4 + O(r⁴); the consistent limit is **+2**
   (certified: E[f_xxx|pins] = 2.000000049 at r = 3.1e-4). Full limit jet in §1.
4. **"v_t ∝ r⁴" refuted (a result, not a formality).** At fixed y (the WP zone),
   v_t(y;r) = v_{t,0}(y)(1 + O(r)) with v_{t,0} > 0 (an r⁰ law); at cluster scale
   y = r·ŷ, v_t ~ r⁸ (and v ~ r⁶). The r⁴-class survives only as the crossover
   v_t ~ d⁸·w(ê) → const at d = O(1) (e.g. v_t ~ r⁴ at d ~ √r). The prior
   campaign's C-3 guess (v_t ∝ r⁴ on the zone) was wrong in the optimistic
   direction for the envelope's needs: on the zone the window width sees a
   NON-vanishing s_F, so the window probability is ∝ ℓ = r³/6 there (§4), and the
   WP r-dependence is carried by the z-score drift, not by a vanishing v_t.

## 0. Model and conventions (validated)

Normalized periodized Bargmann–Fock field on T²₂₄, Var f = 1, kernel
K(x) = K1(x₁)K1(x₂), K1(s) = Z⁻¹ Σ_j e^{−κ_j²/2} e^{iκ_j s}, κ_j = πj/12 (W2's K-B
convention). Wrapped dual: K1(s) = Σ_n e^{−(s+24n)²/2} (wrap error < 4e^{−288});
K1⁽ᵃ⁾(s) = Σ_n (−1)ᵃ He_a(s+24n) e^{−(s+24n)²/2}. Moments exactly Gaussian:
m2 = 1, m4 = 3, m6 = 15, m8 = 105 (certified; wrapped ≡ spectral to 1.8e-59).

Pins at rung r: M = (−r/2,0), S = (r/2,0), Y = M + r(−0.76, 0.24) = r(−1.26, 0.24);
(f, ∇f) at each (9 pins); f(M) = b = 6/5, f(S) = b − ℓ, ℓ = r³/6,
f(Y) = μ_t(r) := E[f(Y) | 6 pins at M,S, ∇f(Y)=0], gradients 0.

**Validation anchors (all PASS):** c(0.025) = −0.4997159 (W2 probe); arch-ray
v-triple 0.01770/0.55748/0.97764 (W2: 0.0177/0.5575/0.9776); W5/C034 station row
v = 0.0005554263945655643 (ledger …5655), φ₂ = 0.226140802688 (ledger …32848),
φ_f = 1.9664442e-10; W4's frozen 25-digit probe laws at (0, 0.5672) matched to 16
digits; W4's E_win probe 2.020901410980e-05 matched to 11 digits.

## 1. Mandate 1 — the exact limit jet

**Limit span.** The 9 pin functionals converge to the 9-dimensional subspace of the
≤3-jet dual
  S₀ = ann(n),  n = (0, 1, 21/2, 209/3)  on (f_xxx, f_xxy, f_xyy, f_yyy);
the unique free third-order direction is n·jet = f_xxy + (21/2)f_xyy + (209/3)f_yyy
(conditional variance stays O(1): certified 28510.43 over rungs 0.005 → 0.0003125).
Certified basis (each element verified pinned: Var(·|pins) → 0 at r⁴-rate, collapse
×256 to ×4×10⁹):

| functional | definition | limit value |
|---|---|---|
| f    | f(0) | b = 6/5 |
| f_x, f_y | | 0, 0 |
| f_xx, f_xy, f_yy | | 0, 0, 0 |
| f_xxx| | **+2** |
| A3   | d1²f_xxx + 2d1d2 f_xxy + d2²f_xyy | 1/2 |
| D3   | (d1²+3/4)f_xxy + 2d1d2 f_xyy + d2²f_yyy | (1.52 − 2c)/0.24 = 10.4994085… |

with d = (d1, d2) = (−1.26, 0.24) = Y/r, c = −3476069/6953125.

**Derivation (exact Taylor-matching).**
- (P1+P2)/2 = f + (r²/8)f_xx + … = b − r³/12 ⇒ f → b. (P3+P4)/2 = f_x + (r²/8)f_xxx
  = 0 ⇒ f_x = −(r²/8)f_xxx. (P2−P1)/r = f_x + (r²/24)f_xxx + … = −r²/6.
  Substituting: −r²/6 = −(r²/12)f_xxx ⇒ **f_xxx → +2**, f_x(0) = −r²/4 + O(r⁴).
  **Consistency trap:** reading f_x ≈ 0 gives the trap value −4; the gradient pins
  force the drift, and +2 is consistent (certified).
- f_y = −(r²/8)f_xxy → 0; f_xx = −(r²/24)f_xxxx → 0; f_xy → 0; f_yy → 0 at rate
  O(r) (E[f_yy]/r → −12.7414).
- A3 from P8 = f_x(Y) = 0: f_x + r(d1f_xx + d2f_xy) + (r²/2)A3 + O(r³) = 0 with
  f_x = −r²/4 ⇒ **A3 → 1/2**.
- D3 from P9 = f_y(Y) = 0 after eliminating lower-order approximations from
  P1..P7: the surviving third-order functional is (1/3)B3 + (1/4)f_xxy −
  (2d1/3d2)A3 + (d1/6d2)f_xxx (B3 := d1²f_xxy + 2d1d2 f_xyy + d2²f_yyy), i.e.
  **D3 = B3 + (3/4)f_xxy** mod pinned; the 3/4 traces to the (P5+P6)/2 ≈
  f_y + (r²/8)f_xxy contamination. Value −(12/d2)(c/6 + 1/12 + d1/6) =
  (1.52 − 2c)/0.24 (A3/f_xxx value corrections cancel exactly). Certified:
  E[D3|pins] Richardson limit 10.49941 vs 10.4994085.

**Limit-jet mean vector (10 components), certified:**
E[(f, f_x, f_y, f_xx, f_xy, f_yy, f_xxx, f_xxy, f_xyy, f_yyy) | F = v*] =
(1.2, 0, 0, 0, 0, 0, 2, 4.3835205968, −0.4174781777, 0), residual variances
(0,0,0,0,0,0,0, 1.1567e-3, 0.127526, 5.61395) — exactly rank-1 along n
(ratios 1 : 10.5² : (209/3)² certified).

## 2. Mandate 2 — the limiting conditional law

For fixed y, the pin-conditioned law of (f(y), ∇f(y), H_y) converges to the
conditioning on (F, v*) of §1 (certified at the C034 station: max |cov − lim| ≤
2e-3 at r = 3.9e-4, means ≤ 8e-5, O(r) rate). Limit Gram G₀: 9×9, SPD, eigenvalues
[5.44e-3, 56.16] (Cholesky PASS); certified inverse ‖G₀G₀⁻¹ − I‖_max = 2.6e-81
(dps 80). Entries closed form (Gaussian moments × polynomials in (d1, d2));
diagonal: G₀[f,f] = 1, [f_x] = 1, [f_y] = 1, [f_xx] = 3, [f_xy] = 1, [f_yy] = 3,
[f_xxx] = 15, [A3,A3] = 39.46308336, [D3,D3] = 18.34811136. Conditional law:
m₀(y) = κ(y)ᵀG₀⁻¹v*, Σ₀(y) = S(y) − κ(y)ᵀG₀⁻¹κ(y), κ_k(y) = Cov(F_k, ·)
(explicit kernel derivatives).

## 3. Mandate 3 — residual variance laws (orders + coefficients + remainders)

**(i) Fixed y (WP-zone scale).** v(y;r), v_t(y;r) := Var(f(y)|pins,∇f(y)=0),
Σ_∇(y;r) obey **r⁰ laws** with O(r) remainders:
v(y;r) = v₀(y)(1 + a(y)r + O(r²)), v_t(y;r) = v_{t,0}(y)(1 + a_t(y)r + O(r²)).
Representative certified values (arch-side strip — verified dominant WP region §5;
C034 station at angle 2.35 rad, d = 1):

| y | v₀(y) | v_{t,0}(y) |
|---|---|---|
| (0, 0.4) | 5.685e-4 | 2.79e-6 |
| (0, 0.5672) | 2.16e-2 | 4.66e-5 (s_F = 6.83e-3) |
| (0, 0.7) | 1.287e-2 | 2.487e-4 |
| (0, 1.0) | 7.637e-2 | 4.409e-3 |

**(ii) Cluster (y = r·ŷ).** v(y;r) ~ r⁶ (v/r⁶ → const: 6.49e-3 at ŷ = (0.5,0.5),
0.1816 at ŷ = (0,1)); **v_t(y;r) ~ r⁸** (vt/r⁸ → const 0.0498 at ŷ = (0.5,0.5);
vt/r⁴, vt/r⁶ → 0). Certified.

**Result (headline 4):** "v_t ∝ r⁴" is refuted in both regimes; the r⁴-class is
only the crossover between v_t ~ d⁸·w(ê) and the d = O(1) constant.

## 4. Mandate 4 — r-dependence of the WP integrand

Frozen integrand (G1): ρ(y;r) = p_grad(y;r)·E[|det H|·1{det H<0}·1{b−ℓ<f(y)<b} |
9 pins, ∇f(y)=0]. At fixed y (certified display at the wedge peak (0, 0.5672),
transcripts_wp_grid1.txt §P2):

- **r-invariant** (→ positive limits, O(r) remainder): p_grad (→ 5.75),
  √(E[det²H|∇=0]) (O(1)), s_F = √v_t (→ 6.83e-3), m(y;r) → m₀(y).
- **carries r:** the window probability P_W = Φ((b−m)/s_F) − Φ((b−ℓ−m)/s_F)
  ≈ (r³/6)·φ(ẑ)/s_F, ẑ = dist(m, [b−ℓ,b])/s_F. P_W/r³: 0.00095 → 2.06 → 8.96 →
  9.51 over r = 0.05 → 0.00625 — the z-score drift is the entire transient. Hence:
  **the r³ window law is asymptotically correct but z-drift-delayed; the apparent
  decade slope 1.45–1.6 on [0.005, 0.05] is the transient, not the asymptote** —
  and the honest successor remainder is (1 − O(r))·(1 − O(r^{1.45})), not O(r³).

Measured truth I_WP(r) = ∫ρ dy (own integrator, validated vs W4 to 11 digits;
wedge + tails, convergence-checked; wedge grid convergence 4e-5 relative):

| r | I_WP | halving exponent |
|---|---|---|
| 0.05 | 3.5794e-5 | 2.26 |
| 0.025 | 7.4993e-6 | 1.94 |
| 0.0125 | 1.9569e-6 | 1.72 |
| 0.00625 | 5.9411e-7 | 1.64 |
| 0.003125 | 1.9019e-7 | 2.71 |
| 0.0015625 | 2.9125e-8 | 3.97 |
| 0.00078125 | 1.8537e-9 | — |

The exponent U-turns at r ≈ 0.003: the dominant wedge peak marches inward
(0.85 → 0.42 → 0.2) and the near-cluster boost dies (inner annulus radius < 0.06
contributes ≤ 1.5e-18 at r = 0.001, 3e-224 at 0.01 — certified). Note
I_WP(0.025) = 7.50e-6 > 3.328125e-6 = 0.213r³, consistent with the registered
formal refutation of the old 0.213r³ upper-bound claim.

## 5. Mandate 5 / Gate G5 — WP-min modulus (W10 §5.6 spec)

**Lemma (WP-min, W6 exact-integrand form).** E⁰_r[N_ws(B₃ ∖ collars)] ≤ I_WP(r) ≤
**E_WP(r) = 3.5e-3·r^{3/2}** for all r ∈ (0, r₀ = 0.05].

- **Validity (proved).** Kac–Rice equality for the frozen G1 integrand:
  E⁰_r[N_ws(zone)] = ∫_zone ρ_exact(y;r) dy with ρ_exact = p_{∇f̃(y)}(0)·
  E[|det H|1{det H<0}1{F∈W} | 9 pins, ∇f(y)=0] ≥ 0 (certified ρ ≥ 0 on 144
  samples). Since zone = B₃ ∖ collars ⊆ B₃ and ρ_exact ≥ 0,
  E⁰_r[N_ws(zone)] ≤ ∫_{B₃} ρ_exact dy =: I_WP(r) — **no Cauchy–Schwarz loss**.
  The exact integrand is validated against W4's frozen probe (E_win at
  (0,0.5672;0.025) = 2.020901410980e-05, matched to 11 digits; ρ = 1.0529e-4).
- **Rung-grid certificate (PASS, `w6_wp2.py`).** 11 rungs {0.05, 0.04, 0.035, 0.03,
  0.025, 0.02, 0.015, 0.0125, 0.01, 0.0075, 0.00625}: C_I(r_k) = I_WP(r_k)/r_k^{3/2}
  = {3.202, 2.635, 2.374, 2.127, 1.897, 1.684, 1.490, 1.400, 1.316, 1.238, 1.202}e-3
  — **sup 3.202e-3 at r₀ = 0.05**, all ≤ 3.5e-3 (margin 9.3%). The smaller fullzone
  rungs 0.003125/0.0015625/0.00078125 give C_I = 1.087/0.471/0.085 e-3 (monotone
  decreasing continues). E_WP(0.05) = 3.5e-3·0.05^{3/2} = 3.91e-5 ≲ 0.06 ✓
  (margin ×1533). Quadrature at r₀ is grid-converged (4e-5 relative); the sup
  location at r₀ makes the certificate insensitive to small-r quadrature error.
- **Named premise P-mono (interval coverage), quantified.** Sufficient condition:
  d log I_WP/d log r ≥ 3/2 on (0, r₀] (equivalently C_I(r) nondecreasing in r, so
  its sup on (0, r₀] is at r₀). Then C_I(r) ≤ C_I(r₀) = 3.202e-3 ≤ 3.5e-3 for all
  r ∈ (0, r₀]. Measured: every discrete log-slope ∈ [1.658, 2.372] (11-rung grid)
  and the asymptotic law I_WP ~ (C₀/6)r³ gives log-slope → 3 > 3/2; the premise
  holds with margin ≥ 0.16 everywhere checked. Its certified derivative bound
  (stations move linearly in r; the chain is mechanical) is the open formality,
  of exactly the C-3 pattern; the sub-0.00625 tail rides on the same premise
  (measured C_I continues to decrease: 1.20 → 1.09 → 0.47 → 0.085 e-3).

**Refuted alternative (documented for the register).** The earlier corrected
global-CS form E_WP = 4.2·r^{3/2} is **invalid as stated and is superseded**: the
pointwise bound ρ ≤ ρ_CS = p_grad·√(E[det²])·√(P_W) (W2 Thm W2-7) is valid
(certified, max exact/CS = 0.045), but ρ_CS has a spurious near-cluster
singularity on the arch direction (where the third-order jet coefficient
J3 = E[f_yyy] ≈ 0): there p_grad ~ d⁻⁴ while the φ(z_F) kill is only O(1), so
ρ_CS ~ ℓ^{1/2}d⁻⁴ and I_CS(r)/r^{3/2} fails to stay ≤ 4.2 — under quadrature
refinement at r = 0.0015 it grows 3.97 → 4.05 → 4.43 → 4.87 → 6.29 (grids
48→120, near-cluster peak at d ≈ 0.2 sharply angular-localized), exceeding 4.2.
The truth is dead there (W4: ρ ≤ 1e-171 on rings 0.002–0.05); the blowup is the
CS √-loss maximized where P_W is small (CS/exact = 420–66000× at d = 0.1–0.567,
r = 0.0015). This is why the exact-integrand form (Kac–Rice equality, no loss)
is the correct modulus object. (`w6_cs.py` and the `transcripts_wp_grid*.txt`
are retained as the record of the refuted approach.)

## 6. Deliverables, certificates, determinism

- Code: `w6_kernel.py` (kernel + conditioning), `w6_analysis.py` (limit law),
  `w6_rho.py`, `w6_rhofast.py` (exact integrand), `w6_zone.py`, `w6_fullzone.py`
  (truth quadrature), `w6_cs.py` (CS bound), `w6_varlaws.py`, `w6_cluster.py`,
  `w6_cert.py`, `w6_wp.py`.
- `w6_cert.py` (mandates 1–3): exit 0; byte-identical normal/-O (verified);
  transcript `transcripts_cert_main.txt`,
  sha256 000bfb456d51614aa3f29b00f23f5e7be64ca3611fa37ef16be584ce9c1608a9.
- `w6_wp2.py` (mandates 4–5, EXACT-integrand modulus — the operative certificate):
  exit 0; transcripts `transcripts_wp2_fullgrid.txt` (full 11-rung grid 0.05–0.00625
  incl. anchors Q0/Q1 + P-mono slopes) and `transcripts_wp2_fullgrid_O.txt`
  (byte-identical `-O` run; cmp-verified). These match the earlier split-half runs
  byte-for-byte on every overlapping rung and were independently reproduced by
  W13's clean room.
- `w6_wp.py` + `transcripts_wp_grid1/2.txt`: REFUTED global-CS form (§5), retained
  as record only — do not consume.
- No randomness; mpmath fixed dps; deterministic quadrature grids; receipts
  (this section + sha256 lines) stored separately from labels/prose.

## 7. Open pieces (named, quantified)

- P-mono (= campaign C-3 pattern): certified derivative bound d log I_WP/d log r
  ≥ 3/2 on (0, 0.05] (evidence: discrete log-slopes ∈ [1.658, 2.372] on the
  11-rung grid; asymptote → 3). Discharges both inter-rung coverage and the
  sub-0.00625 tail (measured C_I continues decreasing: 1.20 → 0.085 e-3).
- F3: interval-certified (Bates-grade) zone quadrature for I_WP replacing
  estimate+refinement grade (wedge convergence 4e-5 relative; the sup-at-r₀
  structure makes the certificate robust, and W3's G3/G6 machinery is the
  natural enclosure engine; interfaces with W3/W4's certified I_WP enclosures).

Everything else in this report is certified as stated.

---
*Revision note (housekeeping, W13 finding F1):* the wp2 certificate's stored
transcripts were consolidated from two split-half runs into the single full-grid
transcript `transcripts_wp2_fullgrid.txt` (+ byte-identical `-O` twin), matching
every overlapping rung byte-for-byte and independently reproduced by W13's clean
room. No mathematical change. P-mono remains the single named premise for
interval coverage; F3 remains the Bates-grade quadrature note.
