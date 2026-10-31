# KIMI-DER-027c — RIDGE: certified two-sided control at both ridge maxima

**Program:** SIDE24/q0 · AO48-WO-063 Task 4c · Lemma LB-RATE hardening campaign, ridge channel.
**Inputs:** work order AO48-WO-063, sha256
`e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2` (echoed in the
certificate transcript).
**Verdict:** **PROVED** — tasks 4c(i)–(iv) are machine-certified at the LB-1 census rung of
record r = 0.025, with outward-rounded intervals and explicit margins throughout, by the
fail-closed certificate below. No status or Boolean claims are made or relied on.
**Certificate:** `verify_ridge_two_sided_v1.py`
(sha256 `d5f4a4ca84c7bbf09a4e26d57f6b299dde068b1c19d6df98f042326210c539a4`),
`ck()` fail-closed (any check → `SystemExit`), 87 checks, runtime ≈ 5 s; transcripts
`t_normal.txt` and `t_O.txt` byte-identical
(sha256 `c2eb24eda3376b8c95d2cbb0518cd19f865ed9a885b3b3b5622b12f56b8f9ac5`).

---

## 1. Object and setting (verbatim ledger construction)

Field: periodized 2D Bargmann–Fock field on the side-24 torus, covariance
K(u) = e^{−|u|²/2} (unit variance); spectral lattice (π/12)ℤ², masses e^{−|k|²/2},
truncation |k| ≤ 30. Level b = 6/5 (EXACT); window ℓ = r³/6; rung r = 0.025 (EXACT,
LB-1's certified census rung — the rung at which the WO-quoted ridge data stand).
The 9-pin conditional mean m₉ = k(x)ᵀ Σ_PP⁻¹ **v** (Gaussian regression mean) with
1-jet pins (f, f_x, f_y):

- M = (−r/2, 0) at value b; S = (+r/2, 0) at value b − ℓ;
- Y = r·(−1.26, 0.24) at value v* = clip(μ_t, [b−ℓ, b]), μ_t the 6-pin conditional mean
  at Y given ∇f(Y) = 0. Certified: v* = μ_t = b − 0.499715906773·ℓ (decimal-dps-80),
  strictly inside the window, in the C026 band of the limit −0.4999290.

Precision labels: EXACT (lattice, pins, b, r, ℓ as rationals) vs decimal-dps-80 (all
evaluations, envelope E_EVAL = 1e-50, 21 orders above the measured inverse residual
3.23e-71 and 5 above the coefficient envelope dc = 5.98e-55) vs float64-audited (grid
engine, certified allowance EPS_F = 5e-7 > analytic worst-case 4.46e-7, 450× the
audited max discrepancy 1.1e-9).

## 2. The exact periodized kernel (WO machinery requirement)

The certificate evaluates every covariance in **two independent representations**
(certificate [A1], [A3]):

1. **Spectral lattice (primary per the WO):** factorized 1D lattice sums
   K1ⁿ(t) = (√(2π)/24)·(−1)^⌈n/2⌉ Σ_{|j|≤114} e^{−k_j²/2} k_j^n {cos,sin}(k_j t),
   k_j = jπ/12, with certified truncation tails
   Σ_{|j|≥115} e^{−k²/2}|k|^n ≤ 2e^{−K_A²/2}K_A^n(1+4/K_A²)·(√(2π)/24) at K_A = 115π/12:
   ≤ 3.12e-198 (n=0), 2.82e-195 (n=2), 2.56e-192 (n=4), 2.32e-189 (n=6) — the WO's
   "certified tail < 1e-60" is met with 129+ orders of slack.
2. **Planar closed form** (Hermite): certified equal to the periodized kernel by
   Poisson summation: image sums ≤ 4.3e-31 (|u| ≤ 12, value order), ≤ 1e-100 at pin
   separations (orders ≤ 2), propagated through the certified Gram inverse to
   |m₉^per − m₉^plan| ≤ 1.42e-104 (Gram) and ≤ 7.87e-98 (root evaluations) —
   40+ orders below every margin.

Cross-representation agreement: covariances at 6 probes × 18 derivative orders agree to
9.49e-81; the **full independent spectral pipeline** (Gram, regression coefficients,
m₉, ∇m₉, Hess m₉ at both maxima) agrees with the planar pipeline to ≤ 3.23e-65
(gate 1e-60). All kernel evaluations for the results below are therefore the exact
periodized kernel's values up to the stated envelopes.

## 3. Rigorous global bounds (RKHS, certificate [A2])

‖m₉‖²_H = **v**ᵀΣ_PP⁻¹**v** = **v**·coef = 13.219736462363 (decimal-dps-80; agrees with
LB-2's 13.21973646). By Cauchy–Schwarz in the RKHS and Var(∂ᵃf | pins) ≤ Var(∂ᵃf)
(= (2a₁−1)!!(2a₂−1)!!), every derivative obeys |∂ᵃm₉(x)| ≤ ‖m₉‖·√((2a₁−1)!!(2a₂−1)!!)
pointwise on the whole torus, giving the certified global constants
G1 = 3.635896, G2 = 6.297556, G3 = 14.081763, D2F = 10.907687 (‖H‖_F),
D3F = 48.780658 (‖dH‖_F/dx). Every exclusion and Taylor remainder below uses only
these certified constants — no fitted Lipschitz bounds.

## 4. Task 4c(i) — existence and uniqueness of P* and Q* (Kantorovich, certificate [B])

For F = ∇m₉, at each Newton-refined point x₀ (residual < 5e-75, decimal-dps-80):
η = β(|F(x₀)| + E_EVAL), β = (min|eig Hm₉(x₀)|·(1−1e-15))⁻¹ (exact for the symmetric
Hessian), γ = D3F (global RKHS Lipschitz of DF). Kantorovich α = βγη < 1/2 ⇒ a unique
critical point exists in B(x₀, ρ₂), lying in B(x₀, ρ₁), ρ₁,₂ = (1∓√(1−2α))/(βγ).

| maximum | residual η | min\|eig\| (floor) | β | α | root enclosure ρ₁ | uniqueness ball ρ₂ |
|---|---|---|---|---|---|---|
| P* | 3.86e-51 | 2.587401874 | 0.38648809 | 7.29e-50 | 3.86e-51 | **0.106083** |
| Q* | 5.1e-51 | 1.958943924 | 0.51047914 | 1.27e-49 | 5.1e-51 | **0.0803164** |

Hessian eigenvalue intervals (outward, ±1e-50): P*: λ ∈ [−2.76737639265, −2.58740187433];
Q*: λ ∈ [−2.75639190410, −1.95894392430]. Both maxima are certified nondegenerate with
definiteness margins **≥ 2.5874 (P*)** and **≥ 1.9589 (Q*)** — these are the explicit
eigenvalue margins the WO requires.

## 5. Task 4c(ii) — certified values and distances (outward-rounded, certificate [C])

| quantity | certified value (decimal-dps-80) | half-width | 12-decimal outward interval |
|---|---|---|---|
| m(P*) | 1.93119035755789484 | 2.41e-50 | [1.931190357557, 1.931190357558] |
| m(Q*) | 1.66659821936183025 | 2.86e-50 | [1.666598219361, 1.666598219362] |
| d(P*) | 1.4747477269211 | 1.39e-50 | [1.474747726921, 1.474747726922] |
| d(Q*) | 1.41162073528202 | 1.51e-50 | [1.411620735282, 1.411620735283] |

(d = distance from the origin, LB-1's census convention.) Half-widths combine the
Kantorovich root enclosure and the evaluation envelope. Consistency gates (all pass):
WO quotes 1.931 / 1.667 / 1.475 / 1.412 within 5e-4; LB-1's prints P* = (1.30561,
0.68576), Q* = (−1.10400, 0.87969), m = 1.9312 / 1.6666 within their rounding.
Conditional variances (assembly noise scale, decimal-dps-80): v(P*) = 0.174043537433
(sd 0.4172, (m−b)/sd = 1.75); v(Q*) = 0.126943975593 (sd 0.3563, (m−b)/sd = 1.31).

## 6. Task 4c(iii) — the zero-above-b-saddle census (certificate [D])

**D.1 (every candidate typed).** All seven critical points of m₉ in B₂.₅ are
Kantorovich-certified (existence, uniqueness, type, value), with pairwise-disjoint
uniqueness balls:

| point | type | value (decimal-dps-80) | min\|eig\| | ρ₂ |
|---|---|---|---|---|
| M = (−0.0125, 0) | MAX | b exactly (±1e-40) | 0.0149431 | 6.13e-4 |
| S = (+0.0125, 0) | SADDLE | b − ℓ exactly (±1e-40) | 0.0333728 | 1.37e-3 |
| Y = (−0.0315, 0.006) | SADDLE | v* exactly (±1e-40) | 0.0233729 | 9.58e-4 |
| P* | MAX | 1.93119035755789 | 2.5874019 | 0.106083 |
| Q* | MAX | 1.66659821936183 | 1.9589439 | 0.0803164 |
| min1 = (−1.740887, −1.077149) | MIN | −0.58051424651 | 1.3664257 | 0.056 |
| min2 = (1.563971, −1.480020) | MIN | −0.33177854916 | 1.0436669 | 0.043 |

**D.2 (every other region excluded, second-order).** Adaptive grid exclusion over
B₂.₅ (cell side quartered 0.02 → 4.9e-6): a cell of circumradius ρ about center c contains no
critical point whenever |∇m₉(c)| > (‖H‖_F(c) + D3F·ρ)·ρ + EPS_F — a rigorous
second-order (Taylor-with-Hessian-remainder) test using only the certified global
constants. All 167 terminal survivor cells lie inside the certified uniqueness balls
(gate fail-closed: any survivor outside every ball aborts the certificate). Hence
**Crit(m₉) ∩ B₂.₅ = {M, S, Y, P*, Q*, min1, min2} exactly.**

**D.3 (outside B₂.₅, value exclusion).** m₉(x) ≤ m₉(c) + |∇m₉(c)|ρ + (D2F/2)ρ² + EPS_F
< b certified on 22,556 cells of the annulus 2.5 < d ≤ 3 (max upper bound 0.948134)
and on 220,057 cells of the torus exterior d > 3 in the fundamental domain
(max upper bound 0.416956). No above-b critical point of any type exists outside B₂.₅.

**Conclusion (torus-global).** The critical points of m₉ with m₉ ≥ b are exactly
P* (max, m = 1.9312), Q* (max, m = 1.6666), and M (max, m = b exactly). The only
saddles are S (b − ℓ) and Y (v*), both below b. **The mean has ZERO above-b saddles** —
a deterministic, certified property of m₉; LB-1's zero-above-b-saddle census is
confirmed at theorem grade with second-order localization throughout.

## 7. Task 4c(iv) — margins for the assembly's ridge channel

Theorem-grade two-sided control constants (all certified above):

- **Root enclosures:** P* ∈ B(x₀, 3.9e-51), Q* ∈ B(x₀, 5.1e-51) — the maxima are
  pinned to 50 decimals.
- **Uniqueness balls:** ρ₂(P*) = 0.106083, ρ₂(Q*) = 0.0803164 — no other critical
  point of m₉ within macroscopic balls around the ridge maxima.
- **Hessian definiteness floors / IFT displacement factors:** min|eig| = 2.587401874,
  β = 0.38649 (P*); min|eig| = 1.958943924, β = 0.51048 (Q*). Under any perturbation
  h with ‖h‖_{C²} < min|eig|, each maximum persists (displacement ≤ β‖h‖, value drift
  ≤ ‖h‖); the max typing and the above-b status survive ‖h‖_{C²} < min(m−b, min|eig|):
  **0.7312 (P*)** and **0.4666 (Q*)** — macroscopic, ℓ-free margins, the same structural
  simplification (frozen pins) that LB-2's ridge-channel closure exploits.
- **Value gaps above b:** m(P*) − b = 0.7311903576; m(Q*) − b = 0.4665982194.
- **Census margins:** exclusion is certified cell-by-cell with the float64 allowance
  EPS_F = 5e-7 (> analytic worst case 4.46e-7); the audited gradient floor scale is
  ~3.8e-4 in the M–S plateau and ~0.05 elsewhere in B₂.₅, both 2+ orders above the
  allowance; value exclusions carry margins ≥ 0.25 (annulus) and ≥ 0.78 (exterior);
  the census itself is EXACT (a deterministic property of m₉, not a probabilistic
  bound).
- **Noise scale at the maxima:** sd √v = 0.4172 (P*), 0.3563 (Q*) — consistent with
  LB-1's O(1) expected critical count above b near P* (the channel is carried by the
  pass-zone localization and the mean-ridge routing, not by any kill at the maxima).

## 8. Hypothesis list

- **H1 (kernel/periodization):** the covariance is the exact periodized kernel;
  planar closed form used only under the certified Poisson perturbations of §2
  (≤ 1e-98 at roots, ≤ 1.83e-19 on grids; lattice tail < 1e-189). No analytic content.
- **H2 (Gaussian regression):** the 9-pin conditional law is Gaussian with mean
  m₉ = kᵀΣ_PP⁻¹**v** (standard; the field is analytic and the pin Gram is certified
  nondegenerate, inverse residual 3.23e-71). The task's statements are about the
  deterministic mean m₉, so no Kac–Rice or count hypothesis is consumed here.
- **H3 (region boundary):** the census region B₂.₅ plus value exclusions to the full
  torus make the zero-above-b-saddle statement torus-global and self-contained.
  Downstream consumption by the assembly (how above-b saddles of the *noisy* field are
  counted and routed) remains LB-1's Lemma H-SUP / pass-zone localization and the
  C029 B3(iii) mean-ridge channel — dependencies, not inputs re-proved here.
- **H4 (rung):** certification is at the EXACT rung r = 0.025, the LB-1 census rung of
  record for the WO-quoted ridge data.

## 9. Dependency table

| dependency | content consumed | hash / freeze |
|---|---|---|
| AO48-WO-063 | tasking, ridge data quotes | e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2 |
| C024 | estimand (pin values b, b−ℓ, window ℓ = r³/6) | freeze 0df10fa6…139 |
| C025 | pin construction (v* = clip μ_t); Lemma FD region language | freeze 0f8cb3dc…60b |
| C026 | (b−v*)/ℓ → −0.4999290 band (verified 0.499715906773) | freeze 6ac9a33c…7df |
| C027 | derived-on-grid grade standard for exclusions | freeze f0833443…dc2 / pkg c9d1466a…3b4 |
| C029 | B3(iii) mean-ridge channel (routing of ridge saddles) | freeze ddd8596f…0ca |
| C030/C031 | horizon tables; ledger quotes 1.931/1.667/1.475/1.412 | ae20f4e3…f4b / e165821b1479f619af8ebdb1438460edfebe88f44c189553ca86f73f8fe1afbe |
| LB-1 (KIMI, sup-over-zone) | census of record (7 critical points; zero above-b saddles); falsifier exhibit | `lb1_sup_over_zone_certificate.py` af19344df3524d03c8d54e188bddb30f41aa5a980414a5f588e84be995775c5e |
| LB-2 (KIMI, mean-ridge) | RKHS bound scheme; Kantorovich scheme; ridge-channel closure at frozen rungs | `lb2_cert.py` bf76df4189033525526d6fd4a73ca15ef62095de119d114548e033d3e137175d |

Independence disclosure: this deliverable was produced by an independent sub-agent
session. All numbers were recomputed from the kernel; LB-1/LB-2 machinery was used as
documented schemes (RKHS bounds, Kantorovich form), not as trusted values — every
constant (‖m₉‖²_H, G1–D3F, eigenvalues, census) is recomputed and re-certified here.

## 10. The falsifier named

An independent recomputation of m₉'s jet from the exact periodized kernel that
disagrees with any certified interval of §4–§5 beyond its stated half-width; **or** a
certified critical point of m₉ in B₂.₅ outside the seven uniqueness balls (the
certificate's survivor gate is designed to catch exactly this and aborts FAIL); **or**
a certified point x on the torus with m₉(x) ≥ b outside B₂.₅ (the value-exclusion
gates); **or** a saddle-typed (det H < 0) critical point of m₉ with value > b anywhere
on the torus. None known; the census agrees with LB-1's independent
Newton-plus-grid census at every printed digit.

## 11. Remaining gap (one paragraph)

The certification is at the single EXACT rung r = 0.025 — the rung of record for the
WO-quoted ridge data and LB-1's falsifier exhibit. The same construction replicates at
r = 0.05 (spot-verified in-session at 60 dps: P* = (1.343178, 0.649897), m = 1.94020;
Q* = (−1.165687, 0.856997), m = 1.67118 — decimal-dps-60, not interval-certified here),
and the certificate script certifies any rung by changing one constant, but an
r-uniform enclosure (interval arithmetic in r, or a certified continuity modulus of the
ridge data in r) is NOT established: nothing here asserts the maxima's positions or
values at rungs other than 0.025. Second, the zero-above-b-saddle census is a statement
about the *mean* m₉; the corresponding count for the noisy conditioned field f̃ in the
ridge annulus remains LB-1's Lemma-H-SUP bound with its named on-grid analyticity
formality (house grade of C027/C031) — this deliverable certifies the deterministic
backbone (mean census, uniqueness balls, persistence margins) against which that
formality operates, and does not re-derive it. Neither gap touches the assembly's
consumption: the ridge channel was closed by LB-2 at the frozen rungs, and the present
margins (definiteness floors ≥ 1.9589, uniqueness balls ≥ 0.080, value gaps ≥ 0.4666)
are macroscopic and ℓ-free at the certified rung.

---
*KIMI-DER-027c, AO48-WO-063 Task 4c. Verdict: PROVED. No status or Boolean claims.*
