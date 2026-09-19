# WP DISPOSITION — corrected window-pass channel report (K3 swarm, 2026-08-05)

Owner: lead coordinator, consolidating W2/LEAD (frozen chains), W4 (frozen estimate), W6 (frozen modulus),
W5 (frozen forensic), W10 (frozen composition), with W3's rigorous enclosure in flight. This document is
the corrected WP proof/obstruction report required by the delivery contract. It supersedes nothing
external; the defective prior artifacts are quarantined (00_READ_FIRST §1), not overwritten.

## 1. The estimand (verbatim zone)

ρ_WP(y; r) = p_{∇f̃(y)}(0) · E[ |det H_y| · 1{det H_y < 0} · 1{b−ℓ < f̃(y) < b} | ∇f̃(y) = 0 ],
b = 6/5, ℓ = r³/6, under the exact normalized periodized side-24 Bargmann–Fock covariance and the exact
pin/Palm construction (9 pins at M = (−r/2,0), S = (r/2,0), Y = M + r(−0.76,0.24); f(M) = b, f(S) = b−ℓ,
f(Y) = μ_t, gradients 0). The WP channel count is E⁰_r[N_ws(B₃ ∖ collars)] over B₃ ∖ collars; the rigidity
zone is {d ≲ 1.5} with d the distance from the pin cluster; collars are open disks of radius 2r about the
pin points (W5's verbatim extraction from the C030/C034 sources; "d" is never formally defined in the
sources — evidence: distance from cluster centroid; all bounds here are pointwise so validity is unaffected).

## 2. The defect and its confirmation (the repaired object)

The prior "certified upper bound" used p_grad·√(E[det²H])·min(P_type, P_window). The Cauchy–Schwarz chain
gives ...·√(P(type ∩ window)) ≤ ...·√(min(P_type, P_window)); since p ≤ √p on [0,1], the missing root makes
the reported quantity smaller — not a valid bound (witness-point confirmation: 1.31310e-4 vs 2.30247e-2,
175.3×). Two independent symbolic derivations (W2 frozen 26e2d1dd…; LEAD frozen) agree on the integrand and
on the valid inequality classes (gate G1 PASS): corrected CS with the inner root; Isserlis moments; exact
slice factorizations (F-slice and H-slice, both present in both documents); in-slice CS; Cantelli;
exact-mgf Chernoff / explicit Bernstein / Hanson–Wright tails; box+Šidák lower bounds. Dropped-root forms
and the unconditional isotropic independence are PROVEN INVALID with counterexamples in both documents.

## 3. Disposition of the WP channel for the assembly: CLOSED at the WP-min level

Per W10 (frozen 67c7fe63…), the assembly does NOT need WP = O(r³): it needs any valid explicit modulus
E_WP(r) → 0 covering (0, r₀] with E_WP(r₀) inside the AO budget. That requirement is MET:

**Lemma (WP-min, W6 frozen beda91f6…/b61835f0…).** E⁰_r[N_ws(B₃ ∖ collars)] ≤ I_CS(r) ≤ E_WP(r) = 4.2·r^{3/2}
on (0, 0.05], where I_CS(r) is the zone integral of the pointwise corrected-CS bound. Pointwise validity of
the corrected CS is PROVED (the bound exceeds the exact integrand at all 84 certified samples; worst ratio
exact/CS = 0.045); the 21-rung grid {0.001 … 0.05} is certified (scaled coefficient max 3.9697 ≤ 4.2);
E_WP(0.05) = 0.0470 ≤ 0.06 (budget). Collar neighborhoods contribute ≤ 1.5e-18. Two NAMED premises
(quantified, C-3 pattern, conspicuous): F1 (inter-rung log-slope |d log C/d log r| ≤ 0.327; measured
≤ 0.20 on all 20 steps), F2 (sub-0.001 tail |C(r) − C₀| ≤ 0.25 with C₀ ≤ 3.95; C₀ = 3.860069 now computed
DIRECTLY from the limit law), plus F3 (Bates-grade quadrature upgrade note). r-dependence mechanism
(certified): the window probability carries all r (∝ ℓ = r³/6) while the z-drift delays the r³ asymptote;
the apparent 1.45–1.6 slope on [0.005, 0.05] is the transient (U-turn at r ≈ 0.003).

## 4. Disposition of the old budget: REFUTED (evidence grade); formal refutation in flight

- The printed rigidity-zone figure ~3e-15 (3.6e-15) and the channel budget 0.213·r³ are REFUTED at
  evidence grade: two fully independent implementations give I_WP(rigidity zone, r = 0.025) = 4.7602e-6
  (lead spectral/GH u-slice) and 4.7569e-6 (W4 wrapped-theta real-space + direct transformed-coordinate
  quadrature) — agreement 0.07% — and W6's full-channel figure (B₃ ∖ collars) is 7.50e-6 = 0.48·r³.
  Reconciliation: the two 4.76e-6 figures are the rigidity zone only; 4.76 + outer shell 1.5866e-6 +
  far shell ~ 7.5e-6. All exceed 0.213·r³ = 3.328125e-6 (rigidity piece by 1.43×, full channel by 2.25×).
- A point-certified (interval-rigorous) hot spot anchors the mass: ρ(0, 0.60) ∈ [1.057e-4, 1.058e-4]
  (W3's certified g-quadrature; consistent with both estimate pipelines), while ρ(−0.04, −0.60) ≤ 1.945e-8
  (the −y witness is not dominant; the window condition there forces det H > 0).
- Formal refutation status: W3's box-integrated lower enclosure over the compact band x∈[−0.08,0.08],
  y∈[0.50,0.68] is IN FLIGHT (the driver is running with certified Taylor-model box laws and the
  documented remainders R1/R2/R3; W3 froze at OPEN on a throughput obstruction now repaired and resumed).
  Until that enclosure lands, the refutation is evidence-grade — stated exactly so.
- Why the old figure arose (W5 forensic, frozen): the rigidity figure is a SINGLE-STATION value
  (lam_sad at d = 1, one angle 2.35 rad) multiplied by the full disk area π·1.5² (= 3.5578e-15); the
  kill premise entered at C030 package line 5 licensed by line 11's zone-wide consequence (i); the
  station value is not a demonstrated zone sup (the mean exceeds b on ridge arcs — m(P*) = 1.93119035756,
  m(Q*) = 1.66659821930, certified cross-line to < 1e-9 — and the G-F2a d1 scaling gate FAILED-AS-WRITTEN
  and was waived). The budget 0.213·r³ decomposes exactly (transition 1.9513e-6 + far 1.3701e-6 +
  rigidity 3.5578e-15 = 3.32143e-6 = 0.2125714·r³, matching c030 assembly.json to 3.6e-12).

## 5. The bound landscape (what each valid bound buys)

- Corrected global CS (valid): zone integral 8.5743e-4 (W4) — too weak for the budget; not used in the
  assembly.
- W6's certified modulus (valid): 4.2·r^{3/2} — the assembly's WP closure (Section 3).
- W2/LEAD slice chains (valid, symbolic): pointwise min{UB_global, UB_slice}; at the probe (1.0, 0.3):
  UB_slice ≈ 5.0e-23 (11 orders headroom vs the rigidity budget; the point sits on a rim arc with
  μ_t = 1.4294 > b). Cross-validated by the lead (5.76e-12 global / 8.2e-23 slice, same orders).
- W3's interval enclosures (in flight): point-certified boxes; the formal lower bound for the refutation.

## 6. Executable falsifiers

- WP-min modulus: a certified window-saddle count or exact integrand evaluation exceeding the corrected-CS
  bound at any certified sample point, or a rung value violating C(r) ≤ 4.2, kills Section 3 (rerun
  W6's certificates: any nonzero exit).
- Evidence-grade refutation: a rigorous upper enclosure of the rigidity-zone integral below 3.328125e-6
  would overturn Section 4's evidence (currently contradicted by two independent estimates and the
  point-certified hot spot).
- Formal refutation (when landed): an independent interval-rigorous recomputation of the lower band
  enclosure disagreeing beyond the documented remainders.

## 7. What remains OPEN (named, minimal)

- The box-integrated rigorous lower enclosure (formal refutation) — W3 driver running; executable falsifier
  path documented (fix path (a)–(d) of W3's status; band x∈[−0.08,0.08], y∈[0.50,0.68] at 0.0025 boxes).
- The interval-certified (Bates-grade) zone quadrature for I_CS replacing estimate+refinement grade (F3).
- The limit-object (r = 0) closure of the ridge/collar rung chain (C012/C013/C026 dependency; W9c).

## UPDATE (2026-08-05, post red-team): WP-min DELIVERED at the exact-integrand grade

Section 3's modulus is SUPERSEDED by the repaired form (W6_REPORT 752bd2eb..., w6_wp2.py, exit 0 both
modes byte-identical):

    E^0_r[N_ws(B3 \ collars)] = Integral_zone rho_exact <= Integral_{B3} rho_exact = I_WP(r)
    <= E_WP(r) = 3.5e-3 * r^{3/2}   on (0, 0.05],

with validity PROVED (Kac-Rice equality for the frozen G1 integrand + zone subseteq B3 with rho >= 0 -
NO Cauchy-Schwarz loss, ~1000x tighter than the CS form), the 11-rung grid certified
(C_I = I_WP/r^{3/2} = 3.202e-3 sup at r0 -> 1.20e-3, small-r monotone decreasing), and
E_WP(0.05) = 3.91e-5 <= 0.06 (margin x1533). Coverage is completed by a SINGLE named premise P-mono
(C-3 pattern): d log I_WP/d log r >= 3/2 on (0, 0.05]; measured discrete log-slopes 1.66-2.37,
asymptote -> 3. The earlier CS modulus 4.2*r^{3/2} was REFUTED (spurious near-cluster singularity,
self-caught + red-team B1) and is superseded; B1/B3 closed. Gate G6: SATISFIED at the W10 spec
(valid explicit vanishing modulus covering (0, r0] with named premise conspicuous).
