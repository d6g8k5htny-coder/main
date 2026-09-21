# C031 — Lemma LB-RATE: the Integration Document of Record
**Freeze:** e165821b1479f619af8ebdb1438460edfebe88f44c189553ca86f73f8fe1afbe. **Scope:** Bargmann–Fock kernel K = exp(−|u|²/2), b = 6/5, ℓ = r³/6, arch station ỹ* = (−0.76, 0.24); rungs r ∈ {0.05, 0.025} unless stated. This document assembles the lower half of Theorem A's rate from the cycle artifacts C022–C030, states every term at its exact grade, registers two composition findings discovered during assembly (one repaired this cycle, one named), and lists the complete named-formality set. Nothing here re-derives a registered result; every grade cites its cycle.

## 1. Target and reduction chain
**Target (lower half of Θ(r³)):** 1 − q(r, b) ≥ c·r³ for small r, with c explicit or measured-with-derived-structure.

**Reduction (C024, freeze 0df10fa6…139, 14/14-verified instrument):** 1 − q ≥ E[N_qual]·(1 − O(r³)) by Bonferroni (pair term O(r⁶) registered at C022), with E[N_qual] = ∫ Λ(y)·AO(y) dA over the arch region. The estimand, verbatim as frozen at C024: Λ(y) = φ₂(∇f(y)=0 | 𝒥₆, with mean) · [Φ((b−μ_t)/s_t) − Φ((b−ℓ−μ_t)/s_t)] · E[W₃1_typed | 9 pins @ v* = clip(μ_t)] / E[W₂1_typed | 𝒥₆]; typing (detM>0, trM<0, detS<0, detY<0); window (b−ℓ, b); pin values ((b,0,0), (b−ℓ,0,0), (v*,0,0)). So LB-RATE = [Λ-side rate: ∫Λ dA ≥ c_Λ·r³] × [AO-side floor: inf AO ≥ c_AO > 0], composed through the C024 reduction.

## 2. The Λ-side: status of the factor limits (S1–S6)
C024 reduced the Λ-side to six factor-limit statements S1–S6 plus the AO floor. Their disposition (C026 freeze 6ac9a33c…7df; C027 freeze f0833443…dc2, package c9d1466a…3b4):

- **Exact (rational, regularizing frame, C026):** the S-limits including den∞ = 3.230979 (closed form) and the numerator product lemma num∞ = |∏c|. **S3 FAILED-AS-WRITTEN and was superseded by the exact value:** m = lim (μ_t − b)/ℓ = −0.4999290, *not* −1/2; the −1/2 conjecture is dead, the exact constant lives (kill registered at C026).
- **Derived-on-grid (C027):** the assembled station integrand λ(ỹ) and its integral **C*_∞ = ∫λ(ỹ) dỹ = 0.9091 ± 0.038 (quad) ± 0.009 (shell)** over a 730-station grid; λ(ỹ*) = 3.4755. Radius certificates: 6-pin r₀ = 0.1011 (proved); 9-pin r₀ = 0.0165 (proved), extended empirically to r = 0.05 via a 1e-13 function identity (named: 9-pin certificate tightening).
- **Measured, with derived structure:** C*(0.05) = 0.9728, C*(0.025) = 0.946 — approach to the limit is clean O(r) (successive-difference ratio 1.73 ≈ 2); all 9-pin mean-Hessian entries are O(r) with tr M coefficient −13.5327 (C027).

**Λ-side grade:** ∫Λ dA = C*·r³ with C* measured at rungs and its r→0 limit derived-on-grid (0.9091). The r³ scaling itself is measured (two-rung ratio 8.23 ≈ 8, C021) with the factor localization derived at C024–C026 (the ℓ window factor carries the rate; the remaining factors have exact limits). The fully rigorous r-uniform lower rate on ∫Λ remains the named residue of this side (grid → continuum: quadrature + shell errors are quantified; the formality is the station-density argument, same class as C027's grade).

## 3. The AO-side: the C025 architecture and per-term grades
**Registered verbatim (C025, freeze 0f8cb3dc…60b):** AO-failure ⊆ {non-adjacency} ∪ {terminal ≤ b}; (R1) the ascent from the arch saddle's outward branch is monotone with height ≥ v*; (R2) P(terminal ≤ b) ≤ E[N_maxband(B_d₀) | 9 pins] + far band-termination; (R3) Lemma FD: conditional-vs-unconditional TV ≤ 2.2% at d ≥ 3, ≤ 1e-4 at d = 5; (R4) OBL-FAR-ASCENT(δ₀), r-free. Assembly as registered: AO ≥ 1 − P(non-adj) − E[N_maxband] − p̄(δ₀) − TV(d₀).

**Per-term grade table (d₀ = 5 per §4's finding G1):**

| Term | Bound | Grade | Cycle artifact | Named residue |
|---|---|---|---|---|
| R1 monotone ascent | exact | derived (gradient-flow monotonicity) | C025 | — |
| Non-adjacency: capture | 27/27 fan certificates (±20°); 3/3 rung skeletons | certified | C025 | — |
| Non-adjacency: measured | capture = 1.0000 (1750/1750 at each of r = 0.15, 0.10; ±20° fans); extra maxima > v*: 0/500; diversion 0; exit 0 | measured on the mp-exact instrument (negmass 2e-15) | C029 (freeze ddd8596f…0ca) | — |
| Non-adjacency: BP reduction | BP-failure ⊆ window-pass ∪ above-b saddle ∪ ridge split ∪ exit | derived (mountain-pass B1–B4) | C029 | — |
| — window-pass channel | E[N_ws(B₃∖collars) \| 9] ≤ **0.213·r³** (rigidity 3.6e-15 + transition 1.95e-6 + far 1.37e-6 at r = 0.025; ℓ-ratio verified 7.76/8.19 ≈ 8) | derived-structure + measured constants (Lemma WP) | C030 (freeze ae20f4e3…f4b) | KR validity; collar exclusion |
| — above-b saddle | superexponentially killed throughout the jet-cluster horizon (v = 0.018/0.55 at d = 1/2; kill e^{−(b−m)²/2v}) — mechanism identified | derived mechanism; closure named | C030 | sup-over-zone argument |
| — ridge split | — | named | C029 | mean-ridge connectivity |
| R2 near term | E[N_maxband(B₅) \| 9] ≤ 0.76·(ℓ/2)·(25−6.25)/(9−6.25) ≈ **2.1·(ℓ/2)** (fixed-zone core DERIVED: ρ_mx(1.2) = 0.043685, E-term 1.41350 ± 0.0017; G-F1 two-route PASS 9.5%) | derived-structure + measured constants (Lemma KR-MB) | C030 | KR validity; collar exclusion |
| R3 quantitative | TV ≤ 2.2% (d ≥ 3), ≤ 1e-4 (d = 5) | proven (Lemma FD) | C022 | — |
| **R3′ positivity transfer** | conditional-barrier: total C¹ perturbation on the d₀ = 5 slab = 0.00363 ≤ gate 0.1307 (36× margin); exit slack 0.1746; ‖m₉‖_H = 3.636; ‖ũ₀′‖ ≤ ‖ũ₀‖ (projection) | **CLOSED this cycle at C028 grade** | C031 (this document; c031_verify.json) | station density on the slab + analyticity (on-grid class, as C027) |
| R4 far ascent | p̄(1/5) ≤ 1 − c, c = e^{−4.82550294/2}·P₀ = 0.089569·P₀ > 0 (support theorem + CM + Anderson; gates G-D1–D4a passed; G-D4b honest null: P₀ < 3.2e-4 at 95%) | theorem (explicit construction) | C028 (freeze 4a343a18…d0f) | OBL-P0-FLOOR (optional) |
| R4 measured | p̄(0.20/0.10/0.05/0.02) = 0.1242/0.0687/0.0334/0.0141 | measured (ν_T instrument) | C025 | — |

## 4. The two composition findings (the integration cycle's own contribution)
**G1 — additive-TV vacuity at theorem grade: FOUND AND REPAIRED (R3′).** As registered, the assembly subtracts TV additively: AO ≥ c − TV − O(r³) with c the C028 theorem constant. Since c = 0.0896·P₀ with P₀ plausibly 1e-4–1e-6-class, c ≪ TV(d₀=3) = 0.022 and the theorem-grade assembly as written is vacuous (negative). **Repair (closed this cycle):** route the barrier *through* the conditioning. The support-theorem positivity argument is conditioning-robust for nondegenerate Gaussians: the conditional law given the 9 pins is Gaussian with CM space {h ∈ H : pins(h) = 0}; the shift ũ₀′ = ũ₀ − Proj_pins(ũ₀) satisfies the pin constraints with ‖ũ₀′‖ ≤ ‖ũ₀‖ (CM factor no worse), and the barrier's C¹ ramp property survives because the two induced perturbations — the conditional mean m₉ and the projection field — have C¹ size on the d₀ = 5 slab bounded by max(|m₉|, |∇m₉|) + ‖ũ₀‖·√(1−v_grad) = 0.0016 + 0.0020 = **0.0036, a 36× underrun of the 0.1307 unallocated margin** (exact mp station table, c031_verify.json). The horizon phenomenon (C030) is precisely what made d₀ = 3 dangerous and d₀ = 5 safe: the pull scales as ‖·‖_H·√(1−v), and v = 1 − O(1e-7) at d = 5. Consequence: theorem-grade far-term positivity needs **no TV subtraction**; Lemma FD is repositioned as the quantitative channel only. Moving d₀ from 3 to 5 raises the R2 constant by the area factor to ≈ 2.1·(ℓ/2) — still O(r³), immaterial.

**G2 — entry-point selection/multiplicity: FOUND AND REGISTERED (OBL-FAR-COMPOSE).** The far term concerns the trajectory's crossing of ∂B_{d₀}. Entries with f(ξ) > b win outright; only band entries f(ξ) ∈ [v*, b] can be bad. Because BF sample paths are a.s. analytic (the same fact behind the C030 horizon), conditioning on the trajectory's past region determines the far field — naive "restart at the crossing point" Palm arguments are therefore delicate and are NOT used. Two rigorous routes, both registered:
(i) *Union bound over band components:* E[#components of the band on ∂B₅] ≤ Rice up-crossings of v* ≈ b: (1/2π)√(λ₂/λ₀)·e^{−b²/2}·2π·5 = **2.43** (+1 wrap) — derived this cycle (V1). Measured-grade composition: far ≤ 2.43·p̄(0.05)·(1 + 1e-4) = 2.43·0.0334 ≈ **0.081**. Theorem-grade via this route requires c > 1 − 1/2.43 ≈ 0.59 — far beyond the C028 barrier class; route (i) is measured-grade only.
(ii) *Selection/disintegration lemma:* make the single-crossing Palm bound rigorous (the crossing point is field-selected; a disintegration over the crossing location with a local-conditioning Palm kernel would give far ≤ sup-Palm p̄ with no multiplicity factor). This is the theorem-grade route and is OPEN. **OBL-FAR-COMPOSE registered with both routes; the theorem-grade AO assembly below is conditional on it.**

## 5. Assembled bounds
**Measured grade (r = 0.025, d₀ = 5):** AO ≥ 1 − [non-adj: 0 measured, channels ≤ 0.21·r³ + named] − [2.1·(ℓ/2) = 3.4e-6] − [far ≤ 0.081] − [TV 1e-4] ≥ **0.918**. Composed with the Λ-side: **1 − q ≥ 0.918 · 0.946 · r³ · (1 − O(r³)) ≈ 0.87·r³** — the measured-grade LB-RATE, consistent with the direct two-sided measurement C* ≈ 0.96 ± 0.06 (C021) since AO there is measured ≈ 1 rather than bounded by the assembly's conservative far term.

**Theorem grade (conditional):** AO ≥ c_cond − 2.1·(ℓ/2) − [named non-adjacency channels] with c_cond > 0 the R3′ conditional-barrier constant — **non-vacuous and positive for small r**, conditional on: OBL-FAR-COMPOSE route (ii); the sup-over-zone step (above-b saddle); mean-ridge connectivity; KR validity + collar exclusion; and the R0/γ-LOC architecture inherited by every AO statement. Composed with the Λ-side: 1 − q ≥ c_cond·c_Λ·r³ conditional on the same set plus the Λ-side grid formality. **The lower half of Θ(r³) is therefore: measured at 0.87·r³ with derived structure throughout; theorem-grade modulo the named set below, with the analytic skeleton now complete** (positivity constants explicit at every node; no term is bare measurement without a derived-structure bound).

## 6. The complete named-formality register (as of C031)
1. **R0 / MS–Sard covering** (derived-architecture since C015) — inherited by all AO statements.
2. **γ-LOC tube-local** (architecture) — inherited by the C022 positivity chain.
3. **(ND′) chain note** — the C022/C023 upper-side ceilings inherit (ND′) (proven under moment conditions; chain stated, not free-standing).
4. **KR validity + collar exclusion** (Lemmas WP, KR-MB; rigidity-class).
5. **Sup-over-zone** (converts the horizon kill into the closed above-b-saddle piece).
6. **Mean-ridge connectivity** (C029 B-chain).
7. **OBL-FAR-COMPOSE** (new, this cycle; routes (i)/(ii) registered).
8. **OBL-P0-FLOOR** (optional; C028 honest null).
9. **OBL-BETA-RELEVANCE** (upper side, C023; the r-uniform *upper* rate — outside LB-RATE's scope but listed for the two-sided statement).
10. **9-pin certificate tightening** (r₀ = 0.0165 → 0.05 empirical).
11. **Collar residual |ỹ| < 0.15** (C027 shell term, quantified ±0.009).
12. **Slab station density + analyticity** (R3′, on-grid class).
13. **Prior-art searches** (network-gated; MANDATORY before novelty language for: the horizon phenomenon vs analytic-kernel prediction theory; the thermodynamic-formalism package; the ΔJ parity law) — no novelty claims are made in this document.

## 7. Constants table (single source)
b = 6/5; ℓ = r³/6; ỹ* = (−0.76, 0.24); m = −0.4999290; den∞ = 3.230979; λ(ỹ*) = 3.4755; **C*_∞ = 0.9091 ± 0.038**; C*(0.05/0.025) = 0.9728/0.946; two-rung ∫Λ ratio 8.23; ν_T: 0.1242/0.0687/0.0334/0.0141 at δ = 0.20/0.10/0.05/0.02; C028: ‖ũ₀‖² = 4.82550294, η = 0.1307 (max 0.2615), a = 0.8087, κ = 0.2679, gain 0.3753, CM factor 0.089569; capture 1.0000; WP ≤ 0.213·r³; KR-MB ≤ 0.76·(ℓ/2) (B₃), ≈ 2.1·(ℓ/2) (B₅); ρ_mx(1.2) = 0.043685; ρ_sad(1.2) = 0.030449; horizon v = 0.018/0.55/0.976 at d = 1/2/3; Rice 0.0775/unit, 2.43 on ∂B₅; ‖m₉‖_H = 3.636; R3′ margin 0.00363 vs 0.1307; measured AO floor 0.918; **measured LB-RATE ≥ 0.87·r³**.

## 8. Honest one-paragraph status
The lower half of Θ(r³) now has a complete two-level structure: a measured-grade bound 1 − q ≥ 0.87·r³ in which every term carries a derived-structure ceiling or an explicit-construction floor (nothing rests on bare measurement alone), and a theorem-grade skeleton — positivity constants explicit at every node via the C028 barrier transferred through the conditioning (R3′, closed this cycle at 36× margin) — whose remaining distance to a proof is exactly the named set of §6, dominated by OBL-FAR-COMPOSE route (ii) (the selection/disintegration lemma), the sup-over-zone step, and the two long-standing architecture items (R0, γ-LOC tube-local). The integration itself contributed the two findings it exists to find: one vacuity repaired within the cycle, one selection gap named with its Rice constant computed. The upper half of Θ(r³) is outside this document's scope and remains as C023 left it (rigorous per-rung anchors; r-uniform upper open as OBL-BETA-RELEVANCE).
