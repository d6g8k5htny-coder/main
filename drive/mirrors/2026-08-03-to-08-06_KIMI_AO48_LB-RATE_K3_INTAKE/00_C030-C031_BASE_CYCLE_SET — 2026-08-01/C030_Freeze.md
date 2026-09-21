# C030 — Freeze: the Two Counting Lemmas (WP and KR-MB) + the Density-Floor Mini-Lemma
**Date:** 2026-07-10. **Purpose:** convert C029-B3's main channel and C025-R2's near term to derived grade.

## Lemma WP (window-pass count, second-order Palm)
E[N_ws(A) | 9 pins] for A ⊆ B_3: intensity λ₉ws(y′) = φ∇(0|9)·φ_f(b | 9, ∇=0)·ℓ·E[|det H|1_sad | 9, ∇=0, f=b]·(1+O(ℓ)) — the band factor extracted analytically (window width ℓ), the rest a 9-pin Schur + typed Hessian expectation at y′. Two zones: in-ball B_2.5r∖collars (expect double suppression: area r² × band ℓ) and fixed zone 0.5 ≤ |y′| ≤ 3 (expect → unconditional per Lemma FD). Deliverable: station table at rungs, zone bounds C_wp-in·r⁵-class + C_wp-far·r³.

## Lemma KR-MB (band-max count)
Same structure, max-typing, band (v*, b] width ℓ/2, over B_3. Fixed-zone core: the UNCONDITIONAL band-max density for BF, derived: at a point, ∇f ⟂ (f, H); H | f=u has mean −uI, Cov: Var(Hxx|f)=Var(Hyy|f)=2 independent, Hxy~N(0,1); ρ_mx(u) = (1/2π)·E[|det H|·1_{H≺0} | f=u]·φ(u). Assembly: E[N_maxband(B_3)|9] ≤ [near, station-measured] + ρ_mx(b)·(ℓ/2)·Area(0.5–3)·(1+TV 2.2%) + [intermediate sup × area].

## Mini-lemma DF (density floor/ceiling, shared)
v(y′) := Var(f(y′) | 9 pins, ∇f(y′)=0) at dist ≥ δ from pins: values at stations; FD limit v → 1 − 0 = 1 at separation (f ⟂ own ∇). Ceiling on φ_f(b|·) = 1/√(2πv)·e^{...} follows.

## Committed gates
- **G-F1 (unconditional density, two routes):** route (a) the conditional-Gaussian reduction + 2e6-draw MC of E[|det|1_max|f=u]; route (b) direct counting on ≥40 synthesized unconditional fields (torus L=24, N=512; synthesis gates: Var within derived band, K(1) within band), band (1.1, 1.2], density = count/(area·width). Agreement within max(10%, 3·combined se).
- **G-F2 (machinery + scaling):** at fixed stations d ∈ {1, 2, 3}: λ₉ws(r=0.05)/λ₉ws(r=0.025) ∈ [6.5, 9.5] (the ℓ ratio 8); at d = 3: λ₉ws within 3× of the unconditional saddle-band analog (FD-consistency; wide band for MC + TV).
- **G-F3 (assembly consistency):** the KR-MB near+zone assembly must be consistent with P-A3's measured 8.9e-9 relative channel and with C029's zero extra maxima (predicted P(≥1) ≤ E[N_in-ball-class] ≪ 1/500).
- Vacuity: MC draws ≥ 2e4 per station; ≥ 6 stations; fields ≥ 40 route (b).

## Labels
WP and KR-MB: derived-structure with measured constants (the Kac–Rice regularity/validity conditions and the collar exclusion argument are NAMED formalities, rigidity-class). DF: exact-at-stations. No novelty claims (no network). BF, b = 6/5.
