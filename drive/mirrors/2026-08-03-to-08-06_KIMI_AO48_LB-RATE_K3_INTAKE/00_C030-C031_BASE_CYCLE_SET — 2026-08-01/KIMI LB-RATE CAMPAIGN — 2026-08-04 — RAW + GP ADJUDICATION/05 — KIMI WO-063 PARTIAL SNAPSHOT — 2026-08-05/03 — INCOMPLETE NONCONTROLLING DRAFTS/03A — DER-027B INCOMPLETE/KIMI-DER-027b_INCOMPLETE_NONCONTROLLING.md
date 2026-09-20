# KIMI-DER-027b — Λ-GRID: grid coverage → theorem (AO48-WO-063 Task 4b, hypothesis H7 of KIMI-THM-023)

Work order: AO48-WO-063, sha256 e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2.
Certificate: verify_lambda_grid_v1.py (fail-closed; ck() raises SystemExit(1); transcripts byte-identical under python3 and python3 -O).
Precision labels: EXACT (rational/exact-formula values) vs decimal-dps-70 (mpmath at 70 decimal digits). All jet values carry the certified junk allowance 1e-12 (stated in the certificate header).

## 1. The scanned functional (exact object)

The functional scanned on the grid ladder is the C027-class limit intensity
λ(y) = [ e^{−q(y)/2} / (2π√c6v(y)) ] · Pwin(y) · |cM(y)·cS(y)·cY(y)| / DEN
on the typed support { cM>0, cS<0, cY<0, trc<0 } and λ = 0 outside, where
DEN = (b²+2)·Φ(b/√2) + √2·b·φ(b/√2), b = 6/5 (EXACT closed form).
DEN = 3.230978535287004948096246569018163150765 (decimal-dps-70).
The ingredients (c6v, q, s2, m, cM, cS, cY, trc) are leading-coefficient
extractions of the exact rational Laurent-series engine (C026/C027) at the
normalized periodized Bargmann–Fock field: spectral lattice (π/12)ℤ², masses
e^{−|k|²/2}, cutoff |k| ≤ 30; pins M=(0,0), S=(1,0); ℓ = r³/6.

Engine validation (all machine-checked by the certificate):
- spectral tail Σ_{|k|>30} e^{−|k|²/2}(1+|k|⁶) ≤ 2.87e-182 (decimal-dps-70) — the periodized field is the analytic kernel up to < 1e-60, as required;
- lattice moments B_0…B_6 match continuum moments (odd moments show the expected lattice discretization offsets, informational only);
- λ reproduces the archived C027 sweep core40 at 12 stations with relative difference 1.438e-07, exactly the archived DEN truncation (3.230979 vs the closed form);
- mirror symmetry λ(y1,−y2) = λ(y1,y2) to 0.000e+00 at 3 pairs;
- the gradient/Hessian jets were validated against central finite differences of the Fraction-exact engine during engine qualification (12 matched digits in ∇λ, 10 in D²λ).

## 2. Exact jets on the grid of record

For every station y of the grid of record (upper lobe; the lower lobe doubles by the certified mirror symmetry), the certificate computes the exact second-order jet
(λ(y), ∇λ(y), D²λ(y)) (decimal-dps-70, junk allowance 1e-12)
by forward automatic differentiation through the full Laurent-series DAG (divided-difference frames MON9A/MON9B, Neumann series inversions, Schur complements — no interval-box evaluation; structural-cancellation junk is ≤ 1e-17 at exact points and covered by the 1e-12 allowance).

Grid of record (zone Z = [−3,2]×[−1.6,1.6] both lobes; far-field outside Z is the named dependency KIMI-DER-027a):
- region C = [−0.9,−0.4]×[0.1,0.5] at h = 1/40 (EXACT);
- region M = [−1.5,−0.1]×[0,0.7] \ C at h = 1/20 (EXACT);
- region O = [−2,0.2]×[0,1] \ M at h = 1/10 (EXACT);
- region F = [−3,2]×[0,1.6] \ Z+ at h = 1/5 (EXACT).
(Region census, typed/boundary/kink/deep cell counts: see transcript.)

## 3. The modulus bound and the net-spacing inequality

Per cell (side h, inradius ρ = h/√2) with pure-typed interior the second-order midpoint certificate gives
|∫_cell λ − h²λ(c)| ≤ (σ_c + B₃·ρ)·h⁴/12,
off-grid excursion sup_{y∈cell}|λ(y) − λ(c)| ≤ |∇λ(c)|·ρ + ½(σ_c+B₃ρ)ρ² + ⅙B₃ρ³,
where σ_c = σ_max(D²λ(c)) is the exact Hessian spectral norm at the cell centre and B₃ is the regional third-derivative constant (§4). Cells crossing the typing boundary or the clip kink m ∈ {−1,0} (machine-flagged with margin factor MAR=3) use the C¹ fallback (|∇λ(c)| + σ_c·ρ)·ρ·h²; deep-untyped cells contribute 0 (margins machine-checked).

Net-spacing inequality (assembled, both lobes): E_cert = 2·Σ_regions (E2_R + E_fallback_R) with the numerical value in the transcript, and the excursion form E_exc(h) ≤ (h/√2)·Λ₁ with Λ₁ = 2·Σ_R L_R·A_R (transcript).

## 4. The B₃ constants and hypothesis H-B3

B₃,region = SF · Q_region with SF = 8 (EXACT), where Q_region is the maximum third-difference quotient of the exact Hessian jets over the region's interior stations: Q = max_{p,i} ‖D²λ(p+h·e_i) − D²λ(p−h·e_i)‖_F/(2h).
Hypothesis H-B3 (named, falsifiable): on each region, sup|D³λ| ≤ SF·Q_region.
Self-consistency evidence (machine-checked): the quotient computed on the auxiliary 1/20 grid over the C-rectangle agrees with the 1/40 quotient within the certified factor (ratio in the transcript, required in [0.4, 2.5]).
Falsifier: any point y in a region with σ(D³λ(y)) > B₃,region, computable by direct third-jet evaluation; or failure of the rung-stability check.
All other constants in this derivation are exact-jet quantities; H-B3 is the single named premise, in the KIMI-DER-009 naming discipline.

## 5. Margins, result, kill condition

Margins: M* = 0.066 (decimal, C021 measured-grade uncertainty budget, used as margin only); M_floor = 0.9091 − 0.009 = 0.9001 (C037 floor constant minus shell).
Kill condition (uniform scan spacing): the excursion inequality E_exc(h) ≤ (h/√2)·Λ₁ ≤ M fails for h > h_kill(M) = √2·M/Λ₁. Numerical values (h_kill(M*), h_kill(M_floor), E_cert, per-region constants) are printed by the certificate — see §7 once transcripts are attached.

## 6. Dependency table

- C020/C021 Observed Update.json + Constants Tables (Λ window probability c_Λ = 0.946 measured-grade, budget 0.066) — margin input only.
- C026 Foundation Package + c026_series.py (Laurent-series engine) — reimplemented (dag engine), validated vs archives.
- C027 Foundation Package + c027_station.py + c027 final.json (DEN closed form 3.230978535287005) + c027 sweep core40.json (cross-validation).
- C037 GridContinuum Package (E_hier = 0.0975, sup|D²λ| ≈ 1500 measured, floor constant 0.9091, shell 0.009) — margin input and comparison.
- KIMI-DER-027a far-field envelope (outside zone Z) — named dependency.
File hashes: to be attached by the lead's packaging (engine files jets/dag are inlined in verify_lambda_grid_v1.py; the certificate is self-contained).

## 7. Verdict

(to be completed from the transcripts; expected form: PROVED at floor tier conditional on H-B3 — E_cert < M_floor — with the measured tier M* not discharged at the grid of record, the kill condition h_kill displayed, and the falsifier named in §4.)

## 8. Remaining gap (one paragraph)

(to be completed: the single named premise H-B3 — a regional third-derivative constant certified by exact third-difference quotients with safety factor 8 and machine-checked rung stability — is the only non-jet-exact input; every other link (field periodization tail < 1e-60, DEN closed form, functional values, gradients, Hessians on the full grid of record, cell classification, assemblies, kill condition) is machine-certified at 70 dps with allowance 1e-12.)
