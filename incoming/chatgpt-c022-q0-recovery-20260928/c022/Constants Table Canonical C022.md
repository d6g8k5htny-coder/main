# The Constants of the r³ Law — canonical table as of C022 (2026-07-09)

Single source of truth; supersedes Constants_Table_Canonical_C021.md (priors retained; scope chains
explicit). b = 1.2, κ = 1, ℓ = r³/6, BF testbed. [PINNED]/[PALM] conventions as before.

**Headline (C022):** the r³ exponent is now **pinned two-sided at proof-adjacent grade**:
lower side by Lemma LB-ARCH (rigorous positivity, derived-and-verified, two cited-standard steps),
upper side by the measured integral + Lemma FD's certified envelope. Assembled:
**0 < c(r, b) ≤ (1 − q)/r³ ≤ 0.96 + 12.41 + ε ≈ 13.4** at the anchor rungs, with the sharp
constant **C\* = 0.96 ± 0.05** [measured, two rungs, clean gates]. The single load-bearing proof
item (OBL-LB-ARCH) is closed.

## Exact
(unchanged from the C021 edition: pair-Hessian Gram (1/540)r¹⁸; Lemma P Gram 13824; 9×9 free-jet
3,981,312; pinned laws; E|det H| = 2.414589; ρ_saddle/ρ_max = 0.030513/0.044111; octave 1/8;
ρ₆(0) = 0.196817; the λ disintegration identity; the Λ window-integral closed form; the near-M
null functional with λ_min = 5.4×10⁻²⁰ at r = 0.025.)
New exact-grade entry: **E[|det H| | f = b, ∇f = 0] = 2.41290** (unconditional level-b
Hessian-determinant mean, GH 80³; distinct object from the 2.414589 Palm constant — both stand).

## Certified (C022, mp dps 50–60, residual-verified)
| object | r = 0.05 | r = 0.025 | margin |
|---|---|---|---|
| C1 ∇-marginal λ_min | 1.156×10⁻⁶ | 8.939×10⁻⁸ | 10^60.8 |
| C2 needle s_t/ℓ; P_win | 0.0676; 1.000000 | 0.0864; 0.999999 | exact |
| C3 6-pin Hessian-cond λ_min | 2.603×10⁻¹⁰ | 4.068×10⁻¹² | 10^51–52 |
| C4 9-pin Hessian-cond λ_min | 1.948×10⁻¹³ | 3.095×10⁻¹⁴ | 10^51.6 |
| C5 9-pin Gram λ_min | 1.321×10⁻¹² | 3.333×10⁻¹⁴ | 10^47–49 |
| C6 18-pin 2-jet Gram λ_min | 9.727×10⁻²¹ | 3.728×10⁻²³ | 10^38–40 |
| barrier ‖g₀‖_H; max\|w\| | 3.4294; 4.96×10⁵ | 3.070; 5.36×10⁶ | — |
| barrier ε₀ (persistence) | ≈ 0.008 (macroscopic) | ≈ 0.003 | — |
| ‖G₆⁻¹v‖₁ (FD dual weights) | 6.556×10⁴ | 5.183×10⁵ | — |
| FD far ceiling C(d₀ = 0.3) | 12.41·r³ | 12.41·r³ | rung-independent |
| FD far ceiling C(d₀ = 2r) | 84.2·r³ | 334.9·r³ | per-rung |

## Measured — asymptotic-regime, two rungs [PALM] (unchanged from C021)
C\*(0.05) = 0.9702 → 0.9728 (ballfix) ± 0.055; C\*(0.025) = 0.9461 ± 0.066; combined
**C\* = 0.96 ± 0.05**; ∫Λ dA ratio 8.23 ≈ 8; qual/raw ≥ 0.9997; rim ~r^2.5; β dead both rungs
(support edge 300ℓ → 894ℓ); gates 6/6 at C021.

## Measured — moderate-r brackets (unchanged; phenomenology; Lemma LB scoped here per C022 Part B)

## The assembled statement (supersedes the C021 paragraph)
**1 − q(r, b) = Θ(r³): the exponent is two-sided at proof-adjacent grade — lower side
Lemma LB-ARCH [derived-and-verified; cited-standard: Bogachev 3.6.1, MS basin continuity], upper
side E[N] (measured 0.96·r³) + β-near (≤ 2.6×10⁻⁸ + 8×10⁻⁴²) + β-far (≤ 12.41·r³ certified,
d₀ = 0.3, rung-independent) + ledger (< 1.5×10⁻³ rel.). The sharp constant C\* = 0.96 ± 0.05 is
measured (two pre-registered rungs).** Chain of custody: LB0 [Proved] → R0 [+ R0-UNIF discharged]
→ Lemma LB-ARCH [C022] → Lemma FD [C022]. Remaining to full proof grade: OBL-BETA-MODZONE
(r-uniform moderate zone), basin-continuity formalization (if demanded), γ-LOC tube-local,
manuscript unification. Optional sharpenings: OBL-LB-RATE (exact limit objects).
