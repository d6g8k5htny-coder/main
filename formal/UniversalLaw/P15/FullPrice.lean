/-!
# P15 full transformed-price budget — endpoint and certificate arithmetic

Informal sources (Layer 0, byte-pinned), both at `Math-` commit
`760340e921ac4ceda296b8118da936f1133e956e`:

* `frontiers/full_price_20260924/PROOF.md`, 11352 bytes, SHA-256
  `87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9`
  (object P15-FULL-TRANSFORMED-PRICE-20260924-v1, author OpenAI / ChatGPT):
  the published 20-digit endpoints of `rho_star` in display (F3), the comparison
  with `6/7` and `16/27`, the block-size inequality `n ≥ 2a + 1`, and the short
  rational certificates of section 6.
* `frontiers/price_budget_20260924/PROOF.md`, 7935 bytes, SHA-256
  `3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535`
  (object P15-TRANSFORMED-PRICE-BUDGET-20260924-v1, author OpenAI / ChatGPT):
  the ratio chain (T3) at its equality instance `a = 1, n = 3` and at finitely
  many further instances, and the explicit rational price of section 4.

The endpoints are transcribed as natural numbers in units of `10^-20`. Core
Lean's `Rat` is not kernel-decidable by `decide` (its `DecidableEq` instance does
not reduce), so every rational statement is the equivalent cross-multiplied
statement over `Nat` with the rational form in the docstring.

What this module does **not** establish: that `rho_star = 1/(3 − log(3e − 2))`
lies between the published endpoints (that needs `Real.exp`, `Real.log` and an
enclosure, which belong to the Mathlib lane), `e < 87/32`, `197/32 < exp(11/6)`,
`h_star > 7/6`, `rho_star < 6/7` as a statement about the real number, the
hazard interpolation lemma (F4)–(F8), the binomial tail comparison (F9)–(F11),
Theorem F (F2), Theorem (T1), the inequalities (T2) and (T3) in general, the
admissibility `c <= phi(p)` of the explicit price, `p_star = 1 − exp(−1)`, or any
prize status. Nothing here changes any Layer 0 status.
-/

namespace UniversalLaw.P15

/-- `0.84547981724898672067` and `0.84547981724898672068` in units of `10^-20`:
the published lower and upper endpoints of display (F3). -/
def rhoLower : Nat := 84547981724898672067
def rhoUpper : Nat := 84547981724898672068

/-- Factorial on `Nat`; kept local so that the library has no dependencies. -/
def fact : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * fact n

/-- Numerator of the degree-5 Taylor partial sum of `exp(11/6)` after clearing
the denominator `6^5 · 5! = 933120`: `∑_{j=0}^{5} 11^j · 6^(5-j) · 5!/j!`. -/
def expElevenSixthsNumerator : Nat :=
  (List.range 6).foldl (fun acc j => acc + 11 ^ j * 6 ^ (5 - j) * (fact 5 / fact j)) 0

/-- Source (F3): "0.84547981724898672067 < rho_star < 0.84547981724898672068 < 6/7".
The published endpoints are distinct and adjacent on the 20-digit grid. -/
theorem rho_star_endpoints_adjacent : rhoLower < rhoUpper ∧ rhoUpper = rhoLower + 1 := by decide

/-- Source (F3): "< 0.84547981724898672068 < 6/7". The upper endpoint is below
`6/7`: `rhoUpper / 10^20 < 6/7 ⇔ rhoUpper · 7 < 6 · 10^20`. -/
theorem rho_star_upper_endpoint_below_six_sevenths : rhoUpper * 7 < 6 * 10 ^ 20 := by decide

/-- Source: "The prior probability<=1/4 theorem ... gives the stronger factor
16/27 on its smaller domain" (source: "factor16/27"). Stronger means smaller:
`16/27 < 6/7 ⇔ 16 · 7 < 6 · 27`, and `16/27` is also below the published lower
endpoint: `16/27 < rhoLower / 10^20 ⇔ 16 · 10^20 < 27 · rhoLower`. The second
comparison is not displayed in the source; it is implied by the displayed numbers. -/
theorem sixteen_twentysevenths_below_six_sevenths :
    16 * 7 < 6 * 27 ∧ 16 * 10 ^ 20 < 27 * rhoLower := by decide

/-- Source: "Our n=ad+1 and d>=2 imply n>=2a+1." and (section 5) "Take one block,
a=1,d=2,n=3". Checked: `1 · 2 + 1 = 3` and `2a + 1 ≤ ad + 1` at five `(a, d)`
instances with `d ≥ 2`. The general implication is not checked. -/
theorem minimal_block_size_instances :
    1 * 2 + 1 = 3 ∧
    ∀ p, p ∈ [(1, 2), (1, 3), (2, 2), (3, 5), (4, 2)] → 2 * p.1 + 1 ≤ p.1 * p.2 + 1 := by
  decide

/-- Source (section 6): "e < 31967/11760 < 87/32, 87/32-31967/11760=11/23520."
Checked: `31967/11760 < 87/32 ⇔ 31967 · 32 < 87 · 11760`, and the difference
`87/32 − 31967/11760 = 11/23520 ⇔ (87·11760 − 31967·32) · 23520 = 11 · (32·11760)`.
The bound `e < 31967/11760` (series plus geometric tail) is not checked. -/
theorem e_rational_bounds_ordering :
    31967 * 32 < 87 * 11760 ∧ (87 * 11760 - 31967 * 32) * 23520 = 11 * (32 * 11760) := by
  decide

/-- Source (section 6): "sum_(j=0)^5 (11/6)^j/j! -197/32=26081/933120>0".
Over the common denominator `933120 = 6^5 · 5! = 32 · 29160`:
the partial sum numerator equals `197 · 29160 + 26081`, i.e. the difference is
`26081/933120 > 0`. That the partial sum is below `exp(11/6)` is analysis. -/
theorem exp_eleven_sixths_taylor_certificate :
    6 ^ 5 * fact 5 = 933120 ∧ 32 * 29160 = 933120 ∧
    expElevenSixthsNumerator = 197 * 29160 + 26081 ∧ 0 < 26081 := by decide

/-- Source (section 6): "Therefore 3e-2<197/32<exp(11/6), giving h_star>7/6 and
rho_star<6/7." The two rational constants: `3 · (87/32) − 2 = 197/32 ⇔
3 · 87 − 2 · 32 = 197`, and `3 − 11/6 = 7/6 ⇔ 3 · 6 − 11 = 7`. The second identity
is not displayed in the source; it is the arithmetic behind `h_star > 7/6` once
`log(3e − 2) < 11/6`. The inequalities involving `e`, `exp`, `log`, `h_star` and
`rho_star` are not checked. -/
theorem h_star_rational_certificate_constants : 3 * 87 - 2 * 32 = 197 ∧ 3 * 6 - 11 = 7 := by
  decide

/-- Source (T3, price_budget): "= (4^(a+1))/(3^n) <= (4/3)*(4/9)^a <=16/27" at the
equality instance `a = 1`, `n = 3`: `(1/4)^(n−a−1)/(3/4)^n = 4^(a+1)/3^n` needs
`4^n = 4^(n−a−1) · 4^(a+1)`, i.e. `4^3 = 4^1 · 4^2`; then `4^2 = 16`, `3^3 = 27`,
and `(4/3)(4/9)^1 = 16/27` is `4 · 4^1 = 16`, `3 · 9^1 = 27`. -/
theorem t3_ratio_at_minimal_block :
    4 ^ 3 = 4 ^ (3 - 1 - 1) * 4 ^ (1 + 1) ∧ 4 ^ (1 + 1) = 16 ∧ 3 ^ 3 = 27 ∧
    4 * 4 ^ 1 = 16 ∧ 3 * 9 ^ 1 = 27 := by decide

/-- Source (T3, price_budget): "<= (4/3)*(4/9)^a <=16/27". Finite instances:
`4^(a+1)/3^n ≤ 16/27 ⇔ 4^(a+1) · 27 ≤ 16 · 3^n` at seven `(a, n)` pairs with
`n ≥ 2a + 1`, and `(4/3)(4/9)^a ≤ 16/27 ⇔ 4 · 4^a · 27 ≤ 16 · (3 · 9^a)` for
`a = 1, …, 6`. The inequalities for all `a ≥ 1`, `n ≥ 2a + 1` are not checked. -/
theorem t3_bound_finite_instances :
    (∀ p, p ∈ [(1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (3, 10), (4, 9)] →
      4 ^ (p.1 + 1) * 27 ≤ 16 * 3 ^ p.2) ∧
    (∀ a, a ∈ [1, 2, 3, 4, 5, 6] → 4 * 4 ^ a * 27 ≤ 16 * (3 * 9 ^ a)) := by decide

/-- Source (section 4, price_budget): "An explicit rational price strictly larger
than p is c=p+p^2/2=81801/3345620000." At `p = 1/40900`:
`p + p^2/2 = (2·40900 + 1)/(2·40900^2) = 81801/3345620000`, and `c > p ⇔
81801 · 40900 > 3345620000`. Admissibility `c ≤ phi(p)` is not checked. -/
theorem explicit_rational_price_above_p :
    2 * 40900 + 1 = 81801 ∧ 2 * 40900 ^ 2 = 3345620000 ∧ 3345620000 < 81801 * 40900 := by
  decide

end UniversalLaw.P15
