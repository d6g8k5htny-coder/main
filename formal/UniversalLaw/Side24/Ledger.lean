/-!
# SIDE24 coefficient note — analytic-image ledger, arithmetic skeleton

Informal source (Layer 0, byte-pinned): `Math-` commit
`9d7b6802424fb4715b31999066aafca8ee2f3cca`, `coefficients/side24_v1/PROOF.md`,
SHA-256 `c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769`,
sections 2–4, and the `image_ledger()` function of `coefficient.py`.

What this module establishes: every *exact integer or rational* identity and
inequality that sections 2–4 of the note rely on is checked by the Lean kernel.
No `sorry`, no `native_decide`, no axioms (see `UniversalLaw/Audit.lean`).

What this module does **not** establish: the analytic facts the arithmetic is
attached to — the Gaussian derivative bound `76|x|^6 e^{-|x|^2/2}` for `|x| ≥ 1`,
the geometric-tail argument, the positive-semidefinite covariance comparison,
the Schur-complement monotonicity, the Gaussian density comparison, and the
`log`/`exp` estimates that turn exponents into the factor `32ε`. Those remain
author-side prose under nonauthor review (Layer 0). Nothing here changes the
scientific status of the coefficient or of its parent theorem.
-/

namespace UniversalLaw.Side24

/-- Factorial on `Nat`; kept local so that the library has no dependencies. -/
def fact : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * fact n

/-- Product-rule pairing count for `q` unit directional derivatives of
`exp(-|x|^2/2)`: `∑_{j=0}^{⌊q/2⌋} q! / (2^j j! (q-2j)!)`. Section 2 of the note
uses its value at `q = 6` and the fact that it dominates the lower orders. -/
def pairingSum (q : Nat) : Nat :=
  (List.range (q / 2 + 1)).foldl
    (fun acc j => acc + fact q / (2 ^ j * fact j * fact (q - 2 * j))) 0

/-- Source: "The coefficient sum for q=6 is 1+15+45+15=76". -/
theorem pairing_sum_six : pairingSum 6 = 76 := by decide

/-- Source: "and it bounds the smaller orders as well". -/
theorem pairing_sum_le_six : ∀ q, q ∈ [0, 1, 2, 3, 4, 5, 6] → pairingSum q ≤ 76 := by
  decide

/-- The individual pairing terms at `q = 6`, as displayed in the note. -/
theorem pairing_terms_six :
    fact 6 / (2 ^ 0 * fact 0 * fact 6) = 1 ∧
    fact 6 / (2 ^ 1 * fact 1 * fact 4) = 15 ∧
    fact 6 / (2 ^ 2 * fact 2 * fact 2) = 45 ∧
    fact 6 / (2 ^ 3 * fact 3 * fact 0) = 15 := by
  decide

/-- Source: "At zero, every such contraction has absolute value at most 15".
The sixth Gaussian moment / sixth derivative of `exp(-x^2/2)` at zero is the
double factorial `5!! = 1·3·5`. Only the arithmetic value is checked here. -/
theorem sixth_moment_double_factorial : 1 * 3 * 5 = 15 := by decide

/-- Source: "there are at most 27j^3 possibilities for d<=3, |n|^6<=27j^6" and
"<=729 sum_{j>=1} j^9 e^(-288j^2) <=1458 e^(-288)". The two factors of 27 give
729; the geometric tail with ratio at most 1/2 gives the factor 2. -/
theorem lattice_shell_constant : 27 * 27 = 729 ∧ 729 * 2 = 1458 := by decide

/-- Source: "successive terms have ratio at most 512e^(-864)". `512 = 2^9` is
the worst ratio `(j+1)^9 / j^9` at `j = 1`; `864 = 288·3` is the exponent gap
`288((j+1)^2 - j^2)` at `j = 1`. -/
theorem geometric_ratio_constants : 2 ^ 9 = 512 ∧ 288 * 3 = 864 := by decide

/-- `24^2 / 2 = 288`: the periodization exponent for lattice period 24. -/
theorem period_exponent : 24 ^ 2 / 2 = 288 := by decide

/-- The image constant `E · 10^125` of display (2):
`1458 · (76 · 24^6 + 15)`. -/
def imageConstant : Nat := 1458 * (76 * 24 ^ 6 + 15)

/-- Source: "E := 1458*(76*24^6+15)*10^(-125) = 21175738586478 *10^(-125)". -/
theorem image_constant_value : imageConstant = 21175738586478 := by decide

/-- Numerator of the degree-20 Taylor partial sum of `exp(288/125)` after
clearing the denominator `125^20 · 20!`:
`∑_{k=0}^{20} 288^k · 125^(20-k) · 20!/k!`. -/
def expTaylorNumerator : Nat :=
  (List.range 21).foldl
    (fun acc k => acc + 288 ^ k * 125 ^ (20 - k) * (fact 20 / fact k)) 0

/-- Source: "The code proves e^(288/125)>10 using the positive rational Taylor
sum through20". Because every Taylor term of `exp` at a positive argument is
positive, the partial sum is a lower bound; this theorem checks that the exact
rational partial sum exceeds 10. The step from the partial sum to `exp` itself
is analysis and is not formalised here. -/
theorem exp_taylor_partial_sum_gt_ten :
    10 * (125 ^ 20 * fact 20) < expTaylorNumerator := by decide

/-- Source: "Each covariance entry differs from the reference by at most2E ...
the spectral norm of the difference is at most20E" (dimension at most 10) and
"C_ref >= I/3 ... epsilon=10^(-108), since60E<epsilon". The relative bound is
`20E · 3 = 60E`, and `60E < 10^(-108)` is equivalent to `60 · 21175738586478 < 10^17`. -/
theorem covariance_relative_bound :
    2 * 10 * 3 = 60 ∧ 60 * imageConstant < 10 ^ 17 := by decide

/-- Source: "The only nontrivial odd block is [[1,-3],[-3,15]], whose eigenvalues
are 8 ± sqrt(58)" (the source writes the sign as plus-or-minus in ASCII).
Trace `16 = 2·8`, determinant `15 - 9 = 6 = 8^2 - 58`. -/
theorem odd_block_eigenvalues :
    (1 : Int) + 15 = 2 * 8 ∧ (1 : Int) * 15 - (-3) * (-3) = 6 ∧ (8 : Int) ^ 2 - 58 = 6 := by
  decide

/-- Source: "Its smaller eigenvalue exceeds 1/3: subtract I/3 and use first
principal minor2/3 and determinant7/9." Scaled by 3: `3M - I = [[2,-9],[-9,44]]`
has first minor `2` and determinant `88 - 81 = 7`. -/
theorem odd_block_shifted_minors :
    (3 : Int) * 1 - 1 = 2 ∧ (3 : Int) * 15 - 1 = 44 ∧
    (2 : Int) * 44 - (-9) * (-9) = 7 := by decide

/-- Direct check of the same claim: `8 - √58 > 1/3 ⇔ 23/3 > √58 ⇔ 23^2 > 58·9`. -/
theorem smaller_eigenvalue_exceeds_third : 58 * 9 < 23 ^ 2 := by decide

/-- Source (section 4): with `m = d-1`, `n = m(m+1)/2`, `a = m + 2/3 + n/2`,
`b = d + n/2`. These are `6a` and `6b` as natural numbers. -/
def sixA (d : Nat) : Nat := 6 * (d - 1) + 4 + 3 * ((d - 1) * d / 2)
def sixB (d : Nat) : Nat := 6 * d + 3 * ((d - 1) * d / 2)

/-- Source: "For d=2,3, a+2b<14 and2a+b<13". In sixths: `6a + 12b < 84` and
`12a + 6b < 78`. For `d = 3` the second inequality is `77 < 78`, so this is a
genuinely tight check. -/
theorem exponent_bounds :
    ∀ d, d ∈ [2, 3] → sixA d + 2 * sixB d < 84 ∧ 2 * sixA d + sixB d < 78 := by
  decide

/-- The exact values behind `exponent_bounds`: `d = 2` gives `a = 13/6`, `b = 5/2`;
`d = 3` gives `a = 25/6`, `b = 9/2`. -/
theorem exponent_values :
    sixA 2 = 13 ∧ sixB 2 = 15 ∧ sixA 3 = 25 ∧ sixB 3 = 27 := by decide

/-- Source: "ratio between1-13epsilon and1+28epsilon, hence safely between
1-32epsilon and1+32epsilon". The analytic `log`/`exp` estimates are not
formalised; the ordering of the constants is. -/
theorem ratio_constant_ordering : 13 ≤ 32 ∧ 28 ≤ 32 := by decide

/-- Source: "|c_d,24/c_d,ref -1| < 10^(-106)" together with `ε = 10^(-108)`
and the factor 32 from section 4: `32 · 10^(-108) < 10^(-106)`. -/
theorem density_comparison_below_reported : 32 * 10 ^ 106 < 10 ^ 108 := by decide

end UniversalLaw.Side24
