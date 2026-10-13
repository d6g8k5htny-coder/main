import Specifications.Side24

/-!
# SIDE24: kernel-checked proofs of elementary registered `Prop`s (statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem below has as its TYPE a registered `Prop` constant of
`UniversalLaw.Spec.Side24` (statement packet `specifications/lean_statements_20261010`, commit
pinned there). A kernel-checked proof of such a `Prop` is evidence about that `Prop` only: it
does not make the informal Layer 0 claim (SIDE24-COEFFICIENT-D23-20260924-v1) accepted,
reviewed or "verified", does not change its LANDING_CLAIMS disposition (REVIEWED_SCOPED,
transcribed), does not enclose the coefficient `c_{d,24}`, and does not touch the statement
register (the statement module is not edited here). Helper lemmas live in the `Aux` namespace
and are not inventory items.

Allowed transitive axioms: `propext`, `Classical.choice`, `Quot.sound` only (checked by
`#print axioms`, see the run receipt). No `sorry`, `native_decide`, `decide` on `ℝ`, or custom
axioms.

Covered (five `Prop`s): `ExponentialBounds`, `SuccessiveTermRatio`, `ConeMomentD2Algebra`,
`ReferenceCoefficientIdentity`, `ElementaryConeIntegral`.

Not covered here: `TraceMoments`, `ConeMomentD1Identity`, `ConeMomentD2Identity`,
`RayleighSquareLaw` (Gaussian moment integrals over an interface measure, not algebra);
`ReferenceOddAndPinBlocks`, `DerivativeBoundAtZero`, `DerivativePairingBound`,
`DerivativeBoundAtLeastOne` (explicit `iteratedFDeriv` of the Gaussian kernel on
`EuclideanSpace` through order six); `LatticeImageSumBound` (a `tsum` over `Fin d → ℤ` with a
shell decomposition); and every interface-level or analytic statement of the module.
-/

namespace UniversalLaw.Proofs.Side24Elementary

open UniversalLaw.Spec.Side24

namespace Aux

/-! ### Exponential bounds -/

/-- The degree-nine Taylor polynomial of `exp` at `288/125` exceeds `10` (exact rational
arithmetic). -/
theorem ten_lt_taylor_sum :
    (10 : ℝ) < ∑ i ∈ Finset.range 10, (288 / 125 : ℝ) ^ i / (Nat.factorial i : ℝ) := by
  simp [Finset.sum_range_succ, Nat.factorial]
  norm_num

/-- `10 < exp (288/125)` by the Taylor lower bound `Real.sum_le_exp_of_nonneg`. -/
theorem ten_lt_exp : (10 : ℝ) < Real.exp (288 / 125) := by
  have h := Real.sum_le_exp_of_nonneg (by norm_num : (0 : ℝ) ≤ 288 / 125) 10
  exact lt_of_lt_of_le ten_lt_taylor_sum h

/-- `10^125 < exp 288`, raising the previous bound to the `125`-th power. -/
theorem pow_ten_lt_exp : (10 : ℝ) ^ 125 < Real.exp 288 := by
  have h : Real.exp 288 = Real.exp (288 / 125) ^ 125 := by
    rw [← Real.exp_nat_mul]; norm_num
  rw [h]
  exact pow_lt_pow_left₀ ten_lt_exp (by norm_num) (by norm_num)

/-- `exp (-288) < 1 / 10^125`. -/
theorem exp_neg_lt : Real.exp (-288) < 1 / 10 ^ 125 := by
  rw [Real.exp_neg, one_div]
  exact inv_strictAnti₀ (by positivity) pow_ten_lt_exp

/-- `1024 < exp 864` by the quadratic lower bound. -/
theorem big_lt_exp : (1024 : ℝ) < Real.exp 864 := by
  have := Real.quadratic_le_exp_of_nonneg (by norm_num : (0 : ℝ) ≤ 864)
  linarith

/-- The geometric-ratio constant: `512 exp (-864) < 1/2`. -/
theorem ratio_const_lt_half : 512 * Real.exp (-864) < (1 / 2 : ℝ) := by
  rw [Real.exp_neg, mul_inv_lt_iff₀ (Real.exp_pos _)]
  have := big_lt_exp
  linarith

/-- The term ratio for `j ≥ 1`: `(j+1)^9 ≤ 512 j^9` since `j + 1 ≤ 2 j`. -/
theorem pow_succ_le (j : ℕ) (hj : 1 ≤ j) : ((j : ℝ) + 1) ^ 9 ≤ 512 * (j : ℝ) ^ 9 := by
  have hj' : (1 : ℝ) ≤ j := by exact_mod_cast hj
  have h : ((j : ℝ) + 1) ≤ 2 * j := by linarith
  calc ((j : ℝ) + 1) ^ 9 ≤ (2 * j) ^ 9 := pow_le_pow_left₀ (by linarith) h 9
    _ = 512 * (j : ℝ) ^ 9 := by ring

/-- The exponential factor for `j ≥ 1`: `exp (-288 (j+1)^2) ≤ exp (-864) exp (-288 j^2)`. -/
theorem exp_succ_le (j : ℕ) (hj : 1 ≤ j) :
    Real.exp (-288 * ((j : ℝ) + 1) ^ 2) ≤ Real.exp (-864) * Real.exp (-288 * (j : ℝ) ^ 2) := by
  have hj' : (1 : ℝ) ≤ j := by exact_mod_cast hj
  rw [← Real.exp_add, Real.exp_le_exp]
  nlinarith

/-! ### Square-root algebra -/

/-- `sqrt (3/8) = sqrt 6 / 4`. -/
theorem sqrt_three_eighths : Real.sqrt (3 / 8) = Real.sqrt 6 / 4 := by
  rw [show (3 / 8 : ℝ) = 6 / 4 ^ 2 by norm_num, Real.sqrt_div' _ (by norm_num),
    Real.sqrt_sq (by norm_num)]

/-- `(sqrt 6)^(4/3) = 24^(1/3) (3/2)^(1/3)`, i.e. `6^(2/3) = 36^(1/3)` and `24 * 3/2 = 36`. -/
theorem sqrt_six_rpow : (Real.sqrt 6) ^ ((4 : ℝ) / 3) =
    (24 : ℝ) ^ ((1 : ℝ) / 3) * (3 / 2 : ℝ) ^ ((1 : ℝ) / 3) := by
  rw [← Real.mul_rpow (by norm_num) (by norm_num), Real.sqrt_eq_rpow,
    ← Real.rpow_mul (by norm_num), show (1 / (2 : ℝ)) * (4 / 3) = 2 * (1 / 3) by norm_num,
    Real.rpow_mul (by norm_num), Real.rpow_two]
  norm_num

/-! ### The elementary cone integral -/

/-- The antiderivative `F z = -((a-z)^2 - 4(a-z) + 8) exp (-z/2)` of
`(a-z)^2 exp (-z/2) / 2`. -/
theorem hasDerivAt_antideriv (a z : ℝ) :
    HasDerivAt (fun z : ℝ => -(((a - z) ^ 2 - 4 * (a - z) + 8) * Real.exp (-z / 2)))
      ((a - z) ^ 2 * Real.exp (-z / 2) / 2) z := by
  have h1 : HasDerivAt (fun z : ℝ => a - z) (-1) z :=
    (hasDerivAt_id' z).const_sub a
  have h2 : HasDerivAt (fun z : ℝ => (a - z) ^ 2) (((2 : ℕ) : ℝ) * (a - z) ^ (2 - 1) * (-1)) z :=
    h1.fun_pow 2
  have h3 : HasDerivAt (fun z : ℝ => 4 * (a - z)) (4 * (-1)) z := h1.const_mul 4
  have h4 : HasDerivAt (fun z : ℝ => (a - z) ^ 2 - 4 * (a - z) + 8)
      (((2 : ℕ) : ℝ) * (a - z) ^ (2 - 1) * (-1) - 4 * (-1) + 0) z :=
    (h2.sub h3).add (hasDerivAt_const z (8 : ℝ))
  have h5 : HasDerivAt (fun z : ℝ => -z / 2) (-1 / 2) z :=
    ((hasDerivAt_id' z).neg).div_const 2
  have h6 : HasDerivAt (fun z : ℝ => Real.exp (-z / 2)) (Real.exp (-z / 2) * (-1 / 2)) z := h5.exp
  have h7 := (h4.mul h6).neg
  convert h7 using 1
  push_cast
  ring

end Aux

/-! ## Section 2 exponential and ratio facts -/

/-- Proves: `UniversalLaw.Spec.Side24.ExponentialBounds` (register row `side24-coefficient`).
Method: degree-nine Taylor lower bound `Real.sum_le_exp_of_nonneg` at `288/125` evaluated in
exact rational arithmetic, then `exp 288 = exp (288/125)^125 > 10^125` and inversion.
Does not establish: the lattice-image bound or display (2) of the source, any enclosure of the
coefficient, nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem exponentialBounds : UniversalLaw.Spec.Side24.ExponentialBounds :=
  ⟨Aux.ten_lt_exp, Aux.exp_neg_lt⟩

/-- Proves: `UniversalLaw.Spec.Side24.SuccessiveTermRatio` (register row `side24-coefficient`).
Method: `(j+1)^9 ≤ (2j)^9 = 512 j^9` and `-288 (j+1)^2 ≤ -864 - 288 j^2` for `j ≥ 1`;
`512 exp (-864) < 1/2` from the quadratic lower bound `1 + 864 + 864^2/2 ≤ exp 864`.
Does not establish: the lattice-image sum bound `LatticeImageSumBound` (the geometric
summation itself is not performed here), any enclosure of the coefficient, nonauthor review
or acceptance. -/
theorem successiveTermRatio : UniversalLaw.Spec.Side24.SuccessiveTermRatio := by
  refine ⟨fun j hj => ?_, Aux.ratio_const_lt_half⟩
  calc ((j : ℝ) + 1) ^ 9 * Real.exp (-288 * ((j : ℝ) + 1) ^ 2)
      ≤ (512 * (j : ℝ) ^ 9) * (Real.exp (-864) * Real.exp (-288 * (j : ℝ) ^ 2)) :=
        mul_le_mul (Aux.pow_succ_le j hj) (Aux.exp_succ_le j hj) (Real.exp_pos _).le
          (by positivity)
    _ = 512 * Real.exp (-864) * ((j : ℝ) ^ 9 * Real.exp (-288 * (j : ℝ) ^ 2)) := by ring

/-! ## Section 1 algebra -/

/-- Proves: `UniversalLaw.Spec.Side24.ConeMomentD2Algebra` (register row `side24-coefficient`).
Method: `sqrt (3/8) = sqrt 6 / 4` (`Real.sqrt_div'`, `Real.sqrt_sq`), then `ring`.
Does not establish: the Gaussian identity `ConeMomentD2Identity` (that `D_2` is the cone
expectation), any enclosure of the coefficient, nonauthor review or acceptance. -/
theorem coneMomentD2Algebra : UniversalLaw.Spec.Side24.ConeMomentD2Algebra := by
  unfold UniversalLaw.Spec.Side24.ConeMomentD2Algebra coneMomentD2
  rw [Aux.sqrt_three_eighths]
  ring

/-- Proves: `UniversalLaw.Spec.Side24.ReferenceCoefficientIdentity` (register row
`side24-coefficient`).
Method: case split `d = 2`, `d = 3`; `(sqrt 6)^(4/3) = 24^(1/3) (3/2)^(1/3)` through
`Real.sqrt_eq_rpow`, `Real.rpow_mul`, `Real.mul_rpow`; `(2 pi)^(-d)` through `Real.rpow_neg`
and `Real.rpow_natCast`; then `field_simp` and `ring` with `pi`, `sqrt pi`, `sqrt 3`,
`24^(1/3)` nonzero.
Does not establish: that the left side is the parent formula (15.2) evaluated at the reference
law (rotational invariance of `K_infty` is not specified), the value of `Gamma (7/6)`, equality
of the periodic and reference coefficients (the source says it is NOT claimed), any enclosure
of the coefficient, nonauthor review or acceptance. -/
theorem referenceCoefficientIdentity : UniversalLaw.Spec.Side24.ReferenceCoefficientIdentity := by
  intro d hd
  have hpi := Real.pi_pos
  have hsqpi : 0 < Real.sqrt Real.pi := Real.sqrt_pos.mpr hpi
  have hsq3 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
  have h24 : 0 < (24 : ℝ) ^ ((1 : ℝ) / 3) := Real.rpow_pos_of_pos (by norm_num) _
  have h32 : 0 < (3 / 2 : ℝ) ^ ((1 : ℝ) / 3) := Real.rpow_pos_of_pos (by norm_num) _
  have hG := Real.Gamma_pos_of_pos (by norm_num : (0 : ℝ) < 7 / 6)
  rw [Aux.sqrt_six_rpow, Real.rpow_neg (by positivity), Real.rpow_natCast]
  rcases hd with rfl | rfl
  · simp only [cRef, sphereArea, coneMomentRef]
    norm_num
    field_simp
  · simp only [cRef, sphereArea, coneMomentRef]
    norm_num
    field_simp
    ring

/-- Proves: `UniversalLaw.Spec.Side24.ElementaryConeIntegral` (register row
`side24-coefficient`).
Method: fundamental theorem of calculus `intervalIntegral.integral_eq_sub_of_hasDerivAt` with
the explicit antiderivative `-((a-z)^2 - 4(a-z) + 8) exp (-z/2)` (derivative rules
`HasDerivAt.fun_pow`, `.mul`, `.exp`), integrability from continuity (`fun_prop`).
Does not establish: the Rayleigh law of `R^2` (`RayleighSquareLaw`), the cone-moment identity
`ConeMomentD2Identity`, any enclosure of the coefficient, nonauthor review or acceptance. -/
theorem elementaryConeIntegral : UniversalLaw.Spec.Side24.ElementaryConeIntegral := by
  intro a _
  have hderiv : ∀ z ∈ Set.uIcc (0 : ℝ) a,
      HasDerivAt (fun z : ℝ => -(((a - z) ^ 2 - 4 * (a - z) + 8) * Real.exp (-z / 2)))
        ((a - z) ^ 2 * Real.exp (-z / 2) / 2) z :=
    fun z _ => Aux.hasDerivAt_antideriv a z
  have hint : IntervalIntegrable (fun z : ℝ => (a - z) ^ 2 * Real.exp (-z / 2) / 2)
      MeasureTheory.volume 0 a :=
    (by fun_prop : Continuous fun z : ℝ => (a - z) ^ 2 * Real.exp (-z / 2) / 2).intervalIntegrable 0 a
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint]
  simp
  ring

end UniversalLaw.Proofs.Side24Elementary
