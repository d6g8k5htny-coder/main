import Specifications.Side24

/-!
# SIDE24 analysis proofs: the Section 4 integrand-ratio bounds (`IntegrandRatioBounds`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Analysis.Aux` (prefix `ir_`). The route is
pure real arithmetic in `epsilon = 10^(-108)`, `exponentA d` and `exponentB d`; no derivative
or density input enters. For `d = 2` the exponents are `a = 13/6`, `b = 5/2`; for `d = 3` they
are `a = 25/6`, `b = 9/2`; in both cases `1 ≤ a, b ≤ 5`. For any such `a, b`: Bernoulli's
inequality (`one_add_mul_self_le_rpow_one_add` at `s = -epsilon`) gives
`1 - a epsilon ≤ (1 - epsilon)^a` and `1 - b epsilon ≤ (1 - epsilon)^b`; monotonicity of the
real power in the exponent (`Real.rpow_le_rpow_of_exponent_le`, base `1 + epsilon ≥ 1`) gives
`(1 + epsilon)^a, (1 + epsilon)^b ≤ (1 + epsilon)^5`; the two displayed quotients then reduce,
through `le_div_iff₀` and `div_le_iff₀`, to the exact rational inequalities
`(1 - 13 epsilon)(1 + epsilon)^5 ≤ 1 - 5 epsilon` and `(1 + epsilon)^5 ≤ (1 + 28 epsilon)(1 - 5 epsilon)`,
which `norm_num` decides after unfolding `epsilon`.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24

namespace Aux

/-! ### (a) Elementary facts about `epsilon = 10^(-108)` -/

/-- `0 < epsilon`. -/
theorem ir_epsilon_pos : (0 : ℝ) < epsilon := by
  unfold epsilon
  positivity

/-- `epsilon < 1/100`: a crude upper bound sufficient for every sign condition below. -/
theorem ir_epsilon_lt : epsilon < (1 / 100 : ℝ) := by
  unfold epsilon
  norm_num

/-- `0 < 1 - epsilon`. -/
theorem ir_one_sub_epsilon_pos : (0 : ℝ) < 1 - epsilon := by
  have := ir_epsilon_lt
  linarith

/-- `1 ≤ 1 + epsilon`. -/
theorem ir_one_le_one_add_epsilon : (1 : ℝ) ≤ 1 + epsilon := by
  have := ir_epsilon_pos
  linarith

/-! ### (b) The two one-sided power estimates -/

/-- For a real exponent `b ≤ 5`, `(1 + epsilon)^b ≤ (1 + epsilon)^5`: monotonicity of the real
power in the exponent for a base `≥ 1` (`Real.rpow_le_rpow_of_exponent_le`), with the natural
power on the right via `Real.rpow_natCast`. -/
theorem ir_rpow_le_pow_five (b : ℝ) (hb : b ≤ 5) :
    (1 + epsilon) ^ b ≤ (1 + epsilon) ^ 5 := by
  have h := Real.rpow_le_rpow_of_exponent_le ir_one_le_one_add_epsilon hb
  rwa [show (5 : ℝ) = ((5 : ℕ) : ℝ) by norm_num, Real.rpow_natCast] at h

/-- Bernoulli's inequality at `s = -epsilon`: for a real exponent `a ≥ 1`,
`1 - a epsilon ≤ (1 - epsilon)^a` (`one_add_mul_self_le_rpow_one_add`). -/
theorem ir_bernoulli (a : ℝ) (ha : 1 ≤ a) : 1 - a * epsilon ≤ (1 - epsilon) ^ a := by
  have hε := ir_epsilon_lt
  have h := one_add_mul_self_le_rpow_one_add (show (-1 : ℝ) ≤ -epsilon by linarith) ha
  rw [show (1 : ℝ) + -epsilon = 1 - epsilon by ring] at h
  linarith

/-! ### (c) The two exact rational inequalities (degree five in `epsilon`) -/

/-- `(1 - 13 epsilon)(1 + epsilon)^5 ≤ 1 - 5 epsilon`, exact rational arithmetic. -/
theorem ir_lower_rational : (1 - 13 * epsilon) * (1 + epsilon) ^ 5 ≤ 1 - 5 * epsilon := by
  unfold epsilon
  norm_num

/-- `(1 + epsilon)^5 ≤ (1 + 28 epsilon)(1 - 5 epsilon)`, exact rational arithmetic. -/
theorem ir_upper_rational : (1 + epsilon) ^ 5 ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := by
  unfold epsilon
  norm_num

/-! ### (d) The ratio bounds for arbitrary exponents in `[1, 5]` -/

/-- Lower bound: for `1 ≤ a ≤ 5` and `1 ≤ b ≤ 5`,
`1 - 13 epsilon ≤ (1 - epsilon)^a / (1 + epsilon)^b`. -/
theorem ir_lower_bound (a b : ℝ) (ha1 : 1 ≤ a) (ha5 : a ≤ 5) (hb5 : b ≤ 5) :
    1 - 13 * epsilon ≤ (1 - epsilon) ^ a / (1 + epsilon) ^ b := by
  have hε := ir_epsilon_pos
  have hε1 := ir_epsilon_lt
  have hpb : 0 < (1 + epsilon) ^ b := Real.rpow_pos_of_pos (by linarith) _
  have hae : a * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right ha5 hε.le
  rw [le_div_iff₀ hpb]
  calc (1 - 13 * epsilon) * (1 + epsilon) ^ b
      ≤ (1 - 13 * epsilon) * (1 + epsilon) ^ 5 :=
        mul_le_mul_of_nonneg_left (ir_rpow_le_pow_five b hb5) (by linarith)
    _ ≤ 1 - 5 * epsilon := ir_lower_rational
    _ ≤ 1 - a * epsilon := by linarith
    _ ≤ (1 - epsilon) ^ a := ir_bernoulli a ha1

/-- Upper bound: for `a ≤ 5` and `1 ≤ b ≤ 5`,
`(1 + epsilon)^a / (1 - epsilon)^b ≤ 1 + 28 epsilon`. -/
theorem ir_upper_bound (a b : ℝ) (ha5 : a ≤ 5) (hb1 : 1 ≤ b) (hb5 : b ≤ 5) :
    (1 + epsilon) ^ a / (1 - epsilon) ^ b ≤ 1 + 28 * epsilon := by
  have hε := ir_epsilon_pos
  have hqb : 0 < (1 - epsilon) ^ b := Real.rpow_pos_of_pos ir_one_sub_epsilon_pos _
  have hbe : b * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right hb5 hε.le
  rw [div_le_iff₀ hqb]
  calc (1 + epsilon) ^ a ≤ (1 + epsilon) ^ 5 := ir_rpow_le_pow_five a ha5
    _ ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := ir_upper_rational
    _ ≤ (1 + 28 * epsilon) * (1 - b * epsilon) :=
        mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    _ ≤ (1 + 28 * epsilon) * (1 - epsilon) ^ b :=
        mul_le_mul_of_nonneg_left (ir_bernoulli b hb1) (by linarith)

/-- Both ratio bounds together, for `1 ≤ a ≤ 5` and `1 ≤ b ≤ 5`. -/
theorem ir_ratio_bounds (a b : ℝ) (ha1 : 1 ≤ a) (ha5 : a ≤ 5) (hb1 : 1 ≤ b) (hb5 : b ≤ 5) :
    1 - 13 * epsilon ≤ (1 - epsilon) ^ a / (1 + epsilon) ^ b ∧
      (1 + epsilon) ^ a / (1 - epsilon) ^ b ≤ 1 + 28 * epsilon :=
  ⟨ir_lower_bound a b ha1 ha5 hb5, ir_upper_bound a b ha5 hb1 hb5⟩

/-! ### (e) The exponents in dimensions two and three -/

/-- `exponentA 2 = 13/6` (`m = 1`, `n = 1`: `1 + 2/3 + 1/2`). -/
theorem ir_exponentA_two : exponentA 2 = 13 / 6 := by
  simp [exponentA, symCoordDim]
  norm_num

/-- `exponentB 2 = 5/2` (`d = 2`, `n = 1`: `2 + 1/2`). -/
theorem ir_exponentB_two : exponentB 2 = 5 / 2 := by
  simp [exponentB, symCoordDim]
  norm_num

/-- `exponentA 3 = 25/6` (`m = 2`, `n = 3`: `2 + 2/3 + 3/2`). -/
theorem ir_exponentA_three : exponentA 3 = 25 / 6 := by
  simp [exponentA, symCoordDim]
  norm_num

/-- `exponentB 3 = 9/2` (`d = 3`, `n = 3`: `3 + 3/2`). -/
theorem ir_exponentB_three : exponentB 3 = 9 / 2 := by
  simp [exponentB, symCoordDim]
  norm_num

end Aux

/-! ## The Section 4 integrand-ratio bounds in dimensions two and three -/

/-- Proves: `UniversalLaw.Spec.Side24.IntegrandRatioBounds` (claim `side24-coefficient`).
Method: case split `d = 2`, `d = 3`, where the exponents evaluate to `a = 13/6, b = 5/2` and
`a = 25/6, b = 9/2` (`Aux.ir_exponentA_two`, `Aux.ir_exponentB_two`, `Aux.ir_exponentA_three`,
`Aux.ir_exponentB_three`); for arbitrary `1 ≤ a, b ≤ 5` both bounds follow
(`Aux.ir_ratio_bounds`) from Bernoulli's inequality `1 - a epsilon ≤ (1 - epsilon)^a`
(`one_add_mul_self_le_rpow_one_add`, `Aux.ir_bernoulli`), exponent monotonicity
`(1 + epsilon)^b ≤ (1 + epsilon)^5` (`Real.rpow_le_rpow_of_exponent_le`,
`Aux.ir_rpow_le_pow_five`), `le_div_iff₀` / `div_le_iff₀`, and the two exact rational
inequalities `(1 - 13 epsilon)(1 + epsilon)^5 ≤ 1 - 5 epsilon` and
`(1 + epsilon)^5 ≤ (1 + 28 epsilon)(1 - 5 epsilon)` decided by `norm_num`
(`Aux.ir_lower_rational`, `Aux.ir_upper_rational`).
Does not establish: the informal Section 4 display of the source as the source's object (the
alignment of `epsilon`, `exponentA`, `exponentB` and these power quotients with the source's
`epsilon`, `a`, `b` and integrand ratio is PENDING_INDEPENDENT_REVIEW); that the integrand ratio
of the source equals these power quotients (not specified in the statement packet); the
arithmetic `13 ≤ 32`, `28 ≤ 32` and `32*10^(-108) < 10^(-106)` (kernel-checked on main in
`UniversalLaw.Side24.ratio_constant_ordering` and `density_comparison_below_reported`, not
here); `DensityComparison` or any covariance ordering; any enclosure of the coefficient; the
statement module's own `Does not claim:` text (the statement packet is not edited here); the
disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register,
not derived here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem integrandRatioBounds : UniversalLaw.Spec.Side24.IntegrandRatioBounds := by
  intro d hd
  rcases hd with rfl | rfl
  · rw [Aux.ir_exponentA_two, Aux.ir_exponentB_two]
    exact Aux.ir_ratio_bounds _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · rw [Aux.ir_exponentA_three, Aux.ir_exponentB_three]
    exact Aux.ir_ratio_bounds _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end UniversalLaw.Proofs.Side24Analysis

-- SCRATCH COPY ONLY (not the deliverable): axioms of every theorem, then the constants audit.
#print axioms UniversalLaw.Proofs.Side24Analysis.integrandRatioBounds
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_epsilon_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_epsilon_lt
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_one_sub_epsilon_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_one_le_one_add_epsilon
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_rpow_le_pow_five
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_bernoulli
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_lower_rational
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_upper_rational
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_lower_bound
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_upper_bound
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_ratio_bounds
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentA_two
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentB_two
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentA_three
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.ir_exponentB_three
#check UniversalLaw.Proofs.Side24Analysis.integrandRatioBounds

-- SCRATCH COPY ONLY: list every constant this module adds, with kind and Prop-ness.
open Lean Elab Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let mut rows : Array String := #[]
  for (n, ci) in env.constants.map₂.toList do
    let kind := match ci with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "AXIOM"
      | .opaqueInfo _ => "OPAQUE" | .quotInfo _ => "QUOT" | .inductInfo _ => "ind"
      | .ctorInfo _ => "ctor" | .recInfo _ => "rec"
    let isP ← liftTermElabM (Meta.isProp ci.type)
    rows := rows.push s!"{kind} prop={isP} unsafe={ci.isUnsafe} {n}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo s!"TOTAL {rows.size}"
