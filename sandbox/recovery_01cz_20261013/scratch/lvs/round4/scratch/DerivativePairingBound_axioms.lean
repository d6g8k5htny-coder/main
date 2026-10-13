import Specifications.Side24

/-!
# SIDE24: the Section 2 derivative pairing bound (statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

The inventory theorem below has as its TYPE the registered `Prop` constant
`UniversalLaw.Spec.Side24.DerivativePairingBound` of the statement packet
`specifications/lean_statements_20261010`. A kernel-checked proof of that `Prop` is evidence
about that `Prop` only: it does not make the informal Layer 0 claim
(SIDE24-COEFFICIENT-D23-20260924-v1, register row `side24-coefficient`) accepted, reviewed or
"verified", does not change its transcribed disposition (REVIEWED_SCOPED), does not enclose the
coefficient `c_{d,24}`, and does not touch the statement register (the statement module is not
edited here). Helper lemmas live in the `Aux` namespace (prefix `dp_`) and are not inventory
items.

Route. No closed form for the iterated Fréchet derivative of the Gaussian kernel is used. The
operator norms `b_q := ‖iteratedFDeriv ℝ q (Kinf d) x‖` satisfy the Hermite-type recursion
`b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q`, obtained from `norm_iteratedFDeriv_fderiv`, the explicit
first derivative `fderiv ℝ (Kinf d) y = Kinf d y • (-(innerSL ℝ y))`, and the Leibniz bound
`norm_iteratedFDeriv_smul_le` for the scalar factor `Kinf d` times the operator-valued factor
`y ↦ -(innerSL ℝ y)`, whose iterated derivatives of order `≥ 2` vanish. With `b_0 = Kinf d x`
and `b_1 = Kinf d x * ‖x‖` the recursion gives `b_q ≤ Kinf d x * pairingSum q ‖x‖` for
`q ≤ 6` order by order, and the contraction at a unit tuple is bounded by the operator norm
(`ContinuousMultilinearMap.le_opNorm`).

Allowed transitive axioms: `propext`, `Classical.choice`, `Quot.sound` only.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24
open scoped ContDiff

namespace Aux

/-! ### Smoothness and first derivative of the reference kernel `Kinf` -/

/-- `Kinf d = exp (-‖·‖^2 / 2)` is `C^∞` on `E d`. -/
theorem dp_contDiff_Kinf (d : ℕ) : ContDiff ℝ ∞ (Kinf d) := by
  unfold Kinf
  have h : ContDiff ℝ ∞ (fun x : E d => -(‖x‖ ^ 2) / 2) :=
    (contDiff_norm_sq ℝ (E := E d)).neg.div_const 2
  exact Real.contDiff_exp.comp h

/-- The first derivative of `Kinf d` at `x` is `Kinf d x • (-(innerSL ℝ x))`, i.e.
`v ↦ -⟪x, v⟫ Kinf d x`. -/
theorem dp_hasFDerivAt_Kinf (d : ℕ) (x : E d) :
    HasFDerivAt (Kinf d) (Kinf d x • (-(innerSL ℝ x))) x := by
  have h0 : HasFDerivAt (fun y : E d => ‖y‖ ^ 2) ((2 : ℕ) • innerSL ℝ x) x :=
    (hasStrictFDerivAt_norm_sq x).hasFDerivAt
  have h1 := (h0.const_smul (-(1 / 2 : ℝ))).exp
  convert h1 using 1
  · funext y
    simp only [Kinf, Pi.smul_apply, smul_eq_mul]
    ring_nf
  · ext v
    simp [Kinf]
    left
    ring

/-- The derivative function of `Kinf d`. -/
theorem dp_fderiv_Kinf (d : ℕ) :
    fderiv ℝ (Kinf d) = fun y => Kinf d y • (-(innerSL ℝ y)) :=
  funext fun y => (dp_hasFDerivAt_Kinf d y).fderiv

/-! ### The operator-valued factor `g y = -(innerSL ℝ y)` -/

/-- `y ↦ -(innerSL ℝ y)` is `C^∞` (it is a continuous linear map). -/
theorem dp_contDiff_g (d : ℕ) : ContDiff ℝ ∞ (fun y : E d => -(innerSL ℝ y)) :=
  (ContinuousLinearMap.contDiff (innerSL ℝ)).neg

/-- The factor as the coercion of the continuous linear map `-(innerSL ℝ)`. -/
theorem dp_g_eq (d : ℕ) :
    (fun y : E d => -(innerSL ℝ y)) = ⇑(-(innerSL ℝ (E := E d))) := by
  funext y
  simp

/-- Order zero: `‖g x‖ = ‖x‖` (`innerSL_apply_norm`). -/
theorem dp_norm_g_zero (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 0 (fun y : E d => -(innerSL ℝ y)) x‖ = ‖x‖ := by
  rw [norm_iteratedFDeriv_zero, norm_neg, innerSL_apply_norm]

/-- Order one: `‖D g x‖ = ‖innerSL ℝ‖ ≤ 1` (`norm_innerSL_le`). -/
theorem dp_norm_g_one (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 1 (fun y : E d => -(innerSL ℝ y)) x‖ ≤ 1 := by
  rw [norm_iteratedFDeriv_one, dp_g_eq, ContinuousLinearMap.fderiv, norm_neg]
  exact norm_innerSL_le ℝ

/-- Orders `k ≥ 2` of the linear factor vanish. -/
theorem dp_g_higher (d : ℕ) (x : E d) (k : ℕ) (hk : 2 ≤ k) :
    iteratedFDeriv ℝ k (fun y : E d => -(innerSL ℝ y)) x = 0 := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [dp_g_eq, iteratedFDeriv_succ_eq_comp_right]
  have hconst : fderiv ℝ ⇑(-(innerSL ℝ (E := E d))) = fun _ => -(innerSL ℝ (E := E d)) :=
    funext fun y => (-(innerSL ℝ (E := E d))).fderiv
  simp only [Function.comp_apply, hconst, iteratedFDeriv_const_of_ne (show j ≠ 0 by omega),
    Pi.zero_apply]
  exact LinearIsometryEquiv.map_zero _

/-! ### The Hermite-type recursion for the operator norms -/

/-- Hermite-type recursion for `b_q = ‖D^q Kinf (x)‖`:
`b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q`, from `norm_iteratedFDeriv_fderiv`, the explicit first
derivative, and the Leibniz bound `norm_iteratedFDeriv_smul_le` in which only the terms
`i = q` and `i = q + 1` of the binomial sum survive. -/
theorem dp_recursion (d : ℕ) (x : E d) (q : ℕ) :
    ‖iteratedFDeriv ℝ (q + 2) (Kinf d) x‖ ≤
      ‖x‖ * ‖iteratedFDeriv ℝ (q + 1) (Kinf d) x‖ +
        (q + 1) * ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  rw [← norm_iteratedFDeriv_fderiv, dp_fderiv_Kinf]
  have h := norm_iteratedFDeriv_smul_le (dp_contDiff_Kinf d) (dp_contDiff_g d) x
    (n := q + 1) (WithTop.coe_le_coe.mpr le_top)
  refine h.trans ?_
  rw [Finset.sum_range_succ, Finset.sum_range_succ]
  have hz : ∑ i ∈ Finset.range q, ((q + 1).choose i : ℝ) * ‖iteratedFDeriv ℝ i (Kinf d) x‖ *
      ‖iteratedFDeriv ℝ (q + 1 - i) (fun y : E d => -(innerSL ℝ y)) x‖ = 0 := by
    refine Finset.sum_eq_zero fun i hi => ?_
    rw [Finset.mem_range] at hi
    rw [dp_g_higher d x _ (by omega), norm_zero, mul_zero]
  rw [hz, zero_add, Nat.choose_self, Nat.choose_succ_self_right, Nat.add_sub_cancel_left,
    Nat.sub_self, dp_norm_g_zero]
  have h1 := dp_norm_g_one d x
  have hb : 0 ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := norm_nonneg _
  push_cast
  nlinarith [mul_le_mul_of_nonneg_left h1 (mul_nonneg (by positivity : (0 : ℝ) ≤ (q : ℝ) + 1) hb)]

/-! ### The seven explicit orders -/

/-- `Kinf d x > 0`. -/
theorem dp_Kinf_pos (d : ℕ) (x : E d) : 0 < Kinf d x := Real.exp_pos _

/-- `b_0 = Kinf d x`. -/
theorem dp_b0 (d : ℕ) (x : E d) : ‖iteratedFDeriv ℝ 0 (Kinf d) x‖ = Kinf d x := by
  rw [norm_iteratedFDeriv_zero, Real.norm_eq_abs, abs_of_pos (dp_Kinf_pos d x)]

/-- `b_1 = Kinf d x * ‖x‖`. -/
theorem dp_b1 (d : ℕ) (x : E d) : ‖iteratedFDeriv ℝ 1 (Kinf d) x‖ = Kinf d x * ‖x‖ := by
  rw [norm_iteratedFDeriv_one, dp_fderiv_Kinf, norm_smul, norm_neg, innerSL_apply_norm,
    Real.norm_eq_abs, abs_of_pos (dp_Kinf_pos d x)]

/-- `b_2 ≤ Kinf d x * (‖x‖^2 + 1)`. -/
theorem dp_b2 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 2 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 2 + 1) := by
  have h := dp_recursion d x 0
  rw [dp_b1, dp_b0] at h
  push_cast at h
  nlinarith [h]

/-- `b_3 ≤ Kinf d x * (‖x‖^3 + 3 ‖x‖)`. -/
theorem dp_b3 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 3 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 3 + 3 * ‖x‖) := by
  have h := dp_recursion d x 1
  have h2 := dp_b2 d x
  have h1 := dp_b1 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h2 hr]

/-- `b_4 ≤ Kinf d x * (‖x‖^4 + 6 ‖x‖^2 + 3)`. -/
theorem dp_b4 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 4 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 4 + 6 * ‖x‖ ^ 2 + 3) := by
  have h := dp_recursion d x 2
  have h3 := dp_b3 d x
  have h2 := dp_b2 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h3 hr]

/-- `b_5 ≤ Kinf d x * (‖x‖^5 + 10 ‖x‖^3 + 15 ‖x‖)`. -/
theorem dp_b5 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 5 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 5 + 10 * ‖x‖ ^ 3 + 15 * ‖x‖) := by
  have h := dp_recursion d x 3
  have h4 := dp_b4 d x
  have h3 := dp_b3 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h4 hr]

/-- `b_6 ≤ Kinf d x * (‖x‖^6 + 15 ‖x‖^4 + 45 ‖x‖^2 + 15)`. -/
theorem dp_b6 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 6 (Kinf d) x‖ ≤
      Kinf d x * (‖x‖ ^ 6 + 15 * ‖x‖ ^ 4 + 45 * ‖x‖ ^ 2 + 15) := by
  have h := dp_recursion d x 4
  have h5 := dp_b5 d x
  have h4 := dp_b4 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h5 hr]

/-- For `q ≤ 6`, `b_q ≤ Kinf d x * pairingSum q ‖x‖`: the explicit bounds `dp_b0` to `dp_b6`
match the unfolded pairing sums (`Finset.sum_range_succ`, `Nat.factorial`). -/
theorem dp_pairing_all (d : ℕ) (x : E d) (q : ℕ) (hq : q ≤ 6) :
    ‖iteratedFDeriv ℝ q (Kinf d) x‖ ≤ Kinf d x * pairingSum q ‖x‖ := by
  interval_cases q
  · rw [dp_b0]
    simp [pairingSum]
  · rw [dp_b1]
    simp [pairingSum]
  · refine (dp_b2 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (dp_b3 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (dp_b4 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (dp_b5 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (dp_b6 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring

/-! ### Contraction at a unit tuple -/

/-- At a unit tuple the contraction is bounded by the operator norm
(`ContinuousMultilinearMap.le_opNorm` with `∏ ‖u i‖ = 1`). -/
theorem dp_abs_le_opNorm (d q : ℕ) (x : E d) (u : Fin q → E d) (hu : IsUnitTuple d q u) :
    |iteratedFDeriv ℝ q (Kinf d) x u| ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  have h := (iteratedFDeriv ℝ q (Kinf d) x).le_opNorm u
  have hp : ∏ i, ‖u i‖ = 1 := Finset.prod_eq_one (fun i _ => hu i)
  rw [hp, mul_one] at h
  simpa [Real.norm_eq_abs] using h

end Aux

/-! ## Inventory theorem -/

/-- Proves: `UniversalLaw.Spec.Side24.DerivativePairingBound` (register row `side24-coefficient`,
Section 2 first display of SIDE24-COEFFICIENT-D23-20260924-v1).
Method: the contraction at a unit tuple is at most the operator norm `‖iteratedFDeriv ℝ q (Kinf d) x‖`
(`Aux.dp_abs_le_opNorm`, from `ContinuousMultilinearMap.le_opNorm`); the operator norms satisfy the
Hermite-type recursion `b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q` (`Aux.dp_recursion`, from
`norm_iteratedFDeriv_fderiv`, the explicit derivative `fderiv ℝ (Kinf d) y = Kinf d y • (-(innerSL ℝ y))`
and the Leibniz bound `norm_iteratedFDeriv_smul_le`, the linear factor's derivatives of order `≥ 2`
vanishing), and with `b_0 = Kinf d x`, `b_1 = Kinf d x ‖x‖` the orders `q ≤ 6` are matched to the
unfolded `pairingSum q ‖x‖` (`Aux.dp_pairing_all`, `interval_cases`, `nlinarith`).
Does not establish: the informal Section 2 display or its review and acceptance; the alignment of the
formal objects (`Kinf`, `IsUnitTuple`, `pairingSum`, Mathlib's `iteratedFDeriv ℝ q`) with the source's
(PENDING_INDEPENDENT_REVIEW); validity for `q > 6` (the Prop restricts to `q ≤ 6`, as the source does,
and nothing beyond it is stated or proved here); any enclosure of the coefficient `c_{d,24}`; the
disposition of `side24-coefficient`, which is REVIEWED_SCOPED as transcribed in the register, not
derived here. -/
theorem derivativePairingBound : UniversalLaw.Spec.Side24.DerivativePairingBound := by
  intro d q hq x u hu
  exact (Aux.dp_abs_le_opNorm d q x u hu).trans (Aux.dp_pairing_all d x q hq)

end UniversalLaw.Proofs.Side24Analysis

-- SCRATCH ONLY: axiom audit lines, not part of the deliverable
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_contDiff_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_hasFDerivAt_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_fderiv_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_contDiff_g
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_g_eq
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_norm_g_zero
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_norm_g_one
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_g_higher
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_recursion
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_Kinf_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_b0
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_b1
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_b2
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_b3
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_b4
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_b5
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_b6
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_pairing_all
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dp_abs_le_opNorm
#print axioms UniversalLaw.Proofs.Side24Analysis.derivativePairingBound
#check @UniversalLaw.Proofs.Side24Analysis.derivativePairingBound
