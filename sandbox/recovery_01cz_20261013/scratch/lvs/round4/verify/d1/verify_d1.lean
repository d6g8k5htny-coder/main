import Specifications.Side24

/-!
# SIDE24: the Section 2 derivative bound at `|x| ≥ 1` (statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

The inventory theorem below has as its TYPE the registered `Prop` constant
`UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne` of the statement packet
`specifications/lean_statements_20261010`. A kernel-checked proof of that `Prop` is evidence
about that `Prop` only: it does not make the informal Layer 0 claim
(SIDE24-COEFFICIENT-D23-20260924-v1, register row `side24-coefficient`) accepted, reviewed or
"verified", does not change its transcribed disposition (REVIEWED_SCOPED), does not enclose the
coefficient `c_{d,24}`, and does not touch the statement register (the statement module is not
edited here). Helper lemmas live in the `Aux` namespace (prefix `d1_`) and are not inventory
items.

Route. The Prop follows from the Section 2 pairing bound `DerivativePairingBound` together with
the elementary inequality `pairingSum q r ≤ 76 r^6` for `1 ≤ r` and `q ≤ 6`: each unfolded
pairing sum has nonnegative coefficients summing to the involution numbers `1, 1, 2, 4, 10, 26,
76 ≤ 76`, and every monomial `r^k` with `k ≤ 6` is at most `r^6` when `1 ≤ r`
(`pow_le_pow_right₀`, `one_le_pow₀`). That reduction is the hypothesis-form helper
`Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound`.

Because `DerivativePairingBound` is proved in a sibling file of the same round, the part of its
route needed here is reproduced under the `d1_` tag (`Aux.d1_derivativePairingBound`): the
operator norms `b_q := ‖iteratedFDeriv ℝ q (Kinf d) x‖` satisfy the Hermite-type recursion
`b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q`, obtained from `norm_iteratedFDeriv_fderiv`, the explicit
first derivative `fderiv ℝ (Kinf d) y = Kinf d y • (-(innerSL ℝ y))`, and the Leibniz bound
`norm_iteratedFDeriv_smul_le` for the scalar factor `Kinf d` times the operator-valued factor
`y ↦ -(innerSL ℝ y)`, whose iterated derivatives of order `≥ 2` vanish; with `b_0 = Kinf d x`
and `b_1 = Kinf d x * ‖x‖` the orders `q ≤ 6` are matched to the unfolded `pairingSum q ‖x‖`,
and the contraction at a unit tuple is at most the operator norm
(`ContinuousMultilinearMap.le_opNorm`). The integrator may replace
`Aux.d1_derivativePairingBound` by the sibling inventory theorem and drop the duplicate.

Allowed transitive axioms: `propext`, `Classical.choice`, `Quot.sound` only.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24
open scoped ContDiff

namespace Aux

/-! ### The elementary pairing-sum bound at `r ≥ 1` -/

/-- For `q ≤ 6` and `1 ≤ r`, `pairingSum q r ≤ 76 r^6`. Each pairing sum unfolds
(`Finset.sum_range_succ`, `Nat.factorial`) to a polynomial with nonnegative coefficients summing
to `1, 1, 2, 4, 10, 26, 76` for `q = 0, …, 6`; every monomial `r^k`, `k ≤ 6`, is at most `r^6`
(`pow_le_pow_right₀`, `one_le_pow₀`). -/
theorem d1_pairingSum_le_of_one_le (q : ℕ) (hq : q ≤ 6) (r : ℝ) (hr : 1 ≤ r) :
    pairingSum q r ≤ 76 * r ^ 6 := by
  have h0 : 1 ≤ r ^ 6 := one_le_pow₀ hr
  have h1 : r ≤ r ^ 6 := by
    simpa using pow_le_pow_right₀ hr (show 1 ≤ 6 by norm_num)
  have h2 : r ^ 2 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  have h3 : r ^ 3 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  have h4 : r ^ 4 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  have h5 : r ^ 5 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  interval_cases q <;> simp only [pairingSum, Finset.sum_range_succ, Finset.sum_range_zero,
    Nat.factorial] <;> norm_num <;> nlinarith [h0, h1, h2, h3, h4, h5, hr]

/-! ### Hypothesis form: the Prop from `DerivativePairingBound` -/

/-- `DerivativeBoundAtLeastOne` from `DerivativePairingBound`: at `1 ≤ ‖x‖` the pairing bound
`|D^q Kinf (x) u| ≤ exp(-‖x‖²/2) · pairingSum q ‖x‖` and `pairingSum q ‖x‖ ≤ 76 ‖x‖^6`
(`d1_pairingSum_le_of_one_le`) give the Prop by `mul_le_mul_of_nonneg_left` with
`0 ≤ exp(-‖x‖²/2)` (`Real.exp_pos`). -/
theorem d1_derivativeBoundAtLeastOne_of_derivativePairingBound
    (h : UniversalLaw.Spec.Side24.DerivativePairingBound) :
    UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne := by
  intro d q hq x hx u hu
  have hpair : |iteratedFDeriv ℝ q (Kinf d) x u| ≤
      Real.exp (-(‖x‖ ^ 2) / 2) * pairingSum q ‖x‖ := h d q hq x u hu
  have hps : pairingSum q ‖x‖ ≤ 76 * ‖x‖ ^ 6 := d1_pairingSum_le_of_one_le q hq ‖x‖ hx
  have hexp : 0 ≤ Real.exp (-(‖x‖ ^ 2) / 2) := (Real.exp_pos _).le
  calc |iteratedFDeriv ℝ q (Kinf d) x u|
      ≤ Real.exp (-(‖x‖ ^ 2) / 2) * pairingSum q ‖x‖ := hpair
    _ ≤ Real.exp (-(‖x‖ ^ 2) / 2) * (76 * ‖x‖ ^ 6) := mul_le_mul_of_nonneg_left hps hexp
    _ = 76 * ‖x‖ ^ 6 * Real.exp (-(‖x‖ ^ 2) / 2) := by ring

/-! ### Reproduction of the pairing-bound route (sibling file `DerivativePairingBound`)

Smoothness and first derivative of the reference kernel `Kinf`. -/

/-- `Kinf d = exp (-‖·‖^2 / 2)` is `C^∞` on `E d`. -/
theorem d1_contDiff_Kinf (d : ℕ) : ContDiff ℝ ∞ (Kinf d) := by
  unfold Kinf
  have h : ContDiff ℝ ∞ (fun x : E d => -(‖x‖ ^ 2) / 2) :=
    (contDiff_norm_sq ℝ (E := E d)).neg.div_const 2
  exact Real.contDiff_exp.comp h

/-- The first derivative of `Kinf d` at `x` is `Kinf d x • (-(innerSL ℝ x))`, i.e.
`v ↦ -⟪x, v⟫ Kinf d x`. -/
theorem d1_hasFDerivAt_Kinf (d : ℕ) (x : E d) :
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
theorem d1_fderiv_Kinf (d : ℕ) :
    fderiv ℝ (Kinf d) = fun y => Kinf d y • (-(innerSL ℝ y)) :=
  funext fun y => (d1_hasFDerivAt_Kinf d y).fderiv

/-! The operator-valued factor `g y = -(innerSL ℝ y)`. -/

/-- `y ↦ -(innerSL ℝ y)` is `C^∞` (it is a continuous linear map). -/
theorem d1_contDiff_g (d : ℕ) : ContDiff ℝ ∞ (fun y : E d => -(innerSL ℝ y)) :=
  (ContinuousLinearMap.contDiff (innerSL ℝ)).neg

/-- The factor as the coercion of the continuous linear map `-(innerSL ℝ)`. -/
theorem d1_g_eq (d : ℕ) :
    (fun y : E d => -(innerSL ℝ y)) = ⇑(-(innerSL ℝ (E := E d))) := by
  funext y
  simp

/-- Order zero: `‖g x‖ = ‖x‖` (`innerSL_apply_norm`). -/
theorem d1_norm_g_zero (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 0 (fun y : E d => -(innerSL ℝ y)) x‖ = ‖x‖ := by
  rw [norm_iteratedFDeriv_zero, norm_neg, innerSL_apply_norm]

/-- Order one: `‖D g x‖ = ‖innerSL ℝ‖ ≤ 1` (`norm_innerSL_le`). -/
theorem d1_norm_g_one (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 1 (fun y : E d => -(innerSL ℝ y)) x‖ ≤ 1 := by
  rw [norm_iteratedFDeriv_one, d1_g_eq, ContinuousLinearMap.fderiv, norm_neg]
  exact norm_innerSL_le ℝ

/-- Orders `k ≥ 2` of the linear factor vanish. -/
theorem d1_g_higher (d : ℕ) (x : E d) (k : ℕ) (hk : 2 ≤ k) :
    iteratedFDeriv ℝ k (fun y : E d => -(innerSL ℝ y)) x = 0 := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [d1_g_eq, iteratedFDeriv_succ_eq_comp_right]
  have hconst : fderiv ℝ ⇑(-(innerSL ℝ (E := E d))) = fun _ => -(innerSL ℝ (E := E d)) :=
    funext fun y => (-(innerSL ℝ (E := E d))).fderiv
  simp only [Function.comp_apply, hconst, iteratedFDeriv_const_of_ne (show j ≠ 0 by omega),
    Pi.zero_apply]
  exact LinearIsometryEquiv.map_zero _

/-! The Hermite-type recursion for the operator norms. -/

/-- Hermite-type recursion for `b_q = ‖D^q Kinf (x)‖`:
`b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q`, from `norm_iteratedFDeriv_fderiv`, the explicit first
derivative, and the Leibniz bound `norm_iteratedFDeriv_smul_le` in which only the terms
`i = q` and `i = q + 1` of the binomial sum survive. -/
theorem d1_recursion (d : ℕ) (x : E d) (q : ℕ) :
    ‖iteratedFDeriv ℝ (q + 2) (Kinf d) x‖ ≤
      ‖x‖ * ‖iteratedFDeriv ℝ (q + 1) (Kinf d) x‖ +
        (q + 1) * ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  rw [← norm_iteratedFDeriv_fderiv, d1_fderiv_Kinf]
  have h := norm_iteratedFDeriv_smul_le (d1_contDiff_Kinf d) (d1_contDiff_g d) x
    (n := q + 1) (WithTop.coe_le_coe.mpr le_top)
  refine h.trans ?_
  rw [Finset.sum_range_succ, Finset.sum_range_succ]
  have hz : ∑ i ∈ Finset.range q, ((q + 1).choose i : ℝ) * ‖iteratedFDeriv ℝ i (Kinf d) x‖ *
      ‖iteratedFDeriv ℝ (q + 1 - i) (fun y : E d => -(innerSL ℝ y)) x‖ = 0 := by
    refine Finset.sum_eq_zero fun i hi => ?_
    rw [Finset.mem_range] at hi
    rw [d1_g_higher d x _ (by omega), norm_zero, mul_zero]
  rw [hz, zero_add, Nat.choose_self, Nat.choose_succ_self_right, Nat.add_sub_cancel_left,
    Nat.sub_self, d1_norm_g_zero]
  have h1 := d1_norm_g_one d x
  have hb : 0 ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := norm_nonneg _
  push_cast
  nlinarith [mul_le_mul_of_nonneg_left h1 (mul_nonneg (by positivity : (0 : ℝ) ≤ (q : ℝ) + 1) hb)]

/-! The seven explicit orders. -/

/-- `Kinf d x > 0`. -/
theorem d1_Kinf_pos (d : ℕ) (x : E d) : 0 < Kinf d x := Real.exp_pos _

/-- `b_0 = Kinf d x`. -/
theorem d1_b0 (d : ℕ) (x : E d) : ‖iteratedFDeriv ℝ 0 (Kinf d) x‖ = Kinf d x := by
  rw [norm_iteratedFDeriv_zero, Real.norm_eq_abs, abs_of_pos (d1_Kinf_pos d x)]

/-- `b_1 = Kinf d x * ‖x‖`. -/
theorem d1_b1 (d : ℕ) (x : E d) : ‖iteratedFDeriv ℝ 1 (Kinf d) x‖ = Kinf d x * ‖x‖ := by
  rw [norm_iteratedFDeriv_one, d1_fderiv_Kinf, norm_smul, norm_neg, innerSL_apply_norm,
    Real.norm_eq_abs, abs_of_pos (d1_Kinf_pos d x)]

/-- `b_2 ≤ Kinf d x * (‖x‖^2 + 1)`. -/
theorem d1_b2 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 2 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 2 + 1) := by
  have h := d1_recursion d x 0
  rw [d1_b1, d1_b0] at h
  push_cast at h
  nlinarith [h]

/-- `b_3 ≤ Kinf d x * (‖x‖^3 + 3 ‖x‖)`. -/
theorem d1_b3 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 3 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 3 + 3 * ‖x‖) := by
  have h := d1_recursion d x 1
  have h2 := d1_b2 d x
  have h1 := d1_b1 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h2 hr]

/-- `b_4 ≤ Kinf d x * (‖x‖^4 + 6 ‖x‖^2 + 3)`. -/
theorem d1_b4 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 4 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 4 + 6 * ‖x‖ ^ 2 + 3) := by
  have h := d1_recursion d x 2
  have h3 := d1_b3 d x
  have h2 := d1_b2 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h3 hr]

/-- `b_5 ≤ Kinf d x * (‖x‖^5 + 10 ‖x‖^3 + 15 ‖x‖)`. -/
theorem d1_b5 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 5 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 5 + 10 * ‖x‖ ^ 3 + 15 * ‖x‖) := by
  have h := d1_recursion d x 3
  have h4 := d1_b4 d x
  have h3 := d1_b3 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h4 hr]

/-- `b_6 ≤ Kinf d x * (‖x‖^6 + 15 ‖x‖^4 + 45 ‖x‖^2 + 15)`. -/
theorem d1_b6 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 6 (Kinf d) x‖ ≤
      Kinf d x * (‖x‖ ^ 6 + 15 * ‖x‖ ^ 4 + 45 * ‖x‖ ^ 2 + 15) := by
  have h := d1_recursion d x 4
  have h5 := d1_b5 d x
  have h4 := d1_b4 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h5 hr]

/-- For `q ≤ 6`, `b_q ≤ Kinf d x * pairingSum q ‖x‖`: the explicit bounds `d1_b0` to `d1_b6`
match the unfolded pairing sums (`Finset.sum_range_succ`, `Nat.factorial`). -/
theorem d1_pairing_all (d : ℕ) (x : E d) (q : ℕ) (hq : q ≤ 6) :
    ‖iteratedFDeriv ℝ q (Kinf d) x‖ ≤ Kinf d x * pairingSum q ‖x‖ := by
  interval_cases q
  · rw [d1_b0]
    simp [pairingSum]
  · rw [d1_b1]
    simp [pairingSum]
  · refine (d1_b2 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (d1_b3 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (d1_b4 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (d1_b5 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (d1_b6 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring

/-! Contraction at a unit tuple, and the reproduced pairing bound. -/

/-- At a unit tuple the contraction is bounded by the operator norm
(`ContinuousMultilinearMap.le_opNorm` with `∏ ‖u i‖ = 1`). -/
theorem d1_abs_le_opNorm (d q : ℕ) (x : E d) (u : Fin q → E d) (hu : IsUnitTuple d q u) :
    |iteratedFDeriv ℝ q (Kinf d) x u| ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  have h := (iteratedFDeriv ℝ q (Kinf d) x).le_opNorm u
  have hp : ∏ i, ‖u i‖ = 1 := Finset.prod_eq_one (fun i _ => hu i)
  rw [hp, mul_one] at h
  simpa [Real.norm_eq_abs] using h

/-- The Section 2 pairing bound, reproduced under the `d1_` tag so that this file is
self-contained; the sibling file proves the same registered Prop as its inventory theorem, and
the integrator may substitute it here. -/
theorem d1_derivativePairingBound : UniversalLaw.Spec.Side24.DerivativePairingBound := by
  intro d q hq x u hu
  exact (d1_abs_le_opNorm d q x u hu).trans (d1_pairing_all d x q hq)

end Aux

/-! ## Inventory theorem -/

/-- Proves: `UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne` (register row `side24-coefficient`,
Section 2 of SIDE24-COEFFICIENT-D23-20260924-v1, "At |x|>=1 this is at most 76 |x|^6 e^(-|x|^2/2)").
Method: the Section 2 pairing bound `|D^q Kinf (x) u| ≤ exp(-‖x‖²/2) · pairingSum q ‖x‖` (here
`Aux.d1_derivativePairingBound`, the Hermite-type recursion `b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q` for the
operator norms from `norm_iteratedFDeriv_fderiv`, `norm_iteratedFDeriv_smul_le` and the explicit
`fderiv ℝ (Kinf d)`, with `ContinuousMultilinearMap.le_opNorm` at a unit tuple) is combined, through
the hypothesis-form helper `Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound`, with the
elementary inequality `pairingSum q r ≤ 76 r^6` for `1 ≤ r`, `q ≤ 6` (`Aux.d1_pairingSum_le_of_one_le`:
`interval_cases`, `Finset.sum_range_succ`, `Nat.factorial`, `norm_num`, and the monomial bounds
`r^k ≤ r^6` from `pow_le_pow_right₀`, `one_le_pow₀`), by `mul_le_mul_of_nonneg_left` with
`0 ≤ exp(-‖x‖²/2)`.
Does not establish: the informal Section 2 display or its review and acceptance; the alignment of the
formal objects (`Kinf`, `IsUnitTuple`, `pairingSum`, Mathlib's `iteratedFDeriv ℝ q`) with the source's
(PENDING_INDEPENDENT_REVIEW); any identification with the main-branch targets
`UniversalLaw.Side24.pairing_sum_six` and `pairing_sum_le_six`, which kernel-check the coefficient
arithmetic `1+15+45+15=76` and `76` bounding the smaller orders there, not here; validity for `q > 6` or
for `‖x‖ < 1` (the Prop restricts to `q ≤ 6` and `1 ≤ ‖x‖`, as the source does); any enclosure of the
coefficient `c_{d,24}`; the disposition of `side24-coefficient`, which is REVIEWED_SCOPED as transcribed
in the register, not derived here. -/
theorem derivativeBoundAtLeastOne : UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne :=
  Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound Aux.d1_derivativePairingBound

end UniversalLaw.Proofs.Side24Analysis

-- VERIFIER SCRATCH TAIL (not part of the deliverable)
#check @UniversalLaw.Proofs.Side24Analysis.derivativeBoundAtLeastOne
#check @UniversalLaw.Proofs.Side24Analysis.Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound
#check @UniversalLaw.Proofs.Side24Analysis.Aux.d1_derivativePairingBound
#print UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne
#print UniversalLaw.Spec.Side24.DerivativePairingBound
#print UniversalLaw.Spec.Side24.IsUnitTuple
#print UniversalLaw.Spec.Side24.Kinf
#print UniversalLaw.Spec.Side24.pairingSum
#print UniversalLaw.Spec.Side24.E
#print UniversalLaw.Proofs.Side24Analysis.derivativeBoundAtLeastOne
#print axioms UniversalLaw.Proofs.Side24Analysis.derivativeBoundAtLeastOne
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_pairingSum_le_of_one_le
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_contDiff_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_hasFDerivAt_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_fderiv_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_contDiff_g
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_g_eq
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_norm_g_zero
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_norm_g_one
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_g_higher
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_recursion
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_Kinf_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_b0
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_b1
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_b2
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_b3
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_b4
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_b5
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_b6
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_pairing_all
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_abs_le_opNorm
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d1_derivativePairingBound

open Lean Elab Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  -- (A) syntactic type exactness against the registered Prop constants
  let dbalo := Expr.const `UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne []
  let dpb := Expr.const `UniversalLaw.Spec.Side24.DerivativePairingBound []
  match env.find? `UniversalLaw.Proofs.Side24Analysis.derivativeBoundAtLeastOne with
  | none => logInfo "INVENTORY MISSING"
  | some ci =>
    logInfo s!"INVENTORY type == const DerivativeBoundAtLeastOne (syntactic): {ci.type == dbalo}; levelParams={ci.levelParams}; isThm={ci matches .thmInfo _}"
  match env.find? `UniversalLaw.Proofs.Side24Analysis.Aux.d1_derivativeBoundAtLeastOne_of_derivativePairingBound with
  | none => logInfo "HELPER MISSING"
  | some ci =>
    let ok := match ci.type with
      | .forallE _ dom body _ => dom == dpb && body == dbalo
      | _ => false
    logInfo s!"HELPER type == (DerivativePairingBound → DerivativeBoundAtLeastOne) (syntactic, nondependent): {ok}"
  match env.find? `UniversalLaw.Proofs.Side24Analysis.Aux.d1_derivativePairingBound with
  | none => logInfo "DPB-REPRO MISSING"
  | some ci =>
    logInfo s!"DPB-REPRO type == const DerivativePairingBound (syntactic): {ci.type == dpb}"
  -- (B) the registered Prop constants are defs (as in the statement packet), unchanged by this file
  for n in [`UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne, `UniversalLaw.Spec.Side24.DerivativePairingBound] do
    match env.find? n with
    | some (.defnInfo d) => logInfo s!"{n}: def, in local map₂? {env.constants.map₂.contains n}; value head: {d.value.getAppFn.ctorName}"
    | some _ => logInfo s!"{n}: NOT a defnInfo"
    | none => logInfo s!"{n}: MISSING"
  -- (C) environment audit of every constant this module adds
  let mut rows : Array String := #[]
  let mut nonThm := 0
  let mut matchers := 0
  let mut badAx := 0
  for (n, ci) in env.constants.map₂.toList do
    let kind := match ci with
      | .thmInfo _ => "thm" | .defnInfo _ => "DEF" | .axiomInfo _ => "AXIOM"
      | .opaqueInfo _ => "OPAQUE" | .quotInfo _ => "QUOT" | .inductInfo _ => "ind"
      | .ctorInfo _ => "ctor" | .recInfo _ => "rec"
    if !(ci matches .thmInfo _) then nonThm := nonThm + 1
    let isP ← liftTermElabM (Meta.isProp ci.type)
    let isM := Meta.isMatcherCore env n
    if isM then matchers := matchers + 1
    let axs ← collectAxioms n
    let axsOk := axs.all (fun a => a == ``propext || a == ``Classical.choice || a == ``Quot.sound)
    if !axsOk then badAx := badAx + 1
    rows := rows.push s!"{kind} prop={isP} unsafe={ci.isUnsafe} matcher={isM} axioms={axs.qsort (·.toString < ·.toString)} axOK={axsOk} {n}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo s!"AUDIT TOTAL={rows.size} nonThm={nonThm} matchers={matchers} badAxioms={badAx}"
