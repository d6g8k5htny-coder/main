import Lean
import Specifications.Side24

/-!
# SIDE24: the Section 2 derivative bound at the origin (statement module `Specifications.Side24`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

The inventory theorem below has as its TYPE the registered `Prop` constant
`UniversalLaw.Spec.Side24.DerivativeBoundAtZero` of the statement packet
`specifications/lean_statements_20261010`. A kernel-checked proof of that `Prop` is evidence
about that `Prop` only: it does not make the informal Layer 0 claim
(SIDE24-COEFFICIENT-D23-20260924-v1, register row `side24-coefficient`) accepted, reviewed or
"verified", does not change its transcribed disposition (REVIEWED_SCOPED), does not enclose the
coefficient `c_{d,24}`, and does not touch the statement register (the statement module is not
edited here). Helper lemmas live in the `Aux` namespace (prefix `d0_`) and are not inventory
items.

Route. The bound at the origin is the pairing bound of `DerivativePairingBound` specialised to
`x = 0`, where `‖x‖ = 0`, `exp (-(0 ^ 2) / 2) = 1` and `pairingSum q 0 ∈ {1, 0, 1, 0, 3, 0, 15}`
for `q = 0, …, 6`, all at most `15`. Two forms are delivered: the hypothesis-form helper
`Aux.d0_derivativeBoundAtZero_of_derivativePairingBound`, which takes the registered `Prop`
`DerivativePairingBound` as a hypothesis, and the inventory theorem itself, which discharges that
hypothesis with a reproduction of the pairing bound under the `d0_` tag
(`Aux.d0_derivativePairingBound`). The reproduction uses no closed form for the iterated
Fréchet derivative of the Gaussian kernel: the operator norms `b_q := ‖iteratedFDeriv ℝ q (Kinf d) x‖`
satisfy the Hermite-type recursion `b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q`, obtained from
`norm_iteratedFDeriv_fderiv`, the explicit first derivative
`fderiv ℝ (Kinf d) y = Kinf d y • (-(innerSL ℝ y))`, and the Leibniz bound
`norm_iteratedFDeriv_smul_le` for the scalar factor `Kinf d` times the operator-valued factor
`y ↦ -(innerSL ℝ y)`, whose iterated derivatives of order `≥ 2` vanish. With `b_0 = Kinf d x`
and `b_1 = Kinf d x * ‖x‖` the recursion gives `b_q ≤ Kinf d x * pairingSum q ‖x‖` for `q ≤ 6`
order by order, and the contraction at a unit tuple is bounded by the operator norm
(`ContinuousMultilinearMap.le_opNorm`). The integrator may replace the `d0_` reproduction by the
`dp_` development of the sibling `DerivativePairingBound` file; the hypothesis-form helper is
written so that only the final line of the inventory theorem then changes.

Allowed transitive axioms: `propext`, `Classical.choice`, `Quot.sound` only.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24
open scoped ContDiff

namespace Aux

/-! ### The pairing sums at the origin -/

/-- At `r = 0` the pairing sums of order `q ≤ 6` are `1, 0, 1, 0, 3, 0, 15` (only the `j = q/2`
term of an even order survives, `0 ^ 0 = 1`), hence at most `15`. -/
theorem d0_pairingSum_zero_le (q : ℕ) (hq : q ≤ 6) : pairingSum q 0 ≤ 15 := by
  interval_cases q <;> simp [pairingSum, Finset.sum_range_succ, Nat.factorial] <;> norm_num

/-! ### The hypothesis form: the bound at the origin from the pairing bound -/

/-- `DerivativeBoundAtZero` from `DerivativePairingBound` taken as a hypothesis: specialise the
pairing bound to `x = 0`, where `‖0‖ = 0` and `exp (-(0 ^ 2) / 2) = 1`, and bound
`pairingSum q 0 ≤ 15` by `d0_pairingSum_zero_le`. -/
theorem d0_derivativeBoundAtZero_of_derivativePairingBound
    (h : UniversalLaw.Spec.Side24.DerivativePairingBound) :
    UniversalLaw.Spec.Side24.DerivativeBoundAtZero := by
  intro d q hq u hu
  have h0 := h d q hq 0 u hu
  have hx : Real.exp (-(‖(0 : E d)‖ ^ 2) / 2) * pairingSum q ‖(0 : E d)‖ = pairingSum q 0 := by
    simp
  rw [hx] at h0
  exact h0.trans (d0_pairingSum_zero_le q hq)

/-! ### Reproduction of the pairing bound: smoothness and first derivative of `Kinf` -/

/-- `Kinf d = exp (-‖·‖^2 / 2)` is `C^∞` on `E d`. -/
theorem d0_contDiff_Kinf (d : ℕ) : ContDiff ℝ ∞ (Kinf d) := by
  unfold Kinf
  have h : ContDiff ℝ ∞ (fun x : E d => -(‖x‖ ^ 2) / 2) :=
    (contDiff_norm_sq ℝ (E := E d)).neg.div_const 2
  exact Real.contDiff_exp.comp h

/-- The first derivative of `Kinf d` at `x` is `Kinf d x • (-(innerSL ℝ x))`, i.e.
`v ↦ -⟪x, v⟫ Kinf d x`. -/
theorem d0_hasFDerivAt_Kinf (d : ℕ) (x : E d) :
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
theorem d0_fderiv_Kinf (d : ℕ) :
    fderiv ℝ (Kinf d) = fun y => Kinf d y • (-(innerSL ℝ y)) :=
  funext fun y => (d0_hasFDerivAt_Kinf d y).fderiv

/-! ### The operator-valued factor `g y = -(innerSL ℝ y)` -/

/-- `y ↦ -(innerSL ℝ y)` is `C^∞` (it is a continuous linear map). -/
theorem d0_contDiff_g (d : ℕ) : ContDiff ℝ ∞ (fun y : E d => -(innerSL ℝ y)) :=
  (ContinuousLinearMap.contDiff (innerSL ℝ)).neg

/-- The factor as the coercion of the continuous linear map `-(innerSL ℝ)`. -/
theorem d0_g_eq (d : ℕ) :
    (fun y : E d => -(innerSL ℝ y)) = ⇑(-(innerSL ℝ (E := E d))) := by
  funext y
  simp

/-- Order zero: `‖g x‖ = ‖x‖` (`innerSL_apply_norm`). -/
theorem d0_norm_g_zero (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 0 (fun y : E d => -(innerSL ℝ y)) x‖ = ‖x‖ := by
  rw [norm_iteratedFDeriv_zero, norm_neg, innerSL_apply_norm]

/-- Order one: `‖D g x‖ = ‖innerSL ℝ‖ ≤ 1` (`norm_innerSL_le`). -/
theorem d0_norm_g_one (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 1 (fun y : E d => -(innerSL ℝ y)) x‖ ≤ 1 := by
  rw [norm_iteratedFDeriv_one, d0_g_eq, ContinuousLinearMap.fderiv, norm_neg]
  exact norm_innerSL_le ℝ

/-- Orders `k ≥ 2` of the linear factor vanish. -/
theorem d0_g_higher (d : ℕ) (x : E d) (k : ℕ) (hk : 2 ≤ k) :
    iteratedFDeriv ℝ k (fun y : E d => -(innerSL ℝ y)) x = 0 := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [d0_g_eq, iteratedFDeriv_succ_eq_comp_right]
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
theorem d0_recursion (d : ℕ) (x : E d) (q : ℕ) :
    ‖iteratedFDeriv ℝ (q + 2) (Kinf d) x‖ ≤
      ‖x‖ * ‖iteratedFDeriv ℝ (q + 1) (Kinf d) x‖ +
        (q + 1) * ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  rw [← norm_iteratedFDeriv_fderiv, d0_fderiv_Kinf]
  have h := norm_iteratedFDeriv_smul_le (d0_contDiff_Kinf d) (d0_contDiff_g d) x
    (n := q + 1) (WithTop.coe_le_coe.mpr le_top)
  refine h.trans ?_
  rw [Finset.sum_range_succ, Finset.sum_range_succ]
  have hz : ∑ i ∈ Finset.range q, ((q + 1).choose i : ℝ) * ‖iteratedFDeriv ℝ i (Kinf d) x‖ *
      ‖iteratedFDeriv ℝ (q + 1 - i) (fun y : E d => -(innerSL ℝ y)) x‖ = 0 := by
    refine Finset.sum_eq_zero fun i hi => ?_
    rw [Finset.mem_range] at hi
    rw [d0_g_higher d x _ (by omega), norm_zero, mul_zero]
  rw [hz, zero_add, Nat.choose_self, Nat.choose_succ_self_right, Nat.add_sub_cancel_left,
    Nat.sub_self, d0_norm_g_zero]
  have h1 := d0_norm_g_one d x
  have hb : 0 ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := norm_nonneg _
  push_cast
  nlinarith [mul_le_mul_of_nonneg_left h1 (mul_nonneg (by positivity : (0 : ℝ) ≤ (q : ℝ) + 1) hb)]

/-! ### The seven explicit orders -/

/-- `Kinf d x > 0`. -/
theorem d0_Kinf_pos (d : ℕ) (x : E d) : 0 < Kinf d x := Real.exp_pos _

/-- `b_0 = Kinf d x`. -/
theorem d0_b0 (d : ℕ) (x : E d) : ‖iteratedFDeriv ℝ 0 (Kinf d) x‖ = Kinf d x := by
  rw [norm_iteratedFDeriv_zero, Real.norm_eq_abs, abs_of_pos (d0_Kinf_pos d x)]

/-- `b_1 = Kinf d x * ‖x‖`. -/
theorem d0_b1 (d : ℕ) (x : E d) : ‖iteratedFDeriv ℝ 1 (Kinf d) x‖ = Kinf d x * ‖x‖ := by
  rw [norm_iteratedFDeriv_one, d0_fderiv_Kinf, norm_smul, norm_neg, innerSL_apply_norm,
    Real.norm_eq_abs, abs_of_pos (d0_Kinf_pos d x)]

/-- `b_2 ≤ Kinf d x * (‖x‖^2 + 1)`. -/
theorem d0_b2 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 2 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 2 + 1) := by
  have h := d0_recursion d x 0
  rw [d0_b1, d0_b0] at h
  push_cast at h
  nlinarith [h]

/-- `b_3 ≤ Kinf d x * (‖x‖^3 + 3 ‖x‖)`. -/
theorem d0_b3 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 3 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 3 + 3 * ‖x‖) := by
  have h := d0_recursion d x 1
  have h2 := d0_b2 d x
  have h1 := d0_b1 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h2 hr]

/-- `b_4 ≤ Kinf d x * (‖x‖^4 + 6 ‖x‖^2 + 3)`. -/
theorem d0_b4 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 4 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 4 + 6 * ‖x‖ ^ 2 + 3) := by
  have h := d0_recursion d x 2
  have h3 := d0_b3 d x
  have h2 := d0_b2 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h3 hr]

/-- `b_5 ≤ Kinf d x * (‖x‖^5 + 10 ‖x‖^3 + 15 ‖x‖)`. -/
theorem d0_b5 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 5 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 5 + 10 * ‖x‖ ^ 3 + 15 * ‖x‖) := by
  have h := d0_recursion d x 3
  have h4 := d0_b4 d x
  have h3 := d0_b3 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h4 hr]

/-- `b_6 ≤ Kinf d x * (‖x‖^6 + 15 ‖x‖^4 + 45 ‖x‖^2 + 15)`. -/
theorem d0_b6 (d : ℕ) (x : E d) :
    ‖iteratedFDeriv ℝ 6 (Kinf d) x‖ ≤
      Kinf d x * (‖x‖ ^ 6 + 15 * ‖x‖ ^ 4 + 45 * ‖x‖ ^ 2 + 15) := by
  have h := d0_recursion d x 4
  have h5 := d0_b5 d x
  have h4 := d0_b4 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h5 hr]

/-- For `q ≤ 6`, `b_q ≤ Kinf d x * pairingSum q ‖x‖`: the explicit bounds `d0_b0` to `d0_b6`
match the unfolded pairing sums (`Finset.sum_range_succ`, `Nat.factorial`). -/
theorem d0_pairing_all (d : ℕ) (x : E d) (q : ℕ) (hq : q ≤ 6) :
    ‖iteratedFDeriv ℝ q (Kinf d) x‖ ≤ Kinf d x * pairingSum q ‖x‖ := by
  interval_cases q
  · rw [d0_b0]
    simp [pairingSum]
  · rw [d0_b1]
    simp [pairingSum]
  · refine (d0_b2 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (d0_b3 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (d0_b4 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (d0_b5 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (d0_b6 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring

/-! ### Contraction at a unit tuple, and the reproduced pairing bound -/

/-- At a unit tuple the contraction is bounded by the operator norm
(`ContinuousMultilinearMap.le_opNorm` with `∏ ‖u i‖ = 1`). -/
theorem d0_abs_le_opNorm (d q : ℕ) (x : E d) (u : Fin q → E d) (hu : IsUnitTuple d q u) :
    |iteratedFDeriv ℝ q (Kinf d) x u| ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  have h := (iteratedFDeriv ℝ q (Kinf d) x).le_opNorm u
  have hp : ∏ i, ‖u i‖ = 1 := Finset.prod_eq_one (fun i _ => hu i)
  rw [hp, mul_one] at h
  simpa [Real.norm_eq_abs] using h

/-- The pairing bound reproduced under the `d0_` tag (a helper, not an inventory item: the
inventory theorem for `DerivativePairingBound` is delivered in its own file). -/
theorem d0_derivativePairingBound : UniversalLaw.Spec.Side24.DerivativePairingBound := by
  intro d q hq x u hu
  exact (d0_abs_le_opNorm d q x u hu).trans (d0_pairing_all d x q hq)

end Aux

/-! ## Inventory theorem -/

/-- Proves: `UniversalLaw.Spec.Side24.DerivativeBoundAtZero` (register row `side24-coefficient`,
Section 2 of SIDE24-COEFFICIENT-D23-20260924-v1, "contraction has absolute value at most 15").
Method: the hypothesis-form helper `Aux.d0_derivativeBoundAtZero_of_derivativePairingBound`
specialises the pairing bound `DerivativePairingBound` to `x = 0`, where `‖0‖ = 0`,
`exp (-(0 ^ 2) / 2) = 1` and `pairingSum q 0 ≤ 15` for `q ≤ 6` (`Aux.d0_pairingSum_zero_le`:
the values are `1, 0, 1, 0, 3, 0, 15`, by `interval_cases`, `Finset.sum_range_succ`, `norm_num`);
the pairing bound itself is reproduced as `Aux.d0_derivativePairingBound` from the Hermite-type
recursion `b_{q+2} ≤ ‖x‖ b_{q+1} + (q+1) b_q` for the operator norms (`Aux.d0_recursion`, from
`norm_iteratedFDeriv_fderiv`, `fderiv ℝ (Kinf d) y = Kinf d y • (-(innerSL ℝ y))` and the Leibniz
bound `norm_iteratedFDeriv_smul_le`), the orders `q ≤ 6` (`Aux.d0_pairing_all`) and
`ContinuousMultilinearMap.le_opNorm` at a unit tuple (`Aux.d0_abs_le_opNorm`).
Does not establish: the informal Section 2 display or its review and acceptance; the alignment of
the formal objects (`Kinf`, `IsUnitTuple`, `pairingSum`, Mathlib's `iteratedFDeriv ℝ q` at `0`) with
the source's (PENDING_INDEPENDENT_REVIEW); the double-factorial identity `5!! = 15` as such, which
is kernel-checked on main (`UniversalLaw.Side24.sixth_moment_double_factorial`) and is neither
re-proved nor cited here (only `pairingSum 6 0 = 15` is computed); validity for `q > 6` (the Prop
restricts to `q ≤ 6`, as the source does); any value of a contraction (only an upper bound on its
absolute value); any enclosure of the coefficient `c_{d,24}`; the disposition of
`side24-coefficient`, which is REVIEWED_SCOPED as transcribed in the register, not derived here. -/
theorem derivativeBoundAtZero : UniversalLaw.Spec.Side24.DerivativeBoundAtZero :=
  Aux.d0_derivativeBoundAtZero_of_derivativePairingBound Aux.d0_derivativePairingBound

end UniversalLaw.Proofs.Side24Analysis


-- ===== VERIFIER SCRATCH TAIL (not part of the deliverable) =====
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_pairingSum_zero_le
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_derivativeBoundAtZero_of_derivativePairingBound
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_contDiff_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_hasFDerivAt_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_fderiv_Kinf
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_contDiff_g
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_g_eq
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_norm_g_zero
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_norm_g_one
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_g_higher
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_recursion
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_Kinf_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_b0
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_b1
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_b2
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_b3
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_b4
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_b5
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_b6
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_pairing_all
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_abs_le_opNorm
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.d0_derivativePairingBound
#print axioms UniversalLaw.Proofs.Side24Analysis.derivativeBoundAtZero
#check @UniversalLaw.Proofs.Side24Analysis.derivativeBoundAtZero
#check @UniversalLaw.Proofs.Side24Analysis.Aux.d0_derivativeBoundAtZero_of_derivativePairingBound
#print UniversalLaw.Spec.Side24.DerivativeBoundAtZero
#print UniversalLaw.Spec.Side24.DerivativePairingBound
#print UniversalLaw.Spec.Side24.pairingSum
#print UniversalLaw.Spec.Side24.Kinf
#print UniversalLaw.Spec.Side24.IsUnitTuple
open Lean Meta Elab Command

def auditModules : Array Name := #[]
def auditPrefix : Name := `UniversalLaw.Proofs.Side24Analysis
def registeredProps : Array Name := #[`UniversalLaw.Spec.Field.HessianHasInertia, `UniversalLaw.Spec.Field.IsCandidatePair, `UniversalLaw.Spec.Field.IsCriticalPoint, `UniversalLaw.Spec.Field.IsFiniteBar, `UniversalLaw.Spec.Field.IsIndexSaddle, `UniversalLaw.Spec.Field.IsLPeriodic, `UniversalLaw.Spec.Field.IsLifetimeDensityVersion, `UniversalLaw.Spec.Field.IsMorseDistinct, `UniversalLaw.Spec.Field.IsNondegMax, `UniversalLaw.Spec.Field.IsSmooth, `UniversalLaw.Spec.Lifetime.HessianHasInertia, `UniversalLaw.Spec.Lifetime.IsCandidatePair, `UniversalLaw.Spec.Lifetime.IsCriticalPoint, `UniversalLaw.Spec.Lifetime.IsFiniteBar, `UniversalLaw.Spec.Lifetime.IsIndexSaddle, `UniversalLaw.Spec.Lifetime.IsLPeriodic, `UniversalLaw.Spec.Lifetime.IsLifetimeDensityVersion, `UniversalLaw.Spec.Lifetime.IsMorseDistinct, `UniversalLaw.Spec.Lifetime.IsNondegMax, `UniversalLaw.Spec.Lifetime.IsSmooth, `UniversalLaw.Spec.Lifetime.IsUnrestrictedLeadingCoefficient, `UniversalLaw.Spec.Lifetime.IsUnrestrictedPair, `UniversalLaw.Spec.Lifetime.IsWindowElderPair, `UniversalLaw.Spec.Lifetime.IsWindowPair, `UniversalLaw.Spec.Lifetime.ParentCompactWindowDensityB, `UniversalLaw.Spec.Lifetime.ParentUniformSelectionA, `UniversalLaw.Spec.Lifetime.ParentUnrestrictedLeadingDensityC, `UniversalLaw.Spec.Lifetime.PlanarDensityGapE11, `UniversalLaw.Spec.Lifetime.PlanarElderLowerE2, `UniversalLaw.Spec.Lifetime.PlanarTwoSidedRateE4, `UniversalLaw.Spec.Lifetime.RemainderBoundedR1, `UniversalLaw.Spec.P15.Benchmark816Cover, `UniversalLaw.Spec.P15.Benchmark816IsRealized, `UniversalLaw.Spec.P15.Benchmark816MinimalForbiddenCounts, `UniversalLaw.Spec.P15.Benchmark816Palette, `UniversalLaw.Spec.P15.Benchmark816Price, `UniversalLaw.Spec.P15.Benchmark816RationalPrice, `UniversalLaw.Spec.P15.Benchmark816TransformedRange, `UniversalLaw.Spec.P15.Benchmark816UnionBound, `UniversalLaw.Spec.P15.Benchmark816WholeGround, `UniversalLaw.Spec.P15.CapTestAtTwoThirds, `UniversalLaw.Spec.P15.CapacityHazardIsBinomial, `UniversalLaw.Spec.P15.CapacityWorstCase, `UniversalLaw.Spec.P15.CoordinatewiseConcavity, `UniversalLaw.Spec.P15.CoverCostHazardBound, `UniversalLaw.Spec.P15.DemandOneBoundary, `UniversalLaw.Spec.P15.DemandOneRefutation, `UniversalLaw.Spec.P15.ElementaryPaletteFormula, `UniversalLaw.Spec.P15.FullBlockCoverIff, `UniversalLaw.Spec.P15.FullGroundBudgetIff, `UniversalLaw.Spec.P15.FullGroundPriceBound, `UniversalLaw.Spec.P15.FullTransformedPriceBudget, `UniversalLaw.Spec.P15.GlobalAssemblyHazard, `UniversalLaw.Spec.P15.HStarExceedsOne, `UniversalLaw.Spec.P15.HazardPhiBound, `UniversalLaw.Spec.P15.HazardProductBound, `UniversalLaw.Spec.P15.HazardStarPositive, `UniversalLaw.Spec.P15.IndependentBlockHazard, `UniversalLaw.Spec.P15.IsGeneratorCover, `UniversalLaw.Spec.P15.IsGood, `UniversalLaw.Spec.P15.IsKDecomposable, `UniversalLaw.Spec.P15.IsMinimalForbidden, `UniversalLaw.Spec.P15.IsPaletteFeasible, `UniversalLaw.Spec.P15.IsProperDownset, `UniversalLaw.Spec.P15.IsRealizedFamily, `UniversalLaw.Spec.P15.IsTransversal, `UniversalLaw.Spec.P15.LocalDecomposability, `UniversalLaw.Spec.P15.LocalObstructionIsFullBlock, `UniversalLaw.Spec.P15.LocalPriceBoundFull, `UniversalLaw.Spec.P15.LocalPriceChain, `UniversalLaw.Spec.P15.LocalRestriction, `UniversalLaw.Spec.P15.LogOnePlusBelowIdentity, `UniversalLaw.Spec.P15.LogRatioBound, `UniversalLaw.Spec.P15.MinimalForbiddenCharacterisation, `UniversalLaw.Spec.P15.OddMajorityIdentity, `UniversalLaw.Spec.P15.PaletteNumberAttained, `UniversalLaw.Spec.P15.PhiAtPStar, `UniversalLaw.Spec.P15.RatioExponentBound, `UniversalLaw.Spec.P15.RestrictedLocalBudget, `UniversalLaw.Spec.P15.RestrictedTransformedPriceBudget, `UniversalLaw.Spec.P15.RhoStarBelowSixSevenths, `UniversalLaw.Spec.P15.RhoStarEnclosure, `UniversalLaw.Spec.P15.RhoTestSufficient, `UniversalLaw.Spec.P15.SharpnessOfRhoStar, `UniversalLaw.Spec.P15.SixSeventhsCertificate, `UniversalLaw.Spec.P15.TriangleNeedsThree, `UniversalLaw.Spec.P15.TwoCoordGeneratorPrices, `UniversalLaw.Spec.P15.WholeGroundChromaticNumber, `UniversalLaw.Spec.RN.AnnulusBridgeAllHeight2, `UniversalLaw.Spec.RN.ConditionalMeanIdentityN3, `UniversalLaw.Spec.RN.D5aPuncturedPinDisks, `UniversalLaw.Spec.RN.D5bMicrodisk, `UniversalLaw.Spec.RN.D5cCollarC1, `UniversalLaw.Spec.RN.D5dSummedPinNeighbourhoodC2, `UniversalLaw.Spec.RN.D5eIntermediateShellsI3I4, `UniversalLaw.Spec.RN.D5fGlobalWindowFirstMomentI5, `UniversalLaw.Spec.RN.DimensionLiftCompactCollarC, `UniversalLaw.Spec.RN.DimensionLiftGlobalWindowG, `UniversalLaw.Spec.RN.DimensionLiftIntermediateShellsI, `UniversalLaw.Spec.RN.DimensionLiftPuncturedPinBallP, `UniversalLaw.Spec.RN.ExponentialTailLayerCakeN5, `UniversalLaw.Spec.RN.FixedAnnulusWindowCandidateA2, `UniversalLaw.Spec.RN.FixedRemoteCountBoundA3, `UniversalLaw.Spec.RN.FixedRemoteMeanMeasureA2, `UniversalLaw.Spec.RN.FourierCutoffPlanarSecondFactorialF4, `UniversalLaw.Spec.RN.HessianHasInertia, `UniversalLaw.Spec.RN.HolderConsequenceN2, `UniversalLaw.Spec.RN.HolderCountBoundN1, `UniversalLaw.Spec.RN.IsCriticalPoint, `UniversalLaw.Spec.RN.PalmRouteFactorialMomentsQ, `UniversalLaw.Spec.RN.PalmRouteSecondFactorialTheta, `UniversalLaw.Spec.RN.RareClusterPoissonObstructionO, `UniversalLaw.Spec.RN.RemoteCollisionSecondFactorialD, `UniversalLaw.Spec.RN.UniformCounterexampleN4, `UniversalLaw.Spec.RN.WitnessCollisionNearPinPairsTarget, `UniversalLaw.Spec.Side24.CoefficientEnclosure, `UniversalLaw.Spec.Side24.ConeMomentD1Identity, `UniversalLaw.Spec.Side24.ConeMomentD2Algebra, `UniversalLaw.Spec.Side24.ConeMomentD2Identity, `UniversalLaw.Spec.Side24.CovarianceOrdering, `UniversalLaw.Spec.Side24.DensityComparison, `UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne, `UniversalLaw.Spec.Side24.DerivativeBoundAtZero, `UniversalLaw.Spec.Side24.DerivativePairingBound, `UniversalLaw.Spec.Side24.ElementaryConeIntegral, `UniversalLaw.Spec.Side24.EntrywiseCovarianceBound, `UniversalLaw.Spec.Side24.ExponentialBounds, `UniversalLaw.Spec.Side24.ImageBound, `UniversalLaw.Spec.Side24.IntegrandRatioBounds, `UniversalLaw.Spec.Side24.IsUnitTuple, `UniversalLaw.Spec.Side24.LatticeImageSumBound, `UniversalLaw.Spec.Side24.NegDefCone, `UniversalLaw.Spec.Side24.RayleighSquareLaw, `UniversalLaw.Spec.Side24.ReferenceCoefficientIdentity, `UniversalLaw.Spec.Side24.ReferenceComparison, `UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound, `UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks, `UniversalLaw.Spec.Side24.ReferencePinDensityProduct, `UniversalLaw.Spec.Side24.SuccessiveTermRatio, `UniversalLaw.Spec.Side24.ThetaSummable, `UniversalLaw.Spec.Side24.TraceMoments]

run_cmd do
  let env ← getEnv
  let some ci := env.find? `UniversalLaw.Proofs.Side24Analysis.derivativeBoundAtZero | throwError "inventory theorem missing"
  let expected : Expr := .const `UniversalLaw.Spec.Side24.DerivativeBoundAtZero []
  logInfo m!"EXACT_TYPE_INVENTORY {toString (ci.type == expected)} raw={ci.type}"
  let isThm := match ci with | .thmInfo _ => true | _ => false
  logInfo m!"INVENTORY_IS_THEOREM {toString isThm}"
  let some hci := env.find? `UniversalLaw.Proofs.Side24Analysis.Aux.d0_derivativeBoundAtZero_of_derivativePairingBound | throwError "helper missing"
  let ok := match hci.type with
    | .forallE _ (.const `UniversalLaw.Spec.Side24.DerivativePairingBound []) (.const `UniversalLaw.Spec.Side24.DerivativeBoundAtZero []) _ => true
    | _ => false
  logInfo m!"EXACT_TYPE_HELPER {toString ok} raw={hci.type}"
  let some pci := env.find? `UniversalLaw.Spec.Side24.DerivativeBoundAtZero | throwError "Prop missing"
  let modIdx := env.getModuleIdxFor? `UniversalLaw.Spec.Side24.DerivativeBoundAtZero
  let modName := match modIdx with | some i => toString env.header.moduleNames[i.toNat]! | none => "CURRENT_FILE"
  let pkind := match pci with | .defnInfo _ => "def" | .thmInfo _ => "thm" | .axiomInfo _ => "AXIOM" | .opaqueInfo _ => "OPAQUE" | _ => "other"
  logInfo m!"PROP_ORIGIN module={modName} kind={pkind} type={pci.type}"
  let modIdx2 := env.getModuleIdxFor? `UniversalLaw.Spec.Side24.DerivativePairingBound
  let modName2 := match modIdx2 with | some i => toString env.header.moduleNames[i.toNat]! | none => "CURRENT_FILE"
  logInfo m!"UPSTREAM_PROP_ORIGIN module={modName2}"

/-! Every constant of the audited modules (or, when `auditModules` is empty, every constant of
this file under `auditPrefix`, private names included) is classified by Lean's own `ConstantInfo`
and type, never by source text. `theorem`: a theorem that is not private, not under an `Aux`
namespace and carries a declaration range, whose type after `forallTelescope` is a registered Prop
constant applied to exactly the telescope's variables (reported with that constant). `aux_theorem`:
a private theorem or one under `Aux`. `generated`: a range-less theorem (structure `injEq`,
`sizeOf_spec`, `_proof_*`, an equation lemma `eq_*` of an unfolded statement definition, which may
be generated in several modules and is audited once), refused if its head is a registered Prop. `auxiliary`: definitions,
instances, inductive types, constructors and recursors whose type is not a proposition.
`REFUSED:<why>`: an axiom, opaque, quotient or unsafe constant; a definition whose type is a
proposition (a proof written as `def`); a non-aux theorem whose head is not registered (`type`) or
whose arguments are not exactly its own parameters (`premise`: an extra hypothesis or a narrowed
argument); any transitive axiom outside propext, Classical.choice and Quot.sound (`axioms`).
One `AUDIT <class> <name> [<head>]` line per constant, then `AUDIT END <count> <refused>
<theorems>`; any refusal makes the file fail. -/
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut constants : Array ConstantInfo := #[]
  if auditModules.isEmpty then
    constants := env.constants.foldStage2 (fun acc _ ci =>
      let user := (privateToUserName? ci.name).getD ci.name
      if auditPrefix.isPrefixOf user then acc.push ci else acc) #[]
  else
    let mut found : Array Name := #[]
    for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
      if auditModules.contains modName then
        found := found.push modName
        constants := constants ++ data.constants
    unless found.size == auditModules.size do
      throwError "AUDIT REFUSED: audited module not imported: {auditModules.filter (!found.contains ·)}"
  -- an equation lemma of an imported definition (`foo.eq_1`) is generated on demand in every module
  -- that unfolds it and is one constant of the environment: audit each name once
  let mut seen : NameSet := {}
  let mut unique : Array ConstantInfo := #[]
  for ci in constants do
    unless seen.contains ci.name do
      seen := seen.insert ci.name
      unique := unique.push ci
  constants := unique
  let mut refused := 0
  let mut theorems := 0
  for ci in constants do
    let n := ci.name
    let axioms ← collectAxioms n
    let bad := axioms.filter (!allowed.contains ·)
    let range ← findDeclarationRanges? n
    let user := (privateToUserName? n).getD n
    let isAux := isPrivateName n || user.components.contains `Aux
    let (head, exact) ← liftTermElabM do
      forallTelescope ci.type fun xs body => do
        let args := body.getAppArgs
        let exact := args.size == xs.size && args.all (fun a => xs.contains a) && xs.all (fun x => args.contains x)
        pure (body.getAppFn.constName?, exact)
    let headStr : String := match head with | some c => toString c | none => "nonconstant"
    let registeredHead := match head with | some c => registeredProps.contains c | none => false
    let isP ← liftTermElabM (isProp ci.type)
    let kind : String :=
      if ci.isUnsafe then "REFUSED:unsafe" else
      match ci with
        | .axiomInfo _ => "REFUSED:axiom"
        | .opaqueInfo _ => "REFUSED:opaque"
        | .quotInfo _ => "REFUSED:quot"
        | .inductInfo _ => "auxiliary"
        | .ctorInfo _ => "auxiliary"
        | .recInfo _ => "auxiliary"
        | .defnInfo _ => if isP then "REFUSED:proof_as_def" else "auxiliary"
        | .thmInfo _ =>
          if isAux then "aux_theorem"
          else if range.isNone then (if registeredHead then s!"REFUSED:generated_proof:{headStr}" else "generated")
          else if registeredHead then (if exact then "theorem" else s!"REFUSED:premise:{headStr}")
          else s!"REFUSED:type:{headStr}"
    let verdict : String :=
      if kind.startsWith "REFUSED" || bad.isEmpty then kind
      else "REFUSED:axioms:" ++ ",".intercalate (bad.map toString).toList
    if verdict.startsWith "REFUSED" then refused := refused + 1
    if verdict == "theorem" then
      theorems := theorems + 1
      logInfo m!"AUDIT theorem {n} {headStr}"
    else
      logInfo m!"AUDIT {verdict} {n}"
  logInfo m!"AUDIT END {constants.size} {refused} {theorems}"
  if refused > 0 then throwError "AUDIT REFUSED {refused} constant(s); see the AUDIT lines"
