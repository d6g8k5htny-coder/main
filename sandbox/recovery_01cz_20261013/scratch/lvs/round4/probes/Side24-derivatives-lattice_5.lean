import Specifications.Side24

/-! Probe (round 4, group Side24-derivatives-lattice): probe 3 with three cosmetic fixes; assembly of the derivative route
(Leibniz norm bound + Hermite-type recursion, seven explicit orders) for
`DerivativePairingBound`, `DerivativeBoundAtLeastOne`, `DerivativeBoundAtZero`.
Probe file only: `sorry` allowed here and nowhere else. -/

open UniversalLaw.Spec.Side24
open scoped ContDiff

namespace Probe5
variable (d : ℕ)

theorem contDiff_Kinf : ContDiff ℝ ∞ (Kinf d) := by
  unfold Kinf
  have h : ContDiff ℝ ∞ (fun x : E d => -(‖x‖ ^ 2) / 2) :=
    (contDiff_norm_sq ℝ (E := E d)).neg.div_const 2
  exact Real.contDiff_exp.comp h

theorem hasFDerivAt_Kinf (x : E d) :
    HasFDerivAt (Kinf d) (Kinf d x • (-(innerSL ℝ x))) x := by
  have h0 : HasFDerivAt (fun y : E d => ‖y‖ ^ 2) ((2:ℕ) • innerSL ℝ x) x :=
    (hasStrictFDerivAt_norm_sq x).hasFDerivAt
  have h1 := (h0.const_smul (-(1/2 : ℝ))).exp
  convert h1 using 1
  · funext y; simp only [Kinf, Pi.smul_apply, smul_eq_mul]; ring_nf
  · ext v; simp [Kinf]; left; ring

theorem fderiv_Kinf : fderiv ℝ (Kinf d) = fun y => Kinf d y • (-(innerSL ℝ y)) :=
  funext fun y => (hasFDerivAt_Kinf d y).fderiv

theorem contDiff_g : ContDiff ℝ ∞ (fun y : E d => -(innerSL ℝ y)) :=
  (ContinuousLinearMap.contDiff (innerSL ℝ)).neg

theorem g_eq : (fun y : E d => -(innerSL ℝ y)) = ⇑(-(innerSL ℝ (E := E d))) := by
  funext y; simp

theorem norm_g_zero (x : E d) : ‖iteratedFDeriv ℝ 0 (fun y : E d => -(innerSL ℝ y)) x‖ = ‖x‖ := by
  rw [norm_iteratedFDeriv_zero, norm_neg, innerSL_apply_norm]

theorem norm_g_one (x : E d) : ‖iteratedFDeriv ℝ 1 (fun y : E d => -(innerSL ℝ y)) x‖ ≤ 1 := by
  rw [norm_iteratedFDeriv_one, g_eq, ContinuousLinearMap.fderiv, norm_neg]
  exact norm_innerSL_le ℝ

theorem g_higher (x : E d) (k : ℕ) (hk : 2 ≤ k) :
    iteratedFDeriv ℝ k (fun y : E d => -(innerSL ℝ y)) x = 0 := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [g_eq, iteratedFDeriv_succ_eq_comp_right]
  have : fderiv ℝ ⇑(-(innerSL ℝ (E := E d))) = fun _ => -(innerSL ℝ (E := E d)) :=
    funext fun y => (-(innerSL ℝ (E := E d))).fderiv
  simp only [Function.comp_apply, this, iteratedFDeriv_const_of_ne (show j ≠ 0 by omega),
    Pi.zero_apply]
  exact LinearIsometryEquiv.map_zero _

/-- Hermite-type recursion for the operator norms `b_q = ‖D^q Kinf (x)‖`. -/
theorem recursion (x : E d) (q : ℕ) :
    ‖iteratedFDeriv ℝ (q + 2) (Kinf d) x‖ ≤
      ‖x‖ * ‖iteratedFDeriv ℝ (q + 1) (Kinf d) x‖ + (q + 1) * ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  rw [← norm_iteratedFDeriv_fderiv, fderiv_Kinf]
  have h := norm_iteratedFDeriv_smul_le (contDiff_Kinf d) (contDiff_g d) x
    (n := q + 1) (WithTop.coe_le_coe.mpr le_top)
  refine h.trans ?_
  rw [Finset.sum_range_succ, Finset.sum_range_succ]
  have hz : ∑ i ∈ Finset.range q, ((q + 1).choose i : ℝ) * ‖iteratedFDeriv ℝ i (Kinf d) x‖ *
      ‖iteratedFDeriv ℝ (q + 1 - i) (fun y : E d => -(innerSL ℝ y)) x‖ = 0 := by
    refine Finset.sum_eq_zero fun i hi => ?_
    rw [Finset.mem_range] at hi
    rw [g_higher d x _ (by omega), norm_zero, mul_zero]
  rw [hz, zero_add, Nat.choose_self, Nat.choose_succ_self_right, Nat.add_sub_cancel_left,
    Nat.sub_self, norm_g_zero]
  have h1 := norm_g_one d x
  have hb : 0 ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := norm_nonneg _
  push_cast
  nlinarith [mul_le_mul_of_nonneg_left h1 (mul_nonneg (by positivity : (0:ℝ) ≤ (q:ℝ) + 1) hb)]

theorem Kinf_pos (x : E d) : 0 < Kinf d x := Real.exp_pos _

theorem b0 (x : E d) : ‖iteratedFDeriv ℝ 0 (Kinf d) x‖ = Kinf d x := by
  rw [norm_iteratedFDeriv_zero, Real.norm_eq_abs, abs_of_pos (Kinf_pos d x)]

theorem b1 (x : E d) : ‖iteratedFDeriv ℝ 1 (Kinf d) x‖ = Kinf d x * ‖x‖ := by
  rw [norm_iteratedFDeriv_one, fderiv_Kinf, norm_smul, norm_neg, innerSL_apply_norm,
    Real.norm_eq_abs, abs_of_pos (Kinf_pos d x)]

theorem b2 (x : E d) : ‖iteratedFDeriv ℝ 2 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 2 + 1) := by
  have h := recursion d x 0
  rw [b1, b0] at h
  push_cast at h
  nlinarith [h]

theorem b3 (x : E d) : ‖iteratedFDeriv ℝ 3 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 3 + 3 * ‖x‖) := by
  have h := recursion d x 1
  have h2 := b2 d x
  have h1 := b1 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h2 hr]

theorem b4 (x : E d) :
    ‖iteratedFDeriv ℝ 4 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 4 + 6 * ‖x‖ ^ 2 + 3) := by
  have h := recursion d x 2
  have h3 := b3 d x
  have h2 := b2 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h3 hr]

theorem b5 (x : E d) :
    ‖iteratedFDeriv ℝ 5 (Kinf d) x‖ ≤ Kinf d x * (‖x‖ ^ 5 + 10 * ‖x‖ ^ 3 + 15 * ‖x‖) := by
  have h := recursion d x 3
  have h4 := b4 d x
  have h3 := b3 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h4 hr]

theorem b6 (x : E d) :
    ‖iteratedFDeriv ℝ 6 (Kinf d) x‖ ≤
      Kinf d x * (‖x‖ ^ 6 + 15 * ‖x‖ ^ 4 + 45 * ‖x‖ ^ 2 + 15) := by
  have h := recursion d x 4
  have h5 := b5 d x
  have h4 := b4 d x
  have hr : 0 ≤ ‖x‖ := norm_nonneg _
  push_cast at h
  nlinarith [mul_le_mul_of_nonneg_left h5 hr]

theorem pairing_all (x : E d) (q : ℕ) (hq : q ≤ 6) :
    ‖iteratedFDeriv ℝ q (Kinf d) x‖ ≤ Kinf d x * pairingSum q ‖x‖ := by
  interval_cases q
  · rw [b0]; simp [pairingSum]
  · rw [b1]; simp [pairingSum]
  · refine (b2 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (b3 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (b4 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (b5 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring
  · refine (b6 d x).trans (le_of_eq ?_)
    congr 1
    simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
    try ring

theorem abs_le_opNorm (q : ℕ) (x : E d) (u : Fin q → E d) (hu : IsUnitTuple d q u) :
    |iteratedFDeriv ℝ q (Kinf d) x u| ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  have h := (iteratedFDeriv ℝ q (Kinf d) x).le_opNorm u
  have hp : ∏ i, ‖u i‖ = 1 := Finset.prod_eq_one (fun i _ => hu i)
  rw [hp, mul_one] at h
  simpa [Real.norm_eq_abs] using h

theorem derivativePairingBound : DerivativePairingBound := by
  intro d q hq x u hu
  exact (abs_le_opNorm d q x u hu).trans (pairing_all d x q hq)

theorem derivativeBoundAtZero : DerivativeBoundAtZero := by
  intro d q hq u hu
  refine ((abs_le_opNorm d q 0 u hu).trans (pairing_all d 0 q hq)).trans ?_
  simp only [Kinf, norm_zero]
  interval_cases q <;> simp [pairingSum, Finset.sum_range_succ, Nat.factorial] <;> norm_num

theorem derivativeBoundAtLeastOne : DerivativeBoundAtLeastOne := by
  intro d q hq x hx u hu
  refine ((abs_le_opNorm d q x u hu).trans (pairing_all d x q hq)).trans ?_
  have hK : 0 < Kinf d x := Kinf_pos d x
  have h0 : 1 ≤ ‖x‖ ^ 6 := one_le_pow₀ hx
  have h1 : ‖x‖ ≤ ‖x‖ ^ 6 := by
    simpa using pow_le_pow_right₀ hx (show 1 ≤ 6 by norm_num)
  have h2 : ‖x‖ ^ 2 ≤ ‖x‖ ^ 6 := pow_le_pow_right₀ hx (by norm_num)
  have h3 : ‖x‖ ^ 3 ≤ ‖x‖ ^ 6 := pow_le_pow_right₀ hx (by norm_num)
  have h4 : ‖x‖ ^ 4 ≤ ‖x‖ ^ 6 := pow_le_pow_right₀ hx (by norm_num)
  have h5 : ‖x‖ ^ 5 ≤ ‖x‖ ^ 6 := pow_le_pow_right₀ hx (by norm_num)
  have hKe : Kinf d x = Real.exp (-(‖x‖ ^ 2) / 2) := rfl
  rw [← hKe]
  interval_cases q <;> simp only [pairingSum, Finset.sum_range_succ, Finset.sum_range_zero,
    Nat.factorial] <;> norm_num <;> nlinarith

end Probe5

#print axioms Probe5.derivativePairingBound
#print axioms Probe5.derivativeBoundAtZero
#print axioms Probe5.derivativeBoundAtLeastOne
