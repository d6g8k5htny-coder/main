import Specifications.Side24

/-! Probe (round 4, group Side24-derivatives-lattice): derivative route for
`DerivativePairingBound`, `DerivativeBoundAtLeastOne`, `DerivativeBoundAtZero`.
Probe file only: `sorry` and `#check` are allowed here and nowhere else. -/

open UniversalLaw.Spec.Side24
open scoped ContDiff

#check @norm_iteratedFDeriv_smul_le
#check @norm_iteratedFDeriv_mul_le
#check @norm_iteratedFDeriv_fderiv
#check @norm_iteratedFDeriv_zero
#check @norm_iteratedFDeriv_one
#check @innerSL_apply_norm
#check @norm_innerSL_le
#check @ContinuousMultilinearMap.le_opNorm
#check @ContinuousMultilinearMap.opNorm_le_bound
#check @iteratedFDeriv_const_of_ne
#check @ContinuousLinearMap.fderiv
#check @hasStrictFDerivAt_norm_sq
#check @HasFDerivAt.exp
#check @contDiff_norm_sq
#check @Real.contDiff_exp
#check @ContinuousLinearMap.contDiff
#check @iteratedFDeriv_succ_apply_right
#check @iteratedFDeriv_succ_eq_comp_right
#check @iteratedFDeriv_comp_add_right
#check @iteratedFDeriv_const_smul_apply
#check @iteratedFDeriv_tsum
#check @hasFDerivAt_tsum_of_isPreconnected
#check @isBoundedBilinearMap_inner
#check @IsBoundedBilinearMap.toContinuousLinearMap

namespace Probe
variable (d : ℕ)

-- A. smoothness of Kinf
theorem contDiff_Kinf : ContDiff ℝ ∞ (Kinf d) := by
  unfold Kinf
  have h : ContDiff ℝ ∞ (fun x : E d => -(‖x‖ ^ 2) / 2) :=
    (contDiff_norm_sq (E := E d)).neg.div_const 2
  exact Real.contDiff_exp.comp h

-- B. derivative formula
theorem hasFDerivAt_Kinf (x : E d) :
    HasFDerivAt (Kinf d) (Kinf d x • (-(innerSL ℝ x))) x := by
  have h0 : HasFDerivAt (fun y : E d => ‖y‖ ^ 2) ((2:ℕ) • innerSL ℝ x) x :=
    (hasStrictFDerivAt_norm_sq x).hasFDerivAt
  have h1 := (h0.const_smul (-(1/2 : ℝ))).exp
  convert h1 using 1
  · funext y; simp only [Kinf, smul_eq_mul]; ring_nf
  · ext v; simp [Kinf]; ring

theorem fderiv_Kinf : fderiv ℝ (Kinf d) = fun y => Kinf d y • (-(innerSL ℝ y)) :=
  funext fun y => (hasFDerivAt_Kinf d y).fderiv

-- C. contraction ≤ operator norm for unit tuples
theorem abs_le_opNorm (q : ℕ) (x : E d) (u : Fin q → E d) (hu : IsUnitTuple d q u) :
    |iteratedFDeriv ℝ q (Kinf d) x u| ≤ ‖iteratedFDeriv ℝ q (Kinf d) x‖ := by
  have h := (iteratedFDeriv ℝ q (Kinf d) x).le_opNorm u
  have hp : ∏ i, ‖u i‖ = 1 := Finset.prod_eq_one (fun i _ => hu i)
  rw [hp, mul_one] at h
  simpa [Real.norm_eq_abs] using h

-- D. pairingSum explicit values
example (r : ℝ) : pairingSum 6 r = r ^ 6 + 15 * r ^ 4 + 45 * r ^ 2 + 15 := by
  simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
  ring
example : pairingSum 6 0 = 15 := by
  simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
  norm_num
example (r : ℝ) : pairingSum 5 r = r ^ 5 + 10 * r ^ 3 + 15 * r := by
  simp [pairingSum, Finset.sum_range_succ, Nat.factorial]
  ring
example (r : ℝ) (hr : 1 ≤ r) : r ^ 6 + 15 * r ^ 4 + 45 * r ^ 2 + 15 ≤ 76 * r ^ 6 := by
  have h2 : r ^ 2 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  have h4 : r ^ 4 ≤ r ^ 6 := pow_le_pow_right₀ hr (by norm_num)
  have h0 : 1 ≤ r ^ 6 := one_le_pow₀ hr
  linarith

-- F. ContDiff of the CLM-valued factor y ↦ -(innerSL ℝ y)
example : ContDiff ℝ ∞ (fun y : E d => -(innerSL ℝ y)) := by
  first
  | exact (ContinuousLinearMap.contDiff (innerSL ℝ)).neg
  | exact ((innerSL ℝ : E d →L[ℝ] E d →L[ℝ] ℝ).contDiff).neg
  | sorry

-- F'. alternative via the bounded bilinear map
noncomputable def B : E d →L[ℝ] E d →L[ℝ] ℝ :=
  (isBoundedBilinearMap_inner (𝕜 := ℝ) (E := E d)).toContinuousLinearMap
example (y : E d) : B d y = innerSL ℝ y := by
  ext v
  simp [B]
example : ContDiff ℝ ∞ (fun y : E d => -(B d y)) := (B d).contDiff.neg

-- E. Leibniz step
example (x : E d) (q : ℕ) (hg : ContDiff ℝ ∞ (fun y : E d => -(innerSL ℝ y))) :
    ‖iteratedFDeriv ℝ (q + 1) (Kinf d) x‖ ≤
      ∑ i ∈ Finset.range (q + 1), (q.choose i : ℝ) * ‖iteratedFDeriv ℝ i (Kinf d) x‖ *
        ‖iteratedFDeriv ℝ (q - i) (fun y : E d => -(innerSL ℝ y)) x‖ := by
  rw [← norm_iteratedFDeriv_fderiv, fderiv_Kinf]
  refine norm_iteratedFDeriv_smul_le (contDiff_Kinf d) hg x ?_
  first
  | exact WithTop.coe_le_coe.mpr le_top
  | exact_mod_cast le_top
  | simp

-- G. iterated derivatives of a CLM-valued linear function
example (x : E d) (L : E d →L[ℝ] (E d →L[ℝ] ℝ)) : ‖iteratedFDeriv ℝ 1 L x‖ = ‖L‖ := by
  rw [norm_iteratedFDeriv_one, L.fderiv]
example (x : E d) (L : E d →L[ℝ] (E d →L[ℝ] ℝ)) (k : ℕ) (hk : 2 ≤ k) :
    iteratedFDeriv ℝ k L x = 0 := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  rw [iteratedFDeriv_succ_eq_comp_right]
  have : fderiv ℝ L = fun _ => L := funext fun y => L.fderiv
  simp only [Function.comp_apply, this, iteratedFDeriv_const_of_ne (show j ≠ 0 by omega)]
  simp

-- H. norm of the zeroth derivative of the factor
example (x : E d) : ‖iteratedFDeriv ℝ 0 (fun y : E d => -(innerSL ℝ y)) x‖ = ‖x‖ := by
  rw [norm_iteratedFDeriv_zero, norm_neg, innerSL_apply_norm]

end Probe
