import CapFirstExit

open Set unitInterval Filter Topology

namespace CapFirstExit

/-- A pin-preserving quartic perturbation of `h1ToyF`: `q(x) = (x² - 1/4)²/480` vanishes to second
order at both pins, so they stay critical and the gap stays `1/6`, while `f_xxx = 2 + x/20` varies
and `D⁴f = 1/20 ≠ 0`. -/
noncomputable def quarticToyF (p : ℝ × ℝ) : ℝ :=
  p.1 ^ 3 / 3 - p.1 / 4 - 20 * p.2 ^ 2 + (p.1 ^ 2 - 1 / 4) ^ 2 / 480

theorem quarticToy_E :
    (∃ ε > 0, ∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - ε : ℝ) (2 * 1 + ε), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => quarticToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp quarticToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (quarticToyF (-1 / 2, 0)) (quarticToyF (1 / 2, 0)) := by
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt quarticToyF
      ((p.1 ^ 2 - 1 / 4 + (4 * p.1 ^ 3 - p.1) / 480) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have h1 := HasFDerivAt.add (HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hf p) 3) (1 / 3))
      (HasFDerivAt.const_mul (hf p) (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hs p) 2) 20))
      (HasFDerivAt.const_mul (HasFDerivAt.pow ((HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)) 2)
        (1 / 480))
    convert h1 using 1
    · funext q
      simp only [quarticToyF, Pi.sub_apply, Pi.add_apply]
      ring
    · ext <;> simp <;> ring
  have hD2 : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4 + (4 * q.1 ^ 3 - q.1) / 480) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * p.1 + (12 * p.1 ^ 2 - 1) / 480) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) p := by
    intro p
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4 + (4 * q.1 ^ 3 - q.1) / 480)
        ((2 * p.1 + (12 * p.1 ^ 2 - 1) / 480) • ContinuousLinearMap.fst ℝ ℝ ℝ) p := by
      have := ((HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)).add
        (HasFDerivAt.const_mul ((HasFDerivAt.const_mul (HasFDerivAt.pow (hf p) 3) 4).sub (hf p))
          (1 / 480))
      convert this using 1 <;> first | (funext q; simp; ring) | (ext <;> simp; ring)
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2) ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      HasFDerivAt.const_mul (hs p) (-40)
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hcd : ContDiff ℝ 4 quarticToyF := by unfold quarticToyF; fun_prop
  have hfd : fderiv ℝ quarticToyF = fun p => (p.1 ^ 2 - 1 / 4 + (4 * p.1 ^ 3 - p.1) / 480) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ :=
    funext fun p => (hDf p).fderiv
  have h2 : ∀ (y : ℝ × ℝ) (m : Fin 2 → ℝ × ℝ), iteratedFDeriv ℝ 2 quarticToyF y m =
      (2 * y.1 + (12 * y.1 ^ 2 - 1) / 480) * (m 0).1 * (m 1).1 - 40 * (m 0).2 * (m 1).2 := by
    intro y m
    rw [iteratedFDeriv_two_apply, hfd, (hD2 y).fderiv]
    simp
    ring
  have h3 : ∀ (x : ℝ × ℝ) (m : Fin 3 → ℝ × ℝ),
      iteratedFDeriv ℝ 3 quarticToyF x m = (2 + x.1 / 20) * (m 0).1 * (m 1).1 * (m 2).1 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 2 quarticToyF y (Fin.tail m))
        (((2 + x.1 / 20) * (m 1).1 * (m 2).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) x := by
      have hp : HasFDerivAt (fun y : ℝ × ℝ => 2 * y.1 + (12 * y.1 ^ 2 - 1) / 480)
          ((2 + x.1 / 20) • ContinuousLinearMap.fst ℝ ℝ ℝ) x := by
        have := (HasFDerivAt.const_mul (hf x) 2).add
          (HasFDerivAt.const_mul ((HasFDerivAt.const_mul (HasFDerivAt.pow (hf x) 2) 12).sub_const 1)
            (1 / 480))
        convert this using 1 <;> first | (funext q; simp; ring) | (ext <;> simp; ring)
      have := (hp.mul_const ((m 1).1 * (m 2).1)).sub_const (40 * (m 1).2 * (m 2).2)
      convert this using 1
      · funext y; rw [h2]; simp [Fin.tail]; ring
      · (ext <;> simp); ring
    rw [hg.fderiv]
    simp
    ring
  have h4 : ∀ (x : ℝ × ℝ) (m : Fin 4 → ℝ × ℝ),
      iteratedFDeriv ℝ 4 quarticToyF x m = (1 / 20) * (m 0).1 * (m 1).1 * (m 2).1 * (m 3).1 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 3 quarticToyF y (Fin.tail m))
        ((1 / 20 * (m 1).1 * (m 2).1 * (m 3).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) x := by
      have := (((hf x).const_mul (1 / 20)).const_add 2).mul_const ((m 1).1 * (m 2).1 * (m 3).1)
      convert this using 1
      · funext y; rw [h3]; simp [Fin.tail]; ring
      · (ext <;> simp); ring
    rw [hg.fderiv]
    simp
    ring
  have hsrc := ridge_capHyp_E (E := ℝ) (f := quarticToyF) (U := univ) (r := 1) (m := 81 / 40)
    (n := 1 / 20) (lam := 40) one_pos isOpen_univ (subset_univ _) hcd.contDiffOn
    (fun p _ s₁ s₂ s₃ y₁ y₂ y₃ => by
      rw [h3, h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons]
      have h0 : (2 + p.1 / 20) * s₁ * s₂ * s₃ - s₁ * s₂ * s₃ * ((2 + p.1 / 20) * 1 * 1 * 1) = 0 := by
        ring
      rw [h0, abs_zero]
      exact mul_nonneg (by norm_num) (block_nonneg s₁ s₂ s₃ y₁ y₂ y₃))
    (fun x hx => by
      rw [h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons, mul_one]
      rw [abs_le]
      constructor <;> linarith [hx.1, hx.2])
    (fun p _ => by rw [h4]; simp)
    (fun p _ v => by rw [h4]; simp)
    (by norm_num)
    (fun v => by
      rw [h2]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by simp only [quarticToyF]; norm_num)
  obtain ⟨ε, hε, hex, hcap⟩ := hsrc
  have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => quarticToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    have := slice_hasFDerivAt (hDf (x, 0))
    convert this using 1
    ext
    simp
  obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
  exact ⟨⟨ε, hε, hex⟩, hC⟩

end CapFirstExit
