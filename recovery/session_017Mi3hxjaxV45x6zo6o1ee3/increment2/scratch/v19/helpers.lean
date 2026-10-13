
/-- **L3 in `y` for `∂_x ∇_y f`.** If along the segment `s ↦ (x, s • y)` the third derivatives
satisfy `|T (1, 0) (0, v) (0, y)| ≤ m ‖v‖ ‖y‖` (the block `f_xyy`), then
`‖∂_x ∇_y f (x, y) - ∂_x ∇_y f (x, 0)‖ ≤ m ‖y‖`. -/
theorem dx_grad_increment {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {x m : ℝ} {y : E} (hm : 0 ≤ m)
    (hD3 : ∀ s ∈ Icc (0 : ℝ) 1, ∀ a b : ℝ × E,
      HasFDerivAt (fun q => D2f q a b) (T3 (x, s • y) a b) (x, s • y))
    (hT : ∀ s ∈ Icc (0 : ℝ) 1, ∀ v : E, |T3 (x, s • y) (1, 0) (0, v) (0, y)| ≤ m * ‖v‖ * ‖y‖) :
    ‖(D2f (x, y) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) -
      (D2f (x, 0) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ m * ‖y‖ := by
  refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun v => ?_
  have hφ : ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => D2f (x, s • y) (1, 0) (0, v))
      (T3 (x, s • y) (1, 0) (0, v) (0, y)) s := by
    intro s hs
    have hp : HasDerivAt (fun s : ℝ => (x, s • y)) ((0 : ℝ), y) s := by
      simpa using (hasDerivAt_const s x).prodMk ((hasDerivAt_id s).smul_const y)
    exact (hD3 s hs (1, 0) (0, v)).comp_hasDerivAt s hp
  have hφb : ∀ s ∈ Icc (0 : ℝ) 1, ‖T3 (x, s • y) (1, 0) (0, v) (0, y)‖ ≤ m * ‖v‖ * ‖y‖ :=
    fun s hs => by rw [Real.norm_eq_abs]; exact hT s hs v
  have := (convex_Icc (0 : ℝ) 1).norm_image_sub_le_of_norm_hasDerivWithin_le
    (fun s hs => (hφ s hs).hasDerivWithinAt) hφb (left_mem_Icc.mpr zero_le_one)
    (right_mem_Icc.mpr zero_le_one)
  simp only [one_smul, zero_smul, sub_zero, norm_one, mul_one] at this
  simp only [ContinuousLinearMap.sub_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.inr_apply]
  calc ‖(D2f (x, y)) (1, 0) (0, v) - (D2f (x, 0)) (1, 0) (0, v)‖ ≤ m * ‖v‖ * ‖y‖ := this
    _ = m * ‖y‖ * ‖v‖ := by ring

/-- **L9's arithmetic.** From `δ ‖h'‖ ≤ ‖∂_x∇_y f(x, h)‖ ≤ 2mr + m‖h‖` and `δ ‖h‖ ≤ 2mr²`:
`‖h'‖ ≤ 2s + 2s²` with `s = mr/δ` (SP: `u = 2/k + 2/k²`). -/
theorem l9_arith {δ m r a b c : ℝ} (hδ : 0 < δ) (hm : 0 ≤ m) (hb0 : 0 ≤ b) (ha : δ * a ≤ c)
    (hc : c ≤ 2 * m * r + m * b) (hb : δ * b ≤ 2 * m * r ^ 2) :
    a ≤ 2 * (m * r / δ) + 2 * (m * r / δ) ^ 2 := by
  have key : δ * δ * a ≤ 2 * m * r * δ + 2 * m * m * r ^ 2 := by nlinarith
  have hrhs : 2 * (m * r / δ) + 2 * (m * r / δ) ^ 2 =
      (2 * m * r * δ + 2 * m * m * r ^ 2) / (δ * δ) := by
    field_simp
    ring
  rw [hrhs, le_div_iff₀ (by positivity)]
  linarith

/-- **L12's bound at one point.** With the combined block bound, `f_xxx ≥ 7/4`, `‖a‖, ‖b‖ ≤ u`
and `7/4 - m((1 + u)³ - 1) ≥ 1/4` (`l12_constants`): `T (1, b) (1, b) (1, a) ≥ 1/4`. -/
theorem l12_lower {T : ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {m u : ℝ} {a b : E} (hm : 0 ≤ m)
    (hT : ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hxxx : 7 / 4 ≤ T (1, 0) (1, 0) (1, 0)) (hc : 1 / 4 ≤ 7 / 4 - m * ((1 + u) ^ 3 - 1))
    (ha : ‖a‖ ≤ u) (hb : ‖b‖ ≤ u) : 1 / 4 ≤ T (1, b) (1, b) (1, a) := by
  have h1 := trilinear_lower hT a b
  have hnb := norm_nonneg b
  have hna := norm_nonneg a
  have h4 : (1 + ‖b‖) * (1 + ‖b‖) * (1 + ‖a‖) ≤ (1 + u) ^ 3 := by
    calc (1 + ‖b‖) * (1 + ‖b‖) * (1 + ‖a‖) ≤ (1 + u) * (1 + u) * (1 + u) := by gcongr
      _ = (1 + u) ^ 3 := by ring
  have h5 := mul_le_mul_of_nonneg_left (sub_le_sub_right h4 1) hm
  linarith
