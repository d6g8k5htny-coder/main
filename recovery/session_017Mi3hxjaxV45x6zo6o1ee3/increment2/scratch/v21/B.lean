import CapFirstExit

open Set unitInterval

namespace CapFirstExit

section Slice

open Filter Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **`‖∂_x D_y² f‖ ≤ m` from the third-derivative block bound.** -/
theorem hessian_x_bound {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx' : E →L[ℝ] E →L[ℝ] ℝ} {J : Set ℝ} {m x : ℝ}
    (hJ : UniqueDiffWithinAt ℝ J x) (hxJ : x ∈ J) (hm : 0 ≤ m)
    (hH : ∀ t ∈ J, HasFDerivAt (fun y => (Df (t, y)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      (H (t, 0)) 0)
    (hD2 : ∀ t ∈ J, HasFDerivAt Df (D2f (t, 0)) (t, 0))
    (hD3 : ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 (x, 0) a b) (x, 0))
    (hT : ∀ v w : E, |T3 (x, 0) (0, v) (0, w) (1, 0)| ≤ m * (‖v‖ * ‖w‖))
    (hHx : HasDerivAt (fun t => H (t, 0)) Hx' x) : ‖Hx'‖ ≤ m := by
  have hp : HasDerivAt (fun t : ℝ => (t, (0 : E))) ((1 : ℝ), (0 : E)) x :=
    (hasDerivAt_id x).prodMk (hasDerivAt_const x (0 : E))
  have key : ∀ v w : E, Hx' v w = T3 (x, 0) (0, v) (0, w) (1, 0) := by
    intro v w
    have h1 : HasDerivAt (fun t => H (t, 0) v w) (Hx' v w) x := by
      have := (hHx.clm_apply (hasDerivAt_const x v)).clm_apply (hasDerivAt_const x w)
      simpa using this
    have h2 : HasDerivAt (fun t => D2f (t, 0) (0, v) (0, w)) (T3 (x, 0) (0, v) (0, w) (1, 0)) x :=
      HasFDerivAt.comp_hasDerivAt (l := fun q => D2f q (0, v) (0, w)) x (hD3 (0, v) (0, w)) hp
    have h3 : HasDerivWithinAt (fun t => D2f (t, 0) (0, v) (0, w)) (Hx' v w) J x :=
      h1.hasDerivWithinAt.congr_of_mem (fun t ht => (transverse_hessian_eq (hH t ht) (hD2 t ht) v w).symm) hxJ
    exact hJ.eq_deriv _ h3 h2.hasDerivWithinAt
  refine ContinuousLinearMap.opNorm_le_bound _ hm fun v => ?_
  refine ContinuousLinearMap.opNorm_le_bound _ (mul_nonneg hm (norm_nonneg v)) fun w => ?_
  rw [key, Real.norm_eq_abs, mul_assoc]
  exact hT v w

/-- **`‖∂_ρ D_y² f‖ ≤ m ‖y‖` along a radius, from the block bound.** -/
theorem hessian_r_bound {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hr' : E →L[ℝ] E →L[ℝ] ℝ} {x m s : ℝ} {y : E}
    (hm : 0 ≤ m) (hs : s ∈ Icc (0 : ℝ) 1)
    (hH : ∀ τ ∈ Icc (0 : ℝ) 1, HasFDerivAt (fun z => (Df (x, z)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      (H (x, τ • y)) (τ • y))
    (hD2 : ∀ τ ∈ Icc (0 : ℝ) 1, HasFDerivAt Df (D2f (x, τ • y)) (x, τ • y))
    (hD3 : ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 (x, s • y) a b) (x, s • y))
    (hT : ∀ v w : E, |T3 (x, s • y) (0, v) (0, w) (0, y)| ≤ m * (‖v‖ * ‖w‖ * ‖y‖))
    (hHr : HasDerivAt (fun τ : ℝ => H (x, τ • y)) Hr' s) : ‖Hr'‖ ≤ m * ‖y‖ := by
  have hp : HasDerivAt (fun τ : ℝ => (x, τ • y)) ((0 : ℝ), y) s := by
    have := (hasDerivAt_const s x).prodMk ((hasDerivAt_id s).smul_const y)
    simpa using this
  have key : ∀ v w : E, Hr' v w = T3 (x, s • y) (0, v) (0, w) (0, y) := by
    intro v w
    have h1 : HasDerivAt (fun τ => H (x, τ • y) v w) (Hr' v w) s := by
      have := (hHr.clm_apply (hasDerivAt_const s v)).clm_apply (hasDerivAt_const s w)
      simpa using this
    have h2 : HasDerivAt (fun τ : ℝ => D2f (x, τ • y) (0, v) (0, w))
        (T3 (x, s • y) (0, v) (0, w) (0, y)) s :=
      HasFDerivAt.comp_hasDerivAt (l := fun q => D2f q (0, v) (0, w)) s (hD3 (0, v) (0, w)) hp
    have h3 : HasDerivWithinAt (fun τ : ℝ => D2f (x, τ • y) (0, v) (0, w)) (Hr' v w) (Icc 0 1) s :=
      h1.hasDerivWithinAt.congr_of_mem
        (fun τ hτ => (transverse_hessian_eq (hH τ hτ) (hD2 τ hτ) v w).symm) hs
    exact (uniqueDiffOn_Icc zero_lt_one s hs).eq_deriv _ h3 h2.hasDerivWithinAt
  have hmy : 0 ≤ m * ‖y‖ := mul_nonneg hm (norm_nonneg y)
  refine ContinuousLinearMap.opNorm_le_bound _ hmy fun v => ?_
  refine ContinuousLinearMap.opNorm_le_bound _ (mul_nonneg hmy (norm_nonneg v)) fun w => ?_
  rw [key, Real.norm_eq_abs]
  calc |T3 (x, s • y) (0, v) (0, w) (0, y)| ≤ m * (‖v‖ * ‖w‖ * ‖y‖) := hT v w
    _ = m * ‖y‖ * ‖v‖ * ‖w‖ := by ring

/-- **Convex and concave envelopes along the axis from a second-derivative bound.** -/
theorem axis_convexConcave {Df : ℝ × E → ℝ × E →L[ℝ] ℝ}
    {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {a c B : ℝ} {u : ℝ × E}
    (hD2 : ∀ t ∈ Icc a c, HasFDerivAt Df (D2f (t, 0)) (t, 0))
    (hD3 : ∀ t ∈ Icc a c, HasFDerivAt (fun q => D2f q (1, 0) u) (T3 (t, 0) (1, 0) u) (t, 0))
    (hB : ∀ t ∈ Icc a c, |T3 (t, 0) (1, 0) u (1, 0)| ≤ B) :
    ConvexOn ℝ (Icc a c) (fun t => Df (t, 0) u + B / 2 * t ^ 2) ∧
      ConcaveOn ℝ (Icc a c) (fun t => Df (t, 0) u - B / 2 * t ^ 2) := by
  have hp : ∀ t : ℝ, HasDerivAt (fun s : ℝ => (s, (0 : E))) ((1 : ℝ), (0 : E)) t := fun t =>
    (hasDerivAt_id t).prodMk (hasDerivAt_const t (0 : E))
  have hg : ∀ t ∈ Icc a c, HasDerivAt (fun s => Df (s, 0) u) (D2f (t, 0) (1, 0) u) t := by
    intro t ht
    have h1 : HasDerivAt (fun s => Df (s, 0)) (D2f (t, 0) (1, 0)) t :=
      (hD2 t ht).comp_hasDerivAt t (hp t)
    simpa using h1.clm_apply (hasDerivAt_const t u)
  have hg' : ∀ t ∈ Icc a c, HasDerivAt (fun s => D2f (s, 0) (1, 0) u) (T3 (t, 0) (1, 0) u (1, 0)) t :=
    fun t ht => HasFDerivAt.comp_hasDerivAt (l := fun q => D2f q (1, 0) u) t (hD3 t ht) (hp t)
  have hcont : ContinuousOn (fun s => Df (s, 0) u) (Icc a c) :=
    fun t ht => (hg t ht).continuousAt.continuousWithinAt
  have hsq : ∀ t : ℝ, HasDerivAt (fun s : ℝ => B / 2 * s ^ 2) (B * t) t := by
    intro t
    have := (hasDerivAt_pow 2 t).const_mul (B / 2)
    convert this using 1
    ring
  have hlin : ∀ t : ℝ, HasDerivAt (fun s : ℝ => B * s) B t := by
    intro t
    simpa using (hasDerivAt_id t).const_mul B
  have hIo : ∀ t ∈ interior (Icc a c), t ∈ Icc a c := fun t ht => interior_subset ht
  constructor
  · refine convexOn_of_hasDerivWithinAt2_nonneg (convex_Icc a c)
      (f' := fun t => D2f (t, 0) (1, 0) u + B * t) (f'' := fun t => T3 (t, 0) (1, 0) u (1, 0) + B)
      (hcont.add (continuousOn_const.mul (continuousOn_id.pow 2))) ?_ ?_ ?_
    · intro t ht
      exact ((hg t (hIo t ht)).add (hsq t)).hasDerivWithinAt
    · intro t ht
      exact ((hg' t (hIo t ht)).add (hlin t)).hasDerivWithinAt
    · intro t ht
      have := hB t (hIo t ht)
      have := neg_abs_le (T3 (t, 0) (1, 0) u (1, 0))
      show 0 ≤ T3 (t, 0) (1, 0) u (1, 0) + B
      linarith
  · refine concaveOn_of_hasDerivWithinAt2_nonpos (convex_Icc a c)
      (f' := fun t => D2f (t, 0) (1, 0) u - B * t) (f'' := fun t => T3 (t, 0) (1, 0) u (1, 0) - B)
      (hcont.sub (continuousOn_const.mul (continuousOn_id.pow 2))) ?_ ?_ ?_
    · intro t ht
      exact ((hg t (hIo t ht)).sub (hsq t)).hasDerivWithinAt
    · intro t ht
      exact ((hg' t (hIo t ht)).sub (hlin t)).hasDerivWithinAt
    · intro t ht
      have := hB t (hIo t ht)
      have := le_abs_self (T3 (t, 0) (1, 0) u (1, 0))
      show T3 (t, 0) (1, 0) u (1, 0) - B ≤ 0
      linarith

/-- The block weight is nonnegative. -/
theorem block_nonneg {F : Type*} [SeminormedAddCommGroup F] (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : F) :
    0 ≤ (|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃| := by
  have a1 : |s₁| ≤ |s₁| + ‖y₁‖ := le_add_of_nonneg_right (norm_nonneg _)
  have a2 : |s₂| ≤ |s₂| + ‖y₂‖ := le_add_of_nonneg_right (norm_nonneg _)
  have a3 : |s₃| ≤ |s₃| + ‖y₃‖ := le_add_of_nonneg_right (norm_nonneg _)
  have := mul_le_mul (mul_le_mul a1 a2 (abs_nonneg _) (by positivity)) a3 (abs_nonneg _)
    (by positivity)
  linarith

/-- **The continuity step: the block bound from `D` to a collar.** -/
theorem block_bound_collar {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ}
    {T4 : ℝ × E → ℝ × E → ℝ × E → ℝ × E → ℝ} {r m K ε₀ ε : ℝ} (hr : 0 < r) (hK : 0 ≤ K)
    (hε : 0 ≤ ε) (hεε : ε ≤ ε₀)
    (hT : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hD4x : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ a b c : ℝ × E, HasDerivAt (fun t => T3 (t, p.2) a b c) (T4 p a b c) p.1)
    (hT4 : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T4 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T4 p (1, 0) (1, 0) (1, 0)| ≤
        K * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|)) :
    ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        (m + K * ε) * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
  rintro ⟨x, y⟩ ⟨hx, hy⟩ s₁ s₂ s₃ y₁ y₂ y₃
  have hX := block_nonneg s₁ s₂ s₃ y₁ y₂ y₃
  set X := (|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃| with hXdef
  obtain ⟨x₀, hx₀, hxx₀⟩ : ∃ x₀ ∈ Icc (-2 * r) (2 * r), |x - x₀| ≤ ε := by
    by_cases h1 : x ≤ -2 * r
    · refine ⟨-2 * r, ⟨le_rfl, ?_⟩, ?_⟩
      · linarith [hx.1]
      · rw [abs_le]; constructor <;> linarith [hx.1]
    · by_cases h2 : 2 * r ≤ x
      · refine ⟨2 * r, ⟨?_, le_rfl⟩, ?_⟩
        · linarith
        · rw [abs_le]; constructor <;> linarith [hx.2]
      · exact ⟨x, ⟨by linarith, by linarith⟩, by simp [hε]⟩
  set g : ℝ → ℝ := fun t => T3 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃) -
    s₁ * s₂ * s₃ * T3 (t, y) (1, 0) (1, 0) (1, 0) with hg
  have hderiv : ∀ t ∈ Icc (-2 * r - ε₀) (2 * r + ε₀), HasDerivWithinAt g
      (T4 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T4 (t, y) (1, 0) (1, 0) (1, 0))
      (Icc (-2 * r - ε₀) (2 * r + ε₀)) t := by
    intro t ht
    have hp : (t, y) ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r) := ⟨ht, hy⟩
    have h1 : HasDerivAt (fun t => T3 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃))
        (T4 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃)) t := hD4x _ hp _ _ _
    have h2 : HasDerivAt (fun t => T3 (t, y) (1, 0) (1, 0) (1, 0)) (T4 (t, y) (1, 0) (1, 0) (1, 0)) t :=
      hD4x _ hp _ _ _
    exact (h1.sub (h2.const_mul (s₁ * s₂ * s₃))).hasDerivWithinAt
  have hbound : ∀ t ∈ Icc (-2 * r - ε₀) (2 * r + ε₀),
      ‖T4 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T4 (t, y) (1, 0) (1, 0) (1, 0)‖ ≤ K * X :=
    fun t ht => by rw [Real.norm_eq_abs]; exact hT4 (t, y) ⟨ht, hy⟩ _ _ _ _ _ _
  have hxS : x ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) := ⟨by linarith [hx.1], by linarith [hx.2]⟩
  have hx₀S : x₀ ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) :=
    ⟨by linarith [hx₀.1, hε, hεε], by linarith [hx₀.2, hε, hεε]⟩
  have hmvt := (convex_Icc (-2 * r - ε₀) (2 * r + ε₀)).norm_image_sub_le_of_norm_hasDerivWithin_le
    hderiv hbound hx₀S hxS
  rw [Real.norm_eq_abs, Real.norm_eq_abs] at hmvt
  have h0 : |g x₀| ≤ m * X := hT (x₀, y) ⟨hx₀, hy⟩ s₁ s₂ s₃ y₁ y₂ y₃
  have hKX : 0 ≤ K * X := mul_nonneg hK hX
  have h3 : K * X * |x - x₀| ≤ K * X * ε := mul_le_mul_of_nonneg_left hxx₀ hKX
  have h4 := abs_sub_abs_le_abs_sub (g x) (g x₀)
  show |g x| ≤ (m + K * ε) * X
  nlinarith

end Slice

end CapFirstExit
