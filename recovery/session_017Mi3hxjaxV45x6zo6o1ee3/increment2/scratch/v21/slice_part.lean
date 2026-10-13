/-! ### The derivative bounds from the block bound, and SP's bounds on `D` (v2.1)

Up to v2.0 the capstones took six derivative bounds besides the block bound `hT`:
- `hHxb` and `hHrb`, the inputs of L3;
- `hwcv` and `hwcc`, the envelope inputs of L5–L6;
- `hG0cv` and `hG0cc`, the envelope inputs of L1–L2.

All six follow from `hT` and from `|f_xxx| ≤ m` on the pin segment, and `ridge_capHyp_C5` derives
them. Up to v2.0 the bounds `m` were also assumed on a collar `[−2r − ε, 2r + ε]` of SP's `D`.
`ridge_capHyp_D` assumes them on `D` only, through the continuity step `block_bound_collar` (I3). -/

/-- **L3's `x`-input from the block bound.** Let `y ↦ ∂_y f (t, y)` have derivative `H (t, 0)` at `0`
and `Df` have derivative `D2f (t, 0)` at `(t, 0)` for every `t ∈ J`, where `J` is uniquely
differentiable at `x ∈ J`. If `t ↦ H (t, 0)` has derivative `Hx'` at `x`, then
`Hx' v w = T3 (x, 0) [(0, v), (0, w), (1, 0)]`, so `‖Hx'‖ ≤ m` when that block is at most
`m ‖v‖ ‖w‖`. This derives v1.6's hypothesis `hHxb`. -/
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

/-- **L3's radial input from the block bound.** Along the radius `τ ↦ (x, τ y)`, `τ ∈ [0, 1]`, if
`τ ↦ H (x, τ y)` has derivative `Hr'` at `s`, then `Hr' v w = T3 (x, s y) [(0, v), (0, w), (0, y)]`,
so `‖Hr'‖ ≤ m ‖y‖` when that block is at most `m ‖v‖ ‖w‖ ‖y‖`. The identification uses derivatives
within `[0, 1]`, so the endpoint `τ = 1` may lie on the sphere `‖y‖ = 2r`. This derives v1.6's
hypothesis `hHrb`. -/
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

/-- **The envelope inputs of L1 and L5–L6 from a second-derivative bound.** If the second derivative
`T3 (t, 0) [(1, 0), u, (1, 0)]` of `t ↦ Df (t, 0) u` has absolute value at most `B` on `[a, c]`, then
`Df (·, 0) u + (B/2) t²` is convex and `Df (·, 0) u − (B/2) t²` is concave there. With `u = (0, v)`
and `B = m‖v‖` (from the block bound) this gives `hwcv` and `hwcc`. With `u = (1, 0)` and `B = m`
(SP: `|f_xxx| ≤ M₃`) it gives `hG0cv` and `hG0cc`. -/
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

/-- The block weight `∏ (|sᵢ| + ‖yᵢ‖) − ∏ |sᵢ|` is nonnegative. -/
theorem block_nonneg {F : Type*} [SeminormedAddCommGroup F] (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : F) :
    0 ≤ (|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃| := by
  have a1 : |s₁| ≤ |s₁| + ‖y₁‖ := le_add_of_nonneg_right (norm_nonneg _)
  have a2 : |s₂| ≤ |s₂| + ‖y₂‖ := le_add_of_nonneg_right (norm_nonneg _)
  have a3 : |s₃| ≤ |s₃| + ‖y₃‖ := le_add_of_nonneg_right (norm_nonneg _)
  have := mul_le_mul (mul_le_mul a1 a2 (abs_nonneg _) (by positivity)) a3 (abs_nonneg _)
    (by positivity)
  linarith

/-- **The continuity step of I3: the block bound from `D` to a collar.** Assume the block bound holds
with `m` on SP's `D = [−2r, 2r] × B̄(0, 2r)`. Assume also that on the collar
`[−2r − ε₀, 2r + ε₀] × B̄(0, 2r)` the `x`-derivative `T4` of `T3` exists and satisfies the same block
bound with some `K ≥ 0` (SP: `f ∈ C⁴` near `D`, so `‖D⁴ f‖` is bounded on a compact collar). Then the
block bound holds with `m + Kε` on the collar of width `ε ≤ ε₀`. Each collar point `(x, y)` is
within `ε` of a point `(x₀, y) ∈ D`, and the mean value inequality along `x` gives the rest. -/
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

/-- **The cap from the block bound alone.** `ridge_capHyp_C4` with its six derivative-bound
hypotheses derived. `hHxb`, `hHrb`, `hwcv` and `hwcc` come from the block bound `hT`
(`hessian_x_bound`, `hessian_r_bound`, `axis_convexConcave`). `hG0cv` and `hG0cc` come from
`|f_xxx| ≤ m` on the pin segment (`hTe`, SP: `|f_xxx| ≤ M₃`). The conclusion is unchanged. -/
theorem ridge_capHyp_C5 [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {Φ4 : ℝ × E → ℝ × E →L[ℝ] ℝ} {n : ℝ}
    {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hDf : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hD2 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt Df (D2f p) p)
    (hD3 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 p a b) p)
    (hT : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hTe : ∀ x ∈ Icc (-r / 2) (r / 2), |T3 (x, 0) (1, 0) (1, 0) (1, 0)| ≤ m)
    (hD4 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun q => T3 q (1, 0) (1, 0) (1, 0)) (Φ4 p) p)
    (hn4x : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), |Φ4 p (1, 0)| ≤ n)
    (hn4y : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ‖(Φ4 p).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ n)
    (hH2 : r * n ≤ 1 / 20)
    (hGs : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  have hm0 : 0 ≤ m := (abs_nonneg _).trans (hTe 0 ⟨by linarith, by linarith⟩)
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  have hax : ∀ t ∈ Icc (-2 * r - ε) (2 * r + ε),
      (t, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun t ht => ⟨ht, h0Y⟩
  have hpin : ∀ t ∈ Icc (-r / 2) (r / 2),
      (t, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun t ht => hax t ⟨by linarith [ht.1], by linarith [ht.2]⟩
  -- `∂_x D_y² f`
  have hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m := by
    intro x hx
    refine hessian_x_bound (J := Icc (-2 * r - ε) (2 * r + ε))
      (uniqueDiffOn_Icc (by linarith) x hx) hx hm0 (fun t ht => hH _ (hax t ht))
      (fun t ht => hD2 _ (hax t ht)) (hD3 _ (hax x hx)) (fun v w => ?_) (hHx x hx)
    have := hT _ (hax x hx) 0 0 1 v w 0
    simpa using this
  -- `∂_ρ D_y² f`
  have hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖ := by
    rintro ⟨x, y⟩ hp s hs
    have hyb : ‖y‖ ≤ 2 * r := by simpa [Metric.mem_closedBall, dist_zero_right] using hp.2
    have hseg : ∀ τ ∈ Icc (0 : ℝ) 1,
        (x, τ • y) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) := by
      intro τ hτ
      refine ⟨hp.1, ?_⟩
      rw [Metric.mem_closedBall, dist_zero_right, norm_smul, Real.norm_eq_abs, abs_of_nonneg hτ.1]
      nlinarith [hτ.2, norm_nonneg y]
    refine hessian_r_bound hm0 hs (fun τ hτ => hH _ (hseg τ hτ)) (fun τ hτ => hD2 _ (hseg τ hτ))
      (hD3 _ (hseg s hs)) (fun v w => ?_) (hHr _ hp s hs)
    have := hT _ (hseg s hs) 0 0 0 v w y
    simpa using this
  -- `w = ∂_y f (·, 0)` and `G0 = ∂_x f (·, 0)`
  have hw : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
        (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2) ∧
      ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε)) (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2) :=
    fun v => axis_convexConcave (fun t ht => hD2 _ (hax t ht))
      (fun t ht => hD3 _ (hax t ht) (1, 0) (0, v))
      (fun t ht => by have := hT _ (hax t ht) 1 0 1 0 v 0; simpa using this)
  have hG0 := axis_convexConcave (u := ((1 : ℝ), (0 : E))) (fun t ht => hD2 _ (hpin t ht))
    (fun t ht => hD3 _ (hpin t ht) (1, 0) (1, 0)) hTe
  exact ridge_capHyp_C4 hr hε hεr hDf hD2 hD3 hT hD4 hn4x hn4y hH2 hGs hH hHx hHxb hHr hHrb hlam
    hH1 hM hS hG0.1 hG0.2 (fun v => (hw v).1) (fun v => (hw v).2) hnorm

/-- **The cap from SP's bounds on `D` (the continuity step of I3).** The quantitative hypotheses are
assumed on SP's `D` only:
- the block bound `m` and `|f_xxx| ≤ m` on the pin segment;
- SP's `M₄` blocks `n`;
- (H1) and (H2).

On a collar of some width `ε₀ > 0`, only qualitative data are assumed: the derivatives, the strict
derivative of `∂_y f`, and some bound `K ≥ 0` on the `x`-derivative of `T3` in block form (SP:
`f ∈ C⁴` near `D`). Because (H1) is strict, some `ε ∈ (0, ε₀]` keeps `8r(m + Kε)² < λ`.
`ridge_capHyp_C5` then applies with `m + Kε` (`block_bound_collar`). -/
theorem ridge_capHyp_D [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {T4 : ℝ × E → ℝ × E → ℝ × E → ℝ × E → ℝ}
    {Φ4 : ℝ × E → ℝ × E →L[ℝ] ℝ} {n : ℝ}
    {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m K lam ε₀ : ℝ} (hr : 0 < r) (hε₀ : 0 < ε₀)
    (hK : 0 ≤ K)
    (hDf : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hD2 : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt Df (D2f p) p)
    (hD3 : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ a b : ℝ × E,
      HasFDerivAt (fun q => D2f q a b) (T3 p a b) p)
    (hD4x : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ a b c : ℝ × E, HasDerivAt (fun t => T3 (t, p.2) a b c) (T4 p a b c) p.1)
    (hT4 : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T4 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T4 p (1, 0) (1, 0) (1, 0)| ≤
        K * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hT : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hTe : ∀ x ∈ Icc (-r / 2) (r / 2), |T3 (x, 0) (1, 0) (1, 0) (1, 0)| ≤ m)
    (hD4 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun q => T3 q (1, 0) (1, 0) (1, 0)) (Φ4 p) p)
    (hn4x : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), |Φ4 p (1, 0)| ≤ n)
    (hn4y : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ‖(Φ4 p).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ n)
    (hH2 : r * n ≤ 1 / 20)
    (hGs : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε₀) (2 * r + ε₀), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHr : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    ∃ ε, 0 < ε ∧ ε ≤ ε₀ ∧
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  have hev : ∀ᶠ ε in 𝓝[>] (0 : ℝ), 0 < ε ∧ ε ≤ ε₀ ∧ ε ≤ r / 2 ∧ 8 * r * (m + K * ε) ^ 2 < lam := by
    have hc : ContinuousAt (fun ε : ℝ => 8 * r * (m + K * ε) ^ 2) 0 := by fun_prop
    have h1 : ∀ᶠ ε in 𝓝 (0 : ℝ), 8 * r * (m + K * ε) ^ 2 < lam :=
      hc.eventually_lt continuousAt_const (by simpa using hH1)
    have h2 : ∀ᶠ ε in 𝓝 (0 : ℝ), ε < ε₀ := eventually_lt_nhds hε₀
    have h3 : ∀ᶠ ε in 𝓝 (0 : ℝ), ε < r / 2 := eventually_lt_nhds (by linarith)
    filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds h1, nhdsWithin_le_nhds h2,
      nhdsWithin_le_nhds h3] with ε hε h1 h2 h3
    exact ⟨hε, h2.le, h3.le, h1⟩
  obtain ⟨ε, hε, hεε, hεr, hH1'⟩ := hev.exists
  have hsub : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun p hp => ⟨⟨by linarith [hp.1.1], by linarith [hp.1.2]⟩, hp.2⟩
  have hsubI : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), x ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) :=
    fun x hx => ⟨by linarith [hx.1], by linarith [hx.2]⟩
  have hmm : m ≤ m + K * ε := le_add_of_nonneg_right (mul_nonneg hK hε.le)
  exact ⟨ε, hε, hεε, ridge_capHyp_C5 (m := m + K * ε) hr hε hεr (fun p hp => hDf p (hsub p hp))
    (fun p hp => hD2 p (hsub p hp)) (fun p hp => hD3 p (hsub p hp))
    (block_bound_collar hr hK hε.le hεε hT hD4x hT4) (fun x hx => (hTe x hx).trans hmm)
    hD4 hn4x hn4y hH2 (fun p hp => hGs p (hsub p hp)) (fun p hp => hH p (hsub p hp))
    (fun x hx => hHx x (hsubI x hx)) (fun p hp => hHr p (hsub p hp)) hlam hH1' hM hS hnorm⟩
