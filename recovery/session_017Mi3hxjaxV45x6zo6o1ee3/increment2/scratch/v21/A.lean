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

end Slice

end CapFirstExit
