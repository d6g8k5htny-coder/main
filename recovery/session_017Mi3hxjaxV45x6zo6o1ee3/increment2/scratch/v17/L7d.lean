import CapFirstExit

open Set Filter Topology

namespace CapFirstExit

section L7

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem slice_critical_unique {K : Set E} {φ : E → ℝ} {δ : ℝ} (hφ : StrongConcaveOn K δ φ)
    (hδ : 0 < δ) {p q : E} (hp : p ∈ K) (hq : q ∈ K) (hdp : HasFDerivAt φ (0 : E →L[ℝ] ℝ) p)
    (hdq : HasFDerivAt φ (0 : E →L[ℝ] ℝ) q) : p = q := by
  have h1 := slice_le hφ hp hdp hq
  have h2 := slice_le hφ hq hdq hp
  rw [norm_sub_rev] at h2
  have h3 : δ * ‖q - p‖ ^ 2 ≤ 0 := by linarith
  have h4 : ‖q - p‖ ^ 2 ≤ 0 := by
    have : δ * ‖q - p‖ ^ 2 ≤ δ * 0 := by rw [mul_zero]; exact h3
    exact le_of_mul_le_mul_left this hδ
  have h5 : ‖q - p‖ = 0 := by nlinarith [norm_nonneg (q - p)]
  exact (sub_eq_zero.mp (norm_eq_zero.mp h5)).symm

theorem slice_critical_exists [ProperSpace E] {φ : E → ℝ} {Dφ : E → E →L[ℝ] ℝ} {δ R : ℝ}
    (hR : 0 < R) (hφ : StrongConcaveOn (Metric.closedBall (0 : E) R) δ φ)
    (hD : ∀ y ∈ Metric.closedBall (0 : E) R, HasFDerivAt φ (Dφ y) y) (hw : ‖Dφ 0‖ < δ * R) :
    ∃ p ∈ Metric.ball (0 : E) R, HasFDerivAt φ (0 : E →L[ℝ] ℝ) p := by
  have h0 : (0 : E) ∈ Metric.closedBall (0 : E) R := Metric.mem_closedBall_self hR.le
  have hcont : ContinuousOn φ (Metric.closedBall (0 : E) R) :=
    fun y hy => (hD y hy).continuousAt.continuousWithinAt
  obtain ⟨p, hp, hmax⟩ := (isCompact_closedBall (0 : E) R).exists_isMaxOn ⟨0, h0⟩ hcont
  have hpR : ‖p‖ ≤ R := by simpa using hp
  have hlt : ‖p‖ < R := by
    by_contra hc
    have hpn : ‖p‖ = R := le_antisymm hpR (not_lt.mp hc)
    have hseg : segment ℝ p (p + -p) ⊆ Metric.closedBall (0 : E) R := by
      rw [add_neg_cancel]
      exact (convex_closedBall (0 : E) R).segment_subset hp h0
    have hcone := mem_posTangentConeAt_of_segment_subset hseg
    have hnp := hmax.isLocalMaxOn.hasFDerivWithinAt_nonpos (hD p hp).hasFDerivWithinAt hcone
    have e1 := strongConcave_le_of_hasFDerivAt hφ hp (hD p hp) h0
    have e2 := strongConcave_le_of_hasFDerivAt hφ h0 (hD 0 h0) hp
    simp only [zero_sub, sub_zero, norm_neg, map_neg] at e1 e2 hnp
    have hb : Dφ 0 p ≤ ‖Dφ 0‖ * ‖p‖ :=
      (le_abs_self _).trans (by simpa [Real.norm_eq_abs] using (Dφ 0).le_opNorm p)
    rw [hpn] at e1 e2 hb
    have : ‖Dφ 0‖ * R < δ * R * R := by nlinarith
    nlinarith
  have hpb : p ∈ Metric.ball (0 : E) R := by simpa using hlt
  have hloc : IsLocalMax φ p :=
    hmax.isLocalMax (Metric.closedBall_mem_nhds_of_mem hpb)
  refine ⟨p, hpb, ?_⟩
  have hz := hloc.hasFDerivAt_eq_zero (hD p hp)
  rw [← hz]
  exact hD p hp

theorem isInvertible_of_negDef [FiniteDimensional ℝ E] {A : E →L[ℝ] E →L[ℝ] ℝ} {δ : ℝ}
    (hδ : 0 < δ) (hA : ∀ v : E, A v v ≤ -δ * ‖v‖ ^ 2) : A.IsInvertible := by
  have hinj : Function.Injective A := by
    intro v w hvw
    have h1 : A (v - w) = 0 := by rw [map_sub, hvw, sub_self]
    have h2 := hA (v - w)
    rw [h1, zero_apply] at h2
    have h3 : ‖v - w‖ ^ 2 ≤ 0 := by nlinarith [sq_nonneg ‖v - w‖]
    have h4 : ‖v - w‖ = 0 := by nlinarith [norm_nonneg (v - w)]
    exact sub_eq_zero.mp (norm_eq_zero.mp h4)
  have hrank : Module.finrank ℝ E = Module.finrank ℝ (E →L[ℝ] ℝ) := by
    rw [← (LinearMap.toContinuousLinearMap : (E →ₗ[ℝ] ℝ) ≃ₗ[ℝ] E →L[ℝ] ℝ).finrank_eq]
    exact (Subspace.dual_finrank_eq).symm
  have hsurj : Function.Surjective A :=
    (LinearMap.injective_iff_surjective_of_finrank_eq_finrank hrank (f := (A : E →ₗ[ℝ] E →L[ℝ] ℝ))).mp hinj
  refine ⟨(LinearEquiv.ofBijective (A : E →ₗ[ℝ] E →L[ℝ] ℝ) ⟨hinj, hsurj⟩).toContinuousLinearEquiv, ?_⟩
  ext v w
  rfl

theorem ridge_hasStrictFDerivAt [CompleteSpace E] {f : ℝ × E → ℝ} {G : ℝ × E → E →L[ℝ] ℝ}
    {DG : ℝ × E →L[ℝ] E →L[ℝ] ℝ} {h : ℝ → E} {K : Set E} {x0 : ℝ}
    (hGs : HasStrictFDerivAt G DG (x0, h x0))
    (hinv : (DG ∘L ContinuousLinearMap.inr ℝ ℝ E).IsInvertible)
    (hG : ∀ᶠ v in 𝓝 (x0, h x0), HasFDerivAt (fun y => f (v.1, y)) (G v) v.2)
    (hcrit0 : HasFDerivAt (fun y => f (x0, y)) (0 : E →L[ℝ] ℝ) (h x0)) (hK : K ∈ 𝓝 (h x0))
    (huniq : ∀ᶠ x in 𝓝 x0, ∀ y ∈ K, HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) y → y = h x) :
    HasStrictFDerivAt h (-(DG ∘L ContinuousLinearMap.inr ℝ ℝ E).inverse ∘L
      (DG ∘L ContinuousLinearMap.inl ℝ ℝ E)) x0 := by
  have hGu : G (x0, h x0) = 0 := (hG.self_of_nhds).unique hcrit0
  have hψ := hGs.hasStrictFDerivAt_implicitFunctionOfProdDomain hinv
  have hev := hGs.eventually_apply_implicitFunctionOfProdDomain hinv
  have htend := hGs.tendsto_implicitFunctionOfProdDomain hinv
  set ψ := hGs.implicitFunctionOfProdDomain hinv with hψdef
  have hpair : Tendsto (fun x => (x, ψ x)) (𝓝 x0) (𝓝 (x0, h x0)) :=
    (tendsto_id.prodMk_nhds htend)
  have hGψ := hpair.eventually hG
  have hKψ := htend.eventually hK
  have heq : ψ =ᶠ[𝓝 x0] h := by
    filter_upwards [hev, hGψ, hKψ, huniq] with x h1 h2 h3 h4
    rw [h1, hGu] at h2
    exact h4 (ψ x) h3 h2
  exact hψ.congr_of_eventuallyEq heq


theorem ridge_differentiable [FiniteDimensional ℝ E] {f : ℝ × E → ℝ} {G : ℝ × E → E →L[ℝ] ℝ}
    {DG : ℝ → ℝ × E →L[ℝ] E →L[ℝ] ℝ} {h : ℝ → E} {U : Set ℝ} {r δ : ℝ} (hδ : 0 < δ)
    (hU : IsOpen U) (hIU : Icc (-2 * r) (2 * r) ⊆ U)
    (hconc : ∀ x ∈ U,
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hball : ∀ x ∈ U, h x ∈ Metric.closedBall (0 : E) (2 * r))
    (hcrit : ∀ x ∈ U, HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x))
    (hint : ∀ x ∈ Icc (-2 * r) (2 * r), h x ∈ Metric.ball (0 : E) (2 * r))
    (hG : ∀ x ∈ Icc (-2 * r) (2 * r),
      ∀ᶠ v in 𝓝 (x, h x), HasFDerivAt (fun y => f (v.1, y)) (G v) v.2)
    (hGs : ∀ x ∈ Icc (-2 * r) (2 * r), HasStrictFDerivAt G (DG x) (x, h x))
    (hneg : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E,
      (DG x ∘L ContinuousLinearMap.inr ℝ ℝ E) v v ≤ -δ * ‖v‖ ^ 2) :
    ∀ x ∈ Icc (-2 * r) (2 * r), HasStrictFDerivAt h
      (-(DG x ∘L ContinuousLinearMap.inr ℝ ℝ E).inverse ∘L (DG x ∘L ContinuousLinearMap.inl ℝ ℝ E)) x := by
  intro x hx
  have hK : Metric.closedBall (0 : E) (2 * r) ∈ 𝓝 (h x) :=
    Metric.closedBall_mem_nhds_of_mem (hint x hx)
  have huniq : ∀ᶠ x' in 𝓝 x, ∀ y ∈ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (x', y)) (0 : E →L[ℝ] ℝ) y → y = h x' := by
    filter_upwards [hU.mem_nhds (hIU hx)] with x' hx' y hy hdy
    exact slice_critical_unique (hconc x' hx') hδ hy (hball x' hx') hdy (hcrit x' hx')
  exact ridge_hasStrictFDerivAt (hGs x hx) (isInvertible_of_negDef hδ (hneg x hx)) (hG x hx)
    (hcrit x (hIU hx)) hK huniq


theorem ridge_exists [ProperSpace E] {f : ℝ × E → ℝ} {Dφ : ℝ → E → E →L[ℝ] ℝ} {U : Set ℝ}
    {r δ : ℝ} (hr : 0 < r)
    (hconc : ∀ x ∈ U,
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hD : ∀ x ∈ U, ∀ y ∈ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (x, y)) (Dφ x y) y)
    (hw : ∀ x ∈ U, ‖Dφ x 0‖ < δ * (2 * r)) :
    ∃ h : ℝ → E, ∀ x ∈ U, h x ∈ Metric.ball (0 : E) (2 * r) ∧
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := by
  have hex : ∀ x : ℝ, ∃ p : E, x ∈ U → p ∈ Metric.ball (0 : E) (2 * r) ∧
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) p := by
    intro x
    by_cases hx : x ∈ U
    · obtain ⟨p, hp, hdp⟩ := slice_critical_exists (by positivity) (hconc x hx) (hD x hx) (hw x hx)
      exact ⟨p, fun _ => ⟨hp, hdp⟩⟩
    · exact ⟨0, fun h => absurd h hx⟩
  choose h hh using hex
  exact ⟨h, fun x hx => hh x hx⟩

end L7

section L7Toy

theorem h1Toy_ridge_differentiable :
    ∀ x ∈ Icc (-2 * (1 : ℝ)) (2 * 1),
      ∃ L : ℝ →L[ℝ] ℝ, HasStrictFDerivAt (fun _ : ℝ => (0 : ℝ)) L x := by
  have hslope : ∀ x y : ℝ, HasFDerivAt (fun y : ℝ => h1ToyF (x, y))
      (ContinuousLinearMap.mul ℝ ℝ (-40 * y)) y := by
    intro x y
    rw [hasFDerivAt_iff_hasDerivAt]
    have h1 := ((hasDerivAt_pow 2 y).const_mul 20).const_sub (x ^ 3 / 3 - x / 4)
    convert h1 using 1
    · funext t
      simp only [h1ToyF]
    · simp
      ring
  have hGs : ∀ p : ℝ × ℝ, HasStrictFDerivAt (fun p : ℝ × ℝ => ContinuousLinearMap.mul ℝ ℝ (-40 * p.2))
      (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    convert (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L
      ContinuousLinearMap.snd ℝ ℝ ℝ).hasStrictFDerivAt (x := p) using 1
    funext q
    ext
    simp
  intro x hx
  exact ⟨_, ridge_differentiable (E := ℝ) (f := h1ToyF)
    (G := fun p => ContinuousLinearMap.mul ℝ ℝ (-40 * p.2))
    (DG := fun _ => ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ)
    (h := fun _ => (0 : ℝ)) (U := Ioo (-3) 3) (r := 1) (δ := 40) (by norm_num) isOpen_Ioo
    (fun t ht => ⟨by linarith [ht.1], by linarith [ht.2]⟩)
    (fun t _ => by
      rw [strongConcaveOn_iff_convex]
      refine (concaveOn_const (t ^ 3 / 3 - t / 4) (convex_closedBall (0 : ℝ) (2 * 1))).congr ?_
      intro y _
      simp only [h1ToyF, Real.norm_eq_abs, sq_abs]
      ring)
    (fun t _ => by simp) (fun t _ => by simpa using hslope t 0) (fun t _ => by simp)
    (fun t _ => Eventually.of_forall (fun v => hslope v.1 v.2)) (fun t _ => hGs _)
    (fun t _ v => by
      simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.inr_apply,
        ContinuousLinearMap.coe_snd', smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul,
        Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v]) x hx⟩

end L7Toy

end CapFirstExit
