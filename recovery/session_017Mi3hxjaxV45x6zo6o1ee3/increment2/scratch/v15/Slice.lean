import CapFirstExit

open Set Filter Topology

namespace CapFirstExit

section Slice

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem strongConcave_le_of_hasFDerivAt {K : Set E} {φ : E → ℝ} {δ : ℝ} {p : E} {L : E →L[ℝ] ℝ}
    (hφ : StrongConcaveOn K δ φ) (hp : p ∈ K) (hd : HasFDerivAt φ L p) {y : E} (hy : y ∈ K) :
    φ y ≤ φ p + L (y - p) - δ / 2 * ‖y - p‖ ^ 2 := by
  set v := y - p with hv
  have h1 : HasDerivAt (fun t : ℝ => p + t • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add p
  have h2 : HasFDerivAt φ L (p + (0 : ℝ) • v) := by simpa using hd
  have hline : HasDerivAt (fun t : ℝ => φ (p + t • v)) (L v) 0 := h2.comp_hasDerivAt (0 : ℝ) h1
  have hslope := hline.tendsto_slope_zero_right
  have hbound : ∀ᶠ t in 𝓝[>] (0 : ℝ),
      φ y - φ p + (1 - t) * (δ / 2 * ‖v‖ ^ 2) ≤
        t⁻¹ • (φ (p + (0 + t) • v) - φ (p + (0 : ℝ) • v)) := by
    have hlt : ∀ᶠ t in 𝓝[>] (0 : ℝ), t < 1 :=
      nhdsWithin_le_nhds (Iio_mem_nhds one_pos)
    filter_upwards [self_mem_nhdsWithin, hlt] with t (ht0 : 0 < t) ht1
    have key := hφ.2 hp hy (by linarith : (0 : ℝ) ≤ 1 - t) ht0.le (by ring : 1 - t + t = 1)
    have hpt : (1 - t) • p + t • y = p + t • v := by
      simp only [hv, smul_sub, sub_smul, one_smul]; abel
    have hnorm : ‖p - y‖ = ‖v‖ := by rw [hv, norm_sub_rev]
    rw [hpt, hnorm, smul_eq_mul, smul_eq_mul] at key
    simp only [zero_smul, add_zero, zero_add, smul_eq_mul]
    rw [le_inv_mul_iff₀ ht0]
    nlinarith [key]
  have hlim : Tendsto (fun t : ℝ => φ y - φ p + (1 - t) * (δ / 2 * ‖v‖ ^ 2)) (𝓝[>] 0)
      (𝓝 (φ y - φ p + (1 - 0) * (δ / 2 * ‖v‖ ^ 2))) :=
    ((continuous_const.add ((continuous_const.sub continuous_id).mul continuous_const)).tendsto
      0).mono_left nhdsWithin_le_nhds
  have := le_of_tendsto_of_tendsto hlim hslope hbound
  simp only [sub_zero, one_mul] at this
  linarith

theorem slice_le {K : Set E} {φ : E → ℝ} {δ : ℝ} {p : E}
    (hφ : StrongConcaveOn K δ φ) (hp : p ∈ K) (hd : HasFDerivAt φ (0 : E →L[ℝ] ℝ) p)
    {y : E} (hy : y ∈ K) : φ y ≤ φ p - δ / 2 * ‖y - p‖ ^ 2 := by
  simpa using strongConcave_le_of_hasFDerivAt hφ hp hd hy

theorem slice_ridge_bound {K : Set E} {φ : E → ℝ} {δ : ℝ} {p : E} {w : E →L[ℝ] ℝ}
    (hφ : StrongConcaveOn K δ φ) (h0 : (0 : E) ∈ K) (hp : p ∈ K)
    (hd : HasFDerivAt φ (0 : E →L[ℝ] ℝ) p) (hw : HasFDerivAt φ w 0) : δ * ‖p‖ ≤ ‖w‖ := by
  have h1 := slice_le hφ hp hd h0
  have h2 := strongConcave_le_of_hasFDerivAt hφ h0 hw hp
  simp only [zero_sub, norm_neg, sub_zero] at h1 h2
  have h3 : w p ≤ ‖w‖ * ‖p‖ :=
    (le_abs_self _).trans (by simpa [Real.norm_eq_abs] using w.le_opNorm p)
  have h4 : δ * ‖p‖ * ‖p‖ ≤ ‖w‖ * ‖p‖ := by nlinarith
  rcases (norm_nonneg p).eq_or_lt with h | h
  · rw [← h, mul_zero]; exact norm_nonneg w
  · exact le_of_mul_le_mul_right h4 h

theorem slice_curved {φ : E → ℝ} {δ ω R b s : ℝ} {p : E} {w : E →L[ℝ] ℝ}
    (hφ : StrongConcaveOn (Metric.closedBall (0 : E) R) δ φ) (hδ : 0 < δ)
    (hp : p ∈ Metric.closedBall (0 : E) R) (hd : HasFDerivAt φ (0 : E →L[ℝ] ℝ) p)
    (hw : HasFDerivAt φ w 0) (hω : ‖w‖ ≤ ω) (hωR : ω ≤ R * δ) (hb : φ p ≤ b)
    (hcurv : 2 * δ * (b - s) < (R * δ - ω) ^ 2) {y : E} (hy : y ∈ Metric.sphere (0 : E) R) :
    φ y < s := by
  have hyn : ‖y‖ = R := mem_sphere_zero_iff_norm.mp hy
  have hR : 0 ≤ R := hyn ▸ norm_nonneg y
  have h0 : (0 : E) ∈ Metric.closedBall (0 : E) R := Metric.mem_closedBall_self hR
  have hL8 := slice_ridge_bound hφ h0 hp hd hw
  have hL7 := slice_le hφ hp hd (Metric.sphere_subset_closedBall hy)
  have htri : R - ‖p‖ ≤ ‖y - p‖ := by have := norm_sub_norm_le y p; linarith
  have hρ : R * δ - ω ≤ δ * ‖y - p‖ := by nlinarith
  have hsq : (R * δ - ω) ^ 2 ≤ (δ * ‖y - p‖) ^ 2 := pow_le_pow_left₀ (by linarith) hρ 2
  have h5 : δ * (2 * (b - s)) < δ * (δ * ‖y - p‖ ^ 2) := by nlinarith
  have h6 : 2 * (b - s) < δ * ‖y - p‖ ^ 2 := lt_of_mul_lt_mul_left h5 hδ.le
  linarith

theorem slice_hasFDerivAt {f : ℝ × E → ℝ} {x : ℝ} {y : E} {D : ℝ × E →L[ℝ] ℝ}
    (hf : HasFDerivAt f D (x, y)) :
    HasFDerivAt (fun y' => f (x, y')) (D.comp (ContinuousLinearMap.inr ℝ ℝ E)) y :=
  hf.comp y (hasFDerivAt_prodMk_right x y)

theorem ridge_hasDerivAt {f : ℝ × E → ℝ} {h : ℝ → E} {x : ℝ} {D : ℝ × E →L[ℝ] ℝ} {h' : E}
    (hf : HasFDerivAt f D (x, h x)) (hh : HasDerivAt h h' x)
    (hy : D.comp (ContinuousLinearMap.inr ℝ ℝ E) = 0) :
    HasDerivAt (fun t => f (t, h t)) (D (1, 0)) x := by
  have hp : HasDerivAt (fun t => (t, h t)) ((1 : ℝ), h') x := (hasDerivAt_id x).prodMk hh
  have hc := hf.comp_hasDerivAt x hp
  have h0 : D (0, h') = 0 := by
    simpa using congrArg (fun L : E →L[ℝ] ℝ => L h') hy
  have hD : D (1, h') = D (1, 0) := by
    calc D (1, h') = D ((1, 0) + (0, h')) := by simp
      _ = D (1, 0) + D (0, h') := map_add _ _ _
      _ = D (1, 0) := by rw [h0, add_zero]
  rw [← hD]
  exact hc

theorem ridge_capHyp_of_slices {f : ℝ × E → ℝ} {h : ℝ → E} {F : ℝ → ℝ} {w : ℝ → E →L[ℝ] ℝ}
    {r δ ω : ℝ} (hr : 0 < r) (hδ : 0 < δ)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hconc : ∀ x ∈ Icc (-2 * r) (r / 2),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hball : ∀ x ∈ Icc (-2 * r) (r / 2), h x ∈ Metric.closedBall (0 : E) (2 * r))
    (hcrit : ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x))
    (hw : ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (w x) (0 : E))
    (hω : ∀ x ∈ Icc (-2 * r) (r / 2), ‖w x‖ ≤ ω) (hωr : ω ≤ 2 * r * δ)
    (hcurv : 2 * δ * (f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2))) < (2 * r * δ - ω) ^ 2)
    (hgap : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) < 9 * r ^ 3 / 32) :
    CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, h (-r / 2))
        (2 * r, h (2 * r)) (f (-r / 2, h (-r / 2))) (f (r / 2, h (r / 2))) ∧
      ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
        x ≠ (r / 2, h (r / 2)) → f x < f (r / 2, h (r / 2)) := by
  obtain ⟨hceil, -, -, -⟩ := ridge_profile hr hg hFa hFc hconv
  have hT : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      f p ≤ f (p.1, h p.1) := by
    rintro ⟨x, y⟩ ⟨hx, hy⟩
    have h1 := slice_le (hconc x hx) (hball x hx) (hcrit x hx) hy
    have h2 : 0 ≤ δ / 2 * ‖y - h x‖ ^ 2 := by positivity
    simp only at h1 ⊢
    linarith
  have hTs : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), p.2 ≠ h p.1 →
      f p < f (p.1, h p.1) := by
    rintro ⟨x, y⟩ ⟨hx, hy⟩ hne
    have h1 := slice_le (hconc x hx) (hball x hx) (hcrit x hx) hy
    have h2 : 0 < ‖y - h x‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hne)
    have h3 : 0 < δ / 2 * ‖y - h x‖ ^ 2 := mul_pos (by linarith) (pow_pos h2 2)
    simp only at h1 ⊢
    linarith
  have hside : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.sphere (0 : E) (2 * r),
      f p < f (r / 2, h (r / 2)) := by
    rintro ⟨x, y⟩ ⟨hx, hy⟩
    exact slice_curved (hconc x hx) hδ (hball x hx) (hcrit x hx) (hw x hx) (hω x hx) hωr
      (hceil x hx) hcurv hy
  have hM : ‖h (-r / 2)‖ ≤ 2 * r := by
    simpa using hball (-r / 2) ⟨by linarith, by linarith⟩
  exact ⟨ridge_capHyp hr hh hg hFa hFc hconv hT (fun p hp => (hside p hp).le) hM hgap,
    ridge_saddle_strict hr hg hFa hFc hconv hT hTs hside hgap⟩

theorem sp_curved_side_constants {r m δ ω : ℝ} (hr : 0 < r) (hm : 2 ≤ m)
    (hδ : r * m * (8 * m - 5) < δ) (hω0 : 0 ≤ ω) (hω : ω ≤ 2 * m * r ^ 2) :
    ω ≤ 2 * r * δ ∧ 2 * δ * (r ^ 3 / 6) < (2 * r * δ - ω) ^ 2 := by
  have hk : 11 ≤ 8 * m - 5 := by linarith
  have hrm : 0 < r * m := by positivity
  have h11 : 11 * (r * m) < δ := by nlinarith
  have h22 : 22 * r < δ := by nlinarith
  have hωδ : 11 * ω < 2 * r * δ := by nlinarith
  have hpos : 20 * r * δ < 11 * (2 * r * δ - ω) := by nlinarith
  refine ⟨by linarith, ?_⟩
  have hsq : (20 * r * δ) ^ 2 < (11 * (2 * r * δ - ω)) ^ 2 :=
    pow_lt_pow_left₀ hpos (by have hδ0 : 0 < δ := by linarith
                              positivity) two_ne_zero
  nlinarith [hsq]

end Slice

section SliceToy

theorem ridgeToy_slices :
    CapHyp ridgeToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (ridgeToyF (-1 / 2, 0)) (ridgeToyF (1 / 2, 0)) ∧
      ∀ x ∈ frontier (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)),
        x ≠ (1 / 2, 0) → ridgeToyF x < ridgeToyF (1 / 2, 0) := by
  have hslice : ∀ x : ℝ, HasFDerivAt (fun y : ℝ => ridgeToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    rw [hasFDerivAt_iff_hasDerivAt]
    have h1 := (hasDerivAt_pow 2 (0 : ℝ)).const_sub (x ^ 3 / 3 - x / 4)
    convert h1 using 1
    · funext y
      simp only [ridgeToyF]
    · simp
  refine ridge_capHyp_of_slices (E := ℝ) (f := ridgeToyF) (h := fun _ => (0 : ℝ))
    (F := fun x => x ^ 2 - 1 / 4) (w := fun _ => 0) (r := 1) (δ := 2) (ω := 0) one_pos two_pos
    continuousOn_const ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro x _
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext y
      simp only [ridgeToyF, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  · norm_num
  · norm_num
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · intro x _
    rw [strongConcaveOn_iff_convex]
    refine (concaveOn_const (x ^ 3 / 3 - x / 4) (convex_closedBall (0 : ℝ) (2 * 1))).congr ?_
    intro y _
    simp only [ridgeToyF, Real.norm_eq_abs, sq_abs]
    ring
  · intro x _
    simp
  · intro x _
    exact hslice x
  · intro x _
    exact hslice x
  · intro x _
    simp
  · norm_num
  · simp only [ridgeToyF]
    norm_num
  · simp only [ridgeToyF]
    norm_num

end SliceToy

end CapFirstExit
