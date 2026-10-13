
/-! ### The transverse slices (L7, L8, L11, L18)

`ridge_capHyp` takes the transverse maximum (`hT`), its uniqueness (`hTs`), the curved side
(`hside`) and the ridge equation `g' = F` as inputs. SP derives them deterministically from L4
(each slice `y ↦ f (x, y)` is `δ`-strongly concave on `B̄(0, 2r)`), L5 (the transverse derivative
`w x` at `y = 0` has norm at most `ω`) and the ridge equation `∂_y f (x, h x) = 0`. This section
proves those derivations. -/

section Slice

open Filter Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **Strong concavity about a point with a derivative (the engine of L7, L8, L18).** If `φ` is
`δ`-strongly concave on `K`, `p ∈ K`, and `φ` has derivative `L` at `p`, then
`φ y ≤ φ p + L (y - p) - (δ/2)‖y - p‖²` for every `y ∈ K`. `p` need not be interior: only the
segment from `p` to `y`, which lies in `K`, is used. -/
theorem strongConcave_le_of_hasFDerivAt {K : Set E} {φ : E → ℝ} {δ : ℝ} {p : E}
    {L : E →L[ℝ] ℝ} (hφ : StrongConcaveOn K δ φ) (hp : p ∈ K) (hd : HasFDerivAt φ L p) {y : E}
    (hy : y ∈ K) : φ y ≤ φ p + L (y - p) - δ / 2 * ‖y - p‖ ^ 2 := by
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

/-- **L7 (the transverse maximum).** At a critical point `p` of a `δ`-strongly concave slice,
`φ y ≤ φ p - (δ/2)‖y - p‖²` on `K`. With `δ > 0` the maximizer is unique. -/
theorem slice_le {K : Set E} {φ : E → ℝ} {δ : ℝ} {p : E}
    (hφ : StrongConcaveOn K δ φ) (hp : p ∈ K) (hd : HasFDerivAt φ (0 : E →L[ℝ] ℝ) p)
    {y : E} (hy : y ∈ K) : φ y ≤ φ p - δ / 2 * ‖y - p‖ ^ 2 := by
  simpa using strongConcave_le_of_hasFDerivAt hφ hp hd hy

/-- **L8 (the ridge bound).** If `0, p ∈ K`, `p` is critical and `w` is the derivative at `0`,
then `δ ‖p‖ ≤ ‖w‖`, i.e. `‖h(x)‖ ≤ ‖w(x)‖/δ`. -/
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

/-- **L18 (the curved side).** On the sphere of radius `R`, the slice is strictly below `s`
once `φ p ≤ b`, `‖w‖ ≤ ω ≤ Rδ` and `2δ(b - s) < (Rδ - ω)²`. -/
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

/-- The derivative of a slice `y ↦ f (x, y)` is the joint derivative restricted to the
transverse directions. -/
theorem slice_hasFDerivAt {f : ℝ × E → ℝ} {x : ℝ} {y : E} {D : ℝ × E →L[ℝ] ℝ}
    (hf : HasFDerivAt f D (x, y)) :
    HasFDerivAt (fun y' => f (x, y')) (D.comp (ContinuousLinearMap.inr ℝ ℝ E)) y :=
  hf.comp y (hasFDerivAt_prodMk_right x y)

/-- **L11 (the ridge function's derivative).** If `f` has joint derivative `D` at `(x, h x)`,
`h` has derivative `h'` at `x`, and the transverse part of `D` vanishes (the ridge equation
`∂_y f = 0`), then `g = f (·, h ·)` has derivative `F x = D (1, 0) = ∂_x f (x, h x)`: the
`∂_y f · h'` term drops out. -/
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

/-- **(C1)–(C3) and (C2⁺) from the slices.** `ridge_capHyp` and `ridge_saddle_strict` with
`hT`, `hTs`, `hside` and `hM` derived (L7, L8, L18) from L4 (`hconc`), the critical points
`h x ∈ B̄(0, 2r)` (`hball`, `hcrit`; L7's existence part), L5 (`hw`, `hω`) and the curvature
condition `2δ(b - s) < (2rδ - ω)²`, which SP's constants satisfy (`sp_curved_side_constants`). -/
theorem ridge_capHyp_of_slices {f : ℝ × E → ℝ} {h : ℝ → E} {F : ℝ → ℝ} {w : ℝ → E →L[ℝ] ℝ}
    {r δ ω : ℝ} (hr : 0 < r) (hδ : 0 < δ)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hconc : ∀ x ∈ Icc (-2 * r) (r / 2),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hball : ∀ x ∈ Icc (-2 * r) (r / 2), h x ∈ Metric.closedBall (0 : E) (2 * r))
    (hcrit : ∀ x ∈ Icc (-2 * r) (r / 2),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x))
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

/-- **The joint form (L7, L11).** From a joint derivative `D x` of `f` along the ridge, a
differentiable `h` and the ridge equation `(D x) ∘ inr = 0`, the inputs `hh`, `hg` (with
`F x = D x (1, 0)`) and `hcrit` of `ridge_capHyp_of_slices` follow. -/
theorem ridge_inputs_of_joint {f : ℝ × E → ℝ} {h h' : ℝ → E} {D : ℝ → ℝ × E →L[ℝ] ℝ} {r : ℝ}
    (hr : 0 < r) (hD : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt f (D x) (x, h x))
    (hh' : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt h (h' x) x)
    (hridge : ∀ x ∈ Icc (-2 * r) (2 * r), (D x).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0) :
    ContinuousOn h (Icc (-r / 2) (2 * r)) ∧
      (∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (D x (1, 0)) x) ∧
      ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := by
  refine ⟨fun x hx => ?_, fun x hx => ridge_hasDerivAt (hD x hx) (hh' x hx) (hridge x hx),
    fun x hx => ?_⟩
  · exact (hh' x ⟨by linarith [hx.1], hx.2⟩).continuousAt.continuousWithinAt
  · have hx' : x ∈ Icc (-2 * r) (2 * r) := ⟨hx.1, by linarith [hx.2]⟩
    simpa [hridge x hx'] using slice_hasFDerivAt (hD x hx')

/-- **SP's constants satisfy the curvature condition (L4, L5, L18).** With `m ≥ 2`,
`δ > r m (8m - 5)` (L4, `k = 8m - 5`), `0 ≤ ω ≤ 2 m r²` (L5) and SP's gap `b - s = r³/6`:
`ω ≤ 2rδ` and `2δ · r³/6 < (2rδ - ω)²`. -/
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
