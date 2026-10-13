import Mathlib

open Set Filter Topology

namespace Test

theorem strongConcave_le_of_hasFDerivAt {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {K : Set E} {φ : E → ℝ} {δ : ℝ} {p : E} {L : E →L[ℝ] ℝ}
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

end Test
