import CapFirstExit

open Set Filter Topology

namespace CapFirstExit

section L5

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Adding the affine function `(m/2)((t - a)(t - c) - t²)` preserves convexity. -/
theorem convexOn_add_node {u : ℝ → ℝ} {D : Set ℝ} {a c m : ℝ}
    (hcv : ConvexOn ℝ D (fun t => u t + m / 2 * t ^ 2)) :
    ConvexOn ℝ D (fun t => u t + m / 2 * ((t - a) * (t - c))) := by
  refine ⟨hcv.1, fun x hx y hy p q hp hq hpq => ?_⟩
  have h := hcv.2 hx hy hp hq hpq
  simp only [smul_eq_mul] at h ⊢
  have key : (u (p * x + q * y) + m / 2 * ((p * x + q * y - a) * (p * x + q * y - c))) -
      (p * (u x + m / 2 * ((x - a) * (x - c))) + q * (u y + m / 2 * ((y - a) * (y - c)))) =
      (u (p * x + q * y) + m / 2 * (p * x + q * y) ^ 2) -
      (p * (u x + m / 2 * x ^ 2) + q * (u y + m / 2 * y ^ 2)) := by
    have hq' : q = 1 - p := by linarith
    subst hq'
    ring
  linarith

theorem two_node_bound {u : ℝ → ℝ} {D : Set ℝ} {a c m : ℝ}
    (hcv : ConvexOn ℝ D (fun t => u t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ D (fun t => u t - m / 2 * t ^ 2))
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a < c) (ha : u a = 0) (hc : u c = 0) {x : ℝ}
    (hx : x ∈ D) : |u x| ≤ m / 2 * |(x - a) * (x - c)| := by
  have hψ := convexOn_add_node (a := a) (c := c) hcv
  have hcv' : ConvexOn ℝ D (fun t => -u t + m / 2 * t ^ 2) :=
    hcc.neg.congr fun t _ => by simp only [Pi.neg_apply]; ring
  have hχ := convexOn_add_node (a := a) (c := c) hcv'
  have hψa : u a + m / 2 * ((a - a) * (a - c)) = 0 := by rw [ha]; ring
  have hψc : u c + m / 2 * ((c - a) * (c - c)) = 0 := by rw [hc]; ring
  have hχa : -u a + m / 2 * ((a - a) * (a - c)) = 0 := by rw [ha]; ring
  have hχc : -u c + m / 2 * ((c - a) * (c - c)) = 0 := by rw [hc]; ring
  rcases le_or_gt x a with hxa | hxa
  · have h1 := convex_zeros_left hψ hcD hac hψa hψc hx hxa
    have h2 := convex_zeros_left hχ hcD hac hχa hχc hx hxa
    have hP : 0 ≤ (x - a) * (x - c) := mul_nonneg_of_nonpos_of_nonpos (by linarith) (by linarith)
    rw [abs_of_nonneg hP, abs_le]
    constructor <;> linarith
  rcases le_or_gt c x with hcx | hcx
  · have h1 := convex_zeros_right hψ haD hac hψa hψc hx hcx
    have h2 := convex_zeros_right hχ haD hac hχa hχc hx hcx
    have hP : 0 ≤ (x - a) * (x - c) := mul_nonneg (by linarith) (by linarith)
    rw [abs_of_nonneg hP, abs_le]
    constructor <;> linarith
  · have h1 := convex_zeros_between hψ haD hcD hac.le hψa hψc ⟨hxa.le, hcx.le⟩
    have h2 := convex_zeros_between hχ haD hcD hac.le hχa hχc ⟨hxa.le, hcx.le⟩
    have hP : (x - a) * (x - c) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
    rw [abs_of_nonpos hP, abs_le]
    constructor <;> linarith

theorem slice_gradient_bound {w : ℝ → E →L[ℝ] ℝ} {D : Set ℝ} {a c m : ℝ} (hm : 0 ≤ m)
    (hcv : ∀ v : E, ConvexOn ℝ D (fun t => w t v + m * ‖v‖ / 2 * t ^ 2))
    (hcc : ∀ v : E, ConcaveOn ℝ D (fun t => w t v - m * ‖v‖ / 2 * t ^ 2))
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a < c) (ha : w a = 0) (hc : w c = 0) {x : ℝ}
    (hx : x ∈ D) : ‖w x‖ ≤ m / 2 * |(x - a) * (x - c)| := by
  refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun v => ?_
  have h := two_node_bound (u := fun t => w t v) (m := m * ‖v‖) (hcv v) (hcc v) haD hcD hac
    (by simp [ha]) (by simp [hc]) hx
  rw [Real.norm_eq_abs]
  calc |w x v| ≤ m * ‖v‖ / 2 * |(x - a) * (x - c)| := h
    _ = m / 2 * |(x - a) * (x - c)| * ‖v‖ := by ring

end L5

end CapFirstExit
