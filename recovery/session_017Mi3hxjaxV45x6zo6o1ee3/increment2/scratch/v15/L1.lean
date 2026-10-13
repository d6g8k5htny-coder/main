import CapFirstExit

open Set

namespace CapFirstExit

section L1

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

theorem hermite_gap_le {g G : ℝ → ℝ} {a c m : ℝ} (hac : a < c)
    (hg : ∀ x ∈ Icc a c, HasDerivAt g (G x) x)
    (hcv : ConvexOn ℝ (Icc a c) (fun t => G t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ (Icc a c) (fun t => G t - m / 2 * t ^ 2))
    (hGa : G a = 0) (hGc : G c = 0) :
    g a - g c ≤ m * (c - a) ^ 3 / 12 := by
  have hpoly : ∀ x, HasDerivAt (fun t => (c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
      ((x - a) * (c - x)) x := by
    intro x
    have h1 : HasDerivAt (fun t => t - a) 1 x := (hasDerivAt_id x).sub_const a
    have h2 := ((h1.pow 2).const_mul (c - a)).div_const 2
    have h3 := (h1.pow 3).div_const 3
    convert h2.sub h3 using 1
    push_cast
    ring
  set k : ℝ → ℝ := fun t => g t + m / 2 * ((c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
    with hk
  have hkd : ∀ x ∈ Icc a c, HasDerivAt k (G x + m / 2 * ((x - a) * (c - x))) x :=
    fun x hx => (hg x hx).add ((hpoly x).const_mul (m / 2))
  have hmono : MonotoneOn k (Icc a c) := by
    refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc a c)
      (fun x hx => (hkd x hx).continuousAt.continuousWithinAt)
      (fun x hx => (hkd x (interior_subset hx)).hasDerivWithinAt) (fun x hx => ?_)
    have hx := interior_subset hx
    have hb := two_node_bound hcv hcc (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac
      hGa hGc hx
    have hP : (x - a) * (x - c) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by linarith [hx.1]) (by linarith [hx.2])
    rw [abs_of_nonpos hP] at hb
    have := (abs_le.mp hb).1
    nlinarith
  have hkac := hmono (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac.le
  simp only [hk] at hkac
  have e0 : m / 2 * ((c - a) * (a - a) ^ 2 / 2 - (a - a) ^ 3 / 3) = 0 := by ring
  have e1 : m / 2 * ((c - a) * (c - a) ^ 2 / 2 - (c - a) ^ 3 / 3) = m * (c - a) ^ 3 / 12 := by
    ring
  linarith

theorem hermite_m_ge_two {g G : ℝ → ℝ} {r m : ℝ} (hr : 0 < r)
    (hg : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt g (G x) x)
    (hcv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G t - m / 2 * t ^ 2))
    (hGa : G (-r / 2) = 0) (hGc : G (r / 2) = 0) (hgap : g (-r / 2) - g (r / 2) = r ^ 3 / 6) :
    2 ≤ m := by
  have h := hermite_gap_le (by linarith) hg hcv hcc hGa hGc
  have hcr : r / 2 - -r / 2 = r := by ring
  rw [hgap, hcr] at h
  have hr3 : 0 < r ^ 3 := by positivity
  nlinarith

end L1

end CapFirstExit
