import Specifications.Side24

/-! Probe (round 4, group Side24-derivatives-lattice): `ThetaSummable d` for every `d`,
with the two spelling fixes from probe 2 (`Finset.prod_le_prod₀`, `Summable.sum_le_tsum`).
Probe file only: `sorry` allowed here and nowhere else. -/

open UniversalLaw.Spec.Side24

namespace Probe4
variable (d : ℕ)

theorem latticeVec_apply (n : Fin d → ℤ) (i : Fin d) : latticeVec d n i = 24 * (n i : ℝ) := by
  simp [latticeVec]

theorem norm_latticeVec_sq (n : Fin d → ℤ) : ‖latticeVec d n‖ ^ 2 = 576 * intNormSq d n := by
  rw [EuclideanSpace.norm_sq_eq, intNormSq, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [latticeVec_apply, Real.norm_eq_abs, sq_abs]
  ring

theorem summable_gauss_int (c : ℝ) (hc : 0 < c) :
    Summable (fun m : ℤ => Real.exp (-c * (m : ℝ) ^ 2)) := by
  have h := HurwitzKernelBounds.summable_f_int 0 0 (t := c / Real.pi) (by positivity)
  refine h.congr fun m => ?_
  simp only [HurwitzKernelBounds.f_int, add_zero, pow_zero, one_mul]
  congr 1
  field_simp

theorem summable_pow_gauss_int (c : ℝ) (hc : 0 < c) :
    Summable (fun m : ℤ => (m : ℝ) ^ 6 * Real.exp (-c * (m : ℝ) ^ 2)) := by
  have h := HurwitzKernelBounds.summable_f_int 6 0 (t := c / Real.pi) (by positivity)
  refine h.congr fun m => ?_
  simp only [HurwitzKernelBounds.f_int, add_zero]
  rw [show |(m : ℝ)| ^ 6 = (m : ℝ) ^ 6 from (even_two_mul 3).pow_abs _]
  congr 1
  congr 1
  field_simp

theorem summable_pi_prod (g : ℤ → ℝ) (hg0 : ∀ m, 0 ≤ g m) (hg : Summable g) :
    Summable (fun n : Fin d → ℤ => ∏ i, g (n i)) := by
  refine summable_of_sum_le (c := (∑' m, g m) ^ d)
    (fun n => Finset.prod_nonneg fun i _ => hg0 _) fun s => ?_
  set t : Fin d → Finset ℤ := fun i => s.image (fun n => n i) with ht
  have hsub : s ⊆ Fintype.piFinset t := fun n hn => by
    rw [Fintype.mem_piFinset]; intro i; exact Finset.mem_image_of_mem _ hn
  have hle : ∀ i, ∑ m ∈ t i, g m ≤ ∑' m, g m := fun i =>
    hg.sum_le_tsum (t i) (fun m _ => hg0 m)
  calc ∑ n ∈ s, ∏ i, g (n i) ≤ ∑ n ∈ Fintype.piFinset t, ∏ i, g (n i) :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub fun n _ _ => Finset.prod_nonneg fun i _ => hg0 _
    _ = ∏ i, ∑ m ∈ t i, g m := (Finset.prod_univ_sum t fun _ m => g m).symm
    _ ≤ ∏ _i : Fin d, ∑' m, g m :=
        Finset.prod_le_prod₀ (fun i _ => Finset.sum_nonneg fun m _ => hg0 m) (fun i _ => hle i)
    _ = (∑' m, g m) ^ d := by simp

theorem exp_shift_le (z v : E d) :
    Real.exp (-(‖z + v‖ ^ 2) / 2) ≤ Real.exp (‖z‖ ^ 2 / 2) * Real.exp (-(‖v‖ ^ 2) / 4) := by
  rw [← Real.exp_add, Real.exp_le_exp]
  have h1 : ‖v‖ ≤ ‖z + v‖ + ‖z‖ := by
    calc ‖v‖ = ‖(z + v) - z‖ := by rw [add_sub_cancel_left]
      _ ≤ ‖z + v‖ + ‖z‖ := norm_sub_le _ _
  have h2 : 0 ≤ ‖z + v‖ := norm_nonneg _
  have h3 : 0 ≤ ‖z‖ := norm_nonneg _
  have h4 : 0 ≤ ‖v‖ := norm_nonneg _
  nlinarith [sq_nonneg (‖z + v‖ - ‖z‖)]

theorem thetaSummable_all : ∀ d : ℕ, ThetaSummable d := by
  intro d z
  have hg := summable_gauss_int 144 (by norm_num)
  have hprod := summable_pi_prod d _ (fun m => (Real.exp_pos _).le) hg
  refine Summable.of_nonneg_of_le (fun n => (Real.exp_pos _).le) (fun n => ?_)
    (hprod.mul_left (Real.exp (‖z‖ ^ 2 / 2)))
  calc Real.exp (-(‖z + latticeVec d n‖ ^ 2) / 2)
      ≤ Real.exp (‖z‖ ^ 2 / 2) * Real.exp (-(‖latticeVec d n‖ ^ 2) / 4) := exp_shift_le d z _
    _ = Real.exp (‖z‖ ^ 2 / 2) * ∏ i, Real.exp (-144 * ((n i : ℤ) : ℝ) ^ 2) := by
      congr 1
      rw [← Real.exp_sum, norm_latticeVec_sq, intNormSq, Finset.mul_sum, neg_div,
        Finset.sum_div, ← Finset.sum_neg_distrib]
      congr 1
      refine Finset.sum_congr rfl fun i _ => ?_
      ring

-- the interface field of `PeriodicCoefficientInterface` would follow:
example : ∀ d : ℕ, d = 2 ∨ d = 3 → ThetaSummable d := fun d _ => thetaSummable_all d

end Probe4

#print axioms Probe4.thetaSummable_all
