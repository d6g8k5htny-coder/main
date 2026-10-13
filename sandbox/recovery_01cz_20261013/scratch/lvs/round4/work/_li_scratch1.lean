import Specifications.Side24
import SpecificationProofs.Side24Elementary

-- SCRATCH ONLY (not a deliverable): isolate the pieces that failed in run 1.

namespace Scratch1

open UniversalLaw.Spec.Side24

-- V1: probe-like, with `set t`
theorem v1 (d : ℕ) (h : Fin d → ℤ → ℝ) (h0 : ∀ k m, 0 ≤ h k m)
    (hs : ∀ k, Summable (h k)) (s : Finset (Fin d → ℤ)) :
    ∑ n ∈ s, ∏ k, h k (n k) ≤ ∏ k, ∑' m, h k m := by
  set t : Fin d → Finset ℤ := fun k => s.image (fun n => n k)
  have hsub : s ⊆ Fintype.piFinset t := fun n hn => by
    rw [Fintype.mem_piFinset]
    intro k
    exact Finset.mem_image_of_mem _ hn
  have hle : ∀ k, ∑ m ∈ t k, h k m ≤ ∑' m, h k m := fun k =>
    (hs k).sum_le_tsum (t k) (fun m _ => h0 k m)
  calc ∑ n ∈ s, ∏ k, h k (n k) ≤ ∑ n ∈ Fintype.piFinset t, ∏ k, h k (n k) :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub fun n _ _ => Finset.prod_nonneg fun k _ => h0 _ _
    _ = ∏ k, ∑ m ∈ t k, h k m := (Finset.prod_univ_sum t fun k m => h k m).symm
    _ ≤ ∏ k, ∑' m, h k m :=
        Finset.prod_le_prod₀ (fun k _ => Finset.sum_nonneg fun m _ => h0 k m) (fun k _ => hle k)

-- V2: the prod_ite lemma with the non-deprecated names
theorem v2 (d : ℕ) (i : Fin d) (F G : Fin d → ℝ) :
    (∏ k : Fin d, if k = i then F k else G k) = F i * ∏ k ∈ Finset.univ.erase i, G k := by
  rw [← Finset.mul_prod_erase Finset.univ (fun k => if k = i then F k else G k)
    (Finset.mem_univ i), ite_eq_left rfl]
  congr 1
  exact Finset.prod_congr rfl fun k hk => ite_eq_right (Finset.ne_of_mem_erase hk)

-- V3: the `h2` pattern with `set`
theorem v3 (i : Fin 3) (f g : ℤ → ℝ) : True := by
  set A : ℝ := ∑' m : ℤ, g m with hA
  set B : ℝ := ∑' m : ℤ, f m with hB
  have h2 : ∀ k : Fin 3, ∑' m : ℤ, (if k = i then f m else g m) = if k = i then B else A := by
    intro k
    by_cases hk : k = i
    · simp only [hk, ite_true, hB]
    · simp only [hk, ite_false, hA]
  trivial

-- V4: summability-of-ite pattern
theorem v4 (i : Fin 3) (f g : ℤ → ℝ) (hf : Summable f) (hg : Summable g) (k : Fin 3) :
    Summable (fun m => if k = i then f m else g m) := by
  by_cases hk : k = i
  · simp only [hk, ite_true]
    exact hf
  · simp only [hk, ite_false]
    exact hg

end Scratch1
