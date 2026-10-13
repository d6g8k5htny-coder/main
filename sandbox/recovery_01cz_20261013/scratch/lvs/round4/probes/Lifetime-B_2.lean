import Specifications.Lifetime

/-! Probe (Lifetime-B_2). Probe file only. Repairs the one failing step of Lifetime-B_1
(`latticeVec d L 0 = 0`): the cheap witness that the universal closure of
`IsUnrestrictedPair` (hence of `IsWindowPair`, `IsWindowElderPair`) is false, via
`torusDist d L M M ≤ 0`. No `sorry`. -/

open UniversalLaw.Spec.Lifetime

lemma single_zero_B (d : ℕ) (i : Fin d) : EuclideanSpace.single i (0 : ℝ) = 0 := by
  ext j
  simp [EuclideanSpace.single_apply]

lemma latticeVec_zero_B (d : ℕ) (L : ℝ) : latticeVec d L (0 : Fin d → ℤ) = 0 := by
  unfold latticeVec
  simp [single_zero_B]

lemma torusDist_self_le_zero_B (d : ℕ) (L : ℝ) (M : E d) : torusDist d L M M ≤ 0 := by
  unfold torusDist
  have hbdd : BddBelow (Set.range fun n : Fin d → ℤ => ‖M - M + latticeVec d L n‖) := by
    refine ⟨0, ?_⟩
    rintro _ ⟨n, rfl⟩
    exact norm_nonneg _
  refine (ciInf_le hbdd (0 : Fin d → ℤ)).trans ?_
  simp [latticeVec_zero_B]

example (d : ℕ) (L : ℝ) (g : E d → ℝ) (M : E d) : ¬ IsUnrestrictedPair d L g M M := by
  intro h
  have := h.1.2.2
  linarith [torusDist_self_le_zero_B d L M]

example (d : ℕ) (L ρ bLo bHi kLo kHi : ℝ) (g : E d → ℝ) (M : E d) :
    ¬ IsWindowPair d L ρ bLo bHi kLo kHi g M M := by
  intro h
  have := h.1.2.2
  linarith [torusDist_self_le_zero_B d L M]
