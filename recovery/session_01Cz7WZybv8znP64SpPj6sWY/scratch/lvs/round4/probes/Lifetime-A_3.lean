import Specifications.Lifetime

/-! Probe (Lifetime-A_3): the L = 0 degeneracy of IsLPeriodic, spelled out. Probe file only. No `sorry`. -/

open UniversalLaw.Spec.Lifetime

theorem single_zero' (d : ℕ) (i : Fin d) : EuclideanSpace.single i (0:ℝ) = 0 := by
  ext j
  simp [EuclideanSpace.single_apply]

-- IsLPeriodic with L = 0 holds for every g (the lattice vector is 0).
example (d : ℕ) (g : E d → ℝ) : IsLPeriodic d 0 g := by
  intro x n
  simp [latticeVec, single_zero']
