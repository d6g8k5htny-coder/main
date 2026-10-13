import Mathlib
noncomputable section
namespace TorusFiltration
abbrev torus_E (d : ℕ) := EuclideanSpace ℝ (Fin d)
def torus_latticeVec (d : ℕ) (L : ℝ) (n : Fin d → ℤ) : torus_E d :=
  ∑ i, EuclideanSpace.single i (L * (n i : ℝ))
def torus_isPeriodic (d : ℕ) (L : ℝ) (g : torus_E d → ℝ) : Prop :=
  ∀ (x : torus_E d) (n : Fin d → ℤ), g (x + torus_latticeVec d L n) = g x
end TorusFiltration
end
open TorusFiltration
example : ¬ torus_isPeriodic 1 (1 : ℝ) (fun _ => (0 : ℝ)) := by
  simp [torus_isPeriodic]
