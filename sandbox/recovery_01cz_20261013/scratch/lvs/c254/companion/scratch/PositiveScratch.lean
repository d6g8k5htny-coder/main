import Mathlib
noncomputable section
namespace TorusFiltration
abbrev torus_E (d : ℕ) := EuclideanSpace ℝ (Fin d)
def torus_latticeVec (d : ℕ) (L : ℝ) (n : Fin d → ℤ) : torus_E d :=
  ∑ i, EuclideanSpace.single i (L * (n i : ℝ))
def torus_isPeriodic (d : ℕ) (L : ℝ) (g : torus_E d → ℝ) : Prop :=
  ∀ (x : torus_E d) (n : Fin d → ℤ), g (x + torus_latticeVec d L n) = g x
def superlevel_set {X : Type*} (g : X → ℝ) (t : ℝ) : Set X :=
  {x | -t ≤ g x}
theorem superlevel_mono {X : Type*} (g : X → ℝ) {s t : ℝ} (hst : s ≤ t) :
    superlevel_set g s ⊆ superlevel_set g t := by
  intro x hx
  exact le_trans (neg_le_neg hst) hx
end TorusFiltration
end
open TorusFiltration
example : (3 : ℝ) ∈ superlevel_set (fun x : ℝ => x) (-2) := by
  norm_num [superlevel_set]
example : superlevel_set (fun x : ℝ => x) (-2) ⊆ superlevel_set (fun x : ℝ => x) 1 :=
  superlevel_mono (fun x : ℝ => x) (by norm_num)
example (d : ℕ) (L : ℝ) (c : ℝ) : torus_isPeriodic d L (fun _ => c) := fun _ _ => rfl
