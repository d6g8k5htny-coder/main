import Mathlib
noncomputable section
namespace TorusFiltration
def superlevel_set {X : Type*} (g : X → ℝ) (t : ℝ) : Set X :=
  {x | -t ≤ g x}
end TorusFiltration
end
open TorusFiltration
example : () ∈ superlevel_set (fun _ : Unit => (0 : ℝ)) (-1) := by
  norm_num [superlevel_set]
