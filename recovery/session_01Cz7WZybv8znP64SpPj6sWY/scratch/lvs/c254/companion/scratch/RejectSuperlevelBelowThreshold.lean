import Mathlib
noncomputable section
namespace TorusFiltration
def superlevel_set {X : Type*} (g : X → ℝ) (t : ℝ) : Set X :=
  {x | -t ≤ g x}
end TorusFiltration
end
open TorusFiltration
example : (3 : ℝ) ∈ superlevel_set (fun x : ℝ => x) (-4) := by
  norm_num [superlevel_set]
