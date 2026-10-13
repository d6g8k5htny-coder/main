import TorusFiltration
open TorusFiltration
example : (3 : ℝ) ∈ superlevel_set (fun x : ℝ => x) (-4) := by
  norm_num [superlevel_set]
