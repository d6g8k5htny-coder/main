import TorusFiltration
open TorusFiltration
example : () ∈ superlevel_set (fun _ : Unit => (0 : ℝ)) (-1) := by
  norm_num [superlevel_set]
