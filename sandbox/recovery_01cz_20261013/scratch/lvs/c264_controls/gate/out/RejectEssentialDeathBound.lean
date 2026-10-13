import TorusFiltration
open TorusFiltration
example : superlevelBar_deathHeight (fun _ : superlevelBar_E 1 => (-1 : ℝ)) 0 ≤ (-1 : ℝ) := by
  norm_num [superlevelBar_death_const]
