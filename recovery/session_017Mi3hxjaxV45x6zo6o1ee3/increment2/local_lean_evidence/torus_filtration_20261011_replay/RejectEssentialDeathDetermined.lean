import TorusFiltration
open TorusFiltration
example : (superlevelBarField_interface 1 1).deathHeight (fun _ => (0 : ℝ)) 0 = (superlevelBarField_shift 1 1 1).deathHeight (fun _ => (0 : ℝ)) 0 := by
  norm_num [superlevelBarField_deathHeight_eq, superlevelBar_death_const, superlevelBarField_shift_deathHeight_const]
