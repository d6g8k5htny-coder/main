import TorusFiltration
open TorusFiltration
example : ¬ torus_isPeriodic 1 (1 : ℝ) (fun _ => (0 : ℝ)) := by
  simp [torus_isPeriodic]
