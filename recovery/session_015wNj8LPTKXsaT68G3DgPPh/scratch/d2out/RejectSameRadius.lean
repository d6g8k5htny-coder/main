import D2MomentBridge
open ResearchFormalCoreR1
example : 0 < d2Delta (1 : ℝ) 1 1 := by
  norm_num [d2Delta]
