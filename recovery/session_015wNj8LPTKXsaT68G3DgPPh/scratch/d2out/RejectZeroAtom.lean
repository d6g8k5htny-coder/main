import D2MomentBridge
open ResearchFormalCoreR1
example : (1 / 2 : ℝ) ^ 2 < 1 / 2 → 0 < d2Delta (1 / 2) (1 / 2) (1 / 2) := by
  norm_num [d2Delta]
