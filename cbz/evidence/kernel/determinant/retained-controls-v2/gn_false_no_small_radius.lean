import ResearchFormalCoreR1.GaussianNormalizer
open ResearchFormalCoreR1
-- The source's small r*K² budget cannot be discarded from the box floor.
example : (3 : ℝ)/8 ≤ gn_normalizedWeight 1 ![-1,1,9,0,-3/2,-3/2] := by
  norm_num [gn_normalizedWeight]
