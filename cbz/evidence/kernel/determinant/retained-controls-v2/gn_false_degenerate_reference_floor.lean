import ResearchFormalCoreR1.GaussianNormalizer
open MeasureTheory ProbabilityTheory
-- Strictly positive reference mass requires nonzero variance.
example : 0 < ∫ q : ℝ, (max (-q) 0)^2 ∂gaussianReal 1 0 := by
  norm_num [gaussianReal_zero_var]
