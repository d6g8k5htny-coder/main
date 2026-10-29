import ResearchFormalCoreR1.GaussianScalarMoments
open MeasureTheory ProbabilityTheory ResearchFormalCoreR1
-- Dropping the nonzero mean term would predict zero eighth moment for N(1,0).
example : (∫ x : ℝ, |x|^8 ∂gaussianReal 1 0) ≤ 128*((0 : ℝ)^8+0^4*gm8C8) := by
  norm_num [gaussianReal_zero_var]
