import ResearchFormalCoreR1.EndpointGaussianMoments
open ResearchFormalCoreR1
-- Exact endpoints of the invisible-midpoint-jet perturbation, versus the
-- quartic raw-nine-jet radius. These two charts cannot be silently identified.
example : egm_weight 1 ![-9,0,-1,9,0,-1]^2 ≤
    16*(1 : ℝ)^4*dm2Radius ![-1,0,0,0,0,0,0,0,0]^8 := by
  norm_num [egm_weight, egm_HM, egm_HS, dm2_negative_iff,
    Matrix.det_fin_two, dm2Radius, Fin.sum_univ_succ]
