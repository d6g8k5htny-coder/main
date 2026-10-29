import ResearchFormalCoreR1
open ResearchFormalCoreR1
-- Deliberately false: q removed from the radius.
example : let j : DM2Jet := ![(-100),0,0,0,0,0,0,0,0]
    ((dm2HM (1/2) j).det*(dm2HS (1/2) j).det)^2 ≤ 16*((1/2) : ℝ)^4*1^8 := by
  norm_num [dm2HM, dm2HS, dm2Radius, Matrix.det_fin_two, Fin.sum_univ_succ]
