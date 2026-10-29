import ResearchFormalCoreR1
open ResearchFormalCoreR1
-- Deliberately false: r^4 replaced by r^6.
example : let j : DM2Jet := ![(-1),0,0,0,0,0,0,0,0]
    ((dm2HM (1/128) j).det*(dm2HS (1/128) j).det)^2 ≤ 16*((1/128) : ℝ)^6*dm2Radius j^8 := by
  norm_num [dm2HM, dm2HS, dm2Radius, Matrix.det_fin_two, Fin.sum_univ_succ]
