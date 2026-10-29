import ResearchFormalCoreR1.GaussianJetMoments
open ResearchFormalCoreR1
-- The exact radius includes one, even when every Gaussian coordinate vanishes.
example : dm2Radius (fun _i : Fin 9 => 0)^8 ≤
    (10 : ℝ)^7 * ∑ _i : Fin 9, |(0 : ℝ)|^8 := by
  norm_num [dm2Radius]
