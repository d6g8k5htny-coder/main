import ResearchFormalCoreR1.GaussianJetMoments
import ResearchFormalCoreR1.GaussianRegressionKernel
import ResearchFormalCoreR1.EndpointGaussianMoments
open MeasureTheory ProbabilityTheory ResearchFormalCoreR1
open scoped NNReal
-- All nine coordinates can be the SAME Gaussian variable: independence is not used.
example : Integrable (fun z : ℝ => (1+9*|z|)^8) (gaussianReal 0 1) := by
  have h := gm8_jet_integrable_bound (gaussianReal 0 1)
    (fun z : ℝ => (fun _i : Fin 9 => z)) (by fun_prop)
    (fun _ => 0) (fun _ => 1) 0 1 (by norm_num) (by norm_num)
    (by intro i; norm_num) (by intro i; norm_num)
    (by intro i; exact HasLaw.id)
  simpa [dm2Radius, Finset.sum_const] using h.1
-- Nonzero mean and zero variance are preserved, rather than centered away.
example : (∫ x : ℝ, |x|^8 ∂gaussianReal 1 0) = 1 := by
  norm_num [gaussianReal_zero_var]
-- The canonical formula specifies even an observation outside the source support.
example (x : Fin 1 → ℝ) :
    regressionKernel (Measure.dirac (0 : Fin 1 → ℝ))
      (ContinuousLinearMap.id ℝ (Fin 1 → ℝ)) 0 0 x = Measure.dirac x := by
  rw [regressionKernel_apply, Measure.map_dirac' (by fun_prop)]
  simp
-- The two actual endpoint types in the six-coordinate chart are allowed.
example : dm2Negative (egm_HM 1 ![-1,0,-1,1,0,-1]) ∧
    (egm_HS 1 ![-1,0,-1,1,0,-1]).det < 0 := by
  norm_num [egm_HM, egm_HS, dm2_negative_iff, Matrix.det_fin_two]
