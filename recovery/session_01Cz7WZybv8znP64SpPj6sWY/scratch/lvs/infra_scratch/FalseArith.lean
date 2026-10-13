import Specifications.P15
example : (6 : ℝ) / 7 ≤ 6 / 7 - 1 / 10 ^ 20 := by norm_num
theorem tightened : (16 : ℝ) * 7 < 16 * 7 := by norm_num
example : UniversalLaw.Spec.P15.RhoStarBelowSixSevenths → UniversalLaw.Spec.P15.RhoStarBelowSixSevenths := fun h => h
