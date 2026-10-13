import D2MomentBridge
open D2MomentBridge ResearchFormalCoreR1
example (c x : ℝ) : cubicResidual c x = 0 ↔ x = 0 ∨ x ^ 2 = c :=
  cubicResidual_eq_zero_iff c x
example : cubicResidual 1 (0 : ℝ) = 0 := by norm_num [cubicResidual]
example : cubicResidual 1 (2 : ℝ) ≠ 0 := by norm_num [cubicResidual]
example : d2Delta (1 / 2) (1 / 2) (1 / 2) = 0 := zero_atom_counterexample.2
