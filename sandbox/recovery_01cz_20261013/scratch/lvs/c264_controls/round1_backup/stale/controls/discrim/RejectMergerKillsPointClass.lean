import Mathlib
import OrdinaryH0
import OrdinaryH0Controls
open TorusFiltration
example : CategoryTheory.CategoryStruct.comp (ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true) (((AlgebraicTopology.singularHomologyFunctor (ModuleCat.{0} ℤ) 0).obj ordinaryH0Control_intCoeff).map ordinaryH0Control_collapse) = 0 := by
  simp only [ordinaryH0Control_point_ne_zero]
