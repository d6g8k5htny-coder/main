import Mathlib
import OrdinaryH0
import OrdinaryH0Controls
open TorusFiltration
example : ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_intCoeff false := by
  simp [ordinaryH0Control_bool_zeroCoeff_eq]
