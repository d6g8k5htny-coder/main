import Mathlib
import OrdinaryH0
import OrdinaryH0Controls
open TorusFiltration
theorem rejected : ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff false := by
  simp [ordinaryH0Control_bool_zeroCoeff_eq]
