import TorusFiltration
open TorusFiltration
example : ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff true ≠ ordinaryH0_pointClass (TopCat.of Bool) ordinaryH0Control_zeroCoeff false := by
  simp [ordinaryH0Control_bool_zeroCoeff_eq]
