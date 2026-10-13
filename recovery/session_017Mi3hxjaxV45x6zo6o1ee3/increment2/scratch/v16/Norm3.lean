import Mathlib
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
example (T : E →L[ℝ] E →L[ℝ] E →L[ℝ] ℝ) : 0 ≤ ‖T‖ := norm_nonneg _
example (T : E →L[ℝ] (E →L[ℝ] (E →L[ℝ] ℝ))) : 0 ≤ ‖T‖ := norm_nonneg _
