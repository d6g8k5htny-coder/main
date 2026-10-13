import Mathlib
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
noncomputable example : AddCommGroup (E →L[ℝ] E →L[ℝ] ℝ) := inferInstance
noncomputable example : NormedAddCommGroup (E →L[ℝ] E →L[ℝ] E →L[ℝ] ℝ) := inferInstance
noncomputable example : Norm (E →L[ℝ] E →L[ℝ] E →L[ℝ] ℝ) := inferInstance
set_option trace.Meta.synthInstance true in
noncomputable example : AddCommGroup (E →L[ℝ] E →L[ℝ] E →L[ℝ] ℝ) := inferInstance
