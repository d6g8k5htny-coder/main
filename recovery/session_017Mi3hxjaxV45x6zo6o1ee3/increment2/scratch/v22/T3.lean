import Mathlib
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
noncomputable def Ξ0 : ((ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ) →L[ℝ] (E →L[ℝ] E →L[ℝ] ℝ) :=
    (ContinuousLinearMap.postcomp E (ContinuousLinearMap.precomp ℝ (ContinuousLinearMap.inr ℝ ℝ E))).comp
      (ContinuousLinearMap.precomp ((ℝ × E) →L[ℝ] ℝ) (ContinuousLinearMap.inr ℝ ℝ E))
example (g : ℝ → ((ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ)) (g' : (ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ) (x : ℝ)
    (h : HasDerivAt g g' x) : HasDerivAt (fun t => Ξ0 (E := E) (g t)) (Ξ0 (E := E) g') x := by
  have hΞ : HasFDerivAt (Ξ0 (E := E)) (Ξ0 (E := E)) (g x) :=
    ContinuousLinearMap.hasFDerivAt (E := (ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ) (F := E →L[ℝ] E →L[ℝ] ℝ) _
  exact HasFDerivAt.comp_hasDerivAt (l := Ξ0 (E := E)) (f := g) x hΞ h
