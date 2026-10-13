import Mathlib
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
example (g : ℝ × E → ((ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ)) (x : ℝ) (L : (ℝ × E) →L[ℝ] ((ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ))
    (h : HasFDerivAt g L (x, 0)) (hp : HasDerivAt (fun t : ℝ => ((t, (0 : E)) : ℝ × E)) ((1 : ℝ), (0 : E)) x) :
    HasDerivAt (fun t => g (t, 0)) (L (1, 0)) x := by
  exact HasFDerivAt.comp_hasDerivAt (l := g) (f := fun t : ℝ => ((t, (0 : E)) : ℝ × E)) x h hp
example (g : ℝ × E → ((ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ)) (x : ℝ) (L : (ℝ × E) →L[ℝ] ((ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ))
    (h : HasFDerivAt g L (x, 0)) (hp : HasDerivAt (fun t : ℝ => ((t, (0 : E)) : ℝ × E)) ((1 : ℝ), (0 : E)) x) :
    HasDerivAt (fun t => g (t, 0)) (L (1, 0)) x := by
  have := h.comp x hp.hasFDerivAt
  simpa using this.hasDerivAt
