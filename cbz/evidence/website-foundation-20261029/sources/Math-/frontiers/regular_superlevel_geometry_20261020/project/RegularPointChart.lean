import Lifetime

/-!
# Regular scalar chart at a noncritical point

The construction is the finite-dimensional scalar specialization of Mathlib's
strict implicit-function chart.  It uses noncriticality only to prove that the
derivative has full range.
-/

namespace UniversalLaw.Spec.Lifetime

variable {d : ℕ} {g : E d → ℝ} {x : E d}

/-- At a noncritical point, the scalar derivative has full range. -/
theorem regularPointDerivative_range_eq_top (hx : ¬ IsCriticalPoint d g x) :
    (fderiv ℝ g x).range = ⊤ := by
  have hne : (fderiv ℝ g x).toLinearMap ≠ 0 := by
    intro hzero
    apply hx
    unfold IsCriticalPoint
    ext y
    have hy := LinearMap.congr_fun hzero y
    simpa using hy
  exact Module.Dual.range_eq_top_of_ne_zero hne

/-- Smoothness supplies the strict derivative used by the chart. -/
theorem regularPointHasStrictFDerivAt (hg : IsSmooth d g) :
    HasStrictFDerivAt g (fderiv ℝ g x) x :=
  (hg 1).hasStrictFDerivAt (by norm_num)

/-- The literal local scalar chart at a noncritical point. -/
noncomputable def regularPointChart (hg : IsSmooth d g) (hx : ¬ IsCriticalPoint d g x) :
    OpenPartialHomeomorph (E d) (ℝ × (fderiv ℝ g x).ker) :=
  (regularPointHasStrictFDerivAt hg).implicitToOpenPartialHomeomorph
    g (fderiv ℝ g x) (regularPointDerivative_range_eq_top hx)

/-- The chart's first coordinate is exactly `g`, globally as a function equality. -/
@[simp]
theorem regularPointChart_fst (hg : IsSmooth d g) (hx : ¬ IsCriticalPoint d g x) (y : E d) :
    (regularPointChart hg hx y).1 = g y := by
  exact HasStrictFDerivAt.implicitToOpenPartialHomeomorph_fst
    (regularPointHasStrictFDerivAt hg) (regularPointDerivative_range_eq_top hx) y

/-- The chart sends the base point to `(g x, 0)`. -/
@[simp]
theorem regularPointChart_self (hg : IsSmooth d g) (hx : ¬ IsCriticalPoint d g x) :
    regularPointChart hg hx x = (g x, 0) := by
  exact HasStrictFDerivAt.implicitToOpenPartialHomeomorph_self
    (regularPointHasStrictFDerivAt hg) (regularPointDerivative_range_eq_top hx)

/-- The base point belongs to the chart source. -/
theorem mem_regularPointChart_source (hg : IsSmooth d g) (hx : ¬ IsCriticalPoint d g x) :
    x ∈ (regularPointChart hg hx).source := by
  exact HasStrictFDerivAt.mem_implicitToOpenPartialHomeomorph_source
    (regularPointHasStrictFDerivAt hg) (regularPointDerivative_range_eq_top hx)

/-- The image base point belongs to the chart target. -/
theorem mem_regularPointChart_target (hg : IsSmooth d g) (hx : ¬ IsCriticalPoint d g x) :
    (g x, 0) ∈ (regularPointChart hg hx).target := by
  exact HasStrictFDerivAt.mem_implicitToOpenPartialHomeomorph_target
    (regularPointHasStrictFDerivAt hg) (regularPointDerivative_range_eq_top hx)

end UniversalLaw.Spec.Lifetime
