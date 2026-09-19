import Mathlib

namespace ResearchFormalCoreR1

set_option autoImplicit false

/-- Canonical A₂ normal-form potential, parameterized by `s` with λ = s². -/
def foldPotential (s x : ℝ) : ℝ := -(x ^ 3) / 3 + (s ^ 2) * x

/-- EC-005: exact canonical fold-gap identity in the normalized coordinate. -/
theorem ec005_fold_gap (s : ℝ) :
    foldPotential s s - foldPotential s (-s) = (2 * s) ^ 3 / 6 := by
  unfold foldPotential
  ring

/-- EC-008: scalar polynomial companion for the planar collision-Gram determinant.
    This is not the matrix-construction theorem. -/
theorem ec008_factor_expansion (c s : ℝ) :
    2 * s ^ 2 *
        ((s ^ 2 * (3 * c ^ 4 + 3 * c ^ 2 * s ^ 2 + s ^ 4) / 24) *
            (s ^ 2 * (4 * c ^ 2 + s ^ 2) / 2) -
          (-c * s ^ 2 * (2 * c ^ 2 + s ^ 2) / 4) ^ 2) =
      s ^ 8 * (c ^ 2 + s ^ 2) * (3 * c ^ 2 + s ^ 2) / 24 := by
  ring

/-- EC-010, generic chart: declared convex-combination polynomial identity. -/
theorem ec010_generic (α K c η : ℝ) :
    (K + 8 * α * c * η) ^ 2 + 64 * α * (1 - α) * c ^ 2 * η ^ 2 =
      (1 - α) * K ^ 2 + α * (K + 8 * c * η) ^ 2 := by
  ring

/-- EC-010, transverse chart: declared convex-combination polynomial identity. -/
theorem ec010_transverse (α a η : ℝ) :
    (a + (1 - 2 * α) * η) ^ 2 + 4 * α * (1 - α) * η ^ 2 =
      (1 - α) * (a + η) ^ 2 + α * (a - η) ^ 2 := by
  ring

/-- EC-011: scalar cancellation companion to the adjoint-orbit calculation.
    It does not formalize the ODE, Hessian symmetry, or section normalization. -/
theorem ec011_scalar_cancellation (h w g : ℝ) :
    (-h * w) * g + w * (h * g) = 0 := by
  ring

/-- EC-014: exact contact-power scalar identity for nonzero separation.
    It does not formalize the six-pin matrix determinant or density transform. -/
theorem ec014_contact_power (r : ℝ) (hr : r ≠ 0) :
    (r ^ 3 / 6) * (1 / r ^ 5) * r ^ 2 = (1 / 6 : ℝ) := by
  field_simp [hr]
  ring

end ResearchFormalCoreR1
