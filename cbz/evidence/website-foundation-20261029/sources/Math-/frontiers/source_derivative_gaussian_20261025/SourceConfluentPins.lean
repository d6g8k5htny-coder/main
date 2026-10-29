import SourceDerivativeGaussian
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring

/-!
# Actual six endpoint pins and their confluent frame

The raw order is `(fM,fS,fxM,fxS,fyM,fyS)`. This is the literal LM006
endpoint-extension frame, including its arbitrary prescribed height.
See SOURCE_CONFLUENT_PINS.md for the frozen preimplementation contract.
-/

noncomputable section

open I4SmoothCarrier I4SourceDerivativeGaussian
open MeasureTheory ProbabilityTheory
open scoped Topology

namespace I4SourceConfluentPins

/-- Literal raw-to-confluent transform, defined also at zero for notation. -/
def confluentTransform (r : ℝ) (v : Fin 6 → ℝ) : Fin 6 → ℝ :=
  ![(v 0 + v 1) / 2, (v 2 + v 3) / 2, (v 4 + v 5) / 2,
    (v 3 - v 2) / r, (v 5 - v 4) / r,
    (12 / r ^ 2) * ((v 2 + v 3) / 2 - (v 1 - v 0) / r)]

/-- The literal inverse, with the cubic correction to the two field values. -/
def rawTransform (r : ℝ) (p : Fin 6 → ℝ) : Fin 6 → ℝ :=
  ![p 0 - r * p 1 / 2 + r ^ 3 * p 5 / 24,
    p 0 + r * p 1 / 2 - r ^ 3 * p 5 / 24,
    p 1 - r * p 3 / 2, p 1 + r * p 3 / 2,
    p 2 - r * p 4 / 2, p 2 + r * p 4 / 2]

theorem rawTransform_confluentTransform (r : ℝ) (hr : r ≠ 0) (v : Fin 6 → ℝ) :
    rawTransform r (confluentTransform r v) = v := by
  ext i
  fin_cases i <;> simp [rawTransform, confluentTransform]
  all_goals field_simp [hr]
  all_goals ring

theorem confluentTransform_rawTransform (r : ℝ) (hr : r ≠ 0) (p : Fin 6 → ℝ) :
    confluentTransform r (rawTransform r p) = p := by
  ext i
  fin_cases i <;> simp [rawTransform, confluentTransform]
  all_goals field_simp [hr]
  all_goals ring

/-- For every nonzero separation, the confluent frame is an actual continuous
linear equivalence. No covariance invertibility is assumed or asserted. -/
def confluentEquiv (r : ℝ) (hr : r ≠ 0) : (Fin 6 → ℝ) ≃L[ℝ] (Fin 6 → ℝ) where
  toFun := confluentTransform r
  invFun := rawTransform r
  left_inv := rawTransform_confluentTransform r hr
  right_inv := confluentTransform_rawTransform r hr
  map_add' v w := by
    ext i
    fin_cases i <;> simp [confluentTransform] <;> ring
  map_smul' c v := by
    ext i
    fin_cases i <;> simp [confluentTransform] <;> ring
  continuous_toFun := by unfold confluentTransform; fun_prop
  continuous_invFun := by unfold rawTransform; fun_prop

@[simp] theorem confluentEquiv_apply (r : ℝ) (hr : r ≠ 0) (v : Fin 6 → ℝ) :
    confluentEquiv r hr v = confluentTransform r v := rfl

@[simp] theorem confluentEquiv_symm_apply (r : ℝ) (hr : r ≠ 0) (p : Fin 6 → ℝ) :
    (confluentEquiv r hr).symm p = rawTransform r p := rfl

/-- The literal prescribed raw pins keep arbitrary height and have zero first
confluent derivative. The normalized third confluent pin is exactly two. -/
theorem prescribed_target (r b : ℝ) (hr : r ≠ 0) :
    confluentEquiv r hr ![b, b - r ^ 3 / 6, 0, 0, 0, 0] =
      ![b - r ^ 3 / 12, 0, 0, 0, 0, 2] := by
  ext i
  fin_cases i <;> simp [confluentTransform] <;> field_simp [hr] <;> ring

def endpointM (r : ℝ) : Fin 2 → ℝ := ![-r / 2, 0]
def endpointS (r : ℝ) : Fin 2 → ℝ := ![r / 2, 0]
def directionX : Fin 2 → ℝ := ![1, 0]
def directionY : Fin 2 → ℝ := ![0, 1]

/-- Actual field values and actual directional x/y first derivatives of the
unchanged smooth Fourier source, in the literal raw pin order. -/
def rawPins (L r : ℝ) (ω : SmoothSample 2 L) : Fin 6 → ℝ :=
  ![smoothSourceField 2 L ω (endpointM r),
    smoothSourceField 2 L ω (endpointS r),
    fderiv ℝ (smoothSourceField 2 L ω) (endpointM r) directionX,
    fderiv ℝ (smoothSourceField 2 L ω) (endpointS r) directionX,
    fderiv ℝ (smoothSourceField 2 L ω) (endpointM r) directionY,
    fderiv ℝ (smoothSourceField 2 L ω) (endpointS r) directionY]

/-- Joint Gaussianity is derived from one heterogeneous vector of actual
order-zero and order-one observations; no observation independence is used. -/
theorem rawPins_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    HasGaussianLaw (rawPins L r) (smoothLaw 2 L) := by
  let k : Fin 6 → ℕ := ![0, 0, 1, 1, 1, 1]
  let x : Fin 6 → (Fin 2 → ℝ) :=
    ![endpointM r, endpointS r, endpointM r, endpointS r, endpointM r, endpointS r]
  let u : ∀ j : Fin 6, Fin (k j) → (Fin 2 → ℝ) :=
    fun j _ => if j.val < 4 then directionX else directionY
  have hG := smoothSourceField_derivativeVector_hasGaussianLaw 2 hL 6 k x u
  apply hG.congr
  filter_upwards with ω
  ext i
  fin_cases i <;> dsimp [rawPins, k, x, u]
  all_goals first | rfl | exact iteratedFDeriv_one_apply _

/-- Actual confluent observations are the proved linear image of raw pins. -/
def confluentPins (L r : ℝ) (hr : r ≠ 0) (ω : SmoothSample 2 L) : Fin 6 → ℝ :=
  confluentEquiv r hr (rawPins L r ω)

theorem confluentPins_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0) :
    HasGaussianLaw (confluentPins L r hr) (smoothLaw 2 L) :=
  (rawPins_hasGaussianLaw hL r).map_equiv_fun (confluentEquiv r hr)

/-- One literal vector containing the raw six pins followed by endpoint Hessian
entries `(fxxM,fxyM,fyyM,fxxS,fxyS,fyyS)`. No r-dependent midpoint jet is used. -/
def rawPinsWithHessians (L r : ℝ) (ω : SmoothSample 2 L) : Fin 12 → ℝ :=
  ![rawPins L r ω 0, rawPins L r ω 1, rawPins L r ω 2,
    rawPins L r ω 3, rawPins L r ω 4, rawPins L r ω 5,
    iteratedFDeriv ℝ 2 (smoothSourceField 2 L ω) (endpointM r) ![directionX, directionX],
    iteratedFDeriv ℝ 2 (smoothSourceField 2 L ω) (endpointM r) ![directionX, directionY],
    iteratedFDeriv ℝ 2 (smoothSourceField 2 L ω) (endpointM r) ![directionY, directionY],
    iteratedFDeriv ℝ 2 (smoothSourceField 2 L ω) (endpointS r) ![directionX, directionX],
    iteratedFDeriv ℝ 2 (smoothSourceField 2 L ω) (endpointS r) ![directionX, directionY],
    iteratedFDeriv ℝ 2 (smoothSourceField 2 L ω) (endpointS r) ![directionY, directionY]]

/-- The pin and Hessian observations are jointly Gaussian, derived together
from their single actual derivative vector rather than separate marginal laws. -/
theorem rawPinsWithHessians_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    HasGaussianLaw (rawPinsWithHessians L r) (smoothLaw 2 L) := by
  let k : Fin 12 → ℕ := ![0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
  let x : Fin 12 → (Fin 2 → ℝ) :=
    ![endpointM r, endpointS r, endpointM r, endpointS r, endpointM r, endpointS r,
      endpointM r, endpointM r, endpointM r, endpointS r, endpointS r, endpointS r]
  let u : ∀ j : Fin 12, Fin (k j) → (Fin 2 → ℝ) := fun j a =>
    if j.val < 4 then directionX else if j.val < 6 then directionY
    else if j.val % 3 = 0 then directionX
    else if j.val % 3 = 1 ∧ a.val = 0 then directionX else directionY
  have hG := smoothSourceField_derivativeVector_hasGaussianLaw 2 hL 12 k x u
  apply hG.congr
  filter_upwards with ω
  ext i
  fin_cases i <;> dsimp [rawPinsWithHessians, rawPins, k, x, u]
  all_goals first
    | rfl
    | exact iteratedFDeriv_one_apply _
    | congr 1; ext a; fin_cases a <;> rfl

end I4SourceConfluentPins
