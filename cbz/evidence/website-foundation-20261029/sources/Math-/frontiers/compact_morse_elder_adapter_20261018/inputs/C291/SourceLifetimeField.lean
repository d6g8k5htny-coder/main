import SourceLifetimeGeometry
import SourceThetaBridge

/-! The actual smooth source fills the unchanged Lifetime Gaussian field interface.
The only final parameter hypothesis is `0 < L`; theta, Gaussian, covariance and
smoothness properties are proved from the frozen literal source.
-/

noncomputable section
open MeasureTheory ProbabilityTheory
open UniversalLaw.Spec.Lifetime I4SmoothCarrier I4ThetaBridge
open scoped BigOperators ENNReal NNReal

namespace I4LifetimeField

theorem actualField_covariance (d : ℕ) {L : ℝ} (hL : 0 < L)
    (x y : E d) :
    cov[fun ω => actualField d L ω x, fun ω => actualField d L ω y; smoothLaw d L] =
      periodicCovariance d L (x - y) := by
  rw [actualField_covariance_spectral d hL]
  exact sourceSpectralKernel_eq_periodicCovariance d hL (x - y)

theorem actualField_raw_second_moment (d : ℕ) {L : ℝ} (hL : 0 < L)
    (x y : E d) :
    ∫ ω, actualField d L ω x * actualField d L ω y ∂smoothLaw d L =
      periodicCovariance d L (x - y) := by
  simpa only [ProbabilityTheory.covariance, actualField_integral_zero d hL, sub_zero]
    using actualField_covariance d hL x y

theorem actualField_linearCombination_variance (d : ℕ) {L : ℝ} (hL : 0 < L)
    {m : ℕ} (x : Fin m → E d) (a : Fin m → ℝ) :
    Var[fun ω => ∑ i, a i * actualField d L ω (x i); smoothLaw d L] =
      covQuadForm d L x a := by
  let : IsProbabilityMeasure (smoothLaw d L) := smoothLaw_isProbabilityMeasure d hL.ne'
  rw [variance_fun_sum (fun i =>
    (actualField_eval_hasGaussianLaw d hL (x i)).memLp_two.const_mul (a i))]
  simp only [covariance_const_mul_left, covariance_const_mul_right,
    actualField_covariance d hL, covQuadForm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem covQuadForm_nonneg (d : ℕ) {L : ℝ} (hL : 0 < L)
    {m : ℕ} (x : Fin m → E d) (a : Fin m → ℝ) :
    0 ≤ covQuadForm d L x a := by
  rw [← actualField_linearCombination_variance d hL x a]
  exact variance_nonneg _ _

theorem actualField_linearCombination_map (d : ℕ) {L : ℝ} (hL : 0 < L)
    {m : ℕ} (x : Fin m → E d) (a : Fin m → ℝ) :
    (smoothLaw d L).map (fun ω => ∑ i, a i * actualField d L ω (x i)) =
      gaussianReal 0 ⟨covQuadForm d L x a, covQuadForm_nonneg d hL x a⟩ := by
  rw [(actualField_linearCombination_hasGaussianLaw d hL x a).map_eq_gaussianReal,
    actualField_linearCombination_integral_zero d hL,
    actualField_linearCombination_variance d hL]
  congr 1
  apply Subtype.ext
  exact Real.coe_toNNReal _ (covQuadForm_nonneg d hL x a)

/-- The original target on the unchanged coefficient carrier and preserved source law. -/
def actualGaussianFieldInterface (d : ℕ) (L : ℝ) (hL : 0 < L) :
    GaussianFieldInterface d L where
  Ω := SmoothSample d L
  mΩ := inferInstance
  P := smoothLaw d L
  isProb := smoothLaw_isProbabilityMeasure d hL.ne'
  F := actualField d L
  smooth := actualField_smooth d hL
  periodic := actualField_periodic d hL
  summable := theta_summable d hL
  quad_nonneg := covQuadForm_nonneg d hL
  gaussian := actualField_linearCombination_map d hL
  covariance := actualField_raw_second_moment d hL

theorem actualGaussianFieldInterface_exists (d : ℕ) (L : ℝ) (hL : 0 < L) :
    Nonempty (GaussianFieldInterface d L) :=
  ⟨actualGaussianFieldInterface d L hL⟩

@[simp] theorem actualGaussianFieldInterface_carrier (d : ℕ) (L : ℝ) (hL : 0 < L) :
    (actualGaussianFieldInterface d L hL).Ω = SmoothSample d L := rfl

@[simp] theorem actualGaussianFieldInterface_law (d : ℕ) (L : ℝ) (hL : 0 < L) :
    (actualGaussianFieldInterface d L hL).P = smoothLaw d L := rfl

@[simp] theorem actualGaussianFieldInterface_field (d : ℕ) (L : ℝ) (hL : 0 < L)
    (ω : SmoothSample d L) (x : E d) :
    (actualGaussianFieldInterface d L hL).F ω x =
      smoothSourceField d L ω (coordinates d x) := rfl

end I4LifetimeField
