import Lifetime
import SourceSmoothCarrier

/-! Coordinate and probability transport for the literal smooth Fourier source.
No theta identity or target field-law premise is introduced in this module.
-/

noncomputable section
open MeasureTheory ProbabilityTheory
open UniversalLaw.Spec.Lifetime I4SmoothCarrier
open scoped BigOperators ENNReal NNReal

namespace I4LifetimeField

def coordinates (d : ℕ) : E d ≃L[ℝ] (Fin d → ℝ) :=
  EuclideanSpace.equiv (Fin d) ℝ

@[simp] theorem coordinates_apply (d : ℕ) (x : E d) (i : Fin d) :
    coordinates d x i = x i := rfl

theorem coordinates_latticeVec (d : ℕ) (L : ℝ) (n : Fin d → ℤ) :
    coordinates d (latticeVec d L n) = fun i => L * (n i : ℝ) := by
  ext i
  simp [coordinates, latticeVec]

/-- The same carrier, same law, and same literal source; only spatial coordinates change. -/
def actualField (d : ℕ) (L : ℝ) (ω : SmoothSample d L) (x : E d) : ℝ :=
  smoothSourceField d L ω (coordinates d x)

theorem actualField_smooth (d : ℕ) {L : ℝ} (hL : 0 < L)
    (ω : SmoothSample d L) : IsSmooth d (actualField d L ω) := by
  intro n
  exact (smoothSourceField_contDiff d hL.ne' ω n).comp
    (coordinates d).contDiff

theorem actualField_periodic (d : ℕ) {L : ℝ} (hL : 0 < L)
    (ω : SmoothSample d L) : IsLPeriodic d L (actualField d L ω) := by
  intro x n
  unfold actualField smoothSourceField
  rw [map_add, coordinates_latticeVec]
  exact I4Lattice.sourceField_lattice_shift d hL.ne' (sampleInclusion d L ω) n
    (coordinates d x)

theorem actualField_eval_measurable (d : ℕ) (L : ℝ) (x : E d) :
    Measurable (fun ω => actualField d L ω x) :=
  (I4Source.sourceField_eval_measurable d L (coordinates d x)).comp
    (sampleInclusion_measurable d L)

theorem actualField_eval_hasGaussianLaw (d : ℕ) {L : ℝ} (hL : 0 < L)
    (x : E d) : HasGaussianLaw (fun ω => actualField d L ω x) (smoothLaw d L) := by
  exact (smoothSourceField_finiteVector_hasGaussianLaw d hL.ne' 1
    (fun _ => coordinates d x)).eval 0

theorem actualField_linearCombination_measurable (d : ℕ) (L : ℝ) {m : ℕ}
    (x : Fin m → E d) (a : Fin m → ℝ) :
    Measurable (fun ω => ∑ i, a i * actualField d L ω (x i)) := by
  exact Finset.measurable_sum _ (fun i _ =>
    measurable_const.mul (actualField_eval_measurable d L (x i)))

theorem actualField_linearCombination_hasGaussianLaw (d : ℕ) {L : ℝ}
    (hL : 0 < L) {m : ℕ} (x : Fin m → E d) (a : Fin m → ℝ) :
    HasGaussianLaw (fun ω => ∑ i, a i * actualField d L ω (x i)) (smoothLaw d L) := by
  have hX : Measurable (fun η : I4Source.SourceMode d → ℝ =>
      ∑ i, a i * I4Source.sourceField d L η (coordinates d (x i))) :=
    Finset.measurable_sum _ (fun i _ => measurable_const.mul
      (I4Source.sourceField_eval_measurable d L (coordinates d (x i))))
  refine ⟨(actualField_linearCombination_measurable d L x a).aemeasurable, ?_⟩
  change IsGaussian ((smoothLaw d L).map (fun ω =>
    ∑ i, a i * I4Source.sourceField d L (sampleInclusion d L ω) (coordinates d (x i))))
  rw [observation_map_eq d hL.ne' _ hX]
  exact (I4SourceGaussian.source_linearCombination_hasGaussianLaw d hL.ne' m
    (fun i => coordinates d (x i)) a).isGaussian_map

theorem actualField_integral_zero (d : ℕ) {L : ℝ} (hL : 0 < L) (x : E d) :
    ∫ ω, actualField d L ω x ∂smoothLaw d L = 0 := by
  change (∫ ω, I4Source.sourceField d L (sampleInclusion d L ω) (coordinates d x)
    ∂smoothLaw d L) = 0
  rw [(sampleInclusion_measurePreserving d hL.ne').integral_comp
    (MeasurableEmbedding.subtype_coe (goodSet_measurable d L))
    (fun η => I4Source.sourceField d L η (coordinates d x))]
  exact I4SourceCovariance.sourceField_integral_zero d hL.ne' (coordinates d x)

theorem actualField_linearCombination_integral_zero (d : ℕ) {L : ℝ}
    (hL : 0 < L) {m : ℕ} (x : Fin m → E d) (a : Fin m → ℝ) :
    ∫ ω, (∑ i, a i * actualField d L ω (x i)) ∂smoothLaw d L = 0 := by
  rw [integral_finsetSum _ (fun i _ =>
    (actualField_eval_hasGaussianLaw d hL (x i)).integrable.const_mul (a i))]
  simp only [integral_const_mul, actualField_integral_zero d hL, mul_zero,
    Finset.sum_const_zero]

theorem actualField_covariance_spectral (d : ℕ) {L : ℝ} (hL : 0 < L)
    (x y : E d) :
    cov[fun ω => actualField d L ω x, fun ω => actualField d L ω y; smoothLaw d L] =
      I4Kernel.sourceSpectralKernel d L (coordinates d (x - y)) := by
  simpa only [actualField, map_sub] using
    smoothSourceField_covariance d hL.ne' (coordinates d x) (coordinates d y)

end I4LifetimeField
