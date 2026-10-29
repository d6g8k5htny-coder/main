import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Independence.InfinitePi
import Mathlib.MeasureTheory.Function.LpSpace.InfiniteSum
import Mathlib.MeasureTheory.Function.LpSeminorm.SMul

/-! The actual product law and the almost-sure absolute-series foundation.
This does not assert a Fourier covariance, sample smoothness or finite-jet rank. -/

noncomputable section

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace I4Foundation

/-- The same actual probability measure used for every coefficient and series assertion. -/
def coefficientLaw (ι : Type*) : Measure (ι → ℝ) :=
  Measure.infinitePi (fun _ : ι => gaussianReal 0 1)

instance coefficientLaw_isProbabilityMeasure (ι : Type*) :
    IsProbabilityMeasure (coefficientLaw ι) := by
  unfold coefficientLaw
  infer_instance

theorem coordinate_measurePreserving {ι : Type*} (i : ι) :
    MeasurePreserving (fun ω : ι → ℝ => ω i) (coefficientLaw ι)
      (gaussianReal 0 1) := by
  exact measurePreserving_eval_infinitePi (fun _ : ι => gaussianReal 0 1) i

theorem coordinate_marginal {ι : Type*} (i : ι) :
    (coefficientLaw ι).map (fun ω => ω i) = gaussianReal 0 1 := by
  exact (coordinate_measurePreserving i).map_eq

theorem coordinates_independent (ι : Type*) :
    iIndepFun (fun i (ω : ι → ℝ) => ω i) (coefficientLaw ι) := by
  exact iIndepFun_infinitePi (P := fun _ : ι => gaussianReal 0 1)
    (X := fun _ : ι => id) (fun _ => measurable_id)

theorem coordinate_memLp {ι : Type*} (i : ι) (p : ℝ≥0∞) (hp : p ≠ ∞) :
    MemLp (fun ω : ι → ℝ => ω i) p (coefficientLaw ι) := by
  exact (memLp_id_gaussianReal' p hp).comp_measurePreserving
    (coordinate_measurePreserving i)

theorem coordinate_integrable {ι : Type*} (i : ι) :
    Integrable (fun ω : ι → ℝ => ω i) (coefficientLaw ι) := by
  exact memLp_one_iff_integrable.mp (coordinate_memLp i 1 (by simp))

theorem coordinate_abs_integrable {ι : Type*} (i : ι) :
    Integrable (fun ω : ι → ℝ => |ω i|) (coefficientLaw ι) := by
  simpa only [Real.norm_eq_abs] using (coordinate_integrable i).norm

theorem weighted_abs_summable_ae {ι : Type*} [Countable ι]
    (a : ι → ℝ) (ha : ∀ i, 0 ≤ a i) (hs : Summable a) :
    ∀ᵐ ω ∂coefficientLaw ι, Summable (fun i => a i * |ω i|) := by
  let C := eLpNorm id 1 (gaussianReal 0 1)
  have hC : C < ∞ := (memLp_id_gaussianReal' 1 (by simp)).eLpNorm_lt_top
  have hcoord (i : ι) :
      eLpNorm (fun ω : ι → ℝ => ω i) 1 (coefficientLaw ι) = C := by
    exact eLpNorm_comp_measurePreserving
      (memLp_id_gaussianReal' 1 (by simp)).aestronglyMeasurable
      (coordinate_measurePreserving i)
  have hnorm (i : ι) :
      eLpNorm (fun ω : ι → ℝ => a i * ω i) 1 (coefficientLaw ι) =
        ENNReal.ofReal (a i) * C := by
    change eLpNorm (a i • (fun ω : ι → ℝ => ω i)) 1 (coefficientLaw ι) = _
    rw [eLpNorm_const_smul, Real.enorm_eq_ofReal (ha i), hcoord]
  have hsum : (∑' i, eLpNorm (fun ω : ι → ℝ => a i * ω i) 1
      (coefficientLaw ι)) ≠ ∞ := by
    simp_rw [hnorm]
    rw [ENNReal.tsum_mul_right]
    exact ENNReal.mul_ne_top hs.tsum_ofReal_ne_top hC.ne
  have h := summable_norm_of_tsum_eLpNorm_ne_top (p := 1) (le_refl 1) hsum
  filter_upwards [h] with ω hω
  simpa only [Real.norm_eq_abs, abs_mul, abs_of_nonneg (ha _)] using hω

/-- The actual scalar sum; its convergence is proved separately on the product law. -/
def scalarSeries {ι : Type*} (a : ι → ℝ) (ω : ι → ℝ) : ℝ :=
  ∑' i, a i * ω i

theorem scalarSeries_measurable {ι : Type*} [Countable ι] (a : ι → ℝ) :
    Measurable (scalarSeries a) := by
  unfold scalarSeries
  apply Measurable.tsum
  intro i
  exact measurable_const.mul (measurable_pi_apply i)

theorem scalarSeries_hasSum_ae {ι : Type*} [Countable ι]
    (a : ι → ℝ) (hs : Summable a) :
    ∀ᵐ ω ∂coefficientLaw ι, HasSum (fun i => a i * ω i) (scalarSeries a ω) := by
  filter_upwards [weighted_abs_summable_ae (fun i => |a i|)
    (fun i => abs_nonneg _) hs.abs] with ω hω
  have hnorm : Summable (fun i => ‖a i * ω i‖) := by
    simpa only [Real.norm_eq_abs, abs_mul] using hω
  exact hnorm.of_norm.hasSum

end I4Foundation
