import SourceKernel
import SourceGaussianLaw
import Mathlib.Probability.Moments.Covariance
import Mathlib.Probability.Moments.Variance

noncomputable section
open I4Foundation I4Weights I4Source I4Kernel I4SourceGaussian MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal NNReal
namespace I4SourceCovariance

variable {ι : Type*} [Countable ι]

theorem scalarSeries_covariance (a b : ι → ℝ) (ha : Summable a) (hb : Summable b) :
    cov[scalarSeries a, scalarSeries b; coefficientLaw ι] = ∑' i, a i * b i := by
  have hsq (f : ι → ℝ) (hf : Summable f) : Summable (fun i => f i ^ 2) := by
    have hp := summable_mul_of_summable_norm hf.norm hf.norm
    simpa only [Function.comp_def, pow_two] using hp.comp_injective
      (show Function.Injective (fun i : ι => (i,i)) from
        fun i j h => congrArg Prod.fst h)
  have hprod : Summable (fun i => a i * b i) := by
    have hp := summable_mul_of_summable_norm ha.norm hb.norm
    simpa only [Function.comp_def] using hp.comp_injective
      (show Function.Injective (fun i : ι => (i,i)) from
        fun i j h => congrArg Prod.fst h)
  have hLp (f : ι → ℝ) (hf : Summable f) :
      MemLp (scalarSeries f) 2 (coefficientLaw ι) := by
    have h : MemLp id 2 ((coefficientLaw ι).map (scalarSeries f)) := by
      rw [scalarSeries_map_gaussian f hf]
      exact memLp_id_gaussianReal' 2 (by norm_num)
    simpa only [Function.comp_def, Function.id_def] using
      h.comp_of_map (scalarSeries_measurable f).aemeasurable
  have heq : scalarSeries (a + b) =ᵐ[coefficientLaw ι]
      (fun ω => scalarSeries a ω + scalarSeries b ω) := by
    filter_upwards [scalarSeries_hasSum_ae a ha, scalarSeries_hasSum_ae b hb,
      scalarSeries_hasSum_ae (a+b) (ha.add hb)] with ω haω hbω habω
    exact habω.unique (by simpa only [Pi.add_apply, add_mul] using haω.add hbω)
  have hts : (∑' i, (a i + b i)^2) =
      (∑' i, a i ^ 2) + 2 * (∑' i, a i * b i) + (∑' i, b i ^ 2) := by
    calc
      _ = ∑' i, (a i ^ 2 + 2 * (a i * b i) + b i ^ 2) := by
        exact tsum_congr (fun i => by ring)
      _ = _ := by
        rw [((hsq a ha).add (hprod.mul_left 2)).tsum_add (hsq b hb),
          (hsq a ha).tsum_add (hprod.mul_left 2), tsum_mul_left]
  have hv := variance_fun_add (hLp a ha) (hLp b hb)
  rw [← variance_congr heq, scalarSeries_variance (a+b) (ha.add hb),
    scalarSeries_variance a ha, scalarSeries_variance b hb] at hv
  change (∑' i, (a i+b i)^2) = _ at hv
  rw [hts] at hv
  linarith

theorem sourceField_integral_zero (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin d → ℝ) :
    ∫ ω, sourceField d L ω x ∂sourceLaw d = 0 := by
  have hc : Summable (fun m => modeCoefficient d L m x) :=
    (modeMajorant_summable d hL).of_norm_bounded (fun m => by
      simpa only [Real.norm_eq_abs] using modeCoefficient_bound d hL m x)
  exact scalarSeries_integral_zero (fun m => modeCoefficient d L m x) hc

theorem sourceField_covariance (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x y : Fin d → ℝ) :
    cov[fun ω => sourceField d L ω x, fun ω => sourceField d L ω y; sourceLaw d] =
      sourceSpectralKernel d L (x - y) := by
  have hc (z : Fin d → ℝ) : Summable (fun m => modeCoefficient d L m z) :=
    (modeMajorant_summable d hL).of_norm_bounded (fun m => by
      simpa only [Real.norm_eq_abs] using modeCoefficient_bound d hL m z)
  change cov[scalarSeries (fun m => modeCoefficient d L m x),
    scalarSeries (fun m => modeCoefficient d L m y); coefficientLaw (SourceMode d)] = _
  rw [scalarSeries_covariance _ _ (hc x) (hc y), coefficientKernel_eq_fullLattice d hL x y]

theorem sourceField_variance_one (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin d → ℝ) :
    Var[fun ω => sourceField d L ω x; sourceLaw d] = 1 := by
  rw [← covariance_self (sourceField_eval_measurable d L x).aemeasurable,
    sourceField_covariance d hL x x, sub_self, spectralKernel_zero d hL]

end I4SourceCovariance
