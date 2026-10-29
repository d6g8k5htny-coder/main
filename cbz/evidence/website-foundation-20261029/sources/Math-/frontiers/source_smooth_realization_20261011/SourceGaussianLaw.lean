import SourceFourierSeries
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.Probability.Independence.CharacteristicFunction
import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Independence
import Mathlib.Analysis.Normed.Ring.InfiniteSum
import Mathlib.Probability.Distributions.Gaussian.IsGaussianProcess.Def

/-! Actual Gaussian laws of summable coefficient series and finite evaluation combinations.
No covariance/source-kernel, torus, derivative-jet, or pinned-law conclusion is assumed. -/

noncomputable section
open I4Foundation I4Weights I4Source MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal NNReal Topology

namespace I4SourceGaussian

variable {ι : Type*}

private theorem summable_square (a : ι → ℝ) (ha : Summable a) :
    Summable (fun i => a i ^ 2) := by
  have h := summable_mul_of_summable_norm ha.norm ha.norm
  simpa only [Function.comp_def, pow_two] using h.comp_injective
    (show Function.Injective (fun i : ι => (i, i)) from
      fun i j hij => congrArg Prod.fst hij)

variable [Countable ι]

omit [Countable ι] in
private theorem partial_charFun (a : ι → ℝ) (s : Finset ι) (t : ℝ) :
    charFun ((coefficientLaw ι).map (fun ω => ∑ i ∈ s, a i * ω i)) t =
      Complex.exp (-((∑ i ∈ s, a i ^ 2 : ℝ) : ℂ) * (t : ℂ) ^ 2 / 2) := by
  classical
  have hi : iIndepFun (fun i (ω : ι → ℝ) => a i * ω i) (coefficientLaw ι) :=
    (coordinates_independent ι).comp (fun i y => a i * y) (fun _ => by fun_prop)
  have hm (i : ι) : Measurable (fun ω : ι → ℝ => a i * ω i) :=
    measurable_const.mul (measurable_pi_apply i)
  have hchar (i : ι) : charFun ((coefficientLaw ι).map
      (fun ω : ι → ℝ => a i * ω i)) t =
      Complex.exp (-((a i : ℂ) * (t : ℂ)) ^ 2 / 2) := by
    rw [charFun_map_mul_comp (f := fun ω : ι → ℝ => ω i)
      (measurable_pi_apply i).aemeasurable, coordinate_marginal, charFun_gaussianReal]
    simp only [Complex.ofReal_mul, mul_zero, Complex.ofReal_zero, zero_mul,
      NNReal.coe_one, Complex.ofReal_one, one_mul, zero_sub]
    congr 1
    ring
  rw [(hi.restrict s).charFun_map_fun_finsetSum_eq_prod (fun i _ => (hm i).aemeasurable)]
  simp only [Finset.prod_apply]
  simp_rw [hchar]
  rw [← Complex.exp_sum]
  congr 1
  push_cast
  simp only [mul_pow]
  rw [← Finset.sum_div, Finset.sum_neg_distrib, ← Finset.sum_mul]
  rw [neg_mul]

private theorem charFun_partial_tendsto (a : ι → ℝ) (ha : Summable a) (t : ℝ) :
    Filter.Tendsto
      (fun s : Finset ι => charFun ((coefficientLaw ι).map
        (fun ω => ∑ i ∈ s, a i * ω i)) t)
      Filter.atTop (𝓝 (charFun ((coefficientLaw ι).map (scalarSeries a)) t)) := by
  classical
  have hm (s : Finset ι) : Measurable (fun ω : ι → ℝ => ∑ i ∈ s, a i * ω i) :=
    Finset.measurable_sum s (fun i _ => measurable_const.mul (measurable_pi_apply i))
  have heq (s : Finset ι) :
      charFun ((coefficientLaw ι).map (fun ω => ∑ i ∈ s, a i * ω i)) t =
      ∫ ω, Complex.exp (((t * ∑ i ∈ s, a i * ω i : ℝ) : ℂ) * Complex.I)
        ∂coefficientLaw ι := by
    rw [charFun_apply_real, integral_map (hm s).aemeasurable (by fun_prop)]
    push_cast
    rfl
  have heq' : charFun ((coefficientLaw ι).map (scalarSeries a)) t =
      ∫ ω, Complex.exp (((t * scalarSeries a ω : ℝ) : ℂ) * Complex.I)
        ∂coefficientLaw ι := by
    rw [charFun_apply_real, integral_map (scalarSeries_measurable a).aemeasurable
      (by fun_prop)]
    push_cast
    rfl
  simp_rw [heq, heq']
  apply tendsto_integral_filter_of_norm_le_const
  · filter_upwards with s
    have hc : Continuous (fun q : ℝ => Complex.exp (((t * q : ℝ) : ℂ) * Complex.I)) :=
      by fun_prop
    exact (hc.measurable.comp (hm s)).aestronglyMeasurable
  · refine ⟨1, Filter.Eventually.of_forall (fun s => ?_)⟩
    filter_upwards with ω
    rw [Complex.norm_exp_ofReal_mul_I]
  · filter_upwards [scalarSeries_hasSum_ae a ha] with ω hω
    exact ((by fun_prop : Continuous (fun q : ℝ =>
      Complex.exp (((t * q : ℝ) : ℂ) * Complex.I))).tendsto _).comp hω

theorem scalarSeries_map_gaussian (a : ι → ℝ) (ha : Summable a) :
    (coefficientLaw ι).map (scalarSeries a) =
      gaussianReal 0 (∑' i, a i ^ 2).toNNReal := by
  apply Measure.ext_of_charFun
  funext t
  have h1 := charFun_partial_tendsto a ha t
  have h2 : Filter.Tendsto (fun s : Finset ι =>
      charFun ((coefficientLaw ι).map (fun ω => ∑ i ∈ s, a i * ω i)) t)
      Filter.atTop (𝓝 (Complex.exp (-((∑' i, a i ^ 2 : ℝ) : ℂ) * (t : ℂ)^2 / 2))) := by
    simp_rw [partial_charFun]
    exact ((by fun_prop : Continuous (fun q : ℝ =>
      Complex.exp (-(q : ℂ) * (t : ℂ)^2 / 2))).tendsto _).comp
      (summable_square a ha).hasSum
  rw [tendsto_nhds_unique h1 h2, charFun_gaussianReal]
  rw [Real.coe_toNNReal _ (tsum_nonneg (fun i => sq_nonneg (a i)))]
  congr 1
  push_cast
  ring

theorem scalarSeries_integral_zero (a : ι → ℝ) (ha : Summable a) :
    ∫ ω, scalarSeries a ω ∂coefficientLaw ι = 0 := by
  have h := integral_id_gaussianReal (μ := 0) (v := (∑' i, a i ^ 2).toNNReal)
  rw [← scalarSeries_map_gaussian a ha,
    integral_map (scalarSeries_measurable a).aemeasurable (by fun_prop)] at h
  exact h

theorem scalarSeries_variance (a : ι → ℝ) (ha : Summable a) :
    Var[scalarSeries a; coefficientLaw ι] = ∑' i, a i ^ 2 := by
  rw [← variance_id_map (scalarSeries_measurable a).aemeasurable,
    scalarSeries_map_gaussian a ha, variance_id_gaussianReal]
  exact Real.coe_toNNReal _ (tsum_nonneg (fun i => sq_nonneg (a i)))

theorem source_linearCombination_hasGaussianLaw (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (n : ℕ) (x : Fin n → (Fin d → ℝ)) (c : Fin n → ℝ) :
    HasGaussianLaw (fun ω => ∑ j, c j * sourceField d L ω (x j)) (sourceLaw d) := by
  classical
  have hc (j : Fin n) : Summable (fun m => modeCoefficient d L m (x j)) :=
    (modeMajorant_summable d hL).of_norm_bounded (fun m => by
      simpa only [Real.norm_eq_abs] using modeCoefficient_bound d hL m (x j))
  let a : SourceMode d → ℝ := fun m => ∑ j, c j * modeCoefficient d L m (x j)
  have ha : Summable a := by
    exact summable_sum (fun j _ => (hc j).mul_left (c j))
  have hG : HasGaussianLaw (scalarSeries a) (sourceLaw d) := by
    constructor
    · exact (scalarSeries_measurable a).aemeasurable
    · change IsGaussian ((coefficientLaw (SourceMode d)).map (scalarSeries a))
      rw [scalarSeries_map_gaussian a ha]
      infer_instance
  apply hG.congr
  filter_upwards [ae_all_iff.2 (fun j : Fin n =>
      scalarSeries_hasSum_ae (fun m => modeCoefficient d L m (x j)) (hc j)),
    scalarSeries_hasSum_ae a ha] with ω hω hsum
  have h : HasSum (fun m => ∑ j, c j * modeCoefficient d L m (x j) * ω m)
      (∑ j, c j * sourceField d L ω (x j)) := by
    simpa only [sourceField, mul_assoc] using hasSum_sum (fun j _ => (hω j).mul_left (c j))
  exact hsum.unique (by simpa only [a, Finset.sum_mul] using h)

theorem source_finiteVector_hasGaussianLaw (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (n : ℕ) (x : Fin n → (Fin d → ℝ)) :
    HasGaussianLaw (fun ω j => sourceField d L ω (x j)) (sourceLaw d) := by
  classical
  have hm : Measurable (fun ω j => sourceField d L ω (x j)) :=
    Measurable.of_eval (fun j => sourceField_eval_measurable d L (x j))
  refine ⟨hm.aemeasurable, ?_⟩
  apply isGaussian_of_map_eq_gaussianReal
  intro ell
  let c : Fin n → ℝ := fun j => ell (Pi.single j 1)
  have hrep (v : Fin n → ℝ) : ell v = ∑ j, c j * v j := by
    rw [← LinearMap.sum_single_apply (fun _ : Fin n => ℝ) v, map_sum]
    apply Finset.sum_congr rfl
    intro j _
    have hs : Pi.single j (v j) = v j • Pi.single j (1 : ℝ) := by
      rw [← Pi.single_smul, smul_eq_mul, mul_one]
    rw [hs, map_smul]
    simp [c, smul_eq_mul, mul_comm]
  have hg : HasGaussianLaw (fun ω => ell (fun j => sourceField d L ω (x j)))
      (sourceLaw d) :=
    (source_linearCombination_hasGaussianLaw d hL n x c).congr
      (Filter.Eventually.of_forall (fun ω => (hrep _).symm))
  refine ⟨(sourceLaw d)[fun ω => ell (fun j => sourceField d L ω (x j))],
    Var[fun ω => ell (fun j => sourceField d L ω (x j)); sourceLaw d].toNNReal, ?_⟩
  rw [Measure.map_map ell.continuous.measurable hm]
  exact hg.map_eq_gaussianReal

theorem sourceField_isGaussianProcess (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    IsGaussianProcess (fun x ω => sourceField d L ω x) (sourceLaw d) := by
  classical
  constructor
  intro I
  let e : ↥I ≃ Fin (Fintype.card ↥I) := Fintype.equivFin ↥I
  let x : Fin (Fintype.card ↥I) → (Fin d → ℝ) := fun j => (e.symm j).1
  let R : (Fin (Fintype.card ↥I) → ℝ) →L[ℝ] (↥I → ℝ) :=
    ContinuousLinearMap.pi (fun i => ContinuousLinearMap.proj (e i))
  apply ((source_finiteVector_hasGaussianLaw d hL _ x).map_fun R).congr
  filter_upwards with ω
  funext i
  simp [R, x]

end I4SourceGaussian
