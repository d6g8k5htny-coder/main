import SourceEndpointEstimates
import Mathlib.Probability.Independence.Process.Basic
import Mathlib.MeasureTheory.MeasurableSpace.Embedding

/-!
# Canonical conditioning on the actual smooth coefficient carrier

The literal coefficient shift preserves every source summability order. Its
pushforward kernel is specified for every pin value, including null pins.
Actual finite Gaussian coefficient laws establish residual-process independence;
no full-carrier conditional law or independence conclusion is assumed.

Source: SOURCE_SMOOTH_CONDITIONAL.md. OpenAI/Codex author-side proof work;
independent scientific review and geometric Palm identification are separate.
Carrier support coauthored by /root/normalizer_palm_research with
/root/normalizer_palm_research/gaussian_reference_probe, supplied as compiled
scratch CarrierShift.lean.
-/

noncomputable section

open I4Foundation I4Source I4Smooth I4SmoothCarrier I4SourceGaussian
open I4SourceDerivativeGaussian I4SourceDerivativeContinuity I4SourceConfluentLimits
open I4SourceEndpointRegression I4ContactGram MeasureTheory ProbabilityTheory Matrix
open scoped BigOperators
set_option autoImplicit false

namespace I4SourceSmoothConditional

def zeroSample (d : ℕ) (L : ℝ) : SmoothSample d L :=
  ⟨fun _ => 0, by
    intro n
    simpa only [Pi.zero_apply, abs_zero, mul_zero] using
      (summable_zero : Summable (fun _ : SourceMode d => (0 : ℝ)))⟩

theorem exists_measurable_sampleProjection (d : ℕ) (L : ℝ) :
    ∃ q : (SourceMode d → ℝ) → SmoothSample d L,
      Measurable q ∧ ∀ ω : SmoothSample d L, q (sampleInclusion d L ω) = ω := by
  exact (goodSet_measurable d L).exists_measurable_proj
    ⟨(zeroSample d L).val, (zeroSample d L).property⟩

theorem goodSample_add {d : ℕ} {L : ℝ}
    (ω η : SourceMode d → ℝ) (hω : goodSample d L ω) (hη : goodSample d L η) :
    goodSample d L (ω + η) := by
  intro n
  apply Summable.of_nonneg_of_le
    (fun m => mul_nonneg (orderModeBound_nonneg n d L m) (abs_nonneg _))
    (fun m => ?_) ((hω n).add (hη n))
  change orderModeBound n d L m * |ω m + η m| ≤
    orderModeBound n d L m * |ω m| + orderModeBound n d L m * |η m|
  rw [← mul_add]
  exact mul_le_mul_of_nonneg_left (abs_add_le _ _) (orderModeBound_nonneg n d L m)

theorem goodSample_neg {d : ℕ} {L : ℝ}
    (ω : SourceMode d → ℝ) (hω : goodSample d L ω) : goodSample d L (-ω) := by
  intro n
  simpa only [Pi.neg_apply, abs_neg] using hω n


theorem derivativeCoefficient_weighted_summable {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample 2 L) (k : ℕ) (x : Fin 2 → ℝ) (u : Fin k → (Fin 2 → ℝ)) :
    Summable (fun m => derivativeCoefficient 2 L k x u m * sampleInclusion 2 L ω m) := by
  apply ((ω.property k).mul_right (∏ j, ‖u j‖)).of_norm_bounded
  intro m
  rw [norm_mul, Real.norm_eq_abs]
  calc
    _ ≤ derivativeMajorant 2 L k u m * |sampleInclusion 2 L ω m| :=
      mul_le_mul_of_nonneg_right (derivativeCoefficient_bound 2 hL k u m x) (abs_nonneg _)
    _ = (orderModeBound k 2 L m * |sampleInclusion 2 L ω m|) * ∏ j, ‖u j‖ := by
      unfold derivativeMajorant
      ring

theorem finiteCoefficient_weighted_summable {ι : Type*} {n q : ℕ}
    (a : Fin n → ι → ℝ) (ω : ι → ℝ)
    (ha : ∀ i, Summable (fun m => a i m * ω m))
    (T : (Fin n → ℝ) →L[ℝ] (Fin q → ℝ)) (j : Fin q) :
    Summable (fun m => T (fun i => a i m) j * ω m) := by
  have hs : HasSum (fun m i => a i m * ω m) (fun i => ∑' m, a i m * ω m) :=
    Pi.hasSum.mpr (fun i => (ha i).hasSum)
  have ht := hs.map T.toAddMonoidHom T.continuous
  convert (Pi.hasSum.mp ht j).summable using 1
  funext m
  have hm : (fun i => a i m * ω m) = ω m • (fun i => a i m) := by
    funext i
    simp only [Pi.smul_apply, smul_eq_mul, mul_comm]
  change T (fun i => a i m) j * ω m = T (fun i => a i m * ω m) j
  rw [hm, map_smul]
  exact mul_comm _ _

theorem pinCoefficient_weighted_summable {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (ω : SmoothSample 2 L) (i : Fin 6) :
    Summable (fun m => pinCoefficient L r i m * sampleInclusion 2 L ω m) :=
  finiteCoefficient_weighted_summable (rawDerivativeCoefficient L r)
    (sampleInclusion 2 L ω)
    (fun j => derivativeCoefficient_weighted_summable hL ω (rawDerivativeOrder j)
      (rawDerivativePoint r j) (rawDerivativeDirections j)) (pinObservationMap r) i

theorem adjustedCoefficient_weighted_summable {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (ω : SmoothSample 2 L) (i : Fin 6) :
    Summable (fun m => adjustedCoefficient L r i m * sampleInclusion 2 L ω m) :=
  finiteCoefficient_weighted_summable (rawDerivativeCoefficient L r)
    (sampleInclusion 2 L ω)
    (fun j => derivativeCoefficient_weighted_summable hL ω (rawDerivativeOrder j)
      (rawDerivativePoint r j) (rawDerivativeDirections j)) (adjustedObservationMap r) i

theorem sourceConfluentPinsAt_measurable {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    Measurable (sourceConfluentPinsAt L r) := by
  have heq : sourceConfluentPinsAt L r =
      fun ω i => scalarSeries (pinCoefficient L r i) (sampleInclusion 2 L ω) := by
    funext ω
    exact sourceConfluentPinsAt_series hL r ω
  rw [heq]
  exact Measurable.of_eval (fun i =>
    (scalarSeries_measurable _).comp (sampleInclusion_measurable 2 L))

theorem sourceAdjustedEndpoints_measurable {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    Measurable (sourceAdjustedEndpoints L r) := by
  have heq : sourceAdjustedEndpoints L r =
      fun ω i => scalarSeries (adjustedCoefficient L r i) (sampleInclusion 2 L ω) := by
    funext ω
    exact sourceAdjustedEndpoints_series hL r ω
  rw [heq]
  exact Measurable.of_eval (fun i =>
    (scalarSeries_measurable _).comp (sampleInclusion_measurable 2 L))

def pinLift (L r : ℝ) (y : Fin 6 → ℝ) (m : SourceMode 2) : ℝ :=
  ∑ i, pinCoefficient L r i m * (((sourcePinCovarianceAt L r)⁻¹ *ᵥ y) i)

theorem pinLift_zero (L r : ℝ) : pinLift L r 0 = 0 := by
  funext m
  simp [pinLift]

theorem pinLift_add (L r : ℝ) (x y : Fin 6 → ℝ) :
    pinLift L r (x + y) = pinLift L r x + pinLift L r y := by
  funext m
  simp [pinLift, Matrix.mulVec_add, mul_add, Finset.sum_add_distrib]

theorem pinLift_neg (L r : ℝ) (x : Fin 6 → ℝ) :
    pinLift L r (-x) = -pinLift L r x := by
  funext m
  simp [pinLift, Matrix.mulVec_neg]

theorem pinLift_goodSample {L : ℝ} (hL : L ≠ 0) (r : ℝ) (y : Fin 6 → ℝ) :
    goodSample 2 L (pinLift L r y) := by
  intro n
  have hprod (i : Fin 6) : Summable (fun m =>
      orderModeBound n 2 L m * |pinCoefficient L r i m|) :=
    coefficient_product_summable (orderModeBound_summable n 2 hL)
      (pinCoefficient_summable hL r i).abs
  have hs : Summable (fun m => ∑ i,
      (orderModeBound n 2 L m * |pinCoefficient L r i m|) *
        |((sourcePinCovarianceAt L r)⁻¹ *ᵥ y) i|) :=
    summable_sum (fun i _ => (hprod i).mul_right _)
  apply Summable.of_nonneg_of_le (fun m =>
    mul_nonneg (orderModeBound_nonneg n 2 L m) (abs_nonneg _)) _ hs
  intro m
  calc
    _ ≤ orderModeBound n 2 L m *
        ∑ i, |pinCoefficient L r i m * (((sourcePinCovarianceAt L r)⁻¹ *ᵥ y) i)| :=
      mul_le_mul_of_nonneg_left (Finset.abs_sum_le_sum_abs _ _) (orderModeBound_nonneg n 2 L m)
    _ = _ := by simp only [abs_mul, Finset.mul_sum, mul_assoc]

theorem pinLift_measurable (L r : ℝ) : Measurable (pinLift L r) := by
  apply Measurable.of_eval
  intro m
  unfold pinLift
  fun_prop

def addLift {L : ℝ} (hL : L ≠ 0) (r : ℝ) (y : Fin 6 → ℝ)
    (ω : SmoothSample 2 L) : SmoothSample 2 L :=
  ⟨sampleInclusion 2 L ω + pinLift L r y,
    goodSample_add _ _ ω.property (pinLift_goodSample hL r y)⟩

@[simp] theorem addLift_val {L : ℝ} (hL : L ≠ 0) (r : ℝ) (y : Fin 6 → ℝ)
    (ω : SmoothSample 2 L) (m : SourceMode 2) :
    (addLift hL r y ω).val m = ω.val m + pinLift L r y m := rfl

theorem addLift_measurable {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    Measurable (fun p : (Fin 6 → ℝ) × SmoothSample 2 L => addLift hL r p.1 p.2) := by
  unfold addLift
  apply Measurable.subtype_mk
  exact ((sampleInclusion_measurable 2 L).comp measurable_snd).add
    ((pinLift_measurable L r).comp measurable_fst)

def posteriorSample {L : ℝ} (hL : L ≠ 0) (r : ℝ) (x : Fin 6 → ℝ)
    (ω : SmoothSample 2 L) : SmoothSample 2 L :=
  addLift hL r (x - sourceConfluentPinsAt L r ω) ω

@[simp] theorem posteriorSample_val {L : ℝ} (hL : L ≠ 0) (r : ℝ) (x : Fin 6 → ℝ)
    (ω : SmoothSample 2 L) (m : SourceMode 2) :
    (posteriorSample hL r x ω).val m = ω.val m +
      pinLift L r (x - sourceConfluentPinsAt L r ω) m := rfl

theorem posteriorSample_measurable {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    Measurable (fun p : (Fin 6 → ℝ) × SmoothSample 2 L => posteriorSample hL r p.1 p.2) := by
  exact (addLift_measurable hL r).comp
    ((measurable_fst.sub ((sourceConfluentPinsAt_measurable hL r).comp measurable_snd)).prodMk
      measurable_snd)

end I4SourceSmoothConditional

namespace I4SourceSmoothConditional

lemma scalarSeries_add_finiteShift {ι : Type*} {n : ℕ}
    (a : ι → ℝ) (b : Fin n → ι → ℝ) (ω : ι → ℝ) (c : Fin n → ℝ)
    (ha : Summable a) (hb : ∀ j, Summable (b j))
    (hω : Summable (fun m => a m * ω m)) :
    scalarSeries a (fun m => ω m + ∑ j, b j m * c j) =
      scalarSeries a ω + ∑ j, (∑' m, a m * b j m) * c j := by
  have hs : HasSum (fun m => ∑ j, (a m * b j m) * c j)
      (∑ j, (∑' m, a m * b j m) * c j) :=
    hasSum_sum (fun j _ => ((coefficient_product_summable ha (hb j)).hasSum).mul_right (c j))
  have ht := hω.hasSum.add hs
  change (∑' m, a m * (ω m + ∑ j, b j m * c j)) =
    (∑' m, a m * ω m) + ∑ j, (∑' m, a m * b j m) * c j
  convert ht.tsum_eq using 1
  apply tsum_congr
  intro m
  rw [mul_add, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  ring

end I4SourceSmoothConditional


namespace I4SourceSmoothConditional

theorem sourceConfluentPinsAt_addLift {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (y : Fin 6 → ℝ) (ω : SmoothSample 2 L) :
    sourceConfluentPinsAt L r (addLift hL r y ω) = sourceConfluentPinsAt L r ω +
      sourcePinCovarianceAt L r *ᵥ ((sourcePinCovarianceAt L r)⁻¹ *ᵥ y) := by
  ext i
  rw [sourceConfluentPinsAt_series hL r (addLift hL r y ω),
    sourceConfluentPinsAt_series hL r ω]
  change scalarSeries (pinCoefficient L r i)
      (fun m => sampleInclusion 2 L ω m + ∑ j,
        pinCoefficient L r j m * (((sourcePinCovarianceAt L r)⁻¹ *ᵥ y) j)) =
    scalarSeries (pinCoefficient L r i) (sampleInclusion 2 L ω) + _
  rw [scalarSeries_add_finiteShift _ _ _ _ (pinCoefficient_summable hL r i)
    (pinCoefficient_summable hL r) (pinCoefficient_weighted_summable hL r ω i)]
  simp_rw [← sourcePinCovarianceAt_coefficient hL r]
  rfl

theorem sourceAdjustedEndpoints_addLift {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (y : Fin 6 → ℝ) (ω : SmoothSample 2 L) :
    sourceAdjustedEndpoints L r (addLift hL r y ω) = sourceAdjustedEndpoints L r ω +
      sourceRegressionMapAt L r y := by
  ext i
  rw [sourceAdjustedEndpoints_series hL r (addLift hL r y ω),
    sourceAdjustedEndpoints_series hL r ω]
  change scalarSeries (adjustedCoefficient L r i)
      (fun m => sampleInclusion 2 L ω m + ∑ j,
        pinCoefficient L r j m * (((sourcePinCovarianceAt L r)⁻¹ *ᵥ y) j)) =
    scalarSeries (adjustedCoefficient L r i) (sampleInclusion 2 L ω) + _
  rw [scalarSeries_add_finiteShift _ _ _ _ (adjustedCoefficient_summable hL r i)
    (pinCoefficient_summable hL r) (adjustedCoefficient_weighted_summable hL r ω i)]
  simp_rw [← sourceCrossCovarianceAt_coefficient hL r]
  change _ = _ + ((sourceCrossCovarianceAt L r * (sourcePinCovarianceAt L r)⁻¹) *ᵥ y) i
  rw [← Matrix.mulVec_mulVec]
  rfl

theorem sourceConfluentPinsAt_posteriorSample {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (x : Fin 6 → ℝ)
    (ω : SmoothSample 2 L) :
    sourceConfluentPinsAt L r (posteriorSample hL r x ω) = x := by
  let _ : Invertible (sourcePinCovarianceAt L r) := hG.invertible
  rw [posteriorSample, sourceConfluentPinsAt_addLift hL r,
    Matrix.mulVec_mulVec, Matrix.mul_inv_of_invertible, Matrix.one_mulVec]
  abel

theorem sourceAdjustedEndpoints_posteriorSample {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (x : Fin 6 → ℝ) (ω : SmoothSample 2 L) :
    sourceAdjustedEndpoints L r (posteriorSample hL r x ω) =
      sourceAdjustedEndpoints L r ω +
        sourceRegressionMapAt L r (x - sourceConfluentPinsAt L r ω) :=
  sourceAdjustedEndpoints_addLift hL r _ ω

end I4SourceSmoothConditional

open I4SourceEndpointParameters I4SourceEndpointEstimates ResearchFormalCoreR1 Filter
open scoped Topology

namespace I4SourceSmoothConditional

/-- The literal coefficient-shift law, specified at every pin value. -/
def sourceSmoothKernel {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    Kernel (Fin 6 → ℝ) (SmoothSample 2 L) :=
  ((Kernel.id : Kernel (Fin 6 → ℝ) (Fin 6 → ℝ)) ×ₖ
    Kernel.const (Fin 6 → ℝ) (smoothLaw 2 L)).map
      (fun p => posteriorSample hL r p.1 p.2)

instance sourceSmoothKernel_isMarkov {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    IsMarkovKernel (sourceSmoothKernel hL r) := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  exact Kernel.IsMarkovKernel.map _ (posteriorSample_measurable hL r)

theorem sourceSmoothKernel_measurable {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    Measurable (sourceSmoothKernel hL r) := Kernel.measurable _

/-- Exact sampling formula, including every probability-zero pin. -/
theorem sourceSmoothKernel_apply {L : ℝ} (hL : L ≠ 0) (r : ℝ) (x : Fin 6 → ℝ) :
    sourceSmoothKernel hL r x = (smoothLaw 2 L).map (posteriorSample hL r x) := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  rw [sourceSmoothKernel, Kernel.map_apply _ (posteriorSample_measurable hL r),
    Kernel.prod_apply, Kernel.id_apply, Kernel.const_apply, Measure.dirac_prod,
    Measure.map_map (posteriorSample_measurable hL r) (by fun_prop)]
  rfl

theorem sourceSmoothKernel_pin_map {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (x : Fin 6 → ℝ) :
    (sourceSmoothKernel hL r x).map (sourceConfluentPinsAt L r) = Measure.dirac x := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  have hp : Measurable (posteriorSample hL r x) :=
    (posteriorSample_measurable hL r).comp (measurable_const.prodMk measurable_id)
  rw [sourceSmoothKernel_apply hL r x,
    Measure.map_map (sourceConfluentPinsAt_measurable hL r) hp]
  have heq : sourceConfluentPinsAt L r ∘ posteriorSample hL r x = fun _ => x := by
    funext ω
    exact sourceConfluentPinsAt_posteriorSample hL r hG x ω
  rw [heq]
  simp

theorem sourceSmoothKernel_pin_fiber {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (x : Fin 6 → ℝ) :
    sourceSmoothKernel hL r x {ω | sourceConfluentPinsAt L r ω = x} = 1 := by
  calc
    _ = ((sourceSmoothKernel hL r x).map (sourceConfluentPinsAt L r)) {x} := by
      rw [Measure.map_apply (sourceConfluentPinsAt_measurable hL r)
        (measurableSet_singleton x)]
      rfl
    _ = 1 := by rw [sourceSmoothKernel_pin_map hL r hG x]; simp

/-- The actual endpoint kernel is the marginal of the full coefficient shift
at every pin, including prescribed null pins. -/
theorem sourceSmoothKernel_endpoint_map {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (x : Fin 6 → ℝ) :
    (sourceSmoothKernel hL r x).map (sourceAdjustedEndpoints L r) =
      sourceEndpointKernel L r x := by
  have hp : Measurable (posteriorSample hL r x) :=
    (posteriorSample_measurable hL r).comp (measurable_const.prodMk measurable_id)
  rw [sourceSmoothKernel_apply hL r x,
    Measure.map_map (sourceAdjustedEndpoints_measurable hL r) hp,
    sourceEndpointKernel_eq_map_posterior hL r x]
  congr 1
  funext ω
  change sourceAdjustedEndpoints L r (posteriorSample hL r x ω) =
    sourceEndpointResidual L r ω + sourceRegressionMapAt L r x
  rw [sourceAdjustedEndpoints_posteriorSample hL r x ω]
  simp only [sourceEndpointResidual, map_sub]
  abel

end I4SourceSmoothConditional

/- Finite-Gaussian and process-independence support coauthored by
/root/regression_kernel, /root/regression_kernel/endpoint_means, and
/root/regression_kernel/gaussian_api, supplied as compiled scratch
ProcessSupport.lean; coauthorship only, no scientific-review credit. -/

namespace I4SourceSmoothConditional

theorem scalarSeries_finite_hasGaussianLaw {ι τ : Type*} [Countable ι] [Fintype τ]
    (a : τ → ι → ℝ) (ha : ∀ j, Summable (a j)) :
    HasGaussianLaw (fun η j => scalarSeries (a j) η) (coefficientLaw ι) := by
  classical
  let e : τ ≃ Fin (Fintype.card τ) := Fintype.equivFin τ
  let reindex : (Fin (Fintype.card τ) → ℝ) →L[ℝ] (τ → ℝ) :=
    ContinuousLinearMap.pi (fun j => ContinuousLinearMap.proj (e j))
  have hG := (scalarSeries_finiteVector_hasGaussianLaw (Fintype.card τ)
    (fun j => a (e.symm j)) (fun j => ha (e.symm j))).map_fun reindex
  simpa [reindex] using hG

def residualCoefficient (L r : ℝ) (ω : SmoothSample 2 L) (m : SourceMode 2) : ℝ :=
  ω.val m - pinLift L r (sourceConfluentPinsAt L r ω) m

lemma scalarSeries_single {ι : Type*} [DecidableEq ι] (m : ι) (η : ι → ℝ) :
    scalarSeries (Pi.single m (1 : ℝ)) η = η m := by
  simp [scalarSeries, Pi.single_apply]

lemma scalarSeries_coordinate_covariance {ι : Type*} [Countable ι]
    (a : ι → ℝ) (ha : Summable a) (m : ι) :
    covariance (scalarSeries a) (fun ω : ι → ℝ => ω m) (coefficientLaw ι) = a m := by
  classical
  have hd : Summable (Pi.single m (1 : ℝ)) := by
    apply summable_of_ne_finset_zero (s := {m})
    intro n hn
    simp only [Finset.mem_singleton] at hn
    simp [hn]
  have heq : (fun ω : ι → ℝ => ω m) = scalarSeries (Pi.single m 1) := by
    funext ω
    exact (scalarSeries_single m ω).symm
  rw [heq, I4SourceCovariance.scalarSeries_covariance _ _ ha hd]
  simp [Pi.single_apply, mul_ite]

theorem pin_coefficient_covariance {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (i : Fin 6) (m : SourceMode 2) :
    covariance (fun ω => sourceConfluentPinsAt L r ω i)
      (fun ω : SmoothSample 2 L => ω.val m) (smoothLaw 2 L) = pinCoefficient L r i m := by
  simp_rw [sourceConfluentPinsAt_series hL]
  change covariance
    (fun ω => scalarSeries (pinCoefficient L r i) (sampleInclusion 2 L ω))
    (fun ω => (sampleInclusion 2 L ω) m) (smoothLaw 2 L) = _
  rw [covariance_comp_inclusion 2 hL _ _ (scalarSeries_measurable _)
    (measurable_pi_apply m)]
  exact scalarSeries_coordinate_covariance _ (pinCoefficient_summable hL r i) m


theorem pin_finite_coefficients_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (I : Finset (SourceMode 2)) :
    HasGaussianLaw (fun ω : SmoothSample 2 L =>
      (sourceConfluentPinsAt L r ω, fun m : I => ω.val m)) (smoothLaw 2 L) := by
  classical
  let q : (Fin 6 ⊕ I) → SourceMode 2 → ℝ :=
    Sum.elim (pinCoefficient L r) (fun m => Pi.single m.val 1)
  have hq (j : Fin 6 ⊕ I) : Summable (q j) := by
    cases j with
    | inl i => exact pinCoefficient_summable hL r i
    | inr m =>
      apply summable_of_ne_finset_zero (s := {m.val})
      intro n hn
      simp only [Finset.mem_singleton] at hn
      simp [q, hn]
  have hG := scalarSeries_finite_hasGaussianLaw q hq
  have hm : Measurable (fun η j => scalarSeries (q j) η) :=
    Measurable.of_eval (fun j => scalarSeries_measurable (q j))
  have hμ : HasGaussianLaw (fun ω : SmoothSample 2 L =>
      fun j => scalarSeries (q j) (sampleInclusion 2 L ω)) (smoothLaw 2 L) := by
    refine ⟨(hm.comp (sampleInclusion_measurable 2 L)).aemeasurable, ?_⟩
    rw [observation_map_eq 2 hL _ hm]
    exact hG.isGaussian_map
  let T : ((Fin 6 ⊕ I) → ℝ) →L[ℝ] ((Fin 6 → ℝ) × (I → ℝ)) :=
    (ContinuousLinearMap.pi (fun i => ContinuousLinearMap.proj (Sum.inl i))).prod
      (ContinuousLinearMap.pi (fun m => ContinuousLinearMap.proj (Sum.inr m)))
  apply (hμ.map_fun T).congr
  filter_upwards with ω
  apply Prod.ext
  · funext i
    change scalarSeries (pinCoefficient L r i) (sampleInclusion 2 L ω) =
      sourceConfluentPinsAt L r ω i
    exact (congrFun (sourceConfluentPinsAt_series hL r ω) i).symm
  · funext m
    exact scalarSeries_single m.val ω.val

def finiteLift (L r : ℝ) (I : Finset (SourceMode 2)) :
    (Fin 6 → ℝ) →L[ℝ] (I → ℝ) :=
  (Matrix.toLin' ((Matrix.of (fun (m : I) i => pinCoefficient L r i m.val)) *
    (sourcePinCovarianceAt L r)⁻¹)).toContinuousLinearMap

lemma finiteLift_apply (L r : ℝ) (I : Finset (SourceMode 2)) (y : Fin 6 → ℝ) (m : I) :
    finiteLift L r I y m = pinLift L r y m.val := by
  change (((Matrix.of (fun (m : I) i => pinCoefficient L r i m.val)) *
    (sourcePinCovarianceAt L r)⁻¹) *ᵥ y) m = _
  rw [← Matrix.mulVec_mulVec]
  rfl

theorem pin_residual_finite_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (I : Finset (SourceMode 2)) :
    HasGaussianLaw (fun ω : SmoothSample 2 L =>
      (sourceConfluentPinsAt L r ω, fun m : I => residualCoefficient L r ω m.val))
      (smoothLaw 2 L) := by
  let T : ((Fin 6 → ℝ) × (I → ℝ)) →L[ℝ] ((Fin 6 → ℝ) × (I → ℝ)) :=
    (ContinuousLinearMap.fst ℝ (Fin 6 → ℝ) (I → ℝ)).prod
      (ContinuousLinearMap.snd ℝ (Fin 6 → ℝ) (I → ℝ) -
        (finiteLift L r I).comp (ContinuousLinearMap.fst ℝ (Fin 6 → ℝ) (I → ℝ)))
  apply ((pin_finite_coefficients_hasGaussianLaw hL r I).map_fun T).congr
  filter_upwards with ω
  apply Prod.ext
  · rfl
  · funext m
    change ω.val m - finiteLift L r I (sourceConfluentPinsAt L r ω) m = _
    rw [finiteLift_apply]
    rfl

lemma coefficient_hasGaussianLaw {L : ℝ} (hL : L ≠ 0) (m : SourceMode 2) :
    HasGaussianLaw (fun ω : SmoothSample 2 L => ω.val m) (smoothLaw 2 L) := by
  refine ⟨((measurable_pi_apply m).comp measurable_subtype_coe).aemeasurable, ?_⟩
  change IsGaussian ((smoothLaw 2 L).map (fun ω => sampleInclusion 2 L ω m))
  rw [I4SmoothCarrier.coordinate_marginal 2 hL m]
  infer_instance

theorem residualCoefficient_zero_covariance {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (i : Fin 6) (m : SourceMode 2) :
    covariance (fun ω => sourceConfluentPinsAt L r ω i)
      (fun ω => residualCoefficient L r ω m) (smoothLaw 2 L) = 0 := by
  classical
  have hX := (sourcePins_adjusted_hasGaussianLaw hL r).fst
  have : IsProbabilityMeasure (smoothLaw 2 L) := hX.isProbabilityMeasure
  let I : Finset (SourceMode 2) := {m}
  let mm : I := ⟨m, by simp [I]⟩
  let C : Matrix I (Fin 6) ℝ := fun t j => pinCoefficient L r j t.val
  let G := sourcePinCovarianceAt L r
  let _ : Invertible G := hG.invertible
  have hprod : C * G⁻¹ * G = C := Matrix.inv_mul_cancel_right_of_invertible G C
  have hXi (l : Fin 6) : MemLp (fun ω => sourceConfluentPinsAt L r ω l) 2 (smoothLaw 2 L) :=
    (hX.eval l).memLp_two
  have hη : MemLp (fun ω : SmoothSample 2 L => ω.val m) 2 (smoothLaw 2 L) :=
    (coefficient_hasGaussianLaw hL m).memLp_two
  have hpred : MemLp (fun ω => pinLift L r (sourceConfluentPinsAt L r ω) m)
      2 (smoothLaw 2 L) := by
    have hh := ((hX.map_fun (finiteLift L r I)).eval mm).memLp_two
    simpa only [finiteLift_apply] using hh
  have heq : (fun ω => pinLift L r (sourceConfluentPinsAt L r ω) m) =
      (fun ω => ∑ l, (C * G⁻¹) mm l * sourceConfluentPinsAt L r ω l) := by
    funext ω
    exact (finiteLift_apply L r I (sourceConfluentPinsAt L r ω) mm).symm
  have hcov : covariance (fun ω => sourceConfluentPinsAt L r ω i)
      (fun ω => pinLift L r (sourceConfluentPinsAt L r ω) m) (smoothLaw 2 L) =
      pinCoefficient L r i m := by
    rw [heq, covariance_fun_sum_right (fun l => (hXi l).const_mul _) (hXi i)]
    simp_rw [covariance_const_mul_right]
    calc
      _ = ∑ l, (C * G⁻¹) mm l * G l i := by
        apply Finset.sum_congr rfl
        intro l _
        congr 1
        exact covariance_comm _ _
      _ = C mm i := congrArg (fun N => N mm i) hprod
      _ = pinCoefficient L r i m := rfl
  unfold residualCoefficient
  rw [covariance_fun_sub_right (hXi i) hη hpred, pin_coefficient_covariance hL r i m,
    hcov, sub_self]

theorem residualCoefficient_finite_indepFun {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (I : Finset (SourceMode 2)) :
    IndepFun (sourceConfluentPinsAt L r)
      (fun (ω : SmoothSample 2 L) (m : I) => residualCoefficient L r ω m.val) (smoothLaw 2 L) := by
  exact (pin_residual_finite_hasGaussianLaw hL r I).indepFun_of_covariance_eval
    (fun i m => residualCoefficient_zero_covariance hL r hG i m.val)

theorem residualCoefficient_indepFun {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    IndepFun (sourceConfluentPinsAt L r) (fun ω m => residualCoefficient L r ω m)
      (smoothLaw 2 L) := by
  have hX := (sourcePins_adjusted_hasGaussianLaw hL r).fst
  have : IsProbabilityMeasure (smoothLaw 2 L) := hX.isProbabilityMeasure
  apply IndepFun.indepFun_process₀ hX.aemeasurable
  · intro m
    have hφ : Measurable (fun y : Fin 6 → ℝ => pinLift L r y m) := by
      unfold pinLift
      fun_prop
    exact ((measurable_pi_apply m).comp measurable_subtype_coe).aemeasurable.sub
      (hφ.comp_aemeasurable hX.aemeasurable)
  · exact residualCoefficient_finite_indepFun hL r hG

end I4SourceSmoothConditional

open I4SourcePhysicalEndpointKernel

namespace I4SourceSmoothConditional

theorem sourceNormalizedEndpoints_measurable {L : ℝ} (hL : L ≠ 0) (r : ℝ) :
    Measurable (sourceNormalizedEndpoints L r) := by
  have heq : sourceNormalizedEndpoints L r = fun ω => sourceAdjustedEndpoints L r ω +
      sourcePhysicalCorrection r (sourceConfluentPinsAt L r ω) :=
    funext (sourceNormalizedEndpoints_eq_adjusted L r)
  rw [heq]
  exact (sourceAdjustedEndpoints_measurable hL r).add
    ((sourcePhysicalCorrection r).continuous.measurable.comp
      (sourceConfluentPinsAt_measurable hL r))

/-- The literal physical endpoint kernel is also an all-pin marginal of the
whole-carrier conditional sampling law. -/
theorem sourceSmoothKernel_physical_endpoint_map {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (x : Fin 6 → ℝ) :
    (sourceSmoothKernel hL r x).map (sourceNormalizedEndpoints L r) =
      sourcePhysicalEndpointKernel L r x := by
  have hp : Measurable (posteriorSample hL r x) :=
    (posteriorSample_measurable hL r).comp (measurable_const.prodMk measurable_id)
  have hR : Measurable (sourceEndpointResidual L r) :=
    (sourceAdjustedEndpoints_measurable hL r).sub
      ((sourceRegressionMapAt L r).continuous.measurable.comp
        (sourceConfluentPinsAt_measurable hL r))
  rw [sourceSmoothKernel_apply hL r x,
    Measure.map_map (sourceNormalizedEndpoints_measurable hL r) hp,
    sourcePhysicalEndpointKernel_apply hL r x,
    Measure.map_map (by fun_prop) hR]
  congr 1
  funext ω
  change sourceNormalizedEndpoints L r (posteriorSample hL r x ω) =
    sourceEndpointResidual L r ω + sourceRegressionMapAt L r x + sourcePhysicalCorrection r x
  rw [sourceNormalizedEndpoints_eq_adjusted,
    sourceConfluentPinsAt_posteriorSample hL r hG x ω,
    sourceAdjustedEndpoints_posteriorSample hL r x ω]
  simp only [sourceEndpointResidual, map_sub]
  abel

/-- Prescribed null pins select exactly the physical endpoint law already used
by the completed endpoint estimates. -/
theorem sourceSmoothKernel_physical_prescribed_map {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    (sourceSmoothKernel hL r (sourcePrescribedPins r b)).map (sourceNormalizedEndpoints L r) =
      sourcePhysicalEndpointLaw L r b :=
  sourceSmoothKernel_physical_endpoint_map hL r hG (sourcePrescribedPins r b)

end I4SourceSmoothConditional

/- Measurable carrier reconstruction and final conditional-law join
coauthored by /root/normalizer_palm_research, supplied as compiled scratch
Reconstruction.lean. Coauthor work, no scientific-review credit. -/

namespace I4SourceSmoothConditional

theorem posteriorSample_self {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (ω : SmoothSample 2 L) :
    posteriorSample hL r (sourceConfluentPinsAt L r ω) ω = ω := by
  apply Subtype.ext
  funext m
  simp only [posteriorSample_val, sub_self, pinLift_zero, Pi.zero_apply, add_zero]

theorem posteriorSample_eq_addLift_residual {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (x : Fin 6 → ℝ) (ω : SmoothSample 2 L) :
    addLift hL r x (posteriorSample hL r 0 ω) = posteriorSample hL r x ω := by
  apply Subtype.ext
  funext m
  simp only [addLift_val, posteriorSample_val, sub_eq_add_neg,
    pinLift_add, pinLift_neg, pinLift_zero, Pi.zero_apply, Pi.add_apply, Pi.neg_apply]
  abel

theorem addLift_observation_residual {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (ω : SmoothSample 2 L) :
    addLift hL r (sourceConfluentPinsAt L r ω) (posteriorSample hL r 0 ω) = ω := by
  rw [posteriorSample_eq_addLift_residual, posteriorSample_self]

theorem posteriorSample_zero_inclusion {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (ω : SmoothSample 2 L) :
    sampleInclusion 2 L (posteriorSample hL r 0 ω) = residualCoefficient L r ω := by
  funext m
  change (posteriorSample hL r 0 ω).val m = residualCoefficient L r ω m
  simp only [posteriorSample_val, residualCoefficient, sub_eq_add_neg, zero_add,
    pinLift_neg, Pi.neg_apply]

theorem posteriorSample_zero_indepFun {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    IndepFun (sourceConfluentPinsAt L r) (posteriorSample hL r 0) (smoothLaw 2 L) := by
  obtain ⟨q,hq,hret⟩ := exists_measurable_sampleProjection 2 L
  have hi := (residualCoefficient_indepFun hL r hG).comp measurable_id hq
  have heq : q ∘ (fun ω => residualCoefficient L r ω) = posteriorSample hL r 0 := by
    funext ω
    rw [Function.comp_apply, ← posteriorSample_zero_inclusion hL r ω, hret]
  simpa only [Function.id_comp, heq] using hi

theorem sourceSmoothKernel_hasCondDistrib {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    HasCondDistrib (id : SmoothSample 2 L → SmoothSample 2 L)
      (sourceConfluentPinsAt L r) (sourceSmoothKernel hL r) (smoothLaw 2 L) := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  have hX := sourceConfluentPinsAt_measurable hL r
  have hR : Measurable (posteriorSample hL r 0) :=
    (posteriorSample_measurable hL r).comp (measurable_const.prodMk measurable_id)
  have hc := (regression_pair_hasCondDistrib hX.aemeasurable hR.aemeasurable
    (posteriorSample_zero_indepFun hL r hG)).comp_left (addLift_measurable hL r)
  have hrec : (fun p : (Fin 6 → ℝ) × SmoothSample 2 L => addLift hL r p.1 p.2) ∘
      (fun ω => (sourceConfluentPinsAt L r ω, posteriorSample hL r 0 ω)) = id := by
    funext ω
    exact addLift_observation_residual hL r ω
  have hk : ((Kernel.id : Kernel (Fin 6 → ℝ) (Fin 6 → ℝ)) ×ₖ
      Kernel.const (Fin 6 → ℝ) ((smoothLaw 2 L).map (posteriorSample hL r 0))).map
        (fun p => addLift hL r p.1 p.2) = sourceSmoothKernel hL r := by
    ext x : 1
    rw [Kernel.map_apply _ (addLift_measurable hL r), Kernel.prod_apply,
      Kernel.id_apply, Kernel.const_apply, Measure.dirac_prod,
      Measure.map_map (addLift_measurable hL r) (by fun_prop), sourceSmoothKernel_apply hL r x]
    have ha : Measurable (addLift hL r x) :=
      (addLift_measurable hL r).comp (measurable_const.prodMk measurable_id)
    change (((smoothLaw 2 L).map (posteriorSample hL r 0)).map (addLift hL r x)) = _
    rw [Measure.map_map ha hR]
    congr 1
    funext ω
    exact posteriorSample_eq_addLift_residual hL r x ω
  rw [hrec, hk] at hc
  exact hc

end I4SourceSmoothConditional

namespace I4SourceSmoothConditional

/-- Every measurable functional of the actual carrier inherits this specified
conditional kernel. -/
theorem sourceSmoothKernel_map_hasCondDistrib {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) {E : Type*} [MeasurableSpace E]
    (F : SmoothSample 2 L → E) (hF : Measurable F) :
    HasCondDistrib F (sourceConfluentPinsAt L r) ((sourceSmoothKernel hL r).map F)
      (smoothLaw 2 L) := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  simpa only [Function.comp_id] using
    (sourceSmoothKernel_hasCondDistrib hL r hG).comp_left hF

/-- The literal smooth field is measurable for the coordinate product sigma
algebra on functions. No additional topological function-space law is chosen. -/
theorem smoothSourceField_measurable (d : ℕ) (L : ℝ) :
    Measurable (smoothSourceField d L) := by
  apply Measurable.of_eval
  intro x
  exact (sourceField_eval_measurable d L x).comp (sampleInclusion_measurable d L)

theorem sourceSmoothKernel_field_hasCondDistrib {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    HasCondDistrib (smoothSourceField 2 L) (sourceConfluentPinsAt L r)
      ((sourceSmoothKernel hL r).map (smoothSourceField 2 L)) (smoothLaw 2 L) :=
  sourceSmoothKernel_map_hasCondDistrib hL r hG _ (smoothSourceField_measurable 2 L)

/-- The completed actual covariance limits supply invertibility. The final
whole-carrier conditioning statement retains only the actual scale hypothesis. -/
theorem sourceSmoothKernel_eventually_hasCondDistrib {L : ℝ} (hL : L ≠ 0) :
    ∀ᶠ r in 𝓝[>] (0 : ℝ),
      HasCondDistrib (id : SmoothSample 2 L → SmoothSample 2 L)
        (sourceConfluentPinsAt L r) (sourceSmoothKernel hL r) (smoothLaw 2 L) := by
  filter_upwards [(sourceEndpoint_parameter_limits hL 0).1] with r hG
  exact sourceSmoothKernel_hasCondDistrib hL r hG

theorem sourceSmoothKernel_eventually_map_hasCondDistrib {L : ℝ} (hL : L ≠ 0)
    {E : Type*} [MeasurableSpace E] (F : SmoothSample 2 L → E) (hF : Measurable F) :
    ∀ᶠ r in 𝓝[>] (0 : ℝ),
      HasCondDistrib F (sourceConfluentPinsAt L r) ((sourceSmoothKernel hL r).map F)
        (smoothLaw 2 L) := by
  filter_upwards [(sourceEndpoint_parameter_limits hL 0).1] with r hG
  exact sourceSmoothKernel_map_hasCondDistrib hL r hG F hF

end I4SourceSmoothConditional
