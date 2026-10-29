import SourceEndpointParameters
import SourceConfluentLimits
import SourceContactEndpointParameters
import SourcePhysicalEndpointKernel
import GaussianRegressionLimits
import ResearchFormalCoreR1.GaussianNormalizerLimit
import ResearchFormalCoreR1.PalmReweighting

/-!
# Actual canonical endpoint estimates and reweighting

The LM006 endpoint vector is explicitly converted to the literal physical
maximum/saddle determinant weight. These composition lemmas retain the actual
canonical all-pin kernel and its derived Gaussian parameters.
See SOURCE_ENDPOINT_ESTIMATES.md for the frozen source contract.
-/

noncomputable section

open I4SmoothCarrier I4SourceConfluentPins I4SourceEndpointRegression
open I4SourceEndpointParameters ResearchFormalCoreR1
open MeasureTheory ProbabilityTheory Matrix WithLp Filter
open scoped Topology NNReal

set_option autoImplicit false

namespace I4SourceEndpointEstimates

/-- LM006 `(aM,aS,bM,bS,cM,cS)` to actual `(aM,bM,cM,aS,bS,cS)` order. -/
def sourcePhysicalCoordinates (v : Fin 6 → ℝ) : Fin 6 → ℝ :=
  ![v 0,v 2,v 4,v 1,v 3,v 5]

def sourcePhysicalIndex : Fin 6 → Fin 6 := ![0,2,4,1,3,5]

theorem sourcePhysicalCoordinates_apply (v : Fin 6 → ℝ) (i : Fin 6) :
    sourcePhysicalCoordinates v i = v (sourcePhysicalIndex i) := by
  fin_cases i <;> rfl

theorem sourcePhysicalCoordinates_continuous : Continuous sourcePhysicalCoordinates := by
  unfold sourcePhysicalCoordinates
  fun_prop

theorem sourcePhysicalCoordinates_inverse (v : Fin 6 → ℝ) :
    enb_permutation (sourcePhysicalCoordinates v) = v := by
  ext i
  fin_cases i <;> rfl

/-- The literal strict typed physical endpoint determinant weight. -/
def sourceEndpointWeight (r : ℝ) (v : Fin 6 → ℝ) : ℝ :=
  egm_weight r (sourcePhysicalCoordinates v)

theorem sourceEndpointWeight_measurable (r : ℝ) : Measurable (sourceEndpointWeight r) :=
  (egm_weight_measurable r).comp sourcePhysicalCoordinates_continuous.measurable

theorem sourceEndpointWeight_nonneg (r : ℝ) (v : Fin 6 → ℝ) :
    0 ≤ sourceEndpointWeight r v := by
  unfold sourceEndpointWeight egm_weight
  split_ifs
  · exact abs_nonneg _
  · exact le_rfl

theorem sourceEndpointWeight_eq_continuous (r : ℝ) (hr : 0 < r) (v : Fin 6 → ℝ) :
    sourceEndpointWeight r v = r^2 * gn_normalizedWeight r v := by
  simpa only [sourceEndpointWeight, sourcePhysicalCoordinates_inverse] using
    enb_weight_identity r (sourcePhysicalCoordinates v) hr

def sourceEndpointLaw (L r b : ℝ) : Measure (Fin 6 → ℝ) :=
  sourceEndpointKernel L r (sourcePrescribedPins r b)

def sourceEndpointMean (L r b : ℝ) : Fin 6 → ℝ :=
  sourceRegressionMapAt L r (sourcePrescribedPins r b)

theorem sourceEndpointLaw_isProbabilityMeasure {L : ℝ} (hL : L ≠ 0) (r b : ℝ) :
    IsProbabilityMeasure (sourceEndpointLaw L r b) := by
  unfold sourceEndpointLaw
  have : IsMarkovKernel (sourceEndpointKernel L r) := sourceEndpointKernel_markov hL r
  exact inferInstance

/-- The actual canonical every-pin kernel has the derived finite Gaussian law. -/
theorem sourceEndpointLaw_euclidean_hasLaw {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    HasLaw (toLp 2) (multivariateGaussian (toLp 2 (sourceEndpointMean L r b))
      (sourceEndpointSchurAt L r)) (sourceEndpointLaw L r b) where
  aemeasurable := by fun_prop
  map_eq := sourceEndpointKernel_eq_multivariateGaussian hL r hG (sourcePrescribedPins r b)

/-- Every physical coordinate marginal is derived from the actual kernel,
including singular residual covariances and noncentered prescribed heights. -/
theorem sourceEndpointLaw_physical_marginal {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) (i : Fin 6) :
    HasLaw (fun v => sourcePhysicalCoordinates v i)
      (gaussianReal (sourceEndpointMean L r b (sourcePhysicalIndex i))
        ((sourceEndpointSchurAt L r) (sourcePhysicalIndex i) (sourcePhysicalIndex i)).toNNReal)
      (sourceEndpointLaw L r b) := by
  have hPSD := sourceEndpointSchurAt_posSemidef hL r hG
  have h := (measurePreserving_eval_multivariateGaussian
    (μ := toLp 2 (sourceEndpointMean L r b)) hPSD (i := sourcePhysicalIndex i)).hasLaw.fun_comp
      (sourceEndpointLaw_euclidean_hasLaw hL r b hG)
  exact h.congr (Eventually.of_forall (fun v => sourcePhysicalCoordinates_apply v i))

/-- Integrals of the literal physical weight transfer to the exact finite
Gaussian parameter law without an assumed integral representation. -/
theorem sourceEndpointLaw_weight_integral {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    (∫ v, sourceEndpointWeight r v ∂sourceEndpointLaw L r b) =
      ∫ x : EuclideanSpace ℝ (Fin 6), egm_weight r ![x 0,x 2,x 4,x 1,x 3,x 5]
        ∂multivariateGaussian (toLp 2 (sourceEndpointMean L r b)) (sourceEndpointSchurAt L r) := by
  have h := (sourceEndpointLaw_euclidean_hasLaw hL r b hG).integral_comp
    (f := fun x : EuclideanSpace ℝ (Fin 6) => egm_weight r ![x 0,x 2,x 4,x 1,x 3,x 5])
    (show AEStronglyMeasurable
      (fun x : EuclideanSpace ℝ (Fin 6) => egm_weight r ![x 0,x 2,x 4,x 1,x 3,x 5])
      (multivariateGaussian (toLp 2 (sourceEndpointMean L r b)) (sourceEndpointSchurAt L r)) by
      exact ((egm_weight_measurable r).comp (by fun_prop)).aestronglyMeasurable)
  simpa only [sourceEndpointWeight, sourcePhysicalCoordinates, Function.comp_def,
    ofLp_toLp] using h

/-- Per-radius moment composition from explicit parameter bounds. The later
actual-source family theorem derives these bounds from the actual block limits. -/
theorem sourceEndpointLaw_second_moment_of_parameter_bounds {L : ℝ} (hL : L ≠ 0)
    (r b B V : ℝ) (hG : IsUnit (sourcePinCovarianceAt L r))
    (hB : 0 ≤ B) (hV : 0 ≤ V) (hr : 0 < r) (hr1 : r ≤ 1)
    (hmean : ∀ i, |sourceEndpointMean L r b i| ≤ B)
    (hvar : ∀ i, |sourceEndpointSchurAt L r i i| ≤ V) :
    Integrable (fun v => sourceEndpointWeight r v ^ 2) (sourceEndpointLaw L r b) ∧
      (∫ v, sourceEndpointWeight r v ^ 2 ∂sourceEndpointLaw L r b) ≤
        16 * gm8FiniteM8 6 B V * r ^ 4 := by
  have := sourceEndpointLaw_isProbabilityMeasure hL r b
  have hPSD := sourceEndpointSchurAt_posSemidef hL r hG
  apply egm_gaussian_moment (sourceEndpointLaw L r b) sourcePhysicalCoordinates
    sourcePhysicalCoordinates_continuous.measurable
    (fun i => sourceEndpointMean L r b (sourcePhysicalIndex i))
    (fun i => ((sourceEndpointSchurAt L r) (sourcePhysicalIndex i) (sourcePhysicalIndex i)).toNNReal)
    B V r hB hV hr hr1
  · exact fun i => hmean (sourcePhysicalIndex i)
  · intro i
    rw [Real.coe_toNNReal _ hPSD.diag_nonneg]
    exact (le_abs_self _).trans (hvar (sourcePhysicalIndex i))
  · exact sourceEndpointLaw_physical_marginal hL r b hG

/-- Actual physical square integrability needs no supplied parameter bound.
Finite current Gaussian parameters provide their own finite coordinate bounds. -/
theorem sourceEndpointLaw_weight_square_integrable {L : ℝ} (hL : L ≠ 0)
    (r b : ℝ) (hG : IsUnit (sourcePinCovarianceAt L r)) (hr : 0 < r) (hr1 : r ≤ 1) :
    Integrable (fun v => sourceEndpointWeight r v ^ 2) (sourceEndpointLaw L r b) := by
  apply (sourceEndpointLaw_second_moment_of_parameter_bounds hL r b
    ‖sourceEndpointMean L r b‖ ‖fun i => sourceEndpointSchurAt L r i i‖
    hG (norm_nonneg _) (norm_nonneg _) hr hr1 ?_ ?_).1
  · intro i
    simpa only [Real.norm_eq_abs] using norm_le_pi_norm (sourceEndpointMean L r b) i
  · intro i
    simpa only [Real.norm_eq_abs] using
      norm_le_pi_norm (fun i => sourceEndpointSchurAt L r i i) i

theorem sourceEndpointLaw_weight_integrable {L : ℝ} (hL : L ≠ 0)
    (r b : ℝ) (hG : IsUnit (sourcePinCovarianceAt L r)) (hr : 0 < r) (hr1 : r ≤ 1) :
    Integrable (sourceEndpointWeight r) (sourceEndpointLaw L r b) := by
  have := sourceEndpointLaw_isProbabilityMeasure hL r b
  exact prw_integrable_of_integrable_sq _ _ (sourceEndpointWeight_measurable r)
    (sourceEndpointLaw_weight_square_integrable hL r b hG hr hr1)

/-- The literal prescribed height has the actual contact pin as its right limit. -/
theorem sourcePrescribedPins_tendsto (b : ℝ) :
    Tendsto (fun r => sourcePrescribedPins r b) (𝓝[>] (0 : ℝ))
      (𝓝 ![b,0,0,0,0,2]) := by
  have hc : Continuous (fun r => sourcePrescribedPins r b) := by
    unfold sourcePrescribedPins
    fun_prop
  simpa [sourcePrescribedPins] using (hc.tendsto 0).mono_left nhdsWithin_le_nhds

/-- Explicit endpoint probability with the literal normalized weight density. -/
def sourceEndpointWeightedProbability (L r b : ℝ) : Measure (Fin 6 → ℝ) :=
  prwMeasure (sourceEndpointLaw L r b) (sourceEndpointWeight r)

/-- Endpoint reweighting glue; its estimates are derived in the final actual
source theorem after the actual covariance-limit exports are available. -/
theorem sourceEndpoint_weighted_event_bound_of_estimates {L : ℝ} (hL : L ≠ 0)
    (r b C c : ℝ) (hC : 0 ≤ C) (hc : 0 < c) (hr : 0 < r)
    (hsq : Integrable (fun v => sourceEndpointWeight r v ^ 2) (sourceEndpointLaw L r b))
    (hmoment : (∫ v, sourceEndpointWeight r v ^ 2 ∂sourceEndpointLaw L r b) ≤ C*r^4)
    (hfloor : c*r^2 ≤ ∫ v, sourceEndpointWeight r v ∂sourceEndpointLaw L r b)
    (E : Set (Fin 6 → ℝ)) (hE : MeasurableSet E) :
    IsProbabilityMeasure (sourceEndpointWeightedProbability L r b) ∧
      (sourceEndpointWeightedProbability L r b).real E ≤
        (Real.sqrt C/c) * Real.sqrt ((sourceEndpointLaw L r b).real E) := by
  have := sourceEndpointLaw_isProbabilityMeasure hL r b
  exact prw_square_root_event_bound (sourceEndpointLaw L r b) (sourceEndpointWeight r)
    (sourceEndpointWeight_measurable r) (sourceEndpointWeight_nonneg r) hsq
    C c r hC hc hr hmoment hfloor E hE

open I4SourceContact I4SourceConfluentLimits I4GaussianRegressionLimits

theorem sourceEndpoint_parameter_limits {L : ℝ} (hL : L ≠ 0) (b : ℝ) :
    (∀ᶠ r in 𝓝[>] (0 : ℝ), IsUnit (sourcePinCovarianceAt L r)) ∧
      Tendsto (fun r => toLp 2 (sourceEndpointMean L r b)) (𝓝[>] (0 : ℝ))
        (𝓝 (toLp 2 (sourceContactEndpointMean L b))) ∧
      Tendsto (sourceEndpointSchurAt L) (𝓝[>] (0 : ℝ))
        (𝓝 (sourceContactEndpointSchur L)) := by
  have hd : Tendsto (fun r => sourcePrescribedPins r b) (𝓝[>] (0 : ℝ))
      (𝓝 (sourceContactPinValue b)) := by
    simpa only [sourceContactPinValue] using sourcePrescribedPins_tendsto b
  have hp := regression_parameter_limits (𝓝[>] (0 : ℝ))
    (sourcePinCovarianceAt L) (sourceCrossCovarianceAt L) (sourceEndpointCovarianceAt L)
    (fun r => sourcePrescribedPins r b)
    (sourcePinCovariance L) (sourceContactEndpointCrossCovariance L)
    (sourceContactEndpointCovariance L) (sourceContactPinValue b)
    (sourcePinCovarianceAt_tendsto hL) (sourceCrossCovarianceAt_tendsto hL)
    (sourceEndpointCovarianceAt_tendsto hL) hd (sourcePinCovariance_isUnit hL)
  refine ⟨hp.1, ?_, ?_⟩
  · have hm : Tendsto (fun r => sourceEndpointMean L r b) (𝓝[>] (0 : ℝ))
        (𝓝 (sourceContactEndpointMean L b)) := by
      simpa only [regressionMean, sourceEndpointMean, sourceRegressionMapAt_apply,
        sourceRegressionMatrixAt, sourceContactEndpointMean, sourceContactEndpointRegression]
        using hp.2.2.1
    exact ((PiLp.continuous_toLp 2 (fun _ : Fin 6 => ℝ)).tendsto _).comp hm
  · exact hp.2.2.2

theorem sourceEndpoint_uniform_parameter_bounds {L : ℝ} (hL : L ≠ 0) (b : ℝ) :
    ∃ M V δ : ℝ, 0 ≤ M ∧ 0 ≤ V ∧ 0 < δ ∧
      ∀ r : ℝ, 0 < r → r < δ → IsUnit (sourcePinCovarianceAt L r) ∧
        (∀ i, |sourceEndpointMean L r b i| ≤ M) ∧
        (∀ i, |sourceEndpointSchurAt L r i i| ≤ V) := by
  have hd : Tendsto (fun r => sourcePrescribedPins r b) (𝓝[>] (0 : ℝ))
      (𝓝 (sourceContactPinValue b)) := by
    simpa only [sourceContactPinValue] using sourcePrescribedPins_tendsto b
  obtain ⟨M,V,δ,hM,hV,hδ,hbounds⟩ := regression_uniform_bounds
    (sourcePinCovarianceAt L) (sourceCrossCovarianceAt L) (sourceEndpointCovarianceAt L)
    (fun r => sourcePrescribedPins r b)
    (sourcePinCovariance L) (sourceContactEndpointCrossCovariance L)
    (sourceContactEndpointCovariance L) (sourceContactPinValue b)
    (sourcePinCovarianceAt_tendsto hL) (sourceCrossCovarianceAt_tendsto hL)
    (sourceEndpointCovarianceAt_tendsto hL) hd (sourcePinCovariance_isUnit hL)
  refine ⟨M,V,δ,hM,hV,hδ,?_⟩
  intro r hr hrδ
  obtain ⟨hunit,hm,hcov⟩ := hbounds r hr hrδ
  refine ⟨hunit,?_,?_⟩
  · simpa only [regressionMean, sourceEndpointMean, sourceRegressionMapAt_apply,
      sourceRegressionMatrixAt] using hm
  · simpa only [regressionCovariance, sourceEndpointSchurAt] using hcov


/- Normalizer assembly coauthored by actual OpenAI/Codex helper
/root/dm2_landed_composition/peano_strategy; author-side only. -/
/-- The actual every-pin source law has a positive small-radius physical
normalizer coefficient, with arbitrary fixed real prescribed height. -/
theorem sourceEndpoint_normalizer_floor {L : ℝ} (hL : L ≠ 0) (b : ℝ) :
    ∃ c : ℝ, 0 < c ∧ ∃ δ : ℝ, 0 < δ ∧
      ∀ r : ℝ, 0 < r → r < δ →
        c * r^2 ≤ ∫ v, sourceEndpointWeight r v ∂sourceEndpointLaw L r b := by
  obtain ⟨hunit, hm, hS⟩ := sourceEndpoint_parameter_limits hL b
  have hPSD : ∀ᶠ r in 𝓝[>] (0 : ℝ), (sourceEndpointSchurAt L r).PosSemidef := by
    filter_upwards [hunit] with r hr
    exact sourceEndpointSchurAt_posSemidef hL r hr
  have hmC := sourceContactEndpointMean_contact hL b
  have hSC := sourceContactEndpointSchur_contact hL
  obtain ⟨c, hc, δN, hδN, hfloor⟩ :=
    gnl_physical_normalizer_floor_of_parameter_limits
      (fun r => toLp 2 (sourceEndpointMean L r b)) (sourceEndpointSchurAt L)
      (toLp 2 (sourceContactEndpointMean L b)) (sourceContactEndpointSchur L)
      hm hS hPSD (sourceContactEndpointSchur_posSemidef hL)
      hmC.1 hmC.2.1 hmC.2.2.1 hmC.2.2.2
      hSC.1 hSC.2.1 hSC.2.2.1 hSC.2.2.2
      (sourceContactEndpointSchur_referenceVariance_pos hL)
  obtain ⟨δG, hδG, hGinterval⟩ := mem_nhdsGT_iff_exists_Ioo_subset.mp hunit
  refine ⟨c, hc, min δN δG, lt_min hδN hδG, ?_⟩
  intro r hr hrδ
  have hG : IsUnit (sourcePinCovarianceAt L r) :=
    hGinterval ⟨hr, lt_of_lt_of_le hrδ (min_le_right _ _)⟩
  rw [sourceEndpointLaw_weight_integral hL r b hG]
  exact hfloor r hr (lt_of_lt_of_le hrδ (min_le_left _ _))

theorem sourceEndpoint_estimates {L : ℝ} (hL : L ≠ 0) (b : ℝ) :
    ∃ C c δ : ℝ, 0 ≤ C ∧ 0 < c ∧ 0 < δ ∧ δ ≤ 1 ∧
      ∀ r : ℝ, 0 < r → r < δ →
        Integrable (fun v => sourceEndpointWeight r v ^ 2) (sourceEndpointLaw L r b) ∧
        (∫ v, sourceEndpointWeight r v ^ 2 ∂sourceEndpointLaw L r b) ≤ C*r^4 ∧
        c*r^2 ≤ ∫ v, sourceEndpointWeight r v ∂sourceEndpointLaw L r b := by
  obtain ⟨M,V,δM,hM,hV,hδM,hbounds⟩ := sourceEndpoint_uniform_parameter_bounds hL b
  obtain ⟨c,hc,δF,hδF,hfloor⟩ := sourceEndpoint_normalizer_floor hL b
  refine ⟨16 * gm8FiniteM8 6 M V,c,min δM (min δF 1),
    mul_nonneg (by norm_num) (gm8FiniteM8_nonneg 6 M V),hc,
    lt_min hδM (lt_min hδF (by norm_num)),
    (min_le_right _ _).trans (min_le_right _ _),?_⟩
  intro r hr hrδ
  have hrM : r < δM := lt_of_lt_of_le hrδ (min_le_left _ _)
  have hrF : r < δF := lt_of_lt_of_le hrδ ((min_le_right _ _).trans (min_le_left _ _))
  have hr1 : r ≤ 1 := (le_of_lt hrδ).trans ((min_le_right _ _).trans (min_le_right _ _))
  obtain ⟨hunit,hm,hv⟩ := hbounds r hr hrM
  obtain ⟨hsq,hbound⟩ := sourceEndpointLaw_second_moment_of_parameter_bounds hL r b M V
    hunit hM hV hr hr1 hm hv
  exact ⟨hsq,hbound,hfloor r hr hrF⟩

theorem sourceEndpoint_estimates_and_weighted_events {L : ℝ} (hL : L ≠ 0) (b : ℝ) :
    ∃ C c δ : ℝ, 0 ≤ C ∧ 0 < c ∧ 0 < δ ∧ δ ≤ 1 ∧
      ∀ r : ℝ, 0 < r → r < δ →
        Integrable (fun v => sourceEndpointWeight r v ^ 2) (sourceEndpointLaw L r b) ∧
        (∫ v, sourceEndpointWeight r v ^ 2 ∂sourceEndpointLaw L r b) ≤ C*r^4 ∧
        c*r^2 ≤ (∫ v, sourceEndpointWeight r v ∂sourceEndpointLaw L r b) ∧
        IsProbabilityMeasure (sourceEndpointWeightedProbability L r b) ∧
        ∀ E : Set (Fin 6 → ℝ), MeasurableSet E →
          (sourceEndpointWeightedProbability L r b).real E ≤
            (Real.sqrt C/c) * Real.sqrt ((sourceEndpointLaw L r b).real E) := by
  obtain ⟨C,c,δ,hC,hc,hδ,hδ1,he⟩ := sourceEndpoint_estimates hL b
  refine ⟨C,c,δ,hC,hc,hδ,hδ1,?_⟩
  intro r hr hrδ
  obtain ⟨hsq,hm,hf⟩ := he r hr hrδ
  have hp := sourceEndpoint_weighted_event_bound_of_estimates hL r b C c hC hc hr
    hsq hm hf Set.univ MeasurableSet.univ
  refine ⟨hsq,hm,hf,hp.1,?_⟩
  intro E hE
  exact (sourceEndpoint_weighted_event_bound_of_estimates hL r b C c hC hc hr
    hsq hm hf E hE).2


open I4SourcePhysicalEndpointKernel

/-- The canonical kernel of the actual physical normalized Hessian entries. -/
def sourcePhysicalEndpointLaw (L r b : ℝ) : Measure (Fin 6 → ℝ) :=
  sourcePhysicalEndpointKernel L r (sourcePrescribedPins r b)

/-- Literal equality of the specified kernels at the prescribed pins, derived
from their common residual law and the proved zero correction there. -/
theorem sourcePhysicalEndpointLaw_eq {L : ℝ} (hL : L ≠ 0) (r b : ℝ) :
    sourcePhysicalEndpointLaw L r b = sourceEndpointLaw L r b :=
  sourcePhysicalEndpointKernel_at_prescribed hL r b

theorem sourcePhysicalEndpointLaw_isProbabilityMeasure {L : ℝ} (hL : L ≠ 0) (r b : ℝ) :
    IsProbabilityMeasure (sourcePhysicalEndpointLaw L r b) := by
  rw [sourcePhysicalEndpointLaw_eq hL r b]
  exact sourceEndpointLaw_isProbabilityMeasure hL r b

/-- Actual block limits supply current invertibility, hence the physical
kernel is a conditional law at every sufficiently small positive radius. -/
theorem sourcePhysicalEndpointKernel_eventually_hasCondDistrib {L : ℝ} (hL : L ≠ 0) :
    ∀ᶠ r in 𝓝[>] (0 : ℝ),
      HasCondDistrib (sourceNormalizedEndpoints L r) (sourceConfluentPinsAt L r)
        (sourcePhysicalEndpointKernel L r) (smoothLaw 2 L) := by
  filter_upwards [(sourceEndpoint_parameter_limits hL 0).1] with r hr
  exact sourcePhysicalEndpointKernel_hasCondDistrib hL r hr

/-- The explicit probability is weighted from the actual physical endpoint law. -/
def sourcePhysicalEndpointWeightedProbability (L r b : ℝ) : Measure (Fin 6 → ℝ) :=
  prwMeasure (sourcePhysicalEndpointLaw L r b) (sourceEndpointWeight r)

theorem sourcePhysicalEndpointWeightedProbability_eq {L : ℝ} (hL : L ≠ 0) (r b : ℝ) :
    sourcePhysicalEndpointWeightedProbability L r b = sourceEndpointWeightedProbability L r b := by
  unfold sourcePhysicalEndpointWeightedProbability sourceEndpointWeightedProbability
  rw [sourcePhysicalEndpointLaw_eq hL r b]

/-- Actual physical canonical endpoint moment, normalizer and measurable-event
estimates, at arbitrary fixed height. No desired law, covariance limit,
parameter bound or positive normalizer is a premise. -/
theorem sourcePhysicalEndpoint_estimates_and_weighted_events {L : ℝ} (hL : L ≠ 0) (b : ℝ) :
    ∃ C c δ : ℝ, 0 ≤ C ∧ 0 < c ∧ 0 < δ ∧ δ ≤ 1 ∧
      ∀ r : ℝ, 0 < r → r < δ →
        Integrable (fun v => sourceEndpointWeight r v ^ 2) (sourcePhysicalEndpointLaw L r b) ∧
        (∫ v, sourceEndpointWeight r v ^ 2 ∂sourcePhysicalEndpointLaw L r b) ≤ C*r^4 ∧
        c*r^2 ≤ (∫ v, sourceEndpointWeight r v ∂sourcePhysicalEndpointLaw L r b) ∧
        IsProbabilityMeasure (sourcePhysicalEndpointWeightedProbability L r b) ∧
        ∀ E : Set (Fin 6 → ℝ), MeasurableSet E →
          (sourcePhysicalEndpointWeightedProbability L r b).real E ≤
            (Real.sqrt C/c) * Real.sqrt ((sourcePhysicalEndpointLaw L r b).real E) := by
  simpa only [sourcePhysicalEndpointLaw_eq hL,
    sourcePhysicalEndpointWeightedProbability_eq hL] using
      sourceEndpoint_estimates_and_weighted_events hL b

end I4SourceEndpointEstimates
