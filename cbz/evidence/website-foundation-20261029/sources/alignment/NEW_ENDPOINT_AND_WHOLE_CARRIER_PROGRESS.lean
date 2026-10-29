import SourceSmoothConditionalKernel

/-!
# Literal determinant weighting on the actual smooth carrier

The base law is the specified actual whole-carrier kernel at the prescribed
six-pin value. The weight uses the actual endpoint Hessian entries and strict
maximum/saddle typing. Exact endpoint pushforward identities transfer the
proved moment and normalizer estimates to every measurable carrier event.

The normalized density formula below is a probability law on the actual smooth
carrier. It is not identified here with a geometric critical-point-process Palm
disintegration, and supplies no Morse, elder, counting or event measurability claim.
See `SOURCE_SMOOTH_PALM.md` for the frozen source contract.
-/

noncomputable section

open I4SmoothCarrier I4SourceConfluentPins I4SourceEndpointRegression I4SourceSmoothConditional
open I4SourceEndpointEstimates I4SourceEndpointParameters
open MeasureTheory ProbabilityTheory ResearchFormalCoreR1 Matrix Filter
open scoped BigOperators ENNReal Topology

set_option autoImplicit false

namespace I4SourceSmoothPalm

lemma withDensity_map_comp {Ω E : Type*} [MeasurableSpace Ω] [MeasurableSpace E]
    (μ : Measure Ω) (F : Ω → E) (w : E → ℝ≥0∞) (hF : Measurable F)
    (hw : Measurable w) :
    (μ.withDensity (fun ω => w (F ω))).map F = (μ.map F).withDensity w := by
  ext s hs
  rw [Measure.map_apply hF hs, withDensity_apply _ (hs.preimage hF),
    withDensity_apply _ hs, setLIntegral_map hs hw hF]

/-- Normalized weighting commutes exactly with a measurable pushforward.
The two normalizers are derived to be equal; no positivity or integrability premise is needed. -/
theorem prwMeasure_map_comp {Ω E : Type*} [MeasurableSpace Ω] [MeasurableSpace E]
    (μ : Measure Ω) (F : Ω → E) (W : E → ℝ) (hF : Measurable F)
    (hW : Measurable W) :
    (prwMeasure μ (W ∘ F)).map F = prwMeasure (μ.map F) W := by
  have hnormalizer : (∫ ω, (W ∘ F) ω ∂μ) = ∫ y, W y ∂μ.map F :=
    (integral_map hF.aemeasurable hW.aestronglyMeasurable).symm
  unfold prwMeasure
  rw [hnormalizer]
  exact withDensity_map_comp μ F (fun y => ENNReal.ofReal (W y / ∫ z, W z ∂μ.map F))
    hF (by fun_prop)

lemma prwMeasure_absolutelyContinuous {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (W : Ω → ℝ) : prwMeasure μ W ≪ μ :=
  withDensity_absolutelyContinuous _ _

lemma prwMeasure_ae_of_ae {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (W : Ω → ℝ) {p : Ω → Prop} (hp : ∀ᵐ ω ∂μ, p ω) :
    ∀ᵐ ω ∂prwMeasure μ W, p ω :=
  (prwMeasure_absolutelyContinuous μ W).ae_le hp

lemma prwMeasure_ae_weight_pos {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (W : Ω → ℝ) (hW : Measurable W)
    (hZ : 0 < ∫ ω, W ω ∂μ) :
    ∀ᵐ ω ∂prwMeasure μ W, 0 < W ω := by
  unfold prwMeasure
  rw [ae_withDensity_iff (by fun_prop)]
  filter_upwards [] with ω hω
  have hd : 0 < W ω / ∫ x, W x ∂μ := ENNReal.ofReal_pos.mp (pos_iff_ne_zero.mpr hω)
  exact (div_pos_iff_of_pos_right hZ).mp hd

def sourceSmoothLaw {L : ℝ} (hL : L ≠ 0) (r b : ℝ) : Measure (SmoothSample 2 L) :=
  sourceSmoothKernel hL r (sourcePrescribedPins r b)

instance sourceSmoothLaw_isProbabilityMeasure {L : ℝ} (hL : L ≠ 0) (r b : ℝ) :
    IsProbabilityMeasure (sourceSmoothLaw hL r b) := by
  unfold sourceSmoothLaw
  infer_instance

lemma rawPins_prescribed_iff {L : ℝ} (r b : ℝ) (hr : r ≠ 0) (ω : SmoothSample 2 L) :
    rawPins L r ω = ![b,b-r^3/6,0,0,0,0] ↔
      sourceConfluentPinsAt L r ω = sourcePrescribedPins r b := by
  change _ ↔ confluentEquiv r hr (rawPins L r ω) = _
  rw [← show confluentEquiv r hr ![b,b-r^3/6,0,0,0,0] = sourcePrescribedPins r b from
    prescribed_target r b hr]
  exact (confluentEquiv r hr).injective.eq_iff.symm

theorem sourceSmoothLaw_raw_pin_fiber {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hr : r ≠ 0) (hG : IsUnit (sourcePinCovarianceAt L r)) :
    sourceSmoothLaw hL r b {ω | rawPins L r ω = ![b,b-r^3/6,0,0,0,0]} = 1 := by
  have hset : {ω : SmoothSample 2 L | rawPins L r ω = ![b,b-r^3/6,0,0,0,0]} =
      {ω | sourceConfluentPinsAt L r ω = sourcePrescribedPins r b} := by
    ext ω
    exact rawPins_prescribed_iff r b hr ω
  rw [hset]
  exact sourceSmoothKernel_pin_fiber hL r hG (sourcePrescribedPins r b)

lemma sourceRawPins_measurable {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0) :
    Measurable (rawPins L r) := by
  have heq : rawPins L r = fun ω => rawTransform r (sourceConfluentPinsAt L r ω) := by
    funext ω
    exact (rawTransform_confluentTransform r hr (rawPins L r ω)).symm
  rw [heq]
  have hc : Continuous (rawTransform r) := by
    unfold rawTransform
    fun_prop
  exact hc.measurable.comp (sourceConfluentPinsAt_measurable hL r)

theorem sourceSmoothLaw_raw_pin_fiber_ae {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hr : r ≠ 0) (hG : IsUnit (sourcePinCovarianceAt L r)) :
    ∀ᵐ ω ∂sourceSmoothLaw hL r b, rawPins L r ω = ![b,b-r^3/6,0,0,0,0] := by
  apply (mem_ae_iff_prob_eq_one
    (measurableSet_eq_fun (sourceRawPins_measurable hL r hr) measurable_const)).2
  exact sourceSmoothLaw_raw_pin_fiber hL r b hr hG

/-- Every probability measure absolutely continuous with respect to the actual conditioned
carrier law retains its exact prescribed raw values and zero gradients. -/
theorem raw_pin_fiber_of_absolutelyContinuous {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hr : r ≠ 0) (hG : IsUnit (sourcePinCovarianceAt L r))
    (ν : Measure (SmoothSample 2 L)) [IsProbabilityMeasure ν]
    (hν : ν ≪ sourceSmoothLaw hL r b) :
    ν {ω | rawPins L r ω = ![b,b-r^3/6,0,0,0,0]} = 1 := by
  apply (mem_ae_iff_prob_eq_one
    (measurableSet_eq_fun (sourceRawPins_measurable hL r hr) measurable_const)).1
  exact hν.ae_le (sourceSmoothLaw_raw_pin_fiber_ae hL r b hr hG)

theorem prwMeasure_raw_pin_fiber {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hr : r ≠ 0) (hG : IsUnit (sourcePinCovarianceAt L r)) (W : SmoothSample 2 L → ℝ)
    [IsProbabilityMeasure (prwMeasure (sourceSmoothLaw hL r b) W)] :
    prwMeasure (sourceSmoothLaw hL r b) W
      {ω | rawPins L r ω = ![b,b-r^3/6,0,0,0,0]} = 1 := by
  apply raw_pin_fiber_of_absolutelyContinuous hL r b hr hG
  exact withDensity_absolutelyContinuous _ _

/-- Literal actual symmetric endpoint Hessian entries at M. -/
def sourceRawHM (L r : ℝ) (ω : SmoothSample 2 L) : Matrix (Fin 2) (Fin 2) ℝ :=
  let v := rawPinsWithHessians L r ω
  !![v 6,v 7;v 7,v 8]

/-- Literal actual symmetric endpoint Hessian entries at S. -/
def sourceRawHS (L r : ℝ) (ω : SmoothSample 2 L) : Matrix (Fin 2) (Fin 2) ℝ :=
  let v := rawPinsWithHessians L r ω
  !![v 9,v 10;v 10,v 11]

def sourceSmoothTyped (L r : ℝ) (ω : SmoothSample 2 L) : Prop :=
  dm2Negative (sourceRawHM L r ω) ∧ (sourceRawHS L r ω).det < 0

/-- Strict maximum/saddle determinant weight of the actual field at its two endpoints. -/
def sourceSmoothWeight (L r : ℝ) (ω : SmoothSample 2 L) : ℝ := by
  classical
  exact if sourceSmoothTyped L r ω then |(sourceRawHM L r ω).det * (sourceRawHS L r ω).det| else 0

lemma sourceRawHM_eq_normalized (L r : ℝ) (hr : r ≠ 0) (ω : SmoothSample 2 L) :
    sourceRawHM L r ω = egm_HM r (sourcePhysicalCoordinates (sourceNormalizedEndpoints L r ω)) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [sourceRawHM, egm_HM, sourcePhysicalCoordinates, sourceNormalizedEndpoints]
  all_goals field_simp [hr]

lemma sourceRawHS_eq_normalized (L r : ℝ) (hr : r ≠ 0) (ω : SmoothSample 2 L) :
    sourceRawHS L r ω = egm_HS r (sourcePhysicalCoordinates (sourceNormalizedEndpoints L r ω)) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [sourceRawHS, egm_HS, sourcePhysicalCoordinates, sourceNormalizedEndpoints]
  all_goals field_simp [hr]

theorem sourceSmoothWeight_eq (L r : ℝ) (hr : r ≠ 0) (ω : SmoothSample 2 L) :
    sourceSmoothWeight L r ω = sourceEndpointWeight r (sourceNormalizedEndpoints L r ω) := by
  unfold sourceSmoothWeight sourceSmoothTyped sourceEndpointWeight egm_weight
  rw [sourceRawHM_eq_normalized L r hr ω, sourceRawHS_eq_normalized L r hr ω]
  split_ifs <;> rfl

theorem sourceSmoothWeight_nonneg (L r : ℝ) (ω : SmoothSample 2 L) :
    0 ≤ sourceSmoothWeight L r ω := by
  unfold sourceSmoothWeight
  split_ifs
  · exact abs_nonneg _
  · exact le_rfl

theorem sourceSmoothWeight_measurable {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0) :
    Measurable (sourceSmoothWeight L r) := by
  have heq : sourceSmoothWeight L r = sourceEndpointWeight r ∘ sourceNormalizedEndpoints L r :=
    funext (sourceSmoothWeight_eq L r hr)
  rw [heq]
  exact (sourceEndpointWeight_measurable r).comp (sourceNormalizedEndpoints_measurable hL r)

theorem sourceSmoothWeight_pos_iff (L r : ℝ) (ω : SmoothSample 2 L) :
    0 < sourceSmoothWeight L r ω ↔ sourceSmoothTyped L r ω := by
  classical
  by_cases ht : sourceSmoothTyped L r ω
  · have hM : 0 < (sourceRawHM L r ω).det := by
      have h := (dm2_negative_iff _ _ _).mp ht.1
      simpa [sourceRawHM, Matrix.det_fin_two, pow_two] using h.2
    have hn : (sourceRawHM L r ω).det * (sourceRawHS L r ω).det ≠ 0 :=
      mul_ne_zero hM.ne' ht.2.ne
    rw [sourceSmoothWeight, ite_eq_left ht]
    exact ⟨fun _ => ht, fun _ => abs_pos.mpr hn⟩
  · simp [sourceSmoothWeight, ht]

theorem sourceSmoothTyped_measurableSet {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0) :
    MeasurableSet {ω : SmoothSample 2 L | sourceSmoothTyped L r ω} := by
  have heq : {ω : SmoothSample 2 L | sourceSmoothTyped L r ω} =
      {ω | 0 < sourceSmoothWeight L r ω} := by
    ext ω
    exact (sourceSmoothWeight_pos_iff L r ω).symm
  rw [heq]
  exact measurableSet_lt measurable_const (sourceSmoothWeight_measurable hL r hr)

/-- The whole-carrier probability defined by the literal normalized actual determinant density. -/
def sourceSmoothWeightedLaw {L : ℝ} (hL : L ≠ 0) (r b : ℝ) : Measure (SmoothSample 2 L) :=
  prwMeasure (sourceSmoothLaw hL r b) (sourceSmoothWeight L r)

theorem sourceSmoothLaw_endpoint_hasLaw {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hG : IsUnit (sourcePinCovarianceAt L r)) :
    HasLaw (sourceNormalizedEndpoints L r) (sourcePhysicalEndpointLaw L r b)
      (sourceSmoothLaw hL r b) := by
  refine ⟨(sourceNormalizedEndpoints_measurable hL r).aemeasurable, ?_⟩
  exact sourceSmoothKernel_physical_prescribed_map hL r b hG

theorem sourceSmoothLaw_weight_square_integrable {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hr : r ≠ 0) (hG : IsUnit (sourcePinCovarianceAt L r))
    (hsq : Integrable (fun v => sourceEndpointWeight r v ^ 2) (sourcePhysicalEndpointLaw L r b)) :
    Integrable (fun ω => sourceSmoothWeight L r ω ^ 2) (sourceSmoothLaw hL r b) := by
  simpa only [sourceSmoothWeight_eq L r hr, Function.comp_def] using
    (sourceSmoothLaw_endpoint_hasLaw hL r b hG).integrable_comp hsq

theorem sourceSmoothLaw_weight_integral {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hr : r ≠ 0) (hG : IsUnit (sourcePinCovarianceAt L r)) :
    (∫ ω, sourceSmoothWeight L r ω ∂sourceSmoothLaw hL r b) =
      ∫ v, sourceEndpointWeight r v ∂sourcePhysicalEndpointLaw L r b := by
  simpa only [sourceSmoothWeight_eq L r hr, Function.comp_def] using
    (sourceSmoothLaw_endpoint_hasLaw hL r b hG).integral_comp
      (sourceEndpointWeight_measurable r).aestronglyMeasurable

theorem sourceSmoothLaw_weight_square_integral {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hr : r ≠ 0) (hG : IsUnit (sourcePinCovarianceAt L r)) :
    (∫ ω, sourceSmoothWeight L r ω ^ 2 ∂sourceSmoothLaw hL r b) =
      ∫ v, sourceEndpointWeight r v ^ 2 ∂sourcePhysicalEndpointLaw L r b := by
  simpa only [sourceSmoothWeight_eq L r hr, Function.comp_def] using
    (sourceSmoothLaw_endpoint_hasLaw hL r b hG).integral_comp
      ((sourceEndpointWeight_measurable r).pow_const 2).aestronglyMeasurable

/-- Exact endpoint pushforward of the literal weighted whole-carrier law. -/
theorem sourceSmoothWeightedLaw_endpoint_map {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hr : r ≠ 0) (hG : IsUnit (sourcePinCovarianceAt L r)) :
    (sourceSmoothWeightedLaw hL r b).map (sourceNormalizedEndpoints L r) =
      sourcePhysicalEndpointWeightedProbability L r b := by
  have heq : sourceSmoothWeight L r = sourceEndpointWeight r ∘ sourceNormalizedEndpoints L r :=
    funext (sourceSmoothWeight_eq L r hr)
  unfold sourceSmoothWeightedLaw sourcePhysicalEndpointWeightedProbability
  rw [heq, prwMeasure_map_comp _ _ _ (sourceNormalizedEndpoints_measurable hL r)
      (sourceEndpointWeight_measurable r), (sourceSmoothLaw_endpoint_hasLaw hL r b hG).map_eq]

/-- The event estimate applies to every measurable event of the actual smooth carrier. -/
theorem sourceSmooth_weighted_event_bound_of_estimates {L : ℝ} (hL : L ≠ 0)
    (r b C c : ℝ) (hC : 0 ≤ C) (hc : 0 < c) (hr : 0 < r)
    (hsq : Integrable (fun ω => sourceSmoothWeight L r ω ^ 2) (sourceSmoothLaw hL r b))
    (hmoment : (∫ ω, sourceSmoothWeight L r ω ^ 2 ∂sourceSmoothLaw hL r b) ≤ C*r^4)
    (hfloor : c*r^2 ≤ ∫ ω, sourceSmoothWeight L r ω ∂sourceSmoothLaw hL r b)
    (E : Set (SmoothSample 2 L)) (hE : MeasurableSet E) :
    IsProbabilityMeasure (sourceSmoothWeightedLaw hL r b) ∧
      (sourceSmoothWeightedLaw hL r b).real E ≤
        (Real.sqrt C/c) * Real.sqrt ((sourceSmoothLaw hL r b).real E) := by
  exact prw_square_root_event_bound (sourceSmoothLaw hL r b) (sourceSmoothWeight L r)
    (sourceSmoothWeight_measurable hL r hr.ne') (sourceSmoothWeight_nonneg L r) hsq
    C c r hC hc hr hmoment hfloor E hE

/-- The literal density is positive only on the strict maximum/saddle type, which has full mass. -/
theorem sourceSmoothWeightedLaw_typed {L : ℝ} (hL : L ≠ 0) (r b : ℝ) (hr : r ≠ 0)
    [IsProbabilityMeasure (sourceSmoothWeightedLaw hL r b)]
    (hZ : 0 < ∫ ω, sourceSmoothWeight L r ω ∂sourceSmoothLaw hL r b) :
    sourceSmoothWeightedLaw hL r b {ω | sourceSmoothTyped L r ω} = 1 := by
  apply (mem_ae_iff_prob_eq_one (sourceSmoothTyped_measurableSet hL r hr)).1
  have hp := prwMeasure_ae_weight_pos (sourceSmoothLaw hL r b) (sourceSmoothWeight L r)
    (sourceSmoothWeight_measurable hL r hr) hZ
  filter_upwards [hp] with ω hω
  exact (sourceSmoothWeight_pos_iff L r ω).mp hω

/-- The density preserves the exact raw prescribed values and all four zero first derivatives. -/
theorem sourceSmoothWeightedLaw_raw_pins {L : ℝ} (hL : L ≠ 0) (r b : ℝ)
    (hr : r ≠ 0) (hG : IsUnit (sourcePinCovarianceAt L r))
    [IsProbabilityMeasure (sourceSmoothWeightedLaw hL r b)] :
    sourceSmoothWeightedLaw hL r b {ω | rawPins L r ω = ![b,b-r^3/6,0,0,0,0]} = 1 := by
  have : IsProbabilityMeasure (prwMeasure (sourceSmoothLaw hL r b) (sourceSmoothWeight L r)) :=
    ‹IsProbabilityMeasure (sourceSmoothWeightedLaw hL r b)›
  exact prwMeasure_raw_pin_fiber hL r b hr hG (sourceSmoothWeight L r)

/-- Actual whole-carrier determinant moments, normalizer, probability and event transfer.
Only the period and fixed prescribed height are inputs; actual covariance limits supply
all sufficiently-small-radius invertibility needed by the literal endpoint marginal. -/
theorem sourceSmoothPalm_estimates {L : ℝ} (hL : L ≠ 0) (b : ℝ) :
    ∃ C c δ : ℝ, 0 ≤ C ∧ 0 < c ∧ 0 < δ ∧ δ ≤ 1 ∧
      ∀ r : ℝ, 0 < r → r < δ →
        Integrable (fun ω => sourceSmoothWeight L r ω ^ 2) (sourceSmoothLaw hL r b) ∧
        (∫ ω, sourceSmoothWeight L r ω ^ 2 ∂sourceSmoothLaw hL r b) ≤ C*r^4 ∧
        c*r^2 ≤ (∫ ω, sourceSmoothWeight L r ω ∂sourceSmoothLaw hL r b) ∧
        IsProbabilityMeasure (sourceSmoothWeightedLaw hL r b) ∧
        (∀ E : Set (SmoothSample 2 L), MeasurableSet E →
          (sourceSmoothWeightedLaw hL r b).real E ≤
            (Real.sqrt C/c) * Real.sqrt ((sourceSmoothLaw hL r b).real E)) ∧
        (sourceSmoothWeightedLaw hL r b).map (sourceNormalizedEndpoints L r) =
          sourcePhysicalEndpointWeightedProbability L r b ∧
        sourceSmoothWeightedLaw hL r b
          {ω | rawPins L r ω = ![b,b-r^3/6,0,0,0,0]} = 1 ∧
        sourceSmoothWeightedLaw hL r b {ω | sourceSmoothTyped L r ω} = 1 := by
  obtain ⟨C,c,δE,hC,hc,hδE,hδE1,he⟩ := sourcePhysicalEndpoint_estimates_and_weighted_events hL b
  obtain ⟨δG,hδG,hGinterval⟩ := mem_nhdsGT_iff_exists_Ioo_subset.mp
    (sourceEndpoint_parameter_limits hL b).1
  refine ⟨C,c,min δE δG,hC,hc,lt_min hδE hδG,(min_le_left _ _).trans hδE1,?_⟩
  intro r hr hrδ
  have hrE : r < δE := lt_of_lt_of_le hrδ (min_le_left _ _)
  have hG : IsUnit (sourcePinCovarianceAt L r) :=
    hGinterval ⟨hr,lt_of_lt_of_le hrδ (min_le_right _ _)⟩
  obtain ⟨hsqE,hmE,hfE,_,_⟩ := he r hr hrE
  have hsq := sourceSmoothLaw_weight_square_integrable hL r b hr.ne' hG hsqE
  have hm : (∫ ω, sourceSmoothWeight L r ω ^ 2 ∂sourceSmoothLaw hL r b) ≤ C*r^4 := by
    rw [sourceSmoothLaw_weight_square_integral hL r b hr.ne' hG]
    exact hmE
  have hf : c*r^2 ≤ ∫ ω, sourceSmoothWeight L r ω ∂sourceSmoothLaw hL r b := by
    rw [sourceSmoothLaw_weight_integral hL r b hr.ne' hG]
    exact hfE
  have hp := sourceSmooth_weighted_event_bound_of_estimates hL r b C c hC hc hr
    hsq hm hf Set.univ MeasurableSet.univ
  have : IsProbabilityMeasure (sourceSmoothWeightedLaw hL r b) := hp.1
  have hZ : 0 < ∫ ω, sourceSmoothWeight L r ω ∂sourceSmoothLaw hL r b :=
    (mul_pos hc (sq_pos_of_pos hr)).trans_le hf
  refine ⟨hsq,hm,hf,hp.1,?_,sourceSmoothWeightedLaw_endpoint_map hL r b hr.ne' hG,
    sourceSmoothWeightedLaw_raw_pins hL r b hr.ne' hG,
    sourceSmoothWeightedLaw_typed hL r b hr.ne' hZ⟩
  intro E hE
  exact (sourceSmooth_weighted_event_bound_of_estimates hL r b C c hC hc hr hsq hm hf E hE).2

end I4SourceSmoothPalm
