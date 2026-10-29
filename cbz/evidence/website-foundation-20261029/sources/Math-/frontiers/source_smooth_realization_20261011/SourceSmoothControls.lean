import SourceSmoothCarrier

/-! Discriminating controls for the actual smooth-source construction.
These controls are proof obligations, not observed runtime outcomes. They require
orders above four, a common coefficient event, literal source inclusion and exact
law preservation. They make no conditional-law or theta-kernel claim. -/

noncomputable section
open I4Source I4Smooth I4SmoothCarrier I4Kernel MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal

namespace I4SmoothControls

/-- The old a.e. order-four theorem cannot fill the all-orders goal. -/
theorem old_four_does_not_fill_all_orders (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d, ∀ n : ℕ, ContDiff ℝ n (sourceField d L ω) := by
  fail_if_success exact I4C4.sourceField_contDiff_ae d hL
  exact sourceField_smooth_ae d hL

theorem represented_sample_order_five (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample d L) : ContDiff ℝ 5 (smoothSourceField d L ω) :=
  smoothSourceField_contDiff d hL ω 5

theorem represented_sample_order_thirtyseven (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample d L) : ContDiff ℝ 37 (smoothSourceField d L ω) :=
  smoothSourceField_contDiff d hL ω 37

/-- Both higher orders hold for every sample of the very same carrier. -/
theorem one_carrier_for_both_orders (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample d L) :
    ContDiff ℝ 5 (smoothSourceField d L ω) ∧
      ContDiff ℝ 37 (smoothSourceField d L ω) :=
  ⟨represented_sample_order_five d hL ω, represented_sample_order_thirtyseven d hL ω⟩

theorem literal_field_everywhere (d : ℕ) (L : ℝ) (ω : SmoothSample d L)
    (x : Fin d → ℝ) :
    smoothSourceField d L ω x = sourceField d L (sampleInclusion d L ω) x := rfl

theorem entire_coefficient_law_preserved (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    (smoothLaw d L).map (sampleInclusion d L) = sourceLaw d :=
  sampleInclusion_map d hL

theorem arbitrary_observation_law_preserved (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    {E : Type*} [MeasurableSpace E] (X : (SourceMode d → ℝ) → E) (hX : Measurable X) :
    (smoothLaw d L).map (fun ω => X (sampleInclusion d L ω)) = (sourceLaw d).map X :=
  observation_map_eq d hL X hX

/-- Arbitrarily large changes on a finite block leave the actual carrier intact. -/
theorem finite_block_changes_preserve_carrier (d : ℕ) (L : ℝ)
    (ω η : SourceMode d → ℝ) (s : Finset (SourceMode d))
    (hωη : ∀ m ∉ s, ω m = η m) :
    goodSample d L ω ↔ goodSample d L η :=
  goodSample_finite_change_iff d L ω η s hωη

/-- The restricted field keeps variance one, which excludes a zero-field substitution. -/
theorem field_variance_one (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x : Fin d → ℝ) :
    cov[fun ω => smoothSourceField d L ω x,
      fun ω => smoothSourceField d L ω x; smoothLaw d L] = 1 := by
  rw [smoothSourceField_covariance d hL x x, sub_self, spectralKernel_zero d hL]

/-- The underlying restricted measure has total mass one, without conditioning
on a finite pin event or normalizing by a weight. -/
theorem carrier_probability (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    IsProbabilityMeasure (smoothLaw d L) :=
  smoothLaw_isProbabilityMeasure d hL

end I4SmoothControls
