import Specifications.Side24

/-! Probe 3 (Side24-interfaces): the `PeriodicCoefficientInterface` does not determine `c24`.
Given any inhabitant `I`, doubling the GIVEN cone moment and `c24` gives another inhabitant, and
the two cannot both satisfy `CoefficientEnclosure` (nor `ReferenceComparison`). Hence a theorem
`∀ I, CoefficientEnclosure I` would force the interface to be empty. Probe only. -/

open MeasureTheory UniversalLaw.Spec.Side24

/-- The doubled inhabitant: same sphere measure, densities, `tau`; `coneMoment` and `c24` doubled. -/
noncomputable def probe_scale (I : PeriodicCoefficientInterface) : PeriodicCoefficientInterface where
  c24 := fun d => 2 * I.c24 d
  sphereMeasure := I.sphereMeasure
  sphereMeasure_support := I.sphereMeasure_support
  sphereMeasure_area := I.sphereMeasure_area
  pG0 := I.pG0
  pV0 := I.pV0
  tau := I.tau
  coneMoment := fun d u => 2 * I.coneMoment d u
  pG0_pos := I.pG0_pos
  pV0_pos := I.pV0_pos
  tau_pos := I.tau_pos
  coneMoment_pos := fun d hd u hu => by
    have := I.coneMoment_pos d hd u hu
    positivity
  theta_summable := I.theta_summable
  pG0_def := I.pG0_def
  pV0_def := I.pV0_def
  tau_sq := I.tau_sq
  eq_15_2 := fun d hd => by
    rw [I.eq_15_2 d hd]
    have hfun : (fun u : E d => I.pG0 d * I.pV0 d u * I.tau d u ^ ((4 : ℝ) / 3) *
        (2 * I.coneMoment d u)) =
        fun u => 2 * (I.pG0 d * I.pV0 d u * I.tau d u ^ ((4 : ℝ) / 3) * I.coneMoment d u) := by
      funext u; ring
    simp only [hfun, integral_const_mul]
    ring

theorem probe_not_both_enclosure (I : PeriodicCoefficientInterface) (h : CoefficientEnclosure I) :
    ¬ CoefficientEnclosure (probe_scale I) := by
  intro h'
  unfold CoefficientEnclosure at h h'
  simp only [probe_scale] at h'
  obtain ⟨h1, -, -, -⟩ := h
  obtain ⟨-, h2', -, -⟩ := h'
  simp only [d2Lower, d2Upper] at h1 h2'
  norm_num at h1 h2'
  linarith

theorem probe_not_both_reference (I : PeriodicCoefficientInterface) (h : ReferenceComparison I) :
    ¬ ReferenceComparison (probe_scale I) := by
  intro h'
  have h2 := h 2 (Or.inl rfl)
  have h2' := h' 2 (Or.inl rfl)
  simp only [probe_scale] at h2'
  rw [mul_div_assoc, abs_lt] at h2'
  rw [abs_lt] at h2
  obtain ⟨a1, -⟩ := h2
  obtain ⟨-, b2⟩ := h2'
  norm_num at a1 b2
  linarith

/-- The universal closure of `CoefficientEnclosure` forces the interface to be empty. -/
theorem probe_enclosure_closure_implies_empty
    (h : ∀ I : PeriodicCoefficientInterface, CoefficientEnclosure I) :
    IsEmpty PeriodicCoefficientInterface :=
  ⟨fun I => probe_not_both_enclosure I (h I) (h (probe_scale I))⟩

/-- Same for `ReferenceComparison`. -/
theorem probe_reference_closure_implies_empty
    (h : ∀ I : PeriodicCoefficientInterface, ReferenceComparison I) :
    IsEmpty PeriodicCoefficientInterface :=
  ⟨fun I => probe_not_both_reference I (h I) (h (probe_scale I))⟩

#print axioms probe_enclosure_closure_implies_empty
#print axioms probe_reference_closure_implies_empty
