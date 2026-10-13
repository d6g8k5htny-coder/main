import Specifications.Side24

/-! Probe 4 (Side24-interfaces): reductions for `RayleighSquareLaw` and `ConeMomentD2Identity`.
(a) `RayleighSquareLaw I` reduces, through the interface's joint law alone, to the Gaussian
    measure of a disk: `(γ₁ ⊗ γ₁) {p | p.1² + p.2² ≤ a} = ofReal (1 - exp (-a/2))`, left as the
    single `sorry` (polar coordinates); the right-hand interval integral is computed.
(b) the cone `NegDefCone (s, x, y)` is `s < 0 ∧ x² + y² < s²`, and `det A = s² - x² - y²`.
Probe only; `sorry` allowed here, never elsewhere. -/

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

/-- The right-hand side of `RayleighSquareLaw`. -/
theorem probe_rhs_integral (a : ℝ) :
    ∫ z in (0 : ℝ)..a, Real.exp (-z / 2) / 2 = 1 - Real.exp (-a / 2) := by
  have hderiv : ∀ z ∈ Set.uIcc (0 : ℝ) a,
      HasDerivAt (fun z : ℝ => -Real.exp (-z / 2)) (Real.exp (-z / 2) / 2) z := by
    intro z _
    have h1 : HasDerivAt (fun z : ℝ => -z / 2) (-1 / 2) z :=
      ((hasDerivAt_id' z).neg).div_const 2
    have h2 := h1.exp.neg
    convert h2 using 1
    ring
  have hint : IntervalIntegrable (fun z : ℝ => Real.exp (-z / 2) / 2) volume 0 a :=
    (by fun_prop : Continuous fun z : ℝ => Real.exp (-z / 2) / 2).intervalIntegrable 0 a
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv hint]
  simp
  ring

/-- The disk measure under the standard planar Gaussian: the hard step, left open here. -/
theorem probe_disk_measure (a : ℝ) (ha : 0 ≤ a) :
    ((gaussianReal 0 1).prod (gaussianReal 0 1)) {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a} =
      ENNReal.ofReal (1 - Real.exp (-a / 2)) := by
  sorry

/-- A.e.-measurability of the triple from its product law (as in probe 2). -/
theorem probe_aemeasurable_triple (I : ConeMomentInterface) :
    @AEMeasurable _ _ _ I.mΩ (fun ω => (I.s ω, I.x ω, I.y ω)) I.P := by
  letI := I.mΩ
  haveI := I.isProb
  have hv : ((5 : NNReal) / 3) ≠ 0 := by norm_num
  haveI : NullSingletonClass (gaussianReal 0 (5 / 3 : NNReal)) := nullSingletonClass_gaussianReal hv
  by_contra hT
  obtain ⟨c, hc⟩ : ∃ c : ℝ × ℝ × ℝ,
      Measure.map (fun ω => (I.s ω, I.x ω, I.y ω)) I.P = Measure.dirac c :=
    ⟨_, Measure.map_of_not_aemeasurable_of_ne_zero hT (IsProbabilityMeasure.ne_zero I.P)⟩
  have h := I.sxy_law
  rw [hc] at h
  have h1 : Measure.dirac c {c} =
      ((gaussianReal 0 (5 / 3)).prod ((gaussianReal 0 1).prod (gaussianReal 0 1))) {c} := by
    rw [h]
  rw [Measure.dirac_apply_of_mem (Set.mem_singleton c), measure_singleton] at h1
  exact one_ne_zero h1

/-- The joint law of `(x, y)` from the interface. -/
theorem probe_law_xy (I : ConeMomentInterface) :
    @Measure.map _ _ I.mΩ _ (fun ω => (I.x ω, I.y ω)) I.P =
      (gaussianReal 0 1).prod (gaussianReal 0 1) := by
  letI := I.mΩ
  haveI := I.isProb
  have hT := probe_aemeasurable_triple I
  have := AEMeasurable.map_map_of_aemeasurable (measurable_snd.aemeasurable) hT
  rw [I.sxy_law, Measure.map_snd_prod, measure_univ, one_smul] at this
  exact this.symm

/-- Reduction: `RayleighSquareLaw I` follows from the disk measure. -/
theorem probe_rayleighSquareLaw (I : ConeMomentInterface) : RayleighSquareLaw I := by
  unfold RayleighSquareLaw
  letI := I.mΩ
  haveI := I.isProb
  intro a ha
  have hxy : AEMeasurable (fun ω => (I.x ω, I.y ω)) I.P := (probe_aemeasurable_triple I).snd
  have hS : MeasurableSet {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a} :=
    measurableSet_le (by fun_prop) measurable_const
  have e : I.P {ω | I.x ω ^ 2 + I.y ω ^ 2 ≤ a} =
      ((gaussianReal 0 1).prod (gaussianReal 0 1)) {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ a} := by
    rw [← probe_law_xy I, Measure.map_apply_of_aemeasurable hxy hS]
    rfl
  rw [e, probe_disk_measure a ha, probe_rhs_integral, ENNReal.toReal_ofReal]
  have : Real.exp (-a / 2) ≤ 1 := by
    rw [Real.exp_le_one_iff]; linarith
  linarith

/-- (b) `det A = s² - x² - y²` for the `m = 2` transverse matrix. -/
theorem probe_det_transverse (s x y : ℝ) :
    (transverseMatrix s x y).det = s ^ 2 - x ^ 2 - y ^ 2 := by
  unfold transverseMatrix
  rw [Matrix.det_fin_two_of]
  ring

/-- (b) Forward direction of the cone characterisation: positive definiteness of `-A` gives
`s < 0` (from the diagonal) and `x² + y² < s²` (from the determinant). -/
theorem probe_cone_forward (s x y : ℝ) (h : NegDefCone (s, x, y)) :
    s < 0 ∧ x ^ 2 + y ^ 2 < s ^ 2 := by
  unfold NegDefCone at h
  have hdet := h.det_pos
  rw [Matrix.det_neg, probe_det_transverse] at hdet
  have hdiag := h.diag_pos (i := 0)
  have hdiag' := h.diag_pos (i := 1)
  simp [transverseMatrix, Matrix.diag] at hdiag hdiag'
  simp only [Fintype.card_fin, even_two, Even.neg_pow, one_pow, one_mul] at hdet
  constructor
  · linarith
  · linarith

#print axioms probe_rayleighSquareLaw
#print axioms probe_cone_forward
