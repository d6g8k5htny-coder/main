import Specifications.Side24

/-! Probe 2 (Side24-interfaces): full route test for `TraceMoments` (marginal extraction from the
product law, second moment by variance, fourth moment by the MGF derivative, exponential moment by
the Gaussian integral). Probe only; `sorry` allowed here, never elsewhere. -/

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

/-- Derivative of `exp (c t^2 / 2)`. -/
theorem probe_hasDerivAt_exp_sq (c t : ℝ) :
    HasDerivAt (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) (c * t * Real.exp (c * t ^ 2 / 2)) t := by
  have h1 : HasDerivAt (fun t : ℝ => c * t ^ 2 / 2)
      (c * (((2 : ℕ) : ℝ) * t ^ (2 - 1) * 1) / 2) t :=
    (((hasDerivAt_id' t).fun_pow 2).const_mul c).div_const 2
  have h2 := h1.exp
  convert h2 using 1
  push_cast
  ring

/-- Fourth derivative of `exp (c t^2 / 2)` at `0` is `3 c^2`. -/
theorem probe_iteratedDeriv_four (c : ℝ) :
    iteratedDeriv 4 (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) 0 = 3 * c ^ 2 := by
  have d1 : deriv (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) =
      fun t => c * t * Real.exp (c * t ^ 2 / 2) := by
    funext t; exact (probe_hasDerivAt_exp_sq c t).deriv
  have d2 : deriv (fun t : ℝ => c * t * Real.exp (c * t ^ 2 / 2)) =
      fun t => (c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2) := by
    funext t
    have h := ((hasDerivAt_id' t).const_mul c).mul (probe_hasDerivAt_exp_sq c t)
    have h' : HasDerivAt (fun t : ℝ => c * t * Real.exp (c * t ^ 2 / 2)) _ t := h
    rw [h'.deriv]; ring
  have d3 : deriv (fun t : ℝ => (c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2)) =
      fun t => (3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2) := by
    funext t
    have h := ((((hasDerivAt_id' t).fun_pow 2).const_mul (c ^ 2)).const_add c).mul
      (probe_hasDerivAt_exp_sq c t)
    have h' : HasDerivAt (fun t : ℝ => (c + c ^ 2 * t ^ 2) * Real.exp (c * t ^ 2 / 2)) _ t := h
    rw [h'.deriv]; push_cast; ring
  have d4 : deriv (fun t : ℝ => (3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2)) =
      fun t => (3 * c ^ 2 + 6 * c ^ 3 * t ^ 2 + c ^ 4 * t ^ 4) * Real.exp (c * t ^ 2 / 2) := by
    funext t
    have ha := ((hasDerivAt_id' t).const_mul (3 * c ^ 2)).add
      (((hasDerivAt_id' t).fun_pow 3).const_mul (c ^ 3))
    have h := ha.mul (probe_hasDerivAt_exp_sq c t)
    have h' : HasDerivAt (fun t : ℝ => (3 * c ^ 2 * t + c ^ 3 * t ^ 3) * Real.exp (c * t ^ 2 / 2))
      _ t := h
    rw [h'.deriv]; simp only [Pi.add_apply]; push_cast; ring
  have e4 : iteratedDeriv 4 (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) =
      deriv (iteratedDeriv 3 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))) := iteratedDeriv_succ
  have e3 : iteratedDeriv 3 (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) =
      deriv (iteratedDeriv 2 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))) := iteratedDeriv_succ
  have e2 : iteratedDeriv 2 (fun t : ℝ => Real.exp (c * t ^ 2 / 2)) =
      deriv (iteratedDeriv 1 (fun t : ℝ => Real.exp (c * t ^ 2 / 2))) := iteratedDeriv_succ
  rw [e4, e3, e2, iteratedDeriv_one, d1, d2, d3, d4]
  simp

/-- Second moment of the centred Gaussian. -/
theorem probe_second_moment (v : NNReal) : ∫ t, t ^ 2 ∂(gaussianReal 0 v) = (v : ℝ) := by
  have h := variance_fun_id_gaussianReal (μ := 0) (v := v)
  rw [variance_eq_integral (by fun_prop)] at h
  simpa using h

/-- Fourth moment of the centred Gaussian, through `iteratedDeriv_mgf_zero`. -/
theorem probe_fourth_moment (v : NNReal) :
    ∫ t, t ^ 4 ∂(gaussianReal 0 v) = 3 * (v : ℝ) ^ 2 := by
  have h := iteratedDeriv_mgf_zero (X := fun x : ℝ => x) (μ := gaussianReal 0 v) (by simp) 4
  rw [mgf_fun_id_gaussianReal] at h
  simp only [zero_mul, zero_add] at h
  rw [probe_iteratedDeriv_four] at h
  simpa using h.symm

/-- Exponential moment `E exp (-s^2/2) = sqrt (3/8)` for `s ~ N(0, 5/3)`. -/
theorem probe_exp_moment :
    ∫ t, Real.exp (-(t ^ 2) / 2) ∂(gaussianReal 0 (5 / 3 : NNReal)) = Real.sqrt (3 / 8) := by
  rw [integral_gaussianReal_eq_integral_smul (by norm_num)]
  have hpt : ∀ x : ℝ, gaussianPDFReal 0 (5 / 3 : NNReal) x • Real.exp (-(x ^ 2) / 2) =
      (Real.sqrt (10 * Real.pi / 3))⁻¹ * Real.exp (-(4 / 5) * x ^ 2) := by
    intro x
    rw [gaussianPDFReal_def, smul_eq_mul, mul_assoc, ← Real.exp_add]
    push_cast
    congr 2
    · congr 1; ring
    · ring
  simp_rw [hpt]
  rw [integral_const_mul, integral_gaussian, ← Real.sqrt_inv, ← Real.sqrt_mul (by positivity)]
  congr 1
  field_simp
  ring

/-- A map equal to a non-degenerate product Gaussian law is a.e.-measurable. -/
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

/-- The marginal law of `s`. -/
theorem probe_law_s (I : ConeMomentInterface) :
    @Measure.map _ _ I.mΩ _ I.s I.P = gaussianReal 0 (5 / 3 : NNReal) := by
  letI := I.mΩ
  haveI := I.isProb
  have hT := probe_aemeasurable_triple I
  have := AEMeasurable.map_map_of_aemeasurable (measurable_fst.aemeasurable) hT
  rw [I.sxy_law, Measure.map_fst_prod, measure_univ, one_smul] at this
  exact this.symm

/-- Full route test: `TraceMoments` for every inhabitant of the interface. -/
theorem probe_traceMoments (I : ConeMomentInterface) : TraceMoments I := by
  unfold TraceMoments
  letI := I.mΩ
  haveI := I.isProb
  have hs : AEMeasurable I.s I.P := (probe_aemeasurable_triple I).fst
  have hlaw := probe_law_s I
  refine ⟨?_, ?_, ?_⟩
  · have e : ∫ ω, I.s ω ^ 2 ∂I.P = ∫ t, t ^ 2 ∂(Measure.map I.s I.P) :=
      (integral_map (f := fun t : ℝ => t ^ 2) hs
        (by fun_prop : Measurable fun t : ℝ => t ^ 2).aestronglyMeasurable).symm
    rw [e, hlaw, probe_second_moment]; push_cast; ring
  · have e : ∫ ω, I.s ω ^ 4 ∂I.P = ∫ t, t ^ 4 ∂(Measure.map I.s I.P) :=
      (integral_map (f := fun t : ℝ => t ^ 4) hs
        (by fun_prop : Measurable fun t : ℝ => t ^ 4).aestronglyMeasurable).symm
    rw [e, hlaw, probe_fourth_moment]; push_cast; ring
  · have e : ∫ ω, Real.exp (-(I.s ω ^ 2) / 2) ∂I.P =
        ∫ t, Real.exp (-(t ^ 2) / 2) ∂(Measure.map I.s I.P) :=
      (integral_map (f := fun t : ℝ => Real.exp (-(t ^ 2) / 2)) hs
        (by fun_prop : Measurable fun t : ℝ => Real.exp (-(t ^ 2) / 2)).aestronglyMeasurable).symm
    rw [e, hlaw, probe_exp_moment]

#print axioms probe_traceMoments
