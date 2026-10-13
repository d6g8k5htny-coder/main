import Specifications.Side24

/-! Probe 1 (Side24-interfaces): name battery and a full route test for
`ConeMomentD1Identity`. Probe only; `sorry` allowed here, never elsewhere. -/

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

-- Name battery (existence at the pinned revision)
#check @MeasureTheory.Measure.map_of_not_aemeasurable_of_ne_zero
#check @MeasureTheory.Measure.dirac_apply_of_mem
#check @MeasureTheory.IsProbabilityMeasure.ne_zero
#check @ProbabilityTheory.nullSingletonClass_gaussianReal
#check @ProbabilityTheory.gaussianReal_map_neg
#check @ProbabilityTheory.variance_fun_id_gaussianReal
#check @ProbabilityTheory.variance_eq_integral
#check @ProbabilityTheory.integral_id_gaussianReal
#check @ProbabilityTheory.memLp_id_gaussianReal
#check @MeasureTheory.MemLp.integrable_sq
#check @MeasureTheory.integral_map
#check @MeasureTheory.integral_indicator
#check @MeasureTheory.integral_add_compl
#check @MeasureTheory.integral_Ici_eq_integral_Ioi
#check @MeasureTheory.Measure.map_fst_prod
#check @MeasureTheory.Measure.map_snd_prod
#check @AEMeasurable.map_map_of_aemeasurable
#check @AEMeasurable.fst
#check @AEMeasurable.snd
#check @MeasureTheory.Measure.map_apply_of_aemeasurable
#check @ProbabilityTheory.iteratedDeriv_mgf_zero
#check @ProbabilityTheory.mgf_fun_id_gaussianReal
#check @ProbabilityTheory.mgf_gaussianReal
#check @ProbabilityTheory.integrableExpSet_fun_id_gaussianReal
#check @ProbabilityTheory.integral_gaussianReal_eq_integral_smul
#check @ProbabilityTheory.gaussianPDFReal_def
#check @integral_gaussian
#check @integral_comp_polarCoord_symm
#check @polarCoord_symm_apply
#check @MeasureTheory.prod_withDensity
#check @MeasureTheory.integral_prod
#check @MeasureTheory.setIntegral_prod_mul
#check @MeasureTheory.Integrable.mul_prod
#check @MeasureTheory.Measure.prod_restrict
#check @Matrix.posDef_iff_dotProduct_mulVec
#check @Matrix.PosDef.det_pos
#check @Matrix.det_fin_two_of
#check @intervalIntegral.integral_comp_mul_deriv
#check @intervalIntegral.integral_eq_sub_of_hasDerivAt
#check @ProbabilityTheory.HasLaw
#check @ProbabilityTheory.HasLaw.integral_comp
#check @ProbabilityTheory.multivariateGaussian
#check @ProbabilityTheory.condDistrib
#check @MeasureTheory.integral_const_mul
#check @iteratedDeriv_succ
#check @iteratedDeriv_one
#check @Real.sqrt_inv
#check @Real.sqrt_mul

/-- A map equal to a non-degenerate Gaussian is a.e.-measurable: at this revision the map of a
non-a.e.-measurable function is a Dirac measure, which charges a singleton. -/
theorem probe_aemeasurable_of_map_eq_gaussian {Ω : Type} [MeasurableSpace Ω] (P : Measure Ω)
    [IsProbabilityMeasure P] (A : Ω → ℝ) {v : NNReal} (hv : v ≠ 0)
    (h : Measure.map A P = gaussianReal 0 v) : AEMeasurable A P := by
  by_contra hA
  obtain ⟨c, hc⟩ : ∃ c : ℝ, Measure.map A P = Measure.dirac c :=
    ⟨_, Measure.map_of_not_aemeasurable_of_ne_zero hA (IsProbabilityMeasure.ne_zero P)⟩
  rw [hc] at h
  haveI : NullSingletonClass (gaussianReal 0 v) := nullSingletonClass_gaussianReal hv
  have h1 : Measure.dirac c {c} = gaussianReal 0 v {c} := by rw [h]
  rw [Measure.dirac_apply_of_mem (Set.mem_singleton c), measure_singleton] at h1
  exact one_ne_zero h1

/-- Second moment of the centred Gaussian, from Mathlib's variance. -/
theorem probe_second_moment (v : NNReal) : ∫ t, t ^ 2 ∂(gaussianReal 0 v) = (v : ℝ) := by
  have h := variance_fun_id_gaussianReal (μ := 0) (v := v)
  rw [variance_eq_integral (by fun_prop)] at h
  simpa using h

/-- Full route test: `ConeMomentD1Identity` for every inhabitant of the interface. -/
theorem probe_coneMomentD1Identity (I : ConeMomentInterface) : ConeMomentD1Identity I := by
  unfold ConeMomentD1Identity
  letI := I.mΩ
  haveI := I.isProb
  have hv : ((8 : NNReal) / 3) ≠ 0 := by norm_num
  have hA : AEMeasurable I.A I.P := probe_aemeasurable_of_map_eq_gaussian I.P I.A hv I.A_law
  haveI : NullSingletonClass (gaussianReal 0 (8 / 3 : NNReal)) := nullSingletonClass_gaussianReal hv
  have hmeas : Measurable (fun a : ℝ => Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a) :=
    (measurable_id.pow_const 2).indicator measurableSet_Iio
  have e : ∫ ω, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) (I.A ω) ∂I.P =
      ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a ∂(Measure.map I.A I.P) :=
    (integral_map hA hmeas.aestronglyMeasurable).symm
  rw [e, I.A_law]
  set γ := gaussianReal 0 (8 / 3 : NNReal) with hγ
  have h2 : ∫ a, a ^ 2 ∂γ = 8 / 3 := by
    rw [hγ, probe_second_moment]; push_cast; ring
  have hint : Integrable (fun a : ℝ => a ^ 2) γ := by
    have := (memLp_id_gaussianReal (μ := 0) (v := (8 / 3 : NNReal)) 2).integrable_sq
    simpa using this
  have hneg : γ.map (fun x => -x) = γ := by rw [hγ, gaussianReal_map_neg, neg_zero]
  have hsymm : ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂γ = ∫ a in Set.Ioi (0 : ℝ), a ^ 2 ∂γ := by
    calc ∫ a in Set.Iio (0 : ℝ), a ^ 2 ∂γ
        = ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a ∂γ :=
          (integral_indicator measurableSet_Iio).symm
      _ = ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) a ∂(γ.map (fun x => -x)) := by
          rw [hneg]
      _ = ∫ a, Set.indicator (Set.Iio (0 : ℝ)) (fun a => a ^ 2) (-a) ∂γ :=
          integral_map measurable_neg.aemeasurable hmeas.aestronglyMeasurable
      _ = ∫ a, Set.indicator (Set.Ioi (0 : ℝ)) (fun a => a ^ 2) a ∂γ := by
          congr 1; funext a
          simp only [Set.indicator, Set.mem_Iio, Set.mem_Ioi, neg_lt_zero, neg_sq]
      _ = ∫ a in Set.Ioi (0 : ℝ), a ^ 2 ∂γ := integral_indicator measurableSet_Ioi
  have hsplit := integral_add_compl (μ := γ) (s := Set.Iio (0 : ℝ)) measurableSet_Iio hint
  rw [Set.compl_Iio, integral_Ici_eq_integral_Ioi, ← hsymm, h2] at hsplit
  rw [integral_indicator measurableSet_Iio]
  unfold coneMomentD1
  linarith

#print axioms probe_coneMomentD1Identity
