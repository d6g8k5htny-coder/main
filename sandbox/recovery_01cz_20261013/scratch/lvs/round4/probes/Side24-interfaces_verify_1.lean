import Specifications.Side24

/-! Verify probe 1 (Side24-interfaces_verify): adversarial checks on the four
`ConeMomentInterface` Props.
(a) the interface is inhabited (so none of the four Props is vacuous);
(b) name battery for the `ConeMomentD2Identity` / `RayleighSquareLaw` routes beyond probe 1;
(c) the cone converse `s < 0 ∧ x² + y² < s² → NegDefCone (s, x, y)` (hard step (3) of the D2 route);
(d) how the variances `8/3`, `5/3` elaborate in the interface fields.
Probe only; `sorry` allowed here, never elsewhere. -/

open MeasureTheory ProbabilityTheory UniversalLaw.Spec.Side24

-- (d) elaborated field types
#check @ConeMomentInterface.A_law
#check @ConeMomentInterface.sxy_law
#check @ConeMomentD1Identity
#check @ConeMomentD2Identity
#check @TraceMoments
#check @RayleighSquareLaw

-- (b) name battery
#check @lintegral_comp_polarCoord_symm
#check @polarCoord_target
#check @MeasureTheory.withDensity_apply
#check @MeasureTheory.lintegral_withDensity_eq_lintegral_mul
#check @MeasureTheory.integral_withDensity_eq_integral_smul
#check @ProbabilityTheory.measurable_gaussianPDF
#check @ProbabilityTheory.gaussianPDF_def
#check @ProbabilityTheory.gaussianPDFReal_pos
#check @MeasureTheory.ofReal_integral_eq_lintegral_ofReal
#check @MeasureTheory.lintegral_prod
#check @MeasureTheory.Measure.prod_apply
#check @MeasureTheory.Measure.prod_restrict
#check @MeasureTheory.Measure.restrict_prod_eq_prod_univ
#check @MeasureTheory.lintegral_indicator
#check @MeasureTheory.lintegral_indicator_one
#check @Real.volume_Ioo
#check @Real.cos_sq_add_sin_sq
#check @Matrix.det_neg
#check @Matrix.PosDef.diag_pos
#check @Matrix.PosDef.det_pos
#check @Matrix.posDef_iff_dotProduct_mulVec
#check @MeasureTheory.Integrable.mul_prod
#check @MeasureTheory.integral_prod
#check @MeasureTheory.Integrable.mono
#check @MeasureTheory.Integrable.indicator
#check @intervalIntegral.integral_comp_mul_deriv
#check @MeasureTheory.integral_Ioi_eq_integral_Ioc
#check @intervalIntegral.integral_of_le
#check @MeasureTheory.integral_Ioc_eq_integral_Ioo

-- (a) the interface is inhabited: product of the four Gaussians with the projections
noncomputable def probe_inhabitant : ConeMomentInterface where
  Ω := ℝ × (ℝ × ℝ × ℝ)
  P := (gaussianReal 0 (8 / 3)).prod
    ((gaussianReal 0 (5 / 3)).prod ((gaussianReal 0 1).prod (gaussianReal 0 1)))
  A := Prod.fst
  A_law := by rw [Measure.map_fst_prod, measure_univ, one_smul]
  s := fun ω => ω.2.1
  x := fun ω => ω.2.2.1
  y := fun ω => ω.2.2.2
  sxy_law := by
    have e : (fun ω : ℝ × (ℝ × ℝ × ℝ) => (ω.2.1, ω.2.2.1, ω.2.2.2)) = Prod.snd := by
      funext ω; rfl
    rw [e, Measure.map_snd_prod, measure_univ, one_smul]

theorem probe_interface_nonempty : Nonempty ConeMomentInterface := ⟨probe_inhabitant⟩

#print axioms probe_interface_nonempty

-- (c) cone converse
theorem probe_det_transverse (s x y : ℝ) :
    (transverseMatrix s x y).det = s ^ 2 - x ^ 2 - y ^ 2 := by
  unfold transverseMatrix
  rw [Matrix.det_fin_two_of]
  ring

theorem probe_cone_converse (s x y : ℝ) (hs : s < 0) (h : x ^ 2 + y ^ 2 < s ^ 2) :
    NegDefCone (s, x, y) := by
  unfold NegDefCone
  rw [Matrix.posDef_iff_dotProduct_mulVec]
  constructor
  · unfold Matrix.IsHermitian transverseMatrix
    ext i j
    fin_cases i <;> fin_cases j <;> simp
  · intro v hv
    have hv2 : 0 < v 0 ^ 2 + v 1 ^ 2 := by
      by_contra hcon
      push_neg at hcon
      have h0 : v 0 = 0 := by nlinarith [sq_nonneg (v 0), sq_nonneg (v 1)]
      have h1 : v 1 = 0 := by nlinarith [sq_nonneg (v 0), sq_nonneg (v 1)]
      apply hv
      funext i
      fin_cases i <;> simp [h0, h1]
    have hRnn : 0 ≤ Real.sqrt (x ^ 2 + y ^ 2) := Real.sqrt_nonneg _
    have hR : Real.sqrt (x ^ 2 + y ^ 2) < -s := by
      rw [Real.sqrt_lt' (by linarith)]
      nlinarith
    have hL : (x * (v 0 ^ 2 - v 1 ^ 2) + y * (2 * v 0 * v 1)) ^ 2 ≤
        (x ^ 2 + y ^ 2) * (v 0 ^ 2 + v 1 ^ 2) ^ 2 := by
      nlinarith [sq_nonneg (x * (2 * v 0 * v 1) - y * (v 0 ^ 2 - v 1 ^ 2))]
    have key : x * (v 0 ^ 2 - v 1 ^ 2) + y * (2 * v 0 * v 1) ≤
        Real.sqrt (x ^ 2 + y ^ 2) * (v 0 ^ 2 + v 1 ^ 2) := by
      have := Real.abs_le_sqrt hL
      rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (by positivity)] at this
      exact le_trans (le_abs_self _) this
    have hpos : 0 < (-s - Real.sqrt (x ^ 2 + y ^ 2)) * (v 0 ^ 2 + v 1 ^ 2) :=
      mul_pos (by linarith) hv2
    simp only [transverseMatrix, Matrix.mulVec, dotProduct, Fin.sum_univ_two, Matrix.neg_apply,
      Matrix.of_apply, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, star_trivial,
      Matrix.cons_val_fin_one]
    nlinarith [key, hpos]

theorem probe_cone_iff (s x y : ℝ) :
    NegDefCone (s, x, y) ↔ s < 0 ∧ x ^ 2 + y ^ 2 < s ^ 2 := by
  constructor
  · intro h
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
  · rintro ⟨hs, h⟩
    exact probe_cone_converse s x y hs h

#print axioms probe_cone_iff
