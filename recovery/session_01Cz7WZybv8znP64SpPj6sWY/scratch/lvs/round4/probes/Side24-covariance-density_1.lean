import Specifications.Side24

/-! Probe 1 (Side24-covariance-density): name checks and two route tests.
    Probe only; `sorry` allowed here, never elsewhere. -/

open UniversalLaw.Spec.Side24
open scoped RealInnerProductSpace

-- Name battery (existence at the pinned revision)
#check @iteratedFDeriv_succ_apply_right
#check @iteratedFDeriv_clm_apply_const_apply
#check @iteratedFDeriv_one_apply
#check @iteratedFDeriv_two_apply
#check @ContDiff.fderiv_right
#check @hasStrictFDerivAt_norm_sq
#check @HasFDerivAt.inner
#check @HasFDerivAt.exp
#check @fderivInnerCLM_apply
#check @orthonormal_iff_ite
#check @one_add_mul_self_le_rpow_one_add
#check @Real.rpow_le_rpow_of_exponent_le
#check @Real.rpow_natCast
#check @Matrix.det_diagonal
#check @Matrix.posSemidef_iff_dotProduct_mulVec
#check @Matrix.le_iff
#check @Matrix.PosDef.det_pos
#check @Matrix.PosDef.inv
#check @Matrix.PosDef.smul
#check @Matrix.PosSemidef.conjTranspose_mul_mul_same
#check @Matrix.IsHermitian.det_eq_prod_eigenvalues
#check @Matrix.IsHermitian.eigenvalues_mem_spectrum_real
#check @Matrix.posSemidef_iff_isHermitian_and_spectrum_nonneg
#check @Matrix.PosSemidef.det_sqrt
#check @spectrum.sub_singleton_eq
#check @iteratedFDeriv_tsum
#check @norm_iteratedFDeriv_comp_le
#check @ContinuousLinearMap.iteratedFDeriv_comp_right

-- Does the matrix order / CFC sqrt elaborate for real matrices?
open scoped MatrixOrder in
example (A : Matrix (Fin 3) (Fin 3) ℝ) (hA : A.PosSemidef) : (CFC.sqrt A).det = RCLike.sqrt A.det :=
  Matrix.PosSemidef.det_sqrt hA

-- Route test A: IntegrandRatioBounds, full proof attempt.
theorem probe_integrandRatioBounds : UniversalLaw.Spec.Side24.IntegrandRatioBounds := by
  intro d hd
  have hε : (0 : ℝ) < epsilon := by unfold epsilon; positivity
  have hε1 : epsilon < 1 / 100 := by unfold epsilon; norm_num
  have h1e : (0 : ℝ) < 1 - epsilon := by linarith
  have h1e' : (1 : ℝ) ≤ 1 + epsilon := by linarith
  have hb5 : ∀ b : ℝ, b ≤ 5 → (1 + epsilon) ^ b ≤ (1 + epsilon) ^ 5 := by
    intro b hb
    have := Real.rpow_le_rpow_of_exponent_le h1e' hb
    rwa [show (5 : ℝ) = ((5 : ℕ) : ℝ) by norm_num, Real.rpow_natCast] at this
  have hbern : ∀ a : ℝ, 1 ≤ a → 1 - a * epsilon ≤ (1 - epsilon) ^ a := by
    intro a ha
    have := one_add_mul_self_le_rpow_one_add (show (-1 : ℝ) ≤ -epsilon by linarith) ha
    rw [show (1 : ℝ) + -epsilon = 1 - epsilon by ring] at this
    linarith
  have key : ∀ a b : ℝ, 1 ≤ a → a ≤ 5 → 1 ≤ b → b ≤ 5 →
      1 - 13 * epsilon ≤ (1 - epsilon) ^ a / (1 + epsilon) ^ b ∧
      (1 + epsilon) ^ a / (1 - epsilon) ^ b ≤ 1 + 28 * epsilon := by
    intro a b ha1 ha5 hb1 hb5'
    have hpa : 0 < (1 + epsilon) ^ a := Real.rpow_pos_of_pos (by linarith) _
    have hpb : 0 < (1 + epsilon) ^ b := Real.rpow_pos_of_pos (by linarith) _
    have hqa : 0 < (1 - epsilon) ^ a := Real.rpow_pos_of_pos h1e _
    have hqb : 0 < (1 - epsilon) ^ b := Real.rpow_pos_of_pos h1e _
    constructor
    · rw [le_div_iff₀ hpb]
      calc (1 - 13 * epsilon) * (1 + epsilon) ^ b
          ≤ (1 - 13 * epsilon) * (1 + epsilon) ^ 5 :=
            mul_le_mul_of_nonneg_left (hb5 b hb5') (by linarith)
        _ ≤ 1 - 5 * epsilon := by unfold epsilon; norm_num
        _ ≤ 1 - a * epsilon := by nlinarith
        _ ≤ (1 - epsilon) ^ a := hbern a ha1
    · rw [div_le_iff₀ hqb]
      calc (1 + epsilon) ^ a ≤ (1 + epsilon) ^ 5 := hb5 a ha5
        _ ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := by unfold epsilon; norm_num
        _ ≤ (1 + 28 * epsilon) * (1 - b * epsilon) := by nlinarith
        _ ≤ (1 + 28 * epsilon) * (1 - epsilon) ^ b :=
            mul_le_mul_of_nonneg_left (hbern b hb1) (by linarith)
  rcases hd with rfl | rfl
  · have ha : exponentA 2 = 13 / 6 := by simp [exponentA, symCoordDim]; norm_num
    have hb : exponentB 2 = 5 / 2 := by simp [exponentB, symCoordDim]; norm_num
    rw [ha, hb]
    exact key _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · have ha : exponentA 3 = 25 / 6 := by simp [exponentA, symCoordDim]; norm_num
    have hb : exponentB 3 = 9 / 2 := by simp [exponentB, symCoordDim]; norm_num
    rw [ha, hb]
    exact key _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

-- Route test B: order-two derivative contraction of the reference kernel at zero.
section DerivRoute

variable {d : ℕ}

theorem probe_kinf_contDiff : ContDiff ℝ (⊤ : ℕ∞) (Kinf d) := by
  unfold Kinf; fun_prop

theorem probe_fderiv_kinf (y v : E d) :
    fderiv ℝ (Kinf d) y v = -(⟪y, v⟫ * Kinf d y) := by
  have h : HasFDerivAt (Kinf d)
      (Real.exp (-(‖y‖ ^ 2) / 2) • ((1 / (2 : ℝ)) • (-(2 • innerSL ℝ y)))) y := by
    unfold Kinf
    exact ((hasStrictFDerivAt_norm_sq y).hasFDerivAt.neg.div_const 2).exp
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply]
  ring

/-- Peeling the last direction off an iterated derivative of a smooth real function. -/
theorem probe_peel (f : E d → ℝ) (hf : ContDiff ℝ (⊤ : ℕ∞) f) (n : ℕ) (x : E d)
    (m : Fin (n + 1) → E d) :
    iteratedFDeriv ℝ (n + 1) f x m =
      iteratedFDeriv ℝ n (fun y => fderiv ℝ f y (m (Fin.last n))) x (Fin.init m) := by
  rw [iteratedFDeriv_succ_apply_right]
  rw [iteratedFDeriv_clm_apply_const_apply (hf.fderiv_right (m := (⊤ : ℕ∞)) le_top) le_top]

theorem probe_order_two (a b : E d) :
    derivContraction d (Kinf d) 0 [a, b] = -⟪a, b⟫ := by
  unfold derivContraction
  simp only [List.length_cons, List.length_nil]
  rw [probe_peel _ probe_kinf_contDiff]
  rw [iteratedFDeriv_one_apply]
  have hcongr : (fun y => fderiv ℝ (Kinf d) y ([a, b].get (Fin.last 1))) =
      fun y => -(⟪y, b⟫ * Kinf d y) := by
    funext y; rw [probe_fderiv_kinf]; rfl
  rw [hcongr]
  have hK : HasFDerivAt (Kinf d) (Real.exp (-(‖(0 : E d)‖ ^ 2) / 2) •
      ((1 / (2 : ℝ)) • (-(2 • innerSL ℝ (0 : E d))))) 0 := by
    unfold Kinf
    exact ((hasStrictFDerivAt_norm_sq (0 : E d)).hasFDerivAt.neg.div_const 2).exp
  have hin : HasFDerivAt (fun y : E d => ⟪y, b⟫)
      ((fderivInnerCLM ℝ ((0 : E d), b)).comp ((ContinuousLinearMap.id ℝ (E d)).prod 0)) 0 :=
    (hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const b 0)
  have h := (hin.mul hK).neg
  rw [h.fderiv]
  simp [Kinf, fderivInnerCLM_apply, Fin.init]

end DerivRoute
