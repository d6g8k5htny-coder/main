import Specifications.Side24

/-! Probe 2 (Side24-covariance-density). Probe only; `sorry` allowed here, never elsewhere. -/

open UniversalLaw.Spec.Side24
open scoped RealInnerProductSpace

/-! ### A. IntegrandRatioBounds, full proof attempt (fix of probe 1, line 79). -/
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
    have hpb : 0 < (1 + epsilon) ^ b := Real.rpow_pos_of_pos (by linarith) _
    have hqb : 0 < (1 - epsilon) ^ b := Real.rpow_pos_of_pos h1e _
    have hae : a * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right ha5 hε.le
    have hbe : b * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right hb5' hε.le
    constructor
    · rw [le_div_iff₀ hpb]
      calc (1 - 13 * epsilon) * (1 + epsilon) ^ b
          ≤ (1 - 13 * epsilon) * (1 + epsilon) ^ 5 :=
            mul_le_mul_of_nonneg_left (hb5 b hb5') (by linarith)
        _ ≤ 1 - 5 * epsilon := by unfold epsilon; norm_num
        _ ≤ 1 - a * epsilon := by linarith
        _ ≤ (1 - epsilon) ^ a := hbern a ha1
    · rw [div_le_iff₀ hqb]
      calc (1 + epsilon) ^ a ≤ (1 + epsilon) ^ 5 := hb5 a ha5
        _ ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := by unfold epsilon; norm_num
        _ ≤ (1 + 28 * epsilon) * (1 - b * epsilon) :=
            mul_le_mul_of_nonneg_left (by linarith) (by linarith)
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

/-! ### B. Derivative route for the reference kernel (orders 1 and 2). -/
section DerivRoute

variable {d : ℕ}

theorem probe_kinf_contDiff : ContDiff ℝ (⊤ : ℕ∞) (Kinf d) := by
  unfold Kinf
  exact (contDiff_norm_sq.neg.div_const 2).exp

theorem probe_hasFDerivAt_kinf (y : E d) :
    HasFDerivAt (Kinf d)
      (Real.exp (-(‖y‖ ^ 2) / 2) • ((-(2 • innerSL ℝ y)).smulRight (2 : ℝ)⁻¹)) y := by
  have h1 : HasFDerivAt (fun z : E d => -(‖z‖ ^ 2) / 2)
      ((-(2 • innerSL ℝ y)).smulRight (2 : ℝ)⁻¹) y := by
    have := (hasStrictFDerivAt_norm_sq y).hasFDerivAt.neg.mul_const (2 : ℝ)⁻¹
    exact this.congr_of_eventuallyEq (Filter.Eventually.of_forall fun z => div_eq_mul_inv _ _)
  exact h1.exp

theorem probe_fderiv_kinf (y v : E d) :
    fderiv ℝ (Kinf d) y v = -(⟪y, v⟫ * Kinf d y) := by
  rw [(probe_hasFDerivAt_kinf y).fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply]
  ring

/-- Peeling the last direction off an iterated derivative of a `C^∞` real function. -/
theorem probe_peel (f : E d → ℝ) (hf : ContDiff ℝ (⊤ : ℕ∞) f) (n : ℕ) (x : E d)
    (m : Fin (n + 1) → E d) :
    iteratedFDeriv ℝ (n + 1) f x m =
      iteratedFDeriv ℝ n (fun y => fderiv ℝ f y (m (Fin.last n))) x (Fin.init m) := by
  rw [iteratedFDeriv_succ_apply_right]
  rw [iteratedFDeriv_clm_apply_const_apply (contDiff_infty_iff_fderiv.mp hf).2
    (mod_cast le_top)]

theorem probe_order_two (a b : E d) :
    derivContraction d (Kinf d) 0 [a, b] = -⟪a, b⟫ := by
  show iteratedFDeriv ℝ 2 (Kinf d) 0 (fun i => [a, b].get i) = -⟪a, b⟫
  rw [probe_peel _ probe_kinf_contDiff, iteratedFDeriv_one_apply]
  simp only [probe_fderiv_kinf]
  show fderiv ℝ (fun y : E d => -(⟪y, b⟫ * Kinf d y)) 0 a = -⟪a, b⟫
  have hin : HasFDerivAt (fun y : E d => ⟪y, b⟫) _ 0 :=
    (hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const b 0)
  have h : HasFDerivAt (fun y : E d => -(⟪y, b⟫ * Kinf d y)) _ 0 :=
    (hin.mul (probe_hasFDerivAt_kinf 0)).neg
  rw [h.fderiv]
  simp [Kinf, fderivInnerCLM_apply]

/-- Consequence shape for `gradCov`: the gradient block of the reference kernel. -/
theorem probe_gradCov (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    gradCov d (Kinf d) R = 1 := by
  ext i j
  simp only [gradCov, Matrix.of_apply, probe_order_two, neg_neg, Matrix.one_apply]
  exact orthonormal_iff_ite.mp hR i j

end DerivRoute

/-! ### C. ReferencePinDensityProduct: the density algebra, given the two block values. -/
theorem probe_density_algebra (d : ℕ) (hd : 0 < d) :
    gaussDensity d 1 0 *
        gaussDensity d (Matrix.diagonal fun i : Fin d => if i = ⟨0, hd⟩ then (3 : ℝ) else 1) 0 =
      (2 * Real.pi) ^ (-(d : ℝ)) / Real.sqrt 3 := by
  have hdet : (Matrix.diagonal fun i : Fin d => if i = ⟨0, hd⟩ then (3 : ℝ) else 1).det = 3 := by
    rw [Matrix.det_diagonal, Finset.prod_ite_eq']
    simp
  have hpi : (0 : ℝ) < 2 * Real.pi := by positivity
  simp only [gaussDensity, hdet, Matrix.det_one, Matrix.mulVec_zero, dotProduct_zero, neg_zero,
    zero_div, Real.exp_zero, mul_one, Real.one_rpow]
  rw [Real.sqrt_eq_rpow, ← Real.rpow_neg (by norm_num), show -(1 : ℝ) / 2 = -(1 / 2 : ℝ) by ring,
    Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 3), ← Real.rpow_add hpi, div_eq_mul_inv]
  congr 1
  ring_nf

/-! ### D. DensityComparison: Loewner-order ingredients over `ℝ`. -/
section Loewner
open scoped MatrixOrder

#check @Matrix.PosSemidef.nonneg
#check @CFC.sqrt_mul_sqrt_self
#check @CFC.sqrt_nonneg
#check @conjugate_le_conjugate
#check @Matrix.PosSemidef.posDef_iff_isUnit
#check @Matrix.IsHermitian.spectral_theorem
#check @Finset.one_le_prod''
#check @Matrix.PosSemidef.inv_sqrt
#check @CStarAlgebra.inv_le_inv

-- Does the C⋆-algebra inverse-ordering lemma apply to real matrices? (expected: no instance)
example (A B : (Matrix (Fin 3) (Fin 3) ℝ)ˣ) (hA : 0 ≤ (A : Matrix (Fin 3) (Fin 3) ℝ))
    (hAB : (A : Matrix (Fin 3) (Fin 3) ℝ) ≤ B) :
    ((B⁻¹ : (Matrix (Fin 3) (Fin 3) ℝ)ˣ) : Matrix (Fin 3) (Fin 3) ℝ) ≤ A⁻¹ :=
  CStarAlgebra.inv_le_inv hA hAB

-- Conjugation by a Hermitian matrix preserves the Loewner order (needed for the inverse step).
example (A B S : Matrix (Fin 3) (Fin 3) ℝ) (hAB : A ≤ B) : star S * A * S ≤ star S * B * S :=
  conjugate_le_conjugate hAB S

-- Order hypotheses of the Prop are exactly Loewner inequalities.
example (n : ℕ) (C0 C : Matrix (Fin n) (Fin n) ℝ) (h : (C - (1 - epsilon) • C0).PosSemidef) :
    (1 - epsilon) • C0 ≤ C := Matrix.le_iff.mpr h

end Loewner

/-! ### E. Index bookkeeping for the joint covariance (cardinalities). -/
example : Fintype.card (JointIndex 2) = 6 := by decide
example : Fintype.card (JointIndex 3) = 10 := by decide
