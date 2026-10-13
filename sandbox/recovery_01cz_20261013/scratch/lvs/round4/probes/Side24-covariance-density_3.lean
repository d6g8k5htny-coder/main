import Specifications.Side24

/-! Probe 3 (Side24-covariance-density). Probe only; `sorry` allowed here, never elsewhere. -/

open UniversalLaw.Spec.Side24
open scoped RealInnerProductSpace

section DerivRoute

variable {d : ℕ}

theorem probe_kinf_contDiff : ContDiff ℝ (⊤ : ℕ∞) (Kinf d) := by
  unfold Kinf
  exact ContDiff.exp (ContDiff.div_const (ContDiff.neg contDiff_norm_sq) 2)

theorem probe_hasFDerivAt_kinf (y : E d) :
    HasFDerivAt (Kinf d)
      (Real.exp (-(‖y‖ ^ 2) / 2) • ((2 : ℝ)⁻¹ • -((2 : ℕ) • innerSL ℝ y))) y := by
  have h1 : HasFDerivAt (fun z : E d => -(‖z‖ ^ 2) / 2) ((2 : ℝ)⁻¹ • -((2 : ℕ) • innerSL ℝ y)) y := by
    have := (hasStrictFDerivAt_norm_sq y).hasFDerivAt.neg.mul_const (2 : ℝ)⁻¹
    refine this.congr_of_eventuallyEq (Filter.Eventually.of_forall fun z => ?_)
    simp [div_eq_mul_inv]
  exact h1.exp

theorem probe_fderiv_kinf (y v : E d) :
    fderiv ℝ (Kinf d) y v = -(⟪y, v⟫ * Kinf d y) := by
  rw [(probe_hasFDerivAt_kinf y).fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply]
  ring

theorem probe_peel (f : E d → ℝ) (hf : ContDiff ℝ (⊤ : ℕ∞) f) (n : ℕ) (x : E d)
    (m : Fin (n + 1) → E d) :
    iteratedFDeriv ℝ (n + 1) f x m =
      iteratedFDeriv ℝ n (fun y => fderiv ℝ f y (m (Fin.last n))) x (Fin.init m) := by
  rw [iteratedFDeriv_succ_apply_right]
  rw [iteratedFDeriv_clm_apply_const_apply (contDiff_infty_iff_fderiv.mp hf).2
    (mod_cast le_top)]

/-! Level-one function `g₁ᵥ y = -(⟪y, v⟫ K y)`. -/

theorem probe_g1_contDiff (v : E d) :
    ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => -(⟪y, v⟫ * Kinf d y)) := by
  first
  | exact ((contDiff_id.inner ℝ contDiff_const).mul probe_kinf_contDiff).neg
  | exact ((contDiff_id.inner contDiff_const).mul probe_kinf_contDiff).neg

theorem probe_g1_hasFDerivAt (v y : E d) :
    HasFDerivAt (fun y : E d => -(⟪y, v⟫ * Kinf d y)) _ y :=
  (((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul (probe_hasFDerivAt_kinf y)).neg

theorem probe_fderiv_g1 (v y w : E d) :
    fderiv ℝ (fun y : E d => -(⟪y, v⟫ * Kinf d y)) y w =
      -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y := by
  rw [(probe_g1_hasFDerivAt v y).fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-! Level-two function `g₂ y = -(⟪w, v⟫ K y) + ⟪y, v⟫ ⟪y, w⟫ K y`. -/

theorem probe_g2_contDiff (v w : E d) :
    ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) := by
  first
  | exact ((probe_kinf_contDiff.const_mul _).neg.add
      (((contDiff_id.inner ℝ contDiff_const).mul (contDiff_id.inner ℝ contDiff_const)).mul
        probe_kinf_contDiff))
  | exact ((probe_kinf_contDiff.const_mul _).neg.add
      (((contDiff_id.inner contDiff_const).mul (contDiff_id.inner contDiff_const)).mul
        probe_kinf_contDiff))

theorem probe_g2_hasFDerivAt (v w y : E d) :
    HasFDerivAt (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) _ y :=
  (((probe_hasFDerivAt_kinf y).const_mul _).neg).add
    ((((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul
      ((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const w y))).mul (probe_hasFDerivAt_kinf y))

theorem probe_fderiv_g2 (v w y a : E d) :
    fderiv ℝ (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) y a =
      ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
        - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y := by
  rw [(probe_g2_hasFDerivAt v w y).fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-! Level-three function at the origin. -/

theorem probe_g3_hasFDerivAt (v w a y : E d) :
    HasFDerivAt (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) _ y :=
  (((((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const a y)).const_mul _).mul
      (probe_hasFDerivAt_kinf y)).add
    ((((((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const w y)).const_mul _).add
        (((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul_const _))).mul
      (probe_hasFDerivAt_kinf y))).sub
    (((((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul
        ((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const w y))).mul
        ((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const a y))).mul (probe_hasFDerivAt_kinf y))

theorem probe_fderiv_g3_zero (v w a c : E d) :
    fderiv ℝ (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0 c =
      ⟪w, v⟫ * ⟪c, a⟫ + ⟪a, v⟫ * ⟪c, w⟫ + ⟪c, v⟫ * ⟪a, w⟫ := by
  rw [(probe_g3_hasFDerivAt v w a 0).fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-- The fourth-order contraction of the reference kernel at zero: the three pairings. -/
theorem probe_order_four (c a w v : E d) :
    derivContraction d (Kinf d) 0 [c, a, w, v] =
      ⟪w, v⟫ * ⟪c, a⟫ + ⟪a, v⟫ * ⟪c, w⟫ + ⟪c, v⟫ * ⟪a, w⟫ := by
  show iteratedFDeriv ℝ 4 (Kinf d) 0 (fun i => [c, a, w, v].get i) = _
  rw [probe_peel _ probe_kinf_contDiff]
  simp only [probe_fderiv_kinf]
  show iteratedFDeriv ℝ 3 (fun y => -(⟪y, v⟫ * Kinf d y)) 0
    (Fin.init fun i => [c, a, w, v].get i) = _
  rw [probe_peel _ (probe_g1_contDiff v)]
  simp only [probe_fderiv_g1]
  show iteratedFDeriv ℝ 2 (fun y => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) 0
    (Fin.init (Fin.init fun i => [c, a, w, v].get i)) = _
  rw [probe_peel _ (probe_g2_contDiff v w)]
  simp only [probe_fderiv_g2]
  show iteratedFDeriv ℝ 1 (fun y => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0
    (Fin.init (Fin.init (Fin.init fun i => [c, a, w, v].get i))) = _
  rw [iteratedFDeriv_one_apply]
  show fderiv ℝ (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0 c = _
  exact probe_fderiv_g3_zero v w a c

/-- Consequences in an orthonormal frame: the pin block and the `t`–gradient vector. -/
theorem probe_pinCov (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    pinCov d (Kinf d) (R ⟨0, hd⟩) R = Matrix.diagonal (fun i => if i = ⟨0, hd⟩ then 3 else 1) := by
  have hite := orthonormal_iff_ite.mp hR
  ext i j
  simp only [pinCov, Matrix.of_apply, probe_order_four, hite, Matrix.diagonal_apply]
  by_cases hi : i = ⟨0, hd⟩ <;> by_cases hj : j = ⟨0, hd⟩ <;> simp [hi, hj] <;> norm_num
  all_goals first | (subst hi; simp_all) | (subst hj; simp_all) | simp_all [eq_comm]

theorem probe_thirdGradCov (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    thirdGradCov d (Kinf d) (R ⟨0, hd⟩) R = fun j => if j = ⟨0, hd⟩ then -3 else 0 := by
  have hite := orthonormal_iff_ite.mp hR
  funext j
  simp only [thirdGradCov, probe_order_four, hite]
  by_cases hj : j = ⟨0, hd⟩ <;> simp [hj] <;> norm_num

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
  rw [← mul_assoc, ← Real.rpow_add hpi, show -(d : ℝ) / 2 + -(d : ℝ) / 2 = -(d : ℝ) by ring,
    show (-1 : ℝ) / 2 = -(1 / 2) by norm_num, Real.rpow_neg (by norm_num : (0 : ℝ) ≤ 3),
    Real.sqrt_eq_rpow, div_eq_mul_inv]

/-! ### D. Loewner-order ingredients over `ℝ`. -/
section Loewner
open scoped MatrixOrder
#check @star_left_conjugate_le_conjugate
#check @IsSelfAdjoint.conjugate_le_conjugate
#check @conjugate_le_conjugate_of_nonneg
example (A B S : Matrix (Fin 3) (Fin 3) ℝ) (hAB : A ≤ B) : star S * A * S ≤ star S * B * S :=
  star_left_conjugate_le_conjugate hAB S
example (A : Matrix (Fin 3) (Fin 3) ℝ) (hA : A.PosDef) : CFC.sqrt A * CFC.sqrt A = A :=
  CFC.sqrt_mul_sqrt_self A hA.posSemidef.nonneg
end Loewner
