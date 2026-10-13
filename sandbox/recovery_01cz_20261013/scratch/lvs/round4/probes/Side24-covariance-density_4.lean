import Specifications.Side24

/-! Probe 4 (Side24-covariance-density). Probe only; `sorry` allowed here, never elsewhere.
Full-proof attempts for `ReferenceOddAndPinBlocks` and `ReferencePinDensityProduct`. -/

open UniversalLaw.Spec.Side24
open scoped RealInnerProductSpace

section DerivRoute

variable {d : ℕ}

theorem probe_kinf_contDiff : ContDiff ℝ (⊤ : ℕ∞) (Kinf d) := by
  unfold Kinf
  exact ContDiff.exp (ContDiff.div_const (ContDiff.neg (contDiff_norm_sq ℝ)) 2)

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

theorem probe_inner_contDiff (v : E d) : ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => ⟪y, v⟫) :=
  contDiff_id.inner ℝ contDiff_const

/-! Level one: `g₁ᵥ y = -(⟪y, v⟫ K y)`. -/

theorem probe_g1_contDiff (v : E d) :
    ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => -(⟪y, v⟫ * Kinf d y)) :=
  ((probe_inner_contDiff v).mul probe_kinf_contDiff).neg

theorem probe_fderiv_g1 (v y w : E d) :
    fderiv ℝ (fun y : E d => -(⟪y, v⟫ * Kinf d y)) y w =
      -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y := by
  have h : HasFDerivAt (fun y : E d => -(⟪y, v⟫ * Kinf d y)) _ y :=
    (((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul (probe_hasFDerivAt_kinf y)).neg
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-! Level two: `g₂ y = -(⟪w, v⟫ K y) + ⟪y, v⟫ ⟪y, w⟫ K y`. -/

theorem probe_g2_contDiff (v w : E d) :
    ContDiff ℝ (⊤ : ℕ∞) (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) :=
  ((contDiff_const.mul probe_kinf_contDiff).neg.add
    (((probe_inner_contDiff v).mul (probe_inner_contDiff w)).mul probe_kinf_contDiff))

theorem probe_fderiv_g2 (v w y a : E d) :
    fderiv ℝ (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) y a =
      ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
        - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y := by
  have h : HasFDerivAt (fun y : E d => -(⟪w, v⟫ * Kinf d y) + ⟪y, v⟫ * ⟪y, w⟫ * Kinf d y) _ y :=
    (((probe_hasFDerivAt_kinf y).const_mul _).neg).add
      ((((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const v y)).mul
        ((hasFDerivAt_id y).inner ℝ (hasFDerivAt_const w y))).mul (probe_hasFDerivAt_kinf y))
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-! Level three at the origin (general directions). -/

theorem probe_fderiv_g3_zero (v w a c : E d) :
    fderiv ℝ (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) 0 c =
      ⟪w, v⟫ * ⟪c, a⟫ + ⟪a, v⟫ * ⟪c, w⟫ + ⟪c, v⟫ * ⟪a, w⟫ := by
  have h : HasFDerivAt (fun y : E d => ⟪w, v⟫ * ⟪y, a⟫ * Kinf d y
      + (⟪a, v⟫ * ⟪y, w⟫ + ⟪y, v⟫ * ⟪a, w⟫) * Kinf d y
      - ⟪y, v⟫ * ⟪y, w⟫ * ⟪y, a⟫ * Kinf d y) _ 0 :=
    (((((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const a 0)).const_mul _).mul
        (probe_hasFDerivAt_kinf 0)).add
      ((((((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const w 0)).const_mul _).add
          (((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const v 0)).mul_const _))).mul
        (probe_hasFDerivAt_kinf 0))).sub
      (((((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const v 0)).mul
          ((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const w 0))).mul
          ((hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const a 0))).mul
        (probe_hasFDerivAt_kinf 0))
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-- Fourth-order contraction at zero: the three pairings. -/
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

/-! Levels three to five along one direction `u` (Hermite forms), for the sixth derivative. -/

theorem probe_P3_contDiff (u : E d) : ContDiff ℝ (⊤ : ℕ∞)
    (fun y : E d => (3 * ⟪u, u⟫ * ⟪y, u⟫ - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) :=
  ((contDiff_const.mul (probe_inner_contDiff u)).sub
    (((probe_inner_contDiff u).mul (probe_inner_contDiff u)).mul (probe_inner_contDiff u))).mul
    probe_kinf_contDiff

theorem probe_fderiv_P3 (u y : E d) :
    fderiv ℝ (fun y : E d => (3 * ⟪u, u⟫ * ⟪y, u⟫ - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) y u =
      (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y := by
  have hin : HasFDerivAt (fun y : E d => ⟪y, u⟫) _ y :=
    (hasFDerivAt_id y).inner ℝ (hasFDerivAt_const u y)
  have h : HasFDerivAt
      (fun y : E d => (3 * ⟪u, u⟫ * ⟪y, u⟫ - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) _ y :=
    ((hin.const_mul _).sub ((hin.mul hin).mul hin)).mul (probe_hasFDerivAt_kinf y)
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

theorem probe_P4_contDiff (u : E d) : ContDiff ℝ (⊤ : ℕ∞)
    (fun y : E d => (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) :=
  ((contDiff_const.sub (contDiff_const.mul ((probe_inner_contDiff u).mul (probe_inner_contDiff u)))).add
    ((((probe_inner_contDiff u).mul (probe_inner_contDiff u)).mul (probe_inner_contDiff u)).mul
      (probe_inner_contDiff u))).mul probe_kinf_contDiff

theorem probe_fderiv_P4 (u y : E d) :
    fderiv ℝ (fun y : E d => (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) y u =
      (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫ + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y := by
  have hin : HasFDerivAt (fun y : E d => ⟪y, u⟫) _ y :=
    (hasFDerivAt_id y).inner ℝ (hasFDerivAt_const u y)
  have h : HasFDerivAt (fun y : E d => (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) _ y :=
    (((hasFDerivAt_const _ y).sub ((hin.mul hin).const_mul _)).add
      (((hin.mul hin).mul hin).mul hin)).mul (probe_hasFDerivAt_kinf y)
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

theorem probe_fderiv_P5_zero (u : E d) :
    fderiv ℝ (fun y : E d => (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫ + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) 0 u =
      -15 * (⟪u, u⟫ * ⟪u, u⟫ * ⟪u, u⟫) := by
  have hin : HasFDerivAt (fun y : E d => ⟪y, u⟫) _ (0 : E d) :=
    (hasFDerivAt_id (0 : E d)).inner ℝ (hasFDerivAt_const u 0)
  have h : HasFDerivAt (fun y : E d => (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫
        + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) _ 0 :=
    (((hin.const_mul _).add (((hin.mul hin).mul hin).const_mul _)).sub
      ((((hin.mul hin).mul hin).mul hin).mul hin)).mul (probe_hasFDerivAt_kinf 0)
  rw [h.fderiv]
  simp [Kinf, two_smul, innerSL_apply_apply, fderivInnerCLM_apply]
  ring

/-- Sixth-order contraction along one direction at zero: `-15 ‖u‖⁶`. -/
theorem probe_order_six (u : E d) :
    derivContraction d (Kinf d) 0 [u, u, u, u, u, u] = -15 * (⟪u, u⟫ * ⟪u, u⟫ * ⟪u, u⟫) := by
  show iteratedFDeriv ℝ 6 (Kinf d) 0 (fun i => [u, u, u, u, u, u].get i) = _
  rw [probe_peel _ probe_kinf_contDiff]
  simp only [probe_fderiv_kinf]
  show iteratedFDeriv ℝ 5 (fun y => -(⟪y, u⟫ * Kinf d y)) 0 _ = _
  rw [probe_peel _ (probe_g1_contDiff u)]
  simp only [probe_fderiv_g1]
  show iteratedFDeriv ℝ 4 (fun y => -(⟪u, u⟫ * Kinf d y) + ⟪y, u⟫ * ⟪y, u⟫ * Kinf d y) 0 _ = _
  rw [probe_peel _ (probe_g2_contDiff u u)]
  simp only [probe_fderiv_g2]
  show iteratedFDeriv ℝ 3 (fun y => ⟪u, u⟫ * ⟪y, u⟫ * Kinf d y
      + (⟪u, u⟫ * ⟪y, u⟫ + ⟪y, u⟫ * ⟪u, u⟫) * Kinf d y
      - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * Kinf d y) 0 _ = _
  rw [show (fun y : E d => ⟪u, u⟫ * ⟪y, u⟫ * Kinf d y
      + (⟪u, u⟫ * ⟪y, u⟫ + ⟪y, u⟫ * ⟪u, u⟫) * Kinf d y
      - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * Kinf d y) =
      fun y : E d => (3 * ⟪u, u⟫ * ⟪y, u⟫ - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y from
      funext fun y => by ring]
  rw [probe_peel _ (probe_P3_contDiff u)]
  simp only [probe_fderiv_P3]
  show iteratedFDeriv ℝ 2 (fun y : E d => (3 * ⟪u, u⟫ * ⟪u, u⟫ - 6 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫)
        + ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) 0 _ = _
  rw [probe_peel _ (probe_P4_contDiff u)]
  simp only [probe_fderiv_P4]
  show iteratedFDeriv ℝ 1 (fun y : E d => (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫
        + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) 0 _ = _
  rw [iteratedFDeriv_one_apply]
  show fderiv ℝ (fun y : E d => (-15 * (⟪u, u⟫ * ⟪u, u⟫) * ⟪y, u⟫
        + 10 * ⟪u, u⟫ * (⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫)
        - ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫ * ⟪y, u⟫) * Kinf d y) 0 u = _
  exact probe_fderiv_P5_zero u

/-! Blocks in an orthonormal frame. -/

theorem probe_order_two (a b : E d) : derivContraction d (Kinf d) 0 [a, b] = -⟪a, b⟫ := by
  show iteratedFDeriv ℝ 2 (Kinf d) 0 (fun i => [a, b].get i) = -⟪a, b⟫
  rw [probe_peel _ probe_kinf_contDiff, iteratedFDeriv_one_apply]
  simp only [probe_fderiv_kinf]
  show fderiv ℝ (fun y : E d => -(⟪y, b⟫ * Kinf d y)) 0 a = -⟪a, b⟫
  rw [probe_fderiv_g1]
  simp [Kinf]

theorem probe_gradCov (R : Fin d → E d) (hR : Orthonormal ℝ R) : gradCov d (Kinf d) R = 1 := by
  ext i j
  simp only [gradCov, Matrix.of_apply, probe_order_two, neg_neg, Matrix.one_apply]
  exact orthonormal_iff_ite.mp hR i j

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
  by_cases hj : j = ⟨0, hd⟩
  · simp [hj]; norm_num
  · have hj' : ¬ ((⟨0, hd⟩ : Fin d) = j) := fun h => hj h.symm
    simp [hj, hj']

theorem probe_thirdVar (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    thirdVar d (Kinf d) (R ⟨0, hd⟩) = 15 := by
  have hite := orthonormal_iff_ite.mp hR
  simp only [thirdVar, probe_order_six, hite]
  simp

theorem probe_cond (hd : 0 < d) (R : Fin d → E d) (hR : Orthonormal ℝ R) :
    condThirdVariance d (Kinf d) (R ⟨0, hd⟩) R = 6 := by
  unfold condThirdVariance
  rw [probe_thirdVar hd R hR, probe_gradCov R hR, probe_thirdGradCov hd R hR, inv_one,
    Matrix.one_mulVec]
  simp [dotProduct, ite_mul, Finset.sum_ite_eq']
  norm_num

end DerivRoute

/-- Full-proof attempt of the registered Prop. -/
theorem probe_referenceOddAndPinBlocks : UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks := by
  intro d _ hd R hR
  exact ⟨probe_gradCov R hR, probe_thirdVar hd R hR, probe_thirdGradCov hd R hR,
    probe_pinCov hd R hR, probe_cond hd R hR⟩

/-! ### ReferencePinDensityProduct -/

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
    Real.sqrt_eq_rpow]
  exact (div_eq_mul_inv _ _).symm

/-- Full-proof attempt of the registered Prop. -/
theorem probe_referencePinDensityProduct : UniversalLaw.Spec.Side24.ReferencePinDensityProduct := by
  intro d _ hd R hR
  rw [probe_gradCov R hR, probe_pinCov hd R hR]
  exact probe_density_algebra d hd

#print axioms probe_referenceOddAndPinBlocks
#print axioms probe_referencePinDensityProduct
