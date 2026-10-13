import Specifications.Side24

/-! Verify probe 1 (Side24-covariance-density_verify). Probe only; `sorry` allowed here, never
elsewhere. (A) `IntegrandRatioBounds` in isolation with `#print axioms`. (B) DensityComparison
route: the real-matrix Loewner-order ingredients, and a full proof of the spectral step
`1 ≤ M → 1 ≤ det M` that L2 needs. -/

open UniversalLaw.Spec.Side24

/-! ### A. IntegrandRatioBounds, exactly the probe-2 text. -/
theorem verify_integrandRatioBounds : UniversalLaw.Spec.Side24.IntegrandRatioBounds := by
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

#print axioms verify_integrandRatioBounds

/-! ### B. DensityComparison route: real-matrix Loewner ingredients. -/
section Loewner
open scoped MatrixOrder Pointwise

variable {n : ℕ}

-- Hypothesis shape of the Prop is exactly a Loewner inequality.
example (C0 C : Matrix (Fin n) (Fin n) ℝ) (h : (C - (1 - epsilon) • C0).PosSemidef) :
    (1 - epsilon) • C0 ≤ C := Matrix.le_iff.mpr h

-- CFC square root for REAL matrices: existence and the two identities used.
example (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.PosDef) : CFC.sqrt A * CFC.sqrt A = A :=
  CFC.sqrt_mul_sqrt_self A hA.posSemidef.nonneg
example (A : Matrix (Fin n) (Fin n) ℝ) : 0 ≤ CFC.sqrt A := CFC.sqrt_nonneg A
example (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.PosSemidef) : (CFC.sqrt A)⁻¹ = CFC.sqrt A⁻¹ :=
  hA.inv_sqrt
example (A : Matrix (Fin n) (Fin n) ℝ) (hA : A.PosSemidef) :
    (CFC.sqrt A).det = RCLike.sqrt A.det := hA.det_sqrt

-- Conjugation preserves the Loewner order (real matrices).
example (A B S : Matrix (Fin n) (Fin n) ℝ) (hAB : A ≤ B) : star S * A * S ≤ star S * B * S :=
  star_left_conjugate_le_conjugate hAB S

-- Side conditions used by the pointwise step.
example (C0 : Matrix (Fin n) (Fin n) ℝ) (h : C0.PosDef) : ((1 - epsilon) • C0).PosDef :=
  h.smul (by unfold epsilon; norm_num)
example (C : Matrix (Fin n) (Fin n) ℝ) (h : C.PosDef) : C⁻¹.PosDef := h.inv
example (C : Matrix (Fin n) (Fin n) ℝ) (h : C.PosDef) : 0 < C.det := h.det_pos
example (C : Matrix (Fin n) (Fin n) ℝ) (c : ℝ) : (c • C).det = c ^ n * C.det := by
  rw [Matrix.det_smul, Fintype.card_fin]
example (M : Matrix (Fin n) (Fin n) ℝ) (hM : M.PosSemidef) (x : Fin n → ℝ) :
    0 ≤ dotProduct x (M.mulVec x) := by simpa using hM.dotProduct_mulVec_nonneg x

-- The C⋆-algebra inverse lemma needs `[CStarAlgebra A]`; real matrices have no such instance.
#check @CStarAlgebra.inv_le_inv

/-- The spectral step of L2, proved: a real matrix `≥ 1` in the Loewner order has `det ≥ 1`. -/
theorem verify_one_le_det (M : Matrix (Fin n) (Fin n) ℝ) (hM : 1 ≤ M) : 1 ≤ M.det := by
  have hpsd : (M - 1).PosSemidef := Matrix.le_iff.mp hM
  have hherm : M.IsHermitian := by
    have h := hpsd.1.add (Matrix.isHermitian_one (n := Fin n) (α := ℝ))
    simpa using h
  rw [hherm.det_eq_prod_eigenvalues]
  apply Finset.one_le_prod₀
  intro i _
  have hspec := (Matrix.posSemidef_iff_isHermitian_and_spectrum_nonneg.mp hpsd).2
  have hmem : hherm.eigenvalues i ∈ spectrum ℝ M := hherm.eigenvalues_mem_spectrum_real i
  have hmem' : hherm.eigenvalues i - 1 ∈ spectrum ℝ (M - 1) := by
    have : hherm.eigenvalues i - 1 ∈ spectrum ℝ M - ({1} : Set ℝ) :=
      Set.sub_mem_sub hmem (Set.mem_singleton (1 : ℝ))
    rw [spectrum.sub_singleton_eq] at this
    simpa using this
  have := hspec hmem'
  simp only [Set.mem_setOf_eq] at this
  simp only [RCLike.ofReal_real_eq_id, id]
  linarith

-- Statement shapes of L1 and L2 (types only; `sorry` is allowed in probes).
example (A C : Matrix (Fin n) (Fin n) ℝ) (hA : A.PosDef) (h : A ≤ C) : A.det ≤ C.det := by
  sorry
example (A C : Matrix (Fin n) (Fin n) ℝ) (hA : A.PosDef) (h : A ≤ C) : C⁻¹ ≤ A⁻¹ := by
  sorry

end Loewner

-- Degenerate index `m = 0`: both sides of the Prop are `1`, so no degeneracy in the statement.
example (C0 C : Matrix (Fin (symCoordDim 0)) (Fin (symCoordDim 0)) ℝ)
    (x : Fin (symCoordDim 0) → ℝ) : gaussDensity (symCoordDim 0) C x = 1 := by
  have : IsEmpty (Fin (symCoordDim 0)) := inferInstanceAs (IsEmpty (Fin 0))
  unfold gaussDensity
  rw [Matrix.det_isEmpty, dotProduct, Finset.univ_eq_empty, Finset.sum_empty]
  simp [symCoordDim]
