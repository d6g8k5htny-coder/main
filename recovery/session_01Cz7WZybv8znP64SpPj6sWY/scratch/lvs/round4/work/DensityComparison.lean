import Specifications.Side24

/-!
# SIDE24 analysis proofs: the Section 4 Gaussian density comparison (`DensityComparison`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Analysis.Aux` (prefix `dc_`). The route is
pure real matrix analysis. Write `A = (1 - epsilon) • C0`, `B = (1 + epsilon) • C0`, so that the
hypotheses say `A ≤ C ≤ B` in the Loewner (positive-semidefinite) order and `A`, `B`, `C` are
positive definite. Two order lemmas are proved for real matrices:

* (L1, `dc_inv_sub_inv_posSemidef`) `A` positive definite, `C` positive definite and `C - A`
  positive semidefinite imply `A⁻¹ - C⁻¹` positive semidefinite. Proof: the identity
  `(C - A) A⁻¹ (C - A) + (C - A) = C A⁻¹ C - C` shows `C A⁻¹ C - C` positive semidefinite
  (a conjugate of `A⁻¹` plus `C - A`), and `C⁻¹ (C A⁻¹ C - C) C⁻¹ = A⁻¹ - C⁻¹` is a conjugate of it
  (`Matrix.PosSemidef.conjTranspose_mul_mul_same`, `Matrix.mul_nonsing_inv`,
  `Matrix.nonsing_inv_mul`).
* (L2, `dc_det_le_det`) `A` positive definite and `C - A` positive semidefinite imply
  `det A ≤ det C`. Proof: with `S = CFC.sqrt A` (so `S S = A`, `S` Hermitian, `det S ≠ 0`),
  `M = S⁻¹ C S⁻¹` satisfies `M - 1 = S⁻¹ (C - A) S⁻¹`, positive semidefinite; a real matrix with
  `M - 1` positive semidefinite has every eigenvalue `≥ 1`
  (`Matrix.posSemidef_iff_isHermitian_and_spectrum_nonneg`, `spectrum.sub_singleton_eq`,
  `Matrix.IsHermitian.eigenvalues_mem_spectrum_real`), hence `1 ≤ det M`
  (`Matrix.IsHermitian.det_eq_prod_eigenvalues`), and `det C = det S · det M · det S = det A · det M`.

The pointwise comparison then follows from `det A ≤ det C ≤ det B`, the quadratic-form
inequalities `x ⬝ B⁻¹ x ≤ x ⬝ C⁻¹ x ≤ x ⬝ A⁻¹ x`, the scaling identity
`((1 - epsilon)/(1 + epsilon))^(n/2) · det A^(-1/2) = det B^(-1/2)` (and its mirror image), which
is real-power bookkeeping on `det (c • C0) = c^n det C0`, antitonicity of `t ↦ t^(-1/2)`
(`Real.rpow_le_rpow_of_nonpos`) and monotonicity of `Real.exp`.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24
open scoped MatrixOrder
open scoped Pointwise

namespace Aux

/-! ### (a) Elementary facts about `epsilon = 10^(-108)` -/

/-- `0 < epsilon`. -/
theorem dc_epsilon_pos : (0 : ℝ) < epsilon := by
  unfold epsilon
  positivity

/-- `epsilon < 1`. -/
theorem dc_epsilon_lt_one : epsilon < (1 : ℝ) := by
  unfold epsilon
  norm_num

/-- `0 < 1 - epsilon`. -/
theorem dc_one_sub_epsilon_pos : (0 : ℝ) < 1 - epsilon := by
  have := dc_epsilon_lt_one
  linarith

/-- `0 < 1 + epsilon`. -/
theorem dc_one_add_epsilon_pos : (0 : ℝ) < 1 + epsilon := by
  have := dc_epsilon_pos
  linarith

/-! ### (b) Loewner order: antitonicity of the inverse (L1) -/

/-- The expansion identity `(C - A) A⁻¹ (C - A) + (C - A) = C A⁻¹ C - C` for an invertible `A`. -/
theorem dc_expand_identity {k : ℕ} (A C : Matrix (Fin k) (Fin k) ℝ) (hA : IsUnit A.det) :
    (C - A) * A⁻¹ * (C - A) + (C - A) = C * A⁻¹ * C - C := by
  have h1 : A * A⁻¹ = 1 := Matrix.mul_nonsing_inv A hA
  have h2 : A⁻¹ * A = 1 := Matrix.nonsing_inv_mul A hA
  calc (C - A) * A⁻¹ * (C - A) + (C - A)
      = C * A⁻¹ * C - C * (A⁻¹ * A) - (A * A⁻¹) * C + (A * A⁻¹) * A + (C - A) := by
        noncomm_ring
    _ = C * A⁻¹ * C - C := by
        rw [h1, h2]
        simp only [Matrix.mul_one, Matrix.one_mul]
        abel

/-- The conjugation identity `C⁻¹ (C A⁻¹ C - C) C⁻¹ = A⁻¹ - C⁻¹` for an invertible `C`. -/
theorem dc_conj_identity {k : ℕ} (A C : Matrix (Fin k) (Fin k) ℝ) (hC : IsUnit C.det) :
    C⁻¹ * (C * A⁻¹ * C - C) * C⁻¹ = A⁻¹ - C⁻¹ := by
  have h1 : C * C⁻¹ = 1 := Matrix.mul_nonsing_inv C hC
  have h2 : C⁻¹ * C = 1 := Matrix.nonsing_inv_mul C hC
  calc C⁻¹ * (C * A⁻¹ * C - C) * C⁻¹
      = (C⁻¹ * C) * A⁻¹ * (C * C⁻¹) - C⁻¹ * (C * C⁻¹) := by
        noncomm_ring
    _ = A⁻¹ - C⁻¹ := by
        rw [h1, h2]
        simp only [Matrix.mul_one, Matrix.one_mul]

/-- (L1) For real matrices: `A` positive definite, `C` positive definite and `C - A` positive
semidefinite give `A⁻¹ - C⁻¹` positive semidefinite, i.e. `C⁻¹ ≤ A⁻¹` in the Loewner order. -/
theorem dc_inv_sub_inv_posSemidef {k : ℕ} {A C : Matrix (Fin k) (Fin k) ℝ}
    (hA : A.PosDef) (hC : C.PosDef) (hAC : (C - A).PosSemidef) :
    (A⁻¹ - C⁻¹).PosSemidef := by
  have hAu : IsUnit A.det := isUnit_iff_ne_zero.mpr hA.det_pos.ne'
  have hCu : IsUnit C.det := isUnit_iff_ne_zero.mpr hC.det_pos.ne'
  have hP : (C * A⁻¹ * C - C).PosSemidef := by
    have h1 : ((C - A).conjTranspose * A⁻¹ * (C - A)).PosSemidef :=
      hA.inv.posSemidef.conjTranspose_mul_mul_same (C - A)
    rw [hAC.isHermitian.eq] at h1
    have h2 := h1.add hAC
    rwa [dc_expand_identity A C hAu] at h2
  have h3 : ((C⁻¹).conjTranspose * (C * A⁻¹ * C - C) * C⁻¹).PosSemidef :=
    hP.conjTranspose_mul_mul_same C⁻¹
  rw [hC.inv.isHermitian.eq] at h3
  rwa [dc_conj_identity A C hCu] at h3

/-- The quadratic-form consequence of (L1): `x ⬝ C⁻¹ x ≤ x ⬝ A⁻¹ x`. -/
theorem dc_quad_le {k : ℕ} {A C : Matrix (Fin k) (Fin k) ℝ}
    (hA : A.PosDef) (hC : C.PosDef) (hAC : (C - A).PosSemidef) (x : Fin k → ℝ) :
    dotProduct x (Matrix.mulVec C⁻¹ x) ≤ dotProduct x (Matrix.mulVec A⁻¹ x) := by
  have h := (dc_inv_sub_inv_posSemidef hA hC hAC).dotProduct_mulVec_nonneg x
  simp only [star_trivial, Matrix.sub_mulVec, dotProduct_sub] at h
  linarith

/-! ### (c) Loewner order: monotonicity of the determinant (L2) -/

/-- A real matrix `M` with `M - 1` positive semidefinite has determinant at least one: every
eigenvalue of `M` lies in `spectrum ℝ M`, so every eigenvalue minus one lies in
`spectrum ℝ (M - 1) ⊆ [0, ∞)`, and `det M` is the product of the eigenvalues. -/
theorem dc_one_le_det {k : ℕ} {M : Matrix (Fin k) (Fin k) ℝ} (hM : (M - 1).PosSemidef) :
    1 ≤ M.det := by
  have hherm : M.IsHermitian := by
    have h := hM.isHermitian.add (Matrix.isHermitian_one (n := Fin k) (α := ℝ))
    simpa using h
  rw [hherm.det_eq_prod_eigenvalues]
  apply Finset.one_le_prod₀
  intro i _
  have hspec := (Matrix.posSemidef_iff_isHermitian_and_spectrum_nonneg.mp hM).2
  have hmem : hherm.eigenvalues i ∈ spectrum ℝ M := hherm.eigenvalues_mem_spectrum_real i
  have hmem' : hherm.eigenvalues i - 1 ∈ spectrum ℝ (M - 1) := by
    have h : hherm.eigenvalues i - 1 ∈ spectrum ℝ M - ({1} : Set ℝ) :=
      Set.sub_mem_sub hmem (Set.mem_singleton (1 : ℝ))
    rw [spectrum.sub_singleton_eq] at h
    simpa using h
  have h0 : (0 : ℝ) ≤ hherm.eigenvalues i - 1 := hspec hmem'
  simp only [RCLike.ofReal_real_eq_id, id]
  linarith

/-- (L2) For real matrices: `A` positive definite and `C - A` positive semidefinite give
`det A ≤ det C`, through the square root `S = CFC.sqrt A` and `M = S⁻¹ C S⁻¹ ≥ 1`. -/
theorem dc_det_le_det {k : ℕ} {A C : Matrix (Fin k) (Fin k) ℝ}
    (hA : A.PosDef) (hAC : (C - A).PosSemidef) : A.det ≤ C.det := by
  obtain ⟨S, hSS, hSpsd⟩ : ∃ S : Matrix (Fin k) (Fin k) ℝ, S * S = A ∧ S.PosSemidef :=
    ⟨CFC.sqrt A, CFC.sqrt_mul_sqrt_self A hA.posSemidef.nonneg,
      Matrix.nonneg_iff_posSemidef.mp (CFC.sqrt_nonneg A)⟩
  have hSherm : S.conjTranspose = S := hSpsd.isHermitian.eq
  have hdetA : A.det = S.det * S.det := by
    rw [← hSS, Matrix.det_mul]
  have hSdet : S.det ≠ 0 := by
    intro h
    have h0 : A.det = 0 := by
      rw [hdetA, h, mul_zero]
    exact hA.det_pos.ne' h0
  have hSu : IsUnit S.det := isUnit_iff_ne_zero.mpr hSdet
  have hSinv : (S⁻¹).conjTranspose = S⁻¹ := by
    rw [Matrix.conjTranspose_nonsing_inv, hSherm]
  have h1 : S * S⁻¹ = 1 := Matrix.mul_nonsing_inv S hSu
  have h2 : S⁻¹ * S = 1 := Matrix.nonsing_inv_mul S hSu
  have hM : (S⁻¹ * C * S⁻¹ - 1).PosSemidef := by
    have h3 : ((S⁻¹).conjTranspose * (C - A) * S⁻¹).PosSemidef :=
      hAC.conjTranspose_mul_mul_same S⁻¹
    rw [hSinv] at h3
    have h4 : S⁻¹ * (C - A) * S⁻¹ = S⁻¹ * C * S⁻¹ - 1 := by
      rw [← hSS]
      calc S⁻¹ * (C - S * S) * S⁻¹
          = S⁻¹ * C * S⁻¹ - (S⁻¹ * S) * (S * S⁻¹) := by
            noncomm_ring
        _ = S⁻¹ * C * S⁻¹ - 1 := by
            rw [h1, h2, Matrix.mul_one]
    rwa [h4] at h3
  have hdetM : 1 ≤ (S⁻¹ * C * S⁻¹).det := dc_one_le_det hM
  have hCeq : S * (S⁻¹ * C * S⁻¹) * S = C := by
    calc S * (S⁻¹ * C * S⁻¹) * S = (S * S⁻¹) * C * (S⁻¹ * S) := by
          noncomm_ring
      _ = C := by
          rw [h1, h2, Matrix.one_mul, Matrix.mul_one]
  have hdetC : C.det = S.det * S.det * (S⁻¹ * C * S⁻¹).det := by
    calc C.det = (S * (S⁻¹ * C * S⁻¹) * S).det := by
          rw [hCeq]
      _ = S.det * (S⁻¹ * C * S⁻¹).det * S.det := by
          rw [Matrix.det_mul, Matrix.det_mul]
      _ = S.det * S.det * (S⁻¹ * C * S⁻¹).det := by
          ring
  have hSS2 : 0 ≤ S.det * S.det := mul_self_nonneg _
  calc A.det = S.det * S.det := hdetA
    _ = S.det * S.det * 1 := (mul_one _).symm
    _ ≤ S.det * S.det * (S⁻¹ * C * S⁻¹).det := mul_le_mul_of_nonneg_left hdetM hSS2
    _ = C.det := hdetC.symm

/-! ### (d) Determinant scaling and real-power bookkeeping -/

/-- `det (c • C) = c ^ k * det C` for a `k × k` real matrix. -/
theorem dc_det_smul {k : ℕ} (C : Matrix (Fin k) (Fin k) ℝ) (c : ℝ) :
    (c • C).det = c ^ k * C.det := by
  rw [Matrix.det_smul, Fintype.card_fin]

/-- The scaling identity `(a / b)^(k/2) · (a^k D)^(-1/2) = (b^k D)^(-1/2)` for `a, b, D > 0`. -/
theorem dc_scale_identity (a b D : ℝ) (ha : 0 < a) (hb : 0 < b) (hD : 0 < D) (k : ℕ) :
    (a / b) ^ ((k : ℝ) / 2) * (a ^ k * D) ^ (-(1 : ℝ) / 2) = (b ^ k * D) ^ (-(1 : ℝ) / 2) := by
  have hak : (a ^ k) ^ (-(1 : ℝ) / 2) = (a ^ ((k : ℝ) / 2))⁻¹ := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul ha.le, ← Real.rpow_neg ha.le]
    congr 1
    ring
  have hbk : (b ^ k) ^ (-(1 : ℝ) / 2) = (b ^ ((k : ℝ) / 2))⁻¹ := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hb.le, ← Real.rpow_neg hb.le]
    congr 1
    ring
  have hapos : a ^ ((k : ℝ) / 2) ≠ 0 := (Real.rpow_pos_of_pos ha _).ne'
  have hbpos : b ^ ((k : ℝ) / 2) ≠ 0 := (Real.rpow_pos_of_pos hb _).ne'
  rw [Real.div_rpow ha.le hb.le, Real.mul_rpow (pow_nonneg ha.le k) hD.le,
    Real.mul_rpow (pow_nonneg hb.le k) hD.le, hak, hbk]
  field_simp

/-! ### (e) Assembly of the two pointwise inequalities from abstract real factors -/

/-- Lower assembly: `s * dA = dB`, `dB ≤ dC`, `eA ≤ eC` give `s (P dA eA) ≤ P dC eC`. -/
theorem dc_assemble_lower (P s dA dB dC eA eC : ℝ) (hP : 0 ≤ P) (hs : s * dA = dB)
    (hd : dB ≤ dC) (hdC : 0 ≤ dC) (he : eA ≤ eC) (heA : 0 ≤ eA) :
    s * (P * dA * eA) ≤ P * dC * eC := by
  calc s * (P * dA * eA) = P * (s * dA) * eA := by ring
    _ = P * dB * eA := by rw [hs]
    _ ≤ P * dC * eC := mul_le_mul (mul_le_mul_of_nonneg_left hd hP) he heA (mul_nonneg hP hdC)

/-- Upper assembly: `s * dB = dA`, `dC ≤ dA`, `eC ≤ eB` give `P dC eC ≤ s (P dB eB)`. -/
theorem dc_assemble_upper (P s dA dB dC eB eC : ℝ) (hP : 0 ≤ P) (hs : s * dB = dA)
    (hd : dC ≤ dA) (hdA : 0 ≤ dA) (he : eC ≤ eB) (heC : 0 ≤ eC) :
    P * dC * eC ≤ s * (P * dB * eB) := by
  calc P * dC * eC ≤ P * dA * eB :=
        mul_le_mul (mul_le_mul_of_nonneg_left hd hP) he heC (mul_nonneg hP hdA)
    _ = P * (s * dB) * eB := by rw [hs]
    _ = s * (P * dB * eB) := by ring

end Aux

/-! ## The Section 4 Gaussian density comparison -/

/-- Proves: `UniversalLaw.Spec.Side24.DensityComparison` (claim `side24-coefficient`).
Method: with `A = (1 - epsilon) • C0` and `B = (1 + epsilon) • C0` (positive definite by
`Matrix.PosDef.smul`), the Loewner lemmas `Aux.dc_det_le_det` (`det A ≤ det C ≤ det B`, via
`CFC.sqrt`, `Matrix.PosSemidef.conjTranspose_mul_mul_same`,
`Matrix.posSemidef_iff_isHermitian_and_spectrum_nonneg` and
`Matrix.IsHermitian.det_eq_prod_eigenvalues`) and `Aux.dc_quad_le`
(`x ⬝ B⁻¹ x ≤ x ⬝ C⁻¹ x ≤ x ⬝ A⁻¹ x`, via `Aux.dc_inv_sub_inv_posSemidef`) give the two
determinant and exponent comparisons; `Aux.dc_scale_identity` with `Matrix.det_smul` turns
`((1 ∓ epsilon)/(1 ± epsilon))^(n/2) · det((1 ∓ epsilon) • C0)^(-1/2)` into
`det((1 ± epsilon) • C0)^(-1/2)`; `Real.rpow_le_rpow_of_nonpos`, `Real.exp_le_exp` and
`Aux.dc_assemble_lower` / `Aux.dc_assemble_upper` finish after unfolding `gaussDensity`.
Does not establish: the informal Section 4 display of the source as the source's object (the
alignment of `gaussDensity`, `symCoordDim`, `epsilon` and the positive-semidefinite hypotheses
with the source's densities, `n = m(m+1)/2`, `epsilon` and ordering is
PENDING_INDEPENDENT_REVIEW); the cone-moment consequence `(1 ± epsilon)^m D0` and the scaling
`a^m` of `h(A) = det(A)^2 1{A<0}` (not specified in the statement packet); that
`(1 ± epsilon) • C0` or `C` are covariances of any random object; `CovarianceOrdering` or any
ordering of `C24` against `Cref`; any enclosure of the coefficient; the statement module's own
`Does not claim:` text (the statement packet is not edited here); the disposition of
`side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not derived here;
nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem densityComparison : UniversalLaw.Spec.Side24.DensityComparison := by
  intro m C0 C hC0 hC hlow hup x
  have ha : (0 : ℝ) < 1 - epsilon := Aux.dc_one_sub_epsilon_pos
  have hb : (0 : ℝ) < 1 + epsilon := Aux.dc_one_add_epsilon_pos
  have hA : ((1 - epsilon) • C0).PosDef := hC0.smul ha
  have hB : ((1 + epsilon) • C0).PosDef := hC0.smul hb
  have hD : 0 < C0.det := hC0.det_pos
  have hdetC : 0 < C.det := hC.det_pos
  have hdetApos : 0 < ((1 - epsilon) • C0).det := hA.det_pos
  have hdetAC : ((1 - epsilon) • C0).det ≤ C.det := Aux.dc_det_le_det hA hlow
  have hdetCB : C.det ≤ ((1 + epsilon) • C0).det := Aux.dc_det_le_det hC hup
  have hqAC : dotProduct x (Matrix.mulVec C⁻¹ x) ≤
      dotProduct x (Matrix.mulVec ((1 - epsilon) • C0)⁻¹ x) := Aux.dc_quad_le hA hC hlow x
  have hqCB : dotProduct x (Matrix.mulVec ((1 + epsilon) • C0)⁻¹ x) ≤
      dotProduct x (Matrix.mulVec C⁻¹ x) := Aux.dc_quad_le hC hB hup x
  have hdetA : ((1 - epsilon) • C0).det = (1 - epsilon) ^ symCoordDim m * C0.det :=
    Aux.dc_det_smul C0 _
  have hdetB : ((1 + epsilon) • C0).det = (1 + epsilon) ^ symCoordDim m * C0.det :=
    Aux.dc_det_smul C0 _
  have hscale1 : ((1 - epsilon) / (1 + epsilon)) ^ ((symCoordDim m : ℝ) / 2) *
      ((1 - epsilon) • C0).det ^ (-(1 : ℝ) / 2) = ((1 + epsilon) • C0).det ^ (-(1 : ℝ) / 2) := by
    rw [hdetA, hdetB]
    exact Aux.dc_scale_identity _ _ _ ha hb hD _
  have hscale2 : ((1 + epsilon) / (1 - epsilon)) ^ ((symCoordDim m : ℝ) / 2) *
      ((1 + epsilon) • C0).det ^ (-(1 : ℝ) / 2) = ((1 - epsilon) • C0).det ^ (-(1 : ℝ) / 2) := by
    rw [hdetA, hdetB]
    exact Aux.dc_scale_identity _ _ _ hb ha hD _
  have hP : (0 : ℝ) ≤ (2 * Real.pi) ^ (-(symCoordDim m : ℝ) / 2) :=
    (Real.rpow_pos_of_pos (by positivity) _).le
  have hrB : ((1 + epsilon) • C0).det ^ (-(1 : ℝ) / 2) ≤ C.det ^ (-(1 : ℝ) / 2) :=
    Real.rpow_le_rpow_of_nonpos hdetC hdetCB (by norm_num)
  have hrA : C.det ^ (-(1 : ℝ) / 2) ≤ ((1 - epsilon) • C0).det ^ (-(1 : ℝ) / 2) :=
    Real.rpow_le_rpow_of_nonpos hdetApos hdetAC (by norm_num)
  have heA : Real.exp (-(dotProduct x (Matrix.mulVec ((1 - epsilon) • C0)⁻¹ x)) / 2) ≤
      Real.exp (-(dotProduct x (Matrix.mulVec C⁻¹ x)) / 2) :=
    Real.exp_le_exp.mpr (by linarith)
  have heB : Real.exp (-(dotProduct x (Matrix.mulVec C⁻¹ x)) / 2) ≤
      Real.exp (-(dotProduct x (Matrix.mulVec ((1 + epsilon) • C0)⁻¹ x)) / 2) :=
    Real.exp_le_exp.mpr (by linarith)
  unfold gaussDensity
  constructor
  · exact Aux.dc_assemble_lower _ _ _ _ _ _ _ hP hscale1 hrB (Real.rpow_nonneg hdetC.le _) heA
      (Real.exp_pos _).le
  · exact Aux.dc_assemble_upper _ _ _ _ _ _ _ hP hscale2 hrA (Real.rpow_nonneg hdetApos.le _) heB
      (Real.exp_pos _).le

end UniversalLaw.Proofs.Side24Analysis
