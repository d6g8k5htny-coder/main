import Lean
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

-- ===== VERIFIER SCRATCH TAIL (never part of the deliverable) =====
open Lean Meta Elab Command

def auditModules : Array Name := #[]
def auditPrefix : Name := `UniversalLaw.Proofs.Side24Analysis
def registeredProps : Array Name := #[`UniversalLaw.Spec.Field.HessianHasInertia, `UniversalLaw.Spec.Field.IsCandidatePair, `UniversalLaw.Spec.Field.IsCriticalPoint, `UniversalLaw.Spec.Field.IsFiniteBar, `UniversalLaw.Spec.Field.IsIndexSaddle, `UniversalLaw.Spec.Field.IsLPeriodic, `UniversalLaw.Spec.Field.IsLifetimeDensityVersion, `UniversalLaw.Spec.Field.IsMorseDistinct, `UniversalLaw.Spec.Field.IsNondegMax, `UniversalLaw.Spec.Field.IsSmooth, `UniversalLaw.Spec.Lifetime.HessianHasInertia, `UniversalLaw.Spec.Lifetime.IsCandidatePair, `UniversalLaw.Spec.Lifetime.IsCriticalPoint, `UniversalLaw.Spec.Lifetime.IsFiniteBar, `UniversalLaw.Spec.Lifetime.IsIndexSaddle, `UniversalLaw.Spec.Lifetime.IsLPeriodic, `UniversalLaw.Spec.Lifetime.IsLifetimeDensityVersion, `UniversalLaw.Spec.Lifetime.IsMorseDistinct, `UniversalLaw.Spec.Lifetime.IsNondegMax, `UniversalLaw.Spec.Lifetime.IsSmooth, `UniversalLaw.Spec.Lifetime.IsUnrestrictedLeadingCoefficient, `UniversalLaw.Spec.Lifetime.IsUnrestrictedPair, `UniversalLaw.Spec.Lifetime.IsWindowElderPair, `UniversalLaw.Spec.Lifetime.IsWindowPair, `UniversalLaw.Spec.Lifetime.ParentCompactWindowDensityB, `UniversalLaw.Spec.Lifetime.ParentUniformSelectionA, `UniversalLaw.Spec.Lifetime.ParentUnrestrictedLeadingDensityC, `UniversalLaw.Spec.Lifetime.PlanarDensityGapE11, `UniversalLaw.Spec.Lifetime.PlanarElderLowerE2, `UniversalLaw.Spec.Lifetime.PlanarTwoSidedRateE4, `UniversalLaw.Spec.Lifetime.RemainderBoundedR1, `UniversalLaw.Spec.P15.Benchmark816Cover, `UniversalLaw.Spec.P15.Benchmark816IsRealized, `UniversalLaw.Spec.P15.Benchmark816MinimalForbiddenCounts, `UniversalLaw.Spec.P15.Benchmark816Palette, `UniversalLaw.Spec.P15.Benchmark816Price, `UniversalLaw.Spec.P15.Benchmark816RationalPrice, `UniversalLaw.Spec.P15.Benchmark816TransformedRange, `UniversalLaw.Spec.P15.Benchmark816UnionBound, `UniversalLaw.Spec.P15.Benchmark816WholeGround, `UniversalLaw.Spec.P15.CapTestAtTwoThirds, `UniversalLaw.Spec.P15.CapacityHazardIsBinomial, `UniversalLaw.Spec.P15.CapacityWorstCase, `UniversalLaw.Spec.P15.CoordinatewiseConcavity, `UniversalLaw.Spec.P15.CoverCostHazardBound, `UniversalLaw.Spec.P15.DemandOneBoundary, `UniversalLaw.Spec.P15.DemandOneRefutation, `UniversalLaw.Spec.P15.ElementaryPaletteFormula, `UniversalLaw.Spec.P15.FullBlockCoverIff, `UniversalLaw.Spec.P15.FullGroundBudgetIff, `UniversalLaw.Spec.P15.FullGroundPriceBound, `UniversalLaw.Spec.P15.FullTransformedPriceBudget, `UniversalLaw.Spec.P15.GlobalAssemblyHazard, `UniversalLaw.Spec.P15.HStarExceedsOne, `UniversalLaw.Spec.P15.HazardPhiBound, `UniversalLaw.Spec.P15.HazardProductBound, `UniversalLaw.Spec.P15.HazardStarPositive, `UniversalLaw.Spec.P15.IndependentBlockHazard, `UniversalLaw.Spec.P15.IsGeneratorCover, `UniversalLaw.Spec.P15.IsGood, `UniversalLaw.Spec.P15.IsKDecomposable, `UniversalLaw.Spec.P15.IsMinimalForbidden, `UniversalLaw.Spec.P15.IsPaletteFeasible, `UniversalLaw.Spec.P15.IsProperDownset, `UniversalLaw.Spec.P15.IsRealizedFamily, `UniversalLaw.Spec.P15.IsTransversal, `UniversalLaw.Spec.P15.LocalDecomposability, `UniversalLaw.Spec.P15.LocalObstructionIsFullBlock, `UniversalLaw.Spec.P15.LocalPriceBoundFull, `UniversalLaw.Spec.P15.LocalPriceChain, `UniversalLaw.Spec.P15.LocalRestriction, `UniversalLaw.Spec.P15.LogOnePlusBelowIdentity, `UniversalLaw.Spec.P15.LogRatioBound, `UniversalLaw.Spec.P15.MinimalForbiddenCharacterisation, `UniversalLaw.Spec.P15.OddMajorityIdentity, `UniversalLaw.Spec.P15.PaletteNumberAttained, `UniversalLaw.Spec.P15.PhiAtPStar, `UniversalLaw.Spec.P15.RatioExponentBound, `UniversalLaw.Spec.P15.RestrictedLocalBudget, `UniversalLaw.Spec.P15.RestrictedTransformedPriceBudget, `UniversalLaw.Spec.P15.RhoStarBelowSixSevenths, `UniversalLaw.Spec.P15.RhoStarEnclosure, `UniversalLaw.Spec.P15.RhoTestSufficient, `UniversalLaw.Spec.P15.SharpnessOfRhoStar, `UniversalLaw.Spec.P15.SixSeventhsCertificate, `UniversalLaw.Spec.P15.TriangleNeedsThree, `UniversalLaw.Spec.P15.TwoCoordGeneratorPrices, `UniversalLaw.Spec.P15.WholeGroundChromaticNumber, `UniversalLaw.Spec.RN.AnnulusBridgeAllHeight2, `UniversalLaw.Spec.RN.ConditionalMeanIdentityN3, `UniversalLaw.Spec.RN.D5aPuncturedPinDisks, `UniversalLaw.Spec.RN.D5bMicrodisk, `UniversalLaw.Spec.RN.D5cCollarC1, `UniversalLaw.Spec.RN.D5dSummedPinNeighbourhoodC2, `UniversalLaw.Spec.RN.D5eIntermediateShellsI3I4, `UniversalLaw.Spec.RN.D5fGlobalWindowFirstMomentI5, `UniversalLaw.Spec.RN.DimensionLiftCompactCollarC, `UniversalLaw.Spec.RN.DimensionLiftGlobalWindowG, `UniversalLaw.Spec.RN.DimensionLiftIntermediateShellsI, `UniversalLaw.Spec.RN.DimensionLiftPuncturedPinBallP, `UniversalLaw.Spec.RN.ExponentialTailLayerCakeN5, `UniversalLaw.Spec.RN.FixedAnnulusWindowCandidateA2, `UniversalLaw.Spec.RN.FixedRemoteCountBoundA3, `UniversalLaw.Spec.RN.FixedRemoteMeanMeasureA2, `UniversalLaw.Spec.RN.FourierCutoffPlanarSecondFactorialF4, `UniversalLaw.Spec.RN.HessianHasInertia, `UniversalLaw.Spec.RN.HolderConsequenceN2, `UniversalLaw.Spec.RN.HolderCountBoundN1, `UniversalLaw.Spec.RN.IsCriticalPoint, `UniversalLaw.Spec.RN.PalmRouteFactorialMomentsQ, `UniversalLaw.Spec.RN.PalmRouteSecondFactorialTheta, `UniversalLaw.Spec.RN.RareClusterPoissonObstructionO, `UniversalLaw.Spec.RN.RemoteCollisionSecondFactorialD, `UniversalLaw.Spec.RN.UniformCounterexampleN4, `UniversalLaw.Spec.RN.WitnessCollisionNearPinPairsTarget, `UniversalLaw.Spec.Side24.CoefficientEnclosure, `UniversalLaw.Spec.Side24.ConeMomentD1Identity, `UniversalLaw.Spec.Side24.ConeMomentD2Algebra, `UniversalLaw.Spec.Side24.ConeMomentD2Identity, `UniversalLaw.Spec.Side24.CovarianceOrdering, `UniversalLaw.Spec.Side24.DensityComparison, `UniversalLaw.Spec.Side24.DerivativeBoundAtLeastOne, `UniversalLaw.Spec.Side24.DerivativeBoundAtZero, `UniversalLaw.Spec.Side24.DerivativePairingBound, `UniversalLaw.Spec.Side24.ElementaryConeIntegral, `UniversalLaw.Spec.Side24.EntrywiseCovarianceBound, `UniversalLaw.Spec.Side24.ExponentialBounds, `UniversalLaw.Spec.Side24.ImageBound, `UniversalLaw.Spec.Side24.IntegrandRatioBounds, `UniversalLaw.Spec.Side24.IsUnitTuple, `UniversalLaw.Spec.Side24.LatticeImageSumBound, `UniversalLaw.Spec.Side24.NegDefCone, `UniversalLaw.Spec.Side24.RayleighSquareLaw, `UniversalLaw.Spec.Side24.ReferenceCoefficientIdentity, `UniversalLaw.Spec.Side24.ReferenceComparison, `UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound, `UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks, `UniversalLaw.Spec.Side24.ReferencePinDensityProduct, `UniversalLaw.Spec.Side24.SuccessiveTermRatio, `UniversalLaw.Spec.Side24.ThetaSummable, `UniversalLaw.Spec.Side24.TraceMoments]

/-! Every constant of the audited modules (or, when `auditModules` is empty, every constant of
this file under `auditPrefix`, private names included) is classified by Lean's own `ConstantInfo`
and type, never by source text. `theorem`: a theorem that is not private, not under an `Aux`
namespace and carries a declaration range, whose type after `forallTelescope` is a registered Prop
constant applied to exactly the telescope's variables (reported with that constant). `aux_theorem`:
a private theorem or one under `Aux`. `generated`: a range-less theorem (structure `injEq`,
`sizeOf_spec`, `_proof_*`, an equation lemma `eq_*` of an unfolded statement definition, which may
be generated in several modules and is audited once), refused if its head is a registered Prop. `auxiliary`: definitions,
instances, inductive types, constructors and recursors whose type is not a proposition.
`REFUSED:<why>`: an axiom, opaque, quotient or unsafe constant; a definition whose type is a
proposition (a proof written as `def`); a non-aux theorem whose head is not registered (`type`) or
whose arguments are not exactly its own parameters (`premise`: an extra hypothesis or a narrowed
argument); any transitive axiom outside propext, Classical.choice and Quot.sound (`axioms`).
One `AUDIT <class> <name> [<head>]` line per constant, then `AUDIT END <count> <refused>
<theorems>`; any refusal makes the file fail. -/
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut constants : Array ConstantInfo := #[]
  if auditModules.isEmpty then
    constants := env.constants.foldStage2 (fun acc _ ci =>
      let user := (privateToUserName? ci.name).getD ci.name
      if auditPrefix.isPrefixOf user then acc.push ci else acc) #[]
  else
    let mut found : Array Name := #[]
    for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
      if auditModules.contains modName then
        found := found.push modName
        constants := constants ++ data.constants
    unless found.size == auditModules.size do
      throwError "AUDIT REFUSED: audited module not imported: {auditModules.filter (!found.contains ·)}"
  -- an equation lemma of an imported definition (`foo.eq_1`) is generated on demand in every module
  -- that unfolds it and is one constant of the environment: audit each name once
  let mut seen : NameSet := {}
  let mut unique : Array ConstantInfo := #[]
  for ci in constants do
    unless seen.contains ci.name do
      seen := seen.insert ci.name
      unique := unique.push ci
  constants := unique
  let mut refused := 0
  let mut theorems := 0
  for ci in constants do
    let n := ci.name
    let axioms ← collectAxioms n
    let bad := axioms.filter (!allowed.contains ·)
    let range ← findDeclarationRanges? n
    let user := (privateToUserName? n).getD n
    let isAux := isPrivateName n || user.components.contains `Aux
    let (head, exact) ← liftTermElabM do
      forallTelescope ci.type fun xs body => do
        let args := body.getAppArgs
        let exact := args.size == xs.size && args.all (fun a => xs.contains a) && xs.all (fun x => args.contains x)
        pure (body.getAppFn.constName?, exact)
    let headStr : String := match head with | some c => toString c | none => "nonconstant"
    let registeredHead := match head with | some c => registeredProps.contains c | none => false
    let isP ← liftTermElabM (isProp ci.type)
    let kind : String :=
      if ci.isUnsafe then "REFUSED:unsafe" else
      match ci with
        | .axiomInfo _ => "REFUSED:axiom"
        | .opaqueInfo _ => "REFUSED:opaque"
        | .quotInfo _ => "REFUSED:quot"
        | .inductInfo _ => "auxiliary"
        | .ctorInfo _ => "auxiliary"
        | .recInfo _ => "auxiliary"
        | .defnInfo _ => if isP then "REFUSED:proof_as_def" else "auxiliary"
        | .thmInfo _ =>
          if isAux then "aux_theorem"
          else if range.isNone then (if registeredHead then s!"REFUSED:generated_proof:{headStr}" else "generated")
          else if registeredHead then (if exact then "theorem" else s!"REFUSED:premise:{headStr}")
          else s!"REFUSED:type:{headStr}"
    let verdict : String :=
      if kind.startsWith "REFUSED" || bad.isEmpty then kind
      else "REFUSED:axioms:" ++ ",".intercalate (bad.map toString).toList
    if verdict.startsWith "REFUSED" then refused := refused + 1
    if verdict == "theorem" then
      theorems := theorems + 1
      logInfo m!"AUDIT theorem {n} {headStr}"
    else
      logInfo m!"AUDIT {verdict} {n}"
  logInfo m!"AUDIT END {constants.size} {refused} {theorems}"
  if refused > 0 then throwError "AUDIT REFUSED {refused} constant(s); see the AUDIT lines"

#check @UniversalLaw.Proofs.Side24Analysis.densityComparison
#print UniversalLaw.Spec.Side24.DensityComparison
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_epsilon_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_epsilon_lt_one
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_one_sub_epsilon_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_one_add_epsilon_pos
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_expand_identity
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_conj_identity
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_inv_sub_inv_posSemidef
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_quad_le
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_one_le_det
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_det_le_det
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_det_smul
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_scale_identity
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_lower
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_assemble_upper
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_expand_identity._abel_1_1
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.dc_expand_identity._abel_1_2
#print axioms UniversalLaw.Proofs.Side24Analysis.densityComparison

run_cmd do
  let env ← getEnv
  let some ci := env.find? `UniversalLaw.Proofs.Side24Analysis.densityComparison | throwError "VERIFY: theorem missing"
  logInfo m!"VERIFY levelParams={ci.levelParams} type={ci.type}"
  unless ci.type.isConstOf `UniversalLaw.Spec.Side24.DensityComparison do throwError "VERIFY: type is not the bare registered constant"
  unless ci.type.getAppNumArgs == 0 do throwError "VERIFY: type has arguments"
  match ci with
  | .thmInfo _ => logInfo "VERIFY kind=theorem"
  | _ => throwError "VERIFY: not a theorem"
  logInfo "VERIFY_TYPE_EXACT_CONST OK"
