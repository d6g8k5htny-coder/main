import SourceContactCovariance

/-!
# Literal six-pin covariance and positive reference Schur variance

The six pin coordinates are `(f, fx, fy, fxx, fxy, fxxx)` and the remaining
coordinate is `Q = fyy`, all taken from the literal origin contact vector.
The block identity below uses an explicit equivalence of the seven coordinates.
Positive definiteness, invertibility and strict Schur positivity follow from
`L ≠ 0`; no covariance or positivity conclusion is an input hypothesis.

These statements concern the torus source with its actual period. They do not
identify its regression mean or variance with a planar reference constant.
See `SOURCE_CONTACT_SCHUR.md` for the frozen scope.
-/

noncomputable section

open I4SmoothCarrier I4ContactGram MeasureTheory ProbabilityTheory Matrix
open scoped BigOperators Matrix

set_option autoImplicit false

namespace I4SourceContact

/-- Literal six-pin observations in their source order. -/
def sourcePinVector (L : ℝ) (ω : SmoothSample 2 L) (i : Fin 6) : ℝ :=
  sourceContactVector L ω i.castSucc

/-- Literal remaining curvature coordinate Q=fyy. -/
def sourceContactQ (L : ℝ) (ω : SmoothSample 2 L) : ℝ :=
  sourceContactVector L ω 6

/-- Principal covariance of the literal six origin pin coordinates. -/
def sourcePinCovariance (L : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  (sourceContactCovariance L).submatrix Fin.castSucc Fin.castSucc

/-- Covariance of each literal pin coordinate with the remaining curvature Q=fyy. -/
def sourceContactCross (L : ℝ) (i : Fin 6) : ℝ :=
  sourceContactCovariance L i.castSucc 6

/-- Literal Q covariance with itself. -/
def sourceContactQVariance (L : ℝ) : ℝ := sourceContactCovariance L 6 6

def sourceContactCrossMatrix (L : ℝ) : Matrix (Fin 6) Unit ℝ :=
  fun i _ => sourceContactCross L i

def sourceContactQMatrix (L : ℝ) : Matrix Unit Unit ℝ :=
  fun _ _ => sourceContactQVariance L

/-- The scalar Schur complement of the actual six-pin covariance. -/
def sourceContactSchur (L : ℝ) : ℝ :=
  sourceContactQVariance L -
    sourceContactCross L ⬝ᵥ ((sourcePinCovariance L)⁻¹ *ᵥ sourceContactCross L)

/-- Exact 6+1 splitting of the source's seven origin coordinates. -/
def sourceContactBlockEquiv : (Fin 6 ⊕ Unit) ≃ Fin 7 :=
  (Equiv.sumCongr (Equiv.refl (Fin 6)) finOneEquiv.symm).trans finSumFinEquiv

@[simp] lemma sourceContactBlockEquiv_inl (i : Fin 6) :
    sourceContactBlockEquiv (Sum.inl i) = i.castSucc := rfl

@[simp] lemma sourceContactBlockEquiv_inr (u : Unit) :
    sourceContactBlockEquiv (Sum.inr u) = 6 := by
  cases u
  rfl

/-- The literal covariance equals the stated six-pin/Q blocks under the exact coordinate split. -/
theorem sourceContactCovariance_block_identity (L : ℝ) :
    (sourceContactCovariance L).submatrix sourceContactBlockEquiv sourceContactBlockEquiv =
      Matrix.fromBlocks (sourcePinCovariance L) (sourceContactCrossMatrix L)
        (sourceContactCrossMatrix L)ᴴ (sourceContactQMatrix L) := by
  ext i j
  cases i with
  | inl i =>
    cases j with
    | inl j => rfl
    | inr u => cases u; rfl
  | inr u =>
    cases u
    cases j with
    | inl j =>
      change covariance (fun ω => sourceContactVector L ω 6)
        (fun ω => sourceContactVector L ω j.castSucc) (smoothLaw 2 L) =
          covariance (fun ω => sourceContactVector L ω j.castSucc)
            (fun ω => sourceContactVector L ω 6) (smoothLaw 2 L)
      exact covariance_comm _ _
    | inr v => cases v; rfl

theorem sourcePinCovariance_posDef {L : ℝ} (hL : L ≠ 0) :
    (sourcePinCovariance L).PosDef :=
  (sourceContactCovariance_posDef hL).submatrix (Fin.castSucc_injective 6)

theorem sourcePinCovariance_isUnit {L : ℝ} (hL : L ≠ 0) :
    IsUnit (sourcePinCovariance L) :=
  (sourcePinCovariance_posDef hL).isUnit

/-- An actual inverse of the six-pin covariance, with no invertibility premise. -/
@[instance_reducible] def sourcePinCovarianceInvertible {L : ℝ} (hL : L ≠ 0) :
    Invertible (sourcePinCovariance L) :=
  (sourcePinCovariance_isUnit hL).invertible

theorem sourceContactQVariance_eq_variance {L : ℝ} (hL : L ≠ 0) :
    sourceContactQVariance L =
      Var[fun ω => sourceContactVector L ω 6; smoothLaw 2 L] := by
  exact covariance_self ((sourceContactVector_hasGaussianLaw hL).eval 6).aemeasurable

/-- Strict positivity of the actual curvature Schur variance follows from the literal contact law. -/
theorem sourceContactSchur_pos {L : ℝ} (hL : L ≠ 0) :
    0 < sourceContactSchur L := by
  have hblock : (Matrix.fromBlocks (sourcePinCovariance L) (sourceContactCrossMatrix L)
      (sourceContactCrossMatrix L)ᴴ (sourceContactQMatrix L)).PosDef := by
    rw [← sourceContactCovariance_block_identity L]
    exact (sourceContactCovariance_posDef hL).submatrix sourceContactBlockEquiv.injective
  have hs := (contact_schur_pos (sourcePinCovariance L) (sourceContactCrossMatrix L)
    (sourceContactQMatrix L) hblock).2
  have heq : ((sourceContactCrossMatrix L)ᴴ * (sourcePinCovariance L)⁻¹ *
      sourceContactCrossMatrix L) () () =
      sourceContactCross L ⬝ᵥ ((sourcePinCovariance L)⁻¹ *ᵥ sourceContactCross L) := by
    simp only [Matrix.mul_apply, Matrix.conjTranspose_apply, sourceContactCrossMatrix,
      star_trivial, dotProduct, Matrix.mulVec]
    simp_rw [Finset.sum_mul]
    rw [Finset.sum_comm]
    simp only [Finset.mul_sum, mul_assoc]
  change 0 < sourceContactQVariance L -
    ((sourceContactCrossMatrix L)ᴴ * (sourcePinCovariance L)⁻¹ * sourceContactCrossMatrix L) () () at hs
  rw [heq] at hs
  exact hs

/-- Finite Gaussian applications use this L² covariance identity; its normal equations are derived. -/
lemma contact_residual_variance_of_isUnit_covariance {Ω ι : Type*} [MeasurableSpace Ω]
    [Fintype ι] [DecidableEq ι] {P : Measure Ω} [IsFiniteMeasure P]
    (J : Ω → ι → ℝ) (Q : Ω → ℝ)
    (hJ : ∀ i, MemLp (fun ω => J ω i) 2 P) (hQ : MemLp Q 2 P)
    (hG : IsUnit (Matrix.of (fun i j =>
      covariance (fun ω => J ω i) (fun ω => J ω j) P) : Matrix ι ι ℝ)) :
    let G : Matrix ι ι ℝ := fun i j => covariance (fun ω => J ω i) (fun ω => J ω j) P
    let b : ι → ℝ := fun i => covariance (fun ω => J ω i) Q P
    variance (fun ω => Q ω - (G⁻¹ *ᵥ b) ⬝ᵥ J ω) P =
      variance Q P - b ⬝ᵥ (G⁻¹ *ᵥ b) := by
  classical
  let G : Matrix ι ι ℝ := fun i j => covariance (fun ω => J ω i) (fun ω => J ω j) P
  let b : ι → ℝ := fun i => covariance (fun ω => J ω i) Q P
  let a : ι → ℝ := G⁻¹ *ᵥ b
  let U : Ω → ℝ := fun ω => a ⬝ᵥ J ω
  change variance (fun ω => Q ω - U ω) P = variance Q P - b ⬝ᵥ a
  have hG' : IsUnit G := hG
  let _ : Invertible G := hG'.invertible
  have hnormal : G *ᵥ a = b := by
    dsimp [a]
    rw [Matrix.mulVec_mulVec, Matrix.mul_inv_of_invertible, Matrix.one_mulVec]
  have hterms (i : ι) : MemLp (fun ω => a i * J ω i) 2 P :=
    (hJ i).const_mul _
  have hU : MemLp U 2 P := memLp_finsetSum Finset.univ (fun i _ => hterms i)
  have hJU (i : ι) : covariance (fun ω => J ω i) U P = b i := by
    change covariance (fun ω => J ω i) (fun ω => ∑ j, a j * J ω j) P = b i
    rw [covariance_fun_sum_right hterms (hJ i)]
    simp_rw [covariance_const_mul_right]
    simpa [G, Matrix.mulVec, dotProduct, mul_comm] using congrFun hnormal i
  have hQU : covariance Q U P = a ⬝ᵥ b := by
    change covariance Q (fun ω => ∑ i, a i * J ω i) P = a ⬝ᵥ b
    rw [covariance_fun_sum_right hterms hQ]
    simp_rw [covariance_const_mul_right]
    simp only [b, dotProduct, covariance_comm Q]
  have hvarU : variance U P = a ⬝ᵥ b := by
    rw [← covariance_self hU.aemeasurable]
    change covariance (fun ω => ∑ i, a i * J ω i) U P = a ⬝ᵥ b
    rw [covariance_fun_sum_left hterms hU]
    simp_rw [covariance_const_mul_left, hJU]
    rfl
  rw [variance_fun_sub hQ hU, hQU, hvarU, dotProduct_comm b a]
  ring

/-- The actual least-squares curvature residual, without replacing the torus covariance. -/
def sourceContactResidual (L : ℝ) (ω : SmoothSample 2 L) : ℝ :=
  sourceContactQ L ω -
    ((sourcePinCovariance L)⁻¹ *ᵥ sourceContactCross L) ⬝ᵥ sourcePinVector L ω

/-- The strictly positive Schur complement is the actual curvature residual variance. -/
theorem sourceContactResidual_variance {L : ℝ} (hL : L ≠ 0) :
    Var[sourceContactResidual L; smoothLaw 2 L] = sourceContactSchur L := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  have hJ (i : Fin 6) : MemLp (fun ω => sourcePinVector L ω i) 2 (smoothLaw 2 L) :=
    ((sourceContactVector_hasGaussianLaw hL).eval i.castSucc).memLp_two
  have hQ : MemLp (sourceContactQ L) 2 (smoothLaw 2 L) :=
    ((sourceContactVector_hasGaussianLaw hL).eval 6).memLp_two
  have hG : IsUnit (Matrix.of (fun i j =>
      covariance (fun ω => sourcePinVector L ω i) (fun ω => sourcePinVector L ω j)
        (smoothLaw 2 L)) : Matrix (Fin 6) (Fin 6) ℝ) := by
    convert sourcePinCovariance_isUnit hL using 1
    ext i j
    rfl
  have hv := contact_residual_variance_of_isUnit_covariance
    (sourcePinVector L) (sourceContactQ L) hJ hQ hG
  change Var[sourceContactResidual L; smoothLaw 2 L] =
    Var[sourceContactQ L; smoothLaw 2 L] -
      sourceContactCross L ⬝ᵥ ((sourcePinCovariance L)⁻¹ *ᵥ sourceContactCross L) at hv
  rw [sourceContactSchur, sourceContactQVariance_eq_variance hL]
  exact hv

theorem sourceContactResidual_variance_pos {L : ℝ} (hL : L ≠ 0) :
    0 < Var[sourceContactResidual L; smoothLaw 2 L] := by
  rw [sourceContactResidual_variance hL]
  exact sourceContactSchur_pos hL

end I4SourceContact
