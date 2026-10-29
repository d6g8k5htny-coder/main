import SourceContactSchur
import Mathlib.Probability.Moments.CovarianceBilin

/-!
# Actual limiting adjusted-endpoint contact parameters

The endpoint order is `(aM,aS,bM,bS,cM,cS)`.  All covariance blocks below
come from literal origin derivatives of the unchanged smooth Fourier source.
The six pins are `(f,fx,fy,fxx,fxy,fxxx)` and their prescribed cubic value is
`(b,0,0,0,0,2)`.  See `SOURCE_CONTACT_ENDPOINT_PARAMETERS.md`.
-/

noncomputable section

open I4SmoothCarrier I4SourceDerivativeGaussian
open MeasureTheory ProbabilityTheory Matrix
open scoped BigOperators Matrix RealInnerProductSpace

set_option autoImplicit false

namespace I4SourceContact

/-- The literal mixed third derivative A=fxxy at the origin. -/
def sourceContactA (L : ℝ) (ω : SmoothSample 2 L) : ℝ :=
  iteratedFDeriv ℝ 3 (smoothSourceField 2 L ω) 0 ![contactE0,contactE0,contactE1]

/-- The actual limiting adjusted endpoints, in the normalizer's six-coordinate order. -/
def sourceContactEndpointVector (L : ℝ) (ω : SmoothSample 2 L) : Fin 6 → ℝ :=
  ![-sourcePinVector L ω 5 / 2, sourcePinVector L ω 5 / 2,
    -sourceContactA L ω / 2, sourceContactA L ω / 2,
    sourceContactQ L ω, sourceContactQ L ω]

/-- Literal limiting endpoint/pin cross covariance C0. -/
def sourceContactEndpointCrossCovariance (L : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  fun i j => covariance (fun ω => sourceContactEndpointVector L ω i)
    (fun ω => sourcePinVector L ω j) (smoothLaw 2 L)

/-- Literal limiting endpoint covariance B0. -/
def sourceContactEndpointCovariance (L : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  fun i j => covariance (fun ω => sourceContactEndpointVector L ω i)
    (fun ω => sourceContactEndpointVector L ω j) (smoothLaw 2 L)

def sourceContactEndpointRegression (L : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  sourceContactEndpointCrossCovariance L * (sourcePinCovariance L)⁻¹

/-- The prescribed limiting pin retains an arbitrary level b. -/
def sourceContactPinValue (b : ℝ) : Fin 6 → ℝ := ![b,0,0,0,0,2]

def sourceContactEndpointMean (L b : ℝ) : Fin 6 → ℝ :=
  sourceContactEndpointRegression L *ᵥ sourceContactPinValue b

def sourceContactEndpointSchur (L : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  sourceContactEndpointCovariance L -
    sourceContactEndpointCrossCovariance L * (sourcePinCovariance L)⁻¹ *
      (sourceContactEndpointCrossCovariance L)ᵀ

def sourceContactEndpointResidual (L : ℝ) (ω : SmoothSample 2 L) : Fin 6 → ℝ :=
  sourceContactEndpointVector L ω -
    sourceContactEndpointRegression L *ᵥ sourcePinVector L ω

private theorem contact_generic_residual_covariance {Ω ι κ : Type*} [MeasurableSpace Ω]
    [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ]
    {P : Measure Ω} [IsFiniteMeasure P]
    (J : Ω → ι → ℝ) (T : Ω → κ → ℝ)
    (hJ : ∀ i, MemLp (fun ω => J ω i) 2 P)
    (hT : ∀ i, MemLp (fun ω => T ω i) 2 P)
    (hG : IsUnit (Matrix.of (fun i j => covariance (fun ω => J ω i) (fun ω => J ω j) P) : Matrix ι ι ℝ)) :
    let G : Matrix ι ι ℝ := fun i j => covariance (fun ω => J ω i) (fun ω => J ω j) P
    let C : Matrix κ ι ℝ := fun i j => covariance (fun ω => T ω i) (fun ω => J ω j) P
    let B : Matrix κ κ ℝ := fun i j => covariance (fun ω => T ω i) (fun ω => T ω j) P
    let D := C * G⁻¹
    let R := fun ω => T ω - D *ᵥ J ω
    (∀ i j, covariance (fun ω => R ω i) (fun ω => R ω j) P = (B - C * G⁻¹ * Cᵀ) i j) ∧
      (B - C * G⁻¹ * Cᵀ).PosSemidef := by
  classical
  intro G C B D R
  have hG' : IsUnit G := hG
  let _ : Invertible G := hG'.invertible
  let U := fun ω => D *ᵥ J ω
  have hU (i : κ) : MemLp (fun ω => U ω i) 2 P :=
    memLp_finsetSum Finset.univ (fun a _ => (hJ a).const_mul (D i a))
  have hR (i : κ) : MemLp (fun ω => R ω i) 2 P := (hT i).sub (hU i)
  have hnormal : D * G = C := by
    dsimp [D]
    rw [Matrix.mul_assoc, Matrix.inv_mul_of_invertible, Matrix.mul_one]
  have hUJ (i : κ) (j : ι) : covariance (fun ω => U ω i) (fun ω => J ω j) P = C i j := by
    change covariance (fun ω => ∑ a, D i a * J ω a) (fun ω => J ω j) P = C i j
    rw [covariance_fun_sum_left (fun a => (hJ a).const_mul _) (hJ j)]
    simp_rw [covariance_const_mul_left]
    exact congrFun (congrFun hnormal i) j
  have hJR (i : ι) (j : κ) : covariance (fun ω => J ω i) (fun ω => R ω j) P = 0 := by
    change covariance (fun ω => J ω i) (fun ω => T ω j - U ω j) P = 0
    rw [covariance_fun_sub_right (hJ i) (hT j) (hU j),
      covariance_comm (fun ω => J ω i) (fun ω => U ω j), hUJ]
    change covariance (fun ω => J ω i) (fun ω => T ω j) P -
      covariance (fun ω => T ω j) (fun ω => J ω i) P = 0
    rw [covariance_comm, sub_self]
  have hUR (i j : κ) : covariance (fun ω => U ω i) (fun ω => R ω j) P = 0 := by
    change covariance (fun ω => ∑ a, D i a * J ω a) (fun ω => R ω j) P = 0
    rw [covariance_fun_sum_left (fun a => (hJ a).const_mul _) (hR j)]
    simp_rw [covariance_const_mul_left, hJR, mul_zero]
    exact Finset.sum_const_zero
  have hUT (i j : κ) : covariance (fun ω => U ω i) (fun ω => T ω j) P = (D * Cᵀ) i j := by
    change covariance (fun ω => ∑ a, D i a * J ω a) (fun ω => T ω j) P = _
    rw [covariance_fun_sum_left (fun a => (hJ a).const_mul _) (hT j)]
    simp_rw [covariance_const_mul_left]
    simp only [Matrix.mul_apply, C]
    apply Finset.sum_congr rfl
    intro a _
    rw [covariance_comm (fun ω => J ω a)]
    rfl
  have hcov (i j : κ) : covariance (fun ω => R ω i) (fun ω => R ω j) P = (B - C * G⁻¹ * Cᵀ) i j := by
    change covariance (fun ω => R ω i) (fun ω => T ω j - U ω j) P = _
    rw [covariance_fun_sub_right (hR i) (hT j) (hU j),
      covariance_comm (fun ω => R ω i) (fun ω => U ω j), hUR, sub_zero]
    change covariance (fun ω => T ω i - U ω i) (fun ω => T ω j) P = _
    rw [covariance_fun_sub_left (hT i) (hU i) (hT j), hUT]
    rfl
  refine ⟨hcov, ?_⟩
  apply Matrix.PosSemidef.of_dotProduct_mulVec_nonneg
  · ext i j
    simp only [Matrix.conjTranspose_apply, star_trivial]
    rw [← hcov i j, ← hcov j i]
    exact covariance_comm _ _
  · intro x
    have hp := covarianceBilin_self_nonneg (μ := P.map (fun ω => WithLp.toLp 2 (R ω))) (WithLp.toLp 2 x)
    rw [covarianceBilin_apply_pi hR] at hp
    simpa [hcov, dotProduct, Matrix.mulVec, Finset.mul_sum, mul_assoc, mul_left_comm, mul_comm] using hp

private theorem sourceContactA_memLp {L : ℝ} (hL : L ≠ 0) :
    MemLp (sourceContactA L) 2 (smoothLaw 2 L) := by
  exact ((smoothSourceField_derivativeVector_hasGaussianLaw 2 hL 1 (fun _ => 3)
    (fun _ => 0) (fun _ => ![contactE0,contactE0,contactE1])).eval 0).memLp_two

private theorem sourcePin_memLp {L : ℝ} (hL : L ≠ 0) (i : Fin 6) :
    MemLp (fun ω => sourcePinVector L ω i) 2 (smoothLaw 2 L) :=
  ((sourceContactVector_hasGaussianLaw hL).eval i.castSucc).memLp_two

private theorem sourceContactEndpoint_memLp {L : ℝ} (hL : L ≠ 0) (i : Fin 6) :
    MemLp (fun ω => sourceContactEndpointVector L ω i) 2 (smoothLaw 2 L) := by
  have hQ := ((sourceContactVector_hasGaussianLaw hL).eval 6).memLp_two
  fin_cases i
  · simpa [sourceContactEndpointVector, div_eq_mul_inv, mul_comm] using
      ((sourcePin_memLp hL 5).neg).const_mul ((2 : ℝ)⁻¹)
  · simpa [sourceContactEndpointVector, div_eq_mul_inv, mul_comm] using
      (sourcePin_memLp hL 5).const_mul ((2 : ℝ)⁻¹)
  · simpa [sourceContactEndpointVector, div_eq_mul_inv, mul_comm] using
      (sourceContactA_memLp hL).neg.const_mul ((2 : ℝ)⁻¹)
  · simpa [sourceContactEndpointVector, div_eq_mul_inv, mul_comm] using
      (sourceContactA_memLp hL).const_mul ((2 : ℝ)⁻¹)
  · exact hQ
  · exact hQ

private theorem cross_row_zero (L : ℝ) (j : Fin 6) :
    sourceContactEndpointCrossCovariance L 0 j = -sourcePinCovariance L 5 j / 2 := by
  change covariance (fun ω => -sourcePinVector L ω 5 / 2)
    (fun ω => sourcePinVector L ω j) (smoothLaw 2 L) = _
  rw [covariance_fun_div_left, covariance_fun_neg_left]
  rfl

private theorem cross_row_one (L : ℝ) (j : Fin 6) :
    sourceContactEndpointCrossCovariance L 1 j = sourcePinCovariance L 5 j / 2 := by
  change covariance (fun ω => sourcePinVector L ω 5 / 2)
    (fun ω => sourcePinVector L ω j) (smoothLaw 2 L) = _
  rw [covariance_fun_div_left]
  rfl

private theorem cross_row_two (L : ℝ) (j : Fin 6) :
    sourceContactEndpointCrossCovariance L 2 j =
      -sourceContactEndpointCrossCovariance L 3 j := by
  change covariance (fun ω => -sourceContactA L ω / 2)
    (fun ω => sourcePinVector L ω j) (smoothLaw 2 L) =
      -covariance (fun ω => sourceContactA L ω / 2)
        (fun ω => sourcePinVector L ω j) (smoothLaw 2 L)
  rw [covariance_fun_div_left, covariance_fun_div_left, covariance_fun_neg_left]
  ring

private theorem cross_row_four (L : ℝ) (j : Fin 6) :
    sourceContactEndpointCrossCovariance L 4 j =
      sourceContactEndpointCrossCovariance L 5 j := rfl

private theorem regression_pin_coordinates {L : ℝ} (hL : L ≠ 0) (x : Fin 6 → ℝ) :
    (sourceContactEndpointRegression L *ᵥ x) 0 = -x 5 / 2 ∧
    (sourceContactEndpointRegression L *ᵥ x) 1 = x 5 / 2 := by
  let _ : Invertible (sourcePinCovariance L) := (sourcePinCovariance_isUnit hL).invertible
  have hcancel : sourcePinCovariance L *ᵥ ((sourcePinCovariance L)⁻¹ *ᵥ x) = x := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_inv_of_invertible, Matrix.one_mulVec]
  have hc := congrFun hcancel 5
  rw [sourceContactEndpointRegression, ← Matrix.mulVec_mulVec]
  change (∑ j, sourceContactEndpointCrossCovariance L 0 j *
      ((sourcePinCovariance L)⁻¹ *ᵥ x) j) = -x 5 / 2 ∧
    (∑ j, sourceContactEndpointCrossCovariance L 1 j *
      ((sourcePinCovariance L)⁻¹ *ᵥ x) j) = x 5 / 2
  simp_rw [cross_row_zero, cross_row_one]
  change (∑ j, sourcePinCovariance L 5 j *
    ((sourcePinCovariance L)⁻¹ *ᵥ x) j) = x 5 at hc
  constructor
  · calc
      _ = (-1 / 2 : ℝ) * (∑ j, sourcePinCovariance L 5 j *
          ((sourcePinCovariance L)⁻¹ *ᵥ x) j) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = -x 5 / 2 := by rw [hc]; ring
  · calc
      _ = (1 / 2 : ℝ) * (∑ j, sourcePinCovariance L 5 j *
          ((sourcePinCovariance L)⁻¹ *ᵥ x) j) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = x 5 / 2 := by rw [hc]; ring

private theorem regression_other_coordinates (L : ℝ) (x : Fin 6 → ℝ) :
    (sourceContactEndpointRegression L *ᵥ x) 2 =
      -(sourceContactEndpointRegression L *ᵥ x) 3 ∧
    (sourceContactEndpointRegression L *ᵥ x) 4 =
      (sourceContactEndpointRegression L *ᵥ x) 5 := by
  rw [sourceContactEndpointRegression, ← Matrix.mulVec_mulVec]
  constructor
  · simp only [Matrix.mulVec, dotProduct, cross_row_two, neg_mul, Finset.sum_neg_distrib]
  · simp only [Matrix.mulVec, dotProduct, cross_row_four]

theorem sourceContactEndpointMean_contact {L : ℝ} (hL : L ≠ 0) (b : ℝ) :
    sourceContactEndpointMean L b 0 = -1 ∧
    sourceContactEndpointMean L b 1 = 1 ∧
    sourceContactEndpointMean L b 2 + sourceContactEndpointMean L b 3 = 0 ∧
    sourceContactEndpointMean L b 4 - sourceContactEndpointMean L b 5 = 0 := by
  have hp := regression_pin_coordinates hL (sourceContactPinValue b)
  have ho := regression_other_coordinates L (sourceContactPinValue b)
  change _ ∧ _ ∧ _ ∧ _
  dsimp only [sourceContactEndpointMean]
  refine ⟨?_, ?_, ?_, ?_⟩
  · simpa [sourceContactPinValue] using hp.1
  · simpa [sourceContactPinValue] using hp.2
  · rw [ho.1]; ring
  · rw [ho.2]; ring

private theorem sourceContactEndpointResidual_contact {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample 2 L) :
    sourceContactEndpointResidual L ω 0 = 0 ∧
    sourceContactEndpointResidual L ω 1 = 0 ∧
    sourceContactEndpointResidual L ω 2 = -sourceContactEndpointResidual L ω 3 ∧
    sourceContactEndpointResidual L ω 4 = sourceContactEndpointResidual L ω 5 := by
  have hp := regression_pin_coordinates hL (sourcePinVector L ω)
  have ho := regression_other_coordinates L (sourcePinVector L ω)
  dsimp only [sourceContactEndpointResidual, Pi.sub_apply, sourceContactEndpointVector,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hp.1]; ring
  · rw [hp.2]; ring
  · rw [ho.1]; ring
  · rw [ho.2]

/-- The limiting Schur matrix is the actual least-squares endpoint residual covariance. -/
theorem sourceContactEndpointSchur_covariance {L : ℝ} (hL : L ≠ 0) :
    (∀ i j, covariance (fun ω => sourceContactEndpointResidual L ω i)
      (fun ω => sourceContactEndpointResidual L ω j) (smoothLaw 2 L) =
      sourceContactEndpointSchur L i j) ∧ (sourceContactEndpointSchur L).PosSemidef := by
  have : IsProbabilityMeasure (smoothLaw 2 L) := smoothLaw_isProbabilityMeasure 2 hL
  have hG : IsUnit (Matrix.of (fun i j =>
      covariance (fun ω => sourcePinVector L ω i) (fun ω => sourcePinVector L ω j)
        (smoothLaw 2 L)) : Matrix (Fin 6) (Fin 6) ℝ) := by
    convert sourcePinCovariance_isUnit hL using 1
    ext i j
    rfl
  exact contact_generic_residual_covariance (sourcePinVector L) (sourceContactEndpointVector L)
    (sourcePin_memLp hL) (sourceContactEndpoint_memLp hL) hG

theorem sourceContactEndpointSchur_posSemidef {L : ℝ} (hL : L ≠ 0) :
    (sourceContactEndpointSchur L).PosSemidef :=
  (sourceContactEndpointSchur_covariance hL).2

/-- The deterministic endpoint relations become exactly the four required zero variances. -/
theorem sourceContactEndpointSchur_contact {L : ℝ} (hL : L ≠ 0) :
    sourceContactEndpointSchur L 0 0 = 0 ∧
    sourceContactEndpointSchur L 1 1 = 0 ∧
    sourceContactEndpointSchur L 2 2 + 2 * sourceContactEndpointSchur L 2 3 +
      sourceContactEndpointSchur L 3 3 = 0 ∧
    sourceContactEndpointSchur L 4 4 - 2 * sourceContactEndpointSchur L 4 5 +
      sourceContactEndpointSchur L 5 5 = 0 := by
  have hc := (sourceContactEndpointSchur_covariance hL).1
  have hzero : (fun ω => sourceContactEndpointResidual L ω 0) = 0 :=
    funext fun ω => (sourceContactEndpointResidual_contact hL ω).1
  have hone : (fun ω => sourceContactEndpointResidual L ω 1) = 0 :=
    funext fun ω => (sourceContactEndpointResidual_contact hL ω).2.1
  have htwo : (fun ω => sourceContactEndpointResidual L ω 2) =
      -(fun ω => sourceContactEndpointResidual L ω 3) :=
    funext fun ω => (sourceContactEndpointResidual_contact hL ω).2.2.1
  have hfour : (fun ω => sourceContactEndpointResidual L ω 4) =
      (fun ω => sourceContactEndpointResidual L ω 5) :=
    funext fun ω => (sourceContactEndpointResidual_contact hL ω).2.2.2
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [← hc 0 0, hzero]; simp
  · rw [← hc 1 1, hone]; simp
  · rw [← hc 2 2, ← hc 2 3, ← hc 3 3, htwo]
    simp only [covariance_neg_left, covariance_neg_right]
    ring
  · rw [← hc 4 4, ← hc 4 5, ← hc 5 5, hfour]
    ring

private theorem cross_row_Q (L : ℝ) (j : Fin 6) :
    sourceContactEndpointCrossCovariance L 4 j = sourceContactCross L j := by
  exact covariance_comm _ _

/-- The reference entry is the already established actual torus curvature Schur variance. -/
theorem sourceContactEndpointSchur_four_four_eq_schur (L : ℝ) :
    sourceContactEndpointSchur L 4 4 = sourceContactSchur L := by
  have heq : (sourceContactEndpointCrossCovariance L * (sourcePinCovariance L)⁻¹ *
      (sourceContactEndpointCrossCovariance L)ᵀ) 4 4 =
      sourceContactCross L ⬝ᵥ ((sourcePinCovariance L)⁻¹ *ᵥ sourceContactCross L) := by
    simp only [Matrix.mul_apply, Matrix.transpose_apply, cross_row_Q, dotProduct, Matrix.mulVec]
    simp_rw [Finset.sum_mul]
    rw [Finset.sum_comm]
    simp only [Finset.mul_sum, mul_assoc]
  change sourceContactQVariance L - _ = sourceContactQVariance L - _
  rw [heq]

theorem sourceContactEndpointSchur_referenceVariance_pos {L : ℝ} (hL : L ≠ 0) :
    0 < sourceContactEndpointSchur L 4 4 := by
  rw [sourceContactEndpointSchur_four_four_eq_schur]
  exact sourceContactSchur_pos hL

end I4SourceContact
