import SourceLifetimeField

/-! Source-bound endpoint and degenerate-family controls. These declarations are
supporting checks; the constructor theorem itself has unrestricted d,m and L>0.
-/

noncomputable section
open MeasureTheory ProbabilityTheory
open UniversalLaw.Spec.Lifetime I4SmoothCarrier I4LifetimeField
open scoped BigOperators NNReal ENNReal

namespace I4LifetimeControls

def unrestrictedEndpoint (d : ℕ) (L : ℝ) (hL : 0 < L) :
    GaussianFieldInterface d L := actualGaussianFieldInterface d L hL

theorem literalCarrier (d : ℕ) (L : ℝ) (hL : 0 < L) :
    (actualGaussianFieldInterface d L hL).Ω = SmoothSample d L := rfl

theorem literalLaw (d : ℕ) (L : ℝ) (hL : 0 < L) :
    (actualGaussianFieldInterface d L hL).P = smoothLaw d L := rfl

theorem literalField (d : ℕ) (L : ℝ) (hL : 0 < L)
    (ω : SmoothSample d L) (x : E d) :
    (actualGaussianFieldInterface d L hL).F ω x =
      smoothSourceField d L ω (EuclideanSpace.equiv (Fin d) ℝ x) := rfl

theorem sameCarrierOrders (d : ℕ) (L : ℝ) (hL : 0 < L)
    (ω : SmoothSample d L) :
    ContDiff ℝ 5 ((actualGaussianFieldInterface d L hL).F ω) ∧
      ContDiff ℝ 37 ((actualGaussianFieldInterface d L hL).F ω) :=
  ⟨(actualGaussianFieldInterface d L hL).smooth ω 5,
    (actualGaussianFieldInterface d L hL).smooth ω 37⟩

theorem wholeCoefficientLaw (d : ℕ) (L : ℝ) (hL : 0 < L) :
    (actualGaussianFieldInterface d L hL).P.map (sampleInclusion d L) =
      I4Source.sourceLaw d := sampleInclusion_map d hL.ne'

theorem periodicCovariance_zero (d : ℕ) {L : ℝ} (hL : 0 < L) :
    periodicCovariance d L 0 = 1 := by
  exact div_self (ne_of_gt (I4ThetaBridge.theta_zero_pos d hL))

theorem rawDiagonalOne (d : ℕ) {L : ℝ} (hL : 0 < L) (x : E d) :
    ∫ ω, actualField d L ω x * actualField d L ω x ∂smoothLaw d L = 1 := by
  rw [actualField_raw_second_moment d hL, sub_self, periodicCovariance_zero d hL]

theorem zeroMean (d : ℕ) {L : ℝ} (hL : 0 < L) (x : E d) :
    ∫ ω, actualField d L ω x ∂smoothLaw d L = 0 :=
  actualField_integral_zero d hL x

theorem dimensionZeroTheta (L : ℝ) : thetaSum 0 L 0 = 1 := by
  simp [thetaSum, latticeVec]

theorem dimensionZeroSpectralNormalizer (L : ℝ) : I4Weights.sourceZ 0 L = 1 := by
  simp [I4Weights.sourceZ, I4Weights.sourceWeight, I4Weights.productDecay]

theorem dimensionZeroRawVariance (L : ℝ) (hL : 0 < L) :
    ∫ ω, actualField 0 L ω 0 * actualField 0 L ω 0 ∂smoothLaw 0 L = 1 :=
  rawDiagonalOne 0 hL 0

theorem emptyQuad (d : ℕ) (L : ℝ) (x : Fin 0 → E d) (a : Fin 0 → ℝ) :
    covQuadForm d L x a = 0 := by simp [covQuadForm]

theorem emptyFamilyLaw (d : ℕ) {L : ℝ} (hL : 0 < L)
    (x : Fin 0 → E d) (a : Fin 0 → ℝ) :
    (smoothLaw d L).map (fun ω => ∑ i, a i * actualField d L ω (x i)) =
      Measure.dirac 0 := by
  have hv : (⟨covQuadForm d L x a, covQuadForm_nonneg d hL x a⟩ : ℝ≥0) = 0 := by
    apply Subtype.ext
    exact emptyQuad d L x a
  exact (actualField_linearCombination_map d hL x a).trans
    ((congrArg (gaussianReal 0) hv).trans (gaussianReal_zero_var 0))

theorem cancelledQuad (d : ℕ) {L : ℝ} (_hL : 0 < L) (x : E d) :
    covQuadForm d L ![x, x] ![1, -1] = 0 := by
  simp [covQuadForm, Fin.sum_univ_two]

theorem cancelledSample (d : ℕ) (L : ℝ) (x : E d) (ω : SmoothSample d L) :
    (∑ i : Fin 2, (![1, -1] : Fin 2 → ℝ) i *
      actualField d L ω ((![x, x] : Fin 2 → E d) i)) = 0 := by
  simp [Fin.sum_univ_two]

theorem cancelledLaw (d : ℕ) {L : ℝ} (hL : 0 < L) (x : E d) :
    (smoothLaw d L).map (fun ω => ∑ i : Fin 2,
      (![1, -1] : Fin 2 → ℝ) i * actualField d L ω ((![x, x] : Fin 2 → E d) i)) =
      Measure.dirac 0 := by
  have hv : (⟨covQuadForm d L ![x, x] ![1, -1],
      covQuadForm_nonneg d hL ![x, x] ![1, -1]⟩ : ℝ≥0) = 0 := by
    apply Subtype.ext
    exact cancelledQuad d hL x
  exact (actualField_linearCombination_map d hL ![x, x] ![1, -1]).trans
    ((congrArg (gaussianReal 0) hv).trans (gaussianReal_zero_var 0))

theorem reinforcedQuad (d : ℕ) {L : ℝ} (hL : 0 < L) (x : E d) :
    covQuadForm d L ![x, x] ![1, 1] = 4 := by
  norm_num [covQuadForm, Fin.sum_univ_two, periodicCovariance_zero d hL]

theorem reinforcedLaw (d : ℕ) {L : ℝ} (hL : 0 < L) (x : E d) :
    (smoothLaw d L).map (fun ω => ∑ i : Fin 2,
      (![1, 1] : Fin 2 → ℝ) i * actualField d L ω ((![x, x] : Fin 2 → E d) i)) =
      gaussianReal 0 4 := by
  convert actualField_linearCombination_map d hL ![x, x] ![1, 1] using 1
  congr 1
  apply Subtype.ext
  exact (reinforcedQuad d hL x).symm

theorem everyLatticeTranslation (d : ℕ) (L : ℝ) (hL : 0 < L)
    (ω : SmoothSample d L) (x : E d) (n : Fin d → ℤ) :
    actualField d L ω (x + latticeVec d L n) = actualField d L ω x :=
  (actualGaussianFieldInterface d L hL).periodic ω x n

theorem euclideanNormSquared :
    ‖(WithLp.toLp 2 (![1, 1] : Fin 2 → ℝ) : E 2)‖ ^ 2 = 2 := by
  norm_num [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two]

/-- Adding one leaves centered covariance unchanged, but violates the required mean. -/
theorem shiftedMeanOne (d : ℕ) {L : ℝ} (hL : 0 < L) (x : E d) :
    ∫ ω, (actualField d L ω x + 1) ∂smoothLaw d L = 1 := by
  let := smoothLaw_isProbabilityMeasure d hL.ne'
  rw [integral_add (actualField_eval_hasGaussianLaw d hL x).integrable
    (integrable_const 1), actualField_integral_zero d hL]
  simp

end I4LifetimeControls
