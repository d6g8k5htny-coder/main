import Mathlib.Probability.HasCondDistrib
import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Independence

/-!
# Canonical Gaussian regression kernel

For a residual law `ν`, `regressionKernel ν B mX mY x` is the literal affine pushforward
`ν.map (fun r => r + mY + B (x - mX))` at every observation `x`. The kernel is constructed
measurably from a deterministic observation and a constant residual kernel, so it has a
specified value even where the observation has probability zero.

The Gaussian theorem derives observation/residual independence from joint Gaussianity and
zero coordinate cross-covariance. No nondegeneracy or residual covariance inverse is needed.
The direct observed/target theorem derives those residual hypotheses from the coordinate
normal equations for `B`. The construction does not identify concrete derivative observations,
regression or Schur covariance blocks, a Palm weight, or a conditional law of a literal smooth field.

Source contract: `SOURCE_REGRESSION_KERNEL.md` in this packet. Author-side Lean verification
is separate from source alignment and the required review by another model family.
-/

open MeasureTheory ProbabilityTheory
open scoped ProbabilityTheory BigOperators

set_option autoImplicit false

namespace ResearchFormalCoreR1

/-- Copying the first coordinate and retaining an independent second coordinate. -/
lemma regression_compProd_id_const {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (μ : Measure α) (ν : Measure β) [SFinite μ] [SFinite ν] :
    μ ⊗ₘ (Kernel.id ×ₖ Kernel.const α ν) =
      (μ.prod ν).map (fun p : α × β => (p.1, p)) := by
  ext s hs
  rw [Measure.compProd_apply hs, Measure.map_apply (by fun_prop) hs,
    Measure.prod_apply (s := (fun p : α × β => (p.1, p)) ⁻¹' s) (hs.preimage (by fun_prop))]
  simp_rw [Kernel.prod_apply, Kernel.id_apply, Kernel.const_apply, Measure.dirac_prod]
  congr with x
  rw [Measure.map_apply (by fun_prop) (measurable_prodMk_left hs)]
  rfl

/-- An independent residual paired with the observation has the explicit product kernel. -/
lemma regression_pair_hasCondDistrib {Ω α β : Type*} [MeasurableSpace Ω]
    [MeasurableSpace α] [MeasurableSpace β] {P : Measure Ω} [IsProbabilityMeasure P]
    {X : Ω → α} {R : Ω → β} (hX : AEMeasurable X P) (hR : AEMeasurable R P)
    (hXR : IndepFun X R P) :
    HasCondDistrib (fun ω => (X ω, R ω)) X
      (Kernel.id ×ₖ Kernel.const α (P.map R)) P := by
  refine ⟨hX.prodMk (hX.prodMk hR), ?_⟩
  calc
    P.map (fun ω => (X ω, (X ω, R ω))) =
        (P.map (fun ω => (X ω, R ω))).map (fun p : α × β => (p.1, p)) := by
      rw [AEMeasurable.map_map_of_aemeasurable (by fun_prop) (hX.prodMk hR)]
      rfl
    _ = ((P.map X).prod (P.map R)).map (fun p : α × β => (p.1, p)) := by
      rw [hXR.map_prod_eq_prod_map_map hX hR]
    _ = (P.map X) ⊗ₘ (Kernel.id ×ₖ Kernel.const α (P.map R)) :=
      (regression_compProd_id_const (P.map X) (P.map R)).symm


/-- Affine reconstruction of the target from the observation and the regression residual. -/
def regressionReconstruction {n k : ℕ}
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ)
    (p : (Fin n → ℝ) × (Fin k → ℝ)) : Fin k → ℝ :=
  p.2 + mY + B (p.1 - mX)

lemma regressionReconstruction_measurable {n k : ℕ}
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ) :
    Measurable (regressionReconstruction B mX mY) := by
  unfold regressionReconstruction
  fun_prop

/-- The canonical regression law is specified at every observation, including null pins. -/
noncomputable def regressionKernel {n k : ℕ} (ν : Measure (Fin k → ℝ))
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ) :
    Kernel (Fin n → ℝ) (Fin k → ℝ) :=
  ((Kernel.id : Kernel (Fin n → ℝ) (Fin n → ℝ)) ×ₖ
    Kernel.const (Fin n → ℝ) ν).map (regressionReconstruction B mX mY)

instance regressionKernel_isMarkov {n k : ℕ} (ν : Measure (Fin k → ℝ))
    [IsProbabilityMeasure ν] (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ))
    (mX : Fin n → ℝ) (mY : Fin k → ℝ) :
    IsMarkovKernel (regressionKernel ν B mX mY) := by
  unfold regressionKernel
  exact Kernel.IsMarkovKernel.map _ (regressionReconstruction_measurable B mX mY)

lemma regressionKernel_apply {n k : ℕ} (ν : Measure (Fin k → ℝ)) [SFinite ν]
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ)
    (x : Fin n → ℝ) :
    regressionKernel ν B mX mY x = ν.map (fun r => r + mY + B (x - mX)) := by
  rw [regressionKernel, Kernel.map_apply _ (regressionReconstruction_measurable B mX mY),
    Kernel.prod_apply, Kernel.id_apply, Kernel.const_apply, Measure.dirac_prod,
    Measure.map_map (regressionReconstruction_measurable B mX mY) (by fun_prop)]
  rfl

/-- Affine reconstruction from an independent residual has the canonical conditional kernel. -/
lemma independentResidual_hasCondDistrib {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} [IsProbabilityMeasure P] {n k : ℕ}
    {X : Ω → Fin n → ℝ} {R : Ω → Fin k → ℝ}
    (hX : AEMeasurable X P) (hR : AEMeasurable R P) (hXR : IndepFun X R P)
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ) :
    HasCondDistrib (fun ω => R ω + mY + B (X ω - mX)) X
      (regressionKernel (P.map R) B mX mY) P := by
  exact (regression_pair_hasCondDistrib hX hR hXR).comp_left
    (regressionReconstruction_measurable B mX mY)

/-- Joint Gaussianity and vanishing observation/residual cross-covariance give the canonical
conditional law, without choosing an almost-everywhere version at a prescribed observation. -/
theorem gaussianRegression_hasCondDistrib {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} {n k : ℕ} {X : Ω → Fin n → ℝ} {R : Ω → Fin k → ℝ}
    (hXR : HasGaussianLaw (fun ω => (X ω, R ω)) P)
    (hzero : ∀ i j, covariance (fun ω => X ω i) (fun ω => R ω j) P = 0)
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ) :
    HasCondDistrib (fun ω => R ω + mY + B (X ω - mX)) X
      (regressionKernel (P.map R) B mX mY) P := by
  have : IsProbabilityMeasure P := hXR.isProbabilityMeasure
  exact independentResidual_hasCondDistrib hXR.fst.aemeasurable hXR.snd.aemeasurable
    (hXR.indepFun_of_covariance_eval hzero) B mX mY

/-- Kernel measurability is pointwise in the observation and does not require a Gaussian law. -/
lemma regressionKernel_measurable {n k : ℕ} (ν : Measure (Fin k → ℝ))
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ) :
    Measurable (regressionKernel ν B mX mY) :=
  Kernel.measurable _

/-- All parts of the canonical Gaussian conditional-law construction, including the literal
pushforward at every prescribed observation. Zero-dimensional and degenerate laws are allowed. -/
theorem gaussianRegressionKernel_spec {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} {n k : ℕ} {X : Ω → Fin n → ℝ} {R : Ω → Fin k → ℝ}
    (hXR : HasGaussianLaw (fun ω => (X ω, R ω)) P)
    (hzero : ∀ i j, covariance (fun ω => X ω i) (fun ω => R ω j) P = 0)
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ) :
    IsMarkovKernel (regressionKernel (P.map R) B mX mY) ∧
      Measurable (regressionKernel (P.map R) B mX mY) ∧
      (∀ x, regressionKernel (P.map R) B mX mY x =
        (P.map R).map (fun r => r + mY + B (x - mX))) ∧
      HasCondDistrib (fun ω => R ω + mY + B (X ω - mX)) X
        (regressionKernel (P.map R) B mX mY) P := by
  have : IsProbabilityMeasure P := hXR.isProbabilityMeasure
  exact ⟨inferInstance, regressionKernel_measurable _ B mX mY,
    regressionKernel_apply _ B mX mY, gaussianRegression_hasCondDistrib hXR hzero B mX mY⟩

/-- Coordinates of a finite-vector continuous linear map in the standard basis. -/
lemma regression_map_coordinate {n k : ℕ}
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (v : Fin n → ℝ) (j : Fin k) :
    B v j = ∑ l, B (Pi.single l 1) j * v l := by
  calc
    B v j = B (∑ l, v l • (Pi.single l 1 : Fin n → ℝ)) j :=
      congrArg (fun w => B w j) (pi_eq_sum_univ' v)
    _ = ∑ l, B (Pi.single l 1) j * v l := by
      simp [map_sum, mul_comm]

/-- The covariance of a centered finite-vector linear predictor is its normal-equation sum. -/
lemma regression_covariance_map {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} [IsProbabilityMeasure P] {n k : ℕ}
    {X : Ω → Fin n → ℝ} (hX : HasGaussianLaw X P)
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ)
    (i : Fin n) (j : Fin k) :
    covariance (fun ω => X ω i) (fun ω => B (X ω - mX) j) P =
      ∑ l, B (Pi.single l 1) j * covariance (fun ω => X ω i) (fun ω => X ω l) P := by
  have hcoord : (fun ω => B (X ω - mX) j) =
      (fun ω => ∑ l, B (Pi.single l 1) j * (X ω l - mX l)) := by
    funext ω
    exact regression_map_coordinate B (X ω - mX) j
  rw [hcoord, covariance_fun_sum_right]
  · apply Finset.sum_congr rfl
    intro l _
    rw [covariance_const_mul_right, covariance_sub_const_right (hX.eval l).integrable]
  · intro l
    exact ((hX.eval l).memLp_two.sub (memLp_const _)).const_mul _
  · exact (hX.eval i).memLp_two

/-- The actual regression residual has zero observation cross-covariance by the normal equations. -/
lemma regression_residual_zero_covariance {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} {n k : ℕ} {X : Ω → Fin n → ℝ} {Y : Ω → Fin k → ℝ}
    (hXY : HasGaussianLaw (fun ω => (X ω, Y ω)) P)
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ)
    (hnormal : ∀ i j, covariance (fun ω => X ω i) (fun ω => Y ω j) P =
      ∑ l, B (Pi.single l 1) j * covariance (fun ω => X ω i) (fun ω => X ω l) P) :
    ∀ i j, covariance (fun ω => X ω i)
      (fun ω => (Y ω - mY - B (X ω - mX)) j) P = 0 := by
  have : IsProbabilityMeasure P := hXY.isProbabilityMeasure
  intro i j
  have hXi := (hXY.fst.eval i).memLp_two
  have hYj := (hXY.snd.eval j).memLp_two
  have hBj : MemLp (fun ω => B (X ω - mX) j) 2 P := by
    have hcoord : (fun ω => B (X ω - mX) j) =
        (fun ω => ∑ l, B (Pi.single l 1) j * (X ω l - mX l)) := by
      funext ω
      exact regression_map_coordinate B (X ω - mX) j
    rw [hcoord]
    exact memLp_finsetSum _ (fun l _ =>
      ((hXY.fst.eval l).memLp_two.sub (memLp_const _)).const_mul _)
  simp only [Pi.sub_apply]
  rw [covariance_fun_sub_right (Y := fun ω => Y ω j - mY j) hXi (hYj.sub (memLp_const _)) hBj,
    covariance_sub_const_right (hXY.snd.eval j).integrable,
    regression_covariance_map hXY.fst B mX i j, hnormal i j, sub_self]

/-- The observation and the actual affine residual are jointly Gaussian. -/
lemma regression_residual_jointGaussian {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} {n k : ℕ}
    {X : Ω → Fin n → ℝ} {Y : Ω → Fin k → ℝ}
    (hXY : HasGaussianLaw (fun ω => (X ω, Y ω)) P)
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ) :
    HasGaussianLaw (fun ω => (X ω, Y ω - mY - B (X ω - mX))) P := by
  let L : ((Fin n → ℝ) × (Fin k → ℝ)) →L[ℝ]
      ((Fin n → ℝ) × (Fin k → ℝ)) :=
    (ContinuousLinearMap.fst ℝ (Fin n → ℝ) (Fin k → ℝ)).prod
      (ContinuousLinearMap.snd ℝ (Fin n → ℝ) (Fin k → ℝ) -
        B.comp (ContinuousLinearMap.fst ℝ (Fin n → ℝ) (Fin k → ℝ)))
  let c : (Fin n → ℝ) × (Fin k → ℝ) := (0, B mX - mY)
  have hL : HasGaussianLaw (fun ω => L (X ω, Y ω)) P := hXY.map_fun L
  have hshift : HasGaussianLaw (fun ω => L (X ω, Y ω) + c) P := {
    aemeasurable := hL.aemeasurable.add aemeasurable_const
    isGaussian_map := by
      have : IsGaussian (P.map (fun ω => L (X ω, Y ω))) := hL.isGaussian_map
      convert (inferInstance : IsGaussian
        ((P.map (fun ω => L (X ω, Y ω))).map (fun p => p + c))) using 1
      rw [AEMeasurable.map_map_of_aemeasurable (by fun_prop) hL.aemeasurable]
      rfl }
  apply hshift.congr
  filter_upwards [] with ω
  apply Prod.ext
  · simp [L, c]
  · change Y ω - B (X ω) + (B mX - mY) = Y ω - mY - B (X ω - mX)
    rw [map_sub]
    abel



/-- The actual target has the explicit regression conditional law when the coefficient map
satisfies the coordinate normal equations. The residual hypotheses are derived. -/
theorem gaussianRegression_observed_hasCondDistrib {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} {n k : ℕ} {X : Ω → Fin n → ℝ} {Y : Ω → Fin k → ℝ}
    (hXY : HasGaussianLaw (fun ω => (X ω, Y ω)) P)
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ)
    (hnormal : ∀ i j, covariance (fun ω => X ω i) (fun ω => Y ω j) P =
      ∑ l, B (Pi.single l 1) j * covariance (fun ω => X ω i) (fun ω => X ω l) P) :
    HasCondDistrib Y X
      (regressionKernel (P.map (fun ω => Y ω - mY - B (X ω - mX))) B mX mY) P := by
  have hXR := regression_residual_jointGaussian hXY B mX mY
  have hzero := regression_residual_zero_covariance hXY B mX mY hnormal
  have hcond := gaussianRegression_hasCondDistrib hXR hzero B mX mY
  convert hcond using 1
  funext ω
  abel

/-- The canonical conditional-law specification for the actual jointly Gaussian target.
The residual Gaussianity and its zero cross-covariance are conclusions of the construction. -/
theorem gaussianRegression_observed_spec {Ω : Type*} [MeasurableSpace Ω]
    {P : Measure Ω} {n k : ℕ} {X : Ω → Fin n → ℝ} {Y : Ω → Fin k → ℝ}
    (hXY : HasGaussianLaw (fun ω => (X ω, Y ω)) P)
    (B : (Fin n → ℝ) →L[ℝ] (Fin k → ℝ)) (mX : Fin n → ℝ) (mY : Fin k → ℝ)
    (hnormal : ∀ i j, covariance (fun ω => X ω i) (fun ω => Y ω j) P =
      ∑ l, B (Pi.single l 1) j * covariance (fun ω => X ω i) (fun ω => X ω l) P) :
    IsMarkovKernel
        (regressionKernel (P.map (fun ω => Y ω - mY - B (X ω - mX))) B mX mY) ∧
      Measurable
        (regressionKernel (P.map (fun ω => Y ω - mY - B (X ω - mX))) B mX mY) ∧
      (∀ x, regressionKernel (P.map (fun ω => Y ω - mY - B (X ω - mX))) B mX mY x =
        (P.map (fun ω => Y ω - mY - B (X ω - mX))).map
          (fun r => r + mY + B (x - mX))) ∧
      HasCondDistrib Y X
        (regressionKernel (P.map (fun ω => Y ω - mY - B (X ω - mX))) B mX mY) P := by
  have hXR := regression_residual_jointGaussian hXY B mX mY
  have hzero := regression_residual_zero_covariance hXY B mX mY hnormal
  have hspec := gaussianRegressionKernel_spec hXR hzero B mX mY
  exact ⟨hspec.1, hspec.2.1, hspec.2.2.1,
    gaussianRegression_observed_hasCondDistrib hXY B mX mY hnormal⟩

end ResearchFormalCoreR1
