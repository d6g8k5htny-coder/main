import SourceSmoothCarrier
import Mathlib.Analysis.Calculus.SmoothSeries
import Mathlib.Topology.Algebra.Module.Multilinear.Topology

/-! Actual derivative observations of the literal smooth Fourier source.
All derivative orders are differentiated termwise on the existing SmoothSample
carrier. Joint Gaussianity follows from summable deterministic coefficient arrays,
without an assumed derivative law, conditional law or covariance lower bound. -/

noncomputable section

open I4Foundation I4Source I4Smooth I4SourceGaussian I4SmoothCarrier
open MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal NNReal Topology

namespace I4SourceDerivativeGaussian

private theorem modeCoefficient_smooth (d : ℕ) (L : ℝ) (m : SourceMode d) :
    ContDiff ℝ (⊤ : ℕ∞) (modeCoefficient d L m) := by
  cases m with
  | none =>
    change ContDiff ℝ (⊤ : ℕ∞) (fun _ : Fin d → ℝ => Real.sqrt (1 / I4Weights.sourceZ d L))
    fun_prop
  | some p =>
    rcases p with ⟨k, b⟩
    change ContDiff ℝ (⊤ : ℕ∞) (fun x : Fin d → ℝ =>
      Real.sqrt (2 * I4Weights.sourceWeight d L k / I4Weights.sourceZ d L) *
        if b then Real.sin (phase d L k x) else Real.cos (phase d L k x))
    cases b <;> simp only [Bool.false_eq_true, ite_false, ite_true] <;>
      unfold phase <;> fun_prop

private theorem weighted_iteratedFDeriv (d : ℕ) (L : ℝ) (m : SourceMode d)
    (c : ℝ) (k : ℕ) (x : Fin d → ℝ) :
    iteratedFDeriv ℝ k (fun y => modeCoefficient d L m y * c) x =
      c • iteratedFDeriv ℝ k (modeCoefficient d L m) x := by
  have hf : ContDiff ℝ k (modeCoefficient d L m) :=
    (modeCoefficient_smooth d L m).of_le
      (by exact_mod_cast (show (k : ℕ∞) ≤ ⊤ from le_top))
  have heq : (fun y => modeCoefficient d L m y * c) = c • modeCoefficient d L m := by
    funext y
    simp only [Pi.smul_apply, smul_eq_mul, mul_comm]
  rw [heq, iteratedFDeriv_const_smul_apply hf.contDiffAt]

private theorem weighted_iteratedFDeriv_bound (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (m : SourceMode d) (c : ℝ) (k : ℕ) (x : Fin d → ℝ) :
    ‖iteratedFDeriv ℝ k (fun y => modeCoefficient d L m y * c) x‖ ≤
      orderModeBound k d L m * |c| := by
  rw [weighted_iteratedFDeriv, norm_smul, Real.norm_eq_abs]
  simpa only [mul_comm] using mul_le_mul_of_nonneg_left
    (modeCoefficient_iteratedFDeriv_order_bound d hL m k x) (abs_nonneg c)

/-- Deterministic Fourier coefficient for an actual directional derivative. -/
def derivativeCoefficient (d : ℕ) (L : ℝ) (k : ℕ) (x : Fin d → ℝ)
    (u : Fin k → (Fin d → ℝ)) (m : SourceMode d) : ℝ :=
  iteratedFDeriv ℝ k (modeCoefficient d L m) x u

theorem derivativeCoefficient_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (k : ℕ) (x : Fin d → ℝ) (u : Fin k → (Fin d → ℝ)) :
    Summable (derivativeCoefficient d L k x u) := by
  apply ((orderModeBound_summable k d hL).mul_right (∏ i, ‖u i‖)).of_norm_bounded
  intro m
  calc
    ‖derivativeCoefficient d L k x u m‖ ≤
        ‖iteratedFDeriv ℝ k (modeCoefficient d L m) x‖ * ∏ i, ‖u i‖ :=
      (iteratedFDeriv ℝ k (modeCoefficient d L m) x).le_opNorm u
    _ ≤ orderModeBound k d L m * ∏ i, ‖u i‖ :=
      mul_le_mul_of_nonneg_right (modeCoefficient_iteratedFDeriv_order_bound d hL m k x)
        (Finset.prod_nonneg (fun _ _ => norm_nonneg _))

/-- This is a literal identity for every sample in the existing smooth carrier,
including order zero. It is not a finite-mode approximation or a new field. -/
theorem smoothSourceField_derivative_series (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample d L) (k : ℕ) (x : Fin d → ℝ)
    (u : Fin k → (Fin d → ℝ)) :
    iteratedFDeriv ℝ k (smoothSourceField d L ω) x u =
      scalarSeries (derivativeCoefficient d L k x u) (sampleInclusion d L ω) := by
  let f : SourceMode d → (Fin d → ℝ) → ℝ :=
    fun m y => modeCoefficient d L m y * sampleInclusion d L ω m
  let major : ℕ → SourceMode d → ℝ :=
    fun j m => orderModeBound j d L m * |sampleInclusion d L ω m|
  have hf (m : SourceMode d) : ContDiff ℝ (k : ℕ∞) (f m) :=
    ((modeCoefficient_smooth d L m).of_le
      (by exact_mod_cast (show (k : ℕ∞) ≤ ⊤ from le_top))).mul contDiff_const
  have hmajor (j : ℕ) (_hj : (j : ℕ∞) ≤ k) : Summable (major j) :=
    ω.property j
  have hbound (j : ℕ) (m : SourceMode d) (y : Fin d → ℝ)
      (_hj : (j : ℕ∞) ≤ k) : ‖iteratedFDeriv ℝ j (f m) y‖ ≤ major j m :=
    weighted_iteratedFDeriv_bound d hL m (sampleInclusion d L ω m) j y
  have hsum : Summable (fun m => iteratedFDeriv ℝ k (f m) x) :=
    (hmajor k le_rfl).of_norm_bounded (fun m => hbound k m x le_rfl)
  have hdiff := iteratedFDeriv_tsum_apply hf hmajor hbound (k := k) le_rfl x
  change iteratedFDeriv ℝ k (fun y => ∑' m, f m y) x u =
    ∑' m, derivativeCoefficient d L k x u m * sampleInclusion d L ω m
  rw [hdiff, ContinuousMultilinearMap.tsum_eval hsum]
  apply tsum_congr
  intro m
  dsimp only [f]
  rw [weighted_iteratedFDeriv]
  exact mul_comm _ _

private theorem scalarSeries_linearCombination_hasGaussianLaw
    {ι : Type*} [Countable ι] (n : ℕ) (a : Fin n → ι → ℝ)
    (ha : ∀ j, Summable (a j)) (c : Fin n → ℝ) :
    HasGaussianLaw (fun ω => ∑ j, c j * scalarSeries (a j) ω) (coefficientLaw ι) := by
  classical
  let b : ι → ℝ := fun m => ∑ j, c j * a j m
  have hb : Summable b := summable_sum (fun j _ => (ha j).mul_left (c j))
  have hG : HasGaussianLaw (scalarSeries b) (coefficientLaw ι) := by
    constructor
    · exact (scalarSeries_measurable b).aemeasurable
    · rw [scalarSeries_map_gaussian b hb]
      infer_instance
  apply hG.congr
  filter_upwards [ae_all_iff.2 (fun j : Fin n => scalarSeries_hasSum_ae (a j) (ha j)),
    scalarSeries_hasSum_ae b hb] with ω hω hsum
  have heq : HasSum (fun m => ∑ j, c j * a j m * ω m)
      (∑ j, c j * scalarSeries (a j) ω) := by
    simpa only [mul_assoc] using hasSum_sum (fun j _ => (hω j).mul_left (c j))
  exact hsum.unique (by simpa only [b, Finset.sum_mul] using heq)

/-- Arbitrary finite arrays of summable coefficient series are jointly Gaussian.
No independence of the resulting observations is asserted or assumed. -/
theorem scalarSeries_finiteVector_hasGaussianLaw
    {ι : Type*} [Countable ι] (n : ℕ) (a : Fin n → ι → ℝ)
    (ha : ∀ j, Summable (a j)) :
    HasGaussianLaw (fun ω j => scalarSeries (a j) ω) (coefficientLaw ι) := by
  classical
  have hm : Measurable (fun ω j => scalarSeries (a j) ω) :=
    Measurable.of_eval (fun j => scalarSeries_measurable (a j))
  refine ⟨hm.aemeasurable, ?_⟩
  apply isGaussian_of_map_eq_gaussianReal
  intro ell
  let c : Fin n → ℝ := fun j => ell (Pi.single j 1)
  have hrep (z : Fin n → ℝ) : ell z = ∑ j, c j * z j := by
    rw [← LinearMap.sum_single_apply (fun _ : Fin n => ℝ) z, map_sum]
    apply Finset.sum_congr rfl
    intro j _
    have hs : Pi.single j (z j) = z j • Pi.single j (1 : ℝ) := by
      rw [← Pi.single_smul, smul_eq_mul, mul_one]
    rw [hs, map_smul]
    simp [c, smul_eq_mul, mul_comm]
  have hG : HasGaussianLaw (fun ω => ell (fun j => scalarSeries (a j) ω))
      (coefficientLaw ι) :=
    (scalarSeries_linearCombination_hasGaussianLaw n a ha c).congr
      (Filter.Eventually.of_forall (fun ω => (hrep _).symm))
  refine ⟨(coefficientLaw ι)[fun ω => ell (fun j => scalarSeries (a j) ω)],
    Var[fun ω => ell (fun j => scalarSeries (a j) ω); coefficientLaw ι].toNNReal, ?_⟩
  rw [Measure.map_map ell.continuous.measurable hm]
  exact hG.map_eq_gaussianReal

/-- Heterogeneous derivative orders, points and directions of the actual smooth
field are jointly Gaussian under its unchanged literal law. Empty families,
order zero and repeated observations are retained. -/
theorem smoothSourceField_derivativeVector_hasGaussianLaw (d : ℕ) {L : ℝ}
    (hL : L ≠ 0) (n : ℕ) (k : Fin n → ℕ) (x : Fin n → (Fin d → ℝ))
    (u : ∀ j, Fin (k j) → (Fin d → ℝ)) :
    HasGaussianLaw (fun ω j =>
      iteratedFDeriv ℝ (k j) (smoothSourceField d L ω) (x j) (u j)) (smoothLaw d L) := by
  let a : Fin n → SourceMode d → ℝ := fun j => derivativeCoefficient d L (k j) (x j) (u j)
  have ha (j : Fin n) : Summable (a j) :=
    derivativeCoefficient_summable d hL (k j) (x j) (u j)
  have hm : Measurable (fun η j => scalarSeries (a j) η) :=
    Measurable.of_eval (fun j => scalarSeries_measurable (a j))
  have hG : HasGaussianLaw
      (fun ω j => scalarSeries (a j) (sampleInclusion d L ω)) (smoothLaw d L) := by
    refine ⟨(hm.comp (sampleInclusion_measurable d L)).aemeasurable, ?_⟩
    rw [observation_map_eq d hL _ hm]
    exact (scalarSeries_finiteVector_hasGaussianLaw n a ha).isGaussian_map
  apply hG.congr
  filter_upwards with ω
  funext j
  exact (smoothSourceField_derivative_series d hL ω (k j) (x j) (u j)).symm

/-- Actual derivative covariance as a convergent Fourier coefficient pairing.
This does not yet identify the pairing with derivatives of the theta kernel. -/
theorem smoothSourceField_derivative_covariance (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (k l : ℕ) (x y : Fin d → ℝ) (u : Fin k → (Fin d → ℝ))
    (v : Fin l → (Fin d → ℝ)) :
    cov[fun ω => iteratedFDeriv ℝ k (smoothSourceField d L ω) x u,
      fun ω => iteratedFDeriv ℝ l (smoothSourceField d L ω) y v; smoothLaw d L] =
      ∑' m, derivativeCoefficient d L k x u m * derivativeCoefficient d L l y v m := by
  simp_rw [smoothSourceField_derivative_series d hL]
  rw [covariance_comp_inclusion d hL _ _ (scalarSeries_measurable _)
    (scalarSeries_measurable _)]
  exact I4SourceCovariance.scalarSeries_covariance _ _
    (derivativeCoefficient_summable d hL k x u) (derivativeCoefficient_summable d hL l y v)

end I4SourceDerivativeGaussian
