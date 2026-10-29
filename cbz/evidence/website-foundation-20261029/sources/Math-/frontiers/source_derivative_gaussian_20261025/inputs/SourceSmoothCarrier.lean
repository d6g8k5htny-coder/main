import SourceSmooth
import SourceGaussianLaw
import SourceCovariance
import SourceTorusC4
import Mathlib.MeasureTheory.Measure.Restrict
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Constructions.BorelSpace.Real

/-!
# A measurable smooth carrier for the actual source law

The carrier consists of the coefficient samples satisfying every explicit
order-weighted absolute summability condition from SourceSmooth. Its measure is
the pullback of the existing product Gaussian sourceLaw under literal subtype
inclusion. Inclusion pushes this measure back to exactly sourceLaw, so finite
coefficient observations, field evaluation laws, and covariance are unchanged.

The construction concerns the same literal Fourier source, for fixed d and L.
It does not identify the spectral kernel with a periodized theta kernel, define
a conditional kernel or a smooth-function-space law, or discharge a Lifetime
Field adapter, Morse, elder, counting, density, or universality conclusion.
-/

noncomputable section
open I4Foundation I4Source I4Smooth I4SourceGaussian I4SourceCovariance I4Kernel
open I4Torus I4TorusC4 MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal NNReal Manifold Topology

namespace I4SmoothCarrier

/-- The actual common coefficient event, with no smoothness premise added. -/
def goodSet (d : ℕ) (L : ℝ) : Set (SourceMode d → ℝ) :=
  {ω | goodSample d L ω}

private def orderAbsNN (n d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ)
    (m : SourceMode d) : ℝ≥0 :=
  ⟨orderModeBound n d L m * |ω m|,
    mul_nonneg (orderModeBound_nonneg n d L m) (abs_nonneg _)⟩

private def orderAbsTotal (n d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ) : ℝ≥0∞ :=
  ∑' m, (orderAbsNN n d L ω m : ℝ≥0∞)

private theorem orderAbsTotal_ne_top_iff (n d : ℕ) (L : ℝ)
    (ω : SourceMode d → ℝ) :
    orderAbsTotal n d L ω ≠ ∞ ↔
      Summable (fun m => orderModeBound n d L m * |ω m|) := by
  exact ENNReal.tsum_coe_ne_top_iff_summable_coe (f := orderAbsNN n d L ω)

private theorem orderAbsTotal_measurable (n d : ℕ) (L : ℝ) :
    Measurable (orderAbsTotal n d L) := by
  unfold orderAbsTotal
  simp_rw [ENNReal.tsum_eq_iSup_sum]
  apply Measurable.iSup
  intro s
  apply s.measurable_fun_sum
  intro m _
  have hm : Measurable (fun ω : SourceMode d → ℝ => orderAbsNN n d L ω m) := by
    unfold orderAbsNN
    exact (show Measurable (fun ω : SourceMode d → ℝ => orderModeBound n d L m * |ω m|) by
      fun_prop).nnreal_mk
  exact hm.coe_nnreal_ennreal

theorem goodSet_measurable (d : ℕ) (L : ℝ) : MeasurableSet (goodSet d L) := by
  have heq : goodSet d L =
      ⋂ n : ℕ, {ω : SourceMode d → ℝ | orderAbsTotal n d L ω ≠ ∞} := by
    ext ω
    simp only [goodSet, Set.mem_setOf_eq, Set.mem_iInter, goodSample]
    exact forall_congr' fun n => (orderAbsTotal_ne_top_iff n d L ω).symm
  rw [heq]
  apply MeasurableSet.iInter
  intro n
  convert ((orderAbsTotal_measurable n d L) (measurableSet_singleton ∞)).compl using 1
  ext ω
  simp

theorem goodSet_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d, ω ∈ goodSet d L :=
  goodSample_ae d hL

theorem goodSet_measure_one (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    sourceLaw d (goodSet d L) = 1 := by
  exact (mem_ae_iff_prob_eq_one (goodSet_measurable d L)).mp (goodSet_ae d hL)

/-- A coefficient sample on the explicit all-order summability carrier. -/
abbrev SmoothSample (d : ℕ) (L : ℝ) := ↥(goodSet d L)

/-- Literal inclusion, retaining every coefficient of the original sample. -/
def sampleInclusion (d : ℕ) (L : ℝ) : SmoothSample d L → (SourceMode d → ℝ) :=
  Subtype.val

theorem sampleInclusion_measurable (d : ℕ) (L : ℝ) :
    Measurable (sampleInclusion d L) :=
  measurable_subtype_coe

/-- Pull back the original product law; no normalization or new law is chosen. -/
def smoothLaw (d : ℕ) (L : ℝ) : Measure (SmoothSample d L) :=
  (sourceLaw d).comap (sampleInclusion d L)

theorem sampleInclusion_map (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    (smoothLaw d L).map (sampleInclusion d L) = sourceLaw d := by
  change ((sourceLaw d).comap ((↑) : SmoothSample d L → (SourceMode d → ℝ))).map
    ((↑) : SmoothSample d L → (SourceMode d → ℝ)) = sourceLaw d
  rw [map_comap_subtype_coe (goodSet_measurable d L),
    Measure.restrict_eq_self_of_ae_mem (goodSet_ae d hL)]

theorem sampleInclusion_measurePreserving (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    MeasurePreserving (sampleInclusion d L) (smoothLaw d L) (sourceLaw d) :=
  ⟨sampleInclusion_measurable d L, sampleInclusion_map d hL⟩

theorem smoothLaw_isProbabilityMeasure (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    IsProbabilityMeasure (smoothLaw d L) := by
  apply (MeasurableEmbedding.subtype_coe (goodSet_measurable d L)).isProbabilityMeasure_comap
  simpa only [Subtype.range_coe] using goodSet_ae d hL

/-- Every measurable observation has exactly its original pushforward law. -/
theorem observation_map_eq (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    {E : Type*} [MeasurableSpace E] (X : (SourceMode d → ℝ) → E) (hX : Measurable X) :
    (smoothLaw d L).map (fun ω => X (sampleInclusion d L ω)) = (sourceLaw d).map X := by
  change (smoothLaw d L).map (X ∘ sampleInclusion d L) = (sourceLaw d).map X
  rw [← Measure.map_map hX (sampleInclusion_measurable d L), sampleInclusion_map d hL]

theorem coordinate_marginal (d : ℕ) {L : ℝ} (hL : L ≠ 0) (m : SourceMode d) :
    (smoothLaw d L).map (fun ω => sampleInclusion d L ω m) = gaussianReal 0 1 := by
  rw [observation_map_eq d hL (fun ω => ω m) (measurable_pi_apply m)]
  exact I4Foundation.coordinate_marginal m

theorem finiteCoordinate_map_eq (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (n : ℕ) (m : Fin n → SourceMode d) :
    (smoothLaw d L).map (fun ω j => sampleInclusion d L ω (m j)) =
      (sourceLaw d).map (fun ω j => ω (m j)) := by
  exact observation_map_eq d hL _ (Measurable.of_eval (fun j => measurable_pi_apply (m j)))

/-- The unchanged literal real-cover field, evaluated on the carrier. -/
def smoothSourceField (d : ℕ) (L : ℝ) (ω : SmoothSample d L) (x : Fin d → ℝ) : ℝ :=
  sourceField d L (sampleInclusion d L ω) x

theorem smoothSourceField_contDiff (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample d L) (n : ℕ) : ContDiff ℝ n (smoothSourceField d L ω) :=
  sourceField_smooth_of_goodSample d hL (sampleInclusion d L ω) ω.property n

theorem smoothSourceField_smooth (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample d L) : ContDiff ℝ (⊤ : ℕ∞) (smoothSourceField d L ω) :=
  contDiff_infty.mpr (smoothSourceField_contDiff d hL ω)

theorem smoothSourceField_finiteVector_map_eq (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (n : ℕ) (x : Fin n → (Fin d → ℝ)) :
    (smoothLaw d L).map (fun ω j => smoothSourceField d L ω (x j)) =
      (sourceLaw d).map (fun ω j => sourceField d L ω (x j)) := by
  exact observation_map_eq d hL _
    (Measurable.of_eval (fun j => sourceField_eval_measurable d L (x j)))

theorem smoothSourceField_finiteVector_hasGaussianLaw (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (n : ℕ) (x : Fin n → (Fin d → ℝ)) :
    HasGaussianLaw (fun ω j => smoothSourceField d L ω (x j)) (smoothLaw d L) := by
  have hm : Measurable (fun ω j => smoothSourceField d L ω (x j)) :=
    (Measurable.of_eval (fun j => sourceField_eval_measurable d L (x j))).comp
      (sampleInclusion_measurable d L)
  refine ⟨hm.aemeasurable, ?_⟩
  rw [smoothSourceField_finiteVector_map_eq d hL n x]
  exact (source_finiteVector_hasGaussianLaw d hL n x).isGaussian_map

theorem covariance_comp_inclusion (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (X Y : (SourceMode d → ℝ) → ℝ) (hX : Measurable X) (hY : Measurable Y) :
    cov[fun ω => X (sampleInclusion d L ω), fun ω => Y (sampleInclusion d L ω); smoothLaw d L] =
      cov[X, Y; sourceLaw d] := by
  have h := covariance_map_fun (μ := smoothLaw d L) (Z := sampleInclusion d L)
    (X := X) (Y := Y) hX.aestronglyMeasurable hY.aestronglyMeasurable
    (sampleInclusion_measurable d L).aemeasurable
  rw [sampleInclusion_map d hL] at h
  exact h.symm

theorem smoothSourceField_covariance (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (x y : Fin d → ℝ) :
    cov[fun ω => smoothSourceField d L ω x, fun ω => smoothSourceField d L ω y; smoothLaw d L] =
      sourceSpectralKernel d L (x - y) := by
  calc
    _ = cov[fun ω => sourceField d L ω x, fun ω => sourceField d L ω y; sourceLaw d] :=
      covariance_comp_inclusion d hL _ _ (sourceField_eval_measurable d L x)
        (sourceField_eval_measurable d L y)
    _ = _ := sourceField_covariance d hL x y

/-- The unchanged literal torus descent of the same coefficient sample. -/
def smoothTorusSourceField (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (ω : SmoothSample d L) (q : SourceTorus d L) : ℝ :=
  torusSourceField d L (sampleInclusion d L ω) q

attribute [local instance] actualChartedSpace

theorem smoothTorusSourceField_smooth (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (ω : SmoothSample d L) :
    ContMDiff 𝓘(ℝ, Fin d → ℝ) 𝓘(ℝ) (⊤ : ℕ∞) (smoothTorusSourceField d L ω) := by
  let _ := torus_isManifold d L
  have hs := smoothSourceField_smooth d (Fact.out : 0 < L).ne' ω
  intro q
  rw [contMDiffAt_iff_source]
  have heq : smoothTorusSourceField d L ω ∘ (extChartAt 𝓘(ℝ, Fin d → ℝ) q).symm =
      smoothSourceField d L ω := by
    funext x
    change torusSourceField d L (sampleInclusion d L ω)
      ((quotientChart d L (chartCenter d L q)).symm x) = _
    rw [quotientChart_symm]
    exact torusSourceField_pullback d L (sampleInclusion d L ω) x
  rw [heq, contMDiffWithinAt_iff_contDiffWithinAt]
  exact hs.contDiffAt.contDiffWithinAt

theorem smoothTorusSourceField_finiteVector_hasGaussianLaw (d : ℕ) (L : ℝ)
    [Fact (0 < L)] (n : ℕ) (q : Fin n → SourceTorus d L) :
    HasGaussianLaw (fun ω j => smoothTorusSourceField d L ω (q j)) (smoothLaw d L) := by
  have hm : Measurable (fun ω j => smoothTorusSourceField d L ω (q j)) :=
    (Measurable.of_eval (fun j => torusSourceField_eval_measurable d L (q j))).comp
      (sampleInclusion_measurable d L)
  refine ⟨hm.aemeasurable, ?_⟩
  change IsGaussian ((smoothLaw d L).map
    (fun ω j => torusSourceField d L (sampleInclusion d L ω) (q j)))
  rw [observation_map_eq d (Fact.out : 0 < L).ne' _
    (Measurable.of_eval (fun j => torusSourceField_eval_measurable d L (q j)))]
  exact (torusSourceField_finiteVector_hasGaussianLaw d L n q).isGaussian_map

theorem smoothTorusSourceField_covariance (d : ℕ) (L : ℝ) [Fact (0 < L)]
    (q p : SourceTorus d L) :
    cov[fun ω => smoothTorusSourceField d L ω q, fun ω => smoothTorusSourceField d L ω p;
      smoothLaw d L] =
      sourceSpectralKernel d L (torusRepresentative d L q - torusRepresentative d L p) := by
  calc
    _ = cov[fun ω => torusSourceField d L ω q, fun ω => torusSourceField d L ω p; sourceLaw d] :=
      covariance_comp_inclusion d (Fact.out : 0 < L).ne' _ _
        (torusSourceField_eval_measurable d L q) (torusSourceField_eval_measurable d L p)
    _ = _ := torusSourceField_covariance d L q p

end I4SmoothCarrier
