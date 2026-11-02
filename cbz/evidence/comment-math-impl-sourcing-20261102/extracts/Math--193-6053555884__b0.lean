import SourceJets
import SourceGaussianLaw
import SourceCovariance

/-! Gaussian laws and coefficient Gram covariance of the actual sourceField derivatives.
Mixed finite lists may be singular. No positive-rank, prescribed-pin or torus-chart claim. -/

noncomputable section
open I4Foundation I4Weights I4Source I4Jets I4SourceGaussian I4SourceCovariance
  MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal NNReal

namespace I4JetLaw

private theorem jet_eq_series_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (j : ℕ) (hj : j ≤ 4) (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) :
    (fun ω => sourceJet d L j ω x v) =ᵐ[sourceLaw d]
      scalarSeries (fun m => modeJetCoefficient d L m j x v) :=
  (sourceJet_eq_series_ae d hL).mono (fun _ω hω => hω j hj x v)

private theorem jet_aemeasurable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (j : ℕ) (hj : j ≤ 4) (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) :
    AEMeasurable (fun ω => sourceJet d L j ω x v) (sourceLaw d) :=
  (scalarSeries_measurable (fun m => modeJetCoefficient d L m j x v)).aemeasurable.congr
    (jet_eq_series_ae d hL j hj x v).symm

theorem sourceJet_map_gaussian (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (j : ℕ) (hj : j ≤ 4) (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) :
    (sourceLaw d).map (fun ω => sourceJet d L j ω x v) =
      gaussianReal 0 (∑' m, (modeJetCoefficient d L m j x v)^2).toNNReal := by
  rw [Measure.map_congr (jet_eq_series_ae d hL j hj x v)]
  exact scalarSeries_map_gaussian _ (modeJetCoefficient_summable d hL j hj x v)

theorem sourceJet_linearCombination_hasGaussianLaw (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (n : ℕ) (r : Fin n → ℕ) (hr : ∀ i, r i ≤ 4)
    (x : Fin n → (Fin d → ℝ)) (v : (i : Fin n) → Fin (r i) → (Fin d → ℝ))
    (c : Fin n → ℝ) :
    HasGaussianLaw (fun ω => ∑ i, c i * sourceJet d L (r i) ω (x i) (v i))
      (sourceLaw d) := by
  classical
  let a : SourceMode d → ℝ := fun m => ∑ i, c i * modeJetCoefficient d L m (r i) (x i) (v i)
  have hc (i : Fin n) : Summable (fun m => modeJetCoefficient d L m (r i) (x i) (v i)) :=
    modeJetCoefficient_summable d hL (r i) (hr i) (x i) (v i)
  have ha : Summable a := summable_sum (fun i _ => (hc i).mul_left (c i))
  have hG : HasGaussianLaw (scalarSeries a) (sourceLaw d) := by
    constructor
    · exact (scalarSeries_measurable a).aemeasurable
    · change IsGaussian ((coefficientLaw (SourceMode d)).map (scalarSeries a))
      rw [scalarSeries_map_gaussian a ha]
      infer_instance
  apply hG.congr
  filter_upwards [ae_all_iff.2 (fun i : Fin n =>
      scalarSeries_hasSum_ae (fun m => modeJetCoefficient d L m (r i) (x i) (v i)) (hc i)),
    scalarSeries_hasSum_ae a ha, sourceJet_eq_series_ae d hL] with ω hω hsum hjet
  have h : HasSum (fun m => ∑ i, c i * modeJetCoefficient d L m (r i) (x i) (v i) * ω m)
      (∑ i, c i * scalarSeries (fun m => modeJetCoefficient d L m (r i) (x i) (v i)) ω) := by
    simpa only [mul_assoc] using hasSum_sum (fun i _ => (hω i).mul_left (c i))
  have hv : (∑ i, c i * sourceJet d L (r i) ω (x i) (v i)) =
      ∑ i, c i * scalarSeries (fun m => modeJetCoefficient d L m (r i) (x i) (v i)) ω := by
    exact Finset.sum_congr rfl (fun i _ => congrArg (c i * ·) (hjet (r i) (hr i) (x i) (v i)))
  rw [← hv] at h
  exact hsum.unique (by simpa only [a, Finset.sum_mul] using h)

theorem sourceJet_finiteVector_hasGaussianLaw (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (n : ℕ) (r : Fin n → ℕ) (hr : ∀ i, r i ≤ 4)
    (x : Fin n → (Fin d → ℝ)) (v : (i : Fin n) → Fin (r i) → (Fin d → ℝ)) :
    HasGaussianLaw (fun ω i => sourceJet d L (r i) ω (x i) (v i))
      (sourceLaw d) := by
  classical
  have hm : AEMeasurable (fun ω i => sourceJet d L (r i) ω (x i) (v i)) (sourceLaw d) :=
    AEMeasurable.of_eval (fun i => jet_aemeasurable d hL (r i) (hr i) (x i) (v i))
  refine ⟨hm, ?_⟩
  apply isGaussian_of_map_eq_gaussianReal
  intro ell
  let c : Fin n → ℝ := fun i => ell (Pi.single i 1)
  have hrep (u : Fin n → ℝ) : ell u = ∑ i, c i * u i := by
    rw [← LinearMap.sum_single_apply (fun _ : Fin n => ℝ) u, map_sum]
    apply Finset.sum_congr rfl
    intro i _
    have hs : Pi.single i (u i) = u i • Pi.single i (1 : ℝ) := by
      rw [← Pi.single_smul, smul_eq_mul, mul_one]
    rw [hs, map_smul]
    simp [c, smul_eq_mul, mul_comm]
  have hg : HasGaussianLaw (fun ω => ell (fun i => sourceJet d L (r i) ω (x i) (v i)))
      (sourceLaw d) :=
    (sourceJet_linearCombination_hasGaussianLaw d hL n r hr x v c).congr
      (Filter.Eventually.of_forall (fun ω => (hrep _).symm))
  refine ⟨(sourceLaw d)[fun ω => ell (fun i => sourceJet d L (r i) ω (x i) (v i))],
    Var[fun ω => ell (fun i => sourceJet d L (r i) ω (x i) (v i)); sourceLaw d].toNNReal, ?_⟩
  rw [AEMeasurable.map_map_of_aemeasurable ell.continuous.measurable.aemeasurable hm]
  exact hg.map_eq_gaussianReal

theorem sourceJet_integral_zero (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (j : ℕ) (hj : j ≤ 4) (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) :
    ∫ ω, sourceJet d L j ω x v ∂sourceLaw d = 0 := by
  rw [integral_congr_ae (jet_eq_series_ae d hL j hj x v)]
  exact scalarSeries_integral_zero _ (modeJetCoefficient_summable d hL j hj x v)

theorem sourceJet_covariance (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (j k : ℕ) (hj : j ≤ 4) (hk : k ≤ 4) (x y : Fin d → ℝ)
    (v : Fin j → (Fin d → ℝ)) (w : Fin k → (Fin d → ℝ)) :
    cov[fun ω => sourceJet d L j ω x v,
      fun ω => sourceJet d L k ω y w; sourceLaw d] =
      ∑' m, modeJetCoefficient d L m j x v * modeJetCoefficient d L m k y w := by
  have h1 := jet_eq_series_ae d hL j hj x v
  have h2 := jet_eq_series_ae d hL k hk y w
  have hc : cov[fun ω => sourceJet d L j ω x v,
      fun ω => sourceJet d L k ω y w; sourceLaw d] =
      cov[scalarSeries (fun m => modeJetCoefficient d L m j x v),
        scalarSeries (fun m => modeJetCoefficient d L m k y w); sourceLaw d] := by
    unfold covariance
    rw [integral_congr_ae h1, integral_congr_ae h2]
    apply integral_congr_ae
    filter_upwards [h1, h2] with ω ha hb
    rw [ha, hb]
  rw [hc]
  exact scalarSeries_covariance _ _ (modeJetCoefficient_summable d hL j hj x v)
    (modeJetCoefficient_summable d hL k hk y w)

theorem sourceJet_variance (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (j : ℕ) (hj : j ≤ 4) (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) :
    Var[fun ω => sourceJet d L j ω x v; sourceLaw d] =
      ∑' m, (modeJetCoefficient d L m j x v)^2 := by
  rw [variance_congr (jet_eq_series_ae d hL j hj x v)]
  exact scalarSeries_variance _ (modeJetCoefficient_summable d hL j hj x v)

end I4JetLaw

