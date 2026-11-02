import TwoPinRank
import PinKernel
import Mathlib.Topology.Instances.Matrix

noncomputable section
open I4Foundation I4Weights I4Source I4C4 I4Jets I4SourceGaussian I4SourceCovariance
  I4PinMajorant I4PinKernel I4TwoPinRank I4Torus MeasureTheory ProbabilityTheory
open scoped BigOperators Matrix
namespace I4PinReproduction

def coefficientGram (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) :
    Matrix (PinIndex d) (PinIndex d) ℝ :=
  fun i j => ∑' m, pinCoefficient d L sites m i * pinCoefficient d L sites m j
def canonicalR (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) :
    Matrix (PinIndex d) (PinIndex d) ℝ := (pinCovariance d L sites)⁻¹

private theorem row_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (i : PinIndex d) :
    Summable (fun m => pinCoefficient d L sites m i) :=
  modeJetCoefficient_summable d hL (pinOrder d i)
    (by rcases i with ⟨a,j⟩; cases j <;> norm_num [pinOrder]) _ _

private theorem row_product_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (i j : PinIndex d) :
    Summable (fun m => pinCoefficient d L sites m i * pinCoefficient d L sites m j) := by
  simpa only [Function.comp_def] using
    (summable_mul_of_summable_norm (row_summable d hL sites i).norm
      (row_summable d hL sites j).norm).comp_injective
      (show Function.Injective (fun m : SourceMode d => (m,m)) from
        fun _ _ h => congrArg Prod.fst h)

private theorem cov_congr {Omega : Type*} [MeasurableSpace Omega] {mu : Measure Omega}
    {f g f' g' : Omega → ℝ} (hf : f =ᵐ[mu] f') (hg : g =ᵐ[mu] g') :
    cov[f,g;mu] = cov[f',g';mu] := by
  unfold covariance
  rw [integral_congr_ae hf,integral_congr_ae hg]
  apply integral_congr_ae
  filter_upwards [hf,hg] with omega hfo hgo
  rw [hfo,hgo]

private theorem modeCoefficient_contDiff (d : ℕ) (L : ℝ) (m : SourceMode d) :
    ContDiff ℝ 4 (modeCoefficient d L m) := by
  cases m with
  | none =>
    change ContDiff ℝ 4 (fun _ : Fin d → ℝ => Real.sqrt (1 / sourceZ d L))
    fun_prop
  | some p =>
    rcases p with ⟨k, b⟩
    change ContDiff ℝ 4 (fun x : Fin d → ℝ =>
      Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
        (if b then Real.sin (phase d L k x) else Real.cos (phase d L k x)))
    cases b <;> simp [phase] <;> fun_prop

private theorem modeTerm_derivative (d : ℕ) (L : ℝ) (m : SourceMode d)
    (ω : SourceMode d → ℝ) (j : ℕ) (hj : j ≤ 4) (x : Fin d → ℝ) :
    iteratedFDeriv ℝ j (fun y => modeCoefficient d L m y * ω m) x =
      ω m • iteratedFDeriv ℝ j (modeCoefficient d L m) x := by
  have hc : ContDiff ℝ j (modeCoefficient d L m) :=
    (modeCoefficient_contDiff d L m).of_le (by exact_mod_cast hj)
  have heq : (fun y => modeCoefficient d L m y * ω m) = ω m • modeCoefficient d L m := by
    funext y
    simp [smul_eq_mul, mul_comm]
  rw [heq, iteratedFDeriv_const_smul_apply hc.contDiffAt]

private theorem deterministic_jet (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SourceMode d → ℝ) (hω : Summable (fun m => modeMajorant d L m * |ω m|))
    (j : ℕ) (hj : j ≤ 4) (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) :
    sourceJet d L j ω x v = scalarSeries (fun m => modeJetCoefficient d L m j x v) ω := by
  have hc (m : SourceMode d) :
      ContDiff ℝ 4 (fun y => modeCoefficient d L m y * ω m) :=
    (modeCoefficient_contDiff d L m).mul contDiff_const
  have hb (k : ℕ) (hk : (k : ℕ∞) ≤ 4) :
      Summable (fun m => modeMajorant d L m * |ω m|) := hω
  have hd (k : ℕ) (m : SourceMode d) (y : Fin d → ℝ) (hk : (k : ℕ∞) ≤ 4) :
      ‖iteratedFDeriv ℝ k (fun z => modeCoefficient d L m z * ω m) y‖ ≤
        modeMajorant d L m * |ω m| := by
    have hk' : k ≤ 4 := by exact_mod_cast hk
    rw [modeTerm_derivative d L m ω k hk' y, norm_smul, Real.norm_eq_abs]
    simpa only [mul_comm] using mul_le_mul_of_nonneg_left
      (modeCoefficient_iteratedFDeriv_bound d hL m k hk' y) (abs_nonneg (ω m))
  have hj' : (j : ℕ∞) ≤ 4 := by exact_mod_cast hj
  have hs : Summable (fun m =>
      iteratedFDeriv ℝ j (fun y => modeCoefficient d L m y * ω m) x) :=
    hω.of_norm_bounded (fun m => hd j m x hj')
  have heq : iteratedFDeriv ℝ j (sourceField d L ω) x =
      ∑' m, iteratedFDeriv ℝ j (fun y => modeCoefficient d L m y * ω m) x := by
    change iteratedFDeriv ℝ j (fun y => ∑' m, modeCoefficient d L m y * ω m) x = _
    exact iteratedFDeriv_tsum_apply hc hb hd hj' x
  unfold sourceJet
  rw [heq, ContinuousMultilinearMap.tsum_eval hs]
  unfold scalarSeries
  apply tsum_congr
  intro m
  rw [modeTerm_derivative d L m ω j hj x]
  change ω m * modeJetCoefficient d L m j x v =
    modeJetCoefficient d L m j x v * ω m
  exact mul_comm _ _

private theorem pin_eq_actual_majorant (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (omega : SourceMode d → ℝ)
    (homega : Summable (fun m => modeMajorant d L m * |omega m|)) :
    pinObservation d L sites omega = actualPinObservation d L sites omega := by
  funext i
  rcases i with ⟨a,j⟩
  cases j with
  | none =>
    have h := deterministic_jet d hL omega homega 0 (by omega) (sites a) (fun k => Fin.elim0 k)
    simpa [pinObservation,pinCoefficient,pinOrder,pinDirections,actualPinObservation,
      sourceJet_zero] using h.symm
  | some j =>
    have h := deterministic_jet d hL omega homega 1 (by omega) (sites a)
      (fun _ => (Pi.single j 1 : Fin d → ℝ))
    simpa [pinObservation,pinCoefficient,pinOrder,pinDirections,actualPinObservation,
      sourceJet,iteratedFDeriv_one_apply] using h.symm

theorem pinCovariance_eq_gram (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) :
    pinCovariance d L sites = coefficientGram d L sites := by
  ext i j
  have hi : (fun omega => actualPinObservation d L sites omega i) =ᵐ[sourceLaw d]
      scalarSeries (fun m => pinCoefficient d L sites m i) :=
    (pinObservation_eq_actual_ae d hL).mono (fun _ h => (congrFun (h sites) i).symm)
  have hj : (fun omega => actualPinObservation d L sites omega j) =ᵐ[sourceLaw d]
      scalarSeries (fun m => pinCoefficient d L sites m j) :=
    (pinObservation_eq_actual_ae d hL).mono (fun _ h => (congrFun (h sites) j).symm)
  exact (cov_congr hi hj).trans (scalarSeries_covariance _ _
    (row_summable d hL sites i) (row_summable d hL sites j))

theorem canonicalR_inverse (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ))
    (hxy : torusProjection d L (sites 0) ≠ torusProjection d L (sites 1)) :
    pinCovariance d L sites * canonicalR d L sites = 1 ∧
      canonicalR d L sites * pinCovariance d L sites = 1 := by
  have hdet : IsUnit (pinCovariance d L sites).det :=
    (Matrix.isUnit_iff_isUnit_det _).mp (pinCovariance_posDef d hL sites hxy).isUnit
  exact ⟨Matrix.mul_nonsing_inv _ hdet,Matrix.nonsing_inv_mul _ hdet⟩

theorem canonicalR_measurable (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    Measurable (canonicalR d L) := by
  classical
  have hg : Measurable (coefficientGram d L) := by
    apply Measurable.of_eval_matrix
    intro i j
    apply Measurable.tsum
    intro m
    exact ((measurable_pi_apply i).comp (pinCoefficient_joint_measurable d L m)).mul
      ((measurable_pi_apply j).comp (pinCoefficient_joint_measurable d L m))
  have hc : Measurable (pinCovariance d L) := by
    convert hg using 1
    funext sites
    exact pinCovariance_eq_gram d hL sites
  have hi : Measurable (fun A : Matrix (PinIndex d) (PinIndex d) ℝ => A⁻¹) := by
    simp_rw [Matrix.inv_def,Ring.inverse_eq_inv]
    exact (continuous_id.matrix_det.measurable.inv).smul
      continuous_id.matrix_adjugate.measurable
  exact hi.comp hc

private theorem transformed_pin_scalar (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (R : Matrix (PinIndex d) (PinIndex d) ℝ)
    (v : PinIndex d → ℝ) (omega : SourceMode d → ℝ)
    (hs : ∀ i, HasSum (fun m => pinCoefficient d L sites m i * omega m)
      (pinObservation d L sites omega i)) (r : PinIndex d) :
    pinObservation d L sites (regressionCoefficients d L sites R v omega) r =
      pinObservation d L sites omega r -
        (∑ i, ∑ j, coefficientGram d L sites r i * R i j * pinObservation d L sites omega j) +
        (∑ i, ∑ j, coefficientGram d L sites r i * R i j * v j) := by
  classical
  have hc : HasSum (fun m => ∑ i, ∑ j,
      pinCoefficient d L sites m r * pinCoefficient d L sites m i * R i j *
        pinObservation d L sites omega j)
      (∑ i, ∑ j, coefficientGram d L sites r i * R i j * pinObservation d L sites omega j) :=
    hasSum_sum (s := Finset.univ) (fun i _ => hasSum_sum (s := Finset.univ) (fun j _ =>
      ((row_product_summable d hL sites r i).hasSum.mul_right (R i j)).mul_right _))
  have ht : HasSum (fun m => ∑ i, ∑ j,
      pinCoefficient d L sites m r * pinCoefficient d L sites m i * R i j * v j)
      (∑ i, ∑ j, coefficientGram d L sites r i * R i j * v j) :=
    hasSum_sum (s := Finset.univ) (fun i _ => hasSum_sum (s := Finset.univ) (fun j _ =>
      ((row_product_summable d hL sites r i).hasSum.mul_right (R i j)).mul_right _))
  have hh := ((hs r).sub hc).add ht
  have he : HasSum (fun m => pinCoefficient d L sites m r *
      regressionCoefficients d L sites R v omega m)
      (pinObservation d L sites omega r -
        (∑ i, ∑ j, coefficientGram d L sites r i * R i j * pinObservation d L sites omega j) +
        (∑ i, ∑ j, coefficientGram d L sites r i * R i j * v j)) := by
    convert hh using 1
    funext m
    simp only [regressionCoefficients,mul_add,mul_sub,Finset.mul_sum,mul_assoc]
  exact he.tsum_eq

theorem canonical_pin_reproduction (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ))
    (hxy : torusProjection d L (sites 0) ≠ torusProjection d L (sites 1))
    (v : PinIndex d → ℝ) :
    ∀ᵐ eta ∂transformedLaw d L sites (canonicalR d L sites) v,
      actualPinObservation d L sites eta = v := by
  classical
  let : Fact (0 < L) := ⟨hL⟩
  have hGR : coefficientGram d L sites * canonicalR d L sites = 1 := by
    rw [← pinCovariance_eq_gram d hL.ne' sites]
    exact (canonicalR_inverse d hL sites hxy).1
  have hsum (r : PinIndex d) (q : PinIndex d → ℝ) :
      (∑ i, ∑ j, coefficientGram d L sites r i * canonicalR d L sites i j * q j) = q r := by
    have he := congrFun (congrArg (fun A : Matrix (PinIndex d) (PinIndex d) ℝ => A *ᵥ q) hGR) r
    rw [Matrix.one_mulVec] at he
    simp only [Matrix.mulVec,dotProduct,Matrix.mul_apply,Finset.sum_mul] at he
    rw [Finset.sum_comm] at he
    exact he
  have hscalar : ∀ᵐ eta ∂transformedLaw d L sites (canonicalR d L sites) v,
      pinObservation d L sites eta = v := by
    change ∀ᵐ eta ∂(sourceLaw d).map (regressionCoefficients d L sites (canonicalR d L sites) v), _
    apply (ae_map_iff (regressionCoefficients_measurable d L sites (canonicalR d L sites) v).aemeasurable
      (by
        have hm : Measurable (pinObservation d L sites) := by
          apply Measurable.of_eval
          intro i
          exact scalarSeries_measurable _
        exact measurableSet_eq_fun hm measurable_const)).2
    filter_upwards [ae_all_iff.2 (fun i : PinIndex d =>
      scalarSeries_hasSum_ae (fun m => pinCoefficient d L sites m i) (row_summable d hL.ne' sites i))]
      with omega hs
    funext r
    rw [transformed_pin_scalar d hL.ne' sites (canonicalR d L sites) v omega hs r,
      hsum,hsum]
    ring
  filter_upwards [hscalar,transformedLaw_regular d L sites (canonicalR d L sites) v] with eta heta hreg
  rw [← pin_eq_actual_majorant d hL.ne' sites eta hreg.1]
  exact heta

end I4PinReproduction

