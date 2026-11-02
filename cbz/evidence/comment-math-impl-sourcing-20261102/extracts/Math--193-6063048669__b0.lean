import PinReproduction
import Mathlib.Probability.Independence.Process.Basic
import Mathlib.Probability.Kernel.Disintegration.Unique

noncomputable section
open I4Foundation I4Weights I4Source I4Jets I4SourceGaussian I4SourceCovariance
  I4PinMajorant I4PinKernel I4TwoPinRank I4PinReproduction I4Torus
  MeasureTheory ProbabilityTheory
open scoped BigOperators ProbabilityTheory
namespace I4PinDisintegration

abbrev Coefficients (d : ℕ) := SourceMode d → ℝ
def residual (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) :
    Coefficients d → Coefficients d :=
  regressionCoefficients d L sites (canonicalR d L sites) 0
def shift (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (v : PinIndex d → ℝ) (z : Coefficients d) (m : SourceMode d) : ℝ :=
  z m + ∑ i, ∑ j, pinCoefficient d L sites m i * canonicalR d L sites i j * v j
def canonicalKernel (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) :
    Kernel (PinIndex d → ℝ) (Coefficients d) :=
  (regressionKernel d L).comap (fun v => (sites,canonicalR d L sites,v)) (by fun_prop)
def pinJoint (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) :
    Measure ((PinIndex d → ℝ) × Coefficients d) :=
  (sourceLaw d).map (fun omega => (pinObservation d L sites omega,omega))

instance canonicalKernel_markov (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) :
    IsMarkovKernel (canonicalKernel d L sites) := by
  have := regressionKernel_markov d L
  unfold canonicalKernel
  infer_instance
instance pinJoint_probability (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) :
    IsProbabilityMeasure (pinJoint d L sites) := by
  unfold pinJoint
  change IsProbabilityMeasure ((coefficientLaw (SourceMode d)).map _)
  infer_instance

private theorem row_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (i : PinIndex d) :
    Summable (fun m => pinCoefficient d L sites m i) :=
  modeJetCoefficient_summable d hL (pinOrder d i)
    (by rcases i with ⟨a,j⟩; cases j <;> norm_num [pinOrder]) _ _

private theorem series_gaussian {ι : Type*} [Countable ι] (a : ι → ℝ) (ha : Summable a) :
    HasGaussianLaw (scalarSeries a) (coefficientLaw ι) := by
  constructor
  · exact (scalarSeries_measurable a).aemeasurable
  · rw [scalarSeries_map_gaussian a ha]
    infer_instance

private theorem series_family_gaussian {ι J : Type*} [Countable ι] [Fintype J]
    (a : J → ι → ℝ) (ha : ∀ j, Summable (a j)) :
    HasGaussianLaw (fun omega j => scalarSeries (a j) omega) (coefficientLaw ι) := by
  classical
  have hm : Measurable (fun omega j => scalarSeries (a j) omega) :=
    Measurable.of_eval (fun j => scalarSeries_measurable _)
  refine ⟨hm.aemeasurable, ?_⟩
  apply isGaussian_of_map_eq_gaussianReal
  intro ell
  let c : J → ℝ := fun j => ell (Pi.single j 1)
  let b : ι → ℝ := fun m => ∑ j, c j * a j m
  have hb : Summable b := summable_sum (fun j _ => (ha j).mul_left (c j))
  have hrep (v : J → ℝ) : ell v = ∑ j, c j * v j := by
    rw [← LinearMap.sum_single_apply (fun _ : J => ℝ) v, map_sum]
    apply Finset.sum_congr rfl
    intro j _
    have hs : Pi.single j (v j) = v j • Pi.single j (1 : ℝ) := by
      rw [← Pi.single_smul, smul_eq_mul, mul_one]
    rw [hs,map_smul]
    simp [c,smul_eq_mul,mul_comm]
  have heq : scalarSeries b =ᵐ[coefficientLaw ι]
      (fun omega => ell (fun j => scalarSeries (a j) omega)) := by
    filter_upwards [ae_all_iff.2 (fun j => scalarSeries_hasSum_ae (a j) (ha j)),
      scalarSeries_hasSum_ae b hb] with omega hs hbomega
    rw [hrep]
    apply hbomega.unique
    simpa only [b,Finset.sum_mul,mul_assoc] using
      hasSum_sum (fun j _ => (hs j).mul_left (c j))
  have hg := (series_gaussian b hb).congr heq
  refine ⟨(coefficientLaw ι)[fun omega => ell (fun j => scalarSeries (a j) omega)],
    Var[fun omega => ell (fun j => scalarSeries (a j) omega);coefficientLaw ι].toNNReal,?_⟩
  rw [Measure.map_map ell.continuous.measurable hm]
  exact hg.map_eq_gaussianReal

private theorem observation_measurable (d : ℕ) (L : ℝ)
    (sites : Fin 2 → (Fin d → ℝ)) : Measurable (pinObservation d L sites) :=
  Measurable.of_eval (fun _ => scalarSeries_measurable _)
private theorem residual_measurable (d : ℕ) (L : ℝ)
    (sites : Fin 2 → (Fin d → ℝ)) : Measurable (residual d L sites) :=
  regressionCoefficients_measurable d L sites (canonicalR d L sites) 0
private theorem shift_joint_measurable (d : ℕ) (L : ℝ)
    (sites : Fin 2 → (Fin d → ℝ)) :
    Measurable (fun p : (PinIndex d → ℝ) × Coefficients d => shift d L sites p.1 p.2) := by
  classical
  apply Measurable.of_eval
  intro m
  unfold shift
  fun_prop

private theorem coordinate_series (d : ℕ) (m : SourceMode d) (omega : Coefficients d) :
    scalarSeries (Pi.single m 1) omega = omega m := by
  classical
  simp [scalarSeries,Pi.single_apply]

private theorem single_summable {ι : Type*} [DecidableEq ι] (m : ι) :
    Summable (Pi.single m (1 : ℝ)) :=
  (hasSum_single m (fun n hn => by simp [hn])).summable
private theorem shift_measurable (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (v : PinIndex d → ℝ) : Measurable (shift d L sites v) := by
  apply Measurable.of_eval
  intro m
  unfold shift
  fun_prop

private theorem joint_block_gaussian (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (sites : Fin 2 → (Fin d → ℝ)) (S : Finset (SourceMode d)) :
    HasGaussianLaw (fun omega => ((fun m : S => residual d L sites omega m),
      pinObservation d L sites omega)) (sourceLaw d) := by
  classical
  let a : (S ⊕ PinIndex d) → SourceMode d → ℝ :=
    Sum.elim (fun m => Pi.single m.val 1) (fun i m => pinCoefficient d L sites m i)
  have ha : ∀ j, Summable (a j) := by
    intro j
    cases j with
    | inl m => exact single_summable m.val
    | inr i => exact row_summable d hL sites i
  let Q : ((S ⊕ PinIndex d) → ℝ) →L[ℝ] ((S → ℝ) × (PinIndex d → ℝ)) :=
    (ContinuousLinearMap.pi (fun m : S =>
      ContinuousLinearMap.proj (Sum.inl m) - ∑ i, ∑ j,
        (pinCoefficient d L sites m.val i * canonicalR d L sites i j) •
          ContinuousLinearMap.proj (Sum.inr j))).prod
    (ContinuousLinearMap.pi (fun j => ContinuousLinearMap.proj (Sum.inr j)))
  have hg := (series_family_gaussian a ha).map_fun Q
  apply hg.congr
  filter_upwards with omega
  ext i <;> simp [Q,a,residual,regressionCoefficients,coordinate_series,pinObservation,
    smul_eq_mul,mul_assoc]

private theorem shift_reconstruct (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (omega : Coefficients d) :
    shift d L sites (pinObservation d L sites omega) (residual d L sites omega) = omega := by
  classical
  funext m
  simp [shift,residual,regressionCoefficients]

private theorem kernel_as_shift (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (v : PinIndex d → ℝ) :
    canonicalKernel d L sites v = ((sourceLaw d).map (residual d L sites)).map
      (shift d L sites v) := by
  classical
  rw [canonicalKernel,Kernel.comap_apply,regressionKernel_apply,transformedLaw,
    Measure.map_map (shift_measurable d L sites v) (residual_measurable d L sites)]
  congr 1
  funext omega m
  simp [shift,residual,regressionCoefficients]

theorem canonicalKernel_apply (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ))
    (v : PinIndex d → ℝ) :
    canonicalKernel d L sites v = transformedLaw d L sites (canonicalR d L sites) v := by
  exact regressionKernel_apply d L (sites,canonicalR d L sites,v)

set_option maxHeartbeats 800000 in
theorem residual_pin_covariance (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ))
    (hxy : torusProjection d L (sites 0) ≠ torusProjection d L (sites 1))
    (m : SourceMode d) (i : PinIndex d) :
    cov[fun omega => residual d L sites omega m,
      fun omega => pinObservation d L sites omega i;sourceLaw d] = 0 := by
  classical
  have hp (j : PinIndex d) : MemLp (fun omega => pinObservation d L sites omega j) 2
      (sourceLaw d) := (series_gaussian _ (row_summable d hL.ne' sites j)).memLp_two
  have ho : MemLp (fun omega : Coefficients d => omega m) 2 (sourceLaw d) := by
    have he : scalarSeries (Pi.single m 1) = (fun omega : Coefficients d => omega m) :=
      funext (coordinate_series d m)
    rw [← he]
    exact (series_gaussian (Pi.single m 1) (single_summable m)).memLp_two
  have hc (j k : PinIndex d) : MemLp (fun omega =>
      pinCoefficient d L sites m j * canonicalR d L sites j k *
        pinObservation d L sites omega k) 2 (sourceLaw d) :=
    (hp k).const_mul _
  have hs (j : PinIndex d) : MemLp (fun omega => ∑ k,
      pinCoefficient d L sites m j * canonicalR d L sites j k *
        pinObservation d L sites omega k) 2 (sourceLaw d) :=
    memLp_finsetSum Finset.univ (fun k _ => hc j k)
  have htot : MemLp (fun omega => ∑ j, ∑ k,
      pinCoefficient d L sites m j * canonicalR d L sites j k *
        pinObservation d L sites omega k) 2 (sourceLaw d) :=
    memLp_finsetSum Finset.univ (fun j _ => hs j)
  have hcoord : cov[fun omega : Coefficients d => omega m,
      fun omega => pinObservation d L sites omega i;sourceLaw d] =
      pinCoefficient d L sites m i := by
    have he : (fun omega : Coefficients d => omega m) = scalarSeries (Pi.single m 1) :=
      funext (fun omega => (coordinate_series d m omega).symm)
    rw [he]
    exact (scalarSeries_covariance _ _ (single_summable m)
      (row_summable d hL.ne' sites i)).trans (by simp [Pi.single_apply])
  have hpin (j : PinIndex d) : cov[fun omega => pinObservation d L sites omega j,
      fun omega => pinObservation d L sites omega i;sourceLaw d] =
      pinCovariance d L sites j i := by
    change cov[scalarSeries (fun m => pinCoefficient d L sites m j),
      scalarSeries (fun m => pinCoefficient d L sites m i);coefficientLaw (SourceMode d)] = _
    rw [scalarSeries_covariance _ _ (row_summable d hL.ne' sites j)
      (row_summable d hL.ne' sites i)]
    exact congrFun (congrFun (pinCovariance_eq_gram d hL.ne' sites).symm j) i
  simp only [residual,regressionCoefficients,Pi.zero_apply,mul_zero,Finset.sum_const_zero,
    add_zero]
  rw [covariance_fun_sub_left ho htot (hp i),hcoord,
    covariance_fun_sum_left hs (hp i)]
  simp_rw [covariance_fun_sum_left (fun k => hc _ k) (hp i),
    covariance_const_mul_left,hpin]
  have hsum : (∑ j, ∑ k, pinCoefficient d L sites m j * canonicalR d L sites j k *
      pinCovariance d L sites k i) = pinCoefficient d L sites m i := by
    calc
      _ = ∑ j, pinCoefficient d L sites m j *
        (canonicalR d L sites * pinCovariance d L sites) j i := by
          simp only [Matrix.mul_apply,Finset.mul_sum,mul_assoc]
      _ = _ := by rw [(canonicalR_inverse d hL sites hxy).2]; simp [Matrix.one_apply]
  rw [hsum,sub_self]


theorem residual_independent (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ))
    (hxy : torusProjection d L (sites 0) ≠ torusProjection d L (sites 1)) :
    IndepFun (residual d L sites) (pinObservation d L sites) (sourceLaw d) := by
  classical
  apply IndepFun.process_indepFun
    (fun m => (measurable_pi_apply m).comp (residual_measurable d L sites))
    (observation_measurable d L sites)
  intro S
  exact (joint_block_gaussian d hL.ne' sites S).indepFun_of_covariance_eval
    (fun m i => residual_pin_covariance d hL sites hxy m i)


theorem canonical_disintegration (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ))
    (hxy : torusProjection d L (sites 0) ≠ torusProjection d L (sites 1)) :
    pinJoint d L sites = ((sourceLaw d).map (pinObservation d L sites)) ⊗ₘ
      canonicalKernel d L sites := by
  classical
  let mu := (sourceLaw d).map (pinObservation d L sites)
  let nu := (sourceLaw d).map (residual d L sites)
  let P : (PinIndex d → ℝ) × Coefficients d → (PinIndex d → ℝ) × Coefficients d :=
    fun p => (p.1,shift d L sites p.1 p.2)
  have hP : Measurable P := measurable_fst.prodMk (shift_joint_measurable d L sites)
  have hind : (sourceLaw d).map (fun omega =>
      (pinObservation d L sites omega,residual d L sites omega)) = mu.prod nu :=
    (residual_independent d hL sites hxy).symm.map_prod_eq_prod_map_map
      (observation_measurable d L sites).aemeasurable
      (residual_measurable d L sites).aemeasurable
  have hrec : (mu.prod nu).map P = pinJoint d L sites := by
    rw [← hind,Measure.map_map hP
      ((observation_measurable d L sites).prodMk (residual_measurable d L sites))]
    congr 1
    funext omega
    exact congrArg (Prod.mk (pinObservation d L sites omega)) (shift_reconstruct d L sites omega)
  rw [← hrec]
  ext s hs : 1
  rw [Measure.map_apply hP hs,Measure.prod_apply (hs.preimage hP),Measure.compProd_apply hs]
  apply lintegral_congr
  intro v
  rw [kernel_as_shift,Measure.map_apply
    (shift_measurable d L sites v)
    (hs.preimage measurable_prodMk_left)]
  rfl


theorem canonicalKernel_eq_condKernel_ae (d : ℕ) {L : ℝ} (hL : 0 < L)
    (sites : Fin 2 → (Fin d → ℝ))
    (hxy : torusProjection d L (sites 0) ≠ torusProjection d L (sites 1)) :
    ∀ᵐ v ∂(sourceLaw d).map (pinObservation d L sites),
      canonicalKernel d L sites v = (pinJoint d L sites).condKernel v := by
  have hfst : (pinJoint d L sites).fst = (sourceLaw d).map (pinObservation d L sites) := by
    exact Measure.fst_map_prodMk (observation_measurable d L sites) measurable_id
  rw [← hfst]
  apply eq_condKernel_of_measure_eq_compProd
  rw [hfst]
  exact canonical_disintegration d hL sites hxy


end I4PinDisintegration

