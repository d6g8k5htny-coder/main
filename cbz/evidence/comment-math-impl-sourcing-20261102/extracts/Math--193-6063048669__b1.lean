import PinDisintegration

noncomputable section
open I4Foundation I4Weights I4Source I4Jets I4SourceGaussian I4SourceCovariance
  I4PinMajorant I4PinKernel I4TwoPinRank I4PinReproduction I4PinDisintegration I4Torus
  MeasureTheory ProbabilityTheory
open scoped BigOperators Matrix ProbabilityTheory
namespace PinDisintegrationTests

def separatedSites (d : ℕ) : Fin 2 → (Fin (d+1) → ℝ) := ![0,fun _ => 1/2]
private theorem separated (d : ℕ) :
    torusProjection (d+1) 1 (separatedSites d 0) ≠
      torusProjection (d+1) 1 (separatedSites d 1) := by
  let : Fact (0 < (1:ℝ)) := ⟨by norm_num⟩
  intro h
  have h0 := congrFun h (0 : Fin (d+1))
  change ((0:ℝ) : AddCircle (1:ℝ)) = ((1/2:ℝ) : AddCircle (1:ℝ)) at h0
  have he := (AddCircle.coe_eq_coe_iff_of_mem_Ico
    (p := (1:ℝ)) (a := 0) (by norm_num : (0:ℝ) ∈ Set.Ico 0 (0+1))
    (by norm_num : (1/2:ℝ) ∈ Set.Ico 0 (0+1))).mp h0
  norm_num at he

-- Literal same-source residual and full pin observation, not scalar margins alone.
example : IndepFun (residual 1 1 (separatedSites 0))
    (pinObservation 1 1 (separatedSites 0)) (sourceLaw 1) :=
  residual_independent 1 (by norm_num) _ (separated 0)
example : IndepFun (residual 2 1 (separatedSites 1))
    (pinObservation 2 1 (separatedSites 1)) (sourceLaw 2) :=
  residual_independent 2 (by norm_num) _ (separated 1)
example : cov[fun omega => residual 1 1 (separatedSites 0) omega none,
    fun omega => pinObservation 1 1 (separatedSites 0) omega (0,none);sourceLaw 1]=0 :=
  residual_pin_covariance 1 (by norm_num) _ (separated 0) none (0,none)
example : pinJoint 1 1 (separatedSites 0) =
    ((sourceLaw 1).map (pinObservation 1 1 (separatedSites 0))) ⊗ₘ
      canonicalKernel 1 1 (separatedSites 0) :=
  canonical_disintegration 1 (by norm_num) _ (separated 0)
example : pinJoint 2 1 (separatedSites 1) =
    ((sourceLaw 2).map (pinObservation 2 1 (separatedSites 1))) ⊗ₘ
      canonicalKernel 2 1 (separatedSites 1) :=
  canonical_disintegration 2 (by norm_num) _ (separated 1)
example : ∀ᵐ v ∂(sourceLaw 1).map (pinObservation 1 1 (separatedSites 0)),
    canonicalKernel 1 1 (separatedSites 0) v = (pinJoint 1 1 (separatedSites 0)).condKernel v :=
  canonicalKernel_eq_condKernel_ae 1 (by norm_num) _ (separated 0)
example (v : PinIndex 2 → ℝ) : canonicalKernel 2 1 (separatedSites 1) v =
    transformedLaw 2 1 (separatedSites 1) (canonicalR 2 1 (separatedSites 1)) v :=
  canonicalKernel_apply _ _ _ _
example : IsMarkovKernel (canonicalKernel 1 1 (separatedSites 0)) := inferInstance
example (v : PinIndex 1 → ℝ) : IsProbabilityMeasure (canonicalKernel 1 1 (separatedSites 0) v) :=
  inferInstance
-- Every target physical reproduction is imported separately, not inferred by native RCD a.e.
example (v : PinIndex 1 → ℝ) : ∀ᵐ eta ∂canonicalKernel 1 1 (separatedSites 0) v,
    actualPinObservation 1 1 (separatedSites 0) eta = v := by
  rw [canonicalKernel_apply]
  exact canonical_pin_reproduction 1 (by norm_num) _ (separated 0) v
-- Pointwise reconstruction also holds on default-tsum exceptional samples.
example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) (omega : Coefficients d) :
    shift d L sites (pinObservation d L sites omega) (residual d L sites omega) = omega := by
  classical
  funext m
  simp [shift,I4PinDisintegration.residual,regressionCoefficients]
-- ZeroR leaves the actual coefficients unchanged: it cannot generally remove crosscovariance.
example (d : ℕ) (L : ℝ) (sites : Fin 2 → (Fin d → ℝ)) (omega : Coefficients d) :
    regressionCoefficients d L sites 0 0 omega = omega := by
  classical
  funext m
  simp [regressionCoefficients]
example : 0 < cov[fun omega : Coefficients 1 => omega none,
    fun omega => pinObservation 1 1 (separatedSites 0) omega (0,none);sourceLaw 1] := by
  classical
  have hs : Summable (Pi.single (none : SourceMode 1) (1 : ℝ)) :=
    (hasSum_single none (fun n hn => by simp [hn])).summable
  have ha : Summable (fun m => pinCoefficient 1 1 (separatedSites 0) m (0,none)) :=
    modeJetCoefficient_summable 1 (by norm_num) 0 (by norm_num) _ _
  have he : (fun omega : Coefficients 1 => omega none) = scalarSeries (Pi.single none 1) := by
    funext omega
    simp [scalarSeries,Pi.single_apply]
  rw [he]
  change 0 < cov[scalarSeries (Pi.single none 1),
    scalarSeries (fun m => pinCoefficient 1 1 (separatedSites 0) m (0,none));
      coefficientLaw (SourceMode 1)]
  rw [scalarSeries_covariance _ _ hs ha]
  simp only [Pi.single_apply,ite_mul,one_mul,zero_mul,tsum_ite_eq]
  simp only [pinCoefficient,pinOrder,pinDirections,I4Jets.modeJetCoefficient]
  apply Real.sqrt_pos.2
  exact one_div_pos.mpr (sourceZ_pos 1 (by norm_num))
-- The singular canonical inverse fallback returns the unconditioned source law, not arbitrary pins.
private theorem collision_zero (d : ℕ) (L : ℝ) (x : Fin d → ℝ) :
    canonicalR d L (fun _ => x) = 0 := by
  classical
  have hrow : pinCovariance d L (fun _ => x) (0,none) =
      pinCovariance d L (fun _ => x) (1,none) := by funext j; rfl
  have hn : ((0,none) : PinIndex d) ≠ (1,none) := by
    intro h; have hh := congrArg Prod.fst h; norm_num at hh
  have hd := Matrix.det_zero_of_row_eq hn hrow
  unfold canonicalR
  apply Matrix.nonsing_inv_apply_not_isUnit
  rw [hd]
  simp
example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) : canonicalR d L (fun _ => x) = 0 :=
  collision_zero d L x
example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) (v : PinIndex d → ℝ) :
    canonicalKernel d L (fun _ => x) v = sourceLaw d := by
  classical
  rw [canonicalKernel_apply,collision_zero]
  have he : regressionCoefficients d L (fun _ => x) 0 v = id := by
    funext omega m; simp [regressionCoefficients]
  rw [transformedLaw,he,Measure.map_id]
example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) (eta : Coefficients d) :
    actualPinObservation d L (fun _ => x) eta ≠ (fun i => if i.1=0 then 0 else 1) := by
  intro h
  have h0 := congrFun h (0,none)
  have h1 := congrFun h (1,none)
  norm_num [actualPinObservation] at h0 h1
  linarith
-- Collision/alias excluded by the actual quotient condition, not mere cover inequality.
example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) :
    ¬torusProjection d L x ≠ torusProjection d L x := by simp
example : torusProjection 1 1 (0 : Fin 1 → ℝ) = torusProjection 1 1 (fun _ => 1) := by
  funext i; simp [torusProjection,AddCircle.coe_period]
example (L : ℝ) (x y : Fin 0 → ℝ) : torusProjection 0 L x = torusProjection 0 L y := by
  funext i; exact Fin.elim0 i
-- A kernel with the correct pin first coordinate but wrong sample coordinate is not a disintegration.
example : (Measure.dirac ((0,0) : ℝ × ℝ)) ≠
    (Measure.dirac (0 : ℝ)) ⊗ₘ (Kernel.const ℝ (Measure.dirac (1 : ℝ))) := by
  intro h
  have he := congrArg (fun mu : Measure (ℝ × ℝ) => mu {((0,0) : ℝ × ℝ)}) h
  rw [Measure.compProd_const,Measure.prod_dirac] at he
  norm_num at he

end PinDisintegrationTests

