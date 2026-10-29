import SourceEndpointRegression
import SourceDerivativeContinuity
import SourceContactSchur
import SourceContactEndpointParameters
import ConfluentIntegralIdentities
import Mathlib.Analysis.Normed.Group.Tannery
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

/-! Actual Fourier representations and covariance limits for the confluent pins
and pin-adjusted endpoint observations. Singular quotient values at zero are
not identified with their continuous contact extensions. -/

noncomputable section

open I4Foundation I4Source I4Smooth I4SmoothCarrier I4SourceGaussian
open I4SourceDerivativeGaussian I4SourceDerivativeContinuity I4SourceConfluentPins
open I4SourceEndpointRegression I4SourceContact I4ConfluentIntegralIdentities
open MeasureTheory ProbabilityTheory Matrix Filter
open scoped BigOperators NNReal Topology

namespace I4SourceConfluentLimits

abbrev rawDerivativeOrder : Fin 12 → ℕ := ![0,0,1,1,1,1,2,2,2,2,2,2]

def rawDerivativePoint (r : ℝ) : Fin 12 → (Fin 2 → ℝ) :=
  ![endpointM r,endpointS r,endpointM r,endpointS r,endpointM r,endpointS r,
    endpointM r,endpointM r,endpointM r,endpointS r,endpointS r,endpointS r]

def rawDerivativeDirections : (i : Fin 12) → Fin (rawDerivativeOrder i) → (Fin 2 → ℝ) :=
  fun i a => if i.val < 4 then directionX else if i.val < 6 then directionY
    else if i.val % 3 = 0 then directionX
    else if i.val % 3 = 1 ∧ a.val = 0 then directionX else directionY

def rawDerivativeCoefficient (L r : ℝ) (i : Fin 12) (m : SourceMode 2) : ℝ :=
  derivativeCoefficient 2 L (rawDerivativeOrder i) (rawDerivativePoint r i)
    (rawDerivativeDirections i) m

def pinObservationMap (r : ℝ) : (Fin 12 → ℝ) →L[ℝ] (Fin 6 → ℝ) :=
  (ContinuousLinearMap.fst ℝ (Fin 6 → ℝ) (Fin 6 → ℝ)).comp (endpointObservationMap r)

def adjustedObservationMap (r : ℝ) : (Fin 12 → ℝ) →L[ℝ] (Fin 6 → ℝ) :=
  (ContinuousLinearMap.snd ℝ (Fin 6 → ℝ) (Fin 6 → ℝ)).comp (endpointObservationMap r)

def pinCoefficient (L r : ℝ) (i : Fin 6) (m : SourceMode 2) : ℝ :=
  pinObservationMap r (fun j => rawDerivativeCoefficient L r j m) i

def adjustedCoefficient (L r : ℝ) (i : Fin 6) (m : SourceMode 2) : ℝ :=
  adjustedObservationMap r (fun j => rawDerivativeCoefficient L r j m) i

private theorem derivativeCoefficient_weighted_summable {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample 2 L) (k : ℕ) (x : Fin 2 → ℝ) (u : Fin k → (Fin 2 → ℝ)) :
    Summable (fun m => derivativeCoefficient 2 L k x u m * sampleInclusion 2 L ω m) := by
  apply ((ω.property k).mul_right (∏ j, ‖u j‖)).of_norm_bounded
  intro m
  rw [norm_mul, Real.norm_eq_abs]
  calc
    _ ≤ derivativeMajorant 2 L k u m * |sampleInclusion 2 L ω m| :=
      mul_le_mul_of_nonneg_right (derivativeCoefficient_bound 2 hL k u m x) (abs_nonneg _)
    _ = (orderModeBound k 2 L m * |sampleInclusion 2 L ω m|) * ∏ j, ‖u j‖ := by
      unfold derivativeMajorant
      ring

private theorem finiteSeries_map {ι : Type*} {n q : ℕ}
    (a : Fin n → ι → ℝ) (ω : ι → ℝ)
    (ha : ∀ i, Summable (fun m => a i m * ω m))
    (T : (Fin n → ℝ) →L[ℝ] (Fin q → ℝ)) :
    T (fun i => scalarSeries (a i) ω) =
      (fun j => scalarSeries (fun m => T (fun i => a i m) j) ω) := by
  have hs : HasSum (fun m i => a i m * ω m) (fun i => scalarSeries (a i) ω) :=
    Pi.hasSum.mpr (fun i => (ha i).hasSum)
  have ht := hs.map T.toAddMonoidHom T.continuous
  funext j
  have hj : HasSum (fun m => T (fun i => a i m) j * ω m)
      (T (fun i => scalarSeries (a i) ω) j) := by
    convert Pi.hasSum.mp ht j using 1
    · funext m
      have hm : (fun i => a i m * ω m) = ω m • (fun i => a i m) := by
        funext i
        simp only [Pi.smul_apply, smul_eq_mul, mul_comm]
      change T (fun i => a i m) j * ω m = T (fun i => a i m * ω m) j
      rw [hm, map_smul]
      exact mul_comm _ _
    · rfl
  exact hj.tsum_eq.symm

private theorem finiteCoefficient_map_summable {ι : Type*} {n q : ℕ}
    (a : Fin n → ι → ℝ) (ha : ∀ i, Summable (a i))
    (T : (Fin n → ℝ) →L[ℝ] (Fin q → ℝ)) (j : Fin q) :
    Summable (fun m => T (fun i => a i m) j) := by
  have hs : HasSum (fun m i => a i m) (fun i => ∑' m, a i m) :=
    Pi.hasSum.mpr (fun i => (ha i).hasSum)
  exact (Pi.hasSum.mp (hs.map T.toAddMonoidHom T.continuous) j).summable

theorem rawPinsWithHessians_series {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (ω : SmoothSample 2 L) :
    rawPinsWithHessians L r ω =
      (fun i => scalarSeries (rawDerivativeCoefficient L r i) (sampleInclusion 2 L ω)) := by
  funext i
  unfold rawDerivativeCoefficient
  rw [← smoothSourceField_derivative_series 2 hL ω]
  fin_cases i <;>
    dsimp [rawPinsWithHessians, rawPins, rawDerivativeCoefficient, rawDerivativeOrder,
      rawDerivativePoint, rawDerivativeDirections]
  all_goals first
    | rfl
    | exact (iteratedFDeriv_one_apply (rawDerivativeDirections 2)).symm
    | exact (iteratedFDeriv_one_apply (rawDerivativeDirections 3)).symm
    | exact (iteratedFDeriv_one_apply (rawDerivativeDirections 4)).symm
    | exact (iteratedFDeriv_one_apply (rawDerivativeDirections 5)).symm
    | apply congrArg (iteratedFDeriv ℝ 2 (smoothSourceField 2 L ω) (endpointM r))
      ext a
      fin_cases a <;> rfl
    | apply congrArg (iteratedFDeriv ℝ 2 (smoothSourceField 2 L ω) (endpointS r))
      ext a
      fin_cases a <;> rfl

theorem pinCoefficient_summable {L : ℝ} (hL : L ≠ 0) (r : ℝ) (i : Fin 6) :
    Summable (pinCoefficient L r i) :=
  finiteCoefficient_map_summable (rawDerivativeCoefficient L r)
    (fun j => derivativeCoefficient_summable 2 hL (rawDerivativeOrder j)
      (rawDerivativePoint r j) (rawDerivativeDirections j)) (pinObservationMap r) i

theorem adjustedCoefficient_summable {L : ℝ} (hL : L ≠ 0) (r : ℝ) (i : Fin 6) :
    Summable (adjustedCoefficient L r i) :=
  finiteCoefficient_map_summable (rawDerivativeCoefficient L r)
    (fun j => derivativeCoefficient_summable 2 hL (rawDerivativeOrder j)
      (rawDerivativePoint r j) (rawDerivativeDirections j)) (adjustedObservationMap r) i

theorem sourceConfluentPinsAt_series {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (ω : SmoothSample 2 L) :
    sourceConfluentPinsAt L r ω =
      (fun i => scalarSeries (pinCoefficient L r i) (sampleInclusion 2 L ω)) := by
  have hactual := congrArg Prod.fst (endpointObservationMap_actual L r ω)
  change pinObservationMap r (rawPinsWithHessians L r ω) = sourceConfluentPinsAt L r ω
    at hactual
  rw [← hactual, rawPinsWithHessians_series hL]
  exact finiteSeries_map (rawDerivativeCoefficient L r) (sampleInclusion 2 L ω)
    (fun j => derivativeCoefficient_weighted_summable hL ω (rawDerivativeOrder j)
      (rawDerivativePoint r j) (rawDerivativeDirections j)) (pinObservationMap r)

theorem sourceAdjustedEndpoints_series {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (ω : SmoothSample 2 L) :
    sourceAdjustedEndpoints L r ω =
      (fun i => scalarSeries (adjustedCoefficient L r i) (sampleInclusion 2 L ω)) := by
  have hactual := congrArg Prod.snd (endpointObservationMap_actual L r ω)
  change adjustedObservationMap r (rawPinsWithHessians L r ω) = sourceAdjustedEndpoints L r ω
    at hactual
  rw [← hactual, rawPinsWithHessians_series hL]
  exact finiteSeries_map (rawDerivativeCoefficient L r) (sampleInclusion 2 L ω)
    (fun j => derivativeCoefficient_weighted_summable hL ω (rawDerivativeOrder j)
      (rawDerivativePoint r j) (rawDerivativeDirections j)) (adjustedObservationMap r)

theorem sourcePinCovarianceAt_coefficient {L : ℝ} (hL : L ≠ 0) (r : ℝ) (i j : Fin 6) :
    sourcePinCovarianceAt L r i j = ∑' m, pinCoefficient L r i m * pinCoefficient L r j m := by
  unfold sourcePinCovarianceAt
  simp_rw [sourceConfluentPinsAt_series hL]
  rw [covariance_comp_inclusion 2 hL _ _ (scalarSeries_measurable _)
    (scalarSeries_measurable _)]
  exact I4SourceCovariance.scalarSeries_covariance _ _
    (pinCoefficient_summable hL r i) (pinCoefficient_summable hL r j)

theorem sourceCrossCovarianceAt_coefficient {L : ℝ} (hL : L ≠ 0) (r : ℝ) (i j : Fin 6) :
    sourceCrossCovarianceAt L r i j =
      ∑' m, adjustedCoefficient L r i m * pinCoefficient L r j m := by
  unfold sourceCrossCovarianceAt
  simp_rw [sourceConfluentPinsAt_series hL, sourceAdjustedEndpoints_series hL]
  rw [covariance_comp_inclusion 2 hL _ _ (scalarSeries_measurable _)
    (scalarSeries_measurable _)]
  exact I4SourceCovariance.scalarSeries_covariance _ _
    (adjustedCoefficient_summable hL r i) (pinCoefficient_summable hL r j)

theorem sourceEndpointCovarianceAt_coefficient {L : ℝ} (hL : L ≠ 0) (r : ℝ) (i j : Fin 6) :
    sourceEndpointCovarianceAt L r i j =
      ∑' m, adjustedCoefficient L r i m * adjustedCoefficient L r j m := by
  unfold sourceEndpointCovarianceAt
  simp_rw [sourceAdjustedEndpoints_series hL]
  rw [covariance_comp_inclusion 2 hL _ _ (scalarSeries_measurable _)
    (scalarSeries_measurable _)]
  exact I4SourceCovariance.scalarSeries_covariance _ _
    (adjustedCoefficient_summable hL r i) (adjustedCoefficient_summable hL r j)

end I4SourceConfluentLimits


/- The following source-mode chain qualifications were coauthored by
OpenAI/Codex /root/dm2_landed_composition and supplied as compiled scratch
ModeCurveChains.lean; this is proof authorship, not independent review. -/


open I4Source I4Weights I4SourceDerivativeGaussian I4SourceConfluentPins

namespace I4ModeCurveChains

def axisMap : ℝ →L[ℝ] (Fin 2 → ℝ) where
  toFun t := ![t,0]
  map_add' s t := by ext i; fin_cases i <;> simp
  map_smul' c t := by ext i; fin_cases i <;> simp
  cont := by fun_prop

theorem mode_smooth (L : ℝ) (m : SourceMode 2) :
    ContDiff ℝ (⊤ : ℕ∞) (modeCoefficient 2 L m) := by
  cases m with
  | none =>
    change ContDiff ℝ (⊤ : ℕ∞) (fun _ : Fin 2 → ℝ => Real.sqrt (1 / sourceZ 2 L))
    fun_prop
  | some p =>
    rcases p with ⟨k,b⟩
    change ContDiff ℝ (⊤ : ℕ∞) (fun x : Fin 2 → ℝ =>
      Real.sqrt (2 * sourceWeight 2 L k / sourceZ 2 L) *
        if b then Real.sin (phase 2 L k x) else Real.cos (phase 2 L k x))
    cases b <;> simp only [Bool.false_eq_true, ite_false, ite_true] <;>
      unfold phase <;> fun_prop

theorem axis_iteratedDeriv (f : (Fin 2 → ℝ) → ℝ)
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) (k : ℕ) (t : ℝ) :
    iteratedDeriv k (fun s => f ![s,0]) t =
      iteratedFDeriv ℝ k f ![t,0] (fun _ => directionX) := by
  rw [iteratedDeriv_eq_iteratedFDeriv]
  have h := axisMap.iteratedFDeriv_comp_right hf t (i := k) (by simp)
  have hh := congrArg (fun A => A (fun _ => (1 : ℝ))) h
  simpa [Function.comp_def, axisMap, directionX,
    ContinuousMultilinearMap.compContinuousLinearMap_apply] using hh

theorem axis_transverse_smooth (f : (Fin 2 → ℝ) → ℝ)
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) :
    ContDiff ℝ (⊤ : ℕ∞) (fun t => fderiv ℝ f ![t,0] directionY) := by
  have hdf : ContDiff ℝ (⊤ : ℕ∞) (fderiv ℝ f) := hf.fderiv_right (by simp)
  exact (hdf.clm_apply contDiff_const).comp axisMap.contDiff

theorem axis_transverse_iteratedDeriv (f : (Fin 2 → ℝ) → ℝ)
    (hf : ContDiff ℝ (⊤ : ℕ∞) f) (k : ℕ) (t : ℝ) :
    iteratedDeriv k (fun s => fderiv ℝ f ![s,0] directionY) t =
      iteratedFDeriv ℝ (k+1) f ![t,0] (Fin.snoc (fun _ => directionX) directionY) := by
  have hdf : ContDiff ℝ (⊤ : ℕ∞) (fderiv ℝ f) := hf.fderiv_right (by simp)
  have hfun : ContDiff ℝ (⊤ : ℕ∞) (fun x => fderiv ℝ f x directionY) :=
    hdf.clm_apply contDiff_const
  calc
    _ = iteratedFDeriv ℝ k (fun x => fderiv ℝ f x directionY) ![t,0]
        (fun _ => directionX) := axis_iteratedDeriv _ hfun k t
    _ = iteratedFDeriv ℝ k (fderiv ℝ f) ![t,0] (fun _ => directionX) directionY :=
      iteratedFDeriv_clm_apply_const_apply hdf (by simp)
    _ = _ := by
      rw [iteratedFDeriv_succ_apply_right, Fin.init_snoc, Fin.snoc_last]

def modeF (L : ℝ) (m : SourceMode 2) (t : ℝ) : ℝ := modeCoefficient 2 L m ![t,0]
def modeH (L : ℝ) (m : SourceMode 2) (t : ℝ) : ℝ :=
  derivativeCoefficient 2 L 1 ![t,0] ![directionY] m

theorem modeH_eq (L : ℝ) (m : SourceMode 2) :
    modeH L m = fun t => fderiv ℝ (modeCoefficient 2 L m) ![t,0] directionY := by
  funext t
  exact iteratedFDeriv_one_apply _

theorem modeF_C3 (L : ℝ) (m : SourceMode 2) : ContDiff ℝ 3 (modeF L m) :=
  ((mode_smooth L m).comp axisMap.contDiff).of_le (by simp)

theorem modeH_C2 (L : ℝ) (m : SourceMode 2) : ContDiff ℝ 2 (modeH L m) := by
  rw [modeH_eq]
  exact (axis_transverse_smooth _ (mode_smooth L m)).of_le (by simp)

theorem modeF_deriv (L : ℝ) (m : SourceMode 2) (t : ℝ) :
    deriv (modeF L m) t = derivativeCoefficient 2 L 1 ![t,0] ![directionX] m := by
  change deriv (fun s => modeCoefficient 2 L m ![s,0]) t = _
  have h := axis_iteratedDeriv _ (mode_smooth L m) 1 t
  simpa [derivativeCoefficient, iteratedDeriv_one] using h

theorem modeF_second (L : ℝ) (m : SourceMode 2) (t : ℝ) :
    iteratedDeriv 2 (modeF L m) t =
      derivativeCoefficient 2 L 2 ![t,0] ![directionX,directionX] m := by
  change iteratedDeriv 2 (fun s => modeCoefficient 2 L m ![s,0]) t = _
  have h := axis_iteratedDeriv _ (mode_smooth L m) 2 t
  have hd : (fun _ : Fin 2 => directionX) = ![directionX,directionX] := by
    ext i; fin_cases i <;> rfl
  simpa only [derivativeCoefficient, hd] using h

theorem modeF_third (L : ℝ) (m : SourceMode 2) (t : ℝ) :
    iteratedDeriv 3 (modeF L m) t =
      derivativeCoefficient 2 L 3 ![t,0] ![directionX,directionX,directionX] m := by
  change iteratedDeriv 3 (fun s => modeCoefficient 2 L m ![s,0]) t = _
  have h := axis_iteratedDeriv _ (mode_smooth L m) 3 t
  have hd : (fun _ : Fin 3 => directionX) = ![directionX,directionX,directionX] := by
    ext i; fin_cases i <;> rfl
  simpa only [derivativeCoefficient, hd] using h

theorem modeH_deriv (L : ℝ) (m : SourceMode 2) (t : ℝ) :
    deriv (modeH L m) t = derivativeCoefficient 2 L 2 ![t,0] ![directionX,directionY] m := by
  rw [modeH_eq]
  have h := axis_transverse_iteratedDeriv _ (mode_smooth L m) 1 t
  have hd : Fin.snoc (fun _ : Fin 1 => directionX) directionY =
      ![directionX,directionY] := by ext i; fin_cases i <;> rfl
  simpa only [iteratedDeriv_one, derivativeCoefficient, hd] using h

theorem modeH_second (L : ℝ) (m : SourceMode 2) (t : ℝ) :
    iteratedDeriv 2 (modeH L m) t =
      derivativeCoefficient 2 L 3 ![t,0] ![directionX,directionX,directionY] m := by
  rw [modeH_eq]
  have h := axis_transverse_iteratedDeriv _ (mode_smooth L m) 2 t
  have hd : Fin.snoc (fun _ : Fin 2 => directionX) directionY =
      ![directionX,directionX,directionY] := by ext i; fin_cases i <;> rfl
  simpa only [derivativeCoefficient, hd] using h

end I4ModeCurveChains

namespace I4SourceConfluentLimits

open I4ModeCurveChains

def modeYY (L : ℝ) (m : SourceMode 2) (t : ℝ) : ℝ :=
  derivativeCoefficient 2 L 2 ![t,0] ![directionY,directionY] m

def modeRawValues (L r : ℝ) (m : SourceMode 2) : Fin 12 → ℝ :=
  ![modeF L m (-r/2), modeF L m (r/2),
    deriv (modeF L m) (-r/2), deriv (modeF L m) (r/2),
    modeH L m (-r/2), modeH L m (r/2),
    iteratedDeriv 2 (modeF L m) (-r/2), deriv (modeH L m) (-r/2), modeYY L m (-r/2),
    iteratedDeriv 2 (modeF L m) (r/2), deriv (modeH L m) (r/2), modeYY L m (r/2)]

private theorem rawDirections2 : rawDerivativeDirections 2 = ![directionX] := by
  funext a
  fin_cases a
  rfl

private theorem rawDirections3 : rawDerivativeDirections 3 = ![directionX] := by
  funext a
  fin_cases a
  rfl

private theorem rawDirections4 : rawDerivativeDirections 4 = ![directionY] := by
  funext a
  fin_cases a
  rfl

private theorem rawDirections5 : rawDerivativeDirections 5 = ![directionY] := by
  funext a
  fin_cases a
  rfl

private theorem rawDirections6 : rawDerivativeDirections 6 = ![directionX,directionX] := by
  funext a
  fin_cases a <;> rfl

private theorem rawDirections7 : rawDerivativeDirections 7 = ![directionX,directionY] := by
  funext a
  fin_cases a <;> rfl

private theorem rawDirections8 : rawDerivativeDirections 8 = ![directionY,directionY] := by
  funext a
  fin_cases a <;> rfl

private theorem rawDirections9 : rawDerivativeDirections 9 = ![directionX,directionX] := by
  funext a
  fin_cases a <;> rfl

private theorem rawDirections10 : rawDerivativeDirections 10 = ![directionX,directionY] := by
  funext a
  fin_cases a <;> rfl

private theorem rawDirections11 : rawDerivativeDirections 11 = ![directionY,directionY] := by
  funext a
  fin_cases a <;> rfl

theorem rawDerivativeCoefficient_explicit (L r : ℝ) (m : SourceMode 2) :
    (fun i => rawDerivativeCoefficient L r i m) = modeRawValues L r m := by
  funext i
  fin_cases i
  all_goals try rfl
  all_goals first
    | change rawDerivativeCoefficient L r 2 m = modeRawValues L r m 2
    | change rawDerivativeCoefficient L r 3 m = modeRawValues L r m 3
    | change rawDerivativeCoefficient L r 4 m = modeRawValues L r m 4
    | change rawDerivativeCoefficient L r 5 m = modeRawValues L r m 5
    | change rawDerivativeCoefficient L r 6 m = modeRawValues L r m 6
    | change rawDerivativeCoefficient L r 7 m = modeRawValues L r m 7
    | change rawDerivativeCoefficient L r 8 m = modeRawValues L r m 8
    | change rawDerivativeCoefficient L r 9 m = modeRawValues L r m 9
    | change rawDerivativeCoefficient L r 10 m = modeRawValues L r m 10
    | change rawDerivativeCoefficient L r 11 m = modeRawValues L r m 11
  all_goals unfold rawDerivativeCoefficient
  all_goals first
    | rw [rawDirections2]
    | rw [rawDirections3]
    | rw [rawDirections4]
    | rw [rawDirections5]
    | rw [rawDirections6]
    | rw [rawDirections7]
    | rw [rawDirections8]
    | rw [rawDirections9]
    | rw [rawDirections10]
    | rw [rawDirections11]
  all_goals dsimp only [rawDerivativeOrder, rawDerivativePoint, modeRawValues]
  all_goals try simp only [modeF_deriv, modeF_second, modeH_deriv]
  all_goals rfl

def modePinValues (L r : ℝ) (m : SourceMode 2) : Fin 6 → ℝ :=
  ![(modeF L m (-r/2) + modeF L m (r/2))/2,
    (deriv (modeF L m) (-r/2) + deriv (modeF L m) (r/2))/2,
    (modeH L m (-r/2) + modeH L m (r/2))/2,
    pin3 (modeF L m) r, pin4 (modeH L m) r, pin5 (modeF L m) r]

theorem pinCoefficient_explicit (L r : ℝ) (m : SourceMode 2) :
    (fun i => pinCoefficient L r i m) = modePinValues L r m := by
  unfold pinCoefficient
  rw [rawDerivativeCoefficient_explicit]
  funext i
  fin_cases i <;>
    simp [pinObservationMap, endpointObservationMap, confluentTransform, modeRawValues,
      modePinValues, pin3, pin4, pin5, add_comm]

def modeAdjustedValues (L r : ℝ) (m : SourceMode 2) : Fin 6 → ℝ :=
  ![(iteratedDeriv 2 (modeF L m) (-r/2) - pin3 (modeF L m) r)/r,
    (iteratedDeriv 2 (modeF L m) (r/2) - pin3 (modeF L m) r)/r,
    (deriv (modeH L m) (-r/2) - pin4 (modeH L m) r)/r,
    (deriv (modeH L m) (r/2) - pin4 (modeH L m) r)/r,
    modeYY L m (-r/2), modeYY L m (r/2)]

theorem adjustedCoefficient_explicit (L r : ℝ) (m : SourceMode 2) :
    (fun i => adjustedCoefficient L r i m) = modeAdjustedValues L r m := by
  unfold adjustedCoefficient
  rw [rawDerivativeCoefficient_explicit]
  funext i
  fin_cases i <;>
    simp [adjustedObservationMap, endpointObservationMap, confluentTransform, modeRawValues,
      modeAdjustedValues, pin3, pin4]

end I4SourceConfluentLimits

/- Weighted-integral support coauthored by OpenAI/Codex
/root/dm2_landed_composition, supplied as compiled scratch
WeightedModeIntegrals.lean; no scientific-review credit. -/


open I4Source I4SourceDerivativeGaussian I4SourceDerivativeContinuity
open MeasureTheory Set
open scoped Interval

namespace I4WeightedModeIntegrals

def weightedDerivativeIntegral (L : ℝ) (k : ℕ) (u : Fin k → (Fin 2 → ℝ))
    (m : SourceMode 2) (w : ℝ → ℝ) (r : ℝ) : ℝ :=
  ∫ t in (-1/2 : ℝ)..(1/2 : ℝ), w t * derivativeCoefficient 2 L k ![r*t,0] u m

theorem weighted_integrand_bound {L : ℝ} (hL : L ≠ 0) (k : ℕ)
    (u : Fin k → (Fin 2 → ℝ)) (m : SourceMode 2) (w : ℝ → ℝ)
    (K : ℝ) (hK : 0 ≤ K)
    (hwbound : ∀ t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ), |w t| ≤ K)
    (r t : ℝ) (ht : t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ)) :
    ‖w t * derivativeCoefficient 2 L k ![r*t,0] u m‖ ≤
      K * derivativeMajorant 2 L k u m := by
  rw [norm_mul, Real.norm_eq_abs]
  exact mul_le_mul (hwbound t ht) (derivativeCoefficient_bound 2 hL k u m ![r*t,0])
    (norm_nonneg _) hK

theorem weightedDerivativeIntegral_continuous {L : ℝ} (hL : L ≠ 0) (k : ℕ)
    (u : Fin k → (Fin 2 → ℝ)) (m : SourceMode 2) (w : ℝ → ℝ) (hw : Continuous w)
    (K : ℝ) (hK : 0 ≤ K)
    (hwbound : ∀ t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ), |w t| ≤ K) :
    Continuous (weightedDerivativeIntegral L k u m w) := by
  unfold weightedDerivativeIntegral
  apply intervalIntegral.continuous_of_dominated_interval
    (bound := fun _ => K * derivativeMajorant 2 L k u m)
  · intro r
    have hc : Continuous (fun t => w t * derivativeCoefficient 2 L k ![r*t,0] u m) :=
      hw.mul ((derivativeCoefficient_continuous 2 L k u m).comp (by fun_prop))
    exact hc.aestronglyMeasurable
  · intro r
    exact Filter.Eventually.of_forall (fun t ht =>
      weighted_integrand_bound hL k u m w K hK hwbound r t ht)
  · exact continuous_const.intervalIntegrable _ _
  · exact Filter.Eventually.of_forall (fun t _ =>
      continuous_const.mul ((derivativeCoefficient_continuous 2 L k u m).comp (by fun_prop)))

theorem weightedDerivativeIntegral_norm_le {L : ℝ} (hL : L ≠ 0) (k : ℕ)
    (u : Fin k → (Fin 2 → ℝ)) (m : SourceMode 2) (w : ℝ → ℝ)
    (K : ℝ) (hK : 0 ≤ K)
    (hwbound : ∀ t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ), |w t| ≤ K) (r : ℝ) :
    ‖weightedDerivativeIntegral L k u m w r‖ ≤ K * derivativeMajorant 2 L k u m := by
  have h := intervalIntegral.norm_integral_le_of_norm_le_const
    (fun t ht => weighted_integrand_bound hL k u m w K hK hwbound r t ht)
  norm_num at h
  simpa only [weightedDerivativeIntegral, Real.norm_eq_abs, neg_div] using h

theorem weightedDerivativeIntegral_zero (L : ℝ) (k : ℕ)
    (u : Fin k → (Fin 2 → ℝ)) (m : SourceMode 2) (w : ℝ → ℝ) :
    weightedDerivativeIntegral L k u m w 0 =
      (∫ t in (-1/2 : ℝ)..(1/2 : ℝ), w t) * derivativeCoefficient 2 L k 0 u m := by
  unfold weightedDerivativeIntegral
  have hzero : (![0,0] : Fin 2 → ℝ) = 0 := by ext i; fin_cases i <;> rfl
  simp_rw [zero_mul, hzero]
  exact intervalIntegral.integral_mul_const _ _

end I4WeightedModeIntegrals

namespace I4SourceConfluentLimits

open I4ModeCurveChains I4WeightedModeIntegrals

def peanoLeft (t : ℝ) : ℝ := 1/2-t
def peanoRight (t : ℝ) : ℝ := t+1/2
def peanoCubic (t : ℝ) : ℝ := 1/4-t^2

private theorem peano_one_bound (t : ℝ)
    (_ht : t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ)) : |(1 : ℝ)| ≤ 1 := by norm_num

private theorem peano_left_bound (t : ℝ)
    (ht : t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ)) : |peanoLeft t| ≤ 1 := by
  rw [Set.uIoc_of_le (by norm_num)] at ht
  unfold peanoLeft
  exact abs_le.mpr ⟨by linarith [ht.2], by linarith [ht.1]⟩

private theorem peano_right_bound (t : ℝ)
    (ht : t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ)) : |peanoRight t| ≤ 1 := by
  rw [Set.uIoc_of_le (by norm_num)] at ht
  unfold peanoRight
  exact abs_le.mpr ⟨by linarith [ht.1], by linarith [ht.2]⟩

private theorem peano_cubic_bound (t : ℝ)
    (ht : t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ)) : |peanoCubic t| ≤ 1 := by
  rw [Set.uIoc_of_le (by norm_num)] at ht
  have habs : |t| ≤ (1/2 : ℝ) := abs_le.mpr ⟨by linarith [ht.1], ht.2⟩
  have hsq := (sq_le_sq₀ (abs_nonneg t) (by norm_num : (0 : ℝ) ≤ 1/2)).2 habs
  rw [sq_abs] at hsq
  unfold peanoCubic
  exact abs_le.mpr ⟨by nlinarith, by nlinarith [sq_nonneg t]⟩

theorem peano_kernel_masses :
    (∫ _t in (-1/2 : ℝ)..(1/2 : ℝ), (1 : ℝ)) = 1 ∧
    (∫ t in (-1/2 : ℝ)..(1/2 : ℝ), peanoLeft t) = 1/2 ∧
    (∫ t in (-1/2 : ℝ)..(1/2 : ℝ), peanoRight t) = 1/2 ∧
    6 * (∫ t in (-1/2 : ℝ)..(1/2 : ℝ), peanoCubic t) = 1 := by
  have hi : IntervalIntegrable (fun t : ℝ => t) volume (-1/2 : ℝ) (1/2 : ℝ) :=
    continuous_id.intervalIntegrable _ _
  have hp : IntervalIntegrable (fun t : ℝ => t^2) volume (-1/2 : ℝ) (1/2 : ℝ) :=
    (continuous_id.pow 2).intervalIntegrable _ _
  refine ⟨by norm_num, ?_, ?_, ?_⟩
  · unfold peanoLeft
    rw [intervalIntegral.integral_sub (continuous_const.intervalIntegrable _ _) hi]
    norm_num [integral_id]
  · unfold peanoRight
    rw [intervalIntegral.integral_add hi (continuous_const.intervalIntegrable _ _)]
    norm_num [integral_id]
  · unfold peanoCubic
    rw [intervalIntegral.integral_sub (continuous_const.intervalIntegrable _ _) hp]
    norm_num [integral_pow]

/-- The Peano formulas define a genuine continuous extension of each pin mode,
while preserving the singular raw formulas at every nonzero radius. -/
def regularizedPinCoefficient (L r : ℝ) (i : Fin 6) (m : SourceMode 2) : ℝ :=
  ![(modeF L m (-r/2) + modeF L m (r/2))/2,
    (deriv (modeF L m) (-r/2) + deriv (modeF L m) (r/2))/2,
    (modeH L m (-r/2) + modeH L m (r/2))/2,
    weightedDerivativeIntegral L 2 ![directionX,directionX] m (fun _ => 1) r,
    weightedDerivativeIntegral L 2 ![directionX,directionY] m (fun _ => 1) r,
    6 * weightedDerivativeIntegral L 3 ![directionX,directionX,directionX] m peanoCubic r] i

def regularizedAdjustedCoefficient (L r : ℝ) (i : Fin 6) (m : SourceMode 2) : ℝ :=
  ![-weightedDerivativeIntegral L 3 ![directionX,directionX,directionX] m peanoLeft r,
    weightedDerivativeIntegral L 3 ![directionX,directionX,directionX] m peanoRight r,
    -weightedDerivativeIntegral L 3 ![directionX,directionX,directionY] m peanoLeft r,
    weightedDerivativeIntegral L 3 ![directionX,directionX,directionY] m peanoRight r,
    modeYY L m (-r/2), modeYY L m (r/2)] i

theorem pinCoefficient_eq_regularized (L r : ℝ) (hr : r ≠ 0) (i : Fin 6)
    (m : SourceMode 2) : pinCoefficient L r i m = regularizedPinCoefficient L r i m := by
  rw [show pinCoefficient L r i m = modePinValues L r m i from
    congrFun (pinCoefficient_explicit L r m) i]
  fin_cases i
  · rfl
  · rfl
  · rfl
  · change pin3 (modeF L m) r = _
    rw [pin3_integral _ (modeF_C3 L m) r hr]
    simp_rw [modeF_second]
    simp [regularizedPinCoefficient, weightedDerivativeIntegral]
  · change pin4 (modeH L m) r = _
    rw [pin4_integral _ (modeH_C2 L m) r hr]
    simp_rw [modeH_deriv]
    simp [regularizedPinCoefficient, weightedDerivativeIntegral]
  · change pin5 (modeF L m) r = _
    rw [pin5_integral _ (modeF_C3 L m) r hr]
    simp_rw [modeF_third]
    rfl

theorem adjustedCoefficient_eq_regularized (L r : ℝ) (hr : r ≠ 0) (i : Fin 6)
    (m : SourceMode 2) : adjustedCoefficient L r i m = regularizedAdjustedCoefficient L r i m := by
  rw [show adjustedCoefficient L r i m = modeAdjustedValues L r m i from
    congrFun (adjustedCoefficient_explicit L r m) i]
  fin_cases i
  · change (iteratedDeriv 2 (modeF L m) (-r/2) - pin3 (modeF L m) r)/r = _
    rw [adjusted_second_left _ (modeF_C3 L m) r hr]
    simp_rw [modeF_third]
    rfl
  · change (iteratedDeriv 2 (modeF L m) (r/2) - pin3 (modeF L m) r)/r = _
    rw [adjusted_second_right _ (modeF_C3 L m) r hr]
    simp_rw [modeF_third]
    rfl
  · change (deriv (modeH L m) (-r/2) - pin4 (modeH L m) r)/r = _
    rw [adjusted_transverse_left _ (modeH_C2 L m) r hr]
    simp_rw [modeH_second]
    rfl
  · change (deriv (modeH L m) (r/2) - pin4 (modeH L m) r)/r = _
    rw [adjusted_transverse_right _ (modeH_C2 L m) r hr]
    simp_rw [modeH_second]
    rfl
  · rfl
  · rfl

end I4SourceConfluentLimits

/- Literal contact coefficient identification coauthored by OpenAI/Codex
/root/normalizer_palm_research, supplied as compiled scratch
ContactCoefficientLimits.lean; no independent-review credit. -/

open I4Foundation I4Source I4SourceGaussian I4SmoothCarrier I4SourceDerivativeGaussian
open I4SourceContact I4SourceConfluentPins MeasureTheory ProbabilityTheory
open scoped BigOperators
set_option autoImplicit false

namespace I4ContactCoefficientLimits

def contactPinCoefficient (L : ℝ) (i : Fin 6) (m : SourceMode 2) : ℝ :=
  sourceContactCoefficient L i.castSucc m

def contactAdjustedCoefficient (L : ℝ) (i : Fin 6) (m : SourceMode 2) : ℝ :=
  ![-derivativeCoefficient 2 L 3 0 ![directionX,directionX,directionX] m / 2,
    derivativeCoefficient 2 L 3 0 ![directionX,directionX,directionX] m / 2,
    -derivativeCoefficient 2 L 3 0 ![directionX,directionX,directionY] m / 2,
    derivativeCoefficient 2 L 3 0 ![directionX,directionX,directionY] m / 2,
    derivativeCoefficient 2 L 2 0 ![directionY,directionY] m,
    derivativeCoefficient 2 L 2 0 ![directionY,directionY] m] i

theorem contactPinCoefficient_summable {L : ℝ} (hL : L ≠ 0) (i : Fin 6) :
    Summable (contactPinCoefficient L i) :=
  sourceContactCoefficient_summable hL i.castSucc

theorem contactAdjustedCoefficient_summable {L : ℝ} (hL : L ≠ 0) (i : Fin 6) :
    Summable (contactAdjustedCoefficient L i) := by
  change Summable (fun m => contactAdjustedCoefficient L i m)
  have hx := derivativeCoefficient_summable 2 hL 3 0 ![directionX,directionX,directionX]
  have ha := derivativeCoefficient_summable 2 hL 3 0 ![directionX,directionX,directionY]
  have hq := derivativeCoefficient_summable 2 hL 2 0 ![directionY,directionY]
  fin_cases i
  · simpa [contactAdjustedCoefficient, div_eq_mul_inv, mul_comm] using
      hx.neg.mul_left ((2 : ℝ)⁻¹)
  · simpa [contactAdjustedCoefficient, div_eq_mul_inv, mul_comm] using
      hx.mul_left ((2 : ℝ)⁻¹)
  · simpa [contactAdjustedCoefficient, div_eq_mul_inv, mul_comm] using
      ha.neg.mul_left ((2 : ℝ)⁻¹)
  · simpa [contactAdjustedCoefficient, div_eq_mul_inv, mul_comm] using
      ha.mul_left ((2 : ℝ)⁻¹)
  · exact hq
  · exact hq

theorem sourcePinVector_eq_contactSeries {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample 2 L) (i : Fin 6) :
    sourcePinVector L ω i =
      scalarSeries (contactPinCoefficient L i) (sampleInclusion 2 L ω) :=
  smoothSourceField_derivative_series 2 hL ω (contactOrder i.castSucc) 0
    (contactDirections i.castSucc)

private theorem scalarSeries_div {ι : Type*} (a : ι → ℝ) (ω : ι → ℝ) (c : ℝ) :
    scalarSeries (fun m => a m / c) ω = scalarSeries a ω / c := by
  simp [scalarSeries, div_mul_eq_mul_div, tsum_div_const]

private theorem scalarSeries_neg {ι : Type*} (a : ι → ℝ) (ω : ι → ℝ) :
    scalarSeries (fun m => -a m) ω = -scalarSeries a ω := by
  simp [scalarSeries, neg_mul, tsum_neg]

theorem sourceContactEndpointVector_eq_contactSeries {L : ℝ} (hL : L ≠ 0)
    (ω : SmoothSample 2 L) (i : Fin 6) :
    sourceContactEndpointVector L ω i =
      scalarSeries (contactAdjustedCoefficient L i) (sampleInclusion 2 L ω) := by
  have hx := smoothSourceField_derivative_series 2 hL ω 3 0
    ![directionX,directionX,directionX]
  have ha := smoothSourceField_derivative_series 2 hL ω 3 0
    ![directionX,directionX,directionY]
  have hq := smoothSourceField_derivative_series 2 hL ω 2 0 ![directionY,directionY]
  fin_cases i
  · change -iteratedFDeriv ℝ 3 (smoothSourceField 2 L ω) 0
      ![directionX,directionX,directionX] / 2 =
      scalarSeries (fun m => -derivativeCoefficient 2 L 3 0
        ![directionX,directionX,directionX] m / 2) (sampleInclusion 2 L ω)
    rw [scalarSeries_div, scalarSeries_neg, hx]
  · change iteratedFDeriv ℝ 3 (smoothSourceField 2 L ω) 0
      ![directionX,directionX,directionX] / 2 =
      scalarSeries (fun m => derivativeCoefficient 2 L 3 0
        ![directionX,directionX,directionX] m / 2) (sampleInclusion 2 L ω)
    rw [scalarSeries_div, hx]
  · change -iteratedFDeriv ℝ 3 (smoothSourceField 2 L ω) 0
      ![directionX,directionX,directionY] / 2 =
      scalarSeries (fun m => -derivativeCoefficient 2 L 3 0
        ![directionX,directionX,directionY] m / 2) (sampleInclusion 2 L ω)
    rw [scalarSeries_div, scalarSeries_neg, ha]
  · change iteratedFDeriv ℝ 3 (smoothSourceField 2 L ω) 0
      ![directionX,directionX,directionY] / 2 =
      scalarSeries (fun m => derivativeCoefficient 2 L 3 0
        ![directionX,directionX,directionY] m / 2) (sampleInclusion 2 L ω)
    rw [scalarSeries_div, ha]
  · exact hq
  · exact hq

theorem sourcePinCovariance_eq_contactCoefficientGram {L : ℝ} (hL : L ≠ 0)
    (i j : Fin 6) : sourcePinCovariance L i j =
      ∑' m, contactPinCoefficient L i m * contactPinCoefficient L j m := by
  change covariance (fun ω => sourcePinVector L ω i) (fun ω => sourcePinVector L ω j)
    (smoothLaw 2 L) = _
  simp_rw [sourcePinVector_eq_contactSeries hL]
  rw [covariance_comp_inclusion 2 hL _ _ (scalarSeries_measurable _)
    (scalarSeries_measurable _)]
  exact I4SourceCovariance.scalarSeries_covariance _ _
    (contactPinCoefficient_summable hL i) (contactPinCoefficient_summable hL j)

theorem sourceContactEndpointCrossCovariance_eq_contactCoefficientGram {L : ℝ}
    (hL : L ≠ 0) (i j : Fin 6) : sourceContactEndpointCrossCovariance L i j =
      ∑' m, contactAdjustedCoefficient L i m * contactPinCoefficient L j m := by
  unfold sourceContactEndpointCrossCovariance
  simp_rw [sourceContactEndpointVector_eq_contactSeries hL, sourcePinVector_eq_contactSeries hL]
  rw [covariance_comp_inclusion 2 hL _ _ (scalarSeries_measurable _)
    (scalarSeries_measurable _)]
  exact I4SourceCovariance.scalarSeries_covariance _ _
    (contactAdjustedCoefficient_summable hL i) (contactPinCoefficient_summable hL j)

theorem sourceContactEndpointCovariance_eq_contactCoefficientGram {L : ℝ}
    (hL : L ≠ 0) (i j : Fin 6) : sourceContactEndpointCovariance L i j =
      ∑' m, contactAdjustedCoefficient L i m * contactAdjustedCoefficient L j m := by
  unfold sourceContactEndpointCovariance
  simp_rw [sourceContactEndpointVector_eq_contactSeries hL]
  rw [covariance_comp_inclusion 2 hL _ _ (scalarSeries_measurable _)
    (scalarSeries_measurable _)]
  exact I4SourceCovariance.scalarSeries_covariance _ _
    (contactAdjustedCoefficient_summable hL i) (contactAdjustedCoefficient_summable hL j)

end I4ContactCoefficientLimits

namespace I4SourceConfluentLimits

open I4ModeCurveChains I4WeightedModeIntegrals I4ContactCoefficientLimits

def modeComponentMajorant (L : ℝ) (m : SourceMode 2) : Fin 8 → ℝ :=
  ![derivativeMajorant 2 L 0 ![] m,
    derivativeMajorant 2 L 1 ![directionX] m,
    derivativeMajorant 2 L 1 ![directionY] m,
    derivativeMajorant 2 L 2 ![directionX,directionX] m,
    derivativeMajorant 2 L 2 ![directionX,directionY] m,
    derivativeMajorant 2 L 3 ![directionX,directionX,directionX] m,
    derivativeMajorant 2 L 3 ![directionX,directionX,directionY] m,
    derivativeMajorant 2 L 2 ![directionY,directionY] m]

def confluentModeMajorant (L : ℝ) (m : SourceMode 2) : ℝ :=
  6 * ∑ s, modeComponentMajorant L m s

private theorem modeComponentMajorant_nonneg (L : ℝ) (m : SourceMode 2) (s : Fin 8) :
    0 ≤ modeComponentMajorant L m s := by
  fin_cases s <;> dsimp [modeComponentMajorant]
  all_goals exact derivativeMajorant_nonneg 2 L _ _ m

theorem confluentModeMajorant_nonneg (L : ℝ) (m : SourceMode 2) :
    0 ≤ confluentModeMajorant L m :=
  mul_nonneg (by norm_num) (Finset.sum_nonneg (fun s _ => modeComponentMajorant_nonneg L m s))

theorem confluentModeMajorant_summable {L : ℝ} (hL : L ≠ 0) :
    Summable (confluentModeMajorant L) := by
  apply Summable.mul_left 6
  apply summable_sum
  intro s _
  fin_cases s <;> dsimp [modeComponentMajorant]
  all_goals exact derivativeMajorant_summable 2 hL _ _

private theorem six_component_le_majorant (L : ℝ) (m : SourceMode 2) (s : Fin 8) :
    6 * modeComponentMajorant L m s ≤ confluentModeMajorant L m := by
  apply mul_le_mul_of_nonneg_left _ (by norm_num)
  exact Finset.single_le_sum (fun t _ => modeComponentMajorant_nonneg L m t) (Finset.mem_univ s)

private theorem component_le_majorant (L : ℝ) (m : SourceMode 2) (s : Fin 8) :
    modeComponentMajorant L m s ≤ confluentModeMajorant L m := by
  have h := six_component_le_majorant L m s
  have hn := modeComponentMajorant_nonneg L m s
  linarith

private theorem average_norm_le (a b M : ℝ) (ha : ‖a‖ ≤ M) (hb : ‖b‖ ≤ M) :
    ‖(a+b)/2‖ ≤ M := by
  have h := (norm_add_le a b).trans (add_le_add ha hb)
  calc
    _ = ‖a+b‖ / 2 := by rw [norm_div]; norm_num
    _ ≤ (M+M)/2 := div_le_div_of_nonneg_right h (by norm_num)
    _ = M := by ring

private theorem integral_bound_one {L : ℝ} (hL : L ≠ 0) (k : ℕ)
    (u : Fin k → (Fin 2 → ℝ)) (m : SourceMode 2) (w : ℝ → ℝ)
    (hw : ∀ t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ), |w t| ≤ 1) (r : ℝ) :
    ‖weightedDerivativeIntegral L k u m w r‖ ≤ derivativeMajorant 2 L k u m := by
  have h := weightedDerivativeIntegral_norm_le hL k u m w 1 (by norm_num) hw r
  simpa only [one_mul] using h

private theorem modeF_norm_bound {L : ℝ} (hL : L ≠ 0) (m : SourceMode 2) (t : ℝ) :
    ‖modeF L m t‖ ≤ derivativeMajorant 2 L 0 ![] m := by
  simpa only [derivativeCoefficient, iteratedFDeriv_zero_apply, modeF] using
    derivativeCoefficient_bound 2 hL 0 ![] m ![t,0]

theorem regularizedPinCoefficient_bound {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (i : Fin 6) (m : SourceMode 2) :
    ‖regularizedPinCoefficient L r i m‖ ≤ confluentModeMajorant L m := by
  fin_cases i
  · exact (average_norm_le _ _ _ (modeF_norm_bound hL m (-r/2))
      (modeF_norm_bound hL m (r/2))).trans (component_le_majorant L m 0)
  · change ‖(deriv (modeF L m) (-r/2) + deriv (modeF L m) (r/2))/2‖ ≤ _
    simp_rw [modeF_deriv]
    exact (average_norm_le _ _ _ (derivativeCoefficient_bound 2 hL 1 ![directionX] m _)
      (derivativeCoefficient_bound 2 hL 1 ![directionX] m _)).trans (component_le_majorant L m 1)
  · exact (average_norm_le _ _ _ (derivativeCoefficient_bound 2 hL 1 ![directionY] m _)
      (derivativeCoefficient_bound 2 hL 1 ![directionY] m _)).trans (component_le_majorant L m 2)
  · exact (integral_bound_one hL 2
      ![directionX,directionX] m (fun _ => 1) peano_one_bound r).trans
      (component_le_majorant L m 3)
  · exact (integral_bound_one hL 2
      ![directionX,directionY] m (fun _ => 1) peano_one_bound r).trans
      (component_le_majorant L m 4)
  · change ‖6 * weightedDerivativeIntegral L 3 ![directionX,directionX,directionX] m peanoCubic r‖ ≤ _
    rw [norm_mul, show ‖(6 : ℝ)‖ = 6 from by norm_num]
    have h := integral_bound_one hL 3 ![directionX,directionX,directionX] m
      peanoCubic peano_cubic_bound r
    calc
      _ ≤ 6 * derivativeMajorant 2 L 3 ![directionX,directionX,directionX] m :=
        mul_le_mul_of_nonneg_left h (by norm_num)
      _ ≤ confluentModeMajorant L m := six_component_le_majorant L m 5

theorem regularizedAdjustedCoefficient_bound {L : ℝ} (hL : L ≠ 0) (r : ℝ)
    (i : Fin 6) (m : SourceMode 2) :
    ‖regularizedAdjustedCoefficient L r i m‖ ≤ confluentModeMajorant L m := by
  fin_cases i
  · change ‖-weightedDerivativeIntegral L 3 ![directionX,directionX,directionX] m peanoLeft r‖ ≤ _
    rw [norm_neg]
    exact (integral_bound_one hL 3
      ![directionX,directionX,directionX] m peanoLeft peano_left_bound r).trans
      (component_le_majorant L m 5)
  · exact (integral_bound_one hL 3
      ![directionX,directionX,directionX] m peanoRight peano_right_bound r).trans
      (component_le_majorant L m 5)
  · change ‖-weightedDerivativeIntegral L 3 ![directionX,directionX,directionY] m peanoLeft r‖ ≤ _
    rw [norm_neg]
    exact (integral_bound_one hL 3
      ![directionX,directionX,directionY] m peanoLeft peano_left_bound r).trans
      (component_le_majorant L m 6)
  · exact (integral_bound_one hL 3
      ![directionX,directionX,directionY] m peanoRight peano_right_bound r).trans
      (component_le_majorant L m 6)
  · exact (derivativeCoefficient_bound 2 hL 2 ![directionY,directionY] m _).trans
      (component_le_majorant L m 7)
  · exact (derivativeCoefficient_bound 2 hL 2 ![directionY,directionY] m _).trans
      (component_le_majorant L m 7)

end I4SourceConfluentLimits

/- Coauthor support: OpenAI/Codex /root/dm2_landed_composition: regularized coefficient continuity.
Copied from the supplied compiled scratch proof; no review credit. -/


open I4Source I4SourceDerivativeGaussian I4SourceDerivativeContinuity
open I4SourceConfluentPins I4ModeCurveChains I4WeightedModeIntegrals
open MeasureTheory Set
open scoped Interval

namespace I4SourceConfluentLimits

private theorem helper_left_bound (t : ℝ)
    (ht : t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ)) : |peanoLeft t| ≤ 1 := by
  rw [Set.uIoc_of_le (by norm_num)] at ht
  unfold peanoLeft
  exact abs_le.mpr ⟨by linarith [ht.2], by linarith [ht.1]⟩

private theorem helper_right_bound (t : ℝ)
    (ht : t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ)) : |peanoRight t| ≤ 1 := by
  rw [Set.uIoc_of_le (by norm_num)] at ht
  unfold peanoRight
  exact abs_le.mpr ⟨by linarith [ht.1], by linarith [ht.2]⟩

private theorem helper_cubic_bound (t : ℝ)
    (ht : t ∈ Set.uIoc (-1/2 : ℝ) (1/2 : ℝ)) : |peanoCubic t| ≤ 1 := by
  rw [Set.uIoc_of_le (by norm_num)] at ht
  have habs : |t| ≤ (1/2 : ℝ) := abs_le.mpr ⟨by linarith [ht.1], ht.2⟩
  have hsq := (sq_le_sq₀ (abs_nonneg t) (by norm_num : (0 : ℝ) ≤ 1/2)).2 habs
  rw [sq_abs] at hsq
  unfold peanoCubic
  exact abs_le.mpr ⟨by nlinarith, by nlinarith [sq_nonneg t]⟩

theorem regularizedPinCoefficient_continuous {L : ℝ} (hL : L ≠ 0)
    (i : Fin 6) (m : SourceMode 2) :
    Continuous (fun r => regularizedPinCoefficient L r i m) := by
  have hF := (modeF_C3 L m).continuous
  have hH := (modeH_C2 L m).continuous
  have hF' : Continuous (deriv (modeF L m)) := by
    simpa only [iteratedDeriv_one] using
      (modeF_C3 L m).continuous_iteratedDeriv 1 (by norm_num)
  have hpoly : Continuous peanoCubic := by unfold peanoCubic; fun_prop
  fin_cases i
  · change Continuous (fun r : ℝ => (modeF L m (-r/2) + modeF L m (r/2))/2)
    exact ((hF.comp (by fun_prop)).add (hF.comp (by fun_prop))).div_const 2
  · change Continuous (fun r : ℝ => (deriv (modeF L m) (-r/2) + deriv (modeF L m) (r/2))/2)
    exact ((hF'.comp (by fun_prop)).add (hF'.comp (by fun_prop))).div_const 2
  · change Continuous (fun r : ℝ => (modeH L m (-r/2) + modeH L m (r/2))/2)
    exact ((hH.comp (by fun_prop)).add (hH.comp (by fun_prop))).div_const 2
  · exact weightedDerivativeIntegral_continuous hL 2 ![directionX,directionX] m
      (fun _ => 1) continuous_const 1 (by norm_num) (by intro t ht; norm_num)
  · exact weightedDerivativeIntegral_continuous hL 2 ![directionX,directionY] m
      (fun _ => 1) continuous_const 1 (by norm_num) (by intro t ht; norm_num)
  · exact (weightedDerivativeIntegral_continuous hL 3 ![directionX,directionX,directionX] m
      peanoCubic hpoly 1 (by norm_num) helper_cubic_bound).const_mul 6

theorem regularizedAdjustedCoefficient_continuous {L : ℝ} (hL : L ≠ 0)
    (i : Fin 6) (m : SourceMode 2) :
    Continuous (fun r => regularizedAdjustedCoefficient L r i m) := by
  have hleft : Continuous peanoLeft := by unfold peanoLeft; fun_prop
  have hright : Continuous peanoRight := by unfold peanoRight; fun_prop
  have hYY : Continuous (modeYY L m) :=
    (derivativeCoefficient_continuous 2 L 2 ![directionY,directionY] m).comp (by fun_prop)
  fin_cases i
  · exact (weightedDerivativeIntegral_continuous hL 3 ![directionX,directionX,directionX] m
      peanoLeft hleft 1 (by norm_num) helper_left_bound).neg
  · exact weightedDerivativeIntegral_continuous hL 3 ![directionX,directionX,directionX] m
      peanoRight hright 1 (by norm_num) helper_right_bound
  · exact (weightedDerivativeIntegral_continuous hL 3 ![directionX,directionX,directionY] m
      peanoLeft hleft 1 (by norm_num) helper_left_bound).neg
  · exact weightedDerivativeIntegral_continuous hL 3 ![directionX,directionX,directionY] m
      peanoRight hright 1 (by norm_num) helper_right_bound
  · exact hYY.comp (by fun_prop)
  · exact hYY.comp (by fun_prop)

end I4SourceConfluentLimits

/- Coauthor support: OpenAI/Codex /root/normalizer_palm_research: exact contact coefficient values.
Copied from the supplied compiled scratch proof; no review credit. -/

open I4Source I4SourceDerivativeGaussian I4SourceConfluentPins I4SourceContact
open I4SourceConfluentLimits I4ModeCurveChains I4WeightedModeIntegrals I4ContactCoefficientLimits
open MeasureTheory
open scoped BigOperators Interval
set_option autoImplicit false

namespace I4SourceConfluentLimits

theorem regularizedPinCoefficient_zero (L : ℝ) (i : Fin 6) (m : SourceMode 2) :
    regularizedPinCoefficient L 0 i m = contactPinCoefficient L i m := by
  have hpoint : (![0,0] : Fin 2 → ℝ) = 0 := by ext j; fin_cases j <;> rfl
  have hm := peano_kernel_masses
  fin_cases i
  · change (modeF L m (-0/2) + modeF L m (0/2))/2 =
      derivativeCoefficient 2 L 0 0 ![] m
    simp [modeF, derivativeCoefficient, iteratedFDeriv_zero_apply, hpoint]
  · change (deriv (modeF L m) (-0/2) + deriv (modeF L m) (0/2))/2 =
      derivativeCoefficient 2 L 1 0 ![directionX] m
    simp only [neg_zero, zero_div, modeF_deriv, hpoint]
    ring
  · change (modeH L m (-0/2) + modeH L m (0/2))/2 =
      derivativeCoefficient 2 L 1 0 ![directionY] m
    simp only [neg_zero, zero_div, modeH, hpoint]
    ring
  · change weightedDerivativeIntegral L 2 ![directionX,directionX] m (fun _ => 1) 0 =
      derivativeCoefficient 2 L 2 0 ![directionX,directionX] m
    rw [weightedDerivativeIntegral_zero, hm.1, one_mul]
  · change weightedDerivativeIntegral L 2 ![directionX,directionY] m (fun _ => 1) 0 =
      derivativeCoefficient 2 L 2 0 ![directionX,directionY] m
    rw [weightedDerivativeIntegral_zero, hm.1, one_mul]
  · change 6 * weightedDerivativeIntegral L 3 ![directionX,directionX,directionX]
      m peanoCubic 0 = derivativeCoefficient 2 L 3 0 ![directionX,directionX,directionX] m
    rw [weightedDerivativeIntegral_zero, ← mul_assoc, hm.2.2.2, one_mul]

theorem regularizedAdjustedCoefficient_zero (L : ℝ) (i : Fin 6) (m : SourceMode 2) :
    regularizedAdjustedCoefficient L 0 i m = contactAdjustedCoefficient L i m := by
  have hpoint : (![0,0] : Fin 2 → ℝ) = 0 := by ext j; fin_cases j <;> rfl
  have hm := peano_kernel_masses
  fin_cases i
  · change -weightedDerivativeIntegral L 3 ![directionX,directionX,directionX]
      m peanoLeft 0 = -derivativeCoefficient 2 L 3 0 ![directionX,directionX,directionX] m / 2
    rw [weightedDerivativeIntegral_zero, hm.2.1]
    ring
  · change weightedDerivativeIntegral L 3 ![directionX,directionX,directionX]
      m peanoRight 0 = derivativeCoefficient 2 L 3 0 ![directionX,directionX,directionX] m / 2
    rw [weightedDerivativeIntegral_zero, hm.2.2.1]
    ring
  · change -weightedDerivativeIntegral L 3 ![directionX,directionX,directionY]
      m peanoLeft 0 = -derivativeCoefficient 2 L 3 0 ![directionX,directionX,directionY] m / 2
    rw [weightedDerivativeIntegral_zero, hm.2.1]
    ring
  · change weightedDerivativeIntegral L 3 ![directionX,directionX,directionY]
      m peanoRight 0 = derivativeCoefficient 2 L 3 0 ![directionX,directionX,directionY] m / 2
    rw [weightedDerivativeIntegral_zero, hm.2.2.1]
    ring
  · change modeYY L m (-0/2) = derivativeCoefficient 2 L 2 0 ![directionY,directionY] m
    simp only [neg_zero, zero_div, modeYY, hpoint]
  · change modeYY L m (0/2) = derivativeCoefficient 2 L 2 0 ![directionY,directionY] m
    simp only [zero_div, modeYY, hpoint]

end I4SourceConfluentLimits

namespace I4SourceConfluentLimits

open I4ContactCoefficientLimits

def regularizedPinCovariance (L r : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  fun i j => ∑' m, regularizedPinCoefficient L r i m * regularizedPinCoefficient L r j m

def regularizedCrossCovariance (L r : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  fun i j => ∑' m, regularizedAdjustedCoefficient L r i m * regularizedPinCoefficient L r j m

def regularizedEndpointCovariance (L r : ℝ) : Matrix (Fin 6) (Fin 6) ℝ :=
  fun i j => ∑' m, regularizedAdjustedCoefficient L r i m * regularizedAdjustedCoefficient L r j m

private theorem regularizedGram_continuous {L : ℝ} (hL : L ≠ 0)
    (a b : ℝ → Fin 6 → SourceMode 2 → ℝ)
    (ha : ∀ i m, Continuous (fun r => a r i m))
    (hb : ∀ i m, Continuous (fun r => b r i m))
    (hba : ∀ r i m, ‖a r i m‖ ≤ confluentModeMajorant L m)
    (hbb : ∀ r i m, ‖b r i m‖ ≤ confluentModeMajorant L m) :
    Continuous (fun r => (fun i j => ∑' m, a r i m * b r j m : Matrix (Fin 6) (Fin 6) ℝ)) := by
  apply continuous_pi
  intro i
  apply continuous_pi
  intro j
  apply continuous_tsum
  · intro m
    exact (ha i m).mul (hb j m)
  · exact I4ContactGram.coefficient_product_summable
      (confluentModeMajorant_summable hL) (confluentModeMajorant_summable hL)
  · intro m r
    rw [norm_mul]
    exact mul_le_mul (hba r i m) (hbb r j m) (norm_nonneg _)
      (confluentModeMajorant_nonneg L m)

theorem regularizedPinCovariance_continuous {L : ℝ} (hL : L ≠ 0) :
    Continuous (regularizedPinCovariance L) :=
  regularizedGram_continuous hL (regularizedPinCoefficient L) (regularizedPinCoefficient L)
    (regularizedPinCoefficient_continuous hL) (regularizedPinCoefficient_continuous hL)
    (regularizedPinCoefficient_bound hL) (regularizedPinCoefficient_bound hL)

theorem regularizedCrossCovariance_continuous {L : ℝ} (hL : L ≠ 0) :
    Continuous (regularizedCrossCovariance L) :=
  regularizedGram_continuous hL (regularizedAdjustedCoefficient L) (regularizedPinCoefficient L)
    (regularizedAdjustedCoefficient_continuous hL) (regularizedPinCoefficient_continuous hL)
    (regularizedAdjustedCoefficient_bound hL) (regularizedPinCoefficient_bound hL)

theorem regularizedEndpointCovariance_continuous {L : ℝ} (hL : L ≠ 0) :
    Continuous (regularizedEndpointCovariance L) :=
  regularizedGram_continuous hL (regularizedAdjustedCoefficient L) (regularizedAdjustedCoefficient L)
    (regularizedAdjustedCoefficient_continuous hL) (regularizedAdjustedCoefficient_continuous hL)
    (regularizedAdjustedCoefficient_bound hL) (regularizedAdjustedCoefficient_bound hL)

theorem sourcePinCovarianceAt_eq_regularized {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0) :
    sourcePinCovarianceAt L r = regularizedPinCovariance L r := by
  ext i j
  rw [sourcePinCovarianceAt_coefficient hL]
  unfold regularizedPinCovariance
  apply tsum_congr
  intro m
  rw [pinCoefficient_eq_regularized L r hr i m, pinCoefficient_eq_regularized L r hr j m]

theorem sourceCrossCovarianceAt_eq_regularized {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0) :
    sourceCrossCovarianceAt L r = regularizedCrossCovariance L r := by
  ext i j
  rw [sourceCrossCovarianceAt_coefficient hL]
  unfold regularizedCrossCovariance
  apply tsum_congr
  intro m
  rw [adjustedCoefficient_eq_regularized L r hr i m, pinCoefficient_eq_regularized L r hr j m]

theorem sourceEndpointCovarianceAt_eq_regularized {L : ℝ} (hL : L ≠ 0) (r : ℝ) (hr : r ≠ 0) :
    sourceEndpointCovarianceAt L r = regularizedEndpointCovariance L r := by
  ext i j
  rw [sourceEndpointCovarianceAt_coefficient hL]
  unfold regularizedEndpointCovariance
  apply tsum_congr
  intro m
  rw [adjustedCoefficient_eq_regularized L r hr i m, adjustedCoefficient_eq_regularized L r hr j m]

theorem regularizedPinCovariance_zero {L : ℝ} (hL : L ≠ 0) :
    regularizedPinCovariance L 0 = sourcePinCovariance L := by
  ext i j
  rw [sourcePinCovariance_eq_contactCoefficientGram hL]
  unfold regularizedPinCovariance
  simp_rw [regularizedPinCoefficient_zero]

theorem regularizedCrossCovariance_zero {L : ℝ} (hL : L ≠ 0) :
    regularizedCrossCovariance L 0 = sourceContactEndpointCrossCovariance L := by
  ext i j
  rw [sourceContactEndpointCrossCovariance_eq_contactCoefficientGram hL]
  unfold regularizedCrossCovariance
  simp_rw [regularizedPinCoefficient_zero, regularizedAdjustedCoefficient_zero]

theorem regularizedEndpointCovariance_zero {L : ℝ} (hL : L ≠ 0) :
    regularizedEndpointCovariance L 0 = sourceContactEndpointCovariance L := by
  ext i j
  rw [sourceContactEndpointCovariance_eq_contactCoefficientGram hL]
  unfold regularizedEndpointCovariance
  simp_rw [regularizedAdjustedCoefficient_zero]

/-- The actual confluent six-pin covariance converges to the literal contact
pin covariance. Only the source period is assumed nonzero. -/
theorem sourcePinCovarianceAt_tendsto {L : ℝ} (hL : L ≠ 0) :
    Tendsto (sourcePinCovarianceAt L) (𝓝[>] (0 : ℝ)) (𝓝 (sourcePinCovariance L)) := by
  have h := (regularizedPinCovariance_continuous hL).tendsto 0
  rw [regularizedPinCovariance_zero hL] at h
  apply (h.mono_left nhdsWithin_le_nhds).congr'
  filter_upwards [self_mem_nhdsWithin] with r hr
  exact (sourcePinCovarianceAt_eq_regularized hL r (ne_of_gt hr)).symm

/-- The actual adjusted-endpoint/pin cross covariance has the stated literal
contact limit, with the pin corrections retained before division. -/
theorem sourceCrossCovarianceAt_tendsto {L : ℝ} (hL : L ≠ 0) :
    Tendsto (sourceCrossCovarianceAt L) (𝓝[>] (0 : ℝ))
      (𝓝 (sourceContactEndpointCrossCovariance L)) := by
  have h := (regularizedCrossCovariance_continuous hL).tendsto 0
  rw [regularizedCrossCovariance_zero hL] at h
  apply (h.mono_left nhdsWithin_le_nhds).congr'
  filter_upwards [self_mem_nhdsWithin] with r hr
  exact (sourceCrossCovarianceAt_eq_regularized hL r (ne_of_gt hr)).symm

/-- The actual pin-adjusted endpoint covariance converges to Cov(T,T), where
T=(-fxxx/2,fxxx/2,-fxxy/2,fxxy/2,fyy,fyy) at the origin. -/
theorem sourceEndpointCovarianceAt_tendsto {L : ℝ} (hL : L ≠ 0) :
    Tendsto (sourceEndpointCovarianceAt L) (𝓝[>] (0 : ℝ))
      (𝓝 (sourceContactEndpointCovariance L)) := by
  have h := (regularizedEndpointCovariance_continuous hL).tendsto 0
  rw [regularizedEndpointCovariance_zero hL] at h
  apply (h.mono_left nhdsWithin_le_nhds).congr'
  filter_upwards [self_mem_nhdsWithin] with r hr
  exact (sourceEndpointCovarianceAt_eq_regularized hL r (ne_of_gt hr)).symm

end I4SourceConfluentLimits
