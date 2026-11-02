import PinReproduction

noncomputable section
open I4Foundation I4Source I4PinMajorant I4PinKernel I4TwoPinRank I4PinReproduction I4Torus
  MeasureTheory ProbabilityTheory
open scoped BigOperators Matrix
namespace PinReproductionTests

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

example (d : ℕ) {L : ℝ} (hL : L ≠ 0) (sites : Fin 2 → (Fin d → ℝ)) :
    pinCovariance d L sites = coefficientGram d L sites := pinCovariance_eq_gram d hL sites

example : pinCovariance 2 1 (separatedSites 1) * canonicalR 2 1 (separatedSites 1) = 1 :=
  (canonicalR_inverse 2 (by norm_num) _ (separated 1)).1

example : canonicalR 2 1 (separatedSites 1) * pinCovariance 2 1 (separatedSites 1) = 1 :=
  (canonicalR_inverse 2 (by norm_num) _ (separated 1)).2

example (d : ℕ) {L : ℝ} (hL : L ≠ 0) : Measurable (canonicalR d L) :=
  canonicalR_measurable d hL

-- Arbitrary nonzero physical values AND gradients are reproduced under the changed law.
example : ∀ᵐ eta ∂transformedLaw 2 1 (separatedSites 1)
    (canonicalR 2 1 (separatedSites 1)) (fun _ => 7),
    actualPinObservation 2 1 (separatedSites 1) eta = fun _ => 7 :=
  canonical_pin_reproduction 2 (by norm_num) _ (separated 1) _

example : ∀ᵐ eta ∂transformedLaw 1 1 (separatedSites 0)
    (canonicalR 1 1 (separatedSites 0)) (fun _ => 3),
    fderiv ℝ (sourceField 1 1 eta) (separatedSites 0 1) (Pi.single 0 1) = 3 := by
  filter_upwards [canonical_pin_reproduction 1 (by norm_num) _ (separated 0) (fun _ => 3)]
    with eta h
  exact congrFun h (1,some 0)

example : ∀ᵐ eta ∂transformedLaw 1 1 (separatedSites 0)
    (canonicalR 1 1 (separatedSites 0)) (fun _ => 3),
    sourceField 1 1 eta (separatedSites 0 0) = 3 := by
  filter_upwards [canonical_pin_reproduction 1 (by norm_num) _ (separated 0) (fun _ => 3)]
    with eta h
  exact congrFun h (0,none)

-- Actual parameterized kernel is the same canonical transformed law.
example : ∀ᵐ eta ∂regressionKernel 2 1
    (separatedSites 1,canonicalR 2 1 (separatedSites 1),fun _ => 9),
    actualPinObservation 2 1 (separatedSites 1) eta = fun _ => 9 := by
  rw [regressionKernel_apply]
  exact canonical_pin_reproduction 2 (by norm_num) _ (separated 1) _

private theorem colliding_inverse_zero (d : ℕ) (L : ℝ) (x : Fin d → ℝ) :
    canonicalR d L (fun _ => x) = 0 := by
  classical
  have hrow : pinCovariance d L (fun _ => x) (0,none) =
      pinCovariance d L (fun _ => x) (1,none) := by
    funext j
    rfl
  have hn : ((0,none) : PinIndex d) ≠ (1,none) := by
    intro h
    have hh := congrArg Prod.fst h
    norm_num at hh
  have hd := Matrix.det_zero_of_row_eq hn hrow
  unfold canonicalR
  apply Matrix.nonsing_inv_apply_not_isUnit
  rw [hd]
  simp

example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) : canonicalR d L (fun _ => x) = 0 :=
  colliding_inverse_zero d L x

-- Borel inverse at singular matrices does not imply reproduction or nondegeneracy.
example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) (v : PinIndex d → ℝ) :
    regressionCoefficients d L (fun _ => x) (canonicalR d L (fun _ => x)) v 0 = 0 := by
  rw [colliding_inverse_zero]
  funext m
  simp [regressionCoefficients]

def incompatibleTarget (d : ℕ) : PinIndex d → ℝ := fun i => if i.1=0 then 0 else 1

example (d : ℕ) (L : ℝ) (x : Fin d → ℝ) (eta : SourceMode d → ℝ) :
    actualPinObservation d L (fun _ => x) eta ≠ incompatibleTarget d := by
  intro h
  have h0 := congrFun h (0,none)
  have h1 := congrFun h (1,none)
  norm_num [actualPinObservation,incompatibleTarget] at h0 h1
  linarith

-- Unequal cover representatives may still denote the SAME torus pin.
example : torusProjection 1 1 (0 : Fin 1 → ℝ) = torusProjection 1 1 (fun _ => 1) := by
  funext i; simp [torusProjection,AddCircle.coe_period]

example (eta : SourceMode 1 → ℝ) :
    actualPinObservation 1 1 ![0,fun _ => 1] eta ≠ incompatibleTarget 1 := by
  intro h
  have h0 := congrFun h (0,none)
  have h1 := congrFun h (1,none)
  have he := sourceField_eq_of_torusProjection_eq 1 (by norm_num : (1:ℝ) ≠ 0)
    eta (0 : Fin 1 → ℝ) (fun _ => 1) (by
      funext i; simp [torusProjection,AddCircle.coe_period])
  norm_num [actualPinObservation,incompatibleTarget] at h0 h1
  linarith

-- ZeroR at genuine separated pins is NOT the canonical inverse or a reproduction theorem.
example : actualPinObservation 1 1 (separatedSites 0)
    (regressionCoefficients 1 1 (separatedSites 0) 0 (fun _ => 1) 0) ≠ fun _ => 1 := by
  intro h
  have h0 := congrFun h (0,none)
  norm_num [regressionCoefficients,actualPinObservation,sourceField,I4Foundation.scalarSeries] at h0

end PinReproductionTests

