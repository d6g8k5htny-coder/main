import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.MeasureTheory.Integral.IntervalIntegral.IntegrationByParts
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Exact integral formulas for confluent pins and adjusted endpoints

All integrals use the ordinary oriented interval `[-1/2,1/2]`. The physical
radius can have either sign. These deterministic scalar formulas use only C3
regularity of F and C2 regularity of H; no integral identity is a premise.
See SOURCE_CONFLUENT_INTEGRALS.md for the frozen source contract.
-/

noncomputable section

open MeasureTheory Set
open scoped Interval

namespace I4ConfluentIntegralIdentities

def pin3 (F : ℝ → ℝ) (r : ℝ) : ℝ :=
  (deriv F (r / 2) - deriv F (-r / 2)) / r

def pin4 (H : ℝ → ℝ) (r : ℝ) : ℝ :=
  (H (r / 2) - H (-r / 2)) / r

def pin5 (F : ℝ → ℝ) (r : ℝ) : ℝ :=
  (12 / r ^ 2) * ((deriv F (r / 2) + deriv F (-r / 2)) / 2 -
    (F (r / 2) - F (-r / 2)) / r)

private theorem scaled_hasDerivAt (g g' : ℝ → ℝ)
    (hg : ∀ x, HasDerivAt g (g' x) x) (r u : ℝ) :
    HasDerivAt (fun u => g (r * u)) (r * g' (r * u)) u := by
  have hru : HasDerivAt (fun u : ℝ => r*u) r u := by
    simpa [id] using (hasDerivAt_id u).const_mul r
  convert (hg (r*u)).comp u hru using 1 <;> simp [Function.comp_def, mul_comm]

private theorem average_identity (g g' : ℝ → ℝ)
    (hg : ∀ x, HasDerivAt g (g' x) x) (hc : Continuous g') (r : ℝ) (hr : r ≠ 0) :
    (g (r / 2) - g (-r / 2)) / r =
      ∫ u in (-1/2 : ℝ)..(1/2 : ℝ), g' (r * u) := by
  have hc' : Continuous (fun u => r * g' (r * u)) :=
    continuous_const.mul (hc.comp (continuous_const.mul continuous_id))
  have h := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (a := (-1/2 : ℝ)) (b := (1/2 : ℝ))
    (fun u _ => scaled_hasDerivAt g g' hg r u) (hc'.intervalIntegrable _ _)
  rw [intervalIntegral.integral_const_mul] at h
  have hleft : r * (-1/2) = -r / 2 := by ring
  have hright : r * (1/2) = r / 2 := by ring
  rw [hleft, hright] at h
  apply (div_eq_iff hr).2
  nlinarith [h]

private theorem left_endpoint_identity (g g' : ℝ → ℝ)
    (hg : ∀ x, HasDerivAt g (g' x) x) (hc : Continuous g') (r : ℝ) (hr : r ≠ 0) :
    (g (-r/2) - ∫ u in (-1/2 : ℝ)..(1/2 : ℝ), g (r*u)) / r =
      -(∫ u in (-1/2 : ℝ)..(1/2 : ℝ), (1/2-u) * g' (r*u)) := by
  have hgc : Continuous g :=
    (show Differentiable ℝ g from fun x => (hg x).differentiableAt).continuous
  have hc' : Continuous (fun u => r * g' (r*u)) :=
    continuous_const.mul (hc.comp (continuous_const.mul continuous_id))
  have h := intervalIntegral.integral_mul_deriv_eq_deriv_mul_of_hasDerivAt
    (a := (-1/2 : ℝ)) (b := (1/2 : ℝ))
    (u := fun u : ℝ => 1/2-u) (u' := fun _ : ℝ => -1)
    (v := fun u => g (r*u)) (v' := fun u => r * g' (r*u))
    (by fun_prop) (hgc.comp (continuous_const.mul continuous_id)).continuousOn
    (fun u _ => by
      convert (hasDerivAt_id u).const_sub (1/2 : ℝ) using 1 <;> simp [id])
    (fun u _ => scaled_hasDerivAt g g' hg r u)
    (continuous_const.intervalIntegrable _ _) (hc'.intervalIntegrable _ _)
  have hfun : (fun u : ℝ => (1/2-u) * (r * g' (r*u))) =
      (fun u => r * ((1/2-u) * g' (r*u))) := by funext u; ring
  rw [hfun, intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul] at h
  norm_num [mul_div_assoc, mul_neg] at h
  apply (div_eq_iff hr).2
  norm_num at h ⊢
  rw [show -(r * (1/2 : ℝ)) = -r/2 by ring] at h
  nlinarith [h]

private theorem right_endpoint_identity (g g' : ℝ → ℝ)
    (hg : ∀ x, HasDerivAt g (g' x) x) (hc : Continuous g') (r : ℝ) (hr : r ≠ 0) :
    (g (r/2) - ∫ u in (-1/2 : ℝ)..(1/2 : ℝ), g (r*u)) / r =
      ∫ u in (-1/2 : ℝ)..(1/2 : ℝ), (u+1/2) * g' (r*u) := by
  have hgc : Continuous g :=
    (show Differentiable ℝ g from fun x => (hg x).differentiableAt).continuous
  have hc' : Continuous (fun u => r * g' (r*u)) :=
    continuous_const.mul (hc.comp (continuous_const.mul continuous_id))
  have h := intervalIntegral.integral_mul_deriv_eq_deriv_mul_of_hasDerivAt
    (a := (-1/2 : ℝ)) (b := (1/2 : ℝ))
    (u := fun u : ℝ => u+1/2) (u' := fun _ : ℝ => 1)
    (v := fun u => g (r*u)) (v' := fun u => r * g' (r*u))
    (by fun_prop) (hgc.comp (continuous_const.mul continuous_id)).continuousOn
    (fun u _ => by simpa [id] using (hasDerivAt_id u).add_const (1/2 : ℝ))
    (fun u _ => scaled_hasDerivAt g g' hg r u)
    (continuous_const.intervalIntegrable _ _) (hc'.intervalIntegrable _ _)
  have hfun : (fun u : ℝ => (u+1/2) * (r * g' (r*u))) =
      (fun u => r * ((u+1/2) * g' (r*u))) := by funext u; ring
  rw [hfun, intervalIntegral.integral_const_mul] at h
  norm_num [mul_div_assoc, mul_neg] at h
  apply (div_eq_iff hr).2
  norm_num at h ⊢
  rw [show r * (1/2 : ℝ) = r/2 by ring] at h
  nlinarith [h]

private theorem peano_identity (f f1 f2 f3 : ℝ → ℝ)
    (h0 : ∀ x, HasDerivAt f (f1 x) x)
    (h1 : ∀ x, HasDerivAt f1 (f2 x) x)
    (h2 : ∀ x, HasDerivAt f2 (f3 x) x) (hc3 : Continuous f3)
    (r : ℝ) (hr : r ≠ 0) :
    (12 / r ^ 2) * ((f1 (r/2) + f1 (-r/2))/2 - (f (r/2) - f (-r/2))/r) =
      6 * ∫ u in (-1/2 : ℝ)..(1/2 : ℝ), (1/4-u^2) * f3 (r*u) := by
  let A : ℝ → ℝ := fun u =>
    (1/4-u^2)*f2 (r*u) + (2/r)*u*f1 (r*u) - (2/r^2)*f (r*u)
  have hA (u : ℝ) : HasDerivAt A (r*((1/4-u^2)*f3 (r*u))) u := by
    have hru : HasDerivAt (fun u : ℝ => r*u) r u := by
      simpa [id] using (hasDerivAt_id u).const_mul r
    have hK : HasDerivAt (fun u : ℝ => 1/4-u^2) (-2*u) u := by
      convert ((hasDerivAt_id u).pow 2).const_sub (1/4 : ℝ) using 1 <;> simp [id]
    have hU : HasDerivAt (fun u : ℝ => (2/r)*u) (2/r) u := by
      simpa [id] using (hasDerivAt_id u).const_mul (2/r)
    have h := ((hK.mul ((h2 (r*u)).comp u hru)).add
      (hU.mul ((h1 (r*u)).comp u hru))).sub (((h0 (r*u)).comp u hru).const_mul (2/r^2))
    convert h using 1
    · rfl
    · dsimp [Function.comp_def]
      field_simp [hr]
      ring
  have hcont : Continuous (fun u : ℝ => r*((1/4-u^2)*f3 (r*u))) :=
    (continuous_const.sub (continuous_id.pow 2)).mul
      (hc3.comp (continuous_const.mul continuous_id)) |>.const_mul r
  have hft := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun u _ => hA u) (hcont.intervalIntegrable (-1/2 : ℝ) (1/2 : ℝ))
  rw [intervalIntegral.integral_const_mul] at hft
  norm_num [A, mul_div_assoc, mul_neg] at hft
  field_simp [hr] at hft ⊢
  nlinarith [hft]

private theorem iterated_hasDerivAt {F : ℝ → ℝ} {n : ℕ}
    (hF : ContDiff ℝ n F) (k : ℕ) (hk : k < n) (x : ℝ) :
    HasDerivAt (iteratedDeriv k F) (iteratedDeriv (k+1) F x) x := by
  rw [iteratedDeriv_succ]
  exact (hF.differentiable_iteratedDeriv k (by exact_mod_cast hk) x).hasDerivAt

/-- The x-derivative divided difference is the exact average second derivative. -/
theorem pin3_integral (F : ℝ → ℝ) (hF : ContDiff ℝ 3 F) (r : ℝ) (hr : r ≠ 0) :
    pin3 F r = ∫ u in (-1/2 : ℝ)..(1/2 : ℝ), iteratedDeriv 2 F (r*u) := by
  simpa [pin3, iteratedDeriv_one] using average_identity (iteratedDeriv 1 F)
    (iteratedDeriv 2 F) (iterated_hasDerivAt hF 1 (by norm_num))
    (hF.continuous_iteratedDeriv 2 (by norm_num)) r hr

/-- H is the y-derivative curve: the pin uses H itself, not its derivative. -/
theorem pin4_integral (H : ℝ → ℝ) (hH : ContDiff ℝ 2 H) (r : ℝ) (hr : r ≠ 0) :
    pin4 H r = ∫ u in (-1/2 : ℝ)..(1/2 : ℝ), deriv H (r*u) := by
  simpa [pin4, iteratedDeriv_one] using average_identity (iteratedDeriv 0 H)
    (iteratedDeriv 1 H) (iterated_hasDerivAt hH 0 (by norm_num))
    (hH.continuous_iteratedDeriv 1 (by norm_num)) r hr

/-- The exact cubic confluent pin is its normalized Peano-kernel integral. -/
theorem pin5_integral (F : ℝ → ℝ) (hF : ContDiff ℝ 3 F) (r : ℝ) (hr : r ≠ 0) :
    pin5 F r = 6 * ∫ u in (-1/2 : ℝ)..(1/2 : ℝ),
      (1/4-u^2) * iteratedDeriv 3 F (r*u) := by
  simpa [pin5, iteratedDeriv_one] using peano_identity (iteratedDeriv 0 F)
    (iteratedDeriv 1 F) (iteratedDeriv 2 F) (iteratedDeriv 3 F)
    (iterated_hasDerivAt hF 0 (by norm_num))
    (iterated_hasDerivAt hF 1 (by norm_num))
    (iterated_hasDerivAt hF 2 (by norm_num))
    (hF.continuous_iteratedDeriv 3 (by norm_num)) r hr

theorem adjusted_second_left (F : ℝ → ℝ) (hF : ContDiff ℝ 3 F) (r : ℝ) (hr : r ≠ 0) :
    (iteratedDeriv 2 F (-r/2) - pin3 F r) / r =
      -(∫ u in (-1/2 : ℝ)..(1/2 : ℝ), (1/2-u) * iteratedDeriv 3 F (r*u)) := by
  rw [pin3_integral F hF r hr]
  exact left_endpoint_identity (iteratedDeriv 2 F) (iteratedDeriv 3 F)
    (iterated_hasDerivAt hF 2 (by norm_num))
    (hF.continuous_iteratedDeriv 3 (by norm_num)) r hr

theorem adjusted_second_right (F : ℝ → ℝ) (hF : ContDiff ℝ 3 F) (r : ℝ) (hr : r ≠ 0) :
    (iteratedDeriv 2 F (r/2) - pin3 F r) / r =
      ∫ u in (-1/2 : ℝ)..(1/2 : ℝ), (u+1/2) * iteratedDeriv 3 F (r*u) := by
  rw [pin3_integral F hF r hr]
  exact right_endpoint_identity (iteratedDeriv 2 F) (iteratedDeriv 3 F)
    (iterated_hasDerivAt hF 2 (by norm_num))
    (hF.continuous_iteratedDeriv 3 (by norm_num)) r hr

theorem adjusted_transverse_left (H : ℝ → ℝ) (hH : ContDiff ℝ 2 H) (r : ℝ) (hr : r ≠ 0) :
    (deriv H (-r/2) - pin4 H r) / r =
      -(∫ u in (-1/2 : ℝ)..(1/2 : ℝ), (1/2-u) * iteratedDeriv 2 H (r*u)) := by
  rw [pin4_integral H hH r hr]
  simpa [iteratedDeriv_one] using left_endpoint_identity (iteratedDeriv 1 H)
    (iteratedDeriv 2 H) (iterated_hasDerivAt hH 1 (by norm_num))
    (hH.continuous_iteratedDeriv 2 (by norm_num)) r hr

theorem adjusted_transverse_right (H : ℝ → ℝ) (hH : ContDiff ℝ 2 H) (r : ℝ) (hr : r ≠ 0) :
    (deriv H (r/2) - pin4 H r) / r =
      ∫ u in (-1/2 : ℝ)..(1/2 : ℝ), (u+1/2) * iteratedDeriv 2 H (r*u) := by
  rw [pin4_integral H hH r hr]
  simpa [iteratedDeriv_one] using right_endpoint_identity (iteratedDeriv 1 H)
    (iteratedDeriv 2 H) (iterated_hasDerivAt hH 1 (by norm_num))
    (hH.continuous_iteratedDeriv 2 (by norm_num)) r hr

end I4ConfluentIntegralIdentities
