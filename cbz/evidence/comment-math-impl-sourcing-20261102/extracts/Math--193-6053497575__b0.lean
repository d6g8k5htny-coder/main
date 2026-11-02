import SourceC4
import Mathlib.Topology.Algebra.Module.Multilinear.Topology

/-! Actual Fréchet derivatives of the unchanged sourceField, on one shared sample event.
Finite jet Gaussian laws, full rank and prescribed-pin regression are separate. -/

noncomputable section
open I4Foundation I4Weights I4Source I4C4 MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace I4Jets

def modeJetCoefficient (d : ℕ) (L : ℝ) (m : SourceMode d) (j : ℕ)
    (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) : ℝ :=
  (iteratedFDeriv ℝ j (modeCoefficient d L m) x) v

def sourceJet (d : ℕ) (L : ℝ) (j : ℕ) (ω : SourceMode d → ℝ)
    (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) : ℝ :=
  (iteratedFDeriv ℝ j (sourceField d L ω) x) v

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

theorem modeJetCoefficient_bound (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (m : SourceMode d) (j : ℕ) (hj : j ≤ 4)
    (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) :
    |modeJetCoefficient d L m j x v| ≤ modeMajorant d L m * ∏ i, ‖v i‖ := by
  change ‖(iteratedFDeriv ℝ j (modeCoefficient d L m) x) v‖ ≤ _
  calc
    _ ≤ ‖iteratedFDeriv ℝ j (modeCoefficient d L m) x‖ * ∏ i, ‖v i‖ :=
      (iteratedFDeriv ℝ j (modeCoefficient d L m) x).le_opNorm v
    _ ≤ _ := mul_le_mul_of_nonneg_right
      (modeCoefficient_iteratedFDeriv_bound d hL m j hj x)
      (Finset.prod_nonneg (fun i _ => norm_nonneg (v i)))

theorem modeJetCoefficient_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (j : ℕ) (hj : j ≤ 4) (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)) :
    Summable (fun m => modeJetCoefficient d L m j x v) := by
  apply ((modeMajorant_summable d hL).mul_right (∏ i, ‖v i‖)).of_norm_bounded
  intro m
  simpa only [Real.norm_eq_abs] using modeJetCoefficient_bound d hL m j hj x v

theorem sourceJet_eq_series_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d, ∀ (j : ℕ), j ≤ 4 →
      ∀ (x : Fin d → ℝ) (v : Fin j → (Fin d → ℝ)),
        sourceJet d L j ω x v =
          scalarSeries (fun m => modeJetCoefficient d L m j x v) ω := by
  filter_upwards [sourceMajorant_ae d hL] with ω hω
  intro j hj x v
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

theorem sourceJet_zero (d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ)
    (x : Fin d → ℝ) (v : Fin 0 → (Fin d → ℝ)) :
    sourceJet d L 0 ω x v = sourceField d L ω x := by
  exact iteratedFDeriv_zero_apply v

end I4Jets

