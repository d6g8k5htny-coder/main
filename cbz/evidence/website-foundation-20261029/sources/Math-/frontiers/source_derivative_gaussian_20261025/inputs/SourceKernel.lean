import SourceLattice
import Mathlib.Analysis.Normed.Ring.InfiniteSum

noncomputable section
open I4Foundation I4Weights I4Source I4Lattice MeasureTheory
open scoped BigOperators
namespace I4Kernel

def frequencyEquiv (d : ℕ) : SourceMode d ≃ (Fin d → ℤ) :=
  Equiv.ofBijective (latticeFrequency d) (latticeFrequency_bijective d)

/-- The prescribed normalized full-lattice real Fourier kernel; image/Poisson linkage is separate. -/
def sourceSpectralKernel (d : ℕ) (L : ℝ) (z : Fin d → ℝ) : ℝ :=
  ∑' k : Fin d → ℤ, sourceWeight d L k / sourceZ d L * Real.cos (phase d L k z)

private theorem tsum_mode (d : ℕ) (f : SourceMode d → ℝ) (hf : Summable f) :
    (∑' m, f m) = f none + ∑' k : ↥(halfLattice d),
      (f (some (k,false)) + f (some (k,true))) := by
  classical
  let e := Equiv.optionEquivSumPUnit.{0,0} (↥(halfLattice d) × Bool)
  have hp : Summable (fun p : ↥(halfLattice d) × Bool => f (some p)) :=
    hf.comp_injective (Option.some_injective (↥(halfLattice d) × Bool))
  calc
    (∑' m, f m) = ∑' q, f (e.symm q) := (e.symm.tsum_eq f).symm
    _ = (∑' p : ↥(halfLattice d) × Bool, f (some p)) + f none := by
      rw [Summable.tsum_sum (by simpa [e, Function.comp_def] using hp)
        (hasSum_fintype _).summable]
      simp [e]
    _ = _ := by
      rw [hp.tsum_prod]
      simp_rw [tsum_bool]
      ring

theorem modeProduct_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x y : Fin d → ℝ) :
    Summable (fun m => modeCoefficient d L m x * modeCoefficient d L m y) := by
  have hc (z : Fin d → ℝ) : Summable (fun m => modeCoefficient d L m z) := by
    apply Summable.of_norm_bounded (modeMajorant_summable d hL)
    intro m
    simpa only [Real.norm_eq_abs] using modeCoefficient_bound d hL m z
  have hp : Summable (fun p : SourceMode d × SourceMode d =>
      modeCoefficient d L p.1 x * modeCoefficient d L p.2 y) :=
    summable_mul_of_summable_norm (hc x).norm (hc y).norm
  simpa only [Function.comp_def] using hp.comp_injective
    (show Function.Injective (fun m : SourceMode d => (m,m)) from
      fun a b h => congrArg Prod.fst h)

theorem spectralKernel_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0) (z : Fin d → ℝ) :
    Summable (fun k : Fin d → ℤ =>
      sourceWeight d L k / sourceZ d L * Real.cos (phase d L k z)) := by
  have hz := sourceZ_pos d hL
  apply Summable.of_norm_bounded ((sourceWeight_summable d hL).div_const (sourceZ d L))
  intro k
  have hq : 0 ≤ sourceWeight d L k / sourceZ d L := by
    rw [sourceWeight_eq_physical]
    exact div_nonneg (Real.exp_pos _).le hz.le
  rw [norm_mul, Real.norm_eq_abs, abs_of_nonneg hq, Real.norm_eq_abs]
  simpa only [mul_one] using mul_le_mul_of_nonneg_left (Real.abs_cos_le_one _) hq

theorem coefficientKernel_eq_fullLattice (d : ℕ) {L : ℝ} (hL : L ≠ 0) (x y : Fin d → ℝ) :
    (∑' m, modeCoefficient d L m x * modeCoefficient d L m y) =
      sourceSpectralKernel d L (x - y) := by
  classical
  let g : (Fin d → ℤ) → ℝ := fun k =>
    sourceWeight d L k / sourceZ d L * Real.cos (phase d L k (x - y))
  have hg : Summable g := spectralKernel_summable d hL (x - y)
  have hneg (k : Fin d → ℤ) : g (-k) = g k := by
    have hp : phase d L (-k) (x - y) = -phase d L k (x - y) := by
      simp [phase, Finset.sum_neg_distrib]
    simp only [g, sourceWeight_neg, hp, Real.cos_neg]
  have hzero : g 0 = 1 / sourceZ d L := by
    simp [g, sourceWeight_eq_physical, phase]
  have hc0 : modeCoefficient d L none x * modeCoefficient d L none y = 1 / sourceZ d L := by
    simp only [modeCoefficient]
    exact Real.mul_self_sqrt (one_div_nonneg.mpr (sourceZ_pos d hL).le)
  have hpair (k : ↥(halfLattice d)) :
      modeCoefficient d L (some (k,false)) x * modeCoefficient d L (some (k,false)) y +
      modeCoefficient d L (some (k,true)) x * modeCoefficient d L (some (k,true)) y = 2 * g k := by
    have h := pairedCoefficient_kernel d hL k x y
    rw [Fintype.sum_bool] at h
    calc
      _ = modeCoefficient d L (some (k,true)) x * modeCoefficient d L (some (k,true)) y +
        modeCoefficient d L (some (k,false)) x * modeCoefficient d L (some (k,false)) y := add_comm _ _
      _ = _ := h
      _ = 2 * g k := by dsimp [g]; ring
  have hleft : (∑' m, modeCoefficient d L m x * modeCoefficient d L m y) =
      1 / sourceZ d L + ∑' k : ↥(halfLattice d), 2 * g k := by
    rw [tsum_mode d _ (modeProduct_summable d hL x y), hc0]
    congr 1
    exact tsum_congr hpair
  have hm : Summable (fun m => g (frequencyEquiv d m)) := (frequencyEquiv d).summable_iff.mpr hg
  have hright : sourceSpectralKernel d L (x - y) =
      1 / sourceZ d L + ∑' k : ↥(halfLattice d), 2 * g k := by
    change (∑' k, g k) = _
    rw [← (frequencyEquiv d).tsum_eq g, tsum_mode d _ hm]
    change g 0 + (∑' k : ↥(halfLattice d), (g k.1 + g (-k.1))) = _
    rw [hzero]
    congr 1
    apply tsum_congr
    intro k
    rw [hneg]
    ring
  exact hleft.trans hright.symm

theorem spectralKernel_zero (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    sourceSpectralKernel d L 0 = 1 := by
  unfold sourceSpectralKernel
  simp only [phase, Pi.zero_apply, mul_zero, Finset.sum_const_zero, Real.cos_zero, mul_one]
  rw [tsum_div_const]
  change sourceZ d L / sourceZ d L = 1
  exact div_self (sourceZ_pos d hL).ne'

end I4Kernel
