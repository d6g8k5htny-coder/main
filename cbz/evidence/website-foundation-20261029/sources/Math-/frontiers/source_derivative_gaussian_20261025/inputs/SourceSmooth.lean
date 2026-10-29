import SourceC4
import Mathlib.Data.Nat.Choose.Sum

/-!
# All-order smoothness of the actual infinite Gaussian Fourier source

This module extends the source-derived polynomial majorant from order four to
every natural order. It uses the same SourceMode, sourceLaw and sourceField as
the qualified F7 dependency; no finite-mode approximation or new field is used.
The good sample predicate is a single countable intersection of explicit
coefficient summability conditions. Its measurability and probability-law
restriction are proved in SourceSmoothCarrier.

This is a C261 Stage A prerequisite. It does not identify the spectral covariance
with Lifetime's theta sum, construct a conditional kernel, or discharge a
weighted elder, Morse, count/density, or universality conclusion.
-/

noncomputable section
open I4Foundation I4Weights I4Source I4C4 MeasureTheory ProbabilityTheory
open scoped BigOperators ENNReal

namespace I4Smooth

def orderCoordinateBound (n : ℕ) (a c : ℝ) (k : ℤ) : ℝ :=
  decay a k * (1 + |c * (k : ℝ)|) ^ n

def orderProductBound (n d : ℕ) (a c : ℝ) (k : Fin d → ℤ) : ℝ :=
  ∏ i, orderCoordinateBound n a c (k i)

def orderPairedBound (n d : ℕ) (L : ℝ) (k : Fin d → ℤ) : ℝ :=
  Real.sqrt (2 / sourceZ d L) * orderProductBound n d (sourceA L) (sourceC L) k

def orderModeBound (n d : ℕ) (L : ℝ) (m : SourceMode d) : ℝ :=
  match m with
  | none => Real.sqrt (1 / sourceZ d L)
  | some (k, _) => orderPairedBound n d L k

theorem orderCoordinateBound_summable (n : ℕ) {a c : ℝ} (ha : 0 < a) :
    Summable (orderCoordinateBound n a c) := by
  classical
  let f (r : ℕ) (k : ℤ) : ℝ :=
    ((n.choose r : ℝ) * |c| ^ (n - r)) * (decay a k * |(k : ℝ)| ^ (n - r))
  have hf (r : ℕ) : Summable (f r) :=
    (summable_decay_pow ha (n - r)).mul_left _
  have hsum (s : Finset ℕ) : Summable (fun k : ℤ => ∑ r ∈ s, f r k) := by
    induction s using Finset.induction_on with
    | empty => simpa using (summable_zero : Summable (fun _ : ℤ => (0 : ℝ)))
    | @insert r s hr ih =>
      simpa only [Finset.sum_insert hr] using (hf r).add ih
  convert hsum (Finset.range (n + 1)) using 1
  funext k
  unfold orderCoordinateBound
  rw [add_pow, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro r _
  simp only [one_pow, one_mul, abs_mul, mul_pow, f]
  ring

theorem orderProductBound_summable (n d : ℕ) {a c : ℝ} (ha : 0 < a) :
    Summable (orderProductBound n d a c) := by
  exact finite_product_summable (orderCoordinateBound_summable n ha)
    (fun k => by unfold orderCoordinateBound decay; positivity) d

theorem orderPairedBound_summable (n d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    Summable (orderPairedBound n d L) := by
  exact (orderProductBound_summable n d (sourceA_pos hL)).mul_left _

theorem orderPairedBound_eq_amplitude (n d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (k : Fin d → ℤ) :
    orderPairedBound n d L k =
      Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
        ∏ i, (1 + |sourceC L * (k i : ℝ)|) ^ n := by
  have hsqrt : Real.sqrt (sourceWeight d L k) = productDecay d (sourceA L) k := by
    have hw : sourceWeight d L k = (productDecay d (sourceA L) k) ^ 2 := by
      rw [sourceWeight, productDecay_eq_exp, productDecay_eq_exp, pow_two, ← Real.exp_add]
      congr 1
      ring
    rw [hw, Real.sqrt_sq_eq_abs, abs_of_nonneg]
    unfold productDecay decay
    positivity
  have hamp : Real.sqrt (2 * sourceWeight d L k / sourceZ d L) =
      Real.sqrt (2 / sourceZ d L) * productDecay d (sourceA L) k := by
    have he : 2 * sourceWeight d L k / sourceZ d L =
        (2 / sourceZ d L) * sourceWeight d L k := by ring
    rw [he, Real.sqrt_mul (by have := sourceZ_pos d hL; positivity), hsqrt]
  have hp : orderProductBound n d (sourceA L) (sourceC L) k =
      productDecay d (sourceA L) k * ∏ i, (1 + |sourceC L * (k i : ℝ)|) ^ n := by
    exact Finset.prod_mul_distrib
  rw [orderPairedBound, hp, hamp]
  ring

theorem orderModeBound_nonneg (n d : ℕ) (L : ℝ) (m : SourceMode d) :
    0 ≤ orderModeBound n d L m := by
  cases m with
  | none => exact Real.sqrt_nonneg _
  | some p =>
    unfold orderModeBound orderPairedBound orderProductBound orderCoordinateBound decay
    positivity

theorem orderModeBound_summable (n d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    Summable (orderModeBound n d L) := by
  classical
  have hs : Summable (fun k : ↥(halfLattice d) => orderPairedBound n d L k) :=
    (orderPairedBound_summable n d hL).subtype (halfLattice d)
  have hb : Summable (fun _ : Bool => (1 : ℝ)) := (hasSum_fintype _).summable
  have hp : Summable (fun p : ↥(halfLattice d) × Bool => orderPairedBound n d L p.1) := by
    simpa only [mul_one, Function.comp_def] using hs.mul_of_nonneg hb
      (fun k => orderModeBound_nonneg n d L (some (k, false))) (fun _ => zero_le_one)
  let e := Equiv.optionEquivSumPUnit.{0, 0} (↥(halfLattice d) × Bool)
  apply e.symm.summable_iff.mp
  apply Summable.sum (fun q => orderModeBound n d L (e.symm q))
  · simpa [e, Function.comp_def, orderModeBound] using hp
  · exact (hasSum_fintype _).summable

theorem orderModeBound_four (d : ℕ) (L : ℝ) (m : SourceMode d) :
    orderModeBound 4 d L m = modeMajorant d L m := by
  cases m <;> rfl

theorem modeCoefficient_iteratedFDeriv_order_bound (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (m : SourceMode d) (j : ℕ) (x : Fin d → ℝ) :
    ‖iteratedFDeriv ℝ j (modeCoefficient d L m) x‖ ≤ orderModeBound j d L m := by
  classical
  cases m with
  | none =>
    cases j with
    | zero =>
      simpa only [norm_iteratedFDeriv_zero, Real.norm_eq_abs, orderModeBound, modeMajorant] using
        modeCoefficient_bound d hL none x
    | succ j =>
      change ‖iteratedFDeriv ℝ (j + 1)
        (fun _ : Fin d → ℝ => Real.sqrt (1 / sourceZ d L)) x‖ ≤ Real.sqrt (1 / sourceZ d L)
      rw [iteratedFDeriv_succ_const]
      simp only [Pi.zero_apply, norm_zero]
      exact Real.sqrt_nonneg _
  | some p =>
    rcases p with ⟨k, b⟩
    let A := Real.sqrt (2 * sourceWeight d L k / sourceZ d L)
    let P := ∏ i, (1 + |sourceC L * (k.1 i : ℝ)|)
    let g : ℝ → ℝ := if b then Real.sin else Real.cos
    let ell := phaseCLM d L k
    have hg : ContDiff ℝ j g := by
      cases b <;> simp only [g, Bool.false_eq_true, ite_false, ite_true] <;> fun_prop
    have hgc : ContDiff ℝ j (g ∘ ell) := hg.comp_continuousLinearMap
    have hgn : ‖iteratedFDeriv ℝ j g (ell x)‖ ≤ 1 := by
      cases b with
      | false =>
        simpa only [g, Bool.false_eq_true, ite_false,
          norm_iteratedFDeriv_eq_norm_iteratedDeriv, Real.norm_eq_abs] using
          Real.abs_iteratedDeriv_cos_le_one j (ell x)
      | true =>
        simpa only [g, ite_true, norm_iteratedFDeriv_eq_norm_iteratedDeriv,
          Real.norm_eq_abs] using Real.abs_iteratedDeriv_sin_le_one j (ell x)
    have hl : ‖ell‖ ≤ P := by
      calc
        ‖ell‖ ≤ ∑ i, |sourceC L * (k.1 i : ℝ)| := phaseCLM_norm_le d L k
        _ ≤ 1 + ∑ i, |sourceC L * (k.1 i : ℝ)| := by linarith
        _ ≤ P := frequency_l1_le_product d _ (fun i => abs_nonneg _)
    have hpow : ‖ell‖ ^ j ≤ P ^ j := by gcongr
    have hder : ‖iteratedFDeriv ℝ j (g ∘ ell) x‖ ≤ ‖ell‖ ^ j := by
      rw [ell.iteratedFDeriv_comp_right hg x le_rfl]
      calc
        _ ≤ ‖iteratedFDeriv ℝ j g (ell x)‖ * ∏ _i : Fin j, ‖ell‖ :=
          (iteratedFDeriv ℝ j g (ell x)).norm_compContinuousLinearMap_le (fun _ => ell)
        _ ≤ 1 * ∏ _i : Fin j, ‖ell‖ :=
          mul_le_mul_of_nonneg_right hgn (Finset.prod_nonneg (fun _ _ => norm_nonneg _))
        _ = ‖ell‖ ^ j := by simp
    have heq : modeCoefficient d L (some (k, b)) = A • (g ∘ ell) := by
      funext y
      cases b <;> simp [modeCoefficient, A, g, Function.comp_apply, Pi.smul_apply,
        smul_eq_mul, ell, phaseCLM_apply]
    rw [heq, iteratedFDeriv_const_smul_apply hgc.contDiffAt, norm_smul,
      Real.norm_eq_abs, abs_of_nonneg (Real.sqrt_nonneg _)]
    calc
      _ ≤ A * P ^ j := mul_le_mul_of_nonneg_left (hder.trans hpow) (Real.sqrt_nonneg _)
      _ = orderModeBound j d L (some (k, b)) := by
        rw [orderModeBound, orderPairedBound_eq_amplitude j d hL k]
        simp only [A, P, Finset.prod_pow]

/-- One explicit coefficient event, shared by every derivative order. -/
def goodSample (d : ℕ) (L : ℝ) (ω : SourceMode d → ℝ) : Prop :=
  ∀ n : ℕ, Summable (fun m => orderModeBound n d L m * |ω m|)

theorem goodSample_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d, goodSample d L ω := by
  apply ae_all_iff.mpr
  intro n
  exact weighted_abs_summable_ae _ (orderModeBound_nonneg n d L)
    (orderModeBound_summable n d hL)

theorem sourceField_smooth_of_goodSample (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (ω : SourceMode d → ℝ) (hω : goodSample d L ω) :
    ∀ n : ℕ, ContDiff ℝ n (sourceField d L ω) := by
  intro n
  have hc (m : SourceMode d) : ContDiff ℝ n (modeCoefficient d L m) := by
    cases m with
    | none =>
      change ContDiff ℝ n (fun _ : Fin d → ℝ => Real.sqrt (1 / sourceZ d L))
      fun_prop
    | some p =>
      rcases p with ⟨k, b⟩
      change ContDiff ℝ n (fun x : Fin d → ℝ =>
        Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
          (if b then Real.sin (phase d L k x) else Real.cos (phase d L k x)))
      cases b <;> simp [phase] <;> fun_prop
  change ContDiff ℝ n (fun x => ∑' m, modeCoefficient d L m x * ω m)
  apply contDiff_tsum (v := fun j m => orderModeBound j d L m * |ω m|)
  · intro m
    exact (hc m).mul contDiff_const
  · intro j _
    exact hω j
  · intro j m x hj
    have hcj : ContDiff ℝ j (modeCoefficient d L m) :=
      (hc m).of_le (by exact_mod_cast hj)
    have heq : (fun y => modeCoefficient d L m y * ω m) =
        ω m • modeCoefficient d L m := by
      funext y
      simp [smul_eq_mul, mul_comm]
    rw [heq, iteratedFDeriv_const_smul_apply hcj.contDiffAt, norm_smul, Real.norm_eq_abs]
    simpa only [mul_comm] using mul_le_mul_of_nonneg_left
      (modeCoefficient_iteratedFDeriv_order_bound d hL m j x) (abs_nonneg (ω m))

theorem sourceField_smooth_ae (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    ∀ᵐ ω ∂sourceLaw d, ∀ n : ℕ, ContDiff ℝ n (sourceField d L ω) := by
  filter_upwards [goodSample_ae d hL] with ω hω
  exact sourceField_smooth_of_goodSample d hL ω hω

/-- The carrier depends only on the coefficient tail; finite coordinate changes
cannot destroy any derivative-order summability condition. -/
theorem goodSample_congr_cofinite (d : ℕ) (L : ℝ)
    (ω η : SourceMode d → ℝ) (hωη : ω =ᶠ[Filter.cofinite] η) :
    goodSample d L ω ↔ goodSample d L η := by
  constructor
  · intro hω n
    apply (hω n).congr_cofinite
    filter_upwards [hωη] with m hm
    rw [hm]
  · intro hη n
    apply (hη n).congr_cofinite
    filter_upwards [hωη.symm] with m hm
    rw [hm]

theorem goodSample_finite_change_iff (d : ℕ) (L : ℝ)
    (ω η : SourceMode d → ℝ) (s : Finset (SourceMode d))
    (hωη : ∀ m ∉ s, ω m = η m) :
    goodSample d L ω ↔ goodSample d L η := by
  apply goodSample_congr_cofinite d L ω η
  apply Filter.eventually_cofinite.mpr
  apply s.finite_toSet.subset
  intro m hm
  by_contra hms
  exact hm (hωη m hms)

end I4Smooth
