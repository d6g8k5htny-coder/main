import Mathlib.NumberTheory.ModularForms.JacobiTheta.TwoVariable
import Mathlib.Analysis.Normed.Ring.InfiniteSum
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Algebra.BigOperators.Fin

/-! Concrete square-root Gaussian spectral majorants for the required I4 bridge.
The product majorant dominates the l1 frequency norm for the default Pi sup norm.
This file alone proves neither a Gaussian field law nor sample smoothness. -/

noncomputable section
open Real
open scoped BigOperators
namespace I4Weights

def decay (a : ℝ) (k : ℤ) : ℝ := exp (-a * (k : ℝ) ^ 2)
def coordinateBound (a c : ℝ) (k : ℤ) : ℝ :=
  decay a k * (1 + |c * (k : ℝ)|) ^ 4
def productDecay (d : ℕ) (a : ℝ) (k : Fin d → ℤ) : ℝ :=
  ∏ i, decay a (k i)
def productBound (d : ℕ) (a c : ℝ) (k : Fin d → ℤ) : ℝ :=
  ∏ i, coordinateBound a c (k i)
def sourceA (L : ℝ) : ℝ := π ^ 2 / L ^ 2
def sourceC (L : ℝ) : ℝ := 2 * π / L
def sourceWeight (d : ℕ) (L : ℝ) (k : Fin d → ℤ) : ℝ :=
  productDecay d (2 * sourceA L) k
def sourceZ (d : ℕ) (L : ℝ) : ℝ := ∑' k, sourceWeight d L k
def pairedBound (d : ℕ) (L : ℝ) (k : Fin d → ℤ) : ℝ :=
  Real.sqrt (2 / sourceZ d L) * productBound d (sourceA L) (sourceC L) k

theorem summable_decay_pow {a : ℝ} (ha : 0 < a) (n : ℕ) :
    Summable (fun k : ℤ => decay a k * |(k : ℝ)| ^ n) := by
  have hb := summable_pow_mul_jacobiTheta₂_term_bound 0
    (show 0 < a / π by positivity) n
  have he (k : ℤ) : -π * (a / π * (k : ℝ) ^ 2 - 2 * 0 * |(k : ℝ)|) =
      -a * (k : ℝ) ^ 2 := by
    field_simp
    ring
  simp_rw [Int.cast_abs, he] at hb
  simpa [decay, mul_comm] using hb

theorem coordinateBound_summable {a c : ℝ} (ha : 0 < a) :
    Summable (coordinateBound a c) := by
  have hs := summable_decay_pow ha
  convert ((((hs 0).add ((hs 1).mul_left (4 * |c|))).add
    ((hs 2).mul_left (6 * |c| ^ 2))).add
    ((hs 3).mul_left (4 * |c| ^ 3))).add ((hs 4).mul_left (|c| ^ 4)) using 1
  funext k
  simp only [coordinateBound, abs_mul]
  ring

set_option maxHeartbeats 800000 in
theorem finite_product_summable {f : ℤ → ℝ} (hf : Summable f) (hn : ∀ k, 0 ≤ f k)
    (d : ℕ) : Summable (fun k : Fin d → ℤ => ∏ i, f (k i)) := by
  classical
  induction d with
  | zero => exact (hasSum_fintype _).summable
  | succ d ih =>
    have hp : Summable (fun p : ℤ × (Fin d → ℤ) => f p.1 * ∏ i, f (p.2 i)) :=
      Summable.mul_of_nonneg (ι := ℤ) (ι' := Fin d → ℤ)
        (f := f) (g := fun k : Fin d → ℤ => ∏ i, f (k i))
        hf ih hn (fun k => Finset.prod_nonneg (fun i _ => hn (k i)))
    have heq :
        ((fun k : Fin (d + 1) → ℤ => ∏ i, f (k i)) ∘
          (Fin.consEquiv (fun _ : Fin (d + 1) => ℤ))) =
        (fun p : ℤ × (Fin d → ℤ) => f p.1 * ∏ i, f (p.2 i)) := by
      funext p
      simp only [Function.comp_apply, Fin.consEquiv_apply, Fin.prod_univ_succ,
        Fin.cons_zero, Fin.cons_succ]
    apply (Fin.consEquiv (fun _ : Fin (d + 1) => ℤ)).summable_iff.mp
    rw [heq]
    exact hp

theorem productBound_summable {a c : ℝ} (ha : 0 < a) (d : ℕ) :
    Summable (productBound d a c) := by
  exact finite_product_summable (coordinateBound_summable ha)
    (fun k => by unfold coordinateBound decay; positivity) d

theorem productDecay_eq_exp (d : ℕ) (a : ℝ) (k : Fin d → ℤ) :
    productDecay d a k = exp (-a * ∑ i, (k i : ℝ) ^ 2) := by
  unfold productDecay decay
  rw [← Real.exp_sum, ← Finset.mul_sum]

theorem frequency_l1_le_product (d : ℕ) (v : Fin d → ℝ) (hv : ∀ i, 0 ≤ v i) :
    1 + ∑ i, v i ≤ ∏ i, (1 + v i) := by
  classical
  have H (s : Finset (Fin d)) : 1 + ∑ i ∈ s, v i ≤ ∏ i ∈ s, (1 + v i) := by
    induction s using Finset.induction_on with
    | empty => simp
    | @insert i s hi ih =>
      rw [Finset.sum_insert hi, Finset.prod_insert hi]
      have hsum : 0 ≤ ∑ j ∈ s, v j := Finset.sum_nonneg (fun j _ => hv j)
      calc
        1 + (v i + ∑ j ∈ s, v j) ≤ (1 + v i) * (1 + ∑ j ∈ s, v j) := by
          nlinarith [mul_nonneg (hv i) hsum]
        _ ≤ (1 + v i) * ∏ j ∈ s, (1 + v j) :=
          mul_le_mul_of_nonneg_left ih (by linarith [hv i])
  exact H Finset.univ

theorem sourceA_pos {L : ℝ} (hL : L ≠ 0) : 0 < sourceA L := by
  unfold sourceA
  positivity

theorem sourceWeight_eq_physical (d : ℕ) (L : ℝ) (k : Fin d → ℤ) :
    sourceWeight d L k = exp (-(∑ i, (sourceC L * (k i : ℝ)) ^ 2) / 2) := by
  have hc : sourceC L ^ 2 = 4 * sourceA L := by
    unfold sourceC sourceA
    ring
  have hs : (∑ i, (sourceC L * (k i : ℝ)) ^ 2) =
      4 * sourceA L * ∑ i, (k i : ℝ) ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    rw [mul_pow, hc]
  rw [sourceWeight, productDecay_eq_exp, hs]
  congr 1
  ring

theorem sourceWeight_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    Summable (sourceWeight d L) := by
  have ha : 0 < 2 * sourceA L := mul_pos (by norm_num) (sourceA_pos hL)
  have hs : Summable (decay (2 * sourceA L)) := by
    simpa using summable_decay_pow ha 0
  exact finite_product_summable hs (fun k => (Real.exp_pos _).le) d

theorem sourceZ_pos (d : ℕ) {L : ℝ} (hL : L ≠ 0) : 0 < sourceZ d L := by
  apply (sourceWeight_summable d hL).tsum_pos
    (fun k => by unfold sourceWeight productDecay decay; positivity) (fun _ => 0)
  simp [sourceWeight, productDecay, decay]

theorem pairedBound_eq_amplitude (d : ℕ) {L : ℝ} (hL : L ≠ 0) (k : Fin d → ℤ) :
    pairedBound d L k = Real.sqrt (2 * sourceWeight d L k / sourceZ d L) *
      ∏ i, (1 + |sourceC L * (k i : ℝ)|) ^ 4 := by
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
  have hp : productBound d (sourceA L) (sourceC L) k =
      productDecay d (sourceA L) k * ∏ i, (1 + |sourceC L * (k i : ℝ)|) ^ 4 := by
    exact Finset.prod_mul_distrib
  rw [pairedBound, hp, hamp]
  ring

theorem pairedBound_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0) :
    Summable (pairedBound d L) := by
  exact (productBound_summable (sourceA_pos hL) d).mul_left _

theorem pairedBound_subtype_summable (d : ℕ) {L : ℝ} (hL : L ≠ 0)
    (S : Set (Fin d → ℤ)) : Summable (fun k : S => pairedBound d L k) := by
  exact (pairedBound_summable d hL).subtype S

end I4Weights
