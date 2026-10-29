import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Analysis.Normed.Ring.InfiniteSum
import Mathlib.Analysis.Normed.Module.FiniteDimension
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith

/-!
# Seven-mode certificate for contact-jet covariance

The row order is `(f, fx, fy, fxx, fxy, fxxx, fyy)`, with selected columns
`(DC, cos(1,0), cos(0,1), cos(1,1), sin(1,0), sin(2,0), sin(0,1))`.
Nonzero mode amplitudes are retained. The witness forces strict positivity of an
infinite coefficient Gram matrix by its sum-of-squares quadratic identity.

This is a coefficient certificate. Applying it to the literal smooth source requires
the exact derivative-coefficient identities and physical covariance identification;
these are not premises that may be silently omitted. See `SOURCE_CONTACT_GRAM.md`.
-/

open Matrix
open scoped BigOperators Matrix
set_option autoImplicit false

namespace I4ContactGram

def contactWitness (c : ℝ) : Matrix (Fin 7) (Fin 7) ℝ :=
  !![1,1,1,1,0,0,0;
     0,0,0,0,c,2*c,0;
     0,0,0,0,0,0,c;
     0,-c^2,0,-c^2,0,0,0;
     0,0,0,-c^2,0,0,0;
     0,0,0,0,-c^3,-8*c^3,0;
     0,0,-c^2,-c^2,0,0,0]

lemma contactWitness_transpose_mulVec_eq_zero {c : ℝ} (hc : c ≠ 0)
    (z : Fin 7 → ℝ) (h : (contactWitness c)ᵀ *ᵥ z = 0) : z = 0 := by
  have hs (s : Fin 7) := congrFun h s
  have h0 : z 0 = 0 := by
    simpa [contactWitness, Matrix.mulVec, dotProduct, Fin.sum_univ_succ] using hs 0
  have h1 := hs 1
  have h2 := hs 2
  have h3 := hs 3
  have h4 := hs 4
  have h5 := hs 5
  have h6 := hs 6
  simp [contactWitness, Matrix.mulVec, dotProduct, Fin.sum_univ_succ, h0] at h1 h2 h3 h4 h5 h6
  have hc2 : c^2 ≠ 0 := pow_ne_zero _ hc
  have hc3 : c^3 ≠ 0 := pow_ne_zero _ hc
  have hz3 : z 3 = 0 := h1.resolve_left hc
  have hz6 : z 6 = 0 := h2.resolve_left hc
  have hz4 : z 4 = 0 := (mul_eq_zero.mp (show c^2 * z 4 = 0 by nlinarith [h3])).resolve_left hc2
  have hz5 : z 5 = 0 := (mul_eq_zero.mp (show c^3 * z 5 = 0 by nlinarith [h4, h5])).resolve_left hc3
  have hz1 : z 1 = 0 := (mul_eq_zero.mp (show c * z 1 = 0 by nlinarith [h4])).resolve_left hc
  have hz2 : z 2 = 0 := h6.resolve_left hc
  funext i
  fin_cases i <;> assumption

lemma contactWitness_transpose_injective {c : ℝ} (hc : c ≠ 0) :
    Function.Injective ((contactWitness c)ᵀ).mulVec := by
  intro x y h
  apply sub_eq_zero.mp
  apply contactWitness_transpose_mulVec_eq_zero hc
  simp [Matrix.mulVec_sub, h]

lemma contactWitness_amplitude_transpose_injective {c : ℝ} (hc : c ≠ 0)
    (a : Fin 7 → ℝ) (ha : ∀ s, a s ≠ 0) :
    Function.Injective ((contactWitness c * diagonal a)ᵀ).mulVec := by
  intro x y h
  apply contactWitness_transpose_injective hc
  funext s
  have hs := congrFun h s
  simp only [Matrix.transpose_mul, Matrix.diagonal_transpose, ← Matrix.mulVec_mulVec,
    Matrix.mulVec_diagonal] at hs
  exact mul_left_cancel₀ (ha s) hs

noncomputable def coefficientGram {ι : Type*} {n : ℕ} (coeff : Fin n → ι → ℝ) :
    Matrix (Fin n) (Fin n) ℝ := fun i j => ∑' m, coeff i m * coeff j m

lemma coefficient_product_summable {ι : Type*} {a b : ι → ℝ}
    (ha : Summable a) (hb : Summable b) : Summable (fun m => a m * b m) := by
  have hp := summable_mul_of_summable_norm ha.norm hb.norm
  simpa only [Function.comp_def] using hp.comp_injective
    (show Function.Injective (fun m : ι => (m,m)) from
      fun i j h => congrArg Prod.fst h)

lemma coefficientGram_quadratic_hasSum {ι : Type*} {n : ℕ} (coeff : Fin n → ι → ℝ)
    (hc : ∀ i, Summable (coeff i)) (z : Fin n → ℝ) :
    HasSum (fun m => (∑ i, z i * coeff i m)^2)
      (star z ⬝ᵥ (coefficientGram coeff *ᵥ z)) := by
  have hs : HasSum (fun m => ∑ i, ∑ j, z i * (coeff i m * coeff j m) * z j)
      (∑ i, ∑ j, z i * coefficientGram coeff i j * z j) :=
    hasSum_sum (fun i _ => hasSum_sum (fun j _ =>
      (((coefficient_product_summable (hc i) (hc j)).hasSum).mul_left (z i)).mul_right (z j)))
  convert hs using 1
  · funext m
    rw [pow_two, Finset.sum_mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    apply Finset.sum_congr rfl
    intro j _
    ring
  · simp [Matrix.mulVec, dotProduct, Finset.mul_sum, mul_assoc]

lemma coefficientGram_posDef_of_separating {ι : Type*} {n : ℕ}
    (coeff : Fin n → ι → ℝ) (hc : ∀ i, Summable (coeff i))
    (hsep : ∀ z : Fin n → ℝ, z ≠ 0 → ∃ m, (∑ i, z i * coeff i m) ≠ 0) :
    (coefficientGram coeff).PosDef := by
  apply Matrix.PosDef.of_dotProduct_mulVec_pos
  · ext i j
    simp only [coefficientGram, Matrix.conjTranspose_apply, star_trivial]
    apply tsum_congr
    intro m
    ring
  · intro z hz
    have hs := coefficientGram_quadratic_hasSum coeff hc z
    rw [← hs.tsum_eq]
    obtain ⟨m, hm⟩ := hsep z hz
    exact hs.summable.tsum_pos (fun m => sq_nonneg _) m (sq_pos_of_ne_zero hm)

/-- Seven explicit independent mode columns force positivity of the entire infinite Gram sum.
No positive-definiteness or invertibility conclusion is an input premise. -/
theorem contact_coefficientGram_posDef {ι : Type*} (coeff : Fin 7 → ι → ℝ)
    (hcoeff : ∀ i, Summable (coeff i)) {c : ℝ} (hc : c ≠ 0)
    (mode : Fin 7 → ι) (a : Fin 7 → ℝ) (ha : ∀ s, a s ≠ 0)
    (hcolumns : ∀ i s, coeff i (mode s) = a s * contactWitness c i s) :
    (coefficientGram coeff).PosDef := by
  apply coefficientGram_posDef_of_separating coeff hcoeff
  intro z hz
  classical
  by_cases hex : ∃ m, (∑ i, z i * coeff i m) ≠ 0
  · exact hex
  · exfalso
    apply hz
    apply contactWitness_amplitude_transpose_injective hc a ha
    rw [Matrix.mulVec_zero]
    funext s
    have hm : (∑ i, z i * coeff i (mode s)) = 0 :=
      not_ne_iff.mp (not_exists.mp hex (mode s))
    simp_rw [hcolumns] at hm
    simpa [Matrix.mulVec, dotProduct, Matrix.mul_diagonal, mul_comm, mul_left_comm,
      mul_assoc] using hm

/-- A positive-definite six-plus-one block covariance has strictly positive scalar Schur complement. -/
theorem contact_schur_pos (A : Matrix (Fin 6) (Fin 6) ℝ)
    (B : Matrix (Fin 6) Unit ℝ) (D : Matrix Unit Unit ℝ)
    (hG : (Matrix.fromBlocks A B Bᴴ D).PosDef) :
    A.PosDef ∧ 0 < (D - Bᴴ * A⁻¹ * B) () () := by
  have hA : A.PosDef := by
    convert hG.submatrix (e := Sum.inl) Sum.inl_injective using 1
    ext i j
    rfl
  let _ : Invertible A := hA.isUnit.invertible
  let y : Unit → ℝ := fun _ => 1
  let x : Fin 6 → ℝ := -((A⁻¹ * B) *ᵥ y)
  have hxy : Sum.elim x y ≠ 0 := by
    intro h
    have hh := congrFun h (Sum.inr ())
    simp [y] at hh
  have hp := hG.dotProduct_mulVec_pos hxy
  rw [Matrix.dotProduct_mulVec,
    Matrix.schur_complement_eq₁₁ B D x y hA.isHermitian] at hp
  dsimp [x] at hp
  rw [neg_add_cancel, dotProduct_zero, zero_add] at hp
  refine ⟨hA, ?_⟩
  simpa [y, Matrix.vecMul, dotProduct] using hp


end I4ContactGram
