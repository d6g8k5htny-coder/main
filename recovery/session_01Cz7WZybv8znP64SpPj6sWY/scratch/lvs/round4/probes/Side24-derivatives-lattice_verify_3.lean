import Specifications.Side24
import SpecificationProofs.Side24Elementary

/-! VERIFY probe (adversarial verifier) for `LatticeImageSumBound` (triage: PROVABLE_WITH_WORK).
Checks that every named Mathlib / packet declaration of the route exists at the pin, and tests the
route's elementary steps: the d = 3 power-mean inequality, nonnegativity of `imageTerm`, and the
d = 0 degenerate instance. Probe file only. -/

open UniversalLaw.Spec.Side24

#check @Real.tsum_le_of_sum_le
#check @summable_of_sum_le
#check @HasSum.int_rec
#check @Summable.sum_le_tsum
#check @Finset.sum_le_sum_of_subset_of_nonneg
#check @Fintype.card_piFinset
#check @Fintype.mem_piFinset
#check @Finset.sum_fiberwise
#check @Finset.prod_univ_sum
#check @Finset.prod_le_prod₀
#check @tsum_geometric_two
#check @tsum_geometric_of_lt_one
#check @Real.exp_sum
#check @HurwitzKernelBounds.summable_f_int
#check @UniversalLaw.Proofs.Side24Elementary.successiveTermRatio
#check @UniversalLaw.Proofs.Side24Elementary.Aux.pow_succ_le
#check @UniversalLaw.Proofs.Side24Elementary.Aux.exp_succ_le
#check @UniversalLaw.Proofs.Side24Elementary.Aux.ratio_const_lt_half

-- power-mean step for d = 3: (a+b+c)^3 ≤ 9 (a^3+b^3+c^3) for a,b,c ≥ 0
example (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
    (a + b + c) ^ 3 ≤ 9 * (a ^ 3 + b ^ 3 + c ^ 3) := by
  nlinarith [mul_nonneg (add_nonneg ha hb) (sq_nonneg (a - b)),
    mul_nonneg (add_nonneg hb hc) (sq_nonneg (b - c)),
    mul_nonneg (add_nonneg ha hc) (sq_nonneg (a - c)),
    mul_nonneg (add_nonneg (add_nonneg ha hb) hc)
      (add_nonneg (add_nonneg (sq_nonneg (a - b)) (sq_nonneg (b - c))) (sq_nonneg (c - a)))]

-- d = 2: (a+b)^3 ≤ 4 (a^3+b^3) ≤ 9 (a^3 + b^3)
example (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : (a + b) ^ 3 ≤ 9 * (a ^ 3 + b ^ 3) := by
  nlinarith [mul_nonneg (add_nonneg ha hb) (sq_nonneg (a - b)), pow_nonneg ha 3, pow_nonneg hb 3]

-- nonnegativity of imageTerm
example (d : ℕ) (n : Fin d → ℤ) : 0 ≤ imageTerm d n := by
  unfold imageTerm; split_ifs
  · exact le_rfl
  · exact mul_nonneg (pow_nonneg (Finset.sum_nonneg fun i _ => sq_nonneg _) 3) (Real.exp_pos _).le

-- d = 0 degenerate instance: `Fin 0 → ℤ` is a singleton, imageTerm ≡ 0, both conjuncts hold
example : Summable (fun n : Fin 0 → ℤ => imageTerm 0 n) ∧
    (∑' n : Fin 0 → ℤ, imageTerm 0 n) ≤ 1458 * Real.exp (-288) := by
  have h : ∀ n : Fin 0 → ℤ, imageTerm 0 n = 0 := fun n => by
    simp [imageTerm, Subsingleton.elim n 0]
  constructor
  · exact summable_zero.congr (fun n => (h n).symm)
  · simp [h]; positivity

-- exp(-288 * |n|^2) = ∏ exp(-288 n_i^2) (Real.exp_sum direction)
example (d : ℕ) (n : Fin d → ℤ) :
    Real.exp (-288 * intNormSq d n) = ∏ i, Real.exp (-288 * ((n i : ℤ) : ℝ) ^ 2) := by
  rw [← Real.exp_sum, intNormSq, Finset.mul_sum]

-- geometric step: for j ≥ 1, j^6 e^{-288 j^2} ≤ j^9 e^{-288 j^2}
example (j : ℕ) (hj : 1 ≤ j) :
    (j : ℝ) ^ 6 * Real.exp (-288 * (j : ℝ) ^ 2) ≤ (j : ℝ) ^ 9 * Real.exp (-288 * (j : ℝ) ^ 2) := by
  have hj' : (1 : ℝ) ≤ j := by exact_mod_cast hj
  exact mul_le_mul_of_nonneg_right (pow_le_pow_right₀ hj' (by norm_num)) (Real.exp_pos _).le
