import Specifications.Side24
import SpecificationProofs.Side24Elementary

/-!
# SIDE24 analysis proofs: the lattice-image sum bound (`LatticeImageSumBound`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/Side24.lean` (statement packet `specifications/lean_statements_20261010`, which
is not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop`
only: it does not make the informal Layer 0 note (SIDE24-COEFFICIENT-D23-20260924-v1) reviewed,
accepted or "verified", it does not identify the formal definitions with the informal objects
(that alignment is PENDING_INDEPENDENT_REVIEW), it does not enclose the coefficient `c_{d,24}`,
and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.Side24Analysis.Aux` (prefix `li_`). The route: both
conjuncts of the Prop follow from one uniform bound on the finite partial sums
`∑ n ∈ s, imageTerm d n ≤ 1458 exp(-288)` (`summable_of_sum_le`, `Real.tsum_le_of_sum_le`,
the terms being nonnegative). Pointwise, for `d ≤ 3`, the power-mean inequality
`(∑ a_i)^3 ≤ d^2 ∑ a_i^3 ≤ 9 ∑ a_i^3` (`pow_sum_div_card_le_sum_pow`) and
`exp(-288 ∑ n_k^2) = ∏ exp(-288 n_k^2)` (`Real.exp_sum`) give
`imageTerm d n ≤ 9 ∑ i, ∏ k, h_i k (n k)` with `h_i i m = m^6 exp(-288 m^2)` and
`h_i k m = exp(-288 m^2)` for `k ≠ i`. A finite set `s` of lattice points sits inside a box
`Fintype.piFinset`, over which the sum of such a product factors (`Finset.prod_univ_sum`) and is
bounded by `∏ k, ∑' m, h_i k m = B · A^(d-1)` with `A = ∑'_{m ∈ ℤ} exp(-288 m^2)` and
`B = ∑'_{m ∈ ℤ} m^6 exp(-288 m^2)`. The one-dimensional sums are controlled by splitting `ℤ` into
`ℕ` and the negatives (`tsum_of_nat_of_neg_add_one`) and comparing the tails with the geometric
series: `j^9 exp(-288 j^2) ≤ exp(-288) (1/2)^(j-1)` follows by induction from the registered
`SuccessiveTermRatio` (proved in `SpecificationProofs.Side24Elementary`), so `A ≤ 1 + 4 exp(-288)
≤ 2` and `B ≤ 4 exp(-288)`. Altogether `∑ n ∈ s, imageTerm d n ≤ 9 · d · B · A^(d-1) ≤ 432
exp(-288) ≤ 1458 exp(-288)`. The source's shell count `729 ∑ j^9 exp(-288 j^2)` is not
reproduced (the Prop does not specify it); the constant `1458` is met with room to spare.
-/

namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24

namespace Aux

/-! ### (a) The geometric tail `(n+1)^9 exp(-288 (n+1)^2) ≤ exp(-288) (1/2)^n` -/

/-- Induction on `n` from the registered successive-term ratio: the ratio of consecutive terms
`j^9 exp(-288 j^2)` is at most `512 exp(-864) < 1/2`
(`UniversalLaw.Proofs.Side24Elementary.successiveTermRatio`). -/
theorem li_geom_tail (n : ℕ) :
    ((n : ℝ) + 1) ^ 9 * Real.exp (-288 * ((n : ℝ) + 1) ^ 2) ≤ Real.exp (-288) * (1 / 2) ^ n := by
  induction n with
  | zero => norm_num
  | succ n ih =>
    have h1 := (UniversalLaw.Proofs.Side24Elementary.successiveTermRatio).1 (n + 1) (by omega)
    have h2 := UniversalLaw.Proofs.Side24Elementary.Aux.ratio_const_lt_half
    have hnn : 0 ≤ ((n : ℝ) + 1) ^ 9 * Real.exp (-288 * ((n : ℝ) + 1) ^ 2) := by positivity
    push_cast at h1 ⊢
    rw [pow_succ]
    calc _ ≤ 512 * Real.exp (-864) * (((n : ℝ) + 1) ^ 9 * Real.exp (-288 * ((n : ℝ) + 1) ^ 2)) :=
          h1
      _ ≤ (1 / 2) * (Real.exp (-288) * (1 / 2) ^ n) := mul_le_mul h2.le ih hnn (by norm_num)
      _ = Real.exp (-288) * ((1 / 2) ^ n * (1 / 2)) := by ring

/-- `(n+1)^6 exp(-288 (n+1)^2) ≤ exp(-288) (1/2)^n`: the sixth power is at most the ninth. -/
theorem li_pow6_succ_le (n : ℕ) :
    ((n : ℝ) + 1) ^ 6 * Real.exp (-288 * ((n : ℝ) + 1) ^ 2) ≤ Real.exp (-288) * (1 / 2) ^ n := by
  refine le_trans ?_ (li_geom_tail n)
  have h0 : (0 : ℝ) ≤ n := by positivity
  have h1 : (1 : ℝ) ≤ (n : ℝ) + 1 := by linarith
  exact mul_le_mul_of_nonneg_right (pow_le_pow_right₀ h1 (by norm_num)) (Real.exp_pos _).le

/-- `exp(-288 (n+1)^2) ≤ exp(-288) (1/2)^n`: `1 ≤ (n+1)^9`. -/
theorem li_exp_succ_le (n : ℕ) :
    Real.exp (-288 * ((n : ℝ) + 1) ^ 2) ≤ Real.exp (-288) * (1 / 2) ^ n := by
  refine le_trans ?_ (li_geom_tail n)
  have h0 : (0 : ℝ) ≤ n := by positivity
  have h1 : (1 : ℝ) ≤ ((n : ℝ) + 1) ^ 9 := one_le_pow₀ (by linarith)
  exact le_mul_of_one_le_left (Real.exp_pos _).le h1

/-! ### (b) One-dimensional sums over `ℤ` -/

/-- A nonnegative `φ : ℤ → ℝ` whose values at `n+1` and at `-(n+1)` are at most
`exp(-288) (1/2)^n` is summable over `ℤ`, with `∑' φ ≤ φ 0 + 4 exp(-288)`: split `ℤ` into `ℕ`
and the negatives (`tsum_of_nat_of_neg_add_one`), peel off `φ 0` (`Summable.tsum_eq_zero_add`)
and compare both tails with the geometric series (`tsum_geometric_two`). -/
theorem li_summable_int_tsum_le (φ : ℤ → ℝ) (h0 : ∀ m, 0 ≤ φ m)
    (hpos : ∀ n : ℕ, φ ((n : ℤ) + 1) ≤ Real.exp (-288) * (1 / 2) ^ n)
    (hneg : ∀ n : ℕ, φ (-((n : ℤ) + 1)) ≤ Real.exp (-288) * (1 / 2) ^ n) :
    Summable φ ∧ ∑' m : ℤ, φ m ≤ φ 0 + 4 * Real.exp (-288) := by
  have hgeo : Summable (fun n : ℕ => Real.exp (-288) * (1 / 2 : ℝ) ^ n) :=
    summable_geometric_two.mul_left _
  have hgeo_sum : ∑' n : ℕ, Real.exp (-288) * (1 / 2 : ℝ) ^ n = 2 * Real.exp (-288) := by
    rw [tsum_mul_left, tsum_geometric_two]
    ring
  have hs1 : Summable (fun n : ℕ => φ ((n : ℤ) + 1)) :=
    Summable.of_nonneg_of_le (fun n => h0 _) hpos hgeo
  have hs2 : Summable (fun n : ℕ => φ (-((n : ℤ) + 1))) :=
    Summable.of_nonneg_of_le (fun n => h0 _) hneg hgeo
  have hs0 : Summable (fun n : ℕ => φ (n : ℤ)) := by
    refine (summable_nat_add_iff 1).mp ?_
    refine hs1.congr fun n => ?_
    norm_cast
  have hsum0 : ∑' n : ℕ, φ (n : ℤ) ≤ φ 0 + 2 * Real.exp (-288) := by
    rw [hs0.tsum_eq_zero_add, Nat.cast_zero]
    have htail : ∑' n : ℕ, φ (((n + 1 : ℕ) : ℤ)) ≤ 2 * Real.exp (-288) := by
      rw [← hgeo_sum]
      refine ((summable_nat_add_iff 1).mpr hs0).tsum_le_tsum (fun n => ?_) hgeo
      rw [Nat.cast_succ]
      exact hpos n
    linarith
  have hsum2 : ∑' n : ℕ, φ (-((n : ℤ) + 1)) ≤ 2 * Real.exp (-288) := by
    rw [← hgeo_sum]
    exact hs2.tsum_le_tsum hneg hgeo
  refine ⟨Summable.of_nat_of_neg_add_one hs0 hs2, ?_⟩
  rw [tsum_of_nat_of_neg_add_one hs0 hs2]
  linarith

/-- The Gaussian weight `m ↦ exp(-288 m^2)` is summable over `ℤ` with sum `A ≤ 2`
(`A ≤ 1 + 4 exp(-288)` and `exp(-288) < 10^(-125)`). -/
theorem li_gauss_int :
    Summable (fun m : ℤ => Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)) ∧
      ∑' m : ℤ, Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) ≤ 2 := by
  have h := li_summable_int_tsum_le (fun m : ℤ => Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2))
    (fun m => (Real.exp_pos _).le)
    (fun n => by
      push_cast
      exact li_exp_succ_le n)
    (fun n => by
      push_cast
      rw [neg_sq]
      exact li_exp_succ_le n)
  refine ⟨h.1, h.2.trans ?_⟩
  have he := UniversalLaw.Proofs.Side24Elementary.Aux.exp_neg_lt
  have hp : (1 : ℝ) / 10 ^ 125 ≤ 1 / 4 := by norm_num
  have h0 : Real.exp (-288 * ((0 : ℤ) : ℝ) ^ 2) = 1 := by simp
  simp only [h0]
  linarith

/-- The weight `m ↦ m^6 exp(-288 m^2)` is summable over `ℤ` with sum `B ≤ 4 exp(-288)`. -/
theorem li_pow_gauss_int :
    Summable (fun m : ℤ => ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)) ∧
      ∑' m : ℤ, ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) ≤
        4 * Real.exp (-288) := by
  have h := li_summable_int_tsum_le
    (fun m : ℤ => ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2))
    (fun m => mul_nonneg (by positivity) (Real.exp_pos _).le)
    (fun n => by
      push_cast
      exact li_pow6_succ_le n)
    (fun n => by
      push_cast
      rw [neg_sq, Even.neg_pow (by decide : Even 6)]
      exact li_pow6_succ_le n)
  refine ⟨h.1, h.2.trans ?_⟩
  simp

/-! ### (c) Finite sums of products over `Fin d → ℤ` -/

/-- For `h : Fin d → ℤ → ℝ` nonnegative and summable in each coordinate, the sum of
`∏ k, h k (n k)` over a finite set `s` of lattice points is at most `∏ k, ∑' m, h k m`: `s` lies
in the box `Fintype.piFinset (fun k => s.image (· k))`, over which the sum factors
(`Finset.prod_univ_sum`), and each factor is a partial sum of a nonnegative summable family
(`Summable.sum_le_tsum`). -/
theorem li_sum_prod_le (d : ℕ) (h : Fin d → ℤ → ℝ) (h0 : ∀ k m, 0 ≤ h k m)
    (hs : ∀ k, Summable (h k)) (s : Finset (Fin d → ℤ)) :
    ∑ n ∈ s, ∏ k, h k (n k) ≤ ∏ k, ∑' m, h k m := by
  set t : Fin d → Finset ℤ := fun k => s.image (fun n => n k)
  have hsub : s ⊆ Fintype.piFinset t := fun n hn => by
    rw [Fintype.mem_piFinset]
    intro k
    exact Finset.mem_image_of_mem _ hn
  have hle : ∀ k, ∑ m ∈ t k, h k m ≤ ∑' m, h k m := fun k =>
    (hs k).sum_le_tsum (t k) (fun m _ => h0 k m)
  calc ∑ n ∈ s, ∏ k, h k (n k) ≤ ∑ n ∈ Fintype.piFinset t, ∏ k, h k (n k) :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub fun n _ _ => Finset.prod_nonneg fun k _ => h0 _ _
    _ = ∏ k, ∑ m ∈ t k, h k m := (Finset.prod_univ_sum t fun k m => h k m).symm
    _ ≤ ∏ k, ∑' m, h k m :=
        Finset.prod_le_prod₀ (fun k _ => Finset.sum_nonneg fun m _ => h0 k m) (fun k _ => hle k)

/-- A product over `Fin d` with one distinguished factor:
`∏ k, (if k = i then F k else G k) = F i * ∏ k ∈ univ.erase i, G k`. -/
theorem li_prod_ite_eq (d : ℕ) (i : Fin d) (F G : Fin d → ℝ) :
    (∏ k : Fin d, if k = i then F k else G k) = F i * ∏ k ∈ Finset.univ.erase i, G k := by
  rw [← Finset.mul_prod_erase Finset.univ (fun k => if k = i then F k else G k)
    (Finset.mem_univ i), ite_eq_left rfl]
  congr 1
  exact Finset.prod_congr rfl fun k hk => ite_eq_right (Finset.ne_of_mem_erase hk)

/-- `∏ k, exp(-288 n_k^2) = exp(-288 |n|^2)` (`Real.exp_sum`). -/
theorem li_prod_exp (d : ℕ) (n : Fin d → ℤ) :
    ∏ k : Fin d, Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2) = Real.exp (-288 * intNormSq d n) := by
  rw [← Real.exp_sum, intNormSq, Finset.mul_sum]

/-! ### (d) The pointwise domination for `d ≤ 3` -/

/-- Power mean for at most three nonnegative terms: `(∑ a_i)^3 ≤ 9 ∑ a_i^3` for `d ≤ 3`
(Jensen's `pow_sum_div_card_le_sum_pow` with `d^2 ≤ 9`; the case `d = 0` is `0 ≤ 0`). -/
theorem li_sum_cube_le (d : ℕ) (hd : d ≤ 3) (a : Fin d → ℝ) (ha : ∀ i, 0 ≤ a i) :
    (∑ i, a i) ^ 3 ≤ 9 * ∑ i, a i ^ 3 := by
  rcases Nat.eq_zero_or_pos d with hd0 | hdpos
  · subst hd0
    simp
  · have hdR : (0 : ℝ) < d := by exact_mod_cast hdpos
    have h := pow_sum_div_card_le_sum_pow (s := Finset.univ) (f := a) (fun i _ => ha i) 2
    rw [Finset.card_univ, Fintype.card_fin, div_le_iff₀ (pow_pos hdR 2)] at h
    have hd' : (d : ℝ) ≤ 3 := by exact_mod_cast hd
    have hsum : 0 ≤ ∑ i, a i ^ 3 := Finset.sum_nonneg fun i _ => pow_nonneg (ha i) 3
    calc (∑ i, a i) ^ 3 ≤ (∑ i, a i ^ 3) * (d : ℝ) ^ 2 := h
      _ ≤ (∑ i, a i ^ 3) * 9 := by
          apply mul_le_mul_of_nonneg_left _ hsum
          nlinarith
      _ = 9 * ∑ i, a i ^ 3 := by ring

/-- `imageTerm d n ≥ 0`. -/
theorem li_imageTerm_nonneg (d : ℕ) (n : Fin d → ℤ) : 0 ≤ imageTerm d n := by
  unfold imageTerm
  split_ifs
  · exact le_rfl
  · exact mul_nonneg (pow_nonneg (Finset.sum_nonneg fun i _ => sq_nonneg _) 3) (Real.exp_pos _).le

/-- Pointwise domination for `d ≤ 3`: `imageTerm d n ≤ 9 ∑ i, ∏ k, h_i k (n k)`, where
`h_i i m = m^6 exp(-288 m^2)` and `h_i k m = exp(-288 m^2)` for `k ≠ i`. Each product equals
`(n_i^2)^3 exp(-288 |n|^2)` (`li_prod_ite_eq`, `li_prod_exp`), and `(∑ n_k^2)^3 ≤ 9 ∑ (n_i^2)^3`
is `li_sum_cube_le`. -/
theorem li_imageTerm_le (d : ℕ) (hd : d ≤ 3) (n : Fin d → ℤ) :
    imageTerm d n ≤ 9 * ∑ i : Fin d, ∏ k : Fin d,
      (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
        else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)) := by
  have hprod : ∀ i : Fin d, (∏ k : Fin d,
      (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
        else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2))) =
      (((n i : ℤ) : ℝ) ^ 2) ^ 3 * Real.exp (-288 * intNormSq d n) := by
    intro i
    rw [li_prod_ite_eq d i (fun k => ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2))
      (fun k => Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)), ← li_prod_exp d n,
      ← Finset.mul_prod_erase Finset.univ (fun k => Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2))
        (Finset.mem_univ i)]
    ring
  rw [Finset.sum_congr rfl (fun i _ => hprod i), ← Finset.sum_mul]
  unfold imageTerm
  split_ifs
  · positivity
  · have h := li_sum_cube_le d hd (fun i => ((n i : ℤ) : ℝ) ^ 2) (fun i => sq_nonneg _)
    unfold intNormSq
    calc (∑ i, ((n i : ℤ) : ℝ) ^ 2) ^ 3 * Real.exp (-288 * ∑ i, ((n i : ℤ) : ℝ) ^ 2)
        ≤ (9 * ∑ i, (((n i : ℤ) : ℝ) ^ 2) ^ 3) * Real.exp (-288 * ∑ i, ((n i : ℤ) : ℝ) ^ 2) :=
          mul_le_mul_of_nonneg_right h (Real.exp_pos _).le
      _ = 9 * ((∑ i, (((n i : ℤ) : ℝ) ^ 2) ^ 3) * Real.exp (-288 * ∑ i, ((n i : ℤ) : ℝ) ^ 2)) := by
          ring

/-! ### (e) The uniform partial-sum bound -/

/-- For `d ≤ 3` and every finite `s`, `∑ n ∈ s, imageTerm d n ≤ 1458 exp(-288)`: the pointwise
bound `li_imageTerm_le`, the box bound `li_sum_prod_le` for each `i`, the one-dimensional sums
`A ≤ 2` and `B ≤ 4 exp(-288)` (`li_gauss_int`, `li_pow_gauss_int`), and
`9 · d · B · A^(d-1) ≤ 9 · 3 · 4 exp(-288) · 4 = 432 exp(-288)`. -/
theorem li_partial_sum_le (d : ℕ) (hd : d ≤ 3) (s : Finset (Fin d → ℤ)) :
    ∑ n ∈ s, imageTerm d n ≤ 1458 * Real.exp (-288) := by
  obtain ⟨hgs, hgle⟩ := li_gauss_int
  obtain ⟨hfs, hfle⟩ := li_pow_gauss_int
  have hA0 : 0 ≤ ∑' m : ℤ, Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) :=
    tsum_nonneg fun m => (Real.exp_pos _).le
  have hexp : 0 ≤ Real.exp (-288) := (Real.exp_pos _).le
  set A : ℝ := ∑' m : ℤ, Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) with hA
  set B : ℝ := ∑' m : ℤ, ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2) with hB
  have hidx : ∀ i : Fin d, ∑ n ∈ s, ∏ k : Fin d,
      (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
        else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)) ≤ 16 * Real.exp (-288) := by
    intro i
    have h1 := li_sum_prod_le d
      (fun k m => if k = i then ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)
        else Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2))
      (fun k m => by
        split_ifs
        · positivity
        · positivity)
      (fun k => by
        by_cases hk : k = i
        · simp only [hk, ite_true]
          exact hfs
        · simp only [hk, ite_false]
          exact hgs)
      s
    have h2 : ∀ k : Fin d, ∑' m : ℤ,
        (if k = i then ((m : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)
          else Real.exp (-288 * ((m : ℤ) : ℝ) ^ 2)) = if k = i then B else A := by
      intro k
      by_cases hk : k = i
      · simp only [hk, ite_true, hB]
      · simp only [hk, ite_false, hA]
    calc _ ≤ _ := h1
      _ = ∏ k : Fin d, (if k = i then B else A) := Finset.prod_congr rfl fun k _ => h2 k
      _ = B * ∏ _k ∈ Finset.univ.erase i, A := li_prod_ite_eq d i (fun _ => B) (fun _ => A)
      _ = B * A ^ (d - 1) := by
          rw [Finset.prod_const, Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ,
            Fintype.card_fin]
      _ ≤ (4 * Real.exp (-288)) * 2 ^ (d - 1) :=
          mul_le_mul hfle (pow_le_pow_left₀ hA0 hgle _) (pow_nonneg hA0 _) (by positivity)
      _ ≤ (4 * Real.exp (-288)) * 2 ^ 2 := by
          apply mul_le_mul_of_nonneg_left _ (by positivity)
          exact pow_le_pow_right₀ (by norm_num) (by omega)
      _ = 16 * Real.exp (-288) := by ring
  calc ∑ n ∈ s, imageTerm d n
      ≤ ∑ n ∈ s, 9 * ∑ i : Fin d, ∏ k : Fin d,
          (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
            else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)) :=
        Finset.sum_le_sum fun n _ => li_imageTerm_le d hd n
    _ = 9 * ∑ i : Fin d, ∑ n ∈ s, ∏ k : Fin d,
          (if k = i then ((n k : ℤ) : ℝ) ^ 6 * Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)
            else Real.exp (-288 * ((n k : ℤ) : ℝ) ^ 2)) := by
        rw [← Finset.mul_sum, Finset.sum_comm]
    _ ≤ 9 * ∑ _i : Fin d, 16 * Real.exp (-288) :=
        mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => hidx i) (by norm_num)
    _ = 9 * ((d : ℝ) * (16 * Real.exp (-288))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    _ ≤ 9 * (3 * (16 * Real.exp (-288))) := by
        have hd' : (d : ℝ) ≤ 3 := by exact_mod_cast hd
        gcongr
    _ ≤ 1458 * Real.exp (-288) := by linarith

end Aux

/-! ## The lattice-image sum bound for `d ≤ 3` -/

/-- Proves: `UniversalLaw.Spec.Side24.LatticeImageSumBound` (claim `side24-coefficient`).
Method: both conjuncts from the uniform partial-sum bound `Aux.li_partial_sum_le` via
`summable_of_sum_le` and `Real.tsum_le_of_sum_le` (terms nonnegative, `Aux.li_imageTerm_nonneg`);
the partial sums are dominated pointwise by `9 ∑ i, ∏ k, h_i k (n k)` (`Aux.li_imageTerm_le`,
from the power mean `pow_sum_div_card_le_sum_pow` and `Real.exp_sum`), each finite sum of
products sits in a `Fintype.piFinset` box and factors (`Aux.li_sum_prod_le`, via
`Finset.prod_univ_sum` and `Summable.sum_le_tsum`), and the one-dimensional sums over `ℤ` are
bounded by `2` and `4 exp(-288)` (`Aux.li_gauss_int`, `Aux.li_pow_gauss_int`) through
`tsum_of_nat_of_neg_add_one`, the geometric series `tsum_geometric_two` and the registered
`SuccessiveTermRatio` (`SpecificationProofs.Side24Elementary.successiveTermRatio`), giving
`432 exp(-288) ≤ 1458 exp(-288)`.
Does not establish: the informal Section 2 lattice display of the source as the source's object
(the alignment of `intNormSq`, `imageTerm` and this `tsum` over `Fin d → ℤ` with the source's
`sum_{n ≠ 0} |n|^6 e^(-288|n|^2)` is PENDING_INDEPENDENT_REVIEW); the intermediate shell count
`729 ∑_j j^9 e^(-288 j^2)` of the source, which the Prop does not specify and this proof does not
reproduce; the arithmetic `27·27 = 729`, `729·2 = 1458` kernel-checked on main, not here;
`ImageBound`, the derivative bounds of Section 2 or any enclosure of the coefficient; the
disposition of `side24-coefficient` (REVIEWED_SCOPED), which is transcribed in the register, not
derived here; nonauthor review or acceptance of SIDE24-COEFFICIENT-D23-20260924-v1. -/
theorem latticeImageSumBound : UniversalLaw.Spec.Side24.LatticeImageSumBound := by
  intro d hd
  exact ⟨summable_of_sum_le (fun n => Aux.li_imageTerm_nonneg d n) (Aux.li_partial_sum_le d hd),
    Real.tsum_le_of_sum_le (fun n => Aux.li_imageTerm_nonneg d n) (Aux.li_partial_sum_le d hd)⟩

end UniversalLaw.Proofs.Side24Analysis

-- SCRATCH COPY ONLY: axiom report for the integrator; not part of the deliverable.
#print axioms UniversalLaw.Proofs.Side24Analysis.latticeImageSumBound
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_geom_tail
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_pow6_succ_le
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_exp_succ_le
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_summable_int_tsum_le
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_gauss_int
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_pow_gauss_int
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_sum_prod_le
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_prod_ite_eq
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_prod_exp
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_sum_cube_le
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_imageTerm_nonneg
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_imageTerm_le
#print axioms UniversalLaw.Proofs.Side24Analysis.Aux.li_partial_sum_le
#check @UniversalLaw.Proofs.Side24Analysis.latticeImageSumBound
example : UniversalLaw.Spec.Side24.LatticeImageSumBound :=
  UniversalLaw.Proofs.Side24Analysis.latticeImageSumBound
