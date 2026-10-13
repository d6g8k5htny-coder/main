import Specifications.P15
import SpecificationProofs.P15Elementary
import SpecificationProofs.P15Hazard
import SpecificationProofs.P15Covers
import SpecificationProofs.P15Budget

/-!
# P15 interpolation proofs: Theorem F's hazard interpolation (displays (F4)-(F8), (F12), (F2))
and the (F3) enclosure of `rho_star`

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/P15.lean` (statement packet `specifications/lean_statements_20261010`, which is
not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop` only:
it does not make the informal Layer 0 note reviewed, accepted or "verified", it does not
identify the formal definitions with the informal objects (that alignment is
PENDING_INDEPENDENT_REVIEW), and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.P15Interpolation.Aux`, tagged by the display they
serve (`f3_`, `cc_`, `f6_`, `f7_`, `f8_`, `bi_`, `lp_`, `f2_`); the gate inventories only the
eight theorems outside that namespace. The mathematics, in the order of the module:

* (F3) `rhoStarLower < rho_star < rhoStarUpper < 6/7`, with `rho_star = 1 / (3 - log(3e - 2))`.
  Exact rational arithmetic throughout: `rho_star < U` iff `3e - 2 < exp(3 - 1/U)` and
  `L < rho_star` iff `exp(3 - 1/L) < 3e - 2`; writing `3 - 1/U = 2 + x_U` with `|x_U| < 1/5`,
  `exp(3 - 1/U) = (exp 1)^2 exp x_U`, and `Real.exp_bound` at degree 26 (for `e`) and degree 16
  (for `exp x_U`, `exp x_L`) gives rational enclosures that `norm_num` compares against the
  decimal literals (margins about `1.08 * 10^-20` and `7.53 * 10^-20`).
* (F4), (F5) The split of `mu_p(A)` at one coordinate: with `p_i` replaced by `x`,
  `mu(A) = x S_in + (1 - x) S_out`, where `S_in`, `S_out` are the sums, over the members of `A`
  containing, respectively avoiding, the coordinate, of the weight over the other coordinates.
  For a downset `S_in ≤ S_out` (erasing the coordinate injects the first family into the second
  and preserves the reduced weight) and `S_out > 0` once `∅ ∈ A` and every other `p_j < 1`.
  With `x = 1 - exp(-s)` this is `S_in + (S_out - S_in) exp(-s)`, and `s ↦ -log(a + b exp(-s))`
  with `a, b ≥ 0`, `a + b > 0` has second derivative `-a b exp(-s) / (a + b exp(-s))^2 ≤ 0`
  (`concaveOn_of_deriv2_nonpos`): separate coordinatewise concavity of the hazard `F_A`.
* (F6) The chord inequality of that concavity on `[0,1]`, `f(s) ≥ (1 - s) f(0) + s f(1) ≥ s f(1)`
  since `f(0) = -log mu ≥ 0`, iterated over the coordinates (`Finset.induction_on`, raising one
  coordinate to `1` at a time): `(prod_i t_i) H_A ≤ F_A(t)`, as `F_A(1, …, 1) = H_A`.
* (F7) For `p` in the cube put `t_i := phi(p_i) ∈ [0,1]` and `p'_i := 1 - exp(-t_i) ≤ p_i`;
  `mu(A)` is antitone in the probability vector for a downset (the same split), so
  `-log mu_p(A) ≥ -log mu_p'(A) = F_A(t) ≥ (prod_i phi(p_i)) H_A`.
* (F8) `prod_i c_i ≤ prod_i phi(p_i) ≤ 1` for `0 ≤ c_i ≤ phi(p_i)`, and when `mu_p(A) > 0`,
  `prod_i phi(p_i) ≤ (1/H_A) [-log mu_p(A)]` by (F7) with `H_A > 0`; the exactness remark: the
  capped budget at `rho = 1` holds for all admissible `p, c` iff `H_A ≥ 1` (the (⇒) direction
  instantiates `p ≡ p_star`, `c ≡ 1`, admissible since `phi p_star = 1`).
* (F12) Each block `X_i` of a realized family is enumerated by `Fin n_i`, `n_i = a_i d_i + 1`;
  along the enumeration `mu_p(D_i)` is the product measure of the capacity family
  `{T : |T| ≤ a_i}` (`LocalRestriction`), a proper downset, so (F7) and `CapacityWorstCase`
  (`h_star ≤ H_{n_i, a_i}`) give `price(X_i) ≤ rho_star [-log mu_p(D_i)]` and `price(X_i) ≤ 1`.
* (F2) `covercost ≤ 1` (the cover `{∅}`); for `K ≥ K_H(d)` the full blocks cover, so
  `covercost ≤ sum_i price(X_i) ≤ rho_star sum_i [-log mu_p(D_i)] ≤ rho_star [-log mu_p(D)]`
  by (F12) and the global assembly (F13), `P15Hazard.globalAssemblyHazard`.
-/

namespace UniversalLaw.Proofs.P15Interpolation

open UniversalLaw.Spec.P15

namespace Aux

/-! ## (F3) The enclosure of `rho_star`

### A 29-decimal enclosure of `e` from the degree-26 Taylor bound -/

/-- `2.71828182845904523536028746620 < e`: the 26-term Taylor sum of `exp 1` minus the tail bound
`27 / (26! * 26)` of `Real.exp_bound`, evaluated exactly by `norm_num`. -/
theorem f3_exp_one_gt : (271828182845904523536028746620 : ℝ) / 10 ^ 29 < Real.exp 1 := by
  have h := Real.exp_bound (x := (1 : ℝ)) (by simp) (n := 26) (by norm_num)
  rw [abs_sub_le_iff] at h
  obtain ⟨-, h2⟩ := h
  norm_num [Finset.sum_range_succ] at h2
  linarith

/-- `e < 2.71828182845904523536028747136`: the 26-term Taylor sum of `exp 1` plus the tail bound
`27 / (26! * 26)` of `Real.exp_bound`, evaluated exactly by `norm_num`. -/
theorem f3_exp_one_lt : Real.exp 1 < (271828182845904523536028747136 : ℝ) / 10 ^ 29 := by
  have h := Real.exp_bound (x := (1 : ℝ)) (by simp) (n := 26) (by norm_num)
  rw [abs_sub_le_iff] at h
  obtain ⟨h1, -⟩ := h
  norm_num [Finset.sum_range_succ] at h1
  linarith

/-! ### Degree-16 Taylor enclosures of `exp x_U` and `exp x_L`

`x_U = 1 - 1/rhoStarUpper = -3863004568775331983 / 21136995431224668017` (reduced) and
`x_L = 1 - 1/rhoStarLower = -15452018275101327933 / 84547981724898672067`, both about `-0.18276`. -/

/-- Lower bound `0.83296775704110158100003160908803 ≤ exp x_U` from `Real.exp_bound` at degree 16. -/
theorem f3_exp_xU_ge :
    (83296775704110158100003160908803 : ℝ) / 10 ^ 32 ≤
      Real.exp (-3863004568775331983 / 21136995431224668017) := by
  have hx : |(-3863004568775331983 / 21136995431224668017 : ℝ)| ≤ 1 :=
    abs_le.mpr ⟨by norm_num, by norm_num⟩
  have h := Real.exp_bound hx (n := 16) (by norm_num)
  rw [abs_sub_le_iff] at h
  obtain ⟨-, h2⟩ := h
  norm_num [Finset.sum_range_succ] at h2
  linarith

/-- Upper bound `exp x_L ≤ 0.83296775704110158098837919695330` from `Real.exp_bound` at degree 16. -/
theorem f3_exp_xL_le :
    Real.exp (-15452018275101327933 / 84547981724898672067) ≤
      (83296775704110158098837919695330 : ℝ) / 10 ^ 32 := by
  have hx : |(-15452018275101327933 / 84547981724898672067 : ℝ)| ≤ 1 :=
    abs_le.mpr ⟨by norm_num, by norm_num⟩
  have h := Real.exp_bound hx (n := 16) (by norm_num)
  rw [abs_sub_le_iff] at h
  obtain ⟨h1, -⟩ := h
  norm_num [Finset.sum_range_succ] at h1
  linarith

/-! ### The two logarithmic inequalities -/

/-- `log(3e - 2) < 2 + x_U = 3 - 1/rhoStarUpper`: equivalently `3e - 2 < (exp 1)^2 * exp x_U`,
with `e` and `exp x_U` replaced by their rational bounds (`norm_num` margin about `1.08e-20`). -/
theorem f3_log_lt :
    Real.log (3 * Real.exp 1 - 2) < 2 + (-3863004568775331983 / 21136995431224668017 : ℝ) := by
  rw [Real.log_lt_iff_lt_exp P15Elementary.Aux.three_exp_sub_two_pos, Real.exp_add,
    P15Elementary.Aux.exp_two_eq]
  have he_lo := f3_exp_one_gt
  have he_hi := f3_exp_one_lt
  have hx := f3_exp_xU_ge
  have hsq : ((271828182845904523536028746620 : ℝ) / 10 ^ 29) ^ 2 ≤ Real.exp 1 ^ 2 :=
    pow_le_pow_left₀ (by norm_num) he_lo.le 2
  have hprod : ((271828182845904523536028746620 : ℝ) / 10 ^ 29) ^ 2 *
      ((83296775704110158100003160908803 : ℝ) / 10 ^ 32) ≤
      Real.exp 1 ^ 2 * Real.exp (-3863004568775331983 / 21136995431224668017) :=
    mul_le_mul hsq hx (by norm_num) (sq_nonneg _)
  have key : 3 * ((271828182845904523536028747136 : ℝ) / 10 ^ 29) - 2 <
      ((271828182845904523536028746620 : ℝ) / 10 ^ 29) ^ 2 *
        ((83296775704110158100003160908803 : ℝ) / 10 ^ 32) := by
    norm_num
  linarith

/-- `2 + x_L = 3 - 1/rhoStarLower < log(3e - 2)`: equivalently `(exp 1)^2 * exp x_L < 3e - 2`,
with `e` and `exp x_L` replaced by their rational bounds (`norm_num` margin about `7.53e-20`). -/
theorem f3_lt_log :
    2 + (-15452018275101327933 / 84547981724898672067 : ℝ) < Real.log (3 * Real.exp 1 - 2) := by
  rw [Real.lt_log_iff_exp_lt P15Elementary.Aux.three_exp_sub_two_pos, Real.exp_add,
    P15Elementary.Aux.exp_two_eq]
  have he_lo := f3_exp_one_gt
  have he_hi := f3_exp_one_lt
  have hx := f3_exp_xL_le
  have hsq : Real.exp 1 ^ 2 ≤ ((271828182845904523536028747136 : ℝ) / 10 ^ 29) ^ 2 :=
    pow_le_pow_left₀ (Real.exp_pos 1).le he_hi.le 2
  have hprod : Real.exp 1 ^ 2 * Real.exp (-15452018275101327933 / 84547981724898672067) ≤
      ((271828182845904523536028747136 : ℝ) / 10 ^ 29) ^ 2 *
        ((83296775704110158098837919695330 : ℝ) / 10 ^ 32) :=
    mul_le_mul hsq hx (Real.exp_pos _).le (by norm_num)
  have key : ((271828182845904523536028747136 : ℝ) / 10 ^ 29) ^ 2 *
        ((83296775704110158098837919695330 : ℝ) / 10 ^ 32) <
      3 * ((271828182845904523536028746620 : ℝ) / 10 ^ 29) - 2 := by
    norm_num
  linarith

/-! ### The three conjuncts of (F3) -/

/-- `0 < h_star` written out, from `one_lt_hStar`. -/
theorem f3_hStar_pos : (0 : ℝ) < 3 - Real.log (3 * Real.exp 1 - 2) := by
  have h := P15Elementary.Aux.one_lt_hStar
  unfold hStar at h
  linarith

/-- `rho_star < rhoStarUpper`. -/
theorem f3_rhoStar_lt_upper : rhoStar < rhoStarUpper := by
  unfold rhoStar rhoStarUpper
  rw [div_lt_iff₀ f3_hStar_pos]
  have h := f3_log_lt
  linarith

/-- `rhoStarLower < rho_star`. -/
theorem f3_lower_lt_rhoStar : rhoStarLower < rhoStar := by
  unfold rhoStar rhoStarLower
  rw [lt_div_iff₀ f3_hStar_pos]
  have h := f3_lt_log
  linarith

/-- `rhoStarUpper < 6/7`, a rational inequality. -/
theorem f3_upper_lt_six_sevenths : rhoStarUpper < 6 / 7 := by
  unfold rhoStarUpper
  norm_num

open Finset

/-! ## (F4), (F5) The split of the product weight at one coordinate and separate concavity -/

/-- The weight of `U` over the coordinates other than `i`: `prod_{v in U, v ≠ i} p_v *
prod_{v notin U, v ≠ i} (1 - p_v)`. -/
noncomputable def cc_wOther {Y : Type} [Fintype Y] [DecidableEq Y] (p : Y → ℝ) (i : Y)
    (U : Finset Y) : ℝ :=
  (∏ v ∈ U.erase i, p v) * ∏ v ∈ (univ \ U).erase i, (1 - p v)

/-- The reduced weight is nonnegative when every `p_v` lies in `[0, 1]`. -/
theorem cc_wOther_nonneg {Y : Type} [Fintype Y] [DecidableEq Y] {p : Y → ℝ}
    (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (i : Y) (U : Finset Y) : 0 ≤ cc_wOther p i U := by
  unfold cc_wOther
  refine mul_nonneg (Finset.prod_nonneg fun v _ => (hp v).1) (Finset.prod_nonneg fun v _ => ?_)
  linarith [(hp v).2]

/-- The reduced weight of the empty set is positive when every `p_v < 1`. -/
theorem cc_wOther_empty_pos {Y : Type} [Fintype Y] [DecidableEq Y] {p : Y → ℝ}
    (hp : ∀ v, p v < 1) (i : Y) : 0 < cc_wOther p i ∅ := by
  unfold cc_wOther
  rw [Finset.erase_empty, Finset.prod_empty, one_mul]
  exact Finset.prod_pos fun v _ => by linarith [hp v]

/-- Erasing the distinguished coordinate does not change the reduced weight. -/
theorem cc_wOther_erase {Y : Type} [Fintype Y] [DecidableEq Y] (p : Y → ℝ) (i : Y)
    (U : Finset Y) : cc_wOther p i (U.erase i) = cc_wOther p i U := by
  unfold cc_wOther
  have h1 : (U.erase i).erase i = U.erase i := Finset.erase_idem
  have h2 : (univ \ U.erase i).erase i = (univ \ U).erase i := by
    ext v
    simp only [mem_erase, mem_sdiff, mem_univ, true_and]
    tauto
  rw [h1, h2]

/-- A finite product splits off the factor at `i` when `i` belongs to the index set. -/
theorem cc_prod_split_at {Y : Type} [DecidableEq Y] (s : Finset Y) (f : Y → ℝ) (i : Y) :
    ∏ v ∈ s, f v = (if i ∈ s then f i else 1) * ∏ v ∈ s.erase i, f v := by
  by_cases h : i ∈ s
  · rw [ite_eq_left h, Finset.mul_prod_erase s f h]
  · rw [ite_eq_right h, one_mul, Finset.erase_eq_of_notMem h]

/-- The weight of `U` under `p` with coordinate `i` replaced by `x`: the factor at `i` is
`x` or `1 - x` according to `i ∈ U`, times the reduced weight. -/
theorem cc_weight_update {Y : Type} [Fintype Y] [DecidableEq Y] (p : Y → ℝ) (i : Y) (x : ℝ)
    (U : Finset Y) :
    weight (Function.update p i x) U = (if i ∈ U then x else 1 - x) * cc_wOther p i U := by
  unfold weight cc_wOther
  have h1 := cc_prod_split_at U (Function.update p i x) i
  have h2 := cc_prod_split_at (univ \ U) (fun v => 1 - Function.update p i x v) i
  simp only [Function.update_self] at h1 h2
  have e1 : ∏ v ∈ U.erase i, Function.update p i x v = ∏ v ∈ U.erase i, p v :=
    Finset.prod_congr rfl (fun v hv => Function.update_of_ne (ne_of_mem_erase hv) x p)
  have e2 : ∏ v ∈ (univ \ U).erase i, (1 - Function.update p i x v)
      = ∏ v ∈ (univ \ U).erase i, (1 - p v) :=
    Finset.prod_congr rfl (fun v hv => by rw [Function.update_of_ne (ne_of_mem_erase hv)])
  rw [h1, h2, e1, e2]
  by_cases hi : i ∈ U
  · have hi' : i ∉ univ \ U := by simp [hi]
    rw [ite_eq_left hi, ite_eq_right hi', ite_eq_left hi]
    ring
  · have hi' : i ∈ univ \ U := by simp [hi]
    rw [ite_eq_right hi, ite_eq_left hi', ite_eq_right hi]
    ring

/-- Display (F4): the split of `mu_p(A)` at coordinate `i`, for any family `A` (general finite
ground type). -/
theorem cc_muOf_update {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y))
    (p : Y → ℝ) (i : Y) (x : ℝ) :
    muOf A (Function.update p i x) =
      x * ∑ U ∈ A.filter (fun U => i ∈ U), cc_wOther p i U
        + (1 - x) * ∑ U ∈ A.filter (fun U => i ∉ U), cc_wOther p i U := by
  unfold muOf
  simp only [cc_weight_update, ite_mul]
  rw [Finset.sum_ite, Finset.mul_sum, Finset.mul_sum]

/-- For a downset, the sum over the members containing `i` is at most the sum over the members
avoiding `i` (`U ↦ U.erase i` injects the first into the second and preserves `cc_wOther`). -/
theorem cc_sum_in_le_sum_out {Y : Type} [Fintype Y] [DecidableEq Y] {A : Finset (Finset Y)}
    (hA : ∀ U ∈ A, ∀ V : Finset Y, V ⊆ U → V ∈ A) {p : Y → ℝ}
    (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (i : Y) :
    ∑ U ∈ A.filter (fun U => i ∈ U), cc_wOther p i U
      ≤ ∑ U ∈ A.filter (fun U => i ∉ U), cc_wOther p i U := by
  have hinj : Set.InjOn (fun U : Finset Y => U.erase i)
      (A.filter (fun U => i ∈ U) : Set (Finset Y)) := by
    intro U hU V hV hUV
    have hU' : i ∈ U := (Finset.mem_filter.1 (Finset.mem_coe.1 hU)).2
    have hV' : i ∈ V := (Finset.mem_filter.1 (Finset.mem_coe.1 hV)).2
    exact Finset.erase_injOn' i hU' hV' hUV
  have himg : (A.filter (fun U => i ∈ U)).image (fun U => U.erase i)
      ⊆ A.filter (fun U => i ∉ U) := by
    intro V hV
    rw [Finset.mem_image] at hV
    obtain ⟨U, hU, rfl⟩ := hV
    rw [Finset.mem_filter] at hU ⊢
    exact ⟨hA U hU.1 _ (Finset.erase_subset i U), Finset.notMem_erase i U⟩
  calc ∑ U ∈ A.filter (fun U => i ∈ U), cc_wOther p i U
      = ∑ U ∈ A.filter (fun U => i ∈ U), cc_wOther p i (U.erase i) :=
        Finset.sum_congr rfl (fun U _ => (cc_wOther_erase p i U).symm)
    _ = ∑ V ∈ (A.filter (fun U => i ∈ U)).image (fun U => U.erase i), cc_wOther p i V :=
        (Finset.sum_image hinj).symm
    _ ≤ ∑ U ∈ A.filter (fun U => i ∉ U), cc_wOther p i U :=
        Finset.sum_le_sum_of_subset_of_nonneg himg (fun U _ _ => cc_wOther_nonneg hp i U)

/-- The sum over the members avoiding `i` is positive once `∅ ∈ A` and every `p_v < 1`. -/
theorem cc_sum_out_pos {Y : Type} [Fintype Y] [DecidableEq Y] {A : Finset (Finset Y)}
    (h0 : ∅ ∈ A) {p : Y → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v < 1) (i : Y) :
    0 < ∑ U ∈ A.filter (fun U => i ∉ U), cc_wOther p i U := by
  have hmem : ∅ ∈ A.filter (fun U => i ∉ U) :=
    Finset.mem_filter.2 ⟨h0, Finset.notMem_empty i⟩
  have hp1 : ∀ v, 0 ≤ p v ∧ p v ≤ 1 := fun v => ⟨(hp v).1, (hp v).2.le⟩
  exact lt_of_lt_of_le (cc_wOther_empty_pos (fun v => (hp v).2) i)
    (Finset.single_le_sum (fun U _ => cc_wOther_nonneg hp1 i U) hmem)

/-- `mu_p(A)` is antitone in each coordinate for a downset and `p` in the cube. -/
theorem cc_muOf_update_antitone {Y : Type} [Fintype Y] [DecidableEq Y] {A : Finset (Finset Y)}
    (hA : ∀ U ∈ A, ∀ V : Finset Y, V ⊆ U → V ∈ A) {p : Y → ℝ}
    (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (i : Y) {x y : ℝ} (hxy : x ≤ y) :
    muOf A (Function.update p i y) ≤ muOf A (Function.update p i x) := by
  rw [cc_muOf_update, cc_muOf_update]
  have h := cc_sum_in_le_sum_out hA hp i
  nlinarith [mul_nonneg (sub_nonneg.2 hxy) (sub_nonneg.2 h)]

/-! ### The one-variable function `s ↦ -log(a + b exp(-s))` (display (F5)) -/

/-- `a + b exp(-s) > 0` for `a, b ≥ 0` with `a + b > 0`. -/
theorem cc_affineExp_pos {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : 0 < a + b) (s : ℝ) :
    0 < a + b * Real.exp (-s) := by
  have he := Real.exp_pos (-s)
  rcases lt_or_eq_of_le ha with ha' | ha'
  · have : 0 ≤ b * Real.exp (-s) := mul_nonneg hb he.le
    linarith
  · have hb' : 0 < b := by linarith
    have := mul_pos hb' he
    linarith

/-- First derivative: `d/ds (-log(a + b e^{-s})) = b e^{-s} / (a + b e^{-s})`. -/
theorem cc_hasDerivAt_negLog {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : 0 < a + b) (s : ℝ) :
    HasDerivAt (fun x => -Real.log (a + b * Real.exp (-x)))
      (b * Real.exp (-s) / (a + b * Real.exp (-s))) s := by
  have hpos := cc_affineExp_pos ha hb hab s
  have h1 : HasDerivAt (fun x : ℝ => a + b * Real.exp (-x)) (b * (Real.exp (-s) * -1)) s :=
    ((hasDerivAt_neg' (x := s)).exp.const_mul b).const_add a
  have h2 := (h1.log hpos.ne').neg
  exact h2.congr_deriv (by ring)

/-- Second derivative: `d/ds (b e^{-s} / (a + b e^{-s})) = -a b e^{-s} / (a + b e^{-s})^2`. -/
theorem cc_hasDerivAt_ratio {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : 0 < a + b) (s : ℝ) :
    HasDerivAt (fun x => b * Real.exp (-x) / (a + b * Real.exp (-x)))
      (-(a * b * Real.exp (-s)) / (a + b * Real.exp (-s)) ^ 2) s := by
  have hpos := cc_affineExp_pos ha hb hab s
  have hnum : HasDerivAt (fun x : ℝ => b * Real.exp (-x)) (b * (Real.exp (-s) * -1)) s :=
    (hasDerivAt_neg' (x := s)).exp.const_mul b
  have hden : HasDerivAt (fun x : ℝ => a + b * Real.exp (-x)) (b * (Real.exp (-s) * -1)) s :=
    hnum.const_add a
  have h := hnum.div hden hpos.ne'
  exact h.congr_deriv (by ring)

/-- Display (F5), one variable: `s ↦ -log(a + b exp(-s))` is concave on `[0, 1]` for
`a, b ≥ 0`, `a + b > 0` (`concaveOn_of_deriv2_nonpos`). -/
theorem cc_concaveOn_negLog {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : 0 < a + b) :
    ConcaveOn ℝ (Set.Icc (0 : ℝ) 1) (fun s => -Real.log (a + b * Real.exp (-s))) := by
  have hd1 := cc_hasDerivAt_negLog ha hb hab
  have hd2 := cc_hasDerivAt_ratio ha hb hab
  have hderiv : deriv (fun s => -Real.log (a + b * Real.exp (-s)))
      = fun s => b * Real.exp (-s) / (a + b * Real.exp (-s)) := by
    funext s
    exact (hd1 s).deriv
  refine concaveOn_of_deriv2_nonpos (convex_Icc 0 1) ?_ ?_ ?_ ?_
  · exact fun s _ => (hd1 s).continuousAt.continuousWithinAt
  · exact fun s _ => (hd1 s).differentiableAt.differentiableWithinAt
  · rw [hderiv]
    exact fun s _ => (hd2 s).differentiableAt.differentiableWithinAt
  · intro s _
    show deriv (deriv (fun s => -Real.log (a + b * Real.exp (-s)))) s ≤ 0
    rw [hderiv, (hd2 s).deriv]
    apply div_nonpos_of_nonpos_of_nonneg
    · have : 0 ≤ a * b * Real.exp (-s) := mul_nonneg (mul_nonneg ha hb) (Real.exp_pos _).le
      linarith
    · exact sq_nonneg _

/-! ### Assembly of (F5) -/

/-- `F_A(t[i := s])` is `-log mu(A)` at `p(t)` with coordinate `i` replaced by `1 - exp(-s)`. -/
theorem cc_hazardCoord_update_eq {Y : Type} [Fintype Y] [DecidableEq Y]
    (A : Finset (Finset Y)) (t : Y → ℝ) (i : Y) (s : ℝ) :
    hazardCoord A (Function.update t i s) =
      -Real.log (muOf A (Function.update (fun j => 1 - Real.exp (-t j)) i
        (1 - Real.exp (-s)))) := by
  unfold hazardCoord
  have h : (fun j => 1 - Real.exp (-(Function.update t i s j)))
      = Function.update (fun j => 1 - Real.exp (-t j)) i (1 - Real.exp (-s)) := by
    funext j
    by_cases hj : j = i
    · subst hj
      simp
    · rw [Function.update_of_ne hj, Function.update_of_ne hj]
  rw [h]

/-- Display (F5) on a general finite ground type: for a downset containing `∅` and `t` in the
cube, `s ↦ F_A(t[i := s])` is concave on `[0, 1]`. -/
theorem cc_concaveOn_hazardCoord_update {Y : Type} [Fintype Y] [DecidableEq Y]
    {A : Finset (Finset Y)} (h0 : ∅ ∈ A) (hdown : ∀ U ∈ A, ∀ V : Finset Y, V ⊆ U → V ∈ A)
    (i : Y) (t : Y → ℝ) (ht : ∀ j, 0 ≤ t j ∧ t j ≤ 1) :
    ConcaveOn ℝ (Set.Icc (0 : ℝ) 1) (fun s : ℝ => hazardCoord A (Function.update t i s)) := by
  obtain ⟨p, hp_def⟩ : ∃ p : Y → ℝ, p = fun j => 1 - Real.exp (-t j) := ⟨_, rfl⟩
  have hp : ∀ v, 0 ≤ p v ∧ p v < 1 := by
    intro v
    rw [hp_def]
    show 0 ≤ 1 - Real.exp (-t v) ∧ 1 - Real.exp (-t v) < 1
    have h1 := (ht v).1
    have he : Real.exp (-t v) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      linarith
    have he' : 0 < Real.exp (-t v) := Real.exp_pos _
    constructor <;> linarith
  have hp1 : ∀ v, 0 ≤ p v ∧ p v ≤ 1 := fun v => ⟨(hp v).1, (hp v).2.le⟩
  have hle := cc_sum_in_le_sum_out hdown hp1 i
  have hpos := cc_sum_out_pos h0 hp i
  have hSin0 : 0 ≤ ∑ U ∈ A.filter (fun U => i ∈ U), cc_wOther p i U :=
    Finset.sum_nonneg fun U _ => cc_wOther_nonneg hp1 i U
  have hfun : (fun s : ℝ => hazardCoord A (Function.update t i s)) =
      fun s => -Real.log ((∑ U ∈ A.filter (fun U => i ∈ U), cc_wOther p i U)
        + ((∑ U ∈ A.filter (fun U => i ∉ U), cc_wOther p i U)
          - ∑ U ∈ A.filter (fun U => i ∈ U), cc_wOther p i U) * Real.exp (-s)) := by
    funext s
    rw [cc_hazardCoord_update_eq, ← hp_def, cc_muOf_update]
    rw [show ∀ x Sin Sout : ℝ, (1 - x) * Sin + (1 - (1 - x)) * Sout = Sin + (Sout - Sin) * x
      from fun x Sin Sout => by ring]
  rw [hfun]
  exact cc_concaveOn_negLog hSin0 (sub_nonneg.2 hle) (by linarith)

/-! ## (F6) The hazard product bound `F_A(t) ≥ (prod_i t_i) H_A`

### Elementary bounds on `mu_p(A)` and on the hazard `F_A` -/

/-- `mu_p(A) ≤ 1` for `p` in the cube: `A ⊆ univ.powerset`, the weights are nonnegative
(`weight_nonneg`) and all of them sum to one (`sum_weight_eq_one`). -/
theorem f6_muOf_le_one {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y))
    {p : Y → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) : muOf A p ≤ 1 := by
  unfold muOf
  rw [← UniversalLaw.Proofs.P15Elementary.Aux.sum_weight_eq_one p]
  exact Finset.sum_le_sum_of_subset_of_nonneg
    (fun U _ => Finset.mem_powerset.2 (Finset.subset_univ U))
    (fun U _ _ => UniversalLaw.Proofs.P15Budget.Aux.weight_nonneg hp U)

/-- `0 ≤ mu_p(A)` for `p` in the cube. -/
theorem f6_muOf_nonneg {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y))
    {p : Y → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) : 0 ≤ muOf A p :=
  Finset.sum_nonneg fun U _ => UniversalLaw.Proofs.P15Budget.Aux.weight_nonneg hp U

/-- `1 - exp(-u) ∈ [0, 1]` for `u ∈ [0, 1]` (indeed for every `u ≥ 0`). -/
theorem f6_one_sub_exp_neg_mem {u : ℝ} (hu : 0 ≤ u ∧ u ≤ 1) :
    0 ≤ 1 - Real.exp (-u) ∧ 1 - Real.exp (-u) ≤ 1 := by
  constructor
  · have h : Real.exp (-u) ≤ 1 := Real.exp_le_one_iff.2 (by linarith [hu.1])
    linarith
  · have h := Real.exp_pos (-u)
    linarith

/-- The coordinates `p_j(u) = 1 - exp(-u_j)` lie in the cube when `u` does. -/
theorem f6_coords_mem {Y : Type} {u : Y → ℝ} (hu : ∀ j, 0 ≤ u j ∧ u j ≤ 1) :
    ∀ v, 0 ≤ 1 - Real.exp (-u v) ∧ 1 - Real.exp (-u v) ≤ 1 :=
  fun v => f6_one_sub_exp_neg_mem (hu v)

/-- `0 ≤ F_A(u)` on the cube: `-log m ≥ 0` for `0 ≤ m ≤ 1` (`Real.log_nonpos`; `log 0 = 0`, so
no positivity of `mu` is needed). -/
theorem f6_hazardCoord_nonneg {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y))
    {u : Y → ℝ} (hu : ∀ j, 0 ≤ u j ∧ u j ≤ 1) : 0 ≤ hazardCoord A u := by
  unfold hazardCoord
  have hp := f6_coords_mem hu
  have h := Real.log_nonpos (f6_muOf_nonneg A hp) (f6_muOf_le_one A hp)
  linarith

/-- `F_A(1, …, 1) = H_A`, since `1 - exp(-1) = p_star`. -/
theorem f6_hazardCoord_one {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y)) :
    hazardCoord A (fun _ => (1 : ℝ)) = hazardStar A := by
  unfold hazardCoord hazardStar pStar
  rfl

/-! ### The chord inequality and its iteration -/

/-- Chord inequality at one coordinate: if `s ↦ F_A(update u i s)` is concave on `[0, 1]` and
`u` lies in the cube, then `u_i · F_A(update u i 1) ≤ F_A(u)`. -/
theorem f6_chord {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y)) (i : Y)
    (u : Y → ℝ) (hu : ∀ j, 0 ≤ u j ∧ u j ≤ 1)
    (hconc : ConcaveOn ℝ (Set.Icc (0 : ℝ) 1)
      (fun s : ℝ => hazardCoord A (Function.update u i s))) :
    u i * hazardCoord A (Function.update u i 1) ≤ hazardCoord A u := by
  have h0 : (0 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := Set.left_mem_Icc.2 zero_le_one
  have h1 : (1 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := Set.right_mem_Icc.2 zero_le_one
  have hs := hu i
  have hc := hconc.2 h0 h1 (by linarith [hs.2] : (0 : ℝ) ≤ 1 - u i) hs.1 (by ring)
  simp only [smul_eq_mul, mul_zero, zero_add, mul_one, Function.update_eq_self] at hc
  have hf0 : 0 ≤ hazardCoord A (Function.update u i 0) := by
    apply f6_hazardCoord_nonneg
    intro j
    rcases eq_or_ne j i with rfl | hj
    · simp
    · rw [Function.update_of_ne hj]
      exact hu j
  have hprod := mul_nonneg (sub_nonneg.2 hs.2) hf0
  linarith

/-- The vector `u_s` which equals `1` on `s` and `t` elsewhere lies in the cube when `t` does. -/
theorem f6_raised_mem {Y : Type} [DecidableEq Y] {t : Y → ℝ} (ht : ∀ j, 0 ≤ t j ∧ t j ≤ 1)
    (s : Finset Y) :
    ∀ j, 0 ≤ (if j ∈ s then (1 : ℝ) else t j) ∧ (if j ∈ s then (1 : ℝ) else t j) ≤ 1 := by
  intro j
  split_ifs
  · exact ⟨zero_le_one, le_refl 1⟩
  · exact ht j

/-- Raising one more coordinate: `u_{insert a s} = update u_s a 1`. -/
theorem f6_raised_insert {Y : Type} [DecidableEq Y] (t : Y → ℝ) (a : Y) (s : Finset Y) :
    (fun j => if j ∈ insert a s then (1 : ℝ) else t j) =
      Function.update (fun j => if j ∈ s then (1 : ℝ) else t j) a 1 := by
  funext j
  by_cases hj : j = a
  · subst hj
    simp
  · simp [Finset.mem_insert, hj]

/-- Iterated chord inequality: for every finite set `s` of coordinates,
`(prod_{j ∈ s} t_j) · F_A(u_s) ≤ F_A(t)`, where `u_s` is `t` with the coordinates in `s` raised
to `1`. Induction over `s` (`Finset.induction_on`) with `f6_chord` at each step. -/
theorem f6_prod_hazardCoord_le {Y : Type} [Fintype Y] [DecidableEq Y] (A : Finset (Finset Y))
    (hconc : ∀ (i : Y) (u : Y → ℝ), (∀ j, 0 ≤ u j ∧ u j ≤ 1) →
      ConcaveOn ℝ (Set.Icc (0 : ℝ) 1) (fun s : ℝ => hazardCoord A (Function.update u i s)))
    (t : Y → ℝ) (ht : ∀ j, 0 ≤ t j ∧ t j ≤ 1) (s : Finset Y) :
    (∏ j ∈ s, t j) * hazardCoord A (fun j => if j ∈ s then (1 : ℝ) else t j) ≤
      hazardCoord A t := by
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    have hus := f6_raised_mem ht s
    have hchord := f6_chord A a (fun j => if j ∈ s then (1 : ℝ) else t j) hus (hconc a _ hus)
    simp only [ite_eq_right ha] at hchord
    have hprod : 0 ≤ ∏ j ∈ s, t j := Finset.prod_nonneg fun j _ => (ht j).1
    rw [Finset.prod_insert ha, f6_raised_insert t a s]
    calc t a * (∏ j ∈ s, t j) *
          hazardCoord A (Function.update (fun j => if j ∈ s then (1 : ℝ) else t j) a 1)
        = (∏ j ∈ s, t j) *
          (t a * hazardCoord A (Function.update (fun j => if j ∈ s then (1 : ℝ) else t j) a 1)) := by
          ring
      _ ≤ (∏ j ∈ s, t j) * hazardCoord A (fun j => if j ∈ s then (1 : ℝ) else t j) :=
          mul_le_mul_of_nonneg_left hchord hprod
      _ ≤ hazardCoord A t := ih

/-- Display (F6) from display (F5): assuming `CoordinatewiseConcavity`, the hazard product bound
holds. Takes `s = univ` in `f6_prod_hazardCoord_le`, where `u_univ = (1, …, 1)` and
`F_A(1, …, 1) = H_A` (`f6_hazardCoord_one`). -/
theorem f6_hazardProductBound_of_concavity (h5 : UniversalLaw.Spec.P15.CoordinatewiseConcavity) :
    UniversalLaw.Spec.P15.HazardProductBound := by
  intro n A hn hA t ht
  have hconc : ∀ (i : Fin n) (u : Fin n → ℝ), (∀ j, 0 ≤ u j ∧ u j ≤ 1) →
      ConcaveOn ℝ (Set.Icc (0 : ℝ) 1) (fun s : ℝ => hazardCoord A (Function.update u i s)) :=
    fun i u hu => h5 n A hn hA i u hu
  have key := f6_prod_hazardCoord_le A hconc t ht (univ : Finset (Fin n))
  have hfun : (fun j : Fin n => if j ∈ (univ : Finset (Fin n)) then (1 : ℝ) else t j) =
      fun _ => (1 : ℝ) := by
    funext j
    simp
  rw [hfun, f6_hazardCoord_one] at key
  exact key

/-! ## (F7) The hazard `phi` bound `-log mu_p(A) ≥ H_A prod_i phi(p_i)`

### (a) The transformed price `phi` maps `[0,1]` into `[0,1]` -/

/-- `0 ≤ phi x` for `x ≥ 0`: when `x < 1`, `-log(1 - x) ≥ 0` since `0 < 1 - x ≤ 1`. -/
theorem f7_phi_nonneg {x : ℝ} (hx : 0 ≤ x) : 0 ≤ phi x := by
  unfold phi
  split_ifs with h
  · refine le_min zero_le_one ?_
    have hlog : Real.log (1 - x) ≤ 0 := Real.log_nonpos (by linarith) (by linarith)
    linarith
  · exact zero_le_one

/-- `phi x ≤ 1` for every real `x`. -/
theorem f7_phi_le_one (x : ℝ) : phi x ≤ 1 := by
  unfold phi
  split_ifs with h
  · exact min_le_left _ _
  · exact le_refl 1

/-- `phi` maps `[0,1]` into `[0,1]`. -/
theorem f7_phi_mem {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : 0 ≤ phi x ∧ phi x ≤ 1 :=
  ⟨f7_phi_nonneg hx.1, f7_phi_le_one x⟩

/-! ### (b) `p' = 1 - exp(-phi p)` lies in `[0, p]` -/

/-- `0 ≤ 1 - exp(-phi x)` for `x ≥ 0` (since `phi x ≥ 0`). -/
theorem f7_one_sub_exp_neg_phi_nonneg {x : ℝ} (hx : 0 ≤ x) : 0 ≤ 1 - Real.exp (-phi x) := by
  have h : Real.exp (-phi x) ≤ 1 := Real.exp_le_one_iff.2 (by linarith [f7_phi_nonneg hx])
  linarith

/-- `1 - exp(-phi x) ≤ x` on `[0,1]`: for `x < 1`, `phi x ≤ -log(1 - x)` so
`exp(-phi x) ≥ exp(log(1 - x)) = 1 - x`; for `x = 1`, `1 - exp(-1) ≤ 1`. -/
theorem f7_one_sub_exp_neg_phi_le {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) :
    1 - Real.exp (-phi x) ≤ x := by
  by_cases h : x < 1
  · have hphi : phi x ≤ -Real.log (1 - x) := by
      unfold phi
      rw [ite_eq_left h]
      exact min_le_right _ _
    have hpos : 0 < 1 - x := by linarith
    have h1 : Real.exp (Real.log (1 - x)) ≤ Real.exp (-phi x) :=
      Real.exp_le_exp.2 (by linarith)
    rw [Real.exp_log hpos] at h1
    linarith
  · have hx1 : x = 1 := le_antisymm hx.2 (not_lt.1 h)
    have hphi : phi x = 1 := by
      unfold phi
      rw [ite_eq_right h]
    rw [hphi, hx1]
    have := Real.exp_pos (-1 : ℝ)
    linarith

/-! ### (c) Antitonicity of `mu_p(A)` in the probability vector (from the split (F4)) -/

/-- The interpolating vector `r_s` (`q` on `s`, `p` elsewhere) lies in the cube when both do. -/
theorem f7_mix_mem {Y : Type} [DecidableEq Y] {p q : Y → ℝ} (hp : ∀ j, 0 ≤ p j ∧ p j ≤ 1)
    (hq : ∀ j, 0 ≤ q j ∧ q j ≤ 1) (s : Finset Y) :
    ∀ j, 0 ≤ (if j ∈ s then q j else p j) ∧ (if j ∈ s then q j else p j) ≤ 1 := by
  intro j
  split_ifs
  · exact hq j
  · exact hp j

/-- Switching one more coordinate to `q`: `r_{insert a s} = update r_s a (q a)`. -/
theorem f7_mix_insert {Y : Type} [DecidableEq Y] (p q : Y → ℝ) (a : Y) (s : Finset Y) :
    (fun j => if j ∈ insert a s then q j else p j) =
      Function.update (fun j => if j ∈ s then q j else p j) a (q a) := by
  funext j
  by_cases hj : j = a
  · subst hj
    simp
  · simp [Finset.mem_insert, hj]

/-- For `a ∉ s`, `r_s` is its own update at `a` by `p a`. -/
theorem f7_mix_eq_update_self {Y : Type} [DecidableEq Y] (p q : Y → ℝ) (a : Y) (s : Finset Y)
    (ha : a ∉ s) :
    (fun j => if j ∈ s then q j else p j) =
      Function.update (fun j => if j ∈ s then q j else p j) a (p a) := by
  funext j
  by_cases hj : j = a
  · subst hj
    simp [ha]
  · simp [hj]

/-- Moving from `p` down to `q` one coordinate at a time: for every finite set `s` of
coordinates, `mu_p(A) ≤ mu_(r_s)(A)` (induction over `s`, `cc_muOf_update_antitone` at each
step). -/
theorem f7_muOf_le_mix {Y : Type} [Fintype Y] [DecidableEq Y] {A : Finset (Finset Y)}
    (hA : ∀ U ∈ A, ∀ V : Finset Y, V ⊆ U → V ∈ A) {p q : Y → ℝ}
    (hp : ∀ j, 0 ≤ p j ∧ p j ≤ 1) (hq : ∀ j, 0 ≤ q j ∧ q j ≤ 1) (hqp : ∀ j, q j ≤ p j)
    (s : Finset Y) :
    muOf A p ≤ muOf A (fun j => if j ∈ s then q j else p j) := by
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    have hmem := f7_mix_mem hp hq s
    have hstep := cc_muOf_update_antitone hA hmem a (hqp a)
    rw [← f7_mix_eq_update_self p q a s ha] at hstep
    rw [f7_mix_insert p q a s]
    exact ih.trans hstep

/-- Antitonicity of `mu(A)` in the probability vector for a downset: `q ≤ p` pointwise, both in
the cube, gives `mu_p(A) ≤ mu_q(A)`. -/
theorem f7_muOf_antitone {Y : Type} [Fintype Y] [DecidableEq Y] {A : Finset (Finset Y)}
    (hA : ∀ U ∈ A, ∀ V : Finset Y, V ⊆ U → V ∈ A) {p q : Y → ℝ}
    (hp : ∀ j, 0 ≤ p j ∧ p j ≤ 1) (hq : ∀ j, 0 ≤ q j ∧ q j ≤ 1) (hqp : ∀ j, q j ≤ p j) :
    muOf A p ≤ muOf A q := by
  have h := f7_muOf_le_mix hA hp hq hqp (univ : Finset Y)
  have hfun : (fun j : Y => if j ∈ (univ : Finset Y) then q j else p j) = q := by
    funext j
    simp
  rw [hfun] at h
  exact h

/-! ### (d) Display (F7) from display (F6) -/

/-- Display (F7) from display (F6): assuming `HazardProductBound`, the hazard `phi` bound holds.
With `t_i := phi(p_i) ∈ [0,1]` and `p'_i := 1 - exp(-t_i) ∈ [0, p_i]`, antitonicity gives
`0 < mu_p(A) ≤ mu_p'(A)`, so `-log mu_p(A) ≥ -log mu_p'(A) = F_A(t) ≥ (prod_i t_i) H_A`. -/
theorem f7_hazardPhiBound_of_productBound (h6 : UniversalLaw.Spec.P15.HazardProductBound) :
    UniversalLaw.Spec.P15.HazardPhiBound := by
  intro n A hn hA p hp hpos
  have hdown : ∀ U ∈ A, ∀ V : Finset (Fin n), V ⊆ U → V ∈ A := hA.2.2
  have ht : ∀ j, 0 ≤ phi (p j) ∧ phi (p j) ≤ 1 := fun j => f7_phi_mem (hp j)
  have hp' : ∀ j, 0 ≤ 1 - Real.exp (-phi (p j)) ∧ 1 - Real.exp (-phi (p j)) ≤ 1 :=
    fun j => ⟨f7_one_sub_exp_neg_phi_nonneg (hp j).1,
      (f7_one_sub_exp_neg_phi_le (hp j)).trans (hp j).2⟩
  have hle : muOf A p ≤ muOf A (fun j => 1 - Real.exp (-phi (p j))) :=
    f7_muOf_antitone hdown hp hp' (fun j => f7_one_sub_exp_neg_phi_le (hp j))
  have hlog : -Real.log (muOf A (fun j => 1 - Real.exp (-phi (p j)))) ≤ -Real.log (muOf A p) :=
    neg_le_neg (Real.log_le_log hpos hle)
  have h6' := h6 n A hn hA (fun j => phi (p j)) ht
  have hcoord : hazardCoord A (fun j => phi (p j)) =
      -Real.log (muOf A (fun j => 1 - Real.exp (-phi (p j)))) := rfl
  rw [hcoord] at h6'
  calc hazardStar A * ∏ i, phi (p i) = (∏ i, phi (p i)) * hazardStar A := mul_comm _ _
    _ ≤ -Real.log (muOf A (fun j => 1 - Real.exp (-phi (p j)))) := h6'
    _ ≤ -Real.log (muOf A p) := hlog

/-! ## (F8) The full-ground price bound -/

/-- `prod_i phi(p_i) ≤ 1` for `p ∈ [0, 1]^n`, from `0 ≤ phi ≤ 1` on `[0, 1]`
(`Finset.prod_le_one₀`). -/
theorem f8_prod_phi_le_one {n : ℕ} (p : Fin n → ℝ) (hp : ∀ j, 0 ≤ p j ∧ p j ≤ 1) :
    ∏ i, phi (p i) ≤ 1 :=
  Finset.prod_le_one₀ (fun i _ => f7_phi_nonneg (hp i).1) (fun i _ => f7_phi_le_one (p i))

/-- `prod_i c_i ≤ prod_i phi(p_i)` for `0 ≤ c_i ≤ phi(p_i)` (`Finset.prod_le_prod₀`). -/
theorem f8_prod_le_prod_phi {n : ℕ} (p c : Fin n → ℝ) (hc : ∀ j, 0 ≤ c j ∧ c j ≤ phi (p j)) :
    ∏ i, c i ≤ ∏ i, phi (p i) :=
  Finset.prod_le_prod₀ (fun i _ => (hc i).1) (fun i _ => (hc i).2)

/-- Display (F8) from display (F7): given `HazardPhiBound`, the full-ground price bound holds.
Fix `n`, a proper downset `A` on `Fin n`, `p ∈ [0, 1]^n` and `0 ≤ c ≤ phi ∘ p`. The cap `1` is
`f8_prod_le_prod_phi` and `f8_prod_phi_le_one`; when `0 < mu_p(A)`, (F7) gives
`H_A prod_i phi(p_i) ≤ -log mu_p(A)` and `0 < H_A` (`P15Elementary.hazardStarPositive`), so
`prod_i phi(p_i) = H_A⁻¹ (H_A prod_i phi(p_i)) ≤ H_A⁻¹ [-log mu_p(A)]` (`inv_mul_cancel_left₀`,
`mul_le_mul_of_nonneg_left`); `P15Budget.Aux.le_cappedHazard_of` reaches the capped hazard. -/
theorem f8_fullGroundPriceBound_of_phiBound (h7 : UniversalLaw.Spec.P15.HazardPhiBound) :
    UniversalLaw.Spec.P15.FullGroundPriceBound := by
  intro n A hn hA p c hp hc
  have hprod : ∏ i, c i ≤ ∏ i, phi (p i) := f8_prod_le_prod_phi p c hc
  have hphi1 : ∏ i, phi (p i) ≤ 1 := f8_prod_phi_le_one p hp
  refine UniversalLaw.Proofs.P15Budget.Aux.le_cappedHazard_of (hprod.trans hphi1) fun hm => ?_
  have hH : 0 < hazardStar A := UniversalLaw.Proofs.P15Elementary.hazardStarPositive n A hn hA
  have hne : hazardStar A ≠ 0 := hH.ne'
  have h7' : hazardStar A * ∏ i, phi (p i) ≤ -Real.log (muOf A p) := h7 n A hn hA p hp hm
  calc ∏ i, c i ≤ ∏ i, phi (p i) := hprod
    _ = (hazardStar A)⁻¹ * (hazardStar A * ∏ i, phi (p i)) :=
        (inv_mul_cancel_left₀ hne _).symm
    _ ≤ (hazardStar A)⁻¹ * -Real.log (muOf A p) :=
        mul_le_mul_of_nonneg_left h7' (inv_nonneg.mpr hH.le)
    _ = 1 / hazardStar A * -Real.log (muOf A p) := by rw [one_div]

/-! ## (F8), exactness remark: the full-ground budget equivalence -/

/-- `0 ≤ prod_i phi(p_i)` for `p ∈ [0, 1]^n` (`Finset.prod_nonneg`). -/
theorem bi_prod_phi_nonneg {n : ℕ} (p : Fin n → ℝ) (hp : ∀ j, 0 ≤ p j ∧ p j ≤ 1) :
    0 ≤ ∏ i, phi (p i) :=
  Finset.prod_nonneg fun i _ => f7_phi_nonneg (hp i).1

/-- The (⇐) direction, display (F8) with `rho = 1`: given display (F7) and `1 ≤ H_A`, every
`p ∈ [0, 1]^n` and `0 ≤ c ≤ phi ∘ p` satisfy `prod_i c_i ≤ cappedHazard 1 (mu_p(A))`. The cap
`1` is `f8_prod_le_prod_phi` and `f8_prod_phi_le_one`; when `0 < mu_p(A)`,
`prod_i phi(p_i) ≤ H_A prod_i phi(p_i)` (`le_mul_of_one_le_left`, from `0 ≤ prod_i phi(p_i)`
and `1 ≤ H_A`) and `H_A prod_i phi(p_i) ≤ -log mu_p(A)` by (F7);
`P15Budget.Aux.le_cappedHazard_of` reaches the capped hazard. -/
theorem bi_budget_of_one_le_hazardStar (h7 : UniversalLaw.Spec.P15.HazardPhiBound) {n : ℕ}
    {A : Finset (Finset (Fin n))} (hn : 1 ≤ n) (hA : IsProperDownset A)
    (h1 : 1 ≤ hazardStar A) (p c : Fin n → ℝ) (hp : ∀ j, 0 ≤ p j ∧ p j ≤ 1)
    (hc : ∀ j, 0 ≤ c j ∧ c j ≤ phi (p j)) :
    ∏ i, c i ≤ cappedHazard 1 (muOf A p) := by
  have hprod : ∏ i, c i ≤ ∏ i, phi (p i) := f8_prod_le_prod_phi p c hc
  have hphi1 : ∏ i, phi (p i) ≤ 1 := f8_prod_phi_le_one p hp
  refine UniversalLaw.Proofs.P15Budget.Aux.le_cappedHazard_of (hprod.trans hphi1) fun hm => ?_
  have hphi0 : 0 ≤ ∏ i, phi (p i) := bi_prod_phi_nonneg p hp
  have h7' : hazardStar A * ∏ i, phi (p i) ≤ -Real.log (muOf A p) := h7 n A hn hA p hp hm
  calc ∏ i, c i ≤ ∏ i, phi (p i) := hprod
    _ ≤ hazardStar A * ∏ i, phi (p i) := le_mul_of_one_le_left hphi0 h1
    _ ≤ -Real.log (muOf A p) := h7'
    _ = 1 * -Real.log (muOf A p) := (one_mul _).symm

/-- `0 < mu_{p_star}(A)` for a proper downset `A`: every weight at the constant vector
`p_star ∈ (0, 1)` is nonnegative (`P15Budget.Aux.weight_nonneg`), and the member `∅ ∈ A`
contributes `prod_{v} (1 - p_star) > 0` (`Finset.prod_pos`), so `Finset.single_le_sum` gives
the claim. -/
theorem bi_muOf_pStar_pos {n : ℕ} {A : Finset (Finset (Fin n))} (hA : IsProperDownset A) :
    0 < muOf A (fun _ => pStar) := by
  have hp0 : 0 < pStar := UniversalLaw.Proofs.P15Elementary.Aux.pStar_pos
  have hp1 : pStar < 1 := UniversalLaw.Proofs.P15Elementary.Aux.pStar_lt_one
  have hw_nonneg : ∀ U : Finset (Fin n), 0 ≤ weight (fun _ : Fin n => pStar) U :=
    fun U => UniversalLaw.Proofs.P15Budget.Aux.weight_nonneg (fun _ => ⟨hp0.le, hp1.le⟩) U
  have hempty_w : 0 < weight (fun _ : Fin n => pStar) (∅ : Finset (Fin n)) := by
    unfold weight
    rw [prod_empty, one_mul]
    exact Finset.prod_pos fun v _ => by
      show (0 : ℝ) < 1 - pStar
      linarith
  unfold muOf
  exact lt_of_lt_of_le hempty_w (Finset.single_le_sum (fun U _ => hw_nonneg U) hA.1)

/-- The (⇒) direction: if every `p ∈ [0, 1]^n` and `0 ≤ c ≤ phi ∘ p` satisfy
`prod_i c_i ≤ cappedHazard 1 (mu_p(A))`, then `1 ≤ H_A`. Instantiate at `p ≡ p_star`, `c ≡ 1`,
admissible since `0 ≤ p_star ≤ 1` and `phi p_star = 1` (`P15Elementary.phiAtPStar`); then
`prod_i 1 = 1` (`Finset.prod_const_one`), `0 < mu_{p_star}(A)` (`bi_muOf_pStar_pos`) selects
the branch `min 1 (1 * -log mu_{p_star}(A))` of `cappedHazard`, and `le_min_iff` extracts
`1 ≤ -log mu_{p_star}(A)`, which is `hazardStar A` by definition. -/
theorem bi_one_le_hazardStar_of_budget {n : ℕ} {A : Finset (Finset (Fin n))}
    (hA : IsProperDownset A)
    (hall : ∀ (p c : Fin n → ℝ), (∀ j, 0 ≤ p j ∧ p j ≤ 1) → (∀ j, 0 ≤ c j ∧ c j ≤ phi (p j)) →
      ∏ i, c i ≤ cappedHazard 1 (muOf A p)) :
    1 ≤ hazardStar A := by
  have hp0 : 0 < pStar := UniversalLaw.Proofs.P15Elementary.Aux.pStar_pos
  have hp1 : pStar < 1 := UniversalLaw.Proofs.P15Elementary.Aux.pStar_lt_one
  have hphi : phi pStar = 1 := UniversalLaw.Proofs.P15Elementary.phiAtPStar
  have hp : ∀ j : Fin n, 0 ≤ (fun _ : Fin n => pStar) j ∧ (fun _ : Fin n => pStar) j ≤ 1 :=
    fun _ => ⟨hp0.le, hp1.le⟩
  have hc : ∀ j : Fin n, 0 ≤ (fun _ : Fin n => (1 : ℝ)) j ∧
      (fun _ : Fin n => (1 : ℝ)) j ≤ phi ((fun _ : Fin n => pStar) j) :=
    fun _ => ⟨zero_le_one, hphi.symm.le⟩
  have h := hall (fun _ => pStar) (fun _ => (1 : ℝ)) hp hc
  simp only [Finset.prod_const_one] at h
  have hm : 0 < muOf A (fun _ => pStar) := bi_muOf_pStar_pos hA
  unfold cappedHazard at h
  simp only [hm, ↓reduceIte, one_mul] at h
  unfold hazardStar
  exact (le_min_iff.1 h).2

/-- The budget equivalence from display (F7): given `HazardPhiBound`, for `n ≥ 1` and a proper
downset `A` on `Fin n`, the universal capped budget at `rho = 1` holds iff `1 ≤ H_A`. The two
directions are `bi_one_le_hazardStar_of_budget` and `bi_budget_of_one_le_hazardStar`. -/
theorem bi_fullGroundBudgetIff_of_phiBound (h7 : UniversalLaw.Spec.P15.HazardPhiBound) :
    UniversalLaw.Spec.P15.FullGroundBudgetIff := by
  intro n A hn hA
  constructor
  · intro hall
    exact bi_one_le_hazardStar_of_budget hA hall
  · intro h1 p c hp hc
    exact bi_budget_of_one_le_hazardStar h7 hn hA h1 p c hp hc

/-! ## (F12) The local price bound

### Elementary facts -/

/-- The capacity family `{T ⊆ Fin n : |T| ≤ a}` is a proper downset when `a < n`: it contains
the empty set, misses `univ` (of size `n > a`), and is closed under taking subsets. -/
theorem lp_capacityFamily_properDownset (n a : ℕ) (h : a < n) :
    IsProperDownset (capacityFamily n a) := by
  refine ⟨?_, ?_, ?_⟩
  · show ∅ ∈ (univ : Finset (Fin n)).powerset.filter (fun U => U.card ≤ a)
    rw [mem_filter, mem_powerset, card_empty]
    exact ⟨empty_subset _, Nat.zero_le _⟩
  · intro huniv
    have hcard : (univ : Finset (Fin n)).card ≤ a := (mem_filter.mp huniv).2
    rw [card_univ, Fintype.card_fin] at hcard
    exact absurd hcard (not_le.mpr h)
  · intro U hU V hVU
    have hU' : U.card ≤ a := (mem_filter.mp hU).2
    exact mem_filter.mpr ⟨mem_powerset.mpr (subset_univ V), le_trans (card_le_card hVU) hU'⟩

/-! ### Enumerating a finite set by `Fin (card)` -/

/-- Every `B : Finset X` is enumerated injectively by `Fin B.card`, with range exactly `B`. -/
theorem lp_exists_enum {X : Type} (B : Finset X) :
    ∃ e : Fin B.card → X, Function.Injective e ∧ ∀ v, v ∈ B ↔ ∃ k, e k = v := by
  refine ⟨fun k => (B.equivFin.symm k).1, ?_, ?_⟩
  · intro k₁ k₂ h
    exact B.equivFin.symm.injective (Subtype.ext h)
  · intro v
    constructor
    · intro hv
      exact ⟨B.equivFin ⟨v, hv⟩, by simp⟩
    · rintro ⟨k, rfl⟩
      exact (B.equivFin.symm k).2

/-- An injection `e : Fin n → X` with range `B` maps `univ` onto `B`. -/
theorem lp_map_univ_eq {X : Type} {B : Finset X} {n : ℕ} (e : Fin n → X)
    (he : Function.Injective e) (hrange : ∀ v, v ∈ B ↔ ∃ k, e k = v) :
    (univ : Finset (Fin n)).map ⟨e, he⟩ = B := by
  ext v
  rw [hrange v]
  simp only [mem_map, mem_univ, true_and, Function.Embedding.coeFn_mk]

/-- Products over `B` are products over `Fin n` along the enumeration. -/
theorem lp_prod_univ_eq {X : Type} {B : Finset X} {n : ℕ} (e : Fin n → X)
    (he : Function.Injective e) (hrange : ∀ v, v ∈ B ↔ ∃ k, e k = v) (f : X → ℝ) :
    ∏ k, f (e k) = ∏ v ∈ B, f v := by
  rw [← lp_map_univ_eq e he hrange, Finset.prod_map]
  rfl

/-- For `S ⊆ B`, the image of the preimage `{k : e k ∈ S}` is `S`. -/
theorem lp_map_filter_eq {X : Type} [DecidableEq X] {B : Finset X} {n : ℕ} (e : Fin n → X)
    (he : Function.Injective e) (hrange : ∀ v, v ∈ B ↔ ∃ k, e k = v) {S : Finset X}
    (hS : S ⊆ B) :
    ((univ : Finset (Fin n)).filter (fun k => e k ∈ S)).map ⟨e, he⟩ = S := by
  ext v
  simp only [mem_map, mem_filter, mem_univ, true_and, Function.Embedding.coeFn_mk]
  constructor
  · rintro ⟨k, hk, rfl⟩
    exact hk
  · intro hv
    obtain ⟨k, rfl⟩ := (hrange v).mp (hS hv)
    exact ⟨k, hv, rfl⟩

/-- The preimage of the image of `T` is `T` (injectivity of the enumeration). -/
theorem lp_filter_map_eq {X : Type} [DecidableEq X] {n : ℕ} (e : Fin n → X)
    (he : Function.Injective e) (T : Finset (Fin n)) :
    (univ : Finset (Fin n)).filter (fun k => e k ∈ T.map ⟨e, he⟩) = T := by
  ext k
  simp only [mem_filter, mem_univ, true_and, mem_map, Function.Embedding.coeFn_mk]
  constructor
  · rintro ⟨a, ha, hak⟩
    rwa [he hak] at ha
  · intro hk
    exact ⟨k, hk, rfl⟩

/-! ### Transport of the local good probability to the capacity family -/

/-- Along an enumeration `e : Fin n → X` of the block `X_i`, the local good probability
`mu_p(D_i)` is the product measure of the capacity family `{T : |T| ≤ a_i}` on `Fin n` under
the transported vector `p ∘ e`. Uses `LocalRestriction` (a block subset is good iff its size is
at most `a_i`) and the bijection `T ↦ T.map e` between subsets of `Fin n` and subsets of the
block, which preserves size and matches the two weight products. -/
theorem lp_localGood_eq_muOf {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}
    (hF : IsRealizedFamily F) (p : X → ℝ) (i : Fin F.b) {n : ℕ} (e : Fin n → X)
    (he : Function.Injective e) (hrange : ∀ v, v ∈ block F i ↔ ∃ k, e k = v) :
    localGood F p i = muOf (capacityFamily n (F.a i)) (fun k => p (e k)) := by
  have hLR := UniversalLaw.Proofs.P15Covers.localRestriction F hF i
  have hmapuniv := lp_map_univ_eq e he hrange
  have hind : ∀ S ∈ (block F i).powerset,
      Set.indicator (downset F)
        (fun S => (∏ v ∈ S, p v) * ∏ v ∈ block F i \ S, (1 - p v)) S =
      if S.card ≤ F.a i then (∏ v ∈ S, p v) * ∏ v ∈ block F i \ S, (1 - p v) else 0 := by
    intro S hS
    have hgood : S ∈ downset F ↔ S.card ≤ F.a i := by
      show IsGood F S ↔ S.card ≤ F.a i
      exact hLR S (mem_powerset.mp hS)
    by_cases h : S.card ≤ F.a i
    · rw [Set.indicator_of_mem (hgood.mpr h), ite_eq_left h]
    · rw [Set.indicator_of_notMem (fun h' => h (hgood.mp h')), ite_eq_right h]
  unfold localGood
  rw [Finset.sum_congr rfl hind, ← Finset.sum_filter]
  unfold muOf capacityFamily
  symm
  refine Finset.sum_nbij' (fun T => T.map ⟨e, he⟩)
    (fun S => (univ : Finset (Fin n)).filter (fun k => e k ∈ S)) ?_ ?_ ?_ ?_ ?_
  · intro T hT
    rw [mem_filter, mem_powerset] at hT
    rw [mem_filter, mem_powerset]
    refine ⟨?_, ?_⟩
    · rw [← hmapuniv]
      exact map_subset_map.mpr hT.1
    · rw [card_map]
      exact hT.2
  · intro S hS
    rw [mem_filter, mem_powerset] at hS
    rw [mem_filter, mem_powerset]
    refine ⟨subset_univ _, ?_⟩
    rw [← card_map ⟨e, he⟩, lp_map_filter_eq e he hrange hS.1]
    exact hS.2
  · intro T _
    exact lp_filter_map_eq e he T
  · intro S hS
    rw [mem_filter, mem_powerset] at hS
    exact lp_map_filter_eq e he hrange hS.1
  · intro T _
    show weight (fun k => p (e k)) T =
      (∏ v ∈ T.map ⟨e, he⟩, p v) * ∏ v ∈ block F i \ T.map ⟨e, he⟩, (1 - p v)
    unfold weight
    rw [Finset.prod_map, ← hmapuniv, ← Finset.map_sdiff, Finset.prod_map]
    rfl

/-! ### Display (F12) in hypothesis form -/

/-- Display (F12) under the hypothesis `HazardPhiBound` (display (F7)): for every realized
family with `d_i ≥ 2`, every `p ∈ [0,1]^X`, every `0 ≤ c_v ≤ phi(p_v)` and every block with
`mu_p(D_i) > 0`, `price(X_i) ≤ rho_star · (-log mu_p(D_i))` and `price(X_i) ≤ 1`. -/
theorem lp_localPriceBoundFull_of_phiBound (h7 : UniversalLaw.Spec.P15.HazardPhiBound)
    {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.LocalPriceBoundFull F := by
  intro hF hd p c hp hc i hpos
  obtain ⟨e, he, hrange⟩ := lp_exists_enum (block F i)
  have hcard : (block F i).card = F.a i * F.d i + 1 := hF.2.2.1 i
  have ha1 : 1 ≤ F.a i := hF.1 i
  have hd2 : 2 ≤ F.d i := hd i
  have hmul : F.a i * 2 ≤ F.a i * F.d i := Nat.mul_le_mul_left (F.a i) hd2
  have hn1 : 1 ≤ (block F i).card := by rw [hcard]; omega
  have hlt : F.a i < (block F i).card := by rw [hcard]; omega
  have h2a : 2 * F.a i + 1 ≤ (block F i).card := by rw [hcard]; omega
  -- transport to the capacity family on `Fin n`
  have htrans : localGood F p i =
      muOf (capacityFamily (block F i).card (F.a i)) (fun k => p (e k)) :=
    lp_localGood_eq_muOf hF p i e he hrange
  have hproper : IsProperDownset (capacityFamily (block F i).card (F.a i)) :=
    lp_capacityFamily_properDownset _ _ hlt
  have hpos' : 0 < muOf (capacityFamily (block F i).card (F.a i)) (fun k => p (e k)) := by
    rw [← htrans]
    exact hpos
  -- display (F7) on the transported vector
  have h7' : hazardStar (capacityFamily (block F i).card (F.a i)) * ∏ k, phi (p (e k)) ≤
      -Real.log (muOf (capacityFamily (block F i).card (F.a i)) (fun k => p (e k))) :=
    h7 _ _ hn1 hproper (fun k => p (e k)) (fun k => hp (e k)) hpos'
  rw [← htrans] at h7'
  have hprod : ∏ k, phi (p (e k)) = ∏ v ∈ block F i, phi (p v) :=
    lp_prod_univ_eq e he hrange (fun v => phi (p v))
  -- the sharp worst case (F11): `h_star ≤ H_{n,a}` and `0 < h_star`
  obtain ⟨hmono, heq1, heq2, hgt1⟩ := UniversalLaw.Proofs.P15Elementary.capacityWorstCase
  have hHstar : hStar ≤ hazardStar (capacityFamily (block F i).card (F.a i)) := by
    have := hmono (F.a i) _ ha1 h2a
    rwa [heq1, heq2] at this
  have hhpos : 0 < hStar := lt_trans zero_lt_one hgt1
  have hH0 : 0 ≤ hazardStar (capacityFamily (block F i).card (F.a i)) :=
    le_of_lt (lt_of_lt_of_le hhpos hHstar)
  -- the price
  have hG0 : 0 ≤ genPrice c (block F i) := Finset.prod_nonneg (fun v _ => (hc v).1)
  have hGP : genPrice c (block F i) ≤ ∏ v ∈ block F i, phi (p v) :=
    Finset.prod_le_prod₀ (fun v _ => (hc v).1) (fun v _ => (hc v).2)
  have hG1 : genPrice c (block F i) ≤ 1 :=
    Finset.prod_le_one₀ (fun v _ => (hc v).1) (fun v _ => (hc v).2.trans (f7_phi_le_one _))
  refine ⟨?_, hG1⟩
  have key : hStar * genPrice c (block F i) ≤ -Real.log (localGood F p i) := by
    calc hStar * genPrice c (block F i)
        ≤ hazardStar (capacityFamily (block F i).card (F.a i)) * genPrice c (block F i) :=
          mul_le_mul_of_nonneg_right hHstar hG0
      _ ≤ hazardStar (capacityFamily (block F i).card (F.a i)) * ∏ v ∈ block F i, phi (p v) :=
          mul_le_mul_of_nonneg_left hGP hH0
      _ = hazardStar (capacityFamily (block F i).card (F.a i)) * ∏ k, phi (p (e k)) := by
          rw [hprod]
      _ ≤ -Real.log (localGood F p i) := h7'
  have hrho : rhoStar = 1 / hStar := rfl
  rw [hrho, one_div, inv_mul_eq_div, le_div_iff₀ hhpos, mul_comm]
  exact key

/-! ## (F2) The full transformed-price budget -/

/-- `0 ≤ rho_star`: `rho_star = 1 / (3 - log(3e - 2))` is definitionally `1 / h_star`, and
`1 < h_star` (`P15Elementary.Aux.one_lt_hStar`). -/
theorem f2_rhoStar_nonneg : 0 ≤ rhoStar := by
  have h : (1 : ℝ) < 3 - Real.log (3 * Real.exp 1 - 2) :=
    UniversalLaw.Proofs.P15Elementary.Aux.one_lt_hStar
  unfold rhoStar
  exact div_nonneg zero_le_one (by linarith)

/-- Display (F2) from display (F12): given the local price bound for every realized family, the
full transformed-price budget holds for every realized family. The hypothesis `h12` is the
registered `Prop` `LocalPriceBoundFull`, quantified over the ground type so that it can be
discharged by the inventory theorem `localPriceBoundFull`. -/
theorem f2_fullTransformedPriceBudget_of_local
    (h12 : ∀ {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X),
      UniversalLaw.Spec.P15.LocalPriceBoundFull F)
    {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.FullTransformedPriceBudget F := by
  intro hF hd p c hp hc K hK
  have hc0 : ∀ v, 0 ≤ c v := fun v => (hc v).1
  refine UniversalLaw.Proofs.P15Budget.Aux.le_cappedHazard_of
    (UniversalLaw.Proofs.P15Budget.Aux.coverCost_le_one F K hc0) fun hmu => ?_
  calc coverCost F K c
      ≤ ∑ i, genPrice c (block F i) :=
        UniversalLaw.Proofs.P15Budget.Aux.coverCost_le_sum_block hF hc0 hK
    _ ≤ ∑ i, rhoStar * (-Real.log (localGood F p i)) := by
        refine Finset.sum_le_sum fun i _ => ?_
        have hpos : 0 < localGood F p i :=
          UniversalLaw.Proofs.P15Hazard.Aux.localGood_pos_of_mu_pos F hp hmu i
        exact (h12 F hF hd p c hp hc i hpos).1
    _ = rhoStar * ∑ i, -Real.log (localGood F p i) := by
        rw [Finset.mul_sum]
    _ ≤ rhoStar * -Real.log (mu F p) :=
        mul_le_mul_of_nonneg_left
          ((UniversalLaw.Proofs.P15Hazard.globalAssemblyHazard F hF p hp).2 hmu)
          f2_rhoStar_nonneg

end Aux

/-! ## Inventory theorems

Each theorem below has as its type exactly one registered `Prop`; the chain is
`coordinatewiseConcavity → hazardProductBound → hazardPhiBound → fullGroundPriceBound,
fullGroundBudgetIff, localPriceBoundFull → fullTransformedPriceBudget`, and `rhoStarEnclosure`
is independent of it. -/

/-- Proves: `UniversalLaw.Spec.P15.RhoStarEnclosure` (claim `p15-full-price`, display (F3)).
Method: `rho_star < U` iff `3e - 2 < exp(3 - 1/U)` and `L < rho_star` iff `exp(3 - 1/L) < 3e - 2`
(`Real.log_lt_iff_lt_exp`, `Real.lt_log_iff_exp_lt`, `div_lt_iff₀`, `lt_div_iff₀`); with
`3 - 1/U = 2 + x_U`, `exp(3 - 1/U) = (exp 1)^2 * exp x_U` (`Real.exp_add`, `exp_two_eq`), and
`Real.exp_bound` at degree 26 for `e` and degree 16 for `exp x_U`, `exp x_L` gives exact rational
enclosures (`Aux.f3_exp_one_gt/lt`, `Aux.f3_exp_xU_ge`, `Aux.f3_exp_xL_le`) that `norm_num`
compares in `Aux.f3_log_lt`, `Aux.f3_lt_log`; `rhoStarUpper < 6/7` is `norm_num` on rationals.
Does not establish: the informal display (F3) or its review and acceptance; the alignment of the
formal objects with the source's (PENDING_INDEPENDENT_REVIEW); the source's `full_price.py`
Fraction procedure (not specified and not reproduced here: the route above is a different
certificate of the same enclosure), nor any change to the `specified` status that main's
`docs/FORMAL_VERIFICATION_ROLLOUT_20260927.md` records for this target; the disposition of
`p15-full-price`, which is transcribed in the register, not derived here. -/
theorem rhoStarEnclosure : UniversalLaw.Spec.P15.RhoStarEnclosure := by
  unfold UniversalLaw.Spec.P15.RhoStarEnclosure
  exact ⟨Aux.f3_lower_lt_rhoStar, Aux.f3_rhoStar_lt_upper, Aux.f3_upper_lt_six_sevenths⟩

/-- Proves: `UniversalLaw.Spec.P15.CoordinatewiseConcavity` (claim `p15-full-price`, display (F5)).
Method: split `mu_p(A)` at the coordinate `i` (display (F4), `Aux.cc_muOf_update`): with
`p_i = 1 - exp(-s)` it equals `S_in + (S_out - S_in) exp(-s)`, where `0 ≤ S_in ≤ S_out` for a
downset (`Aux.cc_sum_in_le_sum_out`, the injection `U ↦ U.erase i`) and `S_out > 0` since `∅ ∈ A`
and every `p_j(t) < 1` on the cube (`Aux.cc_sum_out_pos`); then `s ↦ -log(a + b exp(-s))` with
`a, b ≥ 0`, `a + b > 0` is concave by `concaveOn_of_deriv2_nonpos`, its second derivative being
`-a b exp(-s) / (a + b exp(-s))^2 ≤ 0` (`Aux.cc_concaveOn_negLog`, via `HasDerivAt.log`,
`HasDerivAt.div`, `HasDerivAt.deriv`).
Does not establish: the informal display (F5) or its review and acceptance; the alignment of the
formal objects with the source's (PENDING_INDEPENDENT_REVIEW); joint concavity, or the derivative
formulas (F4)-(F5) themselves as registered statements (not specified; concavity is their
consequence); the disposition of `p15-full-price`, which is transcribed in the register, not
derived here. -/
theorem coordinatewiseConcavity : UniversalLaw.Spec.P15.CoordinatewiseConcavity := by
  intro n A _ hA i t ht
  obtain ⟨h0, -, hdown⟩ := hA
  exact Aux.cc_concaveOn_hazardCoord_update h0 hdown i t ht

/-- Proves: `UniversalLaw.Spec.P15.HazardProductBound` (claim `p15-full-price`, display (F6)).
Method: chord inequality of the separate coordinatewise concavity `coordinatewiseConcavity`
(display (F5)) on `[0,1]`: `f(s) ≥ (1-s) f(0) + s f(1) ≥ s f(1)` since `f(0) = -log mu ≥ 0`
(`Aux.f6_hazardCoord_nonneg`, from `mu_p(A) ≤ 1` by `sum_weight_eq_one` and `weight_nonneg`);
iterate over the coordinates with `Finset.induction_on` (`Aux.f6_prod_hazardCoord_le`), raising
one coordinate to `1` at each step (`Aux.f6_chord`), and identify `F_A(1, …, 1) = H_A`
(`Aux.f6_hazardCoord_one`, since `1 - exp(-1) = p_star`).
Does not establish: the informal display (F6) or its review and acceptance; the alignment of the
formal objects with the source's (PENDING_INDEPENDENT_REVIEW); the inequality as an informal
statement (the Prop is statement only); the disposition of `p15-full-price`, which is
transcribed in the register, not derived here. -/
theorem hazardProductBound : UniversalLaw.Spec.P15.HazardProductBound :=
  Aux.f6_hazardProductBound_of_concavity coordinatewiseConcavity

/-- Proves: `UniversalLaw.Spec.P15.HazardPhiBound` (claim `p15-full-price`, display (F7)).
Method: put `t_i := phi(p_i) ∈ [0,1]` (`Aux.f7_phi_mem`) and `p'_i := 1 - exp(-t_i) ∈ [0, p_i]`
(`Aux.f7_one_sub_exp_neg_phi_le`, from `phi p ≤ -log(1-p)` and `Real.exp_log`); for a downset
`mu(A)` is antitone in the probability vector (`Aux.f7_muOf_antitone`, built from the
one-coordinate split `Aux.cc_muOf_update` of display (F4) with `S_in ≤ S_out` by
`Finset.sum_image` along `U ↦ U.erase i`, iterated by `Finset.induction_on`), so
`0 < mu_p(A) ≤ mu_p'(A)` and `Real.log_le_log` give `-log mu_p(A) ≥ -log mu_p'(A) = F_A(t)`;
display (F6) `hazardProductBound` then gives `F_A(t) ≥ (prod_i t_i) H_A`, and `mul_comm`
matches the Prop's order.
Does not establish: the informal display (F7) or its review and acceptance; the alignment of the
formal objects with the source's (PENDING_INDEPENDENT_REVIEW); the inequality as an informal
statement (the Prop is statement only), nor anything at `mu_p(A) = 0`, where the source reads
the inequality as an extended one and the Prop states nothing; the disposition of
`p15-full-price`, which is transcribed in the register, not derived here. -/
theorem hazardPhiBound : UniversalLaw.Spec.P15.HazardPhiBound :=
  Aux.f7_hazardPhiBound_of_productBound hazardProductBound

/-- Proves: `UniversalLaw.Spec.P15.FullGroundPriceBound` (claim `p15-full-price`, display (F8)).
Method: `prod_i c_i ≤ prod_i phi(p_i)` for `0 ≤ c_i ≤ phi(p_i)` (`Finset.prod_le_prod₀`) and
`prod_i phi(p_i) ≤ 1` from `0 ≤ phi ≤ 1` on `[0,1]` (`Finset.prod_le_one₀`, `Real.log_nonpos`)
give the cap; when `mu_p(A) > 0`, display (F7), `hazardPhiBound`, gives
`H_A prod_i phi(p_i) ≤ -log mu_p(A)` and `H_A > 0` (`P15Elementary.hazardStarPositive`), so
`prod_i phi(p_i) ≤ (1/H_A) [-log mu_p(A)]` (`inv_mul_cancel_left₀`, `mul_le_mul_of_nonneg_left`);
`P15Budget.Aux.le_cappedHazard_of` reaches the capped hazard.
Does not establish: the informal display (F8) or its review and acceptance; the alignment of the
formal objects with the source's (PENDING_INDEPENDENT_REVIEW); that the full-ground generator
covers any obstruction (the source: "It does not assert that this generator alone covers an
arbitrary obstruction"); the disposition of `p15-full-price`, which is transcribed in the
register, not derived here. -/
theorem fullGroundPriceBound : UniversalLaw.Spec.P15.FullGroundPriceBound :=
  Aux.f8_fullGroundPriceBound_of_phiBound hazardPhiBound

/-- Proves: `UniversalLaw.Spec.P15.FullGroundBudgetIff` (claim `p15-full-price`, display (F8), the
exactness remark following it).
Method: (⇐) is display (F8) with `rho = 1`: `prod_i c_i ≤ prod_i phi(p_i) ≤ 1`
(`Finset.prod_le_prod₀`, `Finset.prod_le_one₀`) and, when `mu_p(A) > 0`,
`prod_i phi(p_i) ≤ H_A prod_i phi(p_i) ≤ -log mu_p(A)` from `1 ≤ H_A` (`le_mul_of_one_le_left`)
and display (F7), `hazardPhiBound`, reaching the cap by `P15Budget.Aux.le_cappedHazard_of`.
(⇒) instantiates the budget at `p ≡ p_star`, `c ≡ 1` (admissible by `P15Elementary.Aux.pStar_pos`,
`pStar_lt_one` and `phi p_star = 1`, `P15Elementary.phiAtPStar`); `mu_{p_star}(A) > 0` since
`∅ ∈ A` contributes `(1 - p_star)^n > 0` (`Finset.single_le_sum`, `P15Budget.Aux.weight_nonneg`),
so `1 ≤ min 1 (-log mu_{p_star}(A))` and `le_min_iff` gives `1 ≤ H_A`.
Does not establish: the informal display (F8) or its review and acceptance; the alignment of the
formal objects with the source's (PENDING_INDEPENDENT_REVIEW); coverage of arbitrary `A` (the
source: "it does not infer coverage of arbitrary A"); the disposition of `p15-full-price`, which
is transcribed in the register, not derived here. -/
theorem fullGroundBudgetIff : UniversalLaw.Spec.P15.FullGroundBudgetIff :=
  Aux.bi_fullGroundBudgetIff_of_phiBound hazardPhiBound

/-- Proves: `UniversalLaw.Spec.P15.LocalPriceBoundFull` (claim `p15-full-price`, display (F12)).
Method: enumerate the block by `Fin n_i` (`Aux.lp_exists_enum`), transport `mu_p(D_i)` to the
capacity family `{|T| ≤ a_i}` via `LocalRestriction` (`Aux.lp_localGood_eq_muOf`), apply
`hazardPhiBound` (display (F7)) to that proper downset (`Aux.lp_capacityFamily_properDownset`),
bound `h_star ≤ H_{n_i,a_i}` by `CapacityWorstCase` and `∏ c ≤ ∏ phi ≤ 1` by
`Finset.prod_le_prod₀`, `Finset.prod_le_one₀`; `rho_star = 1 / h_star` is definitional.
Does not establish: the informal display (F12) or its review and acceptance; the alignment of the
formal objects with the source's (PENDING_INDEPENDENT_REVIEW); the stronger inequality
`price ≤ q_i` ("The inequality price<=q_i is NOT required and is in general stronger than what
is proved here"); the disposition of `p15-full-price`, which is transcribed in the register, not
derived here. -/
theorem localPriceBoundFull {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.LocalPriceBoundFull F :=
  Aux.lp_localPriceBoundFull_of_phiBound hazardPhiBound F

/-- Proves: `UniversalLaw.Spec.P15.FullTransformedPriceBudget` (claim `p15-full-price`, display (F2)).
Method: `covercost ≤ 1` by the cover `{∅}` (`P15Budget.Aux.coverCost_le_one`); for `K ≥ K_H(d)`
the full blocks cover, so `covercost ≤ sum_i price(X_i)` (`P15Budget.Aux.coverCost_le_sum_block`);
when `mu_p(D) > 0` every `mu_p(D_i) > 0` (`P15Hazard.Aux.localGood_pos_of_mu_pos`) and the local
bound (F12), `localPriceBoundFull`, gives `price(X_i) ≤ rho_star [-log mu_p(D_i)]`;
`Finset.sum_le_sum`, `Finset.mul_sum` and the global assembly (F13),
`P15Hazard.globalAssemblyHazard`, give `sum_i price(X_i) ≤ rho_star [-log mu_p(D)]` with
`rho_star ≥ 0`; `P15Budget.Aux.le_cappedHazard_of` reaches the capped hazard.
Does not establish: the informal display (F2) or its review and acceptance; the alignment of the
formal objects with the source's (PENDING_INDEPENDENT_REVIEW); optimality of `rho_star` (that is
`SharpnessOfRhoStar`); any unrestricted downset or prize statement (the source: "palette size
remains K_H(d), which can depend on H and d"); the nonauthor review recorded in LANDING_CLAIMS
(REVIEWED_SCOPED), which is not reproduced or extended here; the disposition of
`p15-full-price`, which is transcribed in the register, not derived here. -/
theorem fullTransformedPriceBudget {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.FullTransformedPriceBudget F :=
  Aux.f2_fullTransformedPriceBudget_of_local (fun F => localPriceBoundFull F) F

end UniversalLaw.Proofs.P15Interpolation
