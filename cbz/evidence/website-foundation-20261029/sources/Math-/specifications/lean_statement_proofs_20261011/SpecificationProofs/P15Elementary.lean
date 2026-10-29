import Specifications.P15

/-!
# P15: kernel-checked proofs of elementary registered `Prop`s (statement module `Specifications.P15`)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem below has as its TYPE a registered `Prop` constant of
`UniversalLaw.Spec.P15` (statement packet `specifications/lean_statements_20261010`, commit
pinned there). A kernel-checked proof of such a `Prop` is evidence about that `Prop` only: it
does not make the informal Layer 0 claim accepted, reviewed or "verified", does not change any
LANDING_CLAIMS disposition, and does not touch the statement register (the statement module is
not edited here). Helper lemmas live in the `Aux` namespace and are not inventory items.

Allowed transitive axioms: `propext`, `Classical.choice`, `Quot.sound` only (checked by
`#print axioms`, see the run receipt). No `sorry`, `native_decide`, `decide` on `ℝ`, or custom
axioms. The three `decide` calls in this module act on finite `Finset`/`Fin` data (ground sets of
size at most three) and are kernel-reduced.

Covered (fifteen `Prop`s): `TwoCoordGeneratorPrices`, `LogOnePlusBelowIdentity`, `LogRatioBound`,
`CapTestAtTwoThirds`, `RatioExponentBound`, `RhoStarBelowSixSevenths`, `SixSeventhsCertificate`,
`HStarExceedsOne`, `PhiAtPStar`, `OddMajorityIdentity`, `HazardStarPositive`,
`DemandOneBoundary`, `DemandOneRefutation`, `CapacityWorstCase`, `TriangleNeedsThree`.

Also covered (two further `Prop`s, proved in the final section of this module, so seventeen in
all): `CapacityHazardIsBinomial`, `SharpnessOfRhoStar`.

Not covered here: `IsProperDownset` (a parametrised predicate, not a concrete-instance `Prop`;
it is false for the empty family, so no theorem of that type can exist), `ElementaryPaletteFormula`
(display (P9) quantified over all `b`, `s`, `d`: genuine combinatorics, not a finite statement),
and everything listed as hard analysis or large-benchmark in the proof-packet contract.
-/

namespace UniversalLaw.Proofs.P15Elementary

open UniversalLaw.Spec.P15

namespace Aux

/-! ### Real-analytic constants -/

/-- `2 < e` from Mathlib's decimal bound. -/
theorem two_lt_exp_one : (2 : ℝ) < Real.exp 1 := by
  have := Real.exp_one_gt_d9; linarith

/-- `3e - 2 > 0`. -/
theorem three_exp_sub_two_pos : (0 : ℝ) < 3 * Real.exp 1 - 2 := by
  have := two_lt_exp_one; linarith

/-- `exp 2 = (exp 1)^2`. -/
theorem exp_two_eq : Real.exp 2 = Real.exp 1 ^ 2 := by
  rw [← Real.exp_nat_mul]; norm_num

/-- `exp (-2) = (exp (-1))^2`. -/
theorem exp_neg_two_eq : Real.exp (-2) = Real.exp (-1) ^ 2 := by
  rw [← Real.exp_nat_mul]; norm_num

/-- `exp (-3) = (exp (-1))^3`. -/
theorem exp_neg_three_eq : Real.exp (-3) = Real.exp (-1) ^ 3 := by
  rw [← Real.exp_nat_mul]; norm_num

/-- `log(3e - 2) < 2`, from `3e - 2 < e^2`, i.e. `(e-1)(e-2) > 0`. -/
theorem log_three_exp_sub_two_lt_two : Real.log (3 * Real.exp 1 - 2) < 2 := by
  rw [Real.log_lt_iff_lt_exp three_exp_sub_two_pos, exp_two_eq]
  have := two_lt_exp_one
  nlinarith

/-- `1 < h_star`. -/
theorem one_lt_hStar : 1 < hStar := by
  unfold hStar; have := log_three_exp_sub_two_lt_two; linarith

/-- The Taylor sum of the Section 6 certificate, evaluated exactly. -/
theorem taylor_sum_eq :
    (∑ j ∈ Finset.range 6, (11 / 6 : ℝ) ^ j / (Nat.factorial j : ℝ)) =
      197 / 32 + 26081 / 933120 := by
  simp [Finset.sum_range_succ, Nat.factorial]
  norm_num

/-- `197/32 < exp(11/6)` by the degree-5 Taylor lower bound. -/
theorem exp_eleven_sixths_gt : (197 / 32 : ℝ) < Real.exp (11 / 6) := by
  have h := Real.sum_le_exp_of_nonneg (by norm_num : (0 : ℝ) ≤ 11 / 6) 6
  rw [taylor_sum_eq] at h
  linarith

/-- `3e - 2 < 197/32` from `e < 2.7182818286`. -/
theorem three_exp_sub_two_lt : 3 * Real.exp 1 - 2 < (197 / 32 : ℝ) := by
  have := Real.exp_one_lt_d9; linarith

/-- `log(3e - 2) < 11/6`. -/
theorem log_three_exp_sub_two_lt : Real.log (3 * Real.exp 1 - 2) < 11 / 6 := by
  rw [Real.log_lt_iff_lt_exp three_exp_sub_two_pos]
  exact lt_trans three_exp_sub_two_lt exp_eleven_sixths_gt

/-- `7/6 < h_star`. -/
theorem seven_sixths_lt_hStar : (7 / 6 : ℝ) < hStar := by
  unfold hStar; have := log_three_exp_sub_two_lt; linarith

/-- `rho_star < 6/7`. -/
theorem rhoStar_lt_six_sevenths : rhoStar < 6 / 7 := by
  unfold rhoStar
  have h : (7 / 6 : ℝ) < 3 - Real.log (3 * Real.exp 1 - 2) := seven_sixths_lt_hStar
  rw [div_lt_iff₀ (by linarith)]
  linarith

/-- `0 < p_star`. -/
theorem pStar_pos : 0 < pStar := by
  unfold pStar; have := Real.exp_neg_one_lt_half; linarith

/-- `1/2 < p_star`. -/
theorem pStar_gt_half : 1 / 2 < pStar := by
  unfold pStar; have := Real.exp_neg_one_lt_half; linarith

/-- `p_star < 1`. -/
theorem pStar_lt_one : pStar < 1 := by
  unfold pStar; have := Real.exp_pos (-1); linarith

/-! ### Product weights -/

/-- The weights of all subsets sum to one (`Finset.prod_add` with `f = p`, `g = 1 - p`). -/
theorem sum_weight_eq_one {Y : Type} [Fintype Y] [DecidableEq Y] (p : Y → ℝ) :
    ∑ U ∈ (Finset.univ : Finset Y).powerset, weight p U = 1 := by
  have h := Finset.prod_add p (fun v => 1 - p v) (Finset.univ : Finset Y)
  have hl : ∏ v : Y, (p v + (1 - p v)) = 1 := by simp
  rw [hl] at h
  unfold weight
  exact h.symm

/-! ### Binomial lower tails -/

/-- One more trial: `P(Bin(n+1,p) ≤ k) = P(Bin(n,p) ≤ k) - C(n,k) p^(k+1) (1-p)^(n-k)` for
`k ≤ n` (Pascal's rule, induction on `k`). -/
theorem binomTail_succ_left (n k : ℕ) (p : ℝ) (hk : k ≤ n) :
    binomTail (n + 1) k p =
      binomTail n k p - (Nat.choose n k : ℝ) * p ^ (k + 1) * (1 - p) ^ (n - k) := by
  induction k with
  | zero =>
    simp [binomTail]
    ring
  | succ k ih =>
    have hk' : k ≤ n := by omega
    have e1 : binomTail (n + 1) (k + 1) p = binomTail (n + 1) k p +
        (Nat.choose (n + 1) (k + 1) : ℝ) * p ^ (k + 1) * (1 - p) ^ (n + 1 - (k + 1)) := by
      unfold binomTail; rw [Finset.sum_range_succ]
    have e2 : binomTail n (k + 1) p = binomTail n k p +
        (Nat.choose n (k + 1) : ℝ) * p ^ (k + 1) * (1 - p) ^ (n - (k + 1)) := by
      unfold binomTail; rw [Finset.sum_range_succ]
    rw [e1, e2, ih hk', Nat.choose_succ_succ']
    have h1 : n + 1 - (k + 1) = n - k := by omega
    have h2 : n - k = (n - (k + 1)) + 1 := by omega
    rw [h1, h2, pow_succ]
    push_cast
    ring

/-- (F9) as an identity: the product measure of the capacity family at a constant vector is the
binomial lower tail (stated for `a ≤ n`). -/
theorem muOf_capacityFamily (n a : ℕ) (han : a ≤ n) (c : ℝ) :
    muOf (capacityFamily n a) (fun _ => c) = binomTail n a c := by
  unfold muOf capacityFamily binomTail
  have hw : ∀ U : Finset (Fin n),
      weight (fun _ : Fin n => c) U = c ^ U.card * (1 - c) ^ (n - U.card) := by
    intro U; unfold weight; simp [Finset.card_univ_sdiff]
  rw [Finset.sum_filter]
  simp_rw [hw]
  rw [Finset.sum_powerset_apply_card (fun m => if m ≤ a then c ^ m * (1 - c) ^ (n - m) else 0)]
  simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_ite, mul_zero]
  rw [← Finset.sum_filter]
  have hr : (Finset.range (n + 1)).filter (fun m => m ≤ a) = Finset.range (a + 1) := by
    ext m; simp; omega
  rw [hr]
  exact Finset.sum_congr rfl (fun j _ => by ring)

/-- The lower tail decreases when one trial is added (`a ≤ n`, `0 ≤ c ≤ 1`). -/
theorem binomTail_antitone_step (n a : ℕ) (han : a ≤ n) (c : ℝ) (h0 : 0 ≤ c) (h1 : c ≤ 1) :
    binomTail (n + 1) a c ≤ binomTail n a c := by
  rw [binomTail_succ_left n a c han]
  have : 0 ≤ (Nat.choose n a : ℝ) * c ^ (a + 1) * (1 - c) ^ (n - a) :=
    mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg h0 _)) (pow_nonneg (by linarith) _)
  linarith

/-- Monotonicity in the number of trials: `binomTail n a c ≤ binomTail m a c` for `a ≤ m ≤ n`. -/
theorem binomTail_le_of_le (a m n : ℕ) (hm : a ≤ m) (hmn : m ≤ n) (c : ℝ) (h0 : 0 ≤ c)
    (h1 : c ≤ 1) : binomTail n a c ≤ binomTail m a c := by
  induction n, hmn using Nat.le_induction with
  | base => exact le_refl _
  | succ k hk ih => exact le_trans (binomTail_antitone_step k a (le_trans hm hk) c h0 h1) ih

/-- The lower tail is positive for `0 ≤ c < 1` (the `j = 0` term is `(1-c)^n > 0`). -/
theorem binomTail_pos (n a : ℕ) (c : ℝ) (h0 : 0 ≤ c) (h1 : c < 1) : 0 < binomTail n a c := by
  unfold binomTail
  rw [Finset.sum_range_succ']
  have hpos : 0 < (Nat.choose n 0 : ℝ) * c ^ 0 * (1 - c) ^ (n - 0) := by
    simp
    exact pow_pos (by linarith) _
  have hnn : 0 ≤ ∑ j ∈ Finset.range a,
      (Nat.choose n (j + 1) : ℝ) * c ^ (j + 1) * (1 - c) ^ (n - (j + 1)) :=
    Finset.sum_nonneg (fun j _ =>
      mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg h0 _)) (pow_nonneg (by linarith) _))
  linarith

/-- `P(Bin(3, p_star) ≤ 1) = 3e^{-2} - 2e^{-3}`. -/
theorem binomTail_three_one : binomTail 3 1 pStar = 3 * Real.exp (-2) - 2 * Real.exp (-3) := by
  unfold binomTail pStar
  simp [Finset.sum_range_succ]
  rw [exp_neg_two_eq, exp_neg_three_eq]
  ring

/-! ### One block, no edges, capacity `cap`, demand one, ground set of size `cap + 1` -/

section OneBlock

variable (cap : ℕ)

/-- The single block is the whole ground set. -/
theorem oneBlock_block (i : Fin 1) : block (demandOneBlock cap) i = Finset.univ := by
  unfold block
  rw [Fin.fin_one_eq_zero i]
  simp [demandOneBlock]

/-- A proper subset of a `(cap+1)`-set has at most `cap` elements. -/
theorem card_le_of_ne_univ (U : Finset (Fin (cap + 1))) (h : U ≠ Finset.univ) :
    U.card ≤ cap := by
  have h1 : U.card ≤ Fintype.card (Fin (cap + 1)) := Finset.card_le_univ U
  have h2 : U.card ≠ Fintype.card (Fin (cap + 1)) :=
    fun e => h ((Finset.card_eq_iff_eq_univ U).mp e)
  rw [Fintype.card_fin] at h1 h2
  omega

/-- Goodness is the capacity constraint. -/
theorem oneBlock_isGood (U : Finset (Fin (cap + 1))) :
    IsGood (demandOneBlock cap) U ↔ U.card ≤ cap := by
  show (∀ i : Fin 1, (U ∩ block (demandOneBlock cap) i).card ≤ cap) ∧
      (∀ e ∈ (∅ : Finset (Finset (Fin 1))), ¬ e ⊆ supp (demandOneBlock cap) U) ↔ U.card ≤ cap
  simp [oneBlock_block]

/-- One-piece decomposability is goodness (for any family). -/
theorem decomp_one {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (U : Finset X) :
    IsKDecomposable F 1 U ↔ IsGood F U := by
  unfold IsKDecomposable
  constructor
  · rintro ⟨parts, hgood, -, hU⟩
    have : U = parts 0 := by rw [← hU]; simp
    rw [this]; exact hgood 0
  · intro h
    refine ⟨fun _ => U, fun _ => h, fun j j' hjj' => absurd (Subsingleton.elim j j') hjj', ?_⟩
    simp

/-- At `K = 1` the sole obstruction is the full block. -/
theorem oneBlock_obstruction (U : Finset (Fin (cap + 1))) :
    U ∈ obstruction (demandOneBlock cap) 1 ↔ U = Finset.univ := by
  unfold obstruction
  change ¬ IsKDecomposable (demandOneBlock cap) 1 U ↔ U = Finset.univ
  rw [decomp_one, oneBlock_isGood, not_le]
  constructor
  · intro h
    by_contra hne
    have := card_le_of_ne_univ cap U hne
    omega
  · rintro rfl
    rw [Finset.card_univ, Fintype.card_fin]; omega

/-- The side conditions hold for `cap ≥ 1`. -/
theorem oneBlock_isRealized (hcap : 1 ≤ cap) : IsRealizedFamily (demandOneBlock cap) := by
  refine ⟨fun _ => hcap, fun _ => le_refl 1, ?_, ?_, ?_⟩
  · change ∀ i : Fin 1, (block (demandOneBlock cap) i).card = cap * 1 + 1
    intro i
    rw [oneBlock_block]
    simp
  · change ∀ e ∈ (∅ : Finset (Finset (Fin 1))), 2 ≤ e.card
    simp
  · change ∀ e ∈ (∅ : Finset (Finset (Fin 1))), ∀ f ∈ (∅ : Finset (Finset (Fin 1))),
      e ⊆ f → e = f
    simp

/-- `mu_p(D) = 1 - prod_v p_v`: every set but the full block is good. -/
theorem oneBlock_mu (p : Fin (cap + 1) → ℝ) :
    mu (demandOneBlock cap) p = 1 - ∏ v, p v := by
  unfold mu
  have hmem : (Finset.univ : Finset (Fin (cap + 1))) ∈
      (Finset.univ : Finset (Fin (cap + 1))).powerset :=
    Finset.mem_powerset.mpr (subset_refl _)
  rw [← Finset.add_sum_erase _ _ hmem]
  have h0 : Set.indicator (downset (demandOneBlock cap)) (weight p) Finset.univ = 0 := by
    apply Set.indicator_of_notMem
    show ¬ IsGood (demandOneBlock cap) Finset.univ
    rw [oneBlock_isGood, Finset.card_univ, Fintype.card_fin]
    omega
  rw [h0, zero_add]
  have h1 : ∀ U ∈ (Finset.univ : Finset (Fin (cap + 1))).powerset.erase Finset.univ,
      Set.indicator (downset (demandOneBlock cap)) (weight p) U = weight p U := by
    intro U hU
    rw [Finset.mem_erase] at hU
    apply Set.indicator_of_mem
    show IsGood (demandOneBlock cap) U
    rw [oneBlock_isGood]
    exact card_le_of_ne_univ cap U hU.1
  rw [Finset.sum_congr rfl h1]
  have h2 := Finset.add_sum_erase (Finset.univ : Finset (Fin (cap + 1))).powerset (weight p) hmem
  rw [sum_weight_eq_one] at h2
  have h3 : weight p Finset.univ = ∏ v, p v := by unfold weight; simp
  linarith

/-- At unit prices every cover of `{X}` costs at least one. -/
theorem oneBlock_coverCost_ge_one :
    1 ≤ coverCost (demandOneBlock cap) 1 (fun _ => (1 : ℝ)) := by
  unfold coverCost
  apply le_csInf
  · refine ⟨1, {Finset.univ}, ?_, ?_⟩
    · intro U hU
      rw [oneBlock_obstruction] at hU
      exact ⟨Finset.univ, Finset.mem_singleton_self _, by rw [hU]⟩
    · simp [famPrice, genPrice]
  · rintro t ⟨G, hG, rfl⟩
    have huniv : Finset.univ ∈ obstruction (demandOneBlock cap) 1 :=
      (oneBlock_obstruction cap _).mpr rfl
    obtain ⟨g, hg, -⟩ := hG Finset.univ huniv
    unfold famPrice
    calc (1 : ℝ) = genPrice (fun _ => (1 : ℝ)) g := by simp [genPrice]
      _ ≤ ∑ g ∈ G, genPrice (fun _ => (1 : ℝ)) g :=
          Finset.single_le_sum (fun g _ => by simp [genPrice]) hg

end OneBlock

/-- The two-coordinate record of the price boundary is the `cap = 1` demand-one block. -/
theorem twoCoord_eq : twoCoord = demandOneBlock 1 := rfl

/-! ### The triangle palette instance -/

/-- Three singletons are a feasible palette of size three for the triangle. -/
theorem triangle_feasible3 : IsPaletteFeasible 3 (completeClutter 3 1) (fun _ => 1) 3 := by
  refine ⟨fun i => {i}, fun i => by simp, ?_⟩
  decide

/-- No palette of size zero. -/
theorem triangle_not_feasible0 : ¬ IsPaletteFeasible 3 (completeClutter 3 1) (fun _ => 1) 0 := by
  rintro ⟨P, hP, -⟩
  have h1 : 1 ≤ (P 0).card := hP 0
  have h0 : (P 0).card ≤ Fintype.card (Fin 0) := Finset.card_le_univ _
  rw [Fintype.card_fin] at h0
  omega

/-- No palette of size one (finite check). -/
theorem triangle_not_feasible1 : ¬ IsPaletteFeasible 3 (completeClutter 3 1) (fun _ => 1) 1 := by
  unfold IsPaletteFeasible
  decide

/-- No palette of size two (finite check over the `4^3` label-set assignments). -/
theorem triangle_not_feasible2 : ¬ IsPaletteFeasible 3 (completeClutter 3 1) (fun _ => 1) 2 := by
  unfold IsPaletteFeasible
  decide

end Aux

/-! ## Price boundary (demand one) -/

/-- Proves: `UniversalLaw.Spec.P15.TwoCoordGeneratorPrices` (register rows `p15-price-boundary`,
`math.p15-price-boundary`).
Method: unfold `genPrice`/`twoCoordC`, evaluate the four finite products with `simp`/`norm_num`.
Does not establish: the demand-one refutation itself (`DemandOneRefutation`), any review or
acceptance of P15-PRICE-BOUNDARY-20260924-v1; the disposition EXACT_COUNTEREXAMPLE is
transcribed in the register, not derived here. -/
theorem twoCoordGeneratorPrices : UniversalLaw.Spec.P15.TwoCoordGeneratorPrices := by
  unfold UniversalLaw.Spec.P15.TwoCoordGeneratorPrices
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [genPrice]
  · simp [genPrice, twoCoordC]
  · simp [genPrice, twoCoordC]
  · simp [genPrice, twoCoordC, Finset.prod_const]
    norm_num

/-- Proves: `UniversalLaw.Spec.P15.LogOnePlusBelowIdentity` (register row `p15-price-boundary`).
Method: `Real.log_lt_sub_one_of_pos` at `1 + x`.
Does not establish: anything about the refutation beyond this elementary inequality. -/
theorem logOnePlusBelowIdentity : UniversalLaw.Spec.P15.LogOnePlusBelowIdentity := by
  intro x hx
  have h := Real.log_lt_sub_one_of_pos (by linarith : (0 : ℝ) < 1 + x)
    (by intro h; linarith : (1 : ℝ) + x ≠ 1)
  linarith

/-- Proves: `UniversalLaw.Spec.P15.DemandOneRefutation` (register rows `p15-price-boundary`,
`math.p15-price-boundary`).
Method: `twoCoord` is definitionally the `cap = 1` demand-one block, so the one-block lemmas
give the side conditions, the obstruction `{X}` and `mu_p(D) = 1 - (1/2)^2 = 3/4`; `K_H(d) = 1`
by `Nat.sInf_le`/`le_csInf` (no palette of size `0`); admissibility and `phi(1/2) = log 2` from
`Real.log_two_gt_d9`/`Real.log_two_lt_d9`; every cover contains a generator `g ⊆ X`, whose price
`(2/3)^|g| ≥ 4/9`, and `{X}` attains `4/9` (`IsLeast.csInf_eq`); `log(4/3) < 1/3` from
`Real.log_lt_sub_one_of_pos`.
Does not establish: that the refutation contradicts original P15-B ("It does not contradict the
original P15-B amalgamation"), anything at the `408` scale or about the unrestricted prize, the
source's integral representation of `log 2`, nonauthor review or acceptance; the disposition
EXACT_COUNTEREXAMPLE is transcribed, not derived. -/
theorem demandOneRefutation : UniversalLaw.Spec.P15.DemandOneRefutation := by
  unfold UniversalLaw.Spec.P15.DemandOneRefutation
  have hlog2 := Real.log_two_gt_d9
  have hlog2' := Real.log_two_lt_d9
  have hphi : phi (1 / 2) = Real.log 2 := by
    unfold phi
    have h : (1 : ℝ) / 2 < 1 := by norm_num
    simp only [h, ↓reduceIte]
    rw [show (1 : ℝ) - 1 / 2 = (2 : ℝ)⁻¹ by norm_num, Real.log_inv, neg_neg]
    exact min_eq_right (by linarith)
  have hobs : ∀ U : Finset (Fin 2), U ∈ obstruction twoCoord 1 ↔ U = Finset.univ :=
    Aux.oneBlock_obstruction 1
  have hmu : mu twoCoord twoCoordP = 3 / 4 := by
    have h := Aux.oneBlock_mu 1 twoCoordP
    rw [show mu twoCoord twoCoordP = mu (demandOneBlock 1) twoCoordP from rfl, h]
    simp [twoCoordP]
    norm_num
  have hlog43 : Real.log (4 / 3) < 1 / 3 := by
    have h := Real.log_lt_sub_one_of_pos (by norm_num : (0 : ℝ) < 4 / 3) (by norm_num)
    linarith
  have hch : cappedHazard 1 (mu twoCoord twoCoordP) = Real.log (4 / 3) := by
    rw [hmu]
    unfold cappedHazard
    have h : (0 : ℝ) < 3 / 4 := by norm_num
    simp only [h, ↓reduceIte, one_mul]
    rw [show (3 / 4 : ℝ) = (4 / 3)⁻¹ by norm_num, Real.log_inv, neg_neg]
    exact min_eq_right (by linarith)
  have hcover : ∀ G : Finset (Finset (Fin 2)), IsGeneratorCover twoCoord 1 G →
      4 / 9 ≤ famPrice twoCoordC G := by
    intro G hG
    have huniv : (Finset.univ : Finset (Fin 2)) ∈ obstruction twoCoord 1 := (hobs _).mpr rfl
    obtain ⟨g, hg, -⟩ := hG _ huniv
    unfold famPrice
    have hgp : genPrice twoCoordC g = (2 / 3) ^ g.card := by
      simp [genPrice, twoCoordC, div_pow]
    have hcard : g.card ≤ 2 := by
      have := Finset.card_le_univ g; simpa using this
    calc (4 / 9 : ℝ) = (2 / 3) ^ 2 := by norm_num
      _ ≤ (2 / 3) ^ g.card := pow_le_pow_of_le_one (by norm_num) (by norm_num) hcard
      _ = genPrice twoCoordC g := hgp.symm
      _ ≤ ∑ g ∈ G, genPrice twoCoordC g :=
          Finset.single_le_sum (fun g _ => by
            unfold genPrice
            exact Finset.prod_nonneg (fun v _ => by unfold twoCoordC; norm_num)) hg
  have hcc : coverCost twoCoord 1 twoCoordC = 4 / 9 := by
    unfold coverCost
    apply IsLeast.csInf_eq
    refine ⟨⟨{Finset.univ}, ?_, ?_⟩, ?_⟩
    · intro U hU
      rw [hobs] at hU
      exact ⟨Finset.univ, Finset.mem_singleton_self _, by rw [hU]⟩
    · simp [famPrice, genPrice, twoCoordC]
      norm_num
    · rintro t ⟨G, hG, rfl⟩
      exact hcover G hG
  have hpal : paletteNumber 1 ∅ (fun _ => 1) = 1 := by
    unfold paletteNumber
    have h1 : (1 : ℕ) ∈ {K : ℕ | IsPaletteFeasible 1 ∅ (fun _ => 1) K} := by
      refine ⟨fun _ => {0}, fun _ => by simp, fun e he => by simp at he⟩
    apply le_antisymm (Nat.sInf_le h1)
    apply le_csInf ⟨1, h1⟩
    intro K hK
    obtain ⟨P, hP, -⟩ := hK
    by_contra hlt
    have hK0 : K = 0 := by omega
    subst hK0
    have h1le : 1 ≤ (P 0).card := hP 0
    have h0 : (P 0).card ≤ Fintype.card (Fin 0) := Finset.card_le_univ _
    rw [Fintype.card_fin] at h0
    omega
  refine ⟨Aux.oneBlock_isRealized 1 le_rfl, hpal, hobs, ?_, hphi, by linarith, hmu, hch, hlog43,
    hcover, hcc, by norm_num, ?_⟩
  · intro v
    show (2 / 3 : ℝ) ≤ phi (1 / 2)
    rw [hphi]; linarith
  · rw [hch, hcc]; linarith

/-! ## Restricted price budget: elementary displays -/

/-- Proves: `UniversalLaw.Spec.P15.LogRatioBound` (register row `p15-price-budget-restricted`).
Method: `Real.log_le_sub_one_of_pos` at `(1 - p)⁻¹` and `Real.log_inv`.
Does not establish: display (T1) or the restricted budget; the source's integral representation
of `-log(1-p)` is not formalised. -/
theorem logRatioBound : UniversalLaw.Spec.P15.LogRatioBound := by
  intro p hp0 hp1
  have h1p : 0 < 1 - p := by linarith
  have h := Real.log_le_sub_one_of_pos (inv_pos.mpr h1p)
  rw [Real.log_inv] at h
  have e : (1 - p)⁻¹ - 1 = p / (1 - p) := by
    field_simp
    ring
  linarith

/-- Proves: `UniversalLaw.Spec.P15.CapTestAtTwoThirds` (register row `p15-price-budget-restricted`).
Method: `log 3 > 1` from `exp 1 < 3` (Mathlib's `Real.exp_one_lt_d9`); for `2/3 ≤ q < 1`,
`-log(1-q) ≥ log 3 > 1` so the `min` is `1`; at `q = 1` the definition of `phi` gives `1`.
Does not establish: the checker program of the source's Section 5, nor any budget statement. -/
theorem capTestAtTwoThirds : UniversalLaw.Spec.P15.CapTestAtTwoThirds := by
  unfold UniversalLaw.Spec.P15.CapTestAtTwoThirds
  have hlog3 : 1 < Real.log 3 := by
    rw [Real.lt_log_iff_exp_lt (by norm_num)]
    have := Real.exp_one_lt_d9
    linarith
  refine ⟨hlog3, ?_⟩
  intro q hq1 hq2
  unfold phi
  rcases lt_or_eq_of_le hq2 with h | h
  · simp only [h, ↓reduceIte]
    have h1q : 0 < 1 - q := by linarith
    have hle : 1 - q ≤ 1 / 3 := by linarith
    have hl : Real.log (1 - q) ≤ Real.log (1 / 3) := Real.log_le_log h1q hle
    rw [one_div, Real.log_inv] at hl
    exact min_eq_left (by linarith)
  · subst h
    simp

/-- Proves: `UniversalLaw.Spec.P15.RatioExponentBound` (register row `p15-price-budget-restricted`).
Method: `3^n ≥ 3^(2a+1)` and the exact identity `4^(a+1)/3^(2a+1) = (4/3)(4/9)^a`; then
`(4/9)^a ≤ 4/9` for `a ≥ 1`.
Does not establish: the ratio steps of (T3) preceding the exponent arithmetic, or (T4)/(T1). -/
theorem ratioExponentBound : UniversalLaw.Spec.P15.RatioExponentBound := by
  intro a n ha hn
  constructor
  · have h3 : (3 : ℝ) ^ (2 * a + 1) ≤ 3 ^ n := pow_le_pow_right₀ (by norm_num) hn
    have h3pos : (0 : ℝ) < 3 ^ (2 * a + 1) := by positivity
    calc (4 : ℝ) ^ (a + 1) / 3 ^ n ≤ 4 ^ (a + 1) / 3 ^ (2 * a + 1) :=
          div_le_div_of_nonneg_left (by positivity) h3pos h3
      _ = (4 / 3) * (4 / 9) ^ a := by
          rw [pow_succ, pow_succ, pow_mul, div_pow]
          norm_num
          ring
  · have h : (4 / 9 : ℝ) ^ a ≤ (4 / 9) ^ 1 := pow_le_pow_of_le_one (by norm_num) (by norm_num) ha
    nlinarith [h]

/-! ## Full price: the constant `rho_star` -/

/-- Proves: `UniversalLaw.Spec.P15.RhoStarBelowSixSevenths` (register row `p15-full-price`).
Method: `3e - 2 < 197/32 < exp(11/6)` (decimal bound on `e`; degree-5 Taylor lower bound on
`exp`), hence `h_star > 7/6` and `rho_star = 1/h_star < 6/7`.
Does not establish: the 20-digit enclosure (F3) (`RhoStarEnclosure`), the budget (F2), or any
review/acceptance of P15-FULL-TRANSFORMED-PRICE-20260924-v1 (disposition REVIEWED_SCOPED is
transcribed, not derived). -/
theorem rhoStarBelowSixSevenths : UniversalLaw.Spec.P15.RhoStarBelowSixSevenths :=
  Aux.rhoStar_lt_six_sevenths

/-- Proves: `UniversalLaw.Spec.P15.SixSeventhsCertificate` (register row `p15-full-price`).
Method: `Real.exp_one_lt_d9` for `e < 31967/11760`; `norm_num` for the two rational identities;
`Real.sum_le_exp_of_nonneg` with six terms for `197/32 < exp(11/6)`; then `log` monotonicity.
Does not establish: the series and geometric-tail bounds the source uses for `e < 31967/11760`
(a sharper Mathlib bound is used instead), nor anything about (F2). -/
theorem sixSeventhsCertificate : UniversalLaw.Spec.P15.SixSeventhsCertificate := by
  unfold UniversalLaw.Spec.P15.SixSeventhsCertificate
  have he := Real.exp_one_lt_d9
  refine ⟨by linarith, by norm_num, by norm_num, ?_, by norm_num, Aux.three_exp_sub_two_lt,
    Aux.exp_eleven_sixths_gt, Aux.seven_sixths_lt_hStar, Aux.rhoStar_lt_six_sevenths⟩
  rw [Aux.taylor_sum_eq]
  norm_num

/-- Proves: `UniversalLaw.Spec.P15.HStarExceedsOne` (register row `p15-full-price`).
Method: `2 < e` from `Real.exp_one_gt_d9`; `ring` for the factorisation; `u = exp(-1) < 1/2`
and `u(1-u)(1-2u) > 0` for `3e^{-2} - 2e^{-3} < e^{-1}`; `log(3e-2) < 2` from `3e-2 < e^2`.
Does not establish: (F11) for general `(n, a)` (that is `CapacityWorstCase`). -/
theorem hStarExceedsOne : UniversalLaw.Spec.P15.HStarExceedsOne := by
  unfold UniversalLaw.Spec.P15.HStarExceedsOne
  have he := Aux.two_lt_exp_one
  refine ⟨he, by ring, mul_pos (by linarith) (by linarith), ?_, Aux.one_lt_hStar⟩
  rw [Aux.exp_neg_two_eq, Aux.exp_neg_three_eq]
  have hu : 0 < Real.exp (-1) := Real.exp_pos _
  have hu2 : Real.exp (-1) < 1 / 2 := Real.exp_neg_one_lt_half
  nlinarith [mul_pos (mul_pos hu (by linarith : (0 : ℝ) < 1 - Real.exp (-1)))
    (by linarith : (0 : ℝ) < 1 - 2 * Real.exp (-1))]

/-- Proves: `UniversalLaw.Spec.P15.PhiAtPStar` (register row `p15-full-price`).
Method: `p_star < 1`, `1 - p_star = exp(-1)`, `Real.log_exp`.
Does not establish: equality of the two sides of (F7) (not specified separately). -/
theorem phiAtPStar : UniversalLaw.Spec.P15.PhiAtPStar := by
  unfold UniversalLaw.Spec.P15.PhiAtPStar phi pStar
  have hpos := Real.exp_pos (-1)
  have hlt : (1 : ℝ) - Real.exp (-1) < 1 := by linarith
  simp only [hlt, ↓reduceIte]
  have e : (1 : ℝ) - (1 - Real.exp (-1)) = Real.exp (-1) := by ring
  rw [e, Real.log_exp]
  norm_num

/-! ## Full price: hazard positivity, binomial identities, worst case -/

/-- Proves: `UniversalLaw.Spec.P15.HazardStarPositive` (register row `p15-full-price`).
Method: at the constant vector `p_star ∈ (0,1)` all weights are nonnegative, the empty set
contributes `(1-p_star)^n > 0`, and since `X ∉ A` the mass is at most `1 - p_star^n < 1`
(all weights sum to one by `Finset.prod_add`); then `Real.log_neg`.
Does not establish: the bounds (F5)-(F8) that use `H_A > 0`. -/
theorem hazardStarPositive : UniversalLaw.Spec.P15.HazardStarPositive := by
  intro n A _ hA
  obtain ⟨hempty, huniv, -⟩ := hA
  unfold hazardStar
  have hp0 := Aux.pStar_pos
  have hp1 := Aux.pStar_lt_one
  have hw_nonneg : ∀ U : Finset (Fin n), 0 ≤ weight (fun _ : Fin n => pStar) U := by
    intro U; unfold weight
    apply mul_nonneg
    · exact Finset.prod_nonneg (fun v _ => hp0.le)
    · exact Finset.prod_nonneg (fun v _ => by linarith)
  have hempty_w : 0 < weight (fun _ : Fin n => pStar) (∅ : Finset (Fin n)) := by
    unfold weight
    simp
    exact pow_pos (by linarith) _
  have hpos : 0 < muOf A (fun _ : Fin n => pStar) := by
    unfold muOf
    exact lt_of_lt_of_le hempty_w (Finset.single_le_sum (fun U _ => hw_nonneg U) hempty)
  have hsub : A ⊆ (Finset.univ : Finset (Fin n)).powerset.erase Finset.univ := by
    intro U hU
    rw [Finset.mem_erase]
    exact ⟨fun h => huniv (h ▸ hU), Finset.mem_powerset.mpr (Finset.subset_univ U)⟩
  have hle : muOf A (fun _ : Fin n => pStar) ≤
      ∑ U ∈ (Finset.univ : Finset (Fin n)).powerset.erase Finset.univ,
        weight (fun _ : Fin n => pStar) U :=
    Finset.sum_le_sum_of_subset_of_nonneg hsub (fun U _ _ => hw_nonneg U)
  have hmem : (Finset.univ : Finset (Fin n)) ∈ (Finset.univ : Finset (Fin n)).powerset :=
    Finset.mem_powerset.mpr (subset_refl _)
  have hsum := Finset.add_sum_erase (Finset.univ : Finset (Fin n)).powerset
    (weight (fun _ : Fin n => pStar)) hmem
  rw [Aux.sum_weight_eq_one] at hsum
  have huniv_w : 0 < weight (fun _ : Fin n => pStar) (Finset.univ : Finset (Fin n)) := by
    unfold weight
    simp
    exact pow_pos hp0 _
  have hlt : muOf A (fun _ : Fin n => pStar) < 1 := by linarith
  have := Real.log_neg hpos hlt
  linarith

/-- Proves: `UniversalLaw.Spec.P15.OddMajorityIdentity` (register row `p15-full-price`).
Method: the one-trial recursion `binomTail (n+1) k = binomTail n k - C(n,k) p^(k+1) (1-p)^(n-k)`
(Pascal's rule, induction on `k`) applied twice, Pascal's rule once more, `ring`; the second
form by `Nat.choose_symm_half`; negativity from `p(1-2p) < 0` and a positive factor.
Does not establish: (F11) itself (the chain over `a` is in `CapacityWorstCase`). -/
theorem oddMajorityIdentity : UniversalLaw.Spec.P15.OddMajorityIdentity := by
  intro a p
  have e1 := Aux.binomTail_succ_left (2 * a + 2) (a + 1) p (by omega)
  have e2 : binomTail (2 * a + 2) (a + 1) p = binomTail (2 * a + 1) (a + 1) p -
      (Nat.choose (2 * a + 1) (a + 1) : ℝ) * p ^ (a + 1 + 1) * (1 - p) ^ (2 * a + 1 - (a + 1)) :=
    Aux.binomTail_succ_left (2 * a + 1) (a + 1) p (by omega)
  have e3 : binomTail (2 * a + 1) (a + 1) p = binomTail (2 * a + 1) a p +
      (Nat.choose (2 * a + 1) (a + 1) : ℝ) * p ^ (a + 1) * (1 - p) ^ (2 * a + 1 - (a + 1)) := by
    unfold binomTail; rw [Finset.sum_range_succ]
  have hc : (Nat.choose (2 * a + 2) (a + 1) : ℝ) =
      (Nat.choose (2 * a + 1) a : ℝ) + (Nat.choose (2 * a + 1) (a + 1) : ℝ) := by
    exact_mod_cast Nat.choose_succ_succ' (2 * a + 1) a
  have x1 : 2 * a + 2 - (a + 1) = a + 1 := by omega
  have x2 : 2 * a + 1 - (a + 1) = a := by omega
  have hd : binomTail (2 * a + 3) (a + 1) p - binomTail (2 * a + 1) a p =
      (1 - p) ^ 2 * ((Nat.choose (2 * a + 1) (a + 1) : ℝ) * p ^ (a + 1) * (1 - p) ^ a) -
        p ^ 2 * ((Nat.choose (2 * a + 1) a : ℝ) * p ^ a * (1 - p) ^ (a + 1)) := by
    rw [show 2 * a + 3 = 2 * a + 2 + 1 by omega, e1, e2, e3, hc, x1, x2]
    ring
  have hd2 : binomTail (2 * a + 3) (a + 1) p - binomTail (2 * a + 1) a p =
      p * (1 - 2 * p) * ((Nat.choose (2 * a + 1) a : ℝ) * p ^ a * (1 - p) ^ (a + 1)) := by
    rw [hd, Nat.choose_symm_half]
    ring
  refine ⟨hd, hd2, ?_⟩
  intro hp1 hp2
  rw [hd2]
  have hneg : p * (1 - 2 * p) < 0 := mul_neg_of_pos_of_neg (by linarith) (by linarith)
  have hpos : 0 < (Nat.choose (2 * a + 1) a : ℝ) * p ^ a * (1 - p) ^ (a + 1) :=
    mul_pos (mul_pos (Nat.cast_pos.mpr (Nat.choose_pos (by omega))) (pow_pos (by linarith) _))
      (pow_pos (by linarith) _)
  exact mul_neg_of_neg_of_pos hneg hpos

namespace Aux

/-- The chain of (F10) at `p ∈ (1/2, 1)`: `P(Bin(2a+1,p) ≤ a) ≤ P(Bin(3,p) ≤ 1)` for `a ≥ 1`. -/
theorem binomTail_odd_chain (a : ℕ) (ha : 1 ≤ a) (c : ℝ) (hc : 1 / 2 < c) (hc1 : c < 1) :
    binomTail (2 * a + 1) a c ≤ binomTail 3 1 c := by
  induction a, ha using Nat.le_induction with
  | base => exact le_refl _
  | succ k hk ih =>
    have h := (oddMajorityIdentity k c).2.2 hc hc1
    rw [show 2 * (k + 1) + 1 = 2 * k + 3 by ring]
    linarith

end Aux

/-- Proves: `UniversalLaw.Spec.P15.CapacityWorstCase` (register row `p15-full-price`).
Method: `mu_{p_star}(capacityFamily n a) = binomTail n a p_star` (`Finset.sum_powerset_apply_card`),
the tail decreases in `n` (one-trial recursion) and along `a ↦ a+1` at `p_star > 1/2` (F10), so
`binomTail n a ≤ binomTail 3 1` and `-log` reverses the order; `binomTail 3 1 p_star` evaluates
to `3e^{-2} - 2e^{-3} = (3e-2)e^{-3}`, whose `-log` is `h_star`; `h_star > 1` as in
`HStarExceedsOne`.
Does not establish: the local price bound (F12) or the assembly (F13); the source's remark "it
is not a finite-grid extrapolation" is a statement about its own proof, not re-derived. -/
theorem capacityWorstCase : UniversalLaw.Spec.P15.CapacityWorstCase := by
  unfold UniversalLaw.Spec.P15.CapacityWorstCase
  have hp0 := Aux.pStar_pos
  have hp12 := Aux.pStar_gt_half
  have hp1 := Aux.pStar_lt_one
  have hid : ∀ n a : ℕ, a ≤ n →
      hazardStar (capacityFamily n a) = -Real.log (binomTail n a pStar) := by
    intro n a h; unfold hazardStar; rw [Aux.muOf_capacityFamily n a h]
  refine ⟨?_, ?_, ?_, Aux.one_lt_hStar⟩
  · intro a n ha hn
    rw [hid 3 1 (by norm_num), hid n a (by omega)]
    have h1 : binomTail n a pStar ≤ binomTail (2 * a + 1) a pStar :=
      Aux.binomTail_le_of_le a (2 * a + 1) n (by omega) hn pStar hp0.le hp1.le
    have h2 : binomTail (2 * a + 1) a pStar ≤ binomTail 3 1 pStar :=
      Aux.binomTail_odd_chain a ha pStar hp12 hp1
    have hpos := Aux.binomTail_pos n a pStar hp0.le hp1
    have := Real.log_le_log hpos (le_trans h1 h2)
    linarith
  · rw [hid 3 1 (by norm_num), Aux.binomTail_three_one]
  · unfold hStar
    have e : 3 * Real.exp (-2) - 2 * Real.exp (-3) = (3 * Real.exp 1 - 2) * Real.exp (-3) := by
      rw [show Real.exp (-2) = Real.exp 1 * Real.exp (-3) by rw [← Real.exp_add]; norm_num]
      ring
    rw [e, Real.log_mul (ne_of_gt Aux.three_exp_sub_two_pos) (Real.exp_pos _).ne', Real.log_exp]
    ring

/-! ## Full price Section 5: the exact demand boundary -/

/-- Proves: `UniversalLaw.Spec.P15.DemandOneBoundary` (register row `p15-full-price`).
Method: one-block lemmas (block = ground set, good ⇔ `|U| ≤ cap`, obstruction `{X}` at `K = 1`,
`mu_p(D) = 1 - prod p_v` via `Finset.prod_add`, unit-price covers cost at least one by
`le_csInf`); `p_star^(cap+1) < p_star` for `cap ≥ 1`; `Real.log_lt_log`; for `rho ≤ 1` the capped
hazard is `min(1, rho(-log mu)) ≤ -log mu < 1 ≤ covercost`.
Does not establish: any retraction of original P15-B ("whose local-budget premise is missing in
that case"); the equality `covercost = 1` (only the lower bound is needed and stated). -/
theorem demandOneBoundary : UniversalLaw.Spec.P15.DemandOneBoundary := by
  intro cap hcap
  have hp0 := Aux.pStar_pos
  have hp1 := Aux.pStar_lt_one
  have h1p : 1 - pStar = Real.exp (-1) := by unfold pStar; ring
  have hmu : mu (demandOneBlock cap) (fun _ => pStar) = 1 - pStar ^ (cap + 1) := by
    rw [Aux.oneBlock_mu]; simp
  have hpow : pStar ^ (cap + 1) < pStar := by
    calc pStar ^ (cap + 1) ≤ pStar ^ 2 := pow_le_pow_of_le_one hp0.le hp1.le (by omega)
      _ < pStar := by nlinarith
  have hexp : Real.exp (-1) < 1 - pStar ^ (cap + 1) := by linarith
  have hmu_pos : 0 < mu (demandOneBlock cap) (fun _ => pStar) := by
    rw [hmu]; have := Real.exp_pos (-1); linarith
  have hmu_lt : mu (demandOneBlock cap) (fun _ => pStar) < 1 := by
    rw [hmu]; have := pow_pos hp0 (cap + 1); linarith
  have hlog : -Real.log (mu (demandOneBlock cap) (fun _ => pStar)) < 1 := by
    rw [hmu]
    have := Real.log_lt_log (Real.exp_pos _) hexp
    rw [Real.log_exp] at this
    linarith
  have hcc := Aux.oneBlock_coverCost_ge_one cap
  refine ⟨Aux.oneBlock_isRealized cap hcap, Aux.oneBlock_obstruction cap, hcc, hmu, hexp, hlog,
    ?_⟩
  intro ρ hρ
  unfold cappedHazard
  simp only [hmu_pos, ↓reduceIte]
  have hL : 0 < -Real.log (mu (demandOneBlock cap) (fun _ => pStar)) := by
    have := Real.log_neg hmu_pos hmu_lt; linarith
  have hmul := mul_le_mul_of_nonneg_right hρ hL.le
  calc min 1 (ρ * -Real.log (mu (demandOneBlock cap) (fun _ => pStar)))
      ≤ ρ * -Real.log (mu (demandOneBlock cap) (fun _ => pStar)) := min_le_right _ _
    _ < 1 := by linarith
    _ ≤ coverCost (demandOneBlock cap) 1 (fun _ => (1 : ℝ)) := hcc

/-! ## Realized covers Section 5: the triangle -/

/-- Proves: `UniversalLaw.Spec.P15.TriangleNeedsThree` (register row `p15-realized-covers`).
Method: `K_H(d) = sInf` of the feasible set; three singletons are feasible at `K = 3`
(`Nat.sInf_le`); sizes `0`, `1`, `2` are infeasible (`K = 0` by cardinality, `K = 1, 2` by
kernel `decide` over the finitely many label-set assignments), so `le_csInf` gives `3 ≤ sInf`.
Does not establish: display (P9) (`ElementaryPaletteFormula`), the matroid palette formula, or
anything about the realized-covers budget (P8). -/
theorem triangleNeedsThree : UniversalLaw.Spec.P15.TriangleNeedsThree := by
  unfold UniversalLaw.Spec.P15.TriangleNeedsThree paletteNumber
  have h3 : (3 : ℕ) ∈ {K : ℕ | IsPaletteFeasible 3 (completeClutter 3 1) (fun _ => 1) K} :=
    Aux.triangle_feasible3
  apply le_antisymm (Nat.sInf_le h3)
  apply le_csInf ⟨3, h3⟩
  intro K hK
  by_contra hlt
  have hK : IsPaletteFeasible 3 (completeClutter 3 1) (fun _ => 1) K := hK
  interval_cases K
  · exact Aux.triangle_not_feasible0 hK
  · exact Aux.triangle_not_feasible1 hK
  · exact Aux.triangle_not_feasible2 hK

/-! ## Full price Sections 3 and 5: (F9) for every `a`, and the sharpness of `rho_star` -/

namespace Aux

/-- (F9) as an identity for every `a` (no `a ≤ n` hypothesis): the extra terms `j > n` of the
tail vanish since `Nat.choose n j = 0`. -/
theorem muOf_capacityFamily_all (n a : ℕ) (c : ℝ) :
    muOf (capacityFamily n a) (fun _ => c) = binomTail n a c := by
  unfold muOf capacityFamily binomTail
  have hw : ∀ U : Finset (Fin n),
      weight (fun _ : Fin n => c) U = c ^ U.card * (1 - c) ^ (n - U.card) := by
    intro U; unfold weight; simp [Finset.card_univ_sdiff]
  rw [Finset.sum_filter]
  simp_rw [hw]
  rw [Finset.sum_powerset_apply_card (fun m => if m ≤ a then c ^ m * (1 - c) ^ (n - m) else 0)]
  simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_ite, mul_zero]
  rw [← Finset.sum_filter]
  have hr : (Finset.range (n + 1)).filter (fun m => m ≤ a) =
      (Finset.range (a + 1)).filter (fun m => m ≤ n) := by
    ext m; simp; omega
  rw [hr, Finset.sum_filter]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  split_ifs with h
  · ring
  · rw [Nat.choose_eq_zero_of_lt (by omega)]; simp

/-! ### The triple block: one block, `a = 1`, `d = 2`, ground set of size three, no edges -/

/-- The single block is the whole ground set. -/
theorem tripleBlock_block (i : Fin 1) : block tripleBlock i = Finset.univ := by
  unfold block
  rw [Fin.fin_one_eq_zero i]
  simp [tripleBlock]

/-- Goodness is the capacity constraint `|U| ≤ 1`. -/
theorem tripleBlock_isGood (U : Finset (Fin 3)) : IsGood tripleBlock U ↔ U.card ≤ 1 := by
  show (∀ i : Fin 1, (U ∩ block tripleBlock i).card ≤ 1) ∧
      (∀ e ∈ (∅ : Finset (Finset (Fin 1))), ¬ e ⊆ supp tripleBlock U) ↔ U.card ≤ 1
  simp [tripleBlock_block]

/-- Two-decomposability is `|U| ≤ 2`: two good parts have at most two elements in all
(`Finset.card_biUnion_le`), and a set of size `≤ 2` splits into singletons and empties. -/
theorem tripleBlock_decomp (U : Finset (Fin 3)) :
    IsKDecomposable tripleBlock 2 U ↔ U.card ≤ 2 := by
  unfold IsKDecomposable
  constructor
  · rintro ⟨parts, hgood, -, hU⟩
    rw [← hU]
    calc ((Finset.univ : Finset (Fin 2)).biUnion parts).card
        ≤ ∑ j : Fin 2, (parts j).card := Finset.card_biUnion_le
      _ ≤ ∑ _j : Fin 2, 1 := Finset.sum_le_sum (fun j _ => (tripleBlock_isGood _).mp (hgood j))
      _ = 2 := by simp
  · intro h
    have hc : U.card = 0 ∨ U.card = 1 ∨ U.card = 2 := by omega
    rcases hc with h0 | h1 | h2
    · rw [Finset.card_eq_zero] at h0
      subst h0
      refine ⟨fun _ => ∅, ?_, ?_, ?_⟩
      · intro j; rw [tripleBlock_isGood]; simp
      · intro j j' _; exact Finset.disjoint_empty_left _
      · ext v
        simp
    · obtain ⟨x, rfl⟩ := Finset.card_eq_one.mp h1
      refine ⟨fun j => if j = 0 then {x} else ∅, ?_, ?_, ?_⟩
      · intro j; rw [tripleBlock_isGood]; dsimp only; split_ifs <;> simp
      · intro j j' hjj'
        fin_cases j <;> fin_cases j' <;> simp_all
      · ext v
        simp
    · obtain ⟨x, y, hxy, rfl⟩ := Finset.card_eq_two.mp h2
      refine ⟨fun j => if j = 0 then {x} else {y}, ?_, ?_, ?_⟩
      · intro j; rw [tripleBlock_isGood]; dsimp only; split_ifs <;> simp
      · intro j j' hjj'
        fin_cases j <;> fin_cases j' <;> simp_all [hxy.symm]
      · ext v
        simp

/-- At `K = 2` the sole obstruction is the full triple. -/
theorem tripleBlock_obstruction (U : Finset (Fin 3)) :
    U ∈ obstruction tripleBlock 2 ↔ U = Finset.univ := by
  unfold obstruction
  change ¬ IsKDecomposable tripleBlock 2 U ↔ U = Finset.univ
  rw [tripleBlock_decomp, not_le]
  constructor
  · intro h
    by_contra hne
    have := card_le_of_ne_univ 2 U hne
    omega
  · rintro rfl
    rw [Finset.card_univ, Fintype.card_fin]; omega

/-- The side conditions: `a = 1 ≥ 1`, `d = 2 ≥ 1`, `|X| = 3 = 1 * 2 + 1`, no edges. -/
theorem tripleBlock_isRealized : IsRealizedFamily tripleBlock := by
  refine ⟨fun _ => le_rfl, fun _ => by simp [tripleBlock], ?_, ?_, ?_⟩
  · change ∀ i : Fin 1, (block tripleBlock i).card = 1 * 2 + 1
    intro i
    rw [tripleBlock_block]
    simp
  · change ∀ e ∈ (∅ : Finset (Finset (Fin 1))), 2 ≤ e.card
    simp
  · change ∀ e ∈ (∅ : Finset (Finset (Fin 1))), ∀ f ∈ (∅ : Finset (Finset (Fin 1))),
      e ⊆ f → e = f
    simp

/-- The full-block family is `{X}`. -/
theorem tripleBlock_fullBlockFamily : fullBlockFamily tripleBlock = {Finset.univ} := by
  ext g
  simp only [fullBlockFamily, Finset.mem_image, Finset.mem_univ, true_and, Finset.mem_singleton]
  constructor
  · rintro ⟨i, rfl⟩
    exact tripleBlock_block i
  · rintro rfl
    exact ⟨(0 : Fin 1), tripleBlock_block 0⟩

/-- `K_H(d) = 2` for one block of demand two and no edges: `{0, 1}` is feasible at `K = 2`, and
a label set of size `≥ 2` needs `K ≥ 2`. -/
theorem tripleBlock_palette : paletteNumber 1 ∅ (fun _ => 2) = 2 := by
  unfold paletteNumber
  have h2 : (2 : ℕ) ∈ {K : ℕ | IsPaletteFeasible 1 ∅ (fun _ => 2) K} := by
    refine ⟨fun _ => Finset.univ, fun _ => by simp, fun e he => by simp at he⟩
  apply le_antisymm (Nat.sInf_le h2)
  apply le_csInf ⟨2, h2⟩
  intro K hK
  obtain ⟨P, hP, -⟩ := hK
  have h2le : 2 ≤ (P 0).card := hP 0
  have h0 : (P 0).card ≤ Fintype.card (Fin K) := Finset.card_le_univ _
  rw [Fintype.card_fin] at h0
  omega

/-- `mu_p(D)` for the triple block is the mass of the capacity family `(3, 1)`. -/
theorem tripleBlock_mu (p : Fin 3 → ℝ) : mu tripleBlock p = muOf (capacityFamily 3 1) p := by
  unfold mu muOf capacityFamily
  rw [Finset.sum_filter]
  refine Finset.sum_congr rfl (fun U _ => ?_)
  by_cases h : U.card ≤ 1
  · rw [Set.indicator_of_mem (show U ∈ downset tripleBlock from (tripleBlock_isGood U).mpr h)]
    simp [h]
  · rw [Set.indicator_of_notMem
      (show U ∉ downset tripleBlock from fun hm => h ((tripleBlock_isGood U).mp hm))]
    simp [h]

/-- At unit prices every generator costs one (the empty generator included). -/
theorem genPrice_one (g : Finset (Fin 3)) : genPrice (fun _ => (1 : ℝ)) g = 1 := by
  simp [genPrice]

/-- At unit prices every cover of `{X}` costs at least one. -/
theorem tripleBlock_cover_ge_one (G : Finset (Finset (Fin 3)))
    (hG : IsGeneratorCover tripleBlock 2 G) : 1 ≤ famPrice (fun _ => (1 : ℝ)) G := by
  have huniv : (Finset.univ : Finset (Fin 3)) ∈ obstruction tripleBlock 2 :=
    (tripleBlock_obstruction _).mpr rfl
  obtain ⟨g, hg, -⟩ := hG Finset.univ huniv
  unfold famPrice
  calc (1 : ℝ) = genPrice (fun _ => (1 : ℝ)) g := (genPrice_one g).symm
    _ ≤ ∑ g ∈ G, genPrice (fun _ => (1 : ℝ)) g :=
        Finset.single_le_sum (fun g _ => by rw [genPrice_one]; norm_num) hg

/-- The full-block family `{X}` covers the obstruction `{X}`. -/
theorem tripleBlock_fullBlock_cover :
    IsGeneratorCover tripleBlock 2 (fullBlockFamily tripleBlock) := by
  intro U hU
  rw [tripleBlock_obstruction] at hU
  rw [tripleBlock_fullBlockFamily]
  exact ⟨Finset.univ, Finset.mem_singleton_self _, by rw [hU]⟩

/-- The full-block cover costs exactly one at unit prices. -/
theorem tripleBlock_fullBlock_price :
    famPrice (fun _ => (1 : ℝ)) (fullBlockFamily tripleBlock) = 1 := by
  rw [tripleBlock_fullBlockFamily]
  simp [famPrice, genPrice]

/-- The exact cover cost is one: `{X}` attains the lower bound one. -/
theorem tripleBlock_coverCost : coverCost tripleBlock 2 (fun _ => (1 : ℝ)) = 1 := by
  unfold coverCost
  apply IsLeast.csInf_eq
  refine ⟨⟨fullBlockFamily tripleBlock, tripleBlock_fullBlock_cover,
    tripleBlock_fullBlock_price⟩, ?_⟩
  rintro t ⟨G, hG, rfl⟩
  exact tripleBlock_cover_ge_one G hG

/-- `mu_{p_star}(D) = P(Bin(3, p_star) ≤ 1) = 3e^{-2} - 2e^{-3}`. -/
theorem tripleBlock_mu_pStar :
    mu tripleBlock (fun _ => pStar) = 3 * Real.exp (-2) - 2 * Real.exp (-3) := by
  rw [tripleBlock_mu, muOf_capacityFamily_all, binomTail_three_one]

/-- The available hazard of the triple block at `p_star` is `h_star`. -/
theorem tripleBlock_hazard : -Real.log (mu tripleBlock (fun _ => pStar)) = hStar := by
  rw [tripleBlock_mu_pStar]
  exact capacityWorstCase.2.2.1

/-- Any `ρ < ρ_star = 1/h_star` makes the capped hazard `min(1, ρ h_star)` strictly less than
one. -/
theorem tripleBlock_sharp (ρ : ℝ) (hρ : ρ < rhoStar) :
    cappedHazard ρ (mu tripleBlock (fun _ => pStar)) < 1 := by
  have hmu_pos : 0 < mu tripleBlock (fun _ => pStar) := by
    rw [tripleBlock_mu, muOf_capacityFamily_all]
    exact binomTail_pos 3 1 pStar pStar_pos.le pStar_lt_one
  have hlog := tripleBlock_hazard
  have hh : 0 < hStar := by linarith [one_lt_hStar]
  have hrho : rhoStar * hStar = 1 := by
    show 1 / hStar * hStar = 1
    exact one_div_mul_cancel hh.ne'
  unfold cappedHazard
  simp only [hmu_pos, ↓reduceIte]
  rw [hlog]
  calc min 1 (ρ * hStar) ≤ ρ * hStar := min_le_right _ _
    _ < rhoStar * hStar := mul_lt_mul_of_pos_right hρ hh
    _ = 1 := hrho

end Aux

/-- Proves: `UniversalLaw.Spec.P15.CapacityHazardIsBinomial` (claim `p15-full-price`, display (F9)).
Method: `hazardStar` unfolds to `-log (muOf (capacityFamily n a) p_star)`, and the mass of the
capacity family at a constant vector is `binomTail n a` for EVERY `a` (`Finset.sum_powerset_apply_card`
groups the subsets by cardinality; for `a > n` the extra terms `j > n` vanish since
`Nat.choose n j = 0`), which extends `Aux.muOf_capacityFamily` beyond `a ≤ n`.
Does not establish: the informal claim's review or acceptance; the alignment of `capacityFamily` and
`binomTail` with the source's `A = {S : |S| ≤ a}` and `P(Bin(n, p_star) ≤ a)`
(PENDING_INDEPENDENT_REVIEW); displays (F10)-(F11) (`CapacityWorstCase`) or anything about the
budget (F2). -/
theorem capacityHazardIsBinomial : UniversalLaw.Spec.P15.CapacityHazardIsBinomial := by
  intro n a
  unfold hazardStar
  rw [Aux.muOf_capacityFamily_all n a]

/-- Proves: `UniversalLaw.Spec.P15.SharpnessOfRhoStar` (claim `math.p15-full-price`, full-price note Section 5).
Method: for `tripleBlock` the single block is the ground set, good sets are those of size `≤ 1`, and a
set is `2`-decomposable iff it has size `≤ 2` (two singletons/empties; `Finset.card_biUnion_le` for the
converse), so the obstruction at `K = 2` is `{X}`; `K_H(d) = 2` by `Nat.sInf_le`/`le_csInf`; at unit
prices every generator costs `1`, every cover of `{X}` costs `≥ 1` (`Finset.single_le_sum`) and the
full-block cover `{X}` costs `1`, hence `coverCost = 1` (`IsLeast.csInf_eq`); `mu_p(D)` is the mass of
the capacity family `(3, 1)`, i.e. `binomTail 3 1 p_star = 3e^{-2} - 2e^{-3}`, whose `-log` is `h_star`
(the identity proved in `capacityWorstCase`); for `ρ < ρ_star = 1/h_star`,
`min 1 (ρ h_star) ≤ ρ h_star < 1`.
Does not establish: the informal sharpness claim's review or acceptance; the alignment of the Lean
`tripleBlock`, `obstruction`, `coverCost` and `cappedHazard` with the source's objects
(PENDING_INDEPENDENT_REVIEW); sharpness for any family other than the triple block; the budget (F2)
itself; any necessity claim for multi-block families with some demand one ("other constraints may
change its budget"). -/
theorem sharpnessOfRhoStar : UniversalLaw.Spec.P15.SharpnessOfRhoStar :=
  ⟨Aux.tripleBlock_isRealized, Aux.tripleBlock_palette, Aux.tripleBlock_obstruction, phiAtPStar,
    Aux.genPrice_one, Aux.tripleBlock_cover_ge_one, Aux.tripleBlock_fullBlock_cover,
    Aux.tripleBlock_fullBlock_price, Aux.tripleBlock_coverCost, Aux.tripleBlock_hazard,
    Aux.tripleBlock_sharp⟩

end UniversalLaw.Proofs.P15Elementary
