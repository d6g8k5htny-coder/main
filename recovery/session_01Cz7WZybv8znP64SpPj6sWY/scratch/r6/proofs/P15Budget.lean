import Specifications.P15
import SpecificationProofs.P15Structure
import SpecificationProofs.P15Elementary
import SpecificationProofs.P15Hazard

/-!
# P15 budget proofs: the hazard bound (P8), the restricted transformed-price budget (T1), (T4),
(T5) and the six-block transformed range

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/P15.lean` (statement packet `specifications/lean_statements_20261010`, which is
not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop` only:
it does not make the informal Layer 0 note reviewed, accepted or "verified", it does not
identify the formal definitions with the informal objects (that alignment is
PENDING_INDEPENDENT_REVIEW), and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.P15Budget.Aux`; the gate inventories only the
theorems outside that namespace. The module assembles the block-local price bounds of
`P15Hazard` (the marginal bound `prod_{v in S} p_v ≤ q_i` for `|S| ≥ a_i + 1`) and of
`P15Elementary` (`-log(1-p) ≤ p/(1-p)` and the exponent arithmetic `4^(a+1)/3^n ≤ 16/27`) with
the cover criterion `fullBlockCoverIff` of `P15Structure` and the factorised hazard
`sum_i q_i ≤ -log mu_p(D)` of `P15Hazard` into bounds on the infimum `coverCost`: the set of cover
prices is bounded below by `0`, the cover `{∅}` has price `1`, and the full-block family covers
the obstruction exactly when `K ≥ K_H(d)`.
-/

namespace UniversalLaw.Proofs.P15Budget

open UniversalLaw.Spec.P15
open Finset

namespace Aux

/-! ### Cover prices and the infimum `coverCost` -/

/-- Family prices are nonnegative for nonnegative prices. -/
theorem famPrice_nonneg {Y : Type} {c : Y → ℝ} (hc : ∀ v, 0 ≤ c v) (G : Finset (Finset Y)) :
    0 ≤ famPrice c G :=
  Finset.sum_nonneg fun _ _ => Finset.prod_nonneg fun v _ => hc v

/-- The set of prices of generator covers is bounded below by `0`. -/
theorem coverSet_bddBelow {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (K : ℕ)
    {c : X → ℝ} (hc : ∀ v, 0 ≤ c v) :
    BddBelow {t : ℝ | ∃ G : Finset (Finset X), IsGeneratorCover F K G ∧ famPrice c G = t} := by
  refine ⟨0, ?_⟩
  rintro t ⟨G, -, rfl⟩
  exact famPrice_nonneg hc G

/-- The infimum is at most the price of any generator cover. -/
theorem coverCost_le_famPrice {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (K : ℕ) {c : X → ℝ} (hc : ∀ v, 0 ≤ c v) {G : Finset (Finset X)}
    (hG : IsGeneratorCover F K G) : coverCost F K c ≤ famPrice c G :=
  csInf_le (coverSet_bddBelow F K hc) ⟨G, hG, rfl⟩

/-- `{∅}` covers every obstruction. -/
theorem isGeneratorCover_singleton_empty {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) (K : ℕ) : IsGeneratorCover F K {∅} :=
  fun U _ => ⟨∅, Finset.mem_singleton_self _, Finset.empty_subset U⟩

/-- The cover `{∅}` has price `1`. -/
theorem famPrice_singleton_empty {Y : Type} (c : Y → ℝ) : famPrice c {∅} = 1 := by
  simp [famPrice, genPrice]

/-- `covercost ≤ 1` (the cover `{∅}`). -/
theorem coverCost_le_one {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (K : ℕ)
    {c : X → ℝ} (hc : ∀ v, 0 ≤ c v) : coverCost F K c ≤ 1 := by
  have h := coverCost_le_famPrice F K hc (isGeneratorCover_singleton_empty F K)
  rwa [famPrice_singleton_empty] at h

/-- For `K ≥ K_H(d)` the full blocks cover, so `covercost ≤ sum_i price(X_i)`. -/
theorem coverCost_le_sum_block {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}
    (hF : IsRealizedFamily F) {c : X → ℝ} (hc : ∀ v, 0 ≤ c v) {K : ℕ}
    (hK : paletteNumber F.b F.H F.d ≤ K) :
    coverCost F K c ≤ ∑ i, genPrice c (block F i) := by
  have hcover : IsGeneratorCover F K (fullBlockFamily F) :=
    (P15Structure.fullBlockCoverIff F hF K).2 hK
  rw [← P15Structure.Aux.famPrice_fullBlockFamily hF c]
  exact coverCost_le_famPrice F K hc hcover

/-- A quantity at most `1` and, for `m > 0`, at most `ρ (-log m)` is at most the capped hazard. -/
theorem le_cappedHazard_of {ρ m x : ℝ} (hx1 : x ≤ 1) (hx2 : 0 < m → x ≤ ρ * -Real.log m) :
    x ≤ cappedHazard ρ m := by
  unfold cappedHazard
  split_ifs with h
  · exact le_min hx1 (hx2 h)
  · exact hx1

/-- `sum_i q_i ≤ -log mu_p(D)` when `mu_p(D) > 0` (display (P5)/(T6), from `P15Hazard`). -/
theorem sum_localFailure_le_neg_log_mu {X : Type} [Fintype X] [DecidableEq X]
    {F : RealizedData X} (hF : IsRealizedFamily F) {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1)
    (hmu : 0 < mu F p) : ∑ i, localFailure F p i ≤ -Real.log (mu F p) := by
  have h := (P15Hazard.independentBlockHazard F hF p hp).2 hmu
  exact h.1.trans h.2

/-! ### The ratio bound of (T3)/(T5) -/

/-- `phi(p) ≤ p/(1-p)` for `0 ≤ p < 1` (display (T2)). -/
theorem phi_le_div {p : ℝ} (h0 : 0 ≤ p) (h1 : p < 1) : phi p ≤ p / (1 - p) := by
  unfold phi
  simp only [h1, ↓reduceIte]
  exact (min_le_right _ _).trans (P15Elementary.logRatioBound p h0 h1)

/-- The first two steps of (T3): `prod_{X_i} c ≤ prod_{X_i} p/(1-p)`, split along `S ⊆ X_i`. -/
theorem genPrice_le_ratio {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p c : X → ℝ} (hc : ∀ v, 0 ≤ c v ∧ c v ≤ phi (p v)) {i : Fin F.b}
    (hp : ∀ v ∈ block F i, 0 ≤ p v ∧ p v < 1) {S : Finset X} (hS : S ⊆ block F i) :
    genPrice c (block F i) ≤
      (∏ v ∈ S, p v) * ((∏ v ∈ block F i \ S, p v) * ∏ v ∈ block F i, (1 - p v)⁻¹) := by
  have h1 : genPrice c (block F i) ≤ ∏ v ∈ block F i, p v * (1 - p v)⁻¹ := by
    unfold genPrice
    refine Finset.prod_le_prod₀ (fun v _ => (hc v).1) fun v hv => ?_
    rw [← div_eq_mul_inv]
    exact (hc v).2.trans (phi_le_div (hp v hv).1 (hp v hv).2)
  refine h1.trans (le_of_eq ?_)
  rw [Finset.prod_mul_distrib,
    show ∏ v ∈ block F i, p v = (∏ v ∈ block F i \ S, p v) * ∏ v ∈ S, p v from
      (Finset.prod_sdiff hS).symm]
  ring

/-- A product of coordinates in `[0, B]` is at most `B^|s|`. -/
theorem prod_le_pow_of_le {X : Type} {s : Finset X} {p : X → ℝ} {B : ℝ}
    (hp : ∀ v ∈ s, 0 ≤ p v ∧ p v ≤ B) : ∏ v ∈ s, p v ≤ B ^ s.card := by
  rw [← Finset.prod_const]
  exact Finset.prod_le_prod₀ (fun v hv => (hp v hv).1) (fun v hv => (hp v hv).2)

/-- A product of `(1 - p_v)⁻¹` with `p_v ≤ B < 1` is at most `((1 - B)⁻¹)^|s|`. -/
theorem prod_inv_le_pow {X : Type} {s : Finset X} {p : X → ℝ} {B : ℝ} (hB : B < 1)
    (hp : ∀ v ∈ s, p v ≤ B) : ∏ v ∈ s, (1 - p v)⁻¹ ≤ ((1 - B)⁻¹) ^ s.card := by
  rw [← Finset.prod_const]
  refine Finset.prod_le_prod₀ (fun v hv => ?_) (fun v hv => ?_)
  · exact inv_nonneg.2 (by linarith [hp v hv])
  · exact inv_anti₀ (by linarith) (by linarith [hp v hv])

/-- The ratio bound: for `0 ≤ c ≤ phi(p)`, `0 ≤ p ≤ B < 1` on the block and `S ⊆ X_i`,
`price(X_i) ≤ (prod_{S} p) * B^(n - |S|) * (1 - B)^(-n)`. -/
theorem genPrice_le_prod_mul_ratio {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p c : X → ℝ} (hc : ∀ v, 0 ≤ c v ∧ c v ≤ phi (p v)) {i : Fin F.b} {B : ℝ} (hB0 : 0 ≤ B)
    (hB : B < 1) (hp : ∀ v ∈ block F i, 0 ≤ p v ∧ p v ≤ B) {S : Finset X}
    (hS : S ⊆ block F i) :
    genPrice c (block F i) ≤
      (∏ v ∈ S, p v) * (B ^ ((block F i).card - S.card) * ((1 - B)⁻¹) ^ (block F i).card) := by
  have hp' : ∀ v ∈ block F i, 0 ≤ p v ∧ p v < 1 :=
    fun v hv => ⟨(hp v hv).1, (hp v hv).2.trans_lt hB⟩
  refine (genPrice_le_ratio F hc hp' hS).trans ?_
  refine mul_le_mul_of_nonneg_left ?_ (Finset.prod_nonneg fun v hv => (hp v (hS hv)).1)
  rw [← Finset.card_sdiff_of_subset hS]
  refine mul_le_mul (prod_le_pow_of_le fun v hv => hp v (mem_sdiff.1 hv).1)
    (prod_inv_le_pow hB fun v hv => (hp v hv).2)
    (Finset.prod_nonneg fun v hv => inv_nonneg.2 (by linarith [(hp' v hv).2]))
    (pow_nonneg hB0 _)

/-- The exponent identity behind (T3): `(1/4)^m (4/3)^(m+k) = 4^k / 3^(m+k)`. -/
theorem quarter_pow_identity (m k : ℕ) :
    (1 / 4 : ℝ) ^ m * (4 / 3) ^ (m + k) = 4 ^ k / 3 ^ (m + k) := by
  rw [div_pow, div_pow, one_pow, pow_add]
  field_simp

/-- Display (T3) in the form used here: for `a ≥ 1`, `n ≥ 2a + 1`,
`(1/4)^(n-a-1) (3/4)^(-n) ≤ 16/27`. -/
theorem quarter_ratio_le {a n : ℕ} (ha : 1 ≤ a) (hn : 2 * a + 1 ≤ n) :
    (1 / 4 : ℝ) ^ (n - (a + 1)) * ((1 - 1 / 4)⁻¹) ^ n ≤ 16 / 27 := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + (a + 1) := ⟨n - (a + 1), (Nat.sub_add_cancel (by omega)).symm⟩
  rw [Nat.add_sub_cancel, show ((1 : ℝ) - 1 / 4)⁻¹ = 4 / 3 by norm_num, quarter_pow_identity]
  have h := P15Elementary.ratioExponentBound a (m + (a + 1)) ha hn
  exact h.1.trans h.2

/-! ### Block-local hypotheses -/

/-- `localGood` depends on `p` only through its values on the block. -/
theorem localGood_congr {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p p' : X → ℝ} {i : Fin F.b} (h : ∀ v ∈ block F i, p v = p' v) :
    localGood F p i = localGood F p' i := by
  unfold localGood
  refine Finset.sum_congr rfl fun S hS => ?_
  rw [mem_powerset] at hS
  have e1 : ∏ v ∈ S, p v = ∏ v ∈ S, p' v := Finset.prod_congr rfl fun v hv => h v (hS hv)
  have e2 : ∏ v ∈ block F i \ S, (1 - p v) = ∏ v ∈ block F i \ S, (1 - p' v) :=
    Finset.prod_congr rfl fun v hv => by rw [h v (mem_sdiff.1 hv).1]
  by_cases hmem : S ∈ downset F
  · rw [Set.indicator_of_mem hmem, Set.indicator_of_mem hmem, e1, e2]
  · rw [Set.indicator_of_notMem hmem, Set.indicator_of_notMem hmem]

/-- `localFailure` depends on `p` only through its values on the block. -/
theorem localFailure_congr {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p p' : X → ℝ} {i : Fin F.b} (h : ∀ v ∈ block F i, p v = p' v) :
    localFailure F p i = localFailure F p' i := by
  unfold localFailure
  rw [localGood_congr F h]

/-- The marginal bound `prod_{v in S} p_v ≤ q_i` of `P15Hazard` under block-local hypotheses
only (`p` is replaced by its restriction to the block, zero elsewhere). -/
theorem prod_le_localFailure_of_card_local {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) {p : X → ℝ} {i : Fin F.b} (hp : ∀ v ∈ block F i, 0 ≤ p v ∧ p v ≤ 1)
    {S : Finset X} (hS : S ⊆ block F i) (hcard : F.a i + 1 ≤ S.card) :
    ∏ v ∈ S, p v ≤ localFailure F p i := by
  set p' : X → ℝ := fun v => if v ∈ block F i then p v else 0 with hp'def
  have hagree : ∀ v ∈ block F i, p v = p' v := fun v hv => by simp [hp'def, hv]
  have hp' : ∀ v, 0 ≤ p' v ∧ p' v ≤ 1 := fun v => by
    by_cases hv : v ∈ block F i
    · simp only [hp'def, hv, ↓reduceIte]
      exact hp v hv
    · simp only [hp'def, hv, ↓reduceIte]
      exact ⟨le_rfl, zero_le_one⟩
  rw [localFailure_congr F hagree, Finset.prod_congr rfl fun v hv => hagree v (hS hv)]
  exact P15Hazard.Aux.prod_le_localFailure_of_card F hp' hS hcard

/-! ### The local budget (T4) -/

/-- Display (T4), first inequality: for `a_i ≥ 1`, `d_i ≥ 2`, `0 ≤ p ≤ 1/4` and
`0 ≤ c ≤ phi(p)`, `price(X_i) ≤ (16/27) q_i`. -/
theorem genPrice_le_budget {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}
    (hF : IsRealizedFamily F) (hd : ∀ i, 2 ≤ F.d i) {p c : X → ℝ}
    (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1 / 4) (hc : ∀ v, 0 ≤ c v ∧ c v ≤ phi (p v)) (i : Fin F.b) :
    genPrice c (block F i) ≤ (16 / 27) * localFailure F p i := by
  have hn : (block F i).card = F.a i * F.d i + 1 := hF.2.2.1 i
  have ha : 1 ≤ F.a i := hF.1 i
  have hmul : F.a i * 2 ≤ F.a i * F.d i := Nat.mul_le_mul_left (F.a i) (hd i)
  have hn2 : 2 * F.a i + 1 ≤ (block F i).card := by rw [hn]; linarith
  have hcard : F.a i + 1 ≤ (block F i).card := by linarith
  obtain ⟨S, hS, hScard⟩ := Finset.exists_subset_card_eq hcard
  have hp1 : ∀ v, 0 ≤ p v ∧ p v ≤ 1 := fun v => ⟨(hp v).1, (hp v).2.trans (by norm_num)⟩
  have h1 := genPrice_le_prod_mul_ratio F hc (by norm_num : (0 : ℝ) ≤ 1 / 4)
    (by norm_num : (1 / 4 : ℝ) < 1) (fun v _ => hp v) hS
  have hprodS : ∏ v ∈ S, p v ≤ localFailure F p i :=
    P15Hazard.Aux.prod_le_localFailure_of_card F hp1 hS (by rw [hScard])
  have hprodS0 : 0 ≤ ∏ v ∈ S, p v := Finset.prod_nonneg fun v _ => (hp v).1
  calc genPrice c (block F i)
      ≤ (∏ v ∈ S, p v) *
          ((1 / 4) ^ ((block F i).card - S.card) * ((1 - 1 / 4)⁻¹) ^ (block F i).card) := h1
    _ ≤ (∏ v ∈ S, p v) * (16 / 27) := by
        refine mul_le_mul_of_nonneg_left ?_ hprodS0
        rw [hScard]
        exact quarter_ratio_le ha hn2
    _ = (16 / 27) * ∏ v ∈ S, p v := mul_comm _ _
    _ ≤ (16 / 27) * localFailure F p i := mul_le_mul_of_nonneg_left hprodS (by norm_num)

/-- The summed budget: `sum_i price(X_i) ≤ (16/27) (-log mu_p(D))` when `mu_p(D) > 0`. -/
theorem sum_genPrice_le_budget {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}
    (hF : IsRealizedFamily F) (hd : ∀ i, 2 ≤ F.d i) {p c : X → ℝ}
    (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1 / 4) (hc : ∀ v, 0 ≤ c v ∧ c v ≤ phi (p v))
    (hmu : 0 < mu F p) :
    ∑ i, genPrice c (block F i) ≤ (16 / 27) * -Real.log (mu F p) := by
  have hp1 : ∀ v, 0 ≤ p v ∧ p v ≤ 1 := fun v => ⟨(hp v).1, (hp v).2.trans (by norm_num)⟩
  calc ∑ i, genPrice c (block F i)
      ≤ ∑ i, (16 / 27) * localFailure F p i :=
        Finset.sum_le_sum fun i _ => genPrice_le_budget hF hd hp hc i
    _ = (16 / 27) * ∑ i, localFailure F p i := by rw [Finset.mul_sum]
    _ ≤ (16 / 27) * -Real.log (mu F p) :=
        mul_le_mul_of_nonneg_left (sum_localFailure_le_neg_log_mu hF hp1 hmu) (by norm_num)

/-! ### Positivity of `mu_p(D)` -/

/-- The weight is nonnegative for `p in [0,1]^X`. -/
theorem weight_nonneg {Y : Type} [Fintype Y] [DecidableEq Y] {p : Y → ℝ}
    (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (U : Finset Y) : 0 ≤ weight p U := by
  unfold weight
  refine mul_nonneg (Finset.prod_nonneg fun v _ => (hp v).1) (Finset.prod_nonneg fun v _ => ?_)
  linarith [(hp v).2]

/-- The empty set is good when every edge of `H` has at least two elements. -/
theorem isGood_empty {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}
    (hH : ∀ e ∈ F.H, 2 ≤ e.card) : IsGood F ∅ := by
  refine ⟨fun i => by simp, fun e he hsub => ?_⟩
  have h0 : e = ∅ := subset_empty.1 (by simpa [supp] using hsub)
  have h2 := hH e he
  rw [h0, card_empty] at h2
  omega

/-- `mu_p(D) > 0` for `0 ≤ p < 1` coordinatewise: the empty set is good and has positive weight. -/
theorem mu_pos {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}
    (hF : IsRealizedFamily F) {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v < 1) : 0 < mu F p := by
  have hp1 : ∀ v, 0 ≤ p v ∧ p v ≤ 1 := fun v => ⟨(hp v).1, (hp v).2.le⟩
  have hterm : ∀ U ∈ (univ : Finset X).powerset, 0 ≤ Set.indicator (downset F) (weight p) U :=
    fun U _ => Set.indicator_nonneg (fun T _ => weight_nonneg hp1 T) U
  have hle := Finset.single_le_sum hterm (mem_powerset.2 (empty_subset univ))
  refine lt_of_lt_of_le ?_ hle
  rw [Set.indicator_of_mem (show (∅ : Finset X) ∈ downset F from isGood_empty hF.2.2.2.1)]
  unfold weight
  rw [prod_empty, one_mul]
  exact Finset.prod_pos fun v _ => by linarith [(hp v).2]

/-! ### The six-block benchmark at `p = 1/40900` -/

/-- `log(40900/40899) ≤ 1/40899`. -/
theorem log_benchmark_le : Real.log (40900 / 40899) ≤ 1 / 40899 := by
  have h := Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 40900 / 40899)
  linarith

/-- `phi(1/40900) = log(40900/40899)`. -/
theorem phi_benchmark : phi (1 / 40900) = Real.log (40900 / 40899) := by
  have hlt : (1 : ℝ) / 40900 < 1 := by norm_num
  unfold phi
  simp only [hlt, ↓reduceIte]
  rw [show (1 : ℝ) - 1 / 40900 = (40900 / 40899)⁻¹ by norm_num, Real.log_inv, neg_neg]
  exact min_eq_right (log_benchmark_le.trans (by norm_num))

/-- Each full-block price at `c ≤ log(40900/40899)` is at most `1/40899`. -/
theorem genPrice_benchmark_le {c : Fin 6 × Fin 409 → ℝ}
    (hc : ∀ v, 0 ≤ c v ∧ c v ≤ Real.log (40900 / 40899)) (i : Fin benchmark816.b) :
    genPrice c (block benchmark816 i) ≤ 1 / 40899 := by
  calc genPrice c (block benchmark816 i)
      ≤ ∏ _v ∈ block benchmark816 i, (1 / 40899 : ℝ) :=
        Finset.prod_le_prod₀ (fun v _ => (hc v).1) (fun v _ => (hc v).2.trans log_benchmark_le)
    _ = (1 / 40899) ^ 409 := by rw [prod_const, P15Structure.Aux.card_block_benchmark816]
    _ ≤ 1 / 40899 := pow_le_of_le_one (by norm_num) (by norm_num) (by norm_num)

/-- The full-block price at `c ≤ log(40900/40899)` is at most `6/40899 ≤ 1`. -/
theorem famPrice_benchmark_le_one {c : Fin 6 × Fin 409 → ℝ}
    (hc : ∀ v, 0 ≤ c v ∧ c v ≤ Real.log (40900 / 40899)) :
    famPrice c (fullBlockFamily benchmark816) ≤ 1 := by
  rw [P15Structure.Aux.famPrice_fullBlockFamily P15Structure.Aux.benchmark816_isRealized c]
  calc ∑ i, genPrice c (block benchmark816 i)
      ≤ ∑ _i : Fin benchmark816.b, (1 / 40899 : ℝ) :=
        Finset.sum_le_sum fun i _ => genPrice_benchmark_le hc i
    _ = 6 * (1 / 40899) := by
        rw [sum_const, card_univ]
        change (6 : ℕ) • ((1 : ℝ) / 40899) = 6 * (1 / 40899)
        rw [nsmul_eq_mul]
        norm_num
    _ ≤ 1 := by norm_num

end Aux

/-! ## Inventory theorems -/

/-- Proves: `UniversalLaw.Spec.P15.CoverCostHazardBound` (claim `p15-realized-covers`, display
(P8)).
Method: the set of cover prices is bounded below by `0`; the cover `{∅}` has price `1`, so
`covercost ≤ 1` (`csInf_le`); for `K ≥ K_H(d)` the full blocks cover (`fullBlockCoverIff`), so
`covercost ≤ sum_i price(X_i) ≤ sum_i q_i` (display (P4), `P15Hazard.localPriceChain`) and, when
`mu_p(D) > 0`, `sum_i q_i ≤ -log mu_p(D)` (display (P5), `P15Hazard.independentBlockHazard`).
Does not establish: the informal display (P8) or its review and acceptance; anything for prices
`c_v > p_v`; attainment of the infimum; the alignment of `coverCost`, `mu`, `localFailure` with
the source's objects (PENDING_INDEPENDENT_REVIEW); the unrestricted P15 prize or original P15-B;
the disposition HOLD_WITH_DOMAIN of `p15-realized-covers`, which is transcribed in the register,
not derived here. -/
theorem coverCostHazardBound {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.CoverCostHazardBound F := by
  intro hF p c hp hc K hK
  have hc0 : ∀ v, 0 ≤ c v := fun v => (hc v).1
  refine Aux.le_cappedHazard_of (Aux.coverCost_le_one F K hc0) fun hmu => ?_
  rw [one_mul]
  calc coverCost F K c ≤ ∑ i, genPrice c (block F i) := Aux.coverCost_le_sum_block hF hc0 hK
    _ ≤ ∑ i, localFailure F p i := by
        refine Finset.sum_le_sum fun i _ => ?_
        have h := P15Hazard.localPriceChain F hF p c hp hc i
        exact h.1.trans h.2.1
    _ ≤ -Real.log (mu F p) := Aux.sum_localFailure_le_neg_log_mu hF hp hmu

/-- Proves: `UniversalLaw.Spec.P15.RestrictedLocalBudget` (claim `p15-price-budget-restricted`,
display (T4)).
Method: `phi(p) ≤ -log(1-p) ≤ p/(1-p)` (display (T2), `P15Elementary.logRatioBound`); a subset
`S ⊆ X_i` with `|S| = a_i + 1` exists since `|X_i| = a_i d_i + 1`; `prod_{X_i} c ≤ prod_{X_i}
p/(1-p) ≤ (prod_S p) (1/4)^(n-a-1) (4/3)^n = (prod_S p) 4^(a+1)/3^n ≤ (16/27) prod_S p`
(display (T3), `P15Elementary.ratioExponentBound` with `n ≥ 2a + 1` from `d_i ≥ 2`); `prod_S p ≤
q_i` (block-local marginal bound of `P15Hazard`); `(16/27) q ≤ q ≤ phi(q)` on `[0, 1]`.
Does not establish: the informal display (T4) or its review and acceptance; sharpness of the
range `p_v ≤ 1/4` (the source: "a sufficient range, not an asserted sharp threshold"); anything
at `d_i = 1` (the source's refutation stands); the alignment of `localFailure`, `phi` with the
source's objects (PENDING_INDEPENDENT_REVIEW); the disposition HOLD_WITH_DOMAIN of
`p15-price-budget-restricted`, which is transcribed in the register, not derived here. -/
theorem restrictedLocalBudget {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.RestrictedLocalBudget F := by
  intro hF hd p c hp hc i
  have hp1 : ∀ v, 0 ≤ p v ∧ p v ≤ 1 := fun v => ⟨(hp v).1, (hp v).2.trans (by norm_num)⟩
  refine ⟨Aux.genPrice_le_budget hF hd hp hc i, ?_⟩
  have hq0 := P15Hazard.Aux.localFailure_nonneg F hp1 i
  have hq1 := P15Hazard.Aux.localFailure_le_one F hp1 i
  calc (16 / 27) * localFailure F p i ≤ 1 * localFailure F p i :=
        mul_le_mul_of_nonneg_right (by norm_num) hq0
    _ = localFailure F p i := one_mul _
    _ ≤ phi (localFailure F p i) := P15Hazard.Aux.le_phi_self hq1

/-- Proves: `UniversalLaw.Spec.P15.RhoTestSufficient` (claim `p15-price-budget-restricted`,
display (T5)).
Method: the same ratio argument as (T3) with the common bound `p_*`: for `S ⊆ X_i` with
`|S| = a_i + 1`, `prod_{X_i} c ≤ prod_{X_i} p/(1-p) ≤ (prod_S p) p_*^(n-a-1) (1-p_*)^(-n) =
(prod_S p) rho(a_i, d_i, p_*) ≤ prod_S p` since `n = a_i d_i + 1`, and `prod_S p ≤ q_i` by the
block-local marginal bound (`p` restricted to the block, as only the block's coordinates enter
`q_i`).
Does not establish: the informal display (T5) or its review and acceptance; anything when
`rho > 1` (the source: "rho>1 is not a counterexample and must not be labeled infeasible"); the
alignment of `localFailure`, `phi`, `rhoTest` with the source's objects
(PENDING_INDEPENDENT_REVIEW); the disposition HOLD_WITH_DOMAIN of `p15-price-budget-restricted`,
which is transcribed in the register, not derived here. -/
theorem rhoTestSufficient {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.RhoTestSufficient F := by
  intro hF p c pBound hB0 hB1 hc i hp hrho
  have hn : (block F i).card = F.a i * F.d i + 1 := hF.2.2.1 i
  have hcard : F.a i + 1 ≤ (block F i).card := by
    rw [hn]
    have := Nat.le_mul_of_pos_right (F.a i) (hF.2.1 i)
    omega
  obtain ⟨S, hS, hScard⟩ := Finset.exists_subset_card_eq hcard
  have h1 := Aux.genPrice_le_prod_mul_ratio F hc hB0 hB1 hp hS
  have hprodS0 : 0 ≤ ∏ v ∈ S, p v := Finset.prod_nonneg fun v hv => (hp v (hS hv)).1
  have hp1 : ∀ v ∈ block F i, 0 ≤ p v ∧ p v ≤ 1 :=
    fun v hv => ⟨(hp v hv).1, (hp v hv).2.trans hB1.le⟩
  have hprodS : ∏ v ∈ S, p v ≤ localFailure F p i :=
    Aux.prod_le_localFailure_of_card_local F hp1 hS (by rw [hScard])
  have hrho' : pBound ^ ((block F i).card - S.card) * ((1 - pBound)⁻¹) ^ (block F i).card =
      rhoTest (F.a i) (F.d i) pBound := by
    rw [hScard, hn, rhoTest, Nat.add_sub_add_right, div_eq_mul_inv, inv_pow]
  calc genPrice c (block F i)
      ≤ (∏ v ∈ S, p v) *
          (pBound ^ ((block F i).card - S.card) * ((1 - pBound)⁻¹) ^ (block F i).card) := h1
    _ = (∏ v ∈ S, p v) * rhoTest (F.a i) (F.d i) pBound := by rw [hrho']
    _ ≤ (∏ v ∈ S, p v) * 1 := mul_le_mul_of_nonneg_left hrho hprodS0
    _ = ∏ v ∈ S, p v := mul_one _
    _ ≤ localFailure F p i := hprodS

/-- Proves: `UniversalLaw.Spec.P15.RestrictedTransformedPriceBudget` (claim
`p15-price-budget-restricted`, display (T1)).
Method: `covercost ≤ 1` by the cover `{∅}`; for `K ≥ K_H(d)` the full blocks cover
(`fullBlockCoverIff`), so `covercost ≤ sum_i price(X_i) ≤ (16/27) sum_i q_i` (display (T4), the
local budget of this module) and, when `mu_p(D) > 0`, `sum_i q_i ≤ -log mu_p(D)` (display (T6),
`P15Hazard.independentBlockHazard`); the capped hazard handles `mu_p(D) = 0` by its convention.
Does not establish: the informal display (T1) or its review and acceptance; any retraction of the
demand-one refutation; the unrestricted prize; new classical coloring theory; attainment of the
infimum; the alignment of `coverCost`, `mu`, `phi` with the source's objects
(PENDING_INDEPENDENT_REVIEW); the disposition HOLD_WITH_DOMAIN of `p15-price-budget-restricted`,
which is transcribed in the register, not derived here. -/
theorem restrictedTransformedPriceBudget {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) : UniversalLaw.Spec.P15.RestrictedTransformedPriceBudget F := by
  intro hF hd p c hp hc K hK
  have hc0 : ∀ v, 0 ≤ c v := fun v => (hc v).1
  refine Aux.le_cappedHazard_of (Aux.coverCost_le_one F K hc0) fun hmu => ?_
  exact (Aux.coverCost_le_sum_block hF hc0 hK).trans (Aux.sum_genPrice_le_budget hF hd hp hc hmu)

/-- Proves: `UniversalLaw.Spec.P15.Benchmark816TransformedRange` (claim
`p15-price-budget-restricted`, Section 4 of price_budget PROOF.md).
Method: `1 - 1/40900 = (40900/40899)⁻¹`, `Real.log_inv` and `log(40900/40899) ≤ 1/40899 ≤ 1`
for the value of `phi`; the cover from `fullBlockCoverIff` and `K_H(408, ..., 408) = 816`
(`P15Structure.benchmark816Palette`); the price bound from six block prices `≤ (1/40899)^409 ≤
1/40899` and from the summed local budget (T4) at `p_v = 1/40900 ≤ 1/4`, `d_i = 408 ≥ 2`, with
`mu_p(D) > 0` because the empty set is good with positive weight; the `covercost` bound is
`RestrictedTransformedPriceBudget` at these data.
Does not establish: the informal Section 4 or its review and acceptance; optimality of the
816-label cover; the whole-ground value `818` (a separate `Prop`); the alignment of the formal
objects with the source's (PENDING_INDEPENDENT_REVIEW); the disposition HOLD_WITH_DOMAIN of
`p15-price-budget-restricted`, which is transcribed in the register, not derived here. -/
theorem benchmark816TransformedRange : UniversalLaw.Spec.P15.Benchmark816TransformedRange := by
  have hF := P15Structure.Aux.benchmark816_isRealized
  have hK : paletteNumber benchmark816.b benchmark816.H benchmark816.d ≤ 816 :=
    le_of_eq P15Structure.benchmark816Palette
  have hcover : IsGeneratorCover benchmark816 816 (fullBlockFamily benchmark816) :=
    (P15Structure.fullBlockCoverIff benchmark816 hF 816).2 hK
  refine ⟨Aux.phi_benchmark, hcover, fun c hc => ?_⟩
  have hp : ∀ v : Fin 6 × Fin 409,
      0 ≤ (fun _ => (1 : ℝ) / 40900) v ∧ (fun _ => (1 : ℝ) / 40900) v ≤ 1 / 4 :=
    fun _ => ⟨by norm_num, by norm_num⟩
  have hc' : ∀ v, 0 ≤ c v ∧ c v ≤ phi ((fun _ => (1 : ℝ) / 40900) v) := fun v => by
    refine ⟨(hc v).1, ?_⟩
    show c v ≤ phi (1 / 40900)
    rw [Aux.phi_benchmark]
    exact (hc v).2
  have hd : ∀ i, 2 ≤ benchmark816.d i := fun _ => by
    show (2 : ℕ) ≤ 408
    norm_num
  have hmu : 0 < mu benchmark816 (fun _ => (1 : ℝ) / 40900) :=
    Aux.mu_pos hF fun _ => ⟨by norm_num, by norm_num⟩
  refine ⟨Aux.le_cappedHazard_of (Aux.famPrice_benchmark_le_one hc) fun _ => ?_,
    restrictedTransformedPriceBudget benchmark816 hF hd _ c hp hc' 816 hK⟩
  rw [P15Structure.Aux.famPrice_fullBlockFamily hF c]
  exact Aux.sum_genPrice_le_budget hF hd hp hc' hmu

end UniversalLaw.Proofs.P15Budget
