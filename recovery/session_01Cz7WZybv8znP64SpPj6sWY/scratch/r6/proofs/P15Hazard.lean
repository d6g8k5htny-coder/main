import Specifications.P15

/-!
# P15 hazard proofs: the local price chain (P4), the independent-block hazard (P5) and the
global assembly (F13)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem outside the `Aux` namespace has as its type exactly one registered `Prop` of
`Specifications/P15.lean` (statement packet `specifications/lean_statements_20261010`, which is
not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop` only:
it does not make the informal Layer 0 note reviewed, accepted or "verified", it does not
identify the formal definitions with the informal objects (that alignment is
PENDING_INDEPENDENT_REVIEW), and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.P15Hazard.Aux`; the gate inventories only the
theorems outside that namespace. The heart of the module is the factorisation
`Aux.mu_le_prod_localGood`: the product weight of a subset `U` of the ground set factors over
the blocks as `prod_i w_i(U ∩ X_i)`, a good set has good block restrictions, and the sum over
`2^X` of a product of block-local quantities is the product of the block-local sums (the
bijection `U ↦ (i ↦ U ∩ X_i)` onto `prod_i 2^(X_i)`). No independence of crossing events is used
or asserted. The block-local marginal identity `Aux.sum_blockWeight_filter_supset` and its
consequence `Aux.prod_le_localFailure_of_card` are exported for the price-budget module.
-/

namespace UniversalLaw.Proofs.P15Hazard

open UniversalLaw.Spec.P15
open Finset

namespace Aux

/-! ### The block-local weight -/

/-- The block-restricted weight `w_i(S) = prod_{v in S} p_v prod_{v in X_i \ S} (1 - p_v)`; the
integrand of `localGood`. -/
noncomputable def blockWeight {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (p : X → ℝ) (i : Fin F.b) (S : Finset X) : ℝ :=
  (∏ v ∈ S, p v) * ∏ v ∈ block F i \ S, (1 - p v)

/-- `localGood` is the sum of the indicator of the downset times the block weight. -/
theorem localGood_eq {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (p : X → ℝ)
    (i : Fin F.b) :
    localGood F p i =
      ∑ S ∈ (block F i).powerset, Set.indicator (downset F) (blockWeight F p i) S := rfl

/-- Block weights are nonnegative for `p in [0,1]^X`. -/
theorem blockWeight_nonneg {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (i : Fin F.b) (S : Finset X) :
    0 ≤ blockWeight F p i S := by
  unfold blockWeight
  refine mul_nonneg (Finset.prod_nonneg fun v _ => (hp v).1) (Finset.prod_nonneg fun v _ => ?_)
  linarith [(hp v).2]

/-- The block weights of all subsets of a block sum to one (`Finset.prod_add`). -/
theorem sum_blockWeight_eq_one {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (p : X → ℝ) (i : Fin F.b) :
    ∑ S ∈ (block F i).powerset, blockWeight F p i S = 1 := by
  have h := Finset.prod_add p (fun v => 1 - p v) (block F i)
  have hl : ∏ v ∈ block F i, (p v + (1 - p v)) = 1 := by simp
  rw [hl] at h
  unfold blockWeight
  exact h.symm

/-- The marginal identity: for `S ⊆ X_i`, the block weights of the supersets of `S` inside the
block sum to `prod_{v in S} p_v` (bijection `T ↦ T \ S` onto `2^(X_i \ S)`). -/
theorem sum_blockWeight_filter_supset {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) (p : X → ℝ) (i : Fin F.b) {S : Finset X} (hS : S ⊆ block F i) :
    ∑ T ∈ (block F i).powerset with S ⊆ T, blockWeight F p i T = ∏ v ∈ S, p v := by
  have h1 : ∑ T ∈ (block F i).powerset with S ⊆ T, blockWeight F p i T =
      ∑ R ∈ (block F i \ S).powerset,
        (∏ v ∈ S, p v) * ((∏ v ∈ R, p v) * ∏ v ∈ (block F i \ S) \ R, (1 - p v)) := by
    refine Finset.sum_nbij' (fun T => T \ S) (fun R => R ∪ S) ?_ ?_ ?_ ?_ ?_
    · intro T hT
      rw [mem_filter, mem_powerset] at hT
      exact mem_powerset.2 (sdiff_subset_sdiff hT.1 (subset_refl _))
    · intro R hR
      rw [mem_powerset] at hR
      rw [mem_filter, mem_powerset]
      exact ⟨union_subset (hR.trans sdiff_subset) hS, subset_union_right⟩
    · intro T hT
      rw [mem_filter] at hT
      exact sdiff_union_of_subset hT.2
    · intro R hR
      rw [mem_powerset] at hR
      exact union_sdiff_cancel_right (sdiff_disjoint.mono_left hR)
    · intro T hT
      rw [mem_filter, mem_powerset] at hT
      unfold blockWeight
      rw [← prod_sdiff hT.2]
      have hset : (block F i \ S) \ (T \ S) = block F i \ T := by
        ext v
        simp only [mem_sdiff, not_and, not_not]
        constructor
        · rintro ⟨⟨hb, hs⟩, h⟩
          exact ⟨hb, fun ht => hs (h ht)⟩
        · rintro ⟨hb, ht⟩
          exact ⟨⟨hb, fun hs => ht (hT.2 hs)⟩, fun ht' => (ht ht').elim⟩
      rw [hset]
      ring
  have h2 := Finset.prod_add p (fun v => 1 - p v) (block F i \ S)
  have hl : ∏ v ∈ block F i \ S, (p v + (1 - p v)) = 1 := by simp
  rw [hl] at h2
  rw [h1, ← Finset.mul_sum, ← h2, mul_one]

/-! ### Bounds on the local good and failure probabilities -/

/-- `0 ≤ mu_p(D_i)`. -/
theorem localGood_nonneg {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (i : Fin F.b) : 0 ≤ localGood F p i := by
  rw [localGood_eq]
  exact Finset.sum_nonneg fun S _ =>
    Set.indicator_nonneg (fun T _ => blockWeight_nonneg F hp i T) S

/-- `mu_p(D_i) ≤ 1`. -/
theorem localGood_le_one {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (i : Fin F.b) : localGood F p i ≤ 1 := by
  rw [localGood_eq, ← sum_blockWeight_eq_one F p i]
  exact Finset.sum_le_sum fun S _ =>
    Set.indicator_apply_le' (fun _ => le_rfl) (fun _ => blockWeight_nonneg F hp i S)

/-- `0 ≤ q_i`. -/
theorem localFailure_nonneg {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (i : Fin F.b) : 0 ≤ localFailure F p i := by
  unfold localFailure
  linarith [localGood_le_one F hp i]

/-- `q_i ≤ 1`. -/
theorem localFailure_le_one {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (i : Fin F.b) : localFailure F p i ≤ 1 := by
  unfold localFailure
  linarith [localGood_nonneg F hp i]

/-- `1 - q_i = mu_p(D_i)`. -/
theorem one_sub_localFailure {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (p : X → ℝ) (i : Fin F.b) : 1 - localFailure F p i = localGood F p i := by
  unfold localFailure
  ring

/-- Every subset of the block containing `a_i + 1` or more coordinates is bad, so
`mu_p(D_i) ≤ 1 - prod_{v in S} p_v` for every such `S ⊆ X_i` (marginal identity plus the
termwise bound `indicator ≤ weight`). -/
theorem localGood_le_one_sub_prod {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) {i : Fin F.b} {S : Finset X}
    (hS : S ⊆ block F i) (hcard : F.a i + 1 ≤ S.card) :
    localGood F p i ≤ 1 - ∏ v ∈ S, p v := by
  rw [localGood_eq, ← Finset.sum_filter_add_sum_filter_not ((block F i).powerset) (fun T => S ⊆ T)]
  have h0 : ∑ T ∈ (block F i).powerset with S ⊆ T,
      Set.indicator (downset F) (blockWeight F p i) T = 0 := by
    refine Finset.sum_eq_zero fun T hT => ?_
    rw [mem_filter, mem_powerset] at hT
    refine Set.indicator_of_notMem (fun hgood => ?_) _
    have h1 : (T ∩ block F i).card ≤ F.a i := hgood.1 i
    rw [Finset.inter_eq_left.2 hT.1] at h1
    have h2 := card_le_card hT.2
    omega
  have h3 : ∑ T ∈ (block F i).powerset with ¬ S ⊆ T,
      Set.indicator (downset F) (blockWeight F p i) T ≤
      ∑ T ∈ (block F i).powerset with ¬ S ⊆ T, blockWeight F p i T :=
    Finset.sum_le_sum fun T _ =>
      Set.indicator_apply_le' (fun _ => le_rfl) (fun _ => blockWeight_nonneg F hp i T)
  have h4 := Finset.sum_filter_add_sum_filter_not ((block F i).powerset) (fun T => S ⊆ T)
    (blockWeight F p i)
  rw [sum_blockWeight_eq_one F p i, sum_blockWeight_filter_supset F p i hS] at h4
  linarith

/-- The block-local price bound of the (T3)/(T5) arguments: for `S ⊆ X_i` with
`|S| ≥ a_i + 1`, `prod_{v in S} p_v ≤ q_i`. -/
theorem prod_le_localFailure_of_card {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) {i : Fin F.b}
    {S : Finset X} (hS : S ⊆ block F i) (hcard : F.a i + 1 ≤ S.card) :
    ∏ v ∈ S, p v ≤ localFailure F p i := by
  unfold localFailure
  linarith [localGood_le_one_sub_prod F hp hS hcard]

/-- The full block has `a_i d_i + 1 ≥ a_i + 1` coordinates, so `prod_{v in X_i} p_v ≤ q_i`. -/
theorem prod_block_le_localFailure {X : Type} [Fintype X] [DecidableEq X]
    {F : RealizedData X} (hF : IsRealizedFamily F) {p : X → ℝ}
    (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (i : Fin F.b) :
    ∏ v ∈ block F i, p v ≤ localFailure F p i := by
  refine prod_le_localFailure_of_card F hp (subset_refl _) ?_
  rw [hF.2.2.1 i]
  have := Nat.le_mul_of_pos_right (F.a i) (hF.2.1 i)
  omega

/-! ### The transformed price `phi` -/

/-- `q ≤ -log(1 - q)` for `q < 1` (from `1 - q ≤ exp(-q)`). -/
theorem neg_log_one_sub_ge {q : ℝ} (hq : q < 1) : q ≤ -Real.log (1 - q) := by
  have h1 : 1 - q ≤ Real.exp (-q) := by
    have := Real.add_one_le_exp (-q)
    linarith
  have h2 := Real.log_le_log (by linarith : (0 : ℝ) < 1 - q) h1
  rw [Real.log_exp] at h2
  linarith

/-- `q ≤ phi(q)` for `q ≤ 1` (in particular on `[0, 1]`). -/
theorem le_phi_self {q : ℝ} (h1 : q ≤ 1) : q ≤ phi q := by
  unfold phi
  split_ifs with h
  · exact le_min h1 (neg_log_one_sub_ge h)
  · exact h1

/-! ### Blocks partition the ground set -/

/-- The traces `T ∩ X_i` of a set on the blocks are pairwise disjoint. -/
theorem pairwiseDisjoint_inter_block {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) (T : Finset X) :
    (↑(univ : Finset (Fin F.b)) : Set (Fin F.b)).PairwiseDisjoint
      (fun i => T ∩ block F i) := by
  intro i _ j _ hij
  show Disjoint (T ∩ block F i) (T ∩ block F j)
  rw [Finset.disjoint_left]
  intro v hvi hvj
  simp only [block, mem_inter, mem_filter, mem_univ, true_and] at hvi hvj
  exact hij (hvi.2.symm.trans hvj.2)

/-- The traces of a set on the blocks reassemble it. -/
theorem biUnion_inter_block {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (T : Finset X) : (univ : Finset (Fin F.b)).biUnion (fun i => T ∩ block F i) = T := by
  ext v
  simp [block]

/-- A product over a set factors over its traces on the blocks. -/
theorem prod_eq_prod_blocks {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (f : X → ℝ) (T : Finset X) : ∏ v ∈ T, f v = ∏ i, ∏ v ∈ T ∩ block F i, f v := by
  calc ∏ v ∈ T, f v = ∏ v ∈ (univ : Finset (Fin F.b)).biUnion (fun i => T ∩ block F i), f v := by
        rw [biUnion_inter_block]
    _ = ∏ i, ∏ v ∈ T ∩ block F i, f v := Finset.prod_biUnion (pairwiseDisjoint_inter_block F T)

/-- The complement's trace on a block is the block minus the trace. -/
theorem compl_inter_block {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (U : Finset X) (i : Fin F.b) : (univ \ U) ∩ block F i = block F i \ (U ∩ block F i) := by
  ext v
  simp only [mem_inter, mem_sdiff, mem_univ, true_and, not_and]
  tauto

/-- The weight of a subset factors over the blocks: `w(U) = prod_i w_i(U ∩ X_i)`. -/
theorem weight_eq_prod_blockWeight {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (p : X → ℝ) (U : Finset X) : weight p U = ∏ i, blockWeight F p i (U ∩ block F i) := by
  unfold weight blockWeight
  rw [show ∏ v ∈ U, p v = ∏ i, ∏ v ∈ U ∩ block F i, p v from prod_eq_prod_blocks F p U,
    show ∏ v ∈ univ \ U, (1 - p v) = ∏ i, ∏ v ∈ (univ \ U) ∩ block F i, (1 - p v) from
      prod_eq_prod_blocks F (fun v => 1 - p v) (univ \ U),
    ← Finset.prod_mul_distrib]
  refine Finset.prod_congr rfl fun i _ => ?_
  rw [compl_inter_block]

/-! ### Good sets have good block restrictions -/

/-- `U in D` implies `U ∩ X_i in D`. -/
theorem isGood_inter_block {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}
    {U : Finset X} (hU : IsGood F U) (i : Fin F.b) : IsGood F (U ∩ block F i) := by
  refine ⟨fun j => ?_, fun e he hsub => hU.2 e he (hsub.trans ?_)⟩
  · exact (card_le_card (inter_subset_inter inter_subset_left (subset_refl _))).trans (hU.1 j)
  · exact image_subset_image inter_subset_left

/-- Termwise: `1_D(U) w(U) ≤ prod_i 1_D(U ∩ X_i) w_i(U ∩ X_i)`. -/
theorem indicator_weight_le {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (U : Finset X) :
    Set.indicator (downset F) (weight p) U ≤
      ∏ i, Set.indicator (downset F) (blockWeight F p i) (U ∩ block F i) := by
  by_cases hU : U ∈ downset F
  · rw [Set.indicator_of_mem hU, weight_eq_prod_blockWeight F p U]
    refine le_of_eq (Finset.prod_congr rfl fun i _ => ?_)
    rw [Set.indicator_of_mem (show U ∩ block F i ∈ downset F from isGood_inter_block hU i)]
  · rw [Set.indicator_of_notMem hU]
    exact Finset.prod_nonneg fun i _ =>
      Set.indicator_nonneg (fun T _ => blockWeight_nonneg F hp i T) _

/-! ### The factorisation of a block-local product over `2^X` -/

/-- `sum_{U ⊆ X} prod_i g_i(U ∩ X_i) = prod_i sum_{S ⊆ X_i} g_i(S)`: the bijection
`U ↦ (i ↦ U ∩ X_i)` from `2^X` onto `prod_i 2^(X_i)` with inverse `x ↦ ⋃_i x_i`. -/
theorem sum_prod_blocks {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (g : Fin F.b → Finset X → ℝ) :
    ∑ U ∈ (univ : Finset X).powerset, ∏ i, g i (U ∩ block F i) =
      ∏ i, ∑ S ∈ (block F i).powerset, g i S := by
  rw [Finset.prod_univ_sum]
  refine Finset.sum_nbij' (fun U => fun i => U ∩ block F i) (fun x => univ.biUnion x) ?_ ?_ ?_ ?_
    ?_
  · intro U _
    exact Fintype.mem_piFinset.2 fun i => mem_powerset.2 inter_subset_right
  · intro x _
    exact mem_powerset.2 (subset_univ _)
  · intro U _
    exact biUnion_inter_block F U
  · intro x hx
    have hx' : ∀ j, x j ⊆ block F j := fun j => mem_powerset.1 (Fintype.mem_piFinset.1 hx j)
    funext i
    ext v
    simp only [mem_inter, mem_biUnion, mem_univ, true_and]
    constructor
    · rintro ⟨⟨j, hj⟩, hi⟩
      have h1 : F.blk v = j := by simpa [block] using hx' j hj
      have h2 : F.blk v = i := by simpa [block] using hi
      rw [← h2, h1]
      exact hj
    · intro hv
      exact ⟨⟨i, hv⟩, hx' i hv⟩
  · intro U _
    rfl

/-- The factorisation `mu_p(D) ≤ prod_i mu_p(D_i)` for `p in [0,1]^X`: termwise domination of
`1_D(U) w(U)` by the product of block restrictions, then the block-product bijection. -/
theorem mu_le_prod_localGood {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) : mu F p ≤ ∏ i, localGood F p i := by
  unfold mu
  calc ∑ U ∈ (univ : Finset X).powerset, Set.indicator (downset F) (weight p) U
      ≤ ∑ U ∈ (univ : Finset X).powerset,
          ∏ i, Set.indicator (downset F) (blockWeight F p i) (U ∩ block F i) :=
        Finset.sum_le_sum fun U _ => indicator_weight_le F hp U
    _ = ∏ i, ∑ S ∈ (block F i).powerset, Set.indicator (downset F) (blockWeight F p i) S :=
        sum_prod_blocks F (fun i S => Set.indicator (downset F) (blockWeight F p i) S)
    _ = ∏ i, localGood F p i := by
        refine Finset.prod_congr rfl fun i _ => ?_
        rw [localGood_eq]

/-- When `mu_p(D) > 0`, every local good probability is positive (a zero factor would zero the
product that dominates `mu_p(D)`). -/
theorem localGood_pos_of_mu_pos {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (hmu : 0 < mu F p) (i : Fin F.b) :
    0 < localGood F p i := by
  by_contra h
  have h0 : localGood F p i = 0 := le_antisymm (not_lt.mp h) (localGood_nonneg F hp i)
  have := mu_le_prod_localGood F hp
  rw [Finset.prod_eq_zero (mem_univ i) h0] at this
  linarith

/-- `-log prod_i mu_p(D_i) = sum_i -log mu_p(D_i)` when `mu_p(D) > 0`. -/
theorem neg_log_prod_localGood {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (hmu : 0 < mu F p) :
    -Real.log (∏ i, localGood F p i) = ∑ i, -Real.log (localGood F p i) := by
  rw [Real.log_prod (fun i _ => (localGood_pos_of_mu_pos F hp hmu i).ne'), Finset.sum_neg_distrib]

/-- `-log prod_i mu_p(D_i) ≤ -log mu_p(D)` when `mu_p(D) > 0`. -/
theorem neg_log_prod_localGood_le {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    {p : X → ℝ} (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (hmu : 0 < mu F p) :
    -Real.log (∏ i, localGood F p i) ≤ -Real.log (mu F p) :=
  neg_le_neg (Real.log_le_log hmu (mu_le_prod_localGood F hp))

end Aux

/-! ## Inventory theorems -/

/-- Proves: `UniversalLaw.Spec.P15.LocalPriceChain` (claim `p15-realized-covers`, display (P4)).
Method: `Finset.prod_le_prod₀` for `0 ≤ c ≤ p`; the block-local marginal bound
`Aux.prod_le_localFailure_of_card` at `S = X_i` (the full block has `a_i d_i + 1 ≥ a_i + 1`
coordinates and is therefore bad); `q ≤ phi(q)` on `[0,1]` from `1 - q ≤ exp(-q)`.
Does not establish: the informal display (P4) or its review and acceptance; the alignment of
`localFailure` with the source's `q_i` (PENDING_INDEPENDENT_REVIEW); the identification
`prod p_v = P(U_i = X_i)` as a probabilistic statement; the disposition HOLD_WITH_DOMAIN of
`p15-realized-covers`, which is transcribed in the register, not derived here. -/
theorem localPriceChain {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.LocalPriceChain F := by
  intro hF p c hp hc i
  refine ⟨?_, Aux.prod_block_le_localFailure hF hp i,
    Aux.le_phi_self (Aux.localFailure_le_one F hp i)⟩
  unfold genPrice
  exact Finset.prod_le_prod₀ (fun v _ => (hc v).1) (fun v _ => (hc v).2)

/-- Proves: `UniversalLaw.Spec.P15.IndependentBlockHazard` (claim `p15-realized-covers`, display
(P5)).
Method: `1 - q_i = mu_p(D_i)`; the factorisation `Aux.mu_le_prod_localGood` (block-product
bijection on `2^X`, no independence assumption); when `mu_p(D) > 0` every `mu_p(D_i) > 0`, so
`Real.log_prod` turns `-log prod` into `sum -log`, each `q_i ≤ -log(1 - q_i)` by
`Aux.neg_log_one_sub_ge`, and `-log prod ≤ -log mu` by monotonicity of `log`.
Does not establish: the informal display (P5) or its review and acceptance; any independence of
crossing events (none is used); the price bound `sum_i price(G_i) ≤ sum_i q_i` (that is (P4)
summed, not part of this `Prop`); the alignment of the formal `mu`, `localFailure` with the
source's objects (PENDING_INDEPENDENT_REVIEW); the disposition HOLD_WITH_DOMAIN of
`p15-realized-covers`. -/
theorem independentBlockHazard {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.IndependentBlockHazard F := by
  intro _ p hp
  have hprod : ∏ i, (1 - localFailure F p i) = ∏ i, localGood F p i :=
    Finset.prod_congr rfl fun i _ => Aux.one_sub_localFailure F p i
  rw [hprod]
  refine ⟨Aux.mu_le_prod_localGood F hp, fun hmu => ⟨?_, Aux.neg_log_prod_localGood_le F hp hmu⟩⟩
  rw [Aux.neg_log_prod_localGood F hp hmu]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [← Aux.one_sub_localFailure F p i]
  refine Aux.neg_log_one_sub_ge ?_
  have := Aux.localGood_pos_of_mu_pos F hp hmu i
  unfold localFailure
  linarith

/-- Proves: `UniversalLaw.Spec.P15.GlobalAssemblyHazard` (claim `p15-full-price`, display (F13)).
Method: the factorisation `Aux.mu_le_prod_localGood` (every good set has good block
restrictions; the weight factors over the blocks; the block-product bijection on `2^X`), then,
when `mu_p(D) > 0`, every `mu_p(D_i) > 0`, `Real.log_prod` and the monotonicity of `log`.
Does not establish: the informal display (F13) or its review and acceptance; the local bound
(F12) or the full-price theorem (F2); any independence of crossing failures (none is used); the
alignment of the formal `mu`, `localGood` with the source's `mu_p(D)`, `mu_p(D_i)`
(PENDING_INDEPENDENT_REVIEW); the disposition of `p15-full-price`, which is transcribed in the
register, not derived here. -/
theorem globalAssemblyHazard {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.GlobalAssemblyHazard F := by
  intro _ p hp
  refine ⟨Aux.mu_le_prod_localGood F hp, fun hmu => ?_⟩
  rw [← Aux.neg_log_prod_localGood F hp hmu]
  exact Aux.neg_log_prod_localGood_le F hp hmu

end UniversalLaw.Proofs.P15Hazard
