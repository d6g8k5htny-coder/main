import Specifications.P15
import SpecificationProofs.P15Structure
import SpecificationProofs.P15Elementary

/-!
# P15 realized-cover proofs: the six-block cover, the local restriction, attainment of the
palette number and the union bound (P11)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem in this module has as its type exactly one registered `Prop` of
`Specifications/P15.lean` (statement packet `specifications/lean_statements_20261010`, which is
not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop` only:
it does not make the informal Layer 0 note reviewed, accepted or "verified", it does not
identify the formal definitions with the informal objects (that alignment is
PENDING_INDEPENDENT_REVIEW), and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.P15Covers.Aux`; the gate inventories only the
theorems outside that namespace. The general lemmas of `Aux` (the marginal identity
`sum_weight_filter_superset`, the union bound `one_sub_mu_le_sum_prod`, the existence of a
minimal forbidden subset of a bad set and the cardinality of a transversal) are stated for an
arbitrary finite ground type so that other modules may reuse them.
-/

namespace UniversalLaw.Proofs.P15Covers

open UniversalLaw.Spec.P15
open Finset

namespace Aux

/-! ### The product measure: nonnegative weights and the marginal of a fixed subset -/

/-- Weights are nonnegative for `p` in `[0,1]^Y`. -/
theorem weight_nonneg {Y : Type} [Fintype Y] [DecidableEq Y] {p : Y → ℝ}
    (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (U : Finset Y) : 0 ≤ weight p U := by
  unfold weight
  exact mul_nonneg (prod_nonneg fun v _ => (hp v).1)
    (prod_nonneg fun v _ => by linarith [(hp v).2])

/-- The sets containing `M` are exactly the sets `M ∪ T` with `T ⊆ univ \ M`. -/
theorem filter_superset_eq_image {Y : Type} [Fintype Y] [DecidableEq Y] (M : Finset Y) :
    (univ : Finset Y).powerset.filter (fun U => M ⊆ U) =
      ((univ : Finset Y) \ M).powerset.image (fun T => M ∪ T) := by
  ext U
  simp only [mem_filter, mem_powerset, subset_univ, true_and, mem_image]
  constructor
  · intro hMU
    exact ⟨U \ M, sdiff_subset_sdiff (subset_univ U) subset_rfl, union_sdiff_of_subset hMU⟩
  · rintro ⟨T, -, rfl⟩
    exact subset_union_left

/-- The weight of `M ∪ T` for `T` disjoint from `M` factors as the product of `p` over `M` times
the weight of `T` relative to the ground set `univ \ M`. -/
theorem weight_union_of_disjoint {Y : Type} [Fintype Y] [DecidableEq Y] (p : Y → ℝ)
    {M T : Finset Y} (h : Disjoint M T) :
    weight p (M ∪ T) =
      (∏ v ∈ M, p v) * ((∏ v ∈ T, p v) * ∏ v ∈ ((univ : Finset Y) \ M) \ T, (1 - p v)) := by
  have hs : (univ : Finset Y) \ (M ∪ T) = ((univ : Finset Y) \ M) \ T := by
    ext v
    simp only [mem_sdiff, mem_univ, true_and, mem_union, not_or]
  unfold weight
  rw [prod_union h, mul_assoc, hs]

/-- The marginal identity: the total weight of the sets containing a fixed `M` is
`∏_{v ∈ M} p v`, for any real vector `p` (no range hypothesis is needed). -/
theorem sum_weight_filter_superset {Y : Type} [Fintype Y] [DecidableEq Y] (p : Y → ℝ)
    (M : Finset Y) :
    ∑ U ∈ (univ : Finset Y).powerset.filter (fun U => M ⊆ U), weight p U = ∏ v ∈ M, p v := by
  rw [filter_superset_eq_image, sum_image]
  · have h : ∀ T ∈ ((univ : Finset Y) \ M).powerset, weight p (M ∪ T) =
        (∏ v ∈ M, p v) * ((∏ v ∈ T, p v) * ∏ v ∈ ((univ : Finset Y) \ M) \ T, (1 - p v)) := by
      intro T hT
      rw [mem_powerset] at hT
      exact weight_union_of_disjoint p (disjoint_of_subset_right hT disjoint_sdiff)
    rw [sum_congr rfl h, ← mul_sum, ← prod_add p (fun v => 1 - p v)]
    have h1 : ∏ v ∈ (univ : Finset Y) \ M, (p v + (1 - p v)) = 1 :=
      prod_eq_one fun v _ => by ring
    rw [h1, mul_one]
  · intro T₁ hT₁ T₂ hT₂ h
    rw [mem_coe, mem_powerset] at hT₁ hT₂
    have h' : M ∪ T₁ = M ∪ T₂ := h
    have e₁ := union_sdiff_cancel_left (disjoint_of_subset_right hT₁ disjoint_sdiff (s := M))
    have e₂ := union_sdiff_cancel_left (disjoint_of_subset_right hT₂ disjoint_sdiff (s := M))
    rw [← e₁, ← e₂, h']

/-- The same marginal identity with the superset condition written as an `if`. -/
theorem sum_weight_ite_superset {Y : Type} [Fintype Y] [DecidableEq Y] (p : Y → ℝ)
    (M : Finset Y) :
    ∑ U ∈ (univ : Finset Y).powerset, (if M ⊆ U then weight p U else 0) = ∏ v ∈ M, p v := by
  rw [← sum_filter, sum_weight_filter_superset]

/-! ### The bad probability and the union bound over a covering family -/

open Classical in
/-- `1 - mu_p(D)` is the total weight of the bad sets (classical decidability of `IsGood`). -/
theorem one_sub_mu_eq {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) (p : X → ℝ) :
    1 - mu F p =
      ∑ U ∈ (univ : Finset X).powerset, (if IsGood F U then 0 else weight p U) := by
  classical
  have hmu : mu F p = ∑ U ∈ (univ : Finset X).powerset, (if IsGood F U then weight p U else 0) := by
    unfold mu
    refine sum_congr rfl fun U _ => ?_
    rw [Set.indicator_apply]
    exact if_congr Iff.rfl rfl rfl
  rw [hmu, ← UniversalLaw.Proofs.P15Elementary.Aux.sum_weight_eq_one p, ← sum_sub_distrib]
  refine sum_congr rfl fun U _ => ?_
  split_ifs <;> ring

/-- The union bound: if every bad set contains a member of the family `𝓜`, then
`1 - mu_p(D) ≤ ∑_{M ∈ 𝓜} ∏_{v ∈ M} p v`, for `p` in `[0,1]^X`. -/
theorem one_sub_mu_le_sum_prod {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X)
    (p : X → ℝ) (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (𝓜 : Finset (Finset X))
    (h𝓜 : ∀ U : Finset X, ¬ IsGood F U → ∃ M ∈ 𝓜, M ⊆ U) :
    1 - mu F p ≤ ∑ M ∈ 𝓜, ∏ v ∈ M, p v := by
  classical
  have hterm : ∀ U : Finset X, ∀ M : Finset X,
      0 ≤ (if M ⊆ U then weight p U else 0) := fun U M => by
    split_ifs
    · exact weight_nonneg hp U
    · exact le_rfl
  have h2 : ∀ U ∈ (univ : Finset X).powerset, (if IsGood F U then 0 else weight p U) ≤
      ∑ M ∈ 𝓜, (if M ⊆ U then weight p U else 0) := by
    intro U _
    split_ifs with hU
    · exact sum_nonneg fun M _ => hterm U M
    · obtain ⟨M, hM, hMU⟩ := h𝓜 U hU
      calc weight p U = (if M ⊆ U then weight p U else 0) := by simp [hMU]
        _ ≤ ∑ M ∈ 𝓜, (if M ⊆ U then weight p U else 0) :=
          single_le_sum (fun M _ => hterm U M) hM
  rw [one_sub_mu_eq]
  calc ∑ U ∈ (univ : Finset X).powerset, (if IsGood F U then 0 else weight p U)
      ≤ ∑ U ∈ (univ : Finset X).powerset, ∑ M ∈ 𝓜, (if M ⊆ U then weight p U else 0) :=
        sum_le_sum h2
    _ = ∑ M ∈ 𝓜, ∑ U ∈ (univ : Finset X).powerset, (if M ⊆ U then weight p U else 0) :=
        sum_comm
    _ = ∑ M ∈ 𝓜, ∏ v ∈ M, p v := sum_congr rfl fun M _ => sum_weight_ite_superset p M

/-! ### Minimal forbidden subsets of bad sets; the size of a transversal -/

/-- Every bad set contains an inclusion-minimal forbidden set. -/
theorem exists_minimalForbidden_subset {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) (U : Finset X) (hU : ¬ IsGood F U) :
    ∃ M : Finset X, IsMinimalForbidden F M ∧ M ⊆ U := by
  induction U using Finset.strongInduction with
  | H U ih =>
    by_cases h : ∃ v ∈ U, ¬ IsGood F (U.erase v)
    · obtain ⟨v, hv, hbad⟩ := h
      obtain ⟨M, hM, hMU⟩ := ih (U.erase v) (erase_ssubset hv) hbad
      exact ⟨M, hM, hMU.trans (erase_subset v U)⟩
    · exact ⟨U, ⟨hU, fun v hv => of_not_not fun hb => h ⟨v, hv, hb⟩⟩, subset_rfl⟩

/-- A transversal of `e` has exactly `|e|` elements (one per block of `e`). -/
theorem card_of_isTransversal {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}
    {e : Finset (Fin F.b)} {U : Finset X} (h : IsTransversal F e U) : U.card = e.card := by
  obtain ⟨hsupp, hone⟩ := h
  have hU : U = e.biUnion (fun i => U ∩ block F i) := by
    ext v
    simp only [mem_biUnion, mem_inter, UniversalLaw.Proofs.P15Structure.Aux.mem_block_iff]
    constructor
    · intro hv
      refine ⟨F.blk v, ?_, hv, rfl⟩
      rw [← hsupp]
      exact mem_image_of_mem _ hv
    · rintro ⟨i, -, hv, -⟩
      exact hv
  rw [hU, card_biUnion]
  · rw [sum_congr rfl hone]
    simp
  · intro i _ j _ hij
    exact (UniversalLaw.Proofs.P15Structure.Aux.block_disjoint F hij).mono
      inter_subset_right inter_subset_right

/-! ### The six-block benchmark at `p_v = 1/40900` -/

open UniversalLaw.Proofs.P15Structure.Aux in
/-- Every bad set of the six-block benchmark contains an internal pair or a transversal
quadruple. -/
theorem benchmark816_bad_covered (U : Finset (Fin 6 × Fin 409))
    (hU : ¬ IsGood benchmark816 U) : ∃ M ∈ pairs816 ∪ transAll816, M ⊆ U := by
  obtain ⟨M, hM, hMU⟩ := exists_minimalForbidden_subset benchmark816 U hU
  refine ⟨M, ?_, hMU⟩
  rw [mem_union, mem_pairs816, mem_transAll816]
  exact (minForb816_iff M).1 hM

-- The style linter `constructorNameAsVariable` (variable names shadowing constructor names)
-- exhausts the recursion limit on this local context (as in `P15Structure`); it is a naming
-- linter, not a check.
set_option linter.constructorNameAsVariable false in
open UniversalLaw.Proofs.P15Structure.Aux in
/-- Display (P11), first line: the union bound over the `6 * C(409,2)` pairs and the
`15 * 409^4` transversal quadruples at `p_v = 1/40900`. -/
theorem benchmark816_one_sub_mu_le :
    1 - mu benchmark816 (fun _ => (1 : ℝ) / 40900) ≤ 500616 / 40900 ^ 2 + 15 / 100 ^ 4 := by
  have hp : ∀ v : Fin 6 × Fin 409,
      0 ≤ (fun _ => (1 : ℝ) / 40900) v ∧ (fun _ => (1 : ℝ) / 40900) v ≤ 1 :=
    fun _ => by norm_num
  refine (one_sub_mu_le_sum_prod benchmark816 _ hp _ benchmark816_bad_covered).trans ?_
  rw [sum_union disjoint_pairs816_transAll816]
  have hpair : ∀ M ∈ pairs816, ∏ v ∈ M, (fun _ => (1 : ℝ) / 40900) v = (1 / 40900) ^ 2 := by
    intro M hM
    obtain ⟨_, -, hcard⟩ := mem_pairs816.1 hM
    rw [prod_const, hcard]
  have htrans : ∀ M ∈ transAll816,
      ∏ v ∈ M, (fun _ => (1 : ℝ) / 40900) v = (1 / 40900) ^ 4 := by
    intro M hM
    rw [mem_transAll816] at hM
    obtain ⟨e, he, hT⟩ := hM
    rw [prod_const, card_of_isTransversal hT, mem_H_benchmark816 he]
  rw [sum_congr rfl hpair, sum_congr rfl htrans, sum_const, sum_const, card_pairs816,
    card_transAll816, nsmul_eq_mul, nsmul_eq_mul]
  have hc : Nat.choose 409 2 = 83436 := by
    rw [Nat.choose_two_right]
  rw [hc]
  push_cast
  norm_num

end Aux

/-- Proves: `UniversalLaw.Spec.P15.Benchmark816Cover` (claim `p15-realized-covers`, Section 6 of
P15_REALIZED_COVERS.md: "At816 colors it covers the obstruction; at815 it does not").
Method: Proposition P2 (`FullBlockCoverIff`) for the six-block data, which satisfy
`IsRealizedFamily` (`Benchmark816IsRealized`), reduces both clauses to `816 ≤ K`, and
`Benchmark816Palette` evaluates the palette number to `816`.
Does not establish: the informal claim, its review or acceptance; alignment of the formal
`IsGeneratorCover`/`obstruction` with the informal notions; optimality of any other cover
(disposition transcribed in the statement: HOLD_WITH_DOMAIN). -/
theorem benchmark816Cover : UniversalLaw.Spec.P15.Benchmark816Cover := by
  have h := UniversalLaw.Proofs.P15Structure.fullBlockCoverIff benchmark816
    UniversalLaw.Proofs.P15Structure.benchmark816IsRealized
  have hpal : paletteNumber benchmark816.b benchmark816.H benchmark816.d = 816 :=
    UniversalLaw.Proofs.P15Structure.benchmark816Palette
  refine ⟨(h 816).2 (by rw [hpal]), fun hc => ?_⟩
  have h815 := (h 815).1 hc
  rw [hpal] at h815
  omega

/-- Proves: `UniversalLaw.Spec.P15.LocalRestriction` (claim `p15-realized-covers`, display (P2) of
P15_REALIZED_COVERS.md: `D_i = D ∩ 2^(X_i) = {S ⊆ X_i : |S| ≤ a_i}`).
Method: for `S ⊆ X_i`, `S ∩ X_i = S`, so goodness gives `|S| ≤ a_i`; conversely a subset of one
block with at most `a_i` elements is good because its support has at most one index and every
edge has at least two (`P15Structure.Aux.good_of_subset_block`).
Does not establish: the informal claim, its review or acceptance; alignment of `IsGood` with the
informal downset `D`. -/
theorem localRestriction {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.LocalRestriction F := by
  intro hF i S hS
  constructor
  · intro hgood
    have h := hgood.1 i
    rwa [inter_eq_left.2 hS] at h
  · intro hcard
    exact UniversalLaw.Proofs.P15Structure.Aux.good_of_subset_block hF hS hcard

/-- Proves: `UniversalLaw.Spec.P15.PaletteNumberAttained` (claim `p15-realized-covers`, Section 4
of P15_REALIZED_COVERS.md: "By discarding excess labels within blocks, one may take |P_i|=d_i";
the feasible set of palette sizes is nonempty).
Method: disjoint label intervals of lengths `d_i` form a feasible palette of size `∑ d_i` when
every edge has at least two indices (`P15Structure.Aux.isPaletteFeasible_paletteNumber`), so the
infimum over `ℕ` is attained (`csInf_mem` of a nonempty set of naturals).
Does not establish: the informal claim, its review or acceptance; the value of the palette
number for any family. -/
theorem paletteNumberAttained {X : Type} [Fintype X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.PaletteNumberAttained F :=
  fun hF => UniversalLaw.Proofs.P15Structure.Aux.isPaletteFeasible_paletteNumber F.H F.d hF.2.2.2.1

/-- Proves: `UniversalLaw.Spec.P15.Benchmark816UnionBound` (claim `p15-realized-covers`, display
(P11) of P15_REALIZED_COVERS.md: `Q = 1 - mu_p(D) ≤ 500616/40900^2 + 15/100^4 =
2449227/8180000000 < 3/10000` at `p_v = 1/40900`).
Method: `1 - mu_p(D)` is the total weight of the bad sets; every bad set contains a minimal
forbidden set, which by Proposition P1 for the six-block data is an internal pair (`6 * C(409,2)`
of them) or a transversal quadruple (`15 * 409^4`); the union bound exchanges the two sums and the
marginal identity `∑_{U ⊇ M} w(U) = ∏_{v ∈ M} p_v` evaluates each term to `(1/40900)^|M|`; the
two rational identities are exact arithmetic (`norm_num`), and the last clause follows from the
first three by `linarith`.
Does not establish: the informal claim, its review or acceptance; the exact value of the bad
probability (the source stresses "only a union bound, not an equality of the global bad
probability"); alignment of `mu`/`IsGood` with the informal objects. -/
theorem benchmark816UnionBound : UniversalLaw.Spec.P15.Benchmark816UnionBound := by
  have h1 := Aux.benchmark816_one_sub_mu_le
  have h3 : (500616 / 40900 ^ 2 + 15 / 100 ^ 4 : ℝ) < 3 / 10000 := by norm_num
  refine ⟨h1, by norm_num, by norm_num, ?_⟩
  linarith

end UniversalLaw.Proofs.P15Covers
