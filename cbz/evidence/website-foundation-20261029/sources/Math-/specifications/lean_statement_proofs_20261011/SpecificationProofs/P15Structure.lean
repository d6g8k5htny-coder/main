import Specifications.P15

/-!
# P15 finite-structure proofs (benchmark instance and general finite statements)

Scientific effect: NONE. Organizational-independence credit: 0 (Anthropic / Claude).

Every theorem in this module has as its type exactly one registered `Prop` of
`Specifications/P15.lean` (statement packet `specifications/lean_statements_20261010`, which is
not modified here). A kernel-checked proof of such a `Prop` is evidence about that `Prop` only:
it does not make the informal Layer 0 note reviewed, accepted or "verified", it does not
identify the formal definitions with the informal objects (that alignment is
PENDING_INDEPENDENT_REVIEW), and it promotes, closes, discharges or reclassifies nothing.

Helper lemmas live in `UniversalLaw.Proofs.P15Structure.Aux`; the gate inventories only the
theorems outside that namespace.
-/

namespace UniversalLaw.Proofs.P15Structure

open UniversalLaw.Spec.P15
open Finset

namespace Aux

/-! ### Blocks of a realized family -/

/-- The blocks of a realized family are nonempty (they have `a_i d_i + 1 ≥ 1` elements). -/
theorem block_nonempty {X : Type} [Fintype X] {F : RealizedData X} (hF : IsRealizedFamily F)
    (i : Fin F.b) : (block F i).Nonempty := by
  rw [← Finset.card_pos, hF.2.2.1 i]
  exact Nat.succ_pos _

/-- An element of `block F i` has block index `i`. -/
theorem blk_eq_of_mem_block {X : Type} [Fintype X] {F : RealizedData X} {i : Fin F.b} {v : X}
    (hv : v ∈ block F i) : F.blk v = i := by
  simpa [block] using hv

/-- Membership in a block. -/
theorem mem_block_iff {X : Type} [Fintype X] {F : RealizedData X} {i : Fin F.b} {v : X} :
    v ∈ block F i ↔ F.blk v = i := by
  simp [block]

/-- The block map of a realized family is injective (blocks are nonempty fibres). -/
theorem block_injective {X : Type} [Fintype X] {F : RealizedData X}
    (hF : IsRealizedFamily F) : Function.Injective (block F) := by
  intro i j hij
  obtain ⟨v, hv⟩ := block_nonempty hF i
  have hvj : v ∈ block F j := hij ▸ hv
  rw [← blk_eq_of_mem_block hv, blk_eq_of_mem_block hvj]

/-- The price of the full-block family is the sum of the block prices. -/
theorem famPrice_fullBlockFamily {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}
    (hF : IsRealizedFamily F) (c : X → ℝ) :
    famPrice c (fullBlockFamily F) = ∑ i, genPrice c (block F i) := by
  unfold famPrice fullBlockFamily
  rw [Finset.sum_image (fun i _ j _ h => block_injective hF h)]

/-! ### The six-block benchmark -/

theorem block_benchmark816 (i : Fin 6) :
    block benchmark816 i = {i} ×ˢ (univ : Finset (Fin 409)) := by
  ext ⟨x, y⟩
  simp only [block, mem_filter, mem_univ, true_and, mem_product, mem_singleton, and_true]
  exact Iff.rfl

theorem card_block_benchmark816' (i : Fin 6) : (block benchmark816 i).card = 409 := by
  rw [block_benchmark816 i, card_product, card_singleton, card_univ, Fintype.card_fin]

/-- The same statement at the dependent index type `Fin benchmark816.b` (definitionally `Fin 6`),
so that it can be used by `rw` on goals produced from the general definitions. -/
theorem card_block_benchmark816 (i : Fin benchmark816.b) : (block benchmark816 i).card = 409 :=
  card_block_benchmark816' i

theorem mem_H_benchmark816' {e : Finset (Fin 6)} (he : e ∈ completeClutter 6 3) : e.card = 4 := by
  rw [completeClutter, mem_powersetCard] at he
  exact he.2

theorem mem_H_benchmark816 {e : Finset (Fin benchmark816.b)} (he : e ∈ benchmark816.H) :
    e.card = 4 :=
  mem_H_benchmark816' he

theorem benchmark816_isRealized : IsRealizedFamily benchmark816 := by
  refine ⟨fun _ => le_rfl, fun _ => ?_, fun i => ?_, fun e he => ?_, fun e he f hf hef => ?_⟩
  · change (1 : ℕ) ≤ 408
    norm_num
  · change (block benchmark816 i).card = 1 * 408 + 1
    rw [card_block_benchmark816 i]
  · rw [mem_H_benchmark816 he]
    norm_num
  · exact eq_of_subset_of_card_le hef (by rw [mem_H_benchmark816 he, mem_H_benchmark816 hf])

/-- The price of the six full blocks at a constant price vector. -/
theorem famPrice_const_benchmark816 (r : ℝ) :
    famPrice (fun _ => r) (fullBlockFamily benchmark816) = 6 * r ^ 409 := by
  rw [famPrice_fullBlockFamily benchmark816_isRealized]
  have h : ∀ i : Fin benchmark816.b, genPrice (fun _ => r) (block benchmark816 i) = r ^ 409 :=
    fun i => by rw [genPrice, prod_const, card_block_benchmark816 i]
  rw [Finset.sum_congr rfl (fun i _ => h i), sum_const, card_univ]
  change (6 : ℕ) • r ^ 409 = 6 * r ^ 409
  rw [nsmul_eq_mul]
  norm_num

/-- Lower bound on `-log(1 - x)` from the Taylor estimate of Mathlib with three terms. -/
theorem neg_log_one_sub_gt (x : ℝ) (hx0 : 0 < x) (hx1 : x < 1 / 4) :
    x + x ^ 2 / 2 < -Real.log (1 - x) := by
  have habs : |x| < 1 := by rw [abs_of_pos hx0]; linarith
  have h := Real.abs_log_sub_add_sum_range_le habs 3
  rw [abs_of_pos hx0] at h
  simp only [Finset.sum_range_succ, Finset.sum_range_zero] at h
  rw [abs_le] at h
  obtain ⟨_, h2⟩ := h
  have hpos : 0 < 1 - x := by linarith
  have key : x ^ 4 / (1 - x) < x ^ 3 / 3 := by
    rw [div_lt_div_iff₀ hpos (by norm_num)]
    nlinarith [pow_pos hx0 3]
  norm_num at h2
  linarith [h2, key]


/-! ### Palette numbers of complete clutters -/

/-- For the clutter of all `(s+1)`-subsets, the edge condition of a palette says exactly that every
label lies in at most `s` of the label sets. -/
theorem completeClutter_edge_iff (b s : ℕ) {K : ℕ} (P : Fin b → Finset (Fin K)) :
    (∀ e ∈ completeClutter b s, ∀ ℓ : Fin K, ∃ i ∈ e, ℓ ∉ P i) ↔
      ∀ ℓ : Fin K, (univ.filter (fun i => ℓ ∈ P i)).card ≤ s := by
  constructor
  · intro h ℓ
    by_contra hlt
    rw [not_le] at hlt
    obtain ⟨e, he, hecard⟩ := exists_subset_card_eq (Nat.succ_le_of_lt hlt)
    have hmem : e ∈ completeClutter b s := by
      rw [completeClutter, mem_powersetCard]
      exact ⟨subset_univ _, hecard⟩
    obtain ⟨i, hi, hni⟩ := h e hmem ℓ
    exact hni (mem_filter.1 (he hi)).2
  · intro h e he ℓ
    rw [completeClutter, mem_powersetCard] at he
    by_contra hcon
    simp only [not_exists, not_and, not_not] at hcon
    have hsub : e ⊆ univ.filter (fun i => ℓ ∈ P i) :=
      fun i hi => mem_filter.2 ⟨mem_univ _, hcon i hi⟩
    have := card_le_card hsub
    have := h ℓ
    omega

/-- Double counting: the total size of the label sets is the sum over labels of their
multiplicities. -/
theorem sum_card_eq_sum_fiber {b K : ℕ} (P : Fin b → Finset (Fin K)) :
    ∑ i, (P i).card = ∑ ℓ : Fin K, (univ.filter (fun i => ℓ ∈ P i)).card := by
  have h1 : ∀ i, (P i).card = ∑ ℓ : Fin K, if ℓ ∈ P i then 1 else 0 := fun i => by
    rw [sum_boole, Nat.cast_id]
    congr 1
    ext ℓ
    simp
  simp_rw [h1, card_filter]
  exact sum_comm

/-- The two necessary conditions for a feasible palette of the complete clutter. -/
theorem palette_lower_bounds {b s K : ℕ} {d : Fin b → ℕ}
    (hK : IsPaletteFeasible b (completeClutter b s) d K) :
    (∀ i, d i ≤ K) ∧ ∑ i, d i ≤ s * K := by
  obtain ⟨P, hPd, hPe⟩ := hK
  rw [completeClutter_edge_iff] at hPe
  refine ⟨fun i => (hPd i).trans (card_finset_fin_le _), ?_⟩
  calc ∑ i, d i ≤ ∑ i, (P i).card := sum_le_sum (fun i _ => hPd i)
    _ = ∑ ℓ : Fin K, (univ.filter (fun i => ℓ ∈ P i)).card := sum_card_eq_sum_fiber P
    _ ≤ ∑ _ℓ : Fin K, s := sum_le_sum (fun ℓ _ => hPe ℓ)
    _ = s * K := by rw [sum_const, card_univ, Fintype.card_fin, smul_eq_mul, mul_comm]

/-- Prefix sums of the demands: `c_i = sum_{j < i} d_j`. -/
def prefixSum {b : ℕ} (d : Fin b → ℕ) (i : Fin b) : ℕ :=
  ∑ j ∈ univ.filter (fun j => j < i), d j

theorem prefixSum_add_le_of_lt {b : ℕ} (d : Fin b → ℕ) {i i' : Fin b} (h : i < i') :
    prefixSum d i + d i ≤ prefixSum d i' := by
  unfold prefixSum
  have hnot : i ∉ univ.filter (fun j => j < i) := by simp
  rw [add_comm, ← sum_insert hnot]
  apply sum_le_sum_of_subset
  intro j hj
  simp only [mem_insert, mem_filter, mem_univ, true_and] at hj ⊢
  rcases hj with rfl | hj
  · exact h
  · exact hj.trans h

theorem prefixSum_add_le {b : ℕ} (d : Fin b → ℕ) (i : Fin b) :
    prefixSum d i + d i ≤ ∑ j, d j := by
  unfold prefixSum
  have hnot : i ∉ univ.filter (fun j => j < i) := by simp
  rw [add_comm, ← sum_insert hnot]
  exact sum_le_sum_of_subset (subset_univ _)

/-- The cyclic label of a position. -/
def cycLabel (K : ℕ) (hK : 0 < K) (n : ℕ) : Fin K := ⟨n % K, Nat.mod_lt _ hK⟩

/-- The cyclic palette: block `i` receives the labels of the positions `c_i, ..., c_i + d_i - 1`
reduced modulo `K`. -/
def cycPalette {b : ℕ} (d : Fin b → ℕ) (K : ℕ) (hK : 0 < K) (i : Fin b) : Finset (Fin K) :=
  (Ico (prefixSum d i) (prefixSum d i + d i)).image (cycLabel K hK)

theorem mem_cycPalette {b : ℕ} (d : Fin b → ℕ) (K : ℕ) (hK : 0 < K) (i : Fin b) (ℓ : Fin K) :
    ℓ ∈ cycPalette d K hK i ↔
      ∃ n, n ∈ Ico (prefixSum d i) (prefixSum d i + d i) ∧ n % K = ℓ.val := by
  unfold cycPalette cycLabel
  simp only [mem_image, Fin.ext_iff]

theorem card_cycPalette {b : ℕ} (d : Fin b → ℕ) (K : ℕ) (hK : 0 < K) (i : Fin b)
    (hd : d i ≤ K) : (cycPalette d K hK i).card = d i := by
  unfold cycPalette
  rw [card_image_of_injOn, Nat.card_Ico, Nat.add_sub_cancel_left]
  intro n hn n' hn' hnn'
  simp only [coe_Ico, Set.mem_Ico] at hn hn'
  simp only [cycLabel, Fin.mk.injEq] at hnn'
  rcases le_total n n' with h | h
  · have h1 := Nat.sub_mod_eq_zero_of_mod_eq hnn'.symm
    have h2 : n' - n < K := by omega
    rw [Nat.mod_eq_of_lt h2] at h1
    omega
  · have h1 := Nat.sub_mod_eq_zero_of_mod_eq hnn'
    have h2 : n - n' < K := by omega
    rw [Nat.mod_eq_of_lt h2] at h1
    omega

open Classical in
/-- The first position of block `i` carrying the label `ℓ` (or `0` if there is none). -/
noncomputable def firstPos {b : ℕ} (d : Fin b → ℕ) (K ℓ : ℕ) (i : Fin b) : ℕ :=
  if h : ∃ n, n ∈ Ico (prefixSum d i) (prefixSum d i + d i) ∧ n % K = ℓ then Nat.find h else 0

theorem firstPos_spec {b : ℕ} (d : Fin b → ℕ) (K ℓ : ℕ) (i : Fin b)
    (hex : ∃ n, n ∈ Ico (prefixSum d i) (prefixSum d i + d i) ∧ n % K = ℓ) :
    firstPos d K ℓ i ∈ Ico (prefixSum d i) (prefixSum d i + d i) ∧ firstPos d K ℓ i % K = ℓ := by
  unfold firstPos
  split_ifs
  exact Nat.find_spec hex

/-- At most `s` positions below `s * K` carry a given residue modulo `K`. -/
theorem card_residue_le {K s D : ℕ} (hK : 0 < K) (hD : D ≤ s * K) (ℓ : ℕ) :
    ((range D).filter (fun n => n % K = ℓ)).card ≤ s := by
  calc ((range D).filter (fun n => n % K = ℓ)).card ≤ (range s).card :=
        card_le_card_of_injOn (fun n => n / K) ?_ ?_
    _ = s := card_range s
  · intro n hn
    simp only [mem_coe, mem_filter, mem_range] at hn ⊢
    exact (Nat.div_lt_iff_lt_mul hK).2 (hn.1.trans_le hD)
  · intro n hn n' hn' h
    simp only [mem_coe, mem_filter, mem_range] at hn hn'
    have h' : n / K = n' / K := h
    calc n = K * (n / K) + n % K := (Nat.div_add_mod n K).symm
      _ = K * (n' / K) + n' % K := by rw [h', hn.2, hn'.2]
      _ = n' := Nat.div_add_mod n' K

/-- The cyclic palette is feasible for the complete clutter whenever the two necessary conditions
hold. -/
theorem palette_witness {b s K : ℕ} (d : Fin b → ℕ) (hdK : ∀ i, d i ≤ K)
    (hsum : ∑ i, d i ≤ s * K) : IsPaletteFeasible b (completeClutter b s) d K := by
  rcases Nat.eq_zero_or_pos K with hK0 | hK
  · subst hK0
    refine ⟨fun _ => ∅, fun i => by simpa using hdK i, fun e _ ℓ => ℓ.elim0⟩
  · refine ⟨cycPalette d K hK, fun i => (card_cycPalette d K hK i (hdK i)).ge, ?_⟩
    rw [completeClutter_edge_iff]
    intro ℓ
    calc (univ.filter (fun i => ℓ ∈ cycPalette d K hK i)).card
        ≤ ((range (∑ i, d i)).filter (fun n => n % K = ℓ.val)).card :=
          card_le_card_of_injOn (firstPos d K ℓ.val) ?_ ?_
      _ ≤ s := card_residue_le hK hsum ℓ.val
    · intro i hi
      simp only [mem_coe, mem_filter, mem_univ, true_and, mem_cycPalette] at hi
      obtain ⟨h1, h2⟩ := firstPos_spec d K ℓ.val i hi
      simp only [mem_coe, mem_filter, mem_range]
      refine ⟨?_, h2⟩
      have := (mem_Ico.1 h1).2
      have := prefixSum_add_le d i
      omega
    · intro i hi i' hi' hii'
      simp only [mem_coe, mem_filter, mem_univ, true_and, mem_cycPalette] at hi hi'
      have h1 := mem_Ico.1 (firstPos_spec d K ℓ.val i hi).1
      have h2 := mem_Ico.1 (firstPos_spec d K ℓ.val i' hi').1
      rw [hii'] at h1
      by_contra hne
      rcases lt_or_gt_of_ne hne with hlt | hlt
      · have := prefixSum_add_le_of_lt d hlt
        omega
      · have := prefixSum_add_le_of_lt d hlt
        omega

/-- The palette number of the complete clutter, display (P9), in expanded form. -/
theorem paletteNumber_completeClutter (b s : ℕ) (d : Fin b → ℕ) (hs : 1 ≤ s) :
    paletteNumber b (completeClutter b s) d =
      max (univ.sup d) ⌈((∑ i, d i : ℕ) : ℚ) / (s : ℚ)⌉₊ := by
  have hs' : (0 : ℚ) < s := by exact_mod_cast hs
  apply IsLeast.csInf_eq
  constructor
  · show IsPaletteFeasible b (completeClutter b s) d _
    apply palette_witness
    · intro i
      exact (le_sup (mem_univ i)).trans (le_max_left _ _)
    · have h1 : ((∑ i, d i : ℕ) : ℚ) / s ≤ ⌈((∑ i, d i : ℕ) : ℚ) / s⌉₊ := Nat.le_ceil _
      have h2 : (⌈((∑ i, d i : ℕ) : ℚ) / s⌉₊ : ℚ) ≤
          (max (univ.sup d) ⌈((∑ i, d i : ℕ) : ℚ) / (s : ℚ)⌉₊ : ℕ) := by
        exact_mod_cast le_max_right _ _
      rw [div_le_iff₀ hs'] at h1
      have h3 : ((∑ i, d i : ℕ) : ℚ) ≤
          (s : ℚ) * (max (univ.sup d) ⌈((∑ i, d i : ℕ) : ℚ) / (s : ℚ)⌉₊ : ℕ) := by
        calc ((∑ i, d i : ℕ) : ℚ) ≤ (⌈((∑ i, d i : ℕ) : ℚ) / s⌉₊ : ℚ) * s := h1
          _ ≤ (max (univ.sup d) ⌈((∑ i, d i : ℕ) : ℚ) / (s : ℚ)⌉₊ : ℕ) * s :=
            mul_le_mul_of_nonneg_right h2 hs'.le
          _ = _ := mul_comm _ _
      exact_mod_cast h3
  · intro K hK
    obtain ⟨h1, h2⟩ := palette_lower_bounds hK
    apply max_le
    · exact Finset.sup_le (fun i _ => h1 i)
    · rw [Nat.ceil_le, div_le_iff₀ hs']
      exact_mod_cast (by rw [mul_comm]; exact h2 : ∑ i, d i ≤ K * s)


/-! ### Palettes exist and are upward closed -/

/-- Disjoint label intervals give a feasible palette of size `sum d_i` for any clutter whose edges
have at least two elements. -/
theorem isPaletteFeasible_sum {b : ℕ} (H : Finset (Finset (Fin b))) (d : Fin b → ℕ)
    (hH : ∀ e ∈ H, 2 ≤ e.card) : IsPaletteFeasible b H d (∑ i, d i) := by
  refine ⟨fun i => (Ico (prefixSum d i) (prefixSum d i + d i)).attachFin
      (fun m hm => lt_of_lt_of_le (mem_Ico.1 hm).2 (prefixSum_add_le d i)), fun i => ?_, ?_⟩
  · rw [card_attachFin, Nat.card_Ico, Nat.add_sub_cancel_left]
  · intro e he ℓ
    by_contra hcon
    simp only [not_exists, not_and, mem_attachFin, mem_Ico] at hcon
    have h2 := hH e he
    obtain ⟨i, hi⟩ : e.Nonempty := card_pos.1 (by omega)
    obtain ⟨j, hj, hji⟩ := exists_mem_ne (by omega : 1 < e.card) i
    have h1 := hcon i hi
    have h2 := hcon j hj
    rcases lt_or_gt_of_ne hji with hlt | hlt
    · have := prefixSum_add_le_of_lt d hlt
      omega
    · have := prefixSum_add_le_of_lt d hlt
      omega

/-- Feasibility is upward closed in the palette size (for clutters with edges of size `≥ 2`). -/
theorem isPaletteFeasible_mono {b : ℕ} (H : Finset (Finset (Fin b))) (d : Fin b → ℕ)
    (hH : ∀ e ∈ H, 2 ≤ e.card) {K K' : ℕ} (hKK' : K ≤ K')
    (h : IsPaletteFeasible b H d K) : IsPaletteFeasible b H d K' := by
  obtain ⟨P, hPd, hPe⟩ := h
  refine ⟨fun i => (P i).map (Fin.castLEEmb hKK'), fun i => ?_, ?_⟩
  · rw [card_map]
    exact hPd i
  · intro e he ℓ
    by_cases hℓ : ℓ.val < K
    · obtain ⟨i, hi, hni⟩ := hPe e he ⟨ℓ.val, hℓ⟩
      refine ⟨i, hi, fun hmem => hni ?_⟩
      rw [mem_map] at hmem
      obtain ⟨ℓ', hℓ', hcast⟩ := hmem
      have : ℓ' = ⟨ℓ.val, hℓ⟩ := by
        apply Fin.ext
        have := congrArg Fin.val hcast
        simpa using this
      rw [← this]
      exact hℓ'
    · have h2 := hH e he
      obtain ⟨i, hi⟩ : e.Nonempty := card_pos.1 (by omega)
      refine ⟨i, hi, fun hmem => hℓ ?_⟩
      rw [mem_map] at hmem
      obtain ⟨ℓ', _, hcast⟩ := hmem
      rw [← hcast]
      simp

/-- The palette number is attained and bounded by `sum d_i` when every edge has size `≥ 2`. -/
theorem isPaletteFeasible_paletteNumber {b : ℕ} (H : Finset (Finset (Fin b))) (d : Fin b → ℕ)
    (hH : ∀ e ∈ H, 2 ≤ e.card) : IsPaletteFeasible b H d (paletteNumber b H d) := by
  have h : sInf {K : ℕ | IsPaletteFeasible b H d K} ∈ {K : ℕ | IsPaletteFeasible b H d K} :=
    csInf_mem ⟨∑ i, d i, isPaletteFeasible_sum H d hH⟩
  exact h

theorem isPaletteFeasible_of_paletteNumber_le {b : ℕ} (H : Finset (Finset (Fin b)))
    (d : Fin b → ℕ) (hH : ∀ e ∈ H, 2 ≤ e.card) {K : ℕ} (hK : paletteNumber b H d ≤ K) :
    IsPaletteFeasible b H d K :=
  isPaletteFeasible_mono H d hH hK (isPaletteFeasible_paletteNumber H d hH)

theorem paletteNumber_le_of_feasible {b : ℕ} (H : Finset (Finset (Fin b))) (d : Fin b → ℕ)
    {K : ℕ} (hK : IsPaletteFeasible b H d K) : paletteNumber b H d ≤ K :=
  csInf_le' (show K ∈ {K : ℕ | IsPaletteFeasible b H d K} from hK)

/-! ### Decompositions of a set into good parts versus palettes -/

section Decomp

variable {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}

/-- A `K`-decomposition of `U` yields a palette with `|U ∩ X_i| ≤ a_i |P_i|`: `P_i` is the set of
parts meeting `X_i`. -/
theorem palette_of_decomposable {K : ℕ} {U : Finset X} (h : IsKDecomposable F K U) :
    ∃ P : Fin F.b → Finset (Fin K), (∀ i, (U ∩ block F i).card ≤ F.a i * (P i).card) ∧
      ∀ e ∈ F.H, ∀ ℓ : Fin K, ∃ i ∈ e, ℓ ∉ P i := by
  obtain ⟨parts, hgood, -, hunion⟩ := h
  refine ⟨fun i => univ.filter (fun ℓ => (parts ℓ ∩ block F i).Nonempty), fun i => ?_, ?_⟩
  · set S := univ.filter (fun ℓ => (parts ℓ ∩ block F i).Nonempty) with hS
    have hU : U ∩ block F i = S.biUnion (fun ℓ => parts ℓ ∩ block F i) := by
      ext v
      simp only [hS, mem_inter, mem_biUnion, mem_filter, mem_univ, true_and]
      constructor
      · rintro ⟨hvU, hvb⟩
        rw [← hunion, mem_biUnion] at hvU
        obtain ⟨ℓ, -, hvℓ⟩ := hvU
        exact ⟨ℓ, ⟨v, mem_inter.2 ⟨hvℓ, hvb⟩⟩, hvℓ, hvb⟩
      · rintro ⟨ℓ, -, hvℓ, hvb⟩
        refine ⟨?_, hvb⟩
        rw [← hunion, mem_biUnion]
        exact ⟨ℓ, mem_univ _, hvℓ⟩
    rw [hU]
    calc (S.biUnion (fun ℓ => parts ℓ ∩ block F i)).card
        ≤ ∑ ℓ ∈ S, (parts ℓ ∩ block F i).card := card_biUnion_le
      _ ≤ ∑ _ℓ ∈ S, F.a i := sum_le_sum (fun ℓ _ => (hgood ℓ).1 i)
      _ = F.a i * S.card := by rw [sum_const, smul_eq_mul, mul_comm]
  · intro e he ℓ
    by_contra hcon
    simp only [not_exists, not_and, not_not, mem_filter, mem_univ, true_and] at hcon
    apply (hgood ℓ).2 e he
    intro i hi
    obtain ⟨v, hv⟩ := hcon i hi
    rw [mem_inter] at hv
    rw [supp, mem_image]
    exact ⟨v, hv.1, blk_eq_of_mem_block hv.2⟩

/-- The index of `v` within `U ∩ X_i` (`0` if `v` is not there). -/
noncomputable def idxIn (F : RealizedData X) (U : Finset X) (i : Fin F.b) (v : X) : ℕ :=
  if h : v ∈ U ∩ block F i then ((U ∩ block F i).equivFin ⟨v, h⟩).val else 0

theorem idxIn_lt (F : RealizedData X) (U : Finset X) (i : Fin F.b) {v : X}
    (hv : v ∈ U ∩ block F i) : idxIn F U i v < (U ∩ block F i).card := by
  unfold idxIn
  split_ifs
  exact Fin.isLt _

theorem idxIn_inj (F : RealizedData X) (U : Finset X) (i : Fin F.b) {v w : X}
    (hv : v ∈ U ∩ block F i) (hw : w ∈ U ∩ block F i) (h : idxIn F U i v = idxIn F U i w) :
    v = w := by
  unfold idxIn at h
  split_ifs at h
  have := (U ∩ block F i).equivFin.injective (Fin.ext h)
  exact congrArg Subtype.val this

/-- The `r`-th label of `P_i`, as an `Option`. -/
noncomputable def labOf {b K : ℕ} (P : Fin b → Finset (Fin K)) (i : Fin b) (r : ℕ) :
    Option (Fin K) :=
  if h : r < (P i).card then some ((P i).orderEmbOfFin rfl ⟨r, h⟩) else none

theorem labOf_mem {b K : ℕ} (P : Fin b → Finset (Fin K)) (i : Fin b) (r : ℕ) (ℓ : Fin K)
    (h : labOf P i r = some ℓ) : ℓ ∈ P i := by
  unfold labOf at h
  split_ifs at h with hr
  rw [Option.some_inj] at h
  rw [← h]
  exact orderEmbOfFin_mem _ _ _

theorem labOf_inj {b K : ℕ} (P : Fin b → Finset (Fin K)) (i : Fin b) {r r' : ℕ} {ℓ : Fin K}
    (h : labOf P i r = some ℓ) (h' : labOf P i r' = some ℓ) : r = r' := by
  unfold labOf at h h'
  split_ifs at h h' with hr hr'
  rw [Option.some_inj] at h h'
  have := ((P i).orderEmbOfFin rfl).injective (h.trans h'.symm)
  exact Fin.mk.inj_iff.1 this

theorem labOf_isSome {b K : ℕ} (P : Fin b → Finset (Fin K)) (i : Fin b) {r : ℕ}
    (hr : r < (P i).card) : ∃ ℓ, labOf P i r = some ℓ := by
  unfold labOf
  split_ifs
  exact ⟨_, rfl⟩

/-- The colour of `v`: the `(idx v / a_i)`-th label of `P_{blk v}`. -/
noncomputable def colOf (F : RealizedData X) (U : Finset X) {K : ℕ}
    (P : Fin F.b → Finset (Fin K)) (v : X) : Option (Fin K) :=
  labOf P (F.blk v) (idxIn F U (F.blk v) v / F.a (F.blk v))

/-- The colour class of `ℓ` inside `U`. -/
noncomputable def partOf (F : RealizedData X) (U : Finset X) {K : ℕ}
    (P : Fin F.b → Finset (Fin K)) (ℓ : Fin K) : Finset X :=
  U.filter (fun v => colOf F U P v = some ℓ)

theorem colOf_isSome (hF : IsRealizedFamily F) {U : Finset X} {K : ℕ}
    {P : Fin F.b → Finset (Fin K)} (hcard : ∀ i, (U ∩ block F i).card ≤ F.a i * (P i).card)
    {v : X} (hv : v ∈ U) : ∃ ℓ, colOf F U P v = some ℓ := by
  apply labOf_isSome
  have hvi : v ∈ U ∩ block F (F.blk v) := mem_inter.2 ⟨hv, mem_block_iff.2 rfl⟩
  have h1 := idxIn_lt F U (F.blk v) hvi
  have h2 := hcard (F.blk v)
  have ha : 0 < F.a (F.blk v) := hF.1 (F.blk v)
  rw [Nat.div_lt_iff_lt_mul ha]
  calc idxIn F U (F.blk v) v < (U ∩ block F (F.blk v)).card := h1
    _ ≤ F.a (F.blk v) * (P (F.blk v)).card := h2
    _ = (P (F.blk v)).card * F.a (F.blk v) := mul_comm _ _

theorem partOf_good (hF : IsRealizedFamily F) {U : Finset X} {K : ℕ}
    {P : Fin F.b → Finset (Fin K)}
    (hedge : ∀ e ∈ F.H, ∀ ℓ : Fin K, ∃ i ∈ e, ℓ ∉ P i) (ℓ : Fin K) :
    IsGood F (partOf F U P ℓ) := by
  refine ⟨fun i => ?_, fun e he hsub => ?_⟩
  · by_cases hne : (partOf F U P ℓ ∩ block F i).Nonempty
    · obtain ⟨v₀, hv₀⟩ := hne
      have ha : 0 < F.a i := hF.1 i
      have hr : ∀ v ∈ partOf F U P ℓ ∩ block F i,
          idxIn F U i v / F.a i = idxIn F U i v₀ / F.a i := by
        intro v hv
        rw [mem_inter, partOf, mem_filter] at hv hv₀
        have hbv := blk_eq_of_mem_block hv.2
        have hbv₀ := blk_eq_of_mem_block hv₀.2
        have h1 := hv.1.2
        have h2 := hv₀.1.2
        unfold colOf at h1 h2
        rw [hbv] at h1
        rw [hbv₀] at h2
        exact labOf_inj P i h1 h2
      calc (partOf F U P ℓ ∩ block F i).card
          ≤ (Ico (idxIn F U i v₀ / F.a i * F.a i)
              ((idxIn F U i v₀ / F.a i + 1) * F.a i)).card :=
            card_le_card_of_injOn (idxIn F U i) ?_ ?_
        _ = F.a i := by rw [Nat.card_Ico, add_mul, one_mul, Nat.add_sub_cancel_left]
      · intro v hv
        rw [mem_coe] at hv
        have hr' := hr v hv
        rw [mem_coe, mem_Ico]
        constructor
        · exact (Nat.le_div_iff_mul_le ha).1 hr'.ge
        · have : idxIn F U i v / F.a i < idxIn F U i v₀ / F.a i + 1 := by
            rw [hr']
            exact Nat.lt_succ_self _
          exact (Nat.div_lt_iff_lt_mul ha).1 this
      · intro v hv w hw hvw
        rw [mem_coe, mem_inter, partOf, mem_filter] at hv hw
        exact idxIn_inj F U i (mem_inter.2 ⟨hv.1.1, hv.2⟩) (mem_inter.2 ⟨hw.1.1, hw.2⟩) hvw
    · rw [not_nonempty_iff_eq_empty] at hne
      rw [hne, card_empty]
      exact Nat.zero_le _
  · obtain ⟨i, hi, hni⟩ := hedge e he ℓ
    have := hsub hi
    rw [supp, mem_image] at this
    obtain ⟨v, hv, hbv⟩ := this
    rw [partOf, mem_filter] at hv
    have h1 := hv.2
    unfold colOf at h1
    rw [hbv] at h1
    exact hni (labOf_mem P i _ ℓ h1)

theorem partOf_disjoint (F : RealizedData X) (U : Finset X) {K : ℕ}
    (P : Fin F.b → Finset (Fin K)) {ℓ ℓ' : Fin K} (h : ℓ ≠ ℓ') :
    Disjoint (partOf F U P ℓ) (partOf F U P ℓ') := by
  rw [disjoint_left]
  intro v hv hv'
  rw [partOf, mem_filter] at hv hv'
  apply h
  rw [← Option.some_inj, ← hv.2, ← hv'.2]

theorem biUnion_partOf (hF : IsRealizedFamily F) {U : Finset X} {K : ℕ}
    {P : Fin F.b → Finset (Fin K)} (hcard : ∀ i, (U ∩ block F i).card ≤ F.a i * (P i).card) :
    (univ : Finset (Fin K)).biUnion (partOf F U P) = U := by
  ext v
  rw [mem_biUnion]
  constructor
  · rintro ⟨ℓ, -, hv⟩
    exact (mem_filter.1 hv).1
  · intro hv
    obtain ⟨ℓ, hℓ⟩ := colOf_isSome hF hcard hv
    exact ⟨ℓ, mem_univ _, mem_filter.2 ⟨hv, hℓ⟩⟩

/-- A palette with `|U ∩ X_i| ≤ a_i |P_i|` yields a `K`-decomposition of `U`. -/
theorem decomposable_of_palette (hF : IsRealizedFamily F) {K : ℕ} {U : Finset X}
    (P : Fin F.b → Finset (Fin K)) (hcard : ∀ i, (U ∩ block F i).card ≤ F.a i * (P i).card)
    (hedge : ∀ e ∈ F.H, ∀ ℓ : Fin K, ∃ i ∈ e, ℓ ∉ P i) : IsKDecomposable F K U :=
  ⟨partOf F U P, fun ℓ => partOf_good hF hedge ℓ, fun _ _ h => partOf_disjoint F U P h,
    biUnion_partOf hF hcard⟩

omit [DecidableEq X] in
/-- Blocks are pairwise disjoint. -/
theorem block_disjoint (F : RealizedData X) {i j : Fin F.b} (h : i ≠ j) :
    Disjoint (block F i) (block F j) := by
  rw [disjoint_left]
  intro v hv hv'
  exact h ((blk_eq_of_mem_block hv).symm.trans (blk_eq_of_mem_block hv'))

/-- Section 3: `I_k(D_i) = {S : |S| ≤ a_i k}`, in expanded form. -/
theorem localDecomposability_expanded (hF : IsRealizedFamily F) (i : Fin F.b) (k : ℕ)
    (S : Finset X) (hS : S ⊆ block F i) : IsKDecomposable F k S ↔ S.card ≤ F.a i * k := by
  constructor
  · intro h
    obtain ⟨P, hcard, -⟩ := palette_of_decomposable h
    have := hcard i
    rw [inter_eq_left.2 hS] at this
    exact this.trans (Nat.mul_le_mul_left _ (card_finset_fin_le _))
  · intro h
    classical
    apply decomposable_of_palette hF (fun j => if j = i then univ else ∅)
    · intro j
      by_cases hj : j = i
      · subst hj
        rw [inter_eq_left.2 hS, ite_eq_left rfl, card_univ, Fintype.card_fin]
        exact h
      · rw [ite_eq_right hj]
        have : S ∩ block F j = ∅ := by
          rw [← disjoint_iff_inter_eq_empty]
          exact (block_disjoint F (Ne.symm hj)).mono_left hS
        rw [this, card_empty, card_empty]
        exact Nat.zero_le _
    · intro e he ℓ
      have h2 := hF.2.2.2.1 e he
      obtain ⟨j, hj, hji⟩ := exists_mem_ne (by omega : 1 < e.card) i
      refine ⟨j, hj, ?_⟩
      rw [ite_eq_right hji]
      exact notMem_empty _

end Decomp


/-! ### Minimal forbidden sets -/

section MinForb

variable {X : Type} [Fintype X] [DecidableEq X] {F : RealizedData X}

omit [Fintype X] [DecidableEq X] in
theorem mem_supp_iff {U : Finset X} {i : Fin F.b} : i ∈ supp F U ↔ ∃ v ∈ U, F.blk v = i := by
  rw [supp, mem_image]

/-- A subset of one block with at most `a_i` elements is good. -/
theorem good_of_subset_block (hF : IsRealizedFamily F) {i : Fin F.b} {S : Finset X}
    (hS : S ⊆ block F i) (hcard : S.card ≤ F.a i) : IsGood F S := by
  refine ⟨fun j => ?_, fun e he hsub => ?_⟩
  · by_cases hj : j = i
    · subst hj
      rw [inter_eq_left.2 hS]
      exact hcard
    · have : S ∩ block F j = ∅ := by
        rw [← disjoint_iff_inter_eq_empty]
        exact (block_disjoint F (Ne.symm hj)).mono_left hS
      rw [this, card_empty]
      exact Nat.zero_le _
  · have hsupp : supp F S ⊆ {i} := by
      intro j hj
      obtain ⟨v, hv, rfl⟩ := mem_supp_iff.1 hj
      rw [mem_singleton]
      exact blk_eq_of_mem_block (hS hv)
    have h1 := card_le_card (hsub.trans hsupp)
    have h2 := hF.2.2.2.1 e he
    rw [card_singleton] at h1
    omega

/-- Removing an element outside `X_i` does not change the intersection with `X_i`. -/
theorem erase_inter_block_of_blk_ne {U : Finset X} {v : X} {i : Fin F.b} (h : F.blk v ≠ i) :
    U.erase v ∩ block F i = U ∩ block F i := by
  ext w
  simp only [mem_inter, mem_erase, mem_block_iff]
  constructor
  · rintro ⟨⟨-, hw⟩, hb⟩
    exact ⟨hw, hb⟩
  · rintro ⟨hw, hb⟩
    refine ⟨⟨fun hwv => h (hwv ▸ hb), hw⟩, hb⟩

/-- Proposition P1 in expanded form. -/
theorem minimalForbidden_iff (hF : IsRealizedFamily F) (U : Finset X) :
    IsMinimalForbidden F U ↔
      ((∃ i, U ⊆ block F i ∧ U.card = F.a i + 1) ∨ ∃ e ∈ F.H, IsTransversal F e U) := by
  constructor
  · rintro ⟨hbad, hmin⟩
    by_cases hcap : ∃ i, F.a i < (U ∩ block F i).card
    · obtain ⟨i, hi⟩ := hcap
      left
      have hsub : U ⊆ block F i := by
        intro v hv
        by_contra hvb
        have hne : F.blk v ≠ i := fun h => hvb (mem_block_iff.2 h)
        have := (hmin v hv).1 i
        rw [erase_inter_block_of_blk_ne hne] at this
        omega
      refine ⟨i, hsub, ?_⟩
      rw [inter_eq_left.2 hsub] at hi
      by_contra hne
      have hgt : F.a i + 1 < U.card := by omega
      obtain ⟨v, hv⟩ : U.Nonempty := card_pos.1 (by omega)
      have := (hmin v hv).1 i
      rw [inter_eq_left.2 ((erase_subset _ _).trans hsub), card_erase_of_mem hv] at this
      omega
    · right
      have hcap' : ∀ i, (U ∩ block F i).card ≤ F.a i := by
        intro i
        by_contra h
        exact hcap ⟨i, by omega⟩
      have hex : ¬ ∀ e ∈ F.H, ¬ e ⊆ supp F U := fun hB => hbad ⟨hcap', hB⟩
      simp only [not_forall, not_not] at hex
      obtain ⟨e, he, hsub⟩ := hex
      -- every element of `U` is alone in its block, and its block lies in `e`
      have key : ∀ v ∈ U, F.blk v ∈ e ∧ ∀ w ∈ U, F.blk w = F.blk v → w = v := by
        intro v hv
        have hgood := (hmin v hv).2 e he
        have : ∃ i ∈ e, i ∉ supp F (U.erase v) := by
          by_contra hall
          simp only [not_exists, not_and, not_not] at hall
          exact hgood (fun i hi => hall i hi)
        obtain ⟨i, hi, hni⟩ := this
        obtain ⟨w, hw, hwi⟩ := mem_supp_iff.1 (hsub hi)
        have hwv : w = v := by
          by_contra hwv
          exact hni (mem_supp_iff.2 ⟨w, mem_erase.2 ⟨hwv, hw⟩, hwi⟩)
        subst hwv
        refine ⟨hwi ▸ hi, fun u hu hub => ?_⟩
        by_contra huw
        exact hni (mem_supp_iff.2 ⟨u, mem_erase.2 ⟨huw, hu⟩, hub.trans hwi⟩)
      refine ⟨e, he, ?_, fun i hi => ?_⟩
      · apply Subset.antisymm
        · intro i hi
          obtain ⟨v, hv, rfl⟩ := mem_supp_iff.1 hi
          exact (key v hv).1
        · exact hsub
      · obtain ⟨v, hv, hvi⟩ := mem_supp_iff.1 (hsub hi)
        rw [card_eq_one]
        refine ⟨v, ?_⟩
        ext w
        simp only [mem_inter, mem_block_iff, mem_singleton]
        constructor
        · rintro ⟨hw, hwi⟩
          exact (key v hv).2 w hw (hwi.trans hvi.symm)
        · rintro rfl
          exact ⟨hv, hvi⟩
  · rintro (⟨i, hU, hcard⟩ | ⟨e, he, hsupp, hone⟩)
    · refine ⟨fun hg => ?_, fun v hv => ?_⟩
      · have := hg.1 i
        rw [inter_eq_left.2 hU] at this
        omega
      · apply good_of_subset_block hF ((erase_subset _ _).trans hU)
        rw [card_erase_of_mem hv, hcard]
        exact Nat.le_refl _
    · refine ⟨fun hg => hg.2 e he (by rw [hsupp]), fun v hv => ⟨fun i => ?_, fun f hf hsub => ?_⟩⟩
      · refine (card_le_card (inter_subset_inter_right (erase_subset v U))).trans ?_
        by_cases hi : i ∈ e
        · rw [hone i hi]
          exact hF.1 i
        · have : U ∩ block F i = ∅ := by
            rw [eq_empty_iff_forall_notMem]
            intro w hw
            rw [mem_inter, mem_block_iff] at hw
            exact hi (hsupp ▸ mem_supp_iff.2 ⟨w, hw.1, hw.2⟩)
          rw [this, card_empty]
          exact Nat.zero_le _
      · have hbv : F.blk v ∈ e := hsupp ▸ mem_supp_iff.2 ⟨v, hv, rfl⟩
        have hsub' : supp F (U.erase v) ⊆ e.erase (F.blk v) := by
          intro j hj
          obtain ⟨w, hw, rfl⟩ := mem_supp_iff.1 hj
          rw [mem_erase] at hw
          rw [mem_erase]
          refine ⟨fun hwv => hw.1 ?_, hsupp ▸ mem_supp_iff.2 ⟨w, hw.2, rfl⟩⟩
          have hle : (U ∩ block F (F.blk v)).card ≤ 1 := by rw [hone (F.blk v) hbv]
          rw [card_le_one] at hle
          exact hle w (mem_inter.2 ⟨hw.2, mem_block_iff.2 hwv⟩) v
            (mem_inter.2 ⟨hv, mem_block_iff.2 rfl⟩)
        have hfe : f ⊆ e := hsub.trans (hsub'.trans (erase_subset _ _))
        have := hF.2.2.2.2 f hf e he hfe
        subst this
        exact notMem_erase (F.blk v) f (hsub'.trans' hsub hbv)

end MinForb


/-! ### Counting the minimal forbidden sets of the six-block benchmark -/

section Count816

/-- Proposition P1 for the six-block data (`a_i = 1`). -/
theorem minForb816_iff (U : Finset (Fin 6 × Fin 409)) :
    IsMinimalForbidden benchmark816 U ↔
      ((∃ i, U ⊆ block benchmark816 i ∧ U.card = 2) ∨
        ∃ e ∈ benchmark816.H, IsTransversal benchmark816 e U) :=
  minimalForbidden_iff benchmark816_isRealized U

theorem supp_eq_singleton_of_pair {U : Finset (Fin 6 × Fin 409)} {i : Fin benchmark816.b}
    (hU : U ⊆ block benchmark816 i) (hcard : U.card = 2) : supp benchmark816 U = {i} := by
  apply Subset.antisymm
  · intro j hj
    obtain ⟨v, hv, rfl⟩ := mem_supp_iff.1 hj
    rw [mem_singleton]
    exact blk_eq_of_mem_block (hU hv)
  · intro j hj
    rw [mem_singleton] at hj
    subst hj
    obtain ⟨v, hv⟩ : U.Nonempty := card_pos.1 (by omega)
    exact mem_supp_iff.2 ⟨v, hv, blk_eq_of_mem_block (hU hv)⟩

/-- The pairs inside a single block. -/
def pairs816 : Finset (Finset (Fin 6 × Fin 409)) :=
  (univ : Finset (Fin benchmark816.b)).biUnion (fun i => (block benchmark816 i).powersetCard 2)

theorem mem_pairs816 {U : Finset (Fin 6 × Fin 409)} :
    U ∈ pairs816 ↔ ∃ i, U ⊆ block benchmark816 i ∧ U.card = 2 := by
  simp only [pairs816, mem_biUnion, mem_univ, true_and, mem_powersetCard]

theorem card_pairs816 : pairs816.card = 6 * Nat.choose 409 2 := by
  unfold pairs816
  rw [card_biUnion]
  · have h : ∀ i : Fin benchmark816.b,
        ((block benchmark816 i).powersetCard 2).card = Nat.choose 409 2 := fun i => by
      rw [card_powersetCard, card_block_benchmark816 i]
    rw [Finset.sum_congr rfl (fun i _ => h i), sum_const, card_univ, Fintype.card_fin]
    have hb : benchmark816.b = 6 := rfl
    rw [hb, smul_eq_mul]
  · intro i _ j _ hij
    rw [Function.onFun, disjoint_left]
    intro U hUi hUj
    rw [mem_powersetCard] at hUi hUj
    obtain ⟨v, hv⟩ : U.Nonempty := card_pos.1 (by omega)
    exact hij ((blk_eq_of_mem_block (hUi.1 hv)).symm.trans (blk_eq_of_mem_block (hUj.1 hv)))

end Count816

section Transversals

variable {ι β : Type} [Fintype ι] [DecidableEq ι] [Fintype β] [DecidableEq β]

/-- The candidate transversals of `e` in a product ground type: the images of `e` under
`i ↦ (σ i, g i)` for `g` ranging over a product set (`b₀` is a dummy value off `e`). -/
def transOf (F : RealizedData (ι × β)) (σ : Fin F.b → ι) (b₀ : β) (e : Finset (Fin F.b)) :
    Finset (Finset (ι × β)) :=
  (Fintype.piFinset (fun i : Fin F.b => if i ∈ e then (univ : Finset β) else {b₀})).image
    (fun g => e.image (fun i => (σ i, g i)))

omit [Fintype ι] [DecidableEq ι] [Fintype β] [DecidableEq β] in
theorem sigma_injective {F : RealizedData (ι × β)} {σ : Fin F.b → ι} (b₀ : β)
    (hσ : ∀ (v : ι × β) (i : Fin F.b), F.blk v = i ↔ v.1 = σ i) : Function.Injective σ := by
  intro i j h
  have hi : F.blk (σ i, b₀) = i := (hσ _ _).2 rfl
  have hj : F.blk (σ i, b₀) = j := (hσ _ _).2 h
  exact hi.symm.trans hj

theorem mem_transOf {F : RealizedData (ι × β)} {σ : Fin F.b → ι} {b₀ : β}
    (hσ : ∀ (v : ι × β) (i : Fin F.b), F.blk v = i ↔ v.1 = σ i) (e : Finset (Fin F.b))
    (U : Finset (ι × β)) : U ∈ transOf F σ b₀ e ↔ IsTransversal F e U := by
  have hinj := sigma_injective b₀ hσ
  constructor
  · intro hU
    rw [transOf, mem_image] at hU
    obtain ⟨g, -, rfl⟩ := hU
    refine ⟨?_, fun i hi => ?_⟩
    · ext j
      rw [mem_supp_iff]
      constructor
      · rintro ⟨v, hv, hvj⟩
        rw [mem_image] at hv
        obtain ⟨i, hi, rfl⟩ := hv
        rw [hσ] at hvj
        exact hinj hvj ▸ hi
      · intro hj
        exact ⟨(σ j, g j), mem_image.2 ⟨j, hj, rfl⟩, (hσ _ _).2 rfl⟩
    · rw [card_eq_one]
      refine ⟨(σ i, g i), ?_⟩
      ext v
      rw [mem_inter, mem_block_iff, mem_singleton, mem_image]
      constructor
      · rintro ⟨⟨j, hj, rfl⟩, hji⟩
        rw [hσ] at hji
        have := hinj hji
        subst this
        rfl
      · rintro rfl
        exact ⟨⟨i, hi, rfl⟩, (hσ _ _).2 rfl⟩
  · rintro ⟨hsupp, hone⟩
    classical
    rw [transOf, mem_image]
    let g : Fin F.b → β := fun i => if h : ∃ y, (σ i, y) ∈ U then Classical.choose h else b₀
    have hg : ∀ i ∈ e, (σ i, g i) ∈ U := by
      intro i hi
      obtain ⟨v, hv, hvi⟩ := mem_supp_iff.1 (hsupp ▸ hi)
      rw [hσ] at hvi
      have hex : ∃ y, (σ i, y) ∈ U := ⟨v.2, by rw [← hvi]; exact hv⟩
      show (σ i, if h : ∃ y, (σ i, y) ∈ U then Classical.choose h else b₀) ∈ U
      rw [dite_eq_left hex]
      exact Classical.choose_spec hex
    refine ⟨g, ?_, ?_⟩
    · rw [Fintype.mem_piFinset]
      intro i
      by_cases hi : i ∈ e
      · rw [ite_eq_left hi]
        exact mem_univ _
      · rw [ite_eq_right hi, mem_singleton]
        have hex : ¬ ∃ y, (σ i, y) ∈ U := by
          rintro ⟨y, hy⟩
          exact hi (hsupp ▸ mem_supp_iff.2 ⟨(σ i, y), hy, (hσ _ _).2 rfl⟩)
        show (if h : ∃ y, (σ i, y) ∈ U then Classical.choose h else b₀) = b₀
        rw [dite_eq_right hex]
    · apply Subset.antisymm
      · intro v hv
        rw [mem_image] at hv
        obtain ⟨i, hi, rfl⟩ := hv
        exact hg i hi
      · intro v hv
        have hv1 : F.blk v ∈ e := hsupp ▸ mem_supp_iff.2 ⟨v, hv, rfl⟩
        have hle : (U ∩ block F (F.blk v)).card ≤ 1 := by rw [hone _ hv1]
        rw [card_le_one] at hle
        have := hle v (mem_inter.2 ⟨hv, mem_block_iff.2 rfl⟩) (σ (F.blk v), g (F.blk v))
          (mem_inter.2 ⟨hg _ hv1, mem_block_iff.2 ((hσ _ _).2 rfl)⟩)
        rw [mem_image]
        exact ⟨F.blk v, hv1, this.symm⟩

omit [Fintype ι] in
theorem card_transOf {F : RealizedData (ι × β)} {σ : Fin F.b → ι} {b₀ : β}
    (hσ : ∀ (v : ι × β) (i : Fin F.b), F.blk v = i ↔ v.1 = σ i) (e : Finset (Fin F.b)) :
    (transOf F σ b₀ e).card = Fintype.card β ^ e.card := by
  have hinj := sigma_injective b₀ hσ
  unfold transOf
  rw [card_image_of_injOn]
  · rw [Fintype.card_piFinset]
    have h : ∀ i : Fin F.b,
        (if i ∈ e then (univ : Finset β) else {b₀}).card = if i ∈ e then Fintype.card β else 1 := by
      intro i
      split_ifs
      · rw [card_univ]
      · rw [card_singleton]
    rw [Finset.prod_congr rfl (fun i _ => h i), prod_ite_mem, univ_inter, prod_const]
  · intro g hg g' hg' hgg'
    rw [mem_coe, Fintype.mem_piFinset] at hg hg'
    have hgg'' : e.image (fun i => (σ i, g i)) = e.image (fun i => (σ i, g' i)) := hgg'
    funext i
    by_cases hi : i ∈ e
    · have : (σ i, g i) ∈ e.image (fun i => (σ i, g' i)) := by
        rw [← hgg'', mem_image]
        exact ⟨i, hi, rfl⟩
      rw [mem_image] at this
      obtain ⟨j, -, hj⟩ := this
      have h1 := congrArg Prod.fst hj
      have h2 := congrArg Prod.snd hj
      simp only at h1 h2
      have := hinj h1
      subst this
      exact h2.symm
    · have h1 := hg i
      have h2 := hg' i
      rw [ite_eq_right hi, mem_singleton] at h1 h2
      rw [h1, h2]

end Transversals

section Count816b

/-- The index map of the six-block benchmark is the identity (definitionally). -/
theorem hσ816 : ∀ (v : Fin 6 × Fin 409) (i : Fin benchmark816.b),
    benchmark816.blk v = i ↔ v.1 = (fun i => i : Fin benchmark816.b → Fin 6) i :=
  fun _ _ => Iff.rfl

/-- All transversals of all edges. -/
def transAll816 : Finset (Finset (Fin 6 × Fin 409)) :=
  benchmark816.H.biUnion (transOf benchmark816 (fun i => i) 0)

theorem mem_transAll816 {U : Finset (Fin 6 × Fin 409)} :
    U ∈ transAll816 ↔ ∃ e ∈ benchmark816.H, IsTransversal benchmark816 e U := by
  simp only [transAll816, mem_biUnion, mem_transOf hσ816]

theorem card_H816 : benchmark816.H.card = 15 := by
  change (completeClutter 6 3).card = 15
  rw [completeClutter, card_powersetCard, card_univ, Fintype.card_fin]
  rfl

theorem card_transAll816 : transAll816.card = 15 * 409 ^ 4 := by
  unfold transAll816
  rw [card_biUnion]
  · have h : ∀ e ∈ benchmark816.H, (transOf benchmark816 (fun i => i) 0 e).card = 409 ^ 4 :=
      fun e he => by rw [card_transOf hσ816, Fintype.card_fin, mem_H_benchmark816 he]
    rw [Finset.sum_congr rfl h, sum_const, card_H816, smul_eq_mul]
  · intro e _ e' _ hee'
    rw [Function.onFun, disjoint_left]
    intro U hU hU'
    rw [mem_transOf hσ816] at hU hU'
    exact hee' (hU.1.symm.trans hU'.1)

-- The style linter `constructorNameAsVariable` (variable names shadowing constructor names)
-- exhausts the recursion limit on this local context; it is a naming linter, not a check.
set_option linter.constructorNameAsVariable false in
theorem disjoint_pairs816_transAll816 : Disjoint pairs816 transAll816 := by
  rw [disjoint_left]
  intro U hU hU'
  rw [mem_pairs816] at hU
  rw [mem_transAll816] at hU'
  obtain ⟨i, hUi, hcard⟩ := hU
  obtain ⟨e, he, hsupp, -⟩ := hU'
  have h1 := supp_eq_singleton_of_pair hUi hcard
  have h2 := mem_H_benchmark816 he
  rw [hsupp] at h1
  rw [h1, card_singleton] at h2
  exact absurd h2 (by norm_num)

theorem minForbSet_eq :
    {U : Finset (Fin 6 × Fin 409) | IsMinimalForbidden benchmark816 U} =
      ↑(pairs816 ∪ transAll816) := by
  ext U
  simp only [Set.mem_ofPred_eq, coe_union, Set.mem_union, mem_coe, mem_pairs816, mem_transAll816]
  exact minForb816_iff U

theorem minForbSet1_eq :
    {U : Finset (Fin 6 × Fin 409) |
        IsMinimalForbidden benchmark816 U ∧ (supp benchmark816 U).card = 1} = ↑pairs816 := by
  ext U
  simp only [Set.mem_ofPred_eq, mem_coe, mem_pairs816, minForb816_iff]
  constructor
  · rintro ⟨h | ⟨e, he, hsupp, -⟩, hcard⟩
    · exact h
    · exfalso
      rw [hsupp, mem_H_benchmark816 he] at hcard
      omega
  · rintro ⟨i, hUi, hcard⟩
    refine ⟨Or.inl ⟨i, hUi, hcard⟩, ?_⟩
    rw [supp_eq_singleton_of_pair hUi hcard, card_singleton]

theorem minForbSet4_eq :
    {U : Finset (Fin 6 × Fin 409) |
        IsMinimalForbidden benchmark816 U ∧ (supp benchmark816 U).card = 4} = ↑transAll816 := by
  ext U
  simp only [Set.mem_ofPred_eq, mem_coe, mem_transAll816, minForb816_iff]
  constructor
  · rintro ⟨⟨i, hUi, hcard⟩ | h, hcard4⟩
    · exfalso
      rw [supp_eq_singleton_of_pair hUi hcard, card_singleton] at hcard4
      omega
    · exact h
  · rintro ⟨e, he, hsupp, hone⟩
    refine ⟨Or.inr ⟨e, he, hsupp, hone⟩, ?_⟩
    rw [hsupp, mem_H_benchmark816 he]

end Count816b

end Aux

/-- Proves: `UniversalLaw.Spec.P15.Benchmark816IsRealized` (claim `p15-realized-covers`, Section 6
of P15_REALIZED_COVERS.md: six blocks of `409` coordinates with `a_i = 1`, `d_i = 408`, `H` all
four-block subsets).
Method: block cardinality `1 * 409 = 409` via the product decomposition of the fibre; clutter
property of `powersetCard 4` by `eq_of_subset_of_card_le`.
Does not establish: anything about the informal six-block realization beyond this data record
satisfying the formal side conditions; no review, acceptance or alignment. -/
theorem benchmark816IsRealized : UniversalLaw.Spec.P15.Benchmark816IsRealized :=
  Aux.benchmark816_isRealized

/-- Proves: `UniversalLaw.Spec.P15.Benchmark816Price` (claim `p15-realized-covers`, Section 6:
"6/(40900^409)>0").
Method: the full-block family is the image of the injective block map, so its price is the sum
of six block prices `(1/40900)^409`.
Does not establish: that this cover is optimal or that it covers the obstruction
(`Benchmark816Cover` is a separate `Prop`); no review or acceptance. -/
theorem benchmark816Price : UniversalLaw.Spec.P15.Benchmark816Price := by
  refine ⟨?_, by positivity⟩
  rw [Aux.famPrice_const_benchmark816, one_div, div_eq_mul_inv, ← inv_pow]

/-- Proves: `UniversalLaw.Spec.P15.Benchmark816RationalPrice` (claim `p15-price-budget-restricted`,
display (T7) and the explicit rational price of Section 4 of price_budget PROOF.md).
Method: `log x ≤ x - 1` for the upper bound; Mathlib's Taylor estimate
`Real.abs_log_sub_add_sum_range_le` with three terms for `p + p^2/2 < -log(1 - p)`; exact
rational arithmetic by `norm_num`; the full-block price as six equal block prices.
Does not establish: the (T1) budget itself, admissibility of any other price vector, or the
informal claim; no review or acceptance. -/
theorem benchmark816RationalPrice : UniversalLaw.Spec.P15.Benchmark816RationalPrice := by
  refine ⟨?_, by norm_num, ?_, ?_⟩
  · have hpos : (0 : ℝ) < 40900 / 40899 := by norm_num
    have hlog_le : Real.log (40900 / 40899) ≤ 1 / 40899 := by
      have := Real.log_le_sub_one_of_pos hpos
      linarith
    have hlog_nn : 0 ≤ Real.log (40900 / 40899) := Real.log_nonneg (by norm_num)
    have hpow := pow_le_pow_left₀ hlog_nn hlog_le 409
    calc 6 * Real.log (40900 / 40899) ^ 409 ≤ 6 * (1 / 40899 : ℝ) ^ 409 :=
          mul_le_mul_of_nonneg_left hpow (by norm_num)
      _ = 6 / 40899 ^ 409 := by rw [one_div, div_eq_mul_inv, ← inv_pow]
  · unfold phi
    split_ifs with h
    · apply lt_min (by norm_num)
      have := Aux.neg_log_one_sub_gt (1 / 40900) (by norm_num) (by norm_num)
      linarith
    · exfalso
      exact h (by norm_num)
  · exact Aux.famPrice_const_benchmark816 _

/-- Proves: `UniversalLaw.Spec.P15.ElementaryPaletteFormula` (claim `p15-realized-covers`, display
(P9) of P15_REALIZED_COVERS.md: for `H` all `(s+1)`-subsets, `K_H(d) = max(max_i d_i,
ceil(sum_i d_i / s))`).
Method: for the complete clutter the edge condition says each label lies in at most `s` label
sets; double counting gives the two necessary conditions `d_i ≤ K` and `sum d_i ≤ s K`; the
cyclic palette (block `i` takes positions `c_i, …, c_i + d_i - 1` reduced modulo `K`) realizes
them; `IsLeast.csInf_eq` identifies `sInf`.
Does not establish: the matroid palette formula of Section 5, anything about non-complete
clutters, or the informal claim; the hypotheses `s ≤ b` and `1 ≤ d_i` of the `Prop` are not
needed by this proof and are simply carried; no review or acceptance. -/
theorem elementaryPaletteFormula : UniversalLaw.Spec.P15.ElementaryPaletteFormula :=
  fun b s d hs _ _ => Aux.paletteNumber_completeClutter b s d hs

/-- Proves: `UniversalLaw.Spec.P15.TriangleNeedsThree` (claim `p15-realized-covers`, Section 5:
the triangle at unit demands needs `3` colors).
Method: the complete-clutter formula at `b = 3`, `s = 1`, `d = 1`: `max 1 ⌈3/1⌉₊ = 3`.
Does not establish: the informal remark about "size-two edge loads"; no review or acceptance. -/
theorem triangleNeedsThree : UniversalLaw.Spec.P15.TriangleNeedsThree := by
  unfold UniversalLaw.Spec.P15.TriangleNeedsThree
  rw [Aux.paletteNumber_completeClutter 3 1 (fun _ => 1) le_rfl]
  rw [sup_const univ_nonempty, sum_const, card_univ, Fintype.card_fin, smul_eq_mul]
  norm_num

/-- Proves: `UniversalLaw.Spec.P15.Benchmark816Palette` (claim `p15-realized-covers`, Section 6:
`K_H(408,...,408) = 816`).
Method: the complete-clutter formula at `b = 6`, `s = 3`, `d = 408`: `max 408 ⌈2448/3⌉₊ = 816`.
Does not establish: that the full blocks cover the obstruction (`Benchmark816Cover`), or the
informal claim; no review or acceptance. -/
theorem benchmark816Palette : UniversalLaw.Spec.P15.Benchmark816Palette := by
  unfold UniversalLaw.Spec.P15.Benchmark816Palette
  rw [Aux.paletteNumber_completeClutter 6 3 (fun _ => 408) (by norm_num)]
  rw [sup_const univ_nonempty, sum_const, card_univ, Fintype.card_fin, smul_eq_mul]
  norm_num

/-- Proves: `UniversalLaw.Spec.P15.LocalDecomposability` (claim `p15-realized-covers`, Section 3 of
P15_REALIZED_COVERS.md: `I_k(D_i) = {S : |S| <= a_i k}`).
Method: a decomposition of `S ⊆ X_i` into `k` good parts has parts of size `≤ a_i`; conversely
the chunks of size `≤ a_i` of an enumeration of `S` are good (one block, support of size `≤ 1`
contains no edge of size `≥ 2`), via the general palette/decomposition correspondence.
Does not establish: the informal claim or its review; alignment of `IsKDecomposable` with the
informal `I_k`. -/
theorem localDecomposability {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.LocalDecomposability F :=
  fun hF i k S hS => Aux.localDecomposability_expanded hF i k S hS

/-- Proves: `UniversalLaw.Spec.P15.LocalObstructionIsFullBlock` (claim `p15-realized-covers`,
display (P3): the only `d_i`-obstructed subset of `X_i` is `X_i`).
Method: `LocalDecomposability` at `k = d_i` and `|X_i| = a_i d_i + 1`.
Does not establish: the informal claim or its review. -/
theorem localObstructionIsFullBlock {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.LocalObstructionIsFullBlock F := by
  intro hF i S hS
  rw [Aux.localDecomposability_expanded hF i (F.d i) S hS, not_le]
  have hcard := hF.2.2.1 i
  constructor
  · intro h
    exact eq_of_subset_of_card_le hS (by omega)
  · intro h
    rw [h, hcard]
    exact Nat.lt_succ_self _

/-- Proves: `UniversalLaw.Spec.P15.FullBlockCoverIff` (claim `p15-realized-covers`, Proposition
P2, display (P7): the family of full blocks covers `O_K(D)` iff `K >= K_H(d)`).
Method: a set containing no full block has `|U ∩ X_i| ≤ a_i d_i`, so a palette for `d` at `K`
colours it; conversely the set missing one chosen vertex per block is `K`-decomposable, and the
parts meeting each block form a feasible palette, so `K_H(d) ≤ K`. Attainment of `K_H(d)` and
upward closure of feasibility are proved via disjoint label intervals.
Does not establish: optimality of other covers, the unrestricted prize, the informal claim or
its review (disposition transcribed in the statement: HOLD_WITH_DOMAIN). -/
theorem fullBlockCoverIff {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.FullBlockCoverIff F := by
  intro hF K
  have hH : ∀ e ∈ F.H, 2 ≤ e.card := hF.2.2.2.1
  constructor
  · intro hcover
    classical
    -- one chosen vertex per block
    let pick : Fin F.b → X := fun i => (Aux.block_nonempty hF i).choose
    have hpick : ∀ i, pick i ∈ block F i := fun i => (Aux.block_nonempty hF i).choose_spec
    let U : Finset X := univ.filter (fun v => v ≠ pick (F.blk v))
    have hUdec : IsKDecomposable F K U := by
      by_contra hobs
      obtain ⟨g, hg, hgU⟩ := hcover U hobs
      rw [fullBlockFamily, mem_image] at hg
      obtain ⟨i, -, rfl⟩ := hg
      have := hgU (hpick i)
      simp only [U, mem_filter, mem_univ, true_and] at this
      exact this (by rw [Aux.blk_eq_of_mem_block (hpick i)])
    obtain ⟨P, hcard, hedge⟩ := Aux.palette_of_decomposable hUdec
    apply Aux.paletteNumber_le_of_feasible
    refine ⟨P, fun i => ?_, hedge⟩
    have h1 := hcard i
    have h2 : U ∩ block F i = (block F i).erase (pick i) := by
      ext v
      simp only [U, mem_inter, mem_filter, mem_univ, true_and, mem_erase]
      constructor
      · rintro ⟨hne, hv⟩
        exact ⟨by rwa [Aux.blk_eq_of_mem_block hv] at hne, hv⟩
      · rintro ⟨hne, hv⟩
        exact ⟨by rwa [Aux.blk_eq_of_mem_block hv], hv⟩
    rw [h2, card_erase_of_mem (hpick i), hF.2.2.1 i, Nat.add_sub_cancel] at h1
    exact Nat.le_of_mul_le_mul_left h1 (hF.1 i)
  · intro hK U hU
    obtain ⟨P, hPd, hPe⟩ := Aux.isPaletteFeasible_of_paletteNumber_le F.H F.d hH hK
    by_contra hnone
    simp only [not_exists, not_and] at hnone
    apply hU
    apply Aux.decomposable_of_palette hF P _ hPe
    intro i
    have hsub : U ∩ block F i ⊆ block F i := inter_subset_right
    have hne : U ∩ block F i ≠ block F i := by
      intro heq
      apply hnone (block F i) (mem_image_of_mem _ (mem_univ i))
      intro v hv
      rw [← heq] at hv
      exact (mem_inter.1 hv).1
    have hlt := card_lt_card (Finset.ssubset_iff_subset_ne.2 ⟨hsub, hne⟩)
    rw [hF.2.2.1 i] at hlt
    calc (U ∩ block F i).card ≤ F.a i * F.d i := by omega
      _ ≤ F.a i * (P i).card := Nat.mul_le_mul_left _ (hPd i)

/-- Proves: `UniversalLaw.Spec.P15.WholeGroundChromaticNumber` (claim `p15-realized-covers`,
Section 4: the chromatic number of the entire ground set is `K_H(d+1)`).
Method: the palette/decomposition correspondence with `|X_i| = a_i d_i + 1`, which lies in
`(a_i d_i, a_i (d_i + 1)]`, so the demands are exactly `d_i + 1`.
Does not establish: the informal claim or its review. -/
theorem wholeGroundChromaticNumber {X : Type} [Fintype X] [DecidableEq X] (F : RealizedData X) :
    UniversalLaw.Spec.P15.WholeGroundChromaticNumber F := by
  intro hF K
  have hH : ∀ e ∈ F.H, 2 ≤ e.card := hF.2.2.2.1
  constructor
  · intro h
    obtain ⟨P, hcard, hedge⟩ := Aux.palette_of_decomposable h
    apply Aux.paletteNumber_le_of_feasible
    refine ⟨P, fun i => ?_, hedge⟩
    have h1 := hcard i
    rw [univ_inter, hF.2.2.1 i] at h1
    show F.d i + 1 ≤ (P i).card
    by_contra hlt
    rw [not_le] at hlt
    have hle : (P i).card ≤ F.d i := Nat.lt_succ_iff.1 hlt
    have : F.a i * (P i).card ≤ F.a i * F.d i := Nat.mul_le_mul_left _ hle
    omega
  · intro hK
    obtain ⟨P, hPd, hPe⟩ :=
      Aux.isPaletteFeasible_of_paletteNumber_le F.H (fun i => F.d i + 1) hH hK
    apply Aux.decomposable_of_palette hF P _ hPe
    intro i
    rw [univ_inter, hF.2.2.1 i]
    have ha := hF.1 i
    calc F.a i * F.d i + 1 ≤ F.a i * (F.d i + 1) := by rw [mul_add, mul_one]; omega
      _ ≤ F.a i * (P i).card := Nat.mul_le_mul_left _ (hPd i)

/-- Proves: `UniversalLaw.Spec.P15.Benchmark816WholeGround` (claim `p15-realized-covers`, Section
6: the whole ground set needs `818` colors).
Method: `WholeGroundChromaticNumber` for the six-block data and the complete-clutter palette
formula at demands `409`: `max 409 ⌈2454/3⌉₊ = 818`.
Does not establish: the informal claim or its review. -/
theorem benchmark816WholeGround : UniversalLaw.Spec.P15.Benchmark816WholeGround := by
  have h := wholeGroundChromaticNumber benchmark816 Aux.benchmark816_isRealized
  have hpal : paletteNumber benchmark816.b benchmark816.H (fun i => benchmark816.d i + 1) = 818 := by
    change paletteNumber 6 (completeClutter 6 3) (fun _ => 409) = 818
    rw [Aux.paletteNumber_completeClutter 6 3 (fun _ => 409) (by norm_num)]
    rw [sup_const univ_nonempty, sum_const, card_univ, Fintype.card_fin, smul_eq_mul]
    norm_num
  constructor
  · rw [h 818, hpal]
  · rw [h 817, hpal]
    norm_num

/-- Proves: `UniversalLaw.Spec.P15.MinimalForbiddenCharacterisation` (claim `p15-realized-covers`,
Proposition P1 of P15_REALIZED_COVERS.md: the inclusion-minimal forbidden sets are exactly the
`(a_i+1)`-subsets of a single block and the transversals of the edges).
Method: a capacity violation in block `i` forces `U ⊆ X_i` with `|U| = a_i + 1` (removing any
outside vertex or any vertex keeps the violation); otherwise an edge `e ⊆ supp U` forces each
vertex to be alone in its block with its block in `e`. Conversely both families are forbidden
and every one-vertex deletion is good (one-block sets of size `≤ a_i`; the clutter property for
transversals).
Does not establish: the informal claim or its review; alignment of `IsMinimalForbidden` with the
informal notion. -/
theorem minimalForbiddenCharacterisation {X : Type} [Fintype X] [DecidableEq X]
    (F : RealizedData X) : UniversalLaw.Spec.P15.MinimalForbiddenCharacterisation F :=
  fun hF U => Aux.minimalForbidden_iff hF U

/-- Proves: `UniversalLaw.Spec.P15.Benchmark816MinimalForbiddenCounts` (claim
`p15-realized-covers`, display (P10): `6 * binom(409,2)` internal pairs, `15 * 409^4` crossing
transversal quadruples, and nothing else).
Method: Proposition P1 (`MinimalForbiddenCharacterisation`) for the six-block data; the pairs
are the disjoint union over the six blocks of `powersetCard 2`; the transversals of a four-edge
are the images of `e` under `i ↦ (i, g i)` for `g` in a product set of size `409^4`, counted by
`card_image_of_injOn` and `Fintype.card_piFinset`; the two families are disjoint by the size of
the support. No enumeration of subsets is performed.
Does not establish: the informal claim or its review; the evaluations `500616` and
`419743994415` (core-Lean targets on main). -/
theorem benchmark816MinimalForbiddenCounts :
    UniversalLaw.Spec.P15.Benchmark816MinimalForbiddenCounts := by
  refine ⟨?_, ?_, ?_⟩
  · rw [Aux.minForbSet1_eq, Set.ncard_coe_finset, Aux.card_pairs816]
  · rw [Aux.minForbSet4_eq, Set.ncard_coe_finset, Aux.card_transAll816]
  · rw [Aux.minForbSet_eq, Set.ncard_coe_finset,
      card_union_of_disjoint Aux.disjoint_pairs816_transAll816, Aux.card_pairs816,
      Aux.card_transAll816]

end UniversalLaw.Proofs.P15Structure
