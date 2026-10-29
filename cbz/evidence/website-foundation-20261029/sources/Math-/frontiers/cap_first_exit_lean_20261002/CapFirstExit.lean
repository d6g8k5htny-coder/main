/-
  CapFirstExit: the topological step from a separating cap to the global elder level.

  `CapHyp f C M z b s` bundles the conclusions that CAP §§2–5 derive from the local good event
  `G_r` (reviews/d1_cap_elder_partner_second_pass_claude_20260930/REVIEW.md, L16–L20):

  (C1) `M ∈ C` and `f ≤ b` on the cap `C`                                   (L16);
  (C2) `f ≤ s` on the whole frontier of `C`                                  (L17–L19);
  (C3) `b < f z`, and a path from `M` to `z` along which `f ≥ s`             (L17, L20).

  The saddle form adds (C2⁺): every frontier point other than `S` has `f < s`   (L19).

  Nothing is assumed about `f` outside `C` beyond the witness path in (C3). The core theorems use
  no metric, smoothness, Morse, compactness or probabilistic hypothesis, and their separation
  statements do not even use continuity of `f`. `X` is an arbitrary topological space; the torus
  is one instance. Only the older-peak theorems add compactness, local connectedness and
  continuity of `f`, all of which hold on the torus.

  Up to v1.8 the analytic steps L6 and L9–L12 (the `w'` bound and the quantitative `‖h'‖` and `F''`
  bounds) entered `ridge_capHyp_H1` as `hconv`. L1–L5, L7 (existence,
  uniqueness and implicit-function regularity of the ridge `h`: `ridge_exists`,
  `ridge_differentiable`), L8, L11's chain rule, L13 and L16–L20 are proved, and since v1.8
  `ridge_capHyp_source` composes them from source data on a collar `[-2r - ε, 2r + ε]` of `D`
  (the source data themselves are hypotheses). Since v1.9, L6 and L9–L12 are proved too, and
  `ridge_capHyp_C3` derives `hconv`; since v2.0 `ridge_capHyp_C4` also derives SP's
  `f_xxx ≥ 7/4` (L4, from L2's second statement and (H2)). Since v2.1 `ridge_capHyp_C5` derives
  the six derivative bounds from the block bound and `|f_xxx| ≤ m` on the pin segment, and
  `ridge_capHyp_D` assumes SP's quantitative bounds on `D` only (the continuity step to the collar
  is proved). Since v2.2 `ridge_capHyp_E` assumes only `f ∈ C⁴` on an open neighbourhood of `D`
  and SP's quantitative bounds on `D`: it builds every derivative datum from `f`, and the collar
  and its bound `K` come from compactness.
  `quarticToy_E` (v2.3) exercises it away from the degenerate face, with `D⁴f ≠ 0` and (H2) sharp.
  Since v2.4 `ridge_capHyp_K` states the cap on P §7's event `G_r` for every pin gap
  `f(M) − f(S) = k r³` with `k > 0`, namely `(4/(3k)) r m² < λ` and `r n ≤ 3k/10`, by rescaling
  `f ↦ f/(6k)` (`ridge_capHyp_E` is the case `k = 1/6`). `ridge_capHyp_norm_K` takes the bounds as
  operator norms `‖D³f‖ ≤ M3`, `‖D⁴f‖ ≤ M4` on `D`.
  Since v2.5 `ridge_capHyp_blocks` takes CAP (1) as written, with CAP's partial-block norms: the
  block inequality for `D³f` is derived from the partial blocks (`block_of_partial`), using the
  symmetry of `D³f` for `f ∈ C³` (`iteratedFDeriv_three_swap₁₂`, `iteratedFDeriv_three_swap₂₃`).
  `mixedToy_blocks` exercises it with a nonzero mixed block `∂_x D_y² f = 2`.
  Since v2.6 `torus_capHyp_blocks` and `torus_elder_death_level_blocks` state the cap and the elder
  death level on the torus `(ℝ/Lℤ)^d` for a field whose periodic lift meets CAP (1) in any frame;
  `torusToy_elder` is a concrete instance on `(ℝ/16ℤ)²`.
  Also not formalized: the
  identification of the H0 persistence pairing with the elder rule (L22, P §8); the
  unstable-branch analysis (L23); and every probabilistic statement.
-/
import Mathlib

open Set unitInterval

namespace CapFirstExit

variable {X : Type*} [TopologicalSpace X]

/-! ### First exit -/

/-- First exit along a path. A path from a point of `C` to a point outside `C` meets the frontier
of `C`. -/
theorem path_meets_frontier {C : Set X} {x y : X} (γ : Path x y) (hx : x ∈ C) (hy : y ∉ C) :
    ∃ t : I, γ t ∈ frontier C := by
  have hS : (γ ⁻¹' C).Nonempty := ⟨0, by simpa using hx⟩
  have hS' : γ ⁻¹' C ≠ univ := by
    intro h
    have h1 : (1 : I) ∈ γ ⁻¹' C := h ▸ mem_univ _
    exact hy (by simpa using h1)
  obtain ⟨t, ht⟩ := nonempty_frontier_iff.mpr ⟨hS, hS'⟩
  exact ⟨t, γ.continuous.frontier_preimage_subset C ht⟩

/-- First exit for connected sets. A preconnected set that contains a point of `C` and does not
meet the frontier of `C` lies inside `C`. No path-connectedness is needed. -/
theorem preconnected_subset_of_frontier_disjoint {C K : Set X} (hK : IsPreconnected K) {M : X}
    (hMK : M ∈ K) (hMC : M ∈ C) (hfr : ∀ x ∈ K, x ∉ frontier C) : K ⊆ C := by
  intro w hwK
  by_contra hwC
  have : PreconnectedSpace K := isPreconnected_iff_preconnectedSpace.mp hK
  have hS : (((↑) : K → X) ⁻¹' C).Nonempty := ⟨⟨M, hMK⟩, hMC⟩
  have hS' : (((↑) : K → X) ⁻¹' C) ≠ univ := by
    intro h
    have h1 : (⟨w, hwK⟩ : K) ∈ (((↑) : K → X) ⁻¹' C) := h ▸ mem_univ _
    exact hwC h1
  obtain ⟨⟨x, hxK⟩, hx⟩ := nonempty_frontier_iff.mpr ⟨hS, hS'⟩
  exact hfr x hxK (continuous_subtype_val.frontier_preimage_subset C hx)

/-! ### The cap hypotheses -/

/-- The cap hypotheses (C1)–(C3) for a local maximum `M` of height `b`, a separating level `s`,
and an older point `z`. -/
structure CapHyp (f : X → ℝ) (C : Set X) (M z : X) (b s : ℝ) : Prop where
  /-- (C1) `M` lies in the cap. -/
  mem : M ∈ C
  /-- (C1) The ceiling: `f ≤ b` on the cap (L16). -/
  ceiling : ∀ x ∈ C, f x ≤ b
  /-- (C2) The frontier bound: `f ≤ s` on the frontier of the cap (L17–L19). -/
  frontier_le : ∀ x ∈ frontier C, f x ≤ s
  /-- (C3) The older endpoint: `f z > b` (L17). -/
  older : b < f z
  /-- (C3) The ridge: a path from `M` to `z` along which `f ≥ s` (L20). -/
  ridge : ∃ γ : Path M z, ∀ t, s ≤ f (γ t)

namespace CapHyp

variable {f : X → ℝ} {C : Set X} {M z : X} {b s : ℝ}

/-- `M` itself is at height at least `s` (the ridge starts at `M`). -/
theorem s_le_f_M (H : CapHyp f C M z b s) : s ≤ f M := by
  obtain ⟨γ, hγ⟩ := H.ridge
  simpa using hγ 0

/-! ### The maximin level `d_f(M)` (L21) -/

/-- The upper barrier. Every path from `M` to a point strictly above `b` passes through a point
of height at most `s`, whatever the path does outside `C`. -/
theorem barrier (H : CapHyp f C M z b s) {w : X} (hw : b < f w) (γ : Path M w) :
    ∃ t : I, f (γ t) ≤ s := by
  have hwC : w ∉ C := fun h => (not_le.mpr hw) (H.ceiling w h)
  obtain ⟨t, ht⟩ := path_meets_frontier γ H.mem hwC
  exact ⟨t, H.frontier_le _ ht⟩

end CapHyp

/-- The connection levels of `M`: heights `m` such that some path from `M` to a point strictly
above `b` stays at height at least `m`. Its supremum is the maximin level `d_f(M)` of L21. -/
def connectionLevels (f : X → ℝ) (M : X) (b : ℝ) : Set ℝ :=
  {m | ∃ (w : X) (γ : Path M w), b < f w ∧ ∀ t, m ≤ f (γ t)}

namespace CapHyp

variable {f : X → ℝ} {C : Set X} {M z : X} {b s : ℝ}

/-- The maximin level is attained and equals `s` (L21). -/
theorem maximin_isGreatest (H : CapHyp f C M z b s) :
    IsGreatest (connectionLevels f M b) s := by
  obtain ⟨γ₀, hγ₀⟩ := H.ridge
  refine ⟨⟨z, γ₀, H.older, hγ₀⟩, ?_⟩
  rintro m ⟨w, γ, hw, hγ⟩
  obtain ⟨t, ht⟩ := H.barrier hw γ
  exact (hγ t).trans ht

/-- `d_f(M) = s`, as a supremum (L21). -/
theorem maximin_eq (H : CapHyp f C M z b s) : sSup (connectionLevels f M b) = s :=
  H.maximin_isGreatest.csSup_eq

/-! ### Superlevel components: the persistence filtration (L22) -/

/-- Component separation. If `U` avoids the frontier of `C`, the connected component of `M` in
`U` lies in `C`, so it contains no point strictly above `b`. -/
theorem component_separated (H : CapHyp f C M z b s) {U : Set X}
    (hU : ∀ x ∈ frontier C, x ∉ U) {w : X} (hw : w ∈ connectedComponentIn U M) : f w ≤ b := by
  have hMU : M ∈ U := by
    by_contra hMU
    rw [connectedComponentIn_eq_empty hMU] at hw
    exact notMem_empty w hw
  exact H.ceiling w (preconnected_subset_of_frontier_disjoint isPreconnected_connectedComponentIn
    (mem_connectedComponentIn hMU) H.mem
    (fun x hx hxf => hU x hxf (connectedComponentIn_subset U M hx)) hw)

end CapHyp

/-- Component joining. If a path from `M` to `z` stays in `U`, then `z` lies in the connected
component of `M` in `U`. -/
theorem component_joined {U : Set X} {M z : X} (γ : Path M z) (hγ : ∀ t, γ t ∈ U) :
    z ∈ connectedComponentIn U M := by
  have hsub : range γ ⊆ U := by
    rintro _ ⟨t, rfl⟩
    exact hγ t
  exact (isConnected_range γ.continuous).isPreconnected.subset_connectedComponentIn
    ⟨0, γ.source⟩ hsub ⟨1, γ.target⟩

namespace CapHyp

variable {f : X → ℝ} {C : Set X} {M z : X} {b s : ℝ}

/-- **Elder merge, closed superlevel filtration `{f ≥ h}`** (the filtration of L22). The connected
component of `M` in `{f ≥ h}` contains a point strictly above `b` if and only if `h ≤ s`. -/
theorem elder_merge_iff_closed (H : CapHyp f C M z b s) (h : ℝ) :
    (∃ w ∈ connectedComponentIn {x | h ≤ f x} M, b < f w) ↔ h ≤ s := by
  constructor
  · rintro ⟨w, hw, hbw⟩
    by_contra hhs
    rw [not_le] at hhs
    have hU : ∀ x ∈ frontier C, x ∉ {x | h ≤ f x} := by
      intro x hx hxU
      have h1 : h ≤ f x := hxU
      have h2 := H.frontier_le x hx
      linarith
    exact absurd (H.component_separated hU hw) (not_le.mpr hbw)
  · intro hhs
    obtain ⟨γ₀, hγ₀⟩ := H.ridge
    exact ⟨z, component_joined γ₀ (fun t => le_trans hhs (hγ₀ t)), H.older⟩

/-- **Elder merge, open superlevel filtration `{f > h}`.** The connected component of `M` in
`{f > h}` contains a point strictly above `b` if and only if `h < s`. -/
theorem elder_merge_iff_open (H : CapHyp f C M z b s) (h : ℝ) :
    (∃ w ∈ connectedComponentIn {x | h < f x} M, b < f w) ↔ h < s := by
  constructor
  · rintro ⟨w, hw, hbw⟩
    by_contra hhs
    rw [not_lt] at hhs
    have hU : ∀ x ∈ frontier C, x ∉ {x | h < f x} := by
      intro x hx hxU
      have h1 : h < f x := hxU
      have h2 := H.frontier_le x hx
      linarith
    exact absurd (H.component_separated hU hw) (not_le.mpr hbw)
  · intro hhs
    obtain ⟨γ₀, hγ₀⟩ := H.ridge
    exact ⟨z, component_joined γ₀ (fun t => lt_of_lt_of_le hhs (hγ₀ t)), H.older⟩

/-- The set of levels at which the class born at `M` is joined to an older point. -/
def joinedLevels (f : X → ℝ) (M : X) (b : ℝ) : Set ℝ :=
  {h | ∃ w ∈ connectedComponentIn {x | h ≤ f x} M, b < f w}

/-- **Elder death level.** In the closed superlevel filtration, the largest level at which the
component of `M` contains an older point exists and equals `s`, whatever `f` does outside the cap.
This is the death level of the class born at `M` in L22. -/
theorem elder_death_level (H : CapHyp f C M z b s) : IsGreatest (joinedLevels f M b) s :=
  ⟨(H.elder_merge_iff_closed s).mpr le_rfl, fun _ hh => (H.elder_merge_iff_closed _).mp hh⟩

/-! ### The merging saddle -/

/-- Every path from `M` to a point strictly above `b` that stays in `{f ≥ s}` passes through `S`,
when `S` is the only frontier point at height `s` (C2⁺, L19). The statement says something only when
`M ≠ S`, as in the source, where the pins are distinct. -/
theorem path_through_saddle (H : CapHyp f C M z b s) {S : X}
    (hS : ∀ x ∈ frontier C, x ≠ S → f x < s) {w : X} (hw : b < f w) (γ : Path M w)
    (hγ : ∀ t, s ≤ f (γ t)) : ∃ t, γ t = S := by
  have hwC : w ∉ C := fun h => (not_le.mpr hw) (H.ceiling w h)
  obtain ⟨t, ht⟩ := path_meets_frontier γ H.mem hwC
  refine ⟨t, ?_⟩
  by_contra hne
  have h1 := hS _ ht hne
  have h2 := hγ t
  linarith

/-- **`S` is a cut point at the death level.** Assume (C2⁺) and `M ≠ S`. Then `M` lies in its own
component of `{f ≥ s} \ {S}`, so the statement is not vacuous, and that component contains no
point strictly above `b`. By contrast, `{f ≥ s}` itself joins `M` to an older point
(`elder_merge_iff_closed` at `h = s`). Removing `S` therefore disconnects `M` from every older
point at the death level. -/
theorem saddle_cut (H : CapHyp f C M z b s) {S : X}
    (hS : ∀ x ∈ frontier C, x ≠ S → f x < s) (hMS : M ≠ S) :
    M ∈ connectedComponentIn ({x | s ≤ f x} \ {S}) M ∧
      ∀ w ∈ connectedComponentIn ({x | s ≤ f x} \ {S}) M, f w ≤ b := by
  refine ⟨mem_connectedComponentIn ⟨H.s_le_f_M, hMS⟩, fun w hw => ?_⟩
  refine H.component_separated (fun x hx hxU => ?_) hw
  obtain ⟨h1, h2⟩ := hxU
  have h3 := hS x hx h2
  have h4 : s ≤ f x := h1
  linarith

/-! ### Exterior invariance -/

/-- **Exterior invariance.** Any `g` that agrees with `f` on the closure of the cap and along one
ridge path satisfies the same cap hypotheses. Every conclusion above (maximin level, elder merge
levels, death level, saddle cut) therefore holds for `g` with the same `s`, however `g` behaves
elsewhere: remote saddles, high corridors and re-entries change nothing. -/
theorem congr {g : X → ℝ} (H : CapHyp f C M z b s) (γ₀ : Path M z) (hγ₀ : ∀ t, s ≤ f (γ₀ t))
    (hgC : ∀ x ∈ closure C, g x = f x) (hgγ : ∀ t, g (γ₀ t) = f (γ₀ t)) :
    CapHyp g C M z b s where
  mem := H.mem
  ceiling x hx := (hgC x (subset_closure hx)).symm ▸ H.ceiling x hx
  frontier_le x hx := (hgC x (frontier_subset_closure hx)).symm ▸ H.frontier_le x hx
  older := by
    have h1 := hgγ 1
    rw [γ₀.target] at h1
    rw [h1]
    exact H.older
  ridge := ⟨γ₀, fun t => (hgγ t).symm ▸ hγ₀ t⟩

end CapHyp

/-! ### The elder-rule interface on a compact, locally connected space

The elder rule pairs the class born at `M` with the level at which its superlevel component first
contains a strictly higher local maximum (an older class). On a compact, locally connected space
with continuous `f` (a closed manifold, in particular the torus), the following theorems identify
that level as `s`. -/

/-- A connected component of a closed set is closed. -/
theorem isClosed_connectedComponentIn {F : Set X} (hF : IsClosed F) (x : X) :
    IsClosed (connectedComponentIn F x) := by
  by_cases hx : x ∈ F
  · refine isClosed_of_closure_subset ?_
    exact isPreconnected_connectedComponentIn.closure.subset_connectedComponentIn
      (subset_closure (mem_connectedComponentIn hx))
      (closure_minimal (connectedComponentIn_subset F x) hF)
  · rw [connectedComponentIn_eq_empty hx]
    exact isClosed_empty

namespace CapHyp

variable {f : X → ℝ} {C : Set X} {M z : X} {b s : ℝ}

/-- **An older peak below the death level.** For every level `h ≤ s`, the component of `M` in
`{f ≥ h}` contains a point `p` with `f p > b` that maximizes `f` over the component and is a local
maximum of `f` on `X`: the older class that absorbs the class born at `M`. -/
theorem older_peak [CompactSpace X] [LocallyConnectedSpace X] (H : CapHyp f C M z b s)
    (hf : Continuous f) {h : ℝ} (hh : h ≤ s) :
    ∃ p ∈ connectedComponentIn {x | h ≤ f x} M, b < f p ∧ IsLocalMax f p ∧
      ∀ w ∈ connectedComponentIn {x | h ≤ f x} M, f w ≤ f p := by
  have hF : IsClosed {x | h ≤ f x} := isClosed_le continuous_const hf
  obtain ⟨w, hwK, hbw⟩ := (H.elder_merge_iff_closed h).mpr hh
  obtain ⟨p, hpK, hpmax⟩ := (isClosed_connectedComponentIn hF M).isCompact.exists_isMaxOn
    ⟨w, hwK⟩ hf.continuousOn
  have hmax : ∀ x ∈ connectedComponentIn {x | h ≤ f x} M, f x ≤ f p := isMaxOn_iff.mp hpmax
  have hbp : b < f p := lt_of_lt_of_le hbw (hmax w hwK)
  have hsb : s ≤ b := le_trans H.s_le_f_M (H.ceiling M H.mem)
  have hpU : p ∈ {x | h < f x} := by
    show h < f p
    linarith
  have hU : IsOpen {x | h < f x} := isOpen_lt continuous_const hf
  have hVK : connectedComponentIn {x | h < f x} p ⊆ connectedComponentIn {x | h ≤ f x} M := by
    rw [connectedComponentIn_eq hpK]
    exact connectedComponentIn_mono p (fun x (hx : h < f x) => (le_of_lt hx : h ≤ f x))
  refine ⟨p, hpK, hbp, ?_, hmax⟩
  exact Filter.eventually_of_mem (hU.connectedComponentIn.mem_nhds (mem_connectedComponentIn hpU))
    (fun x hx => hmax x (hVK hx))

/-- **No older point above the death level.** For every level `h > s`, `M` is a highest point of
its component in `{f ≥ h}`, when `b = f M`: the class born at `M` is the elder of its component. -/
theorem elder_alive (H : CapHyp f C M z b s) (hMb : f M = b) {h : ℝ} (hh : s < h) :
    ∀ w ∈ connectedComponentIn {x | h ≤ f x} M, f w ≤ f M := by
  intro w hw
  rw [hMb]
  by_contra hlt
  rw [not_le] at hlt
  exact absurd ((H.elder_merge_iff_closed h).mp ⟨w, hw, hlt⟩) (not_le.mpr hh)

/-- The levels at which the component of `M` in `{f ≥ h}` contains a strictly higher local
maximum of `f`. -/
def olderPeakLevels (f : X → ℝ) (M : X) : Set ℝ :=
  {h | ∃ p ∈ connectedComponentIn {x | h ≤ f x} M, f M < f p ∧ IsLocalMax f p}

/-- **Elder death level, peak form.** On a compact, locally connected space with continuous `f`
and `b = f M`, the largest level at which the component of `M` contains a strictly higher local
maximum exists and equals `s`. This is the death level of the elder rule. -/
theorem elder_death_level_peak [CompactSpace X] [LocallyConnectedSpace X]
    (H : CapHyp f C M z b s) (hf : Continuous f) (hMb : f M = b) :
    IsGreatest (olderPeakLevels f M) s := by
  refine ⟨?_, fun h hh => ?_⟩
  · obtain ⟨p, hpK, hbp, hloc, -⟩ := H.older_peak hf le_rfl
    exact ⟨p, hpK, hMb ▸ hbp, hloc⟩
  · obtain ⟨p, hpK, hMp, -⟩ := hh
    exact (H.elder_merge_iff_closed h).mp ⟨p, hpK, hMb ▸ hMp⟩

end CapHyp

/-! ### Charts: the cap on the torus and its faces

The analysis of CAP §§2–5 runs in Euclidean coordinates on the periodic lift `f ∘ e` of a field
`f` on the torus, where `e : ℝ^d → T^d` is the covering map. The two lemmas below transport the cap
hypotheses to the torus and list the faces of the cylinder cap. The transport needs only that `e`
is continuous and open, that the target is Hausdorff and that the cap is compact; injectivity is
not used. -/

/-- Frontier of a compact image. If `e` is continuous and open, `X` is Hausdorff and `C` is
compact, every frontier point of `e '' C` is the image of a frontier point of `C`. -/
theorem frontier_image_subset {Y : Type*} [TopologicalSpace Y] [T2Space X] {e : Y → X}
    (hec : Continuous e) (heo : IsOpenMap e) {C : Set Y} (hC : IsCompact C) :
    frontier (e '' C) ⊆ e '' frontier C := by
  intro x hx
  have hcl : IsClosed (e '' C) := (hC.image hec).isClosed
  have hx1 : x ∈ e '' C := by
    have h := hx.1
    rwa [hcl.closure_eq] at h
  obtain ⟨y, hyC, rfl⟩ := hx1
  refine ⟨y, ⟨subset_closure hyC, fun hyi => hx.2 ?_⟩, rfl⟩
  exact interior_maximal (image_mono interior_subset) (heo _ isOpen_interior) ⟨y, hyi, rfl⟩

namespace CapHyp

/-- **Transport along a chart.** The cap hypotheses for the lift `f ∘ e` on a compact cap `C`
give the cap hypotheses for `f` on `e '' C`, with the same `b` and `s`. -/
theorem map {Y : Type*} [TopologicalSpace Y] [T2Space X] {e : Y → X} (hec : Continuous e)
    (heo : IsOpenMap e) {f : X → ℝ} {C : Set Y} (hC : IsCompact C) {M z : Y} {b s : ℝ}
    (H : CapHyp (f ∘ e) C M z b s) : CapHyp f (e '' C) (e M) (e z) b s where
  mem := ⟨M, H.mem, rfl⟩
  ceiling := by
    rintro _ ⟨y, hy, rfl⟩
    exact H.ceiling y hy
  frontier_le := by
    intro x hx
    obtain ⟨y, hy, rfl⟩ := frontier_image_subset hec heo hC hx
    exact H.frontier_le y hy
  older := H.older
  ridge := by
    obtain ⟨γ, hγ⟩ := H.ridge
    exact ⟨γ.map hec, fun t => hγ t⟩

/-- Transport of (C2⁺): if only `S` attains `s` on the frontier of `C` for the lift, only `e S`
attains it on the frontier of `e '' C`. -/
theorem map_saddle {Y : Type*} [TopologicalSpace Y] [T2Space X] {e : Y → X} (hec : Continuous e)
    (heo : IsOpenMap e) {f : X → ℝ} {C : Set Y} (hC : IsCompact C) {S : Y} {s : ℝ}
    (hS : ∀ y ∈ frontier C, y ≠ S → (f ∘ e) y < s) :
    ∀ x ∈ frontier (e '' C), x ≠ e S → f x < s := by
  intro x hx hne
  obtain ⟨y, hy, rfl⟩ := frontier_image_subset hec heo hC hx
  exact hS y hy (fun h => hne (h ▸ rfl))

end CapHyp

/-- **The faces of the cylinder cap** `[a, c] × B̄(0, R)`: the two end faces `{a, c} × B̄(0, R)`
and the curved side `[a, c] × S(0, R)`. These are the faces estimated in L17–L19. -/
theorem frontier_cylinder {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {a c R : ℝ}
    (hac : a ≤ c) (hR : R ≠ 0) :
    frontier (Icc a c ×ˢ Metric.closedBall (0 : E) R) =
      ({a, c} : Set ℝ) ×ˢ Metric.closedBall (0 : E) R ∪ Icc a c ×ˢ Metric.sphere (0 : E) R := by
  rw [frontier_prod_eq, closure_Icc, frontier_closedBall (0 : E) hR, frontier_Icc hac,
    Metric.isClosed_closedBall.closure_eq]
  exact union_comm _ _

/-- The cylinder cap is compact in finite dimension. -/
theorem isCompact_cylinder {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [ProperSpace E]
    (a c R : ℝ) : IsCompact (Icc a c ×ˢ Metric.closedBall (0 : E) R) :=
  isCompact_Icc.prod (isCompact_closedBall 0 R)

/-- The covering map of the flat torus `(ℝ / Lℤ)^d`. -/
def torusCover (L : ℝ) (d : ℕ) : (Fin d → ℝ) → (Fin d → AddCircle L) :=
  Pi.map fun _ => ((↑) : ℝ → AddCircle L)

theorem torusCover_continuous (L : ℝ) (d : ℕ) : Continuous (torusCover L d) :=
  Continuous.piMap fun _ => continuous_quotient_mk'

theorem torusCover_isOpenMap (L : ℝ) (d : ℕ) : IsOpenMap (torusCover L d) :=
  IsOpenMap.piMap (fun _ => QuotientAddGroup.isOpenMap_coe)
    (Filter.Eventually.of_forall fun _ => QuotientAddGroup.mk_surjective)

/-- **The cap on the torus.** Let `φ` be any coordinate frame (a homeomorphism onto `ℝ^d`), and
suppose the periodic lift `f ∘ torusCover L d ∘ φ` satisfies the cap hypotheses on a compact cap
`C`. Then `f` satisfies them on the image of `C` in the torus `(ℝ / Lℤ)^d`, with the same `b` and
`s`. No embedding radius is used. -/
theorem CapHyp.toTorus {Y : Type*} [TopologicalSpace Y] {L : ℝ} {d : ℕ}
    (φ : Y ≃ₜ (Fin d → ℝ)) {f : (Fin d → AddCircle L) → ℝ} {C : Set Y} (hC : IsCompact C)
    {M z : Y} {b s : ℝ} (H : CapHyp (f ∘ (torusCover L d ∘ φ)) C M z b s) :
    CapHyp f ((torusCover L d ∘ φ) '' C) (torusCover L d (φ M)) (torusCover L d (φ z)) b s :=
  H.map ((torusCover_continuous L d).comp φ.continuous)
    ((torusCover_isOpenMap L d).comp φ.isOpenMap) hC

/-! ### From the ridge to the cap hypotheses (L13, L16, L17, L19, L20)

SP §2 normalizes the pins to `x = a = -r/2` and `x = c = r/2`, on the cap
`C = [-2r, r/2] × B̄(0, 2r)` in coordinates `(x, y) ∈ ℝ × E`. The multi-variable analytic steps
enter as hypotheses, each the output of a named step of SP:
- `hT` (L7): on the cap, `f (x, y) ≤ f (x, h x)`, with `h` the transverse maximizer;
- `hg` (L11): the ridge function `g x = f (x, h x)` has derivative `F x`;
- `hFa`, `hFc` (L13): `F a = F c = 0`, because the pins are critical;
- `hconv` (L12): `F - q` is convex, where `q x = (x - a)(x - c)/8`; this is SP's `F'' ≥ 1/4`;
- `hside` (L18): on the curved side, `f ≤ g c`;
- `hgap`: `g a - g c < 9r³/32`. SP's normalization `g a - g c = r³/6` satisfies it.

From these inputs Lean derives the rest:
- L13: the sign pattern of `F`;
- L16: the ceiling `g ≤ g a` on `[-2r, c]`;
- L17: both exact face integrals `9r³/32`, by comparing `g` with the cubic
  `Q x = x³/24 - r²x/32`;
- L19: the face assembly;
- L20: the ridge path. -/

section Ridge

/-- A convex function that vanishes at `a ≤ c` is `≤ 0` between them. -/
theorem convex_zeros_between {φ : ℝ → ℝ} {D : Set ℝ} (hφ : ConvexOn ℝ D φ) {a c : ℝ}
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a ≤ c) (ha : φ a = 0) (hc : φ c = 0) {x : ℝ}
    (hx : x ∈ Icc a c) : φ x ≤ 0 := by
  have h := hφ.le_on_segment haD hcD (by rw [segment_eq_Icc hac]; exact hx)
  rwa [ha, hc, max_self] at h

/-- A convex function that vanishes at `a < c` is `≥ 0` to the right of `c`. -/
theorem convex_zeros_right {φ : ℝ → ℝ} {D : Set ℝ} (hφ : ConvexOn ℝ D φ) {a c : ℝ}
    (haD : a ∈ D) (hac : a < c) (ha : φ a = 0) (hc : φ c = 0) {x : ℝ} (hxD : x ∈ D)
    (hx : c ≤ x) : 0 ≤ φ x := by
  rcases eq_or_lt_of_le hx with rfl | hlt
  · rw [hc]
  · have h := hφ.slope_mono_adjacent haD hxD hac hlt
    rw [ha, hc, sub_self, zero_div, sub_zero] at h
    rw [le_div_iff₀ (sub_pos.mpr hlt), zero_mul] at h
    exact h

/-- A convex function that vanishes at `a < c` is `≥ 0` to the left of `a`. -/
theorem convex_zeros_left {φ : ℝ → ℝ} {D : Set ℝ} (hφ : ConvexOn ℝ D φ) {a c : ℝ}
    (hcD : c ∈ D) (hac : a < c) (ha : φ a = 0) (hc : φ c = 0) {x : ℝ} (hxD : x ∈ D)
    (hx : x ≤ a) : 0 ≤ φ x := by
  rcases eq_or_lt_of_le hx with rfl | hlt
  · rw [ha]
  · have h := hφ.slope_mono_adjacent hxD hcD hlt hac
    rw [ha, hc, sub_self, zero_div, zero_sub] at h
    rw [div_le_iff₀ (sub_pos.mpr hlt), zero_mul] at h
    linarith

/-- **The ridge profile (L13, L16, L17, L20).** On `D = [-2r, 2r]`, let `g' = F`, with `F`
vanishing at the pins `∓r/2` and `F - (x + r/2)(x - r/2)/8` convex. Then:
- `g ≤ g(-r/2)` on `[-2r, r/2]`;
- `g ≥ g(r/2)` on `[-r/2, 2r]`;
- `g(-2r) ≤ g(-r/2) - 9r³/32`;
- `g(2r) ≥ g(r/2) + 9r³/32`. -/
theorem ridge_profile {g F : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt g (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8)) :
    (∀ x ∈ Icc (-2 * r) (r / 2), g x ≤ g (-r / 2)) ∧
      (∀ x ∈ Icc (-r / 2) (2 * r), g (r / 2) ≤ g x) ∧
      g (-2 * r) ≤ g (-r / 2) - 9 * r ^ 3 / 32 ∧ g (r / 2) + 9 * r ^ 3 / 32 ≤ g (2 * r) := by
  have hac : -r / 2 < r / 2 := by linarith
  have haD : -r / 2 ∈ Icc (-2 * r) (2 * r) := ⟨by linarith, by linarith⟩
  have hcD : r / 2 ∈ Icc (-2 * r) (2 * r) := ⟨by linarith, by linarith⟩
  have hLD : Icc (-2 * r) (-r / 2) ⊆ Icc (-2 * r) (2 * r) := Icc_subset_Icc le_rfl (by linarith)
  have hMD : Icc (-r / 2) (r / 2) ⊆ Icc (-2 * r) (2 * r) :=
    Icc_subset_Icc (by linarith) (by linarith)
  have hRD : Icc (r / 2) (2 * r) ⊆ Icc (-2 * r) (2 * r) := Icc_subset_Icc (by linarith) le_rfl
  set φ : ℝ → ℝ := fun x => F x - (x + r / 2) * (x - r / 2) / 8 with hφdef
  have hφa : φ (-r / 2) = 0 := by simp only [hφdef, hFa]; ring
  have hφc : φ (r / 2) = 0 := by simp only [hφdef, hFc]; ring
  have hgc : ContinuousOn g (Icc (-2 * r) (2 * r)) :=
    fun x hx => (hg x hx).continuousAt.continuousWithinAt
  -- L13: the sign pattern of `F`
  have hFl : ∀ x ∈ Icc (-2 * r) (-r / 2), 0 ≤ F x := by
    intro x hx
    have h1 := convex_zeros_left hconv hcD hac hφa hφc (hLD hx) hx.2
    have h2 : 0 ≤ (x + r / 2) * (x - r / 2) / 8 := by
      have : 0 ≤ (x + r / 2) * (x - r / 2) :=
        mul_nonneg_of_nonpos_of_nonpos (by linarith [hx.2]) (by linarith [hx.2])
      positivity
    simp only [hφdef] at h1
    linarith
  have hFm : ∀ x ∈ Icc (-r / 2) (r / 2), F x ≤ 0 := by
    intro x hx
    have h1 := convex_zeros_between hconv haD hcD hac.le hφa hφc hx
    have h2 : (x + r / 2) * (x - r / 2) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by linarith [hx.1]) (by linarith [hx.2])
    simp only [hφdef] at h1
    linarith
  have hFr : ∀ x ∈ Icc (r / 2) (2 * r), 0 ≤ F x := by
    intro x hx
    have h1 := convex_zeros_right hconv haD hac hφa hφc (hRD hx) hx.1
    have h2 : 0 ≤ (x + r / 2) * (x - r / 2) := mul_nonneg (by linarith [hx.1]) (by linarith [hx.1])
    simp only [hφdef] at h1
    linarith
  -- monotonicity of `g` on the three pieces
  have hmonoL : MonotoneOn g (Icc (-2 * r) (-r / 2)) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc _ _) (hgc.mono hLD)
      (fun x hx => (hg x (hLD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => hFl x (interior_subset hx))
  have hantiM : AntitoneOn g (Icc (-r / 2) (r / 2)) :=
    antitoneOn_of_hasDerivWithinAt_nonpos (convex_Icc _ _) (hgc.mono hMD)
      (fun x hx => (hg x (hMD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => hFm x (interior_subset hx))
  have hmonoR : MonotoneOn g (Icc (r / 2) (2 * r)) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc _ _) (hgc.mono hRD)
      (fun x hx => (hg x (hRD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => hFr x (interior_subset hx))
  -- L17: compare `g` with the cubic `Q`, whose derivative is `(x + r/2)(x - r/2)/8`
  set Q : ℝ → ℝ := fun x => x ^ 3 / 24 - r ^ 2 * x / 32 with hQdef
  have hQ : ∀ x, HasDerivAt Q ((x + r / 2) * (x - r / 2) / 8) x := by
    intro x
    have h1 := ((hasDerivAt_pow 3 x).div_const 24).sub ((hasDerivAt_id x).const_mul (r ^ 2 / 32))
    convert h1 using 1
    · funext y
      simp only [hQdef, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  have hG : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => g x - Q x) (φ x) x :=
    fun x hx => (hg x hx).sub (hQ x)
  have hGc : ContinuousOn (fun x => g x - Q x) (Icc (-2 * r) (2 * r)) :=
    fun x hx => (hG x hx).continuousAt.continuousWithinAt
  have hGL : MonotoneOn (fun x => g x - Q x) (Icc (-2 * r) (-r / 2)) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc _ _) (hGc.mono hLD)
      (fun x hx => (hG x (hLD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => convex_zeros_left hconv hcD hac hφa hφc (hLD (interior_subset hx))
        (interior_subset hx).2)
  have hGR : MonotoneOn (fun x => g x - Q x) (Icc (r / 2) (2 * r)) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc _ _) (hGc.mono hRD)
      (fun x hx => (hG x (hRD (interior_subset hx))).hasDerivWithinAt)
      (fun x hx => convex_zeros_right hconv haD hac hφa hφc (hRD (interior_subset hx))
        (interior_subset hx).1)
  have hLa : (-r / 2 : ℝ) ∈ Icc (-2 * r) (-r / 2) := ⟨by linarith, le_rfl⟩
  have hL2 : (-2 * r : ℝ) ∈ Icc (-2 * r) (-r / 2) := ⟨le_rfl, by linarith⟩
  have hRc : (r / 2 : ℝ) ∈ Icc (r / 2) (2 * r) := ⟨le_rfl, by linarith⟩
  have hR2 : (2 * r : ℝ) ∈ Icc (r / 2) (2 * r) := ⟨by linarith, le_rfl⟩
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro x hx
    rcases le_or_gt x (-r / 2) with h | h
    · exact hmonoL ⟨hx.1, h⟩ hLa h
    · exact hantiM ⟨le_rfl, hac.le⟩ ⟨h.le, hx.2⟩ h.le
  · intro x hx
    rcases le_or_gt x (r / 2) with h | h
    · exact hantiM ⟨hx.1, h⟩ ⟨hac.le, le_rfl⟩ h
    · exact hmonoR hRc ⟨h.le, hx.2⟩ h.le
  · have h := hGL hL2 hLa (by linarith)
    simp only [hQdef] at h
    nlinarith [h]
  · have h := hGR hRc hR2 (by linarith)
    simp only [hQdef] at h
    nlinarith [h]

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **The cap hypotheses from the ridge (L16, L17, L19, L20).** Under the analytic inputs listed
above, `f` satisfies (C1)–(C3) on `C = [-2r, r/2] × B̄(0, 2r)`, with `M = (-r/2, h(-r/2))`,
`z = (2r, h(2r))`, `b = g(-r/2)` and `s = g(r/2)`. Every input is local: `g' = F` and the convexity
hold on `[-2r, 2r]`, and `h` is continuous on `[-r/2, 2r]`. -/
theorem ridge_capHyp {f : ℝ × E → ℝ} {h : ℝ → E} {F : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hT : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), f p ≤ f (p.1, h p.1))
    (hside : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.sphere (0 : E) (2 * r),
      f p ≤ f (r / 2, h (r / 2)))
    (hM : ‖h (-r / 2)‖ ≤ 2 * r)
    (hgap : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) < 9 * r ^ 3 / 32) :
    CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, h (-r / 2))
      (2 * r, h (2 * r)) (f (-r / 2, h (-r / 2))) (f (r / 2, h (r / 2))) := by
  obtain ⟨hceil, hfloor, hleft, hright⟩ := ridge_profile hr hg hFa hFc hconv
  have hcyl : frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) =
      ({-2 * r, r / 2} : Set ℝ) ×ˢ Metric.closedBall (0 : E) (2 * r) ∪
        Icc (-2 * r) (r / 2) ×ˢ Metric.sphere (0 : E) (2 * r) :=
    frontier_cylinder (by linarith) (by positivity)
  set xt : I → ℝ := fun t => -r / 2 + (t : ℝ) * (2 * r - -r / 2) with hxt
  have hxc : Continuous xt := continuous_const.add (continuous_subtype_val.mul continuous_const)
  have hx0 : xt 0 = -r / 2 := by simp [hxt]
  have hx1 : xt 1 = 2 * r := by simp [hxt]
  have hxmem : ∀ t, xt t ∈ Icc (-r / 2) (2 * r) := by
    intro t
    have h0 := t.2.1
    have h1 := t.2.2
    simp only [hxt]
    constructor <;> nlinarith
  let γ : Path ((-r / 2 : ℝ), h (-r / 2)) (2 * r, h (2 * r)) :=
    { toFun := fun t => (xt t, h (xt t))
      continuous_toFun := hxc.prodMk (hh.comp_continuous hxc hxmem)
      source' := by simp only [hx0]
      target' := by simp only [hx1] }
  refine ⟨⟨⟨by linarith, by linarith⟩, mem_closedBall_zero_iff.mpr hM⟩, ?_, ?_, ?_, ⟨γ, ?_⟩⟩
  · intro p hp
    exact (hT p hp).trans (hceil p.1 hp.1)
  · intro p hp
    rw [hcyl] at hp
    rcases hp with ⟨hp1, hp2⟩ | hp
    · have hpC : p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r) := by
        refine ⟨?_, hp2⟩
        rcases hp1 with h1 | h1
        · rw [h1]; exact ⟨le_refl _, by linarith⟩
        · rw [h1]; exact ⟨by linarith, le_refl _⟩
      have h0 := hT p hpC
      rcases hp1 with h1 | h1
      · rw [h1] at h0
        linarith
      · rw [h1] at h0
        exact h0
    · exact hside p hp
  · linarith
  · intro t
    exact hfloor (xt t) (hxmem t)

/-- **(C2⁺) from the ridge.** If, in addition, `h x` is the unique transverse maximizer on the
cap and the curved side is strictly below `g c` (L7, L18), then `S = (r/2, h(r/2))` is the only
frontier point at height `s`. -/
theorem ridge_saddle_strict {f : ℝ × E → ℝ} {h : ℝ → E} {F : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hT : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), f p ≤ f (p.1, h p.1))
    (hTs : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), p.2 ≠ h p.1 →
      f p < f (p.1, h p.1))
    (hside : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.sphere (0 : E) (2 * r),
      f p < f (r / 2, h (r / 2)))
    (hgap : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) < 9 * r ^ 3 / 32) :
    ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
      x ≠ (r / 2, h (r / 2)) → f x < f (r / 2, h (r / 2)) := by
  obtain ⟨-, -, hleft, -⟩ := ridge_profile hr hg hFa hFc hconv
  intro p hp hne
  rw [frontier_cylinder (by linarith) (by positivity)] at hp
  rcases hp with ⟨hp1, hp2⟩ | hp
  · rcases hp1 with h1 | h1
    · have hpC : p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
        ⟨by rw [h1]; exact ⟨le_refl _, by linarith⟩, hp2⟩
      have h0 := hT p hpC
      rw [h1] at h0
      linarith
    · have hpC : p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
        ⟨by rw [h1]; exact ⟨by linarith, le_refl _⟩, hp2⟩
      have hy : p.2 ≠ h p.1 := by
        intro hy
        apply hne
        rw [h1] at hy
        exact Prod.ext h1 hy
      have h0 := hTs p hpC hy
      rw [h1] at h0
      exact h0
  · exact hside p hp

end Ridge

/-! ### The transverse slices (L7, L8, L11, L18)

`ridge_capHyp` takes the transverse maximum (`hT`), its uniqueness (`hTs`), the curved side
(`hside`) and the ridge equation `g' = F` as inputs. SP derives them deterministically from L4
(each slice `y ↦ f (x, y)` is `δ`-strongly concave on `B̄(0, 2r)`), L5 (the transverse derivative
`w x` at `y = 0` has norm at most `ω`) and the ridge equation `∂_y f (x, h x) = 0`. This section
proves those derivations. -/

section Slice

open Filter Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **Strong concavity about a point with a derivative (the engine of L7, L8, L18).** If `φ` is
`δ`-strongly concave on `K`, `p ∈ K`, and `φ` has derivative `L` at `p`, then
`φ y ≤ φ p + L (y - p) - (δ/2)‖y - p‖²` for every `y ∈ K`. `p` need not be interior: only the
segment from `p` to `y`, which lies in `K`, is used. -/
theorem strongConcave_le_of_hasFDerivAt {K : Set E} {φ : E → ℝ} {δ : ℝ} {p : E}
    {L : E →L[ℝ] ℝ} (hφ : StrongConcaveOn K δ φ) (hp : p ∈ K) (hd : HasFDerivAt φ L p) {y : E}
    (hy : y ∈ K) : φ y ≤ φ p + L (y - p) - δ / 2 * ‖y - p‖ ^ 2 := by
  set v := y - p with hv
  have h1 : HasDerivAt (fun t : ℝ => p + t • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add p
  have h2 : HasFDerivAt φ L (p + (0 : ℝ) • v) := by simpa using hd
  have hline : HasDerivAt (fun t : ℝ => φ (p + t • v)) (L v) 0 := h2.comp_hasDerivAt (0 : ℝ) h1
  have hslope := hline.tendsto_slope_zero_right
  have hbound : ∀ᶠ t in 𝓝[>] (0 : ℝ),
      φ y - φ p + (1 - t) * (δ / 2 * ‖v‖ ^ 2) ≤
        t⁻¹ • (φ (p + (0 + t) • v) - φ (p + (0 : ℝ) • v)) := by
    have hlt : ∀ᶠ t in 𝓝[>] (0 : ℝ), t < 1 :=
      nhdsWithin_le_nhds (Iio_mem_nhds one_pos)
    filter_upwards [self_mem_nhdsWithin, hlt] with t (ht0 : 0 < t) ht1
    have key := hφ.2 hp hy (by linarith : (0 : ℝ) ≤ 1 - t) ht0.le (by ring : 1 - t + t = 1)
    have hpt : (1 - t) • p + t • y = p + t • v := by
      simp only [hv, smul_sub, sub_smul, one_smul]; abel
    have hnorm : ‖p - y‖ = ‖v‖ := by rw [hv, norm_sub_rev]
    rw [hpt, hnorm, smul_eq_mul, smul_eq_mul] at key
    simp only [zero_smul, add_zero, zero_add, smul_eq_mul]
    rw [le_inv_mul_iff₀ ht0]
    nlinarith [key]
  have hlim : Tendsto (fun t : ℝ => φ y - φ p + (1 - t) * (δ / 2 * ‖v‖ ^ 2)) (𝓝[>] 0)
      (𝓝 (φ y - φ p + (1 - 0) * (δ / 2 * ‖v‖ ^ 2))) :=
    ((continuous_const.add ((continuous_const.sub continuous_id).mul continuous_const)).tendsto
      0).mono_left nhdsWithin_le_nhds
  have := le_of_tendsto_of_tendsto hlim hslope hbound
  simp only [sub_zero, one_mul] at this
  linarith

/-- **L7 (the transverse maximum).** At a critical point `p` of a `δ`-strongly concave slice,
`φ y ≤ φ p - (δ/2)‖y - p‖²` on `K`. With `δ > 0` the maximizer is unique. -/
theorem slice_le {K : Set E} {φ : E → ℝ} {δ : ℝ} {p : E}
    (hφ : StrongConcaveOn K δ φ) (hp : p ∈ K) (hd : HasFDerivAt φ (0 : E →L[ℝ] ℝ) p)
    {y : E} (hy : y ∈ K) : φ y ≤ φ p - δ / 2 * ‖y - p‖ ^ 2 := by
  simpa using strongConcave_le_of_hasFDerivAt hφ hp hd hy

/-- **L8 (the ridge bound).** If `0, p ∈ K`, `p` is critical and `w` is the derivative at `0`,
then `δ ‖p‖ ≤ ‖w‖`, i.e. `‖h(x)‖ ≤ ‖w(x)‖/δ`. -/
theorem slice_ridge_bound {K : Set E} {φ : E → ℝ} {δ : ℝ} {p : E} {w : E →L[ℝ] ℝ}
    (hφ : StrongConcaveOn K δ φ) (h0 : (0 : E) ∈ K) (hp : p ∈ K)
    (hd : HasFDerivAt φ (0 : E →L[ℝ] ℝ) p) (hw : HasFDerivAt φ w 0) : δ * ‖p‖ ≤ ‖w‖ := by
  have h1 := slice_le hφ hp hd h0
  have h2 := strongConcave_le_of_hasFDerivAt hφ h0 hw hp
  simp only [zero_sub, norm_neg, sub_zero] at h1 h2
  have h3 : w p ≤ ‖w‖ * ‖p‖ :=
    (le_abs_self _).trans (by simpa [Real.norm_eq_abs] using w.le_opNorm p)
  have h4 : δ * ‖p‖ * ‖p‖ ≤ ‖w‖ * ‖p‖ := by nlinarith
  rcases (norm_nonneg p).eq_or_lt with h | h
  · rw [← h, mul_zero]; exact norm_nonneg w
  · exact le_of_mul_le_mul_right h4 h

/-- **L18 (the curved side).** On the sphere of radius `R`, the slice is strictly below `s`
once `φ p ≤ b`, `‖w‖ ≤ ω ≤ Rδ` and `2δ(b - s) < (Rδ - ω)²`. -/
theorem slice_curved {φ : E → ℝ} {δ ω R b s : ℝ} {p : E} {w : E →L[ℝ] ℝ}
    (hφ : StrongConcaveOn (Metric.closedBall (0 : E) R) δ φ) (hδ : 0 < δ)
    (hp : p ∈ Metric.closedBall (0 : E) R) (hd : HasFDerivAt φ (0 : E →L[ℝ] ℝ) p)
    (hw : HasFDerivAt φ w 0) (hω : ‖w‖ ≤ ω) (hωR : ω ≤ R * δ) (hb : φ p ≤ b)
    (hcurv : 2 * δ * (b - s) < (R * δ - ω) ^ 2) {y : E} (hy : y ∈ Metric.sphere (0 : E) R) :
    φ y < s := by
  have hyn : ‖y‖ = R := mem_sphere_zero_iff_norm.mp hy
  have hR : 0 ≤ R := hyn ▸ norm_nonneg y
  have h0 : (0 : E) ∈ Metric.closedBall (0 : E) R := Metric.mem_closedBall_self hR
  have hL8 := slice_ridge_bound hφ h0 hp hd hw
  have hL7 := slice_le hφ hp hd (Metric.sphere_subset_closedBall hy)
  have htri : R - ‖p‖ ≤ ‖y - p‖ := by have := norm_sub_norm_le y p; linarith
  have hρ : R * δ - ω ≤ δ * ‖y - p‖ := by nlinarith
  have hsq : (R * δ - ω) ^ 2 ≤ (δ * ‖y - p‖) ^ 2 := pow_le_pow_left₀ (by linarith) hρ 2
  have h5 : δ * (2 * (b - s)) < δ * (δ * ‖y - p‖ ^ 2) := by nlinarith
  have h6 : 2 * (b - s) < δ * ‖y - p‖ ^ 2 := lt_of_mul_lt_mul_left h5 hδ.le
  linarith

/-- The derivative of a slice `y ↦ f (x, y)` is the joint derivative restricted to the
transverse directions. -/
theorem slice_hasFDerivAt {f : ℝ × E → ℝ} {x : ℝ} {y : E} {D : ℝ × E →L[ℝ] ℝ}
    (hf : HasFDerivAt f D (x, y)) :
    HasFDerivAt (fun y' => f (x, y')) (D.comp (ContinuousLinearMap.inr ℝ ℝ E)) y :=
  hf.comp y (hasFDerivAt_prodMk_right x y)

/-- **L11 (the ridge function's derivative).** If `f` has joint derivative `D` at `(x, h x)`,
`h` has derivative `h'` at `x`, and the transverse part of `D` vanishes (the ridge equation
`∂_y f = 0`), then `g = f (·, h ·)` has derivative `F x = D (1, 0) = ∂_x f (x, h x)`: the
`∂_y f · h'` term drops out. -/
theorem ridge_hasDerivAt {f : ℝ × E → ℝ} {h : ℝ → E} {x : ℝ} {D : ℝ × E →L[ℝ] ℝ} {h' : E}
    (hf : HasFDerivAt f D (x, h x)) (hh : HasDerivAt h h' x)
    (hy : D.comp (ContinuousLinearMap.inr ℝ ℝ E) = 0) :
    HasDerivAt (fun t => f (t, h t)) (D (1, 0)) x := by
  have hp : HasDerivAt (fun t => (t, h t)) ((1 : ℝ), h') x := (hasDerivAt_id x).prodMk hh
  have hc := hf.comp_hasDerivAt x hp
  have h0 : D (0, h') = 0 := by
    simpa using congrArg (fun L : E →L[ℝ] ℝ => L h') hy
  have hD : D (1, h') = D (1, 0) := by
    calc D (1, h') = D ((1, 0) + (0, h')) := by simp
      _ = D (1, 0) + D (0, h') := map_add _ _ _
      _ = D (1, 0) := by rw [h0, add_zero]
  rw [← hD]
  exact hc

/-- **(C1)–(C3) and (C2⁺) from the slices.** `ridge_capHyp` and `ridge_saddle_strict` with
`hT`, `hTs`, `hside` and `hM` derived (L7, L8, L18) from L4 (`hconc`), the critical points
`h x ∈ B̄(0, 2r)` (`hball`, `hcrit`; L7's existence part), L5 (`hw`, `hω`) and the curvature
condition `2δ(b - s) < (2rδ - ω)²`, which SP's constants satisfy (`sp_curved_side_constants`). -/
theorem ridge_capHyp_of_slices {f : ℝ × E → ℝ} {h : ℝ → E} {F : ℝ → ℝ} {w : ℝ → E →L[ℝ] ℝ}
    {r δ ω : ℝ} (hr : 0 < r) (hδ : 0 < δ)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hconc : ∀ x ∈ Icc (-2 * r) (r / 2),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hball : ∀ x ∈ Icc (-2 * r) (r / 2), h x ∈ Metric.closedBall (0 : E) (2 * r))
    (hcrit : ∀ x ∈ Icc (-2 * r) (r / 2),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x))
    (hw : ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (w x) (0 : E))
    (hω : ∀ x ∈ Icc (-2 * r) (r / 2), ‖w x‖ ≤ ω) (hωr : ω ≤ 2 * r * δ)
    (hcurv : 2 * δ * (f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2))) < (2 * r * δ - ω) ^ 2)
    (hgap : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) < 9 * r ^ 3 / 32) :
    CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, h (-r / 2))
        (2 * r, h (2 * r)) (f (-r / 2, h (-r / 2))) (f (r / 2, h (r / 2))) ∧
      ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
        x ≠ (r / 2, h (r / 2)) → f x < f (r / 2, h (r / 2)) := by
  obtain ⟨hceil, -, -, -⟩ := ridge_profile hr hg hFa hFc hconv
  have hT : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      f p ≤ f (p.1, h p.1) := by
    rintro ⟨x, y⟩ ⟨hx, hy⟩
    have h1 := slice_le (hconc x hx) (hball x hx) (hcrit x hx) hy
    have h2 : 0 ≤ δ / 2 * ‖y - h x‖ ^ 2 := by positivity
    simp only at h1 ⊢
    linarith
  have hTs : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), p.2 ≠ h p.1 →
      f p < f (p.1, h p.1) := by
    rintro ⟨x, y⟩ ⟨hx, hy⟩ hne
    have h1 := slice_le (hconc x hx) (hball x hx) (hcrit x hx) hy
    have h2 : 0 < ‖y - h x‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hne)
    have h3 : 0 < δ / 2 * ‖y - h x‖ ^ 2 := mul_pos (by linarith) (pow_pos h2 2)
    simp only at h1 ⊢
    linarith
  have hside : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.sphere (0 : E) (2 * r),
      f p < f (r / 2, h (r / 2)) := by
    rintro ⟨x, y⟩ ⟨hx, hy⟩
    exact slice_curved (hconc x hx) hδ (hball x hx) (hcrit x hx) (hw x hx) (hω x hx) hωr
      (hceil x hx) hcurv hy
  have hM : ‖h (-r / 2)‖ ≤ 2 * r := by
    simpa using hball (-r / 2) ⟨by linarith, by linarith⟩
  exact ⟨ridge_capHyp hr hh hg hFa hFc hconv hT (fun p hp => (hside p hp).le) hM hgap,
    ridge_saddle_strict hr hg hFa hFc hconv hT hTs hside hgap⟩

/-- **The joint form (L7, L11).** From a joint derivative `D x` of `f` along the ridge, a
differentiable `h` and the ridge equation `(D x) ∘ inr = 0`, the inputs `hh`, `hg` (with
`F x = D x (1, 0)`) and `hcrit` of `ridge_capHyp_of_slices` follow. -/
theorem ridge_inputs_of_joint {f : ℝ × E → ℝ} {h h' : ℝ → E} {D : ℝ → ℝ × E →L[ℝ] ℝ} {r : ℝ}
    (hr : 0 < r) (hD : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt f (D x) (x, h x))
    (hh' : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt h (h' x) x)
    (hridge : ∀ x ∈ Icc (-2 * r) (2 * r), (D x).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0) :
    ContinuousOn h (Icc (-r / 2) (2 * r)) ∧
      (∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (D x (1, 0)) x) ∧
      ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := by
  refine ⟨fun x hx => ?_, fun x hx => ridge_hasDerivAt (hD x hx) (hh' x hx) (hridge x hx),
    fun x hx => ?_⟩
  · exact (hh' x ⟨by linarith [hx.1], hx.2⟩).continuousAt.continuousWithinAt
  · have hx' : x ∈ Icc (-2 * r) (2 * r) := ⟨hx.1, by linarith [hx.2]⟩
    simpa [hridge x hx'] using slice_hasFDerivAt (hD x hx')

/-- **SP's constants satisfy the curvature condition (L4, L5, L18).** With `m ≥ 2`,
`δ > r m (8m - 5)` (L4, `k = 8m - 5`), `0 ≤ ω ≤ 2 m r²` (L5) and SP's gap `b - s = r³/6`:
`ω ≤ 2rδ` and `2δ · r³/6 < (2rδ - ω)²`. -/
theorem sp_curved_side_constants {r m δ ω : ℝ} (hr : 0 < r) (hm : 2 ≤ m)
    (hδ : r * m * (8 * m - 5) < δ) (hω0 : 0 ≤ ω) (hω : ω ≤ 2 * m * r ^ 2) :
    ω ≤ 2 * r * δ ∧ 2 * δ * (r ^ 3 / 6) < (2 * r * δ - ω) ^ 2 := by
  have hk : 11 ≤ 8 * m - 5 := by linarith
  have hrm : 0 < r * m := by positivity
  have h11 : 11 * (r * m) < δ := by nlinarith
  have h22 : 22 * r < δ := by nlinarith
  have hωδ : 11 * ω < 2 * r * δ := by nlinarith
  have hpos : 20 * r * δ < 11 * (2 * r * δ - ω) := by nlinarith
  refine ⟨by linarith, ?_⟩
  have hsq : (20 * r * δ) ^ 2 < (11 * (2 * r * δ - ω)) ^ 2 :=
    pow_lt_pow_left₀ hpos (by have hδ0 : 0 < δ := by linarith
                              positivity) two_ne_zero
  nlinarith [hsq]

/-- Adding the affine function `(m/2)((t - a)(t - c) - t²)` preserves convexity. -/
theorem convexOn_add_node {u : ℝ → ℝ} {D : Set ℝ} {a c m : ℝ}
    (hcv : ConvexOn ℝ D (fun t => u t + m / 2 * t ^ 2)) :
    ConvexOn ℝ D (fun t => u t + m / 2 * ((t - a) * (t - c))) := by
  refine ⟨hcv.1, fun x hx y hy p q hp hq hpq => ?_⟩
  have h := hcv.2 hx hy hp hq hpq
  simp only [smul_eq_mul] at h ⊢
  have key : (u (p * x + q * y) + m / 2 * ((p * x + q * y - a) * (p * x + q * y - c))) -
      (p * (u x + m / 2 * ((x - a) * (x - c))) + q * (u y + m / 2 * ((y - a) * (y - c)))) =
      (u (p * x + q * y) + m / 2 * (p * x + q * y) ^ 2) -
      (p * (u x + m / 2 * x ^ 2) + q * (u y + m / 2 * y ^ 2)) := by
    have hq' : q = 1 - p := by linarith
    subst hq'
    ring
  linarith

/-- **L5 (two-node interpolation), scalar form.** If `u + (m/2) t²` is convex and `u - (m/2) t²` is
concave on `D` (the weak form of `|u''| ≤ m`) and `u` vanishes at `a < c` in `D`, then
`|u x| ≤ (m/2) |(x - a)(x - c)|` for every `x ∈ D`, inside or outside `[a, c]`. -/
theorem two_node_bound {u : ℝ → ℝ} {D : Set ℝ} {a c m : ℝ}
    (hcv : ConvexOn ℝ D (fun t => u t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ D (fun t => u t - m / 2 * t ^ 2))
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a < c) (ha : u a = 0) (hc : u c = 0) {x : ℝ}
    (hx : x ∈ D) : |u x| ≤ m / 2 * |(x - a) * (x - c)| := by
  have hψ := convexOn_add_node (a := a) (c := c) hcv
  have hcv' : ConvexOn ℝ D (fun t => -u t + m / 2 * t ^ 2) :=
    hcc.neg.congr fun t _ => by simp only [Pi.neg_apply]; ring
  have hχ := convexOn_add_node (a := a) (c := c) hcv'
  have hψa : u a + m / 2 * ((a - a) * (a - c)) = 0 := by rw [ha]; ring
  have hψc : u c + m / 2 * ((c - a) * (c - c)) = 0 := by rw [hc]; ring
  have hχa : -u a + m / 2 * ((a - a) * (a - c)) = 0 := by rw [ha]; ring
  have hχc : -u c + m / 2 * ((c - a) * (c - c)) = 0 := by rw [hc]; ring
  rcases le_or_gt x a with hxa | hxa
  · have h1 := convex_zeros_left hψ hcD hac hψa hψc hx hxa
    have h2 := convex_zeros_left hχ hcD hac hχa hχc hx hxa
    have hP : 0 ≤ (x - a) * (x - c) := mul_nonneg_of_nonpos_of_nonpos (by linarith) (by linarith)
    rw [abs_of_nonneg hP, abs_le]
    constructor <;> linarith
  rcases le_or_gt c x with hcx | hcx
  · have h1 := convex_zeros_right hψ haD hac hψa hψc hx hcx
    have h2 := convex_zeros_right hχ haD hac hχa hχc hx hcx
    have hP : 0 ≤ (x - a) * (x - c) := mul_nonneg (by linarith) (by linarith)
    rw [abs_of_nonneg hP, abs_le]
    constructor <;> linarith
  · have h1 := convex_zeros_between hψ haD hcD hac.le hψa hψc ⟨hxa.le, hcx.le⟩
    have h2 := convex_zeros_between hχ haD hcD hac.le hχa hχc ⟨hxa.le, hcx.le⟩
    have hP : (x - a) * (x - c) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
    rw [abs_of_nonpos hP, abs_le]
    constructor <;> linarith

/-- **L5 (the transverse gradient at `y = 0`).** If the slice derivatives `w t` at `y = 0` vanish at
the pins `a < c` and, for every direction `v`, `t ↦ w t v` has second derivative bounded by
`m ‖v‖` in the weak form below, then `‖w x‖ ≤ (m/2) |(x - a)(x - c)|` on `D`. -/
theorem slice_gradient_bound {w : ℝ → E →L[ℝ] ℝ} {D : Set ℝ} {a c m : ℝ} (hm : 0 ≤ m)
    (hcv : ∀ v : E, ConvexOn ℝ D (fun t => w t v + m * ‖v‖ / 2 * t ^ 2))
    (hcc : ∀ v : E, ConcaveOn ℝ D (fun t => w t v - m * ‖v‖ / 2 * t ^ 2))
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a < c) (ha : w a = 0) (hc : w c = 0) {x : ℝ}
    (hx : x ∈ D) : ‖w x‖ ≤ m / 2 * |(x - a) * (x - c)| := by
  refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun v => ?_
  have h := two_node_bound (u := fun t => w t v) (m := m * ‖v‖) (hcv v) (hcc v) haD hcD hac
    (by simp [ha]) (by simp [hc]) hx
  rw [Real.norm_eq_abs]
  calc |w x v| ≤ m * ‖v‖ / 2 * |(x - a) * (x - c)| := h
    _ = m / 2 * |(x - a) * (x - c)| * ‖v‖ := by ring

/-- **The cap from SP's normalized hypotheses (L2, L4, L5, L7, L11, L12).** With `m ≥ 2` (L2),
slices `δ`-strongly concave for some `δ > r m (8m - 5)` (L4), slice derivatives at `y = 0` with
second `x`-derivative bounded by `m` in the weak form (L5), critical points `h x` with
`h(∓r/2) = 0` (L7) and the gap `b - s ≤ r³/6`, the conclusions of `ridge_capHyp_of_slices` hold.
`hω`, `hωr` and `hcurv` are derived: `‖w‖ ≤ 2 m r²` (`slice_gradient_bound`) and
`sp_curved_side_constants`. -/
theorem ridge_capHyp_sp {f : ℝ × E → ℝ} {h : ℝ → E} {F : ℝ → ℝ} {w : ℝ → E →L[ℝ] ℝ}
    {r m δ : ℝ} (hr : 0 < r) (hm : 2 ≤ m) (hδ : r * m * (8 * m - 5) < δ)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hconc : ∀ x ∈ Icc (-2 * r) (r / 2),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hball : ∀ x ∈ Icc (-2 * r) (r / 2), h x ∈ Metric.closedBall (0 : E) (2 * r))
    (hcrit : ∀ x ∈ Icc (-2 * r) (r / 2),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x))
    (hha : h (-r / 2) = 0) (hhc : h (r / 2) = 0)
    (hw : ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (w x) (0 : E))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r) (r / 2)) (fun t => w t v + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r) (r / 2)) (fun t => w t v - m * ‖v‖ / 2 * t ^ 2))
    (hgap : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) ≤ r ^ 3 / 6) :
    CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, h (-r / 2))
        (2 * r, h (2 * r)) (f (-r / 2, h (-r / 2))) (f (r / 2, h (r / 2))) ∧
      ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
        x ≠ (r / 2, h (r / 2)) → f x < f (r / 2, h (r / 2)) := by
  have hm0 : 0 ≤ m := by linarith
  have hδ0 : 0 < δ := by
    have : 0 < r * m * (8 * m - 5) := by
      have : 0 < 8 * m - 5 := by linarith
      positivity
    linarith
  have ha : -r / 2 ∈ Icc (-2 * r) (r / 2) := ⟨by linarith, by linarith⟩
  have hc : r / 2 ∈ Icc (-2 * r) (r / 2) := ⟨by linarith, le_rfl⟩
  have hwa : w (-r / 2) = 0 := (hw _ ha).unique (by simpa [hha] using hcrit _ ha)
  have hwc : w (r / 2) = 0 := (hw _ hc).unique (by simpa [hhc] using hcrit _ hc)
  have hω : ∀ x ∈ Icc (-2 * r) (r / 2), ‖w x‖ ≤ 2 * m * r ^ 2 := by
    intro x hx
    have h1 := slice_gradient_bound hm0 hwcv hwcc ha hc (by linarith) hwa hwc hx
    have h2 : |(x - -r / 2) * (x - r / 2)| ≤ 4 * r ^ 2 := by
      rw [abs_le]
      constructor <;> nlinarith [hx.1, hx.2]
    calc ‖w x‖ ≤ m / 2 * |(x - -r / 2) * (x - r / 2)| := h1
      _ ≤ m / 2 * (4 * r ^ 2) := by gcongr
      _ = 2 * m * r ^ 2 := by ring
  obtain ⟨hωr, hcurv⟩ := sp_curved_side_constants hr hm hδ (by positivity) le_rfl
  refine ridge_capHyp_of_slices hr hδ0 hh hg hFa hFc hconv hconc hball hcrit hw hω hωr ?_ ?_
  · calc 2 * δ * (f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2))) ≤ 2 * δ * (r ^ 3 / 6) := by
          gcongr
      _ < _ := hcurv
  · have : r ^ 3 / 6 < 9 * r ^ 3 / 32 := by
      have : 0 < r ^ 3 := by positivity
      linarith
    linarith

/-- **L1 (the Hermite identity, as a bound).** If `g' = G` on `[a, c]`, `G` vanishes at both ends
(the pins are critical) and `|G''| ≤ m` in the weak form, then `g(a) - g(c) ≤ m (c - a)³/12`.
SP's identity `g(c) - g(a) = -(1/2)∫ φ g'''` with `∫ φ = (c - a)³/6` gives the same bound; here it
comes from `two_node_bound` and monotonicity of `g + (m/2)Φ`, `Φ' = (t - a)(c - t)`. -/
theorem hermite_gap_le {g G : ℝ → ℝ} {a c m : ℝ} (hac : a < c)
    (hg : ∀ x ∈ Icc a c, HasDerivAt g (G x) x)
    (hcv : ConvexOn ℝ (Icc a c) (fun t => G t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ (Icc a c) (fun t => G t - m / 2 * t ^ 2))
    (hGa : G a = 0) (hGc : G c = 0) :
    g a - g c ≤ m * (c - a) ^ 3 / 12 := by
  have hpoly : ∀ x, HasDerivAt (fun t => (c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
      ((x - a) * (c - x)) x := by
    intro x
    have h1 : HasDerivAt (fun t => t - a) 1 x := (hasDerivAt_id x).sub_const a
    have h2 := ((h1.pow 2).const_mul (c - a)).div_const 2
    have h3 := (h1.pow 3).div_const 3
    convert h2.sub h3 using 1
    push_cast
    ring
  set k : ℝ → ℝ := fun t => g t + m / 2 * ((c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
    with hk
  have hkd : ∀ x ∈ Icc a c, HasDerivAt k (G x + m / 2 * ((x - a) * (c - x))) x :=
    fun x hx => (hg x hx).add ((hpoly x).const_mul (m / 2))
  have hmono : MonotoneOn k (Icc a c) := by
    refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc a c)
      (fun x hx => (hkd x hx).continuousAt.continuousWithinAt)
      (fun x hx => (hkd x (interior_subset hx)).hasDerivWithinAt) (fun x hx => ?_)
    have hx := interior_subset hx
    have hb := two_node_bound hcv hcc (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac
      hGa hGc hx
    have hP : (x - a) * (x - c) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by linarith [hx.1]) (by linarith [hx.2])
    rw [abs_of_nonpos hP] at hb
    have := (abs_le.mp hb).1
    nlinarith
  have hkac := hmono (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac.le
  simp only [hk] at hkac
  have e0 : m / 2 * ((c - a) * (a - a) ^ 2 / 2 - (a - a) ^ 3 / 3) = 0 := by ring
  have e1 : m / 2 * ((c - a) * (c - a) ^ 2 / 2 - (c - a) ^ 3 / 3) = m * (c - a) ^ 3 / 12 := by
    ring
  linarith

/-- **L2 (`m ≥ 2`).** Under SP's normalization `g(-r/2) - g(r/2) = r³/6` on the pin interval, the
weak bound `|g'''| ≤ m` forces `m ≥ 2`. -/
theorem hermite_m_ge_two {g G : ℝ → ℝ} {r m : ℝ} (hr : 0 < r)
    (hg : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt g (G x) x)
    (hcv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G t - m / 2 * t ^ 2))
    (hGa : G (-r / 2) = 0) (hGc : G (r / 2) = 0) (hgap : g (-r / 2) - g (r / 2) = r ^ 3 / 6) :
    2 ≤ m := by
  have h := hermite_gap_le (by linarith) hg hcv hcc hGa hGc
  have hcr : r / 2 - -r / 2 = r := by ring
  rw [hgap, hcr] at h
  have hr3 : 0 < r ^ 3 := by positivity
  nlinarith

/-- **The cap from SP's normalization (L1, L2 and the rest of `ridge_capHyp_sp`).** As
`ridge_capHyp_sp`, with `m ≥ 2` derived instead of assumed: `g0 = f(·, 0)` has derivative `G0`
vanishing at the pins with `|G0''| ≤ m` (weak form), and the gap is exactly `r³/6`. -/
theorem ridge_capHyp_normalized {f : ℝ × E → ℝ} {h : ℝ → E} {F G0 : ℝ → ℝ}
    {w : ℝ → E →L[ℝ] ℝ} {r m δ : ℝ} (hr : 0 < r)
    (hg0 : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt (fun x => f (x, 0)) (G0 x) x)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G0 t + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G0 t - m / 2 * t ^ 2))
    (hG0a : G0 (-r / 2) = 0) (hG0c : G0 (r / 2) = 0)
    (hδ : r * m * (8 * m - 5) < δ)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hconc : ∀ x ∈ Icc (-2 * r) (r / 2),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hball : ∀ x ∈ Icc (-2 * r) (r / 2), h x ∈ Metric.closedBall (0 : E) (2 * r))
    (hcrit : ∀ x ∈ Icc (-2 * r) (r / 2),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x))
    (hha : h (-r / 2) = 0) (hhc : h (r / 2) = 0)
    (hw : ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (w x) (0 : E))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r) (r / 2)) (fun t => w t v + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r) (r / 2)) (fun t => w t v - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) = r ^ 3 / 6) :
    CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, h (-r / 2))
        (2 * r, h (2 * r)) (f (-r / 2, h (-r / 2))) (f (r / 2, h (r / 2))) ∧
      ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
        x ≠ (r / 2, h (r / 2)) → f x < f (r / 2, h (r / 2)) := by
  have hm : 2 ≤ m := hermite_m_ge_two hr hg0 hG0cv hG0cc hG0a hG0c (by simpa [hha, hhc] using hnorm)
  exact ridge_capHyp_sp hr hm hδ hh hg hFa hFc hconv hconc hball hcrit hha hhc hw hwcv hwcc
    hnorm.le

/-- **Second-order condition for strong concavity.** If `φ` has derivative `Dφ` and `Dφ` has
derivative `H` at every point of the convex set `K`, and `H y v v ≤ -δ ‖v‖²` there, then `φ` is
`δ`-strongly concave on `K`. Along each segment, `φ(x + t v) + (δ/2)‖v‖² t²` has a nonpositive
second derivative, so it is concave on `[0, 1]`. -/
theorem strongConcaveOn_of_hessian {K : Set E} {φ : E → ℝ} {Dφ : E → E →L[ℝ] ℝ}
    {H : E → E →L[ℝ] E →L[ℝ] ℝ} {δ : ℝ} (hK : Convex ℝ K)
    (hD : ∀ y ∈ K, HasFDerivAt φ (Dφ y) y) (hH : ∀ y ∈ K, HasFDerivAt Dφ (H y) y)
    (hneg : ∀ y ∈ K, ∀ v : E, H y v v ≤ -δ * ‖v‖ ^ 2) : StrongConcaveOn K δ φ := by
  refine ⟨hK, fun x hx y hy a b ha hb hab => ?_⟩
  set v := y - x with hv
  have hmem : ∀ t ∈ Icc (0 : ℝ) 1, x + t • v ∈ K := fun t ht => hK.add_smul_sub_mem hx hy ht
  set ψ : ℝ → ℝ := fun t => φ (x + t • v) + δ / 2 * ‖v‖ ^ 2 * t ^ 2 with hψ
  set ψ' : ℝ → ℝ := fun t => Dφ (x + t • v) v + δ * ‖v‖ ^ 2 * t with hψ'
  set ψ'' : ℝ → ℝ := fun t => H (x + t • v) v v + δ * ‖v‖ ^ 2 with hψ''
  have hline : ∀ t : ℝ, HasDerivAt (fun s : ℝ => x + s • v) v t := fun t => by
    simpa using ((hasDerivAt_id t).smul_const v).const_add x
  have hd1 : ∀ t ∈ Icc (0 : ℝ) 1, HasDerivAt ψ (ψ' t) t := by
    intro t ht
    have h1 : HasDerivAt (fun s => φ (x + s • v)) (Dφ (x + t • v) v) t :=
      (hD _ (hmem t ht)).comp_hasDerivAt t (hline t)
    have h2 : HasDerivAt (fun s : ℝ => δ / 2 * ‖v‖ ^ 2 * s ^ 2) (δ * ‖v‖ ^ 2 * t) t := by
      convert (hasDerivAt_pow 2 t).const_mul (δ / 2 * ‖v‖ ^ 2) using 1
      push_cast
      ring
    exact h1.add h2
  have hd2 : ∀ t ∈ Icc (0 : ℝ) 1, HasDerivAt ψ' (ψ'' t) t := by
    intro t ht
    have h1 : HasDerivAt (fun s => Dφ (x + s • v)) (H (x + t • v) v) t :=
      (hH _ (hmem t ht)).comp_hasDerivAt t (hline t)
    have h1' : HasDerivAt (fun s => Dφ (x + s • v) v) (H (x + t • v) v v) t := by
      simpa using h1.clm_apply (hasDerivAt_const t v)
    have h2 : HasDerivAt (fun s : ℝ => δ * ‖v‖ ^ 2 * s) (δ * ‖v‖ ^ 2) t := by
      simpa using (hasDerivAt_id t).const_mul (δ * ‖v‖ ^ 2)
    exact h1'.add h2
  have hcc : ConcaveOn ℝ (Icc (0 : ℝ) 1) ψ := by
    refine concaveOn_of_hasDerivWithinAt2_nonpos (convex_Icc 0 1)
      (fun t ht => (hd1 t ht).continuousAt.continuousWithinAt)
      (fun t ht => (hd1 t (interior_subset ht)).hasDerivWithinAt)
      (fun t ht => (hd2 t (interior_subset ht)).hasDerivWithinAt) (fun t ht => ?_)
    have := hneg _ (hmem t (interior_subset ht)) v
    simp only [hψ'']
    linarith
  have key := hcc.2 (left_mem_Icc.mpr zero_le_one) (right_mem_Icc.mpr zero_le_one) ha hb hab
  simp only [hψ, smul_eq_mul, mul_zero, zero_smul, add_zero, mul_one, one_smul, zero_add] at key
  have hab' : a = 1 - b := by linarith
  have hpt : a • x + b • y = x + b • v := by
    rw [hab', hv, smul_sub, sub_smul, one_smul]; abel
  have hn : ‖x - y‖ = ‖v‖ := by rw [hv, norm_sub_rev]
  rw [hpt, hn, smul_eq_mul, smul_eq_mul]
  have hxv : x + v = y := by rw [hv]; abel
  rw [hxv] at key
  show a * φ x + b * φ y + a * b * (δ / 2 * ‖v‖ ^ 2) ≤ φ (x + b • v)
  rw [hab'] at key ⊢
  linear_combination key

/-- **L3–L4 (the transverse Hessian bound).** If the transverse Hessian `H` satisfies
`H(M) ≤ -λ` (SP's `λ = λ_min(-D_y² f(M))`) and its increments from `M = (-r/2, 0)` are bounded by
`m (|Δx| + ‖Δy‖)` (L3, with `m` bounding the third-derivative blocks), then `H(p) ≤ -(λ - 5rm)`
on the cap. The increment is at most `3r/2 + 2r ≤ 5r` there; SP bounds it on all of `D` by
`9r/2 < 5r`. -/
theorem transverse_hessian_le {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {r m lam : ℝ} (hr : 0 < r)
    (hm : 0 ≤ m)
    (hlip : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖))
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) :
    ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      H p v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
  rintro ⟨x, y⟩ ⟨hx, hy⟩ v
  have hy' : ‖y‖ ≤ 2 * r := by simpa using hy
  have hdx : |x - -r / 2| ≤ 3 * r / 2 := by
    rw [abs_le]; constructor <;> linarith [hx.1, hx.2]
  have hinc : ‖H (x, y) - H (-r / 2, 0)‖ ≤ 5 * r * m := by
    have := hlip (x, y) ⟨hx, hy⟩
    calc ‖H (x, y) - H (-r / 2, 0)‖ ≤ m * (|x - -r / 2| + ‖y‖) := this
      _ ≤ m * (3 * r / 2 + 2 * r) := by gcongr
      _ ≤ 5 * r * m := by nlinarith
  have hsplit : H (x, y) v v = H (-r / 2, 0) v v + (H (x, y) - H (-r / 2, 0)) v v := by
    simp
  have hb := (H (x, y) - H (-r / 2, 0)).le_opNorm₂ v v
  have hb' : (H (x, y) - H (-r / 2, 0)) v v ≤ 5 * r * m * ‖v‖ ^ 2 := by
    have h1 := (le_abs_self _).trans (by simpa [Real.norm_eq_abs] using hb)
    calc (H (x, y) - H (-r / 2, 0)) v v ≤ ‖H (x, y) - H (-r / 2, 0)‖ * ‖v‖ * ‖v‖ := h1
      _ ≤ 5 * r * m * ‖v‖ * ‖v‖ := by gcongr
      _ = 5 * r * m * ‖v‖ ^ 2 := by ring
  rw [hsplit]
  have := hlam v
  nlinarith

/-- **L3 (increments of the transverse Hessian).** If `x ↦ H (x, 0)` has derivative `Hx x` with
`‖Hx x‖ ≤ m` on `[-2r, r/2]` (the block `∂_x D_y² f`), and along each ray `s ↦ H (x, s • y)`,
`s ∈ [0, 1]`, the derivative `Hr p s` has `‖Hr p s‖ ≤ m ‖y‖` (the block `D_y³ f` applied to `y`),
then moving first in `x` along `y = 0` and then along the ray gives `‖H p - H M‖ ≤ m (|Δx| + ‖Δy‖)`
from `M = (-r/2, 0)`, by the mean value inequality. -/
theorem hessian_increment {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m : ℝ} (hr : 0 < r)
    (hHx : ∀ x ∈ Icc (-2 * r) (r / 2), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r) (r / 2), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ s ∈ Icc (0 : ℝ) 1,
      HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ s ∈ Icc (0 : ℝ) 1,
      ‖Hr p s‖ ≤ m * ‖p.2‖) :
    ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖) := by
  rintro ⟨x, y⟩ hp
  have hx := hp.1
  have h1 : ‖H (x, (1 : ℝ) • y) - H (x, (0 : ℝ) • y)‖ ≤ m * ‖y‖ * ‖(1 : ℝ) - 0‖ :=
    (convex_Icc (0 : ℝ) 1).norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun s : ℝ => H (x, s • y)) (fun s hs => (hHr (x, y) hp s hs).hasDerivWithinAt)
      (fun s hs => hHrb (x, y) hp s hs) (left_mem_Icc.mpr zero_le_one)
      (right_mem_Icc.mpr zero_le_one)
  have ha : -r / 2 ∈ Icc (-2 * r) (r / 2) := ⟨by linarith, by linarith⟩
  have h2 : ‖H (x, 0) - H (-r / 2, 0)‖ ≤ m * ‖x - -r / 2‖ :=
    (convex_Icc (-2 * r) (r / 2)).norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun t => H (t, 0)) (fun t ht => (hHx t ht).hasDerivWithinAt) hHxb ha hx
  simp only [one_smul, zero_smul, sub_zero, norm_one, mul_one] at h1
  rw [Real.norm_eq_abs] at h2
  calc ‖H (x, y) - H (-r / 2, 0)‖ ≤ ‖H (x, y) - H (x, 0)‖ + ‖H (x, 0) - H (-r / 2, 0)‖ :=
        norm_sub_le_norm_sub_add_norm_sub (H (x, y)) (H (x, 0)) (H (-r / 2, 0))
    _ ≤ m * ‖y‖ + m * |x - -r / 2| := add_le_add h1 h2
    _ = m * (|(x, y).1 - -r / 2| + ‖(x, y).2‖) := by simp only; ring

/-- **L4 (strong concavity of the slices).** Under `transverse_hessian_le`'s hypotheses, with the
slice derivatives `Dφ` and Hessians `H` given on the cap, every slice `y ↦ f (x, y)`,
`x ∈ [-2r, r/2]`, is `(λ - 5rm)`-strongly concave on `B̄(0, 2r)`. -/
theorem slices_strongConcave {f : ℝ × E → ℝ} {Dφ : ℝ × E → E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {r m lam : ℝ} (hr : 0 < r) (hm : 0 ≤ m)
    (hD : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) (Dφ p) p.2)
    (hH : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => Dφ (p.1, y)) (H p) p.2)
    (hlip : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖))
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) :
    ∀ x ∈ Icc (-2 * r) (r / 2),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) (lam - 5 * r * m) (fun y => f (x, y)) := by
  intro x hx
  have hbd := transverse_hessian_le hr hm hlip hlam
  exact strongConcaveOn_of_hessian (Dφ := fun y => Dφ (x, y)) (H := fun y => H (x, y))
    (convex_closedBall _ _) (fun y hy => hD (x, y) ⟨hx, hy⟩) (fun y hy => hH (x, y) ⟨hx, hy⟩)
    (fun y hy v => hbd (x, y) ⟨hx, hy⟩ v)

/-- **The cap from SP's hypothesis (H1) (L1–L5, L7, L8, L11, L18).** As `ridge_capHyp_normalized`,
with the slice strong concavity and `δ > rm(8m - 5)` derived from Hessian data: the slice
derivatives and Hessians on the cap, the increment bound of `H` from `M` with constant `m`,
`H(M) ≤ -λ`, and (H1) `8 r m² < λ`. Then `δ = λ - 5rm > rm(8m - 5)`. -/
theorem ridge_capHyp_H1 {f : ℝ × E → ℝ} {h : ℝ → E} {F G0 : ℝ → ℝ}
    {w : ℝ → E →L[ℝ] ℝ} {Dφ : ℝ × E → E →L[ℝ] ℝ} {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ}
    {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ} {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam : ℝ} (hr : 0 < r)
    (hg0 : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt (fun x => f (x, 0)) (G0 x) x)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G0 t + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G0 t - m / 2 * t ^ 2))
    (hG0a : G0 (-r / 2) = 0) (hG0c : G0 (r / 2) = 0)
    (hD : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) (Dφ p) p.2)
    (hH : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => Dφ (p.1, y)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r) (r / 2), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r) (r / 2), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ s ∈ Icc (0 : ℝ) 1,
      HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ s ∈ Icc (0 : ℝ) 1,
      ‖Hr p s‖ ≤ m * ‖p.2‖)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hh : ContinuousOn h (Icc (-r / 2) (2 * r)))
    (hg : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt (fun x => f (x, h x)) (F x) x)
    (hFa : F (-r / 2) = 0) (hFc : F (r / 2) = 0)
    (hconv : ConvexOn ℝ (Icc (-2 * r) (2 * r)) (fun x => F x - (x + r / 2) * (x - r / 2) / 8))
    (hball : ∀ x ∈ Icc (-2 * r) (r / 2), h x ∈ Metric.closedBall (0 : E) (2 * r))
    (hcrit : ∀ x ∈ Icc (-2 * r) (r / 2),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x))
    (hha : h (-r / 2) = 0) (hhc : h (r / 2) = 0)
    (hw : ∀ x ∈ Icc (-2 * r) (r / 2), HasFDerivAt (fun y => f (x, y)) (w x) (0 : E))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r) (r / 2)) (fun t => w t v + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r) (r / 2)) (fun t => w t v - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, h (-r / 2)) - f (r / 2, h (r / 2)) = r ^ 3 / 6) :
    CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, h (-r / 2))
        (2 * r, h (2 * r)) (f (-r / 2, h (-r / 2))) (f (r / 2, h (r / 2))) ∧
      ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
        x ≠ (r / 2, h (r / 2)) → f x < f (r / 2, h (r / 2)) := by
  have hm : 2 ≤ m :=
    hermite_m_ge_two hr hg0 hG0cv hG0cc hG0a hG0c (by simpa [hha, hhc] using hnorm)
  have hlip := hessian_increment hr hHx hHxb hHr hHrb
  have hconc := slices_strongConcave hr (by linarith) hD hH hlip hlam
  have hδ : r * m * (8 * m - 5) < lam - 5 * r * m := by nlinarith
  exact ridge_capHyp_normalized hr hg0 hG0cv hG0cc hG0a hG0c hδ hh hg hFa hFc hconv hconc hball
    hcrit hha hhc hw hwcv hwcc hnorm

/-- **L7 (uniqueness of the transverse critical point).** A `δ`-strongly concave slice, `δ > 0`,
has at most one critical point in `K`. -/
theorem slice_critical_unique {K : Set E} {φ : E → ℝ} {δ : ℝ} (hφ : StrongConcaveOn K δ φ)
    (hδ : 0 < δ) {p q : E} (hp : p ∈ K) (hq : q ∈ K) (hdp : HasFDerivAt φ (0 : E →L[ℝ] ℝ) p)
    (hdq : HasFDerivAt φ (0 : E →L[ℝ] ℝ) q) : p = q := by
  have h1 := slice_le hφ hp hdp hq
  have h2 := slice_le hφ hq hdq hp
  rw [norm_sub_rev] at h2
  have h3 : δ * ‖q - p‖ ^ 2 ≤ 0 := by linarith
  have h4 : ‖q - p‖ ^ 2 ≤ 0 := by
    have : δ * ‖q - p‖ ^ 2 ≤ δ * 0 := by rw [mul_zero]; exact h3
    exact le_of_mul_le_mul_left this hδ
  have h5 : ‖q - p‖ = 0 := by nlinarith [norm_nonneg (q - p)]
  exact (sub_eq_zero.mp (norm_eq_zero.mp h5)).symm

/-- **L7 (existence of the transverse critical point).** If the slice is `δ`-strongly concave and
differentiable on `B̄(0, R)` and its derivative at `0` has norm `< δR` (SP: `‖w‖ < δ · 2r`), the
maximum over the compact ball is attained in the open ball, so it is a critical point. On the
sphere the radial derivative is `≤ w(p) - δR² < 0` (SP's L7 computation, here from the strong
concavity engine), which contradicts maximality. -/
theorem slice_critical_exists [ProperSpace E] {φ : E → ℝ} {Dφ : E → E →L[ℝ] ℝ} {δ R : ℝ}
    (hR : 0 < R) (hφ : StrongConcaveOn (Metric.closedBall (0 : E) R) δ φ)
    (hD : ∀ y ∈ Metric.closedBall (0 : E) R, HasFDerivAt φ (Dφ y) y) (hw : ‖Dφ 0‖ < δ * R) :
    ∃ p ∈ Metric.ball (0 : E) R, HasFDerivAt φ (0 : E →L[ℝ] ℝ) p := by
  have h0 : (0 : E) ∈ Metric.closedBall (0 : E) R := Metric.mem_closedBall_self hR.le
  have hcont : ContinuousOn φ (Metric.closedBall (0 : E) R) :=
    fun y hy => (hD y hy).continuousAt.continuousWithinAt
  obtain ⟨p, hp, hmax⟩ := (isCompact_closedBall (0 : E) R).exists_isMaxOn ⟨0, h0⟩ hcont
  have hpR : ‖p‖ ≤ R := by simpa using hp
  have hlt : ‖p‖ < R := by
    by_contra hc
    have hpn : ‖p‖ = R := le_antisymm hpR (not_lt.mp hc)
    have hseg : segment ℝ p (p + -p) ⊆ Metric.closedBall (0 : E) R := by
      rw [add_neg_cancel]
      exact (convex_closedBall (0 : E) R).segment_subset hp h0
    have hcone := mem_posTangentConeAt_of_segment_subset hseg
    have hnp := hmax.isLocalMaxOn.hasFDerivWithinAt_nonpos (hD p hp).hasFDerivWithinAt hcone
    have e1 := strongConcave_le_of_hasFDerivAt hφ hp (hD p hp) h0
    have e2 := strongConcave_le_of_hasFDerivAt hφ h0 (hD 0 h0) hp
    simp only [zero_sub, sub_zero, norm_neg, map_neg] at e1 e2 hnp
    have hb : Dφ 0 p ≤ ‖Dφ 0‖ * ‖p‖ :=
      (le_abs_self _).trans (by simpa [Real.norm_eq_abs] using (Dφ 0).le_opNorm p)
    rw [hpn] at e1 e2 hb
    have : ‖Dφ 0‖ * R < δ * R * R := by nlinarith
    nlinarith
  have hpb : p ∈ Metric.ball (0 : E) R := by simpa using hlt
  have hloc : IsLocalMax φ p :=
    hmax.isLocalMax (Metric.closedBall_mem_nhds_of_mem hpb)
  refine ⟨p, hpb, ?_⟩
  have hz := hloc.hasFDerivAt_eq_zero (hD p hp)
  rw [← hz]
  exact hD p hp

/-- In finite dimension, a bilinear form `A : E →L E →L ℝ` with `A v v ≤ -δ‖v‖²`, `δ > 0`, is
invertible as a map `E → E*` (SP L7: "`D_y² f` invertible"). -/
theorem isInvertible_of_negDef [FiniteDimensional ℝ E] {A : E →L[ℝ] E →L[ℝ] ℝ} {δ : ℝ}
    (hδ : 0 < δ) (hA : ∀ v : E, A v v ≤ -δ * ‖v‖ ^ 2) : A.IsInvertible := by
  have hinj : Function.Injective A := by
    intro v w hvw
    have h1 : A (v - w) = 0 := by rw [map_sub, hvw, sub_self]
    have h2 := hA (v - w)
    rw [h1, zero_apply] at h2
    have h3 : ‖v - w‖ ^ 2 ≤ 0 := by nlinarith [sq_nonneg ‖v - w‖]
    have h4 : ‖v - w‖ = 0 := by nlinarith [norm_nonneg (v - w)]
    exact sub_eq_zero.mp (norm_eq_zero.mp h4)
  have hrank : Module.finrank ℝ E = Module.finrank ℝ (E →L[ℝ] ℝ) := by
    rw [← (LinearMap.toContinuousLinearMap : (E →ₗ[ℝ] ℝ) ≃ₗ[ℝ] E →L[ℝ] ℝ).finrank_eq]
    exact (Subspace.dual_finrank_eq).symm
  have hsurj : Function.Surjective A :=
    (LinearMap.injective_iff_surjective_of_finrank_eq_finrank hrank (f := (A : E →ₗ[ℝ] E →L[ℝ] ℝ))).mp hinj
  refine ⟨(LinearEquiv.ofBijective (A : E →ₗ[ℝ] E →L[ℝ] ℝ) ⟨hinj, hsurj⟩).toContinuousLinearEquiv, ?_⟩
  ext v w
  rfl

/-- **L7 (regularity of the ridge, pointwise).** If the transverse gradient `G` is strictly
differentiable at `(x0, h x0)` with invertible transverse part, `h x0` is a critical point, and
near `x0` the critical point of each slice in a neighbourhood `K` of `h x0` is unique and equal to
`h x`, then `h` agrees near `x0` with the implicit function of `G = 0` (Mathlib's
`implicitFunctionOfProdDomain`), so `h` is strictly differentiable at `x0` with
`h' = -(∂_y G)⁻¹ ∂_x G` (SP L9's formula). -/
theorem ridge_hasStrictFDerivAt [CompleteSpace E] {f : ℝ × E → ℝ} {G : ℝ × E → E →L[ℝ] ℝ}
    {DG : ℝ × E →L[ℝ] E →L[ℝ] ℝ} {h : ℝ → E} {K : Set E} {x0 : ℝ}
    (hGs : HasStrictFDerivAt G DG (x0, h x0))
    (hinv : (DG ∘L ContinuousLinearMap.inr ℝ ℝ E).IsInvertible)
    (hG : ∀ᶠ v in 𝓝 (x0, h x0), HasFDerivAt (fun y => f (v.1, y)) (G v) v.2)
    (hcrit0 : HasFDerivAt (fun y => f (x0, y)) (0 : E →L[ℝ] ℝ) (h x0)) (hK : K ∈ 𝓝 (h x0))
    (huniq : ∀ᶠ x in 𝓝 x0, ∀ y ∈ K, HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) y → y = h x) :
    HasStrictFDerivAt h (-(DG ∘L ContinuousLinearMap.inr ℝ ℝ E).inverse ∘L
      (DG ∘L ContinuousLinearMap.inl ℝ ℝ E)) x0 := by
  have hGu : G (x0, h x0) = 0 := (hG.self_of_nhds).unique hcrit0
  have hψ := hGs.hasStrictFDerivAt_implicitFunctionOfProdDomain hinv
  have hev := hGs.eventually_apply_implicitFunctionOfProdDomain hinv
  have htend := hGs.tendsto_implicitFunctionOfProdDomain hinv
  set ψ := hGs.implicitFunctionOfProdDomain hinv with hψdef
  have hpair : Tendsto (fun x => (x, ψ x)) (𝓝 x0) (𝓝 (x0, h x0)) :=
    (tendsto_id.prodMk_nhds htend)
  have hGψ := hpair.eventually hG
  have hKψ := htend.eventually hK
  have heq : ψ =ᶠ[𝓝 x0] h := by
    filter_upwards [hev, hGψ, hKψ, huniq] with x h1 h2 h3 h4
    rw [h1, hGu] at h2
    exact h4 (ψ x) h3 h2
  exact hψ.congr_of_eventuallyEq heq


/-- **L7 (regularity of the ridge on `[-2r, 2r]`).** With the slices `δ`-strongly concave and `h x` a
critical point in `B̄(0, 2r)` for `x` in an open set `U ⊇ [-2r, 2r]` (SP: `f ∈ C⁴` near `D`), `h x`
in the open ball on `[-2r, 2r]`, and the transverse gradient strictly differentiable along the
ridge with negative definite transverse part, `h` is strictly differentiable at every
`x ∈ [-2r, 2r]`. Composed with `ridge_inputs_of_joint`, as `ridge_capHyp_source` does, this supplies
its `hh'`. -/
theorem ridge_differentiable [FiniteDimensional ℝ E] {f : ℝ × E → ℝ} {G : ℝ × E → E →L[ℝ] ℝ}
    {DG : ℝ → ℝ × E →L[ℝ] E →L[ℝ] ℝ} {h : ℝ → E} {U : Set ℝ} {r δ : ℝ} (hδ : 0 < δ)
    (hU : IsOpen U) (hIU : Icc (-2 * r) (2 * r) ⊆ U)
    (hconc : ∀ x ∈ U,
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hball : ∀ x ∈ U, h x ∈ Metric.closedBall (0 : E) (2 * r))
    (hcrit : ∀ x ∈ U, HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x))
    (hint : ∀ x ∈ Icc (-2 * r) (2 * r), h x ∈ Metric.ball (0 : E) (2 * r))
    (hG : ∀ x ∈ Icc (-2 * r) (2 * r),
      ∀ᶠ v in 𝓝 (x, h x), HasFDerivAt (fun y => f (v.1, y)) (G v) v.2)
    (hGs : ∀ x ∈ Icc (-2 * r) (2 * r), HasStrictFDerivAt G (DG x) (x, h x))
    (hneg : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E,
      (DG x ∘L ContinuousLinearMap.inr ℝ ℝ E) v v ≤ -δ * ‖v‖ ^ 2) :
    ∀ x ∈ Icc (-2 * r) (2 * r), HasStrictFDerivAt h
      (-(DG x ∘L ContinuousLinearMap.inr ℝ ℝ E).inverse ∘L (DG x ∘L ContinuousLinearMap.inl ℝ ℝ E)) x := by
  intro x hx
  have hK : Metric.closedBall (0 : E) (2 * r) ∈ 𝓝 (h x) :=
    Metric.closedBall_mem_nhds_of_mem (hint x hx)
  have huniq : ∀ᶠ x' in 𝓝 x, ∀ y ∈ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (x', y)) (0 : E →L[ℝ] ℝ) y → y = h x' := by
    filter_upwards [hU.mem_nhds (hIU hx)] with x' hx' y hy hdy
    exact slice_critical_unique (hconc x' hx') hδ hy (hball x' hx') hdy (hcrit x' hx')
  exact ridge_hasStrictFDerivAt (hGs x hx) (isInvertible_of_negDef hδ (hneg x hx)) (hG x hx)
    (hcrit x (hIU hx)) hK huniq


/-- **L7 (existence of the ridge).** If on an open set `U` of `x`-values the slices are
`δ`-strongly concave and differentiable on `B̄(0, 2r)` with `‖∂_y f(x, 0)‖ < δ · 2r`, a map `h`
with `h x` a critical point in the open ball exists on `U` (by choice; unique by
`slice_critical_unique`). -/
theorem ridge_exists [ProperSpace E] {f : ℝ × E → ℝ} {Dφ : ℝ → E → E →L[ℝ] ℝ} {U : Set ℝ}
    {r δ : ℝ} (hr : 0 < r)
    (hconc : ∀ x ∈ U,
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) δ (fun y => f (x, y)))
    (hD : ∀ x ∈ U, ∀ y ∈ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (x, y)) (Dφ x y) y)
    (hw : ∀ x ∈ U, ‖Dφ x 0‖ < δ * (2 * r)) :
    ∃ h : ℝ → E, ∀ x ∈ U, h x ∈ Metric.ball (0 : E) (2 * r) ∧
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := by
  have hex : ∀ x : ℝ, ∃ p : E, x ∈ U → p ∈ Metric.ball (0 : E) (2 * r) ∧
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) p := by
    intro x
    by_cases hx : x ∈ U
    · obtain ⟨p, hp, hdp⟩ := slice_critical_exists (by positivity) (hconc x hx) (hD x hx) (hw x hx)
      exact ⟨p, fun _ => ⟨hp, hdp⟩⟩
    · exact ⟨0, fun h => absurd h hx⟩
  choose h hh using hex
  exact ⟨h, fun x hx => hh x hx⟩

/-! ### The full domain (v1.8)

SP proves L4 on all of `D`, where the increment from `M` is at most `9r/2 < 5r`, and assumes
`f ∈ C⁴` near `D`. The lemmas below take the source data on a collar `x ∈ [-2r - ε, 2r + ε]`,
`0 < ε ≤ r/2`, where `|x + r/2| ≤ 3r` keeps the increment at most `5r`. So every slice over the open
set `(-2r - ε, 2r + ε) ⊇ [-2r, 2r]` is `(λ - 5rm)`-strongly concave, which is what the L7 lemmas
need, and `ridge_capHyp_source` composes L7 with the cap. -/

/-- **L3 on the full domain.** As `hessian_increment`, with the `x`-range `[-2r - ε, 2r + ε]`. -/
theorem hessian_increment_D {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖) :
    ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖) := by
  rintro ⟨x, y⟩ hp
  have hx := hp.1
  have h1 : ‖H (x, (1 : ℝ) • y) - H (x, (0 : ℝ) • y)‖ ≤ m * ‖y‖ * ‖(1 : ℝ) - 0‖ :=
    (convex_Icc (0 : ℝ) 1).norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun s : ℝ => H (x, s • y)) (fun s hs => (hHr (x, y) hp s hs).hasDerivWithinAt)
      (fun s hs => hHrb (x, y) hp s hs) (left_mem_Icc.mpr zero_le_one)
      (right_mem_Icc.mpr zero_le_one)
  have ha : -r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) := ⟨by linarith, by linarith⟩
  have h2 : ‖H (x, 0) - H (-r / 2, 0)‖ ≤ m * ‖x - -r / 2‖ :=
    (convex_Icc (-2 * r - ε) (2 * r + ε)).norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun t => H (t, 0)) (fun t ht => (hHx t ht).hasDerivWithinAt) hHxb ha hx
  simp only [one_smul, zero_smul, sub_zero, norm_one, mul_one] at h1
  rw [Real.norm_eq_abs] at h2
  calc ‖H (x, y) - H (-r / 2, 0)‖ ≤ ‖H (x, y) - H (x, 0)‖ + ‖H (x, 0) - H (-r / 2, 0)‖ :=
        norm_sub_le_norm_sub_add_norm_sub (H (x, y)) (H (x, 0)) (H (-r / 2, 0))
    _ ≤ m * ‖y‖ + m * |x - -r / 2| := add_le_add h1 h2
    _ = m * (|(x, y).1 - -r / 2| + ‖(x, y).2‖) := by simp only; ring

/-- **L4's Hessian bound on the full domain.** As `transverse_hessian_le`, for
`x ∈ [-2r - ε, 2r + ε]` with `0 < ε ≤ r/2`: there `|x + r/2| ≤ 3r`, so the increment is still at
most `5r`. -/
theorem transverse_hessian_le_D {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ}
    (hm : 0 ≤ m) (hε : 0 < ε) (hεr : ε ≤ r / 2)
    (hlip : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖))
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) :
    ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      H p v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
  rintro ⟨x, y⟩ ⟨hx, hy⟩ v
  have hy' : ‖y‖ ≤ 2 * r := by simpa using hy
  have hdx : |x - -r / 2| ≤ 3 * r := by
    rw [abs_le]; constructor <;> linarith [hx.1, hx.2]
  have hinc : ‖H (x, y) - H (-r / 2, 0)‖ ≤ 5 * r * m := by
    have := hlip (x, y) ⟨hx, hy⟩
    calc ‖H (x, y) - H (-r / 2, 0)‖ ≤ m * (|x - -r / 2| + ‖y‖) := this
      _ ≤ m * (3 * r + 2 * r) := by gcongr
      _ = 5 * r * m := by ring
  have hsplit : H (x, y) v v = H (-r / 2, 0) v v + (H (x, y) - H (-r / 2, 0)) v v := by
    simp
  have hb := (H (x, y) - H (-r / 2, 0)).le_opNorm₂ v v
  have hb' : (H (x, y) - H (-r / 2, 0)) v v ≤ 5 * r * m * ‖v‖ ^ 2 := by
    have h1 := (le_abs_self _).trans (by simpa [Real.norm_eq_abs] using hb)
    calc (H (x, y) - H (-r / 2, 0)) v v ≤ ‖H (x, y) - H (-r / 2, 0)‖ * ‖v‖ * ‖v‖ := h1
      _ ≤ 5 * r * m * ‖v‖ * ‖v‖ := by gcongr
      _ = 5 * r * m * ‖v‖ ^ 2 := by ring
  rw [hsplit]
  have := hlam v
  nlinarith

/-- **L4 on the full domain.** Every slice `y ↦ f (x, y)`, `x ∈ [-2r - ε, 2r + ε]`, is
`(λ - 5rm)`-strongly concave on `B̄(0, 2r)`. -/
theorem slices_strongConcave_D {f : ℝ × E → ℝ} {Dφ : ℝ × E → E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hm : 0 ≤ m) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hD : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) (Dφ p) p.2)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => Dφ (p.1, y)) (H p) p.2)
    (hlip : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖H p - H (-r / 2, 0)‖ ≤ m * (|p.1 - -r / 2| + ‖p.2‖))
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) :
    ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε),
      StrongConcaveOn (Metric.closedBall (0 : E) (2 * r)) (lam - 5 * r * m) (fun y => f (x, y)) := by
  intro x hx
  have hbd := transverse_hessian_le_D hm hε hεr hlip hlam
  exact strongConcaveOn_of_hessian (Dφ := fun y => Dφ (x, y)) (H := fun y => H (x, y))
    (convex_closedBall _ _) (fun y hy => hD (x, y) ⟨hx, hy⟩) (fun y hy => hH (x, y) ⟨hx, hy⟩)
    (fun y hy v => hbd (x, y) ⟨hx, hy⟩ v)

/-- **The cap from SP's source data (L1–L5, L7, L8, L11, L18 composed).** Fix a collar
`0 < ε ≤ r/2` and assume, for `x ∈ [-2r - ε, 2r + ε]` and `y ∈ B̄(0, 2r)`: a joint derivative `Df` of `f`, strictly differentiable in
its transverse part `∂_y f = Df ∘ inr` with derivative `DG`; the transverse Hessian `H` with the
third-derivative bounds `m` of L3; (H1) `8rm² < λ` with `H(M) ≤ -λ`; the critical pins
`Df(∓r/2, 0) = 0`; the weak bounds `|∂_x³ f(·, 0)| ≤ m` on `[-r/2, r/2]` (L1) and
`|∂_x² ∂_y f(·, 0)| ≤ m` on the collar (L5); and the gap `f(M) - f(S) = r³/6`. Then:
- a ridge `h` exists: on `(-2r - ε, 2r + ε)`, `h x` is a critical point of its slice in the open
  ball;
- for any such ridge (unique by `slice_critical_unique`) with L12's `hconv` for
  `F x = ∂_x f(x, h x)`, `h(∓r/2) = 0`, and (C1)–(C3) and (C2⁺) hold on the cap with `M = (-r/2, 0)`,
  `S = (r/2, 0)`, `b = f(M)` and `s = f(S)`.

L8 puts `h` in the open ball on `[-2r, 2r]` (`‖w‖ ≤ 3mr² < 2rδ`); the implicit function theorem
(`ridge_differentiable`) makes `h` differentiable there, with negative definite `DG ∘ inr = H`;
`ridge_inputs_of_joint` gives L11; `ridge_capHyp_H1` gives the cap. Only L12 (`hconv`) remains an
input, with L6 and L9–L11's quantitative bounds that SP uses to prove it. -/
theorem ridge_capHyp_source [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hDf : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt f (Df p) p)
    (hGs : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) - m / 2 * t ^ 2))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        ConvexOn ℝ (Icc (-2 * r) (2 * r))
          (fun x => Df (x, h x) (1, 0) - (x + r / 2) * (x - r / 2) / 8) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  have hUX : Ioo (-2 * r - ε) (2 * r + ε) ⊆ Icc (-2 * r - ε) (2 * r + ε) := Ioo_subset_Icc_self
  have hcapX : Icc (-2 * r) (r / 2) ⊆ Icc (-2 * r - ε) (2 * r + ε) :=
    Icc_subset_Icc (by linarith) (by linarith)
  have hIU : Icc (-2 * r) (2 * r) ⊆ Ioo (-2 * r - ε) (2 * r + ε) :=
    fun t ht => ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  -- the slice derivatives are the transverse parts of `Df`
  have hD : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) ((Df p).comp (ContinuousLinearMap.inr ℝ ℝ E)) p.2 := by
    rintro ⟨x, y⟩ hp
    exact slice_hasFDerivAt (hDf (x, y) hp)
  -- L1–L2: `m ≥ 2`
  have hg0 : ∀ x ∈ Icc (-r / 2) (r / 2),
      HasDerivAt (fun x => f (x, 0)) (Df (x, 0) (1, 0)) x := by
    intro x hx
    have hx' : (x, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
      ⟨⟨by linarith [hx.1], by linarith [hx.2]⟩, h0Y⟩
    have hp : HasDerivAt (fun t : ℝ => (t, (0 : E))) ((1 : ℝ), (0 : E)) x :=
      (hasDerivAt_id x).prodMk (hasDerivAt_const x (0 : E))
    exact (hDf _ hx').comp_hasDerivAt x hp
  have hG0a : Df (-r / 2, 0) (1, 0) = 0 := by simp [hM]
  have hG0c : Df (r / 2, 0) (1, 0) = 0 := by simp [hS]
  have hm : 2 ≤ m := hermite_m_ge_two (g := fun x => f (x, 0)) hr hg0 hG0cv hG0cc hG0a hG0c hnorm
  have hm0 : 0 ≤ m := by linarith
  -- L3–L4 on the full domain
  have hlip := hessian_increment_D hr hε hHx hHxb hHr hHrb
  have hneg := transverse_hessian_le_D hm0 hε hεr hlip hlam
  have hconcX := slices_strongConcave_D hm0 hε hεr hD hH hlip hlam
  have h11 : 11 * (r * m) < lam - 5 * r * m := by
    nlinarith [mul_nonneg (mul_nonneg hr.le hm0) (sub_nonneg.2 hm)]
  have hrm : 0 < r * m := mul_pos hr (by linarith)
  have hδ0 : 0 < lam - 5 * r * m := by linarith
  -- L5 on the full domain: `‖w x‖ ≤ 3mr² < 2rδ`
  have hwv : ∀ (t : ℝ) (v : E),
      ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v = Df (t, 0) (0, v) := fun t v => by simp
  have hwcv' : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v + m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcv v).congr fun t _ => by simp only [hwv]
  have hwcc' : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v - m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcc v).congr fun t _ => by simp only [hwv]
  have hwa : (Df (-r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hM]
  have hwc : (Df (r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hS]
  have hwlt : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε),
      ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ < (lam - 5 * r * m) * (2 * r) := by
    intro x hx
    have h1 := slice_gradient_bound (w := fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      hm0 hwcv' hwcc' (show -r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩)
      (show r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩) (by linarith)
      hwa hwc hx
    have h2 : |(x - -r / 2) * (x - r / 2)| ≤ 6 * r ^ 2 := by
      have hxx : 0 ≤ (x - (-2 * r - ε)) * (2 * r + ε - x) :=
        mul_nonneg (by linarith [hx.1]) (by linarith [hx.2])
      have hee : 0 ≤ (r / 2 - ε) * (9 * r / 2 + ε) :=
        mul_nonneg (by linarith) (by linarith)
      rw [abs_le]; constructor <;> nlinarith [sq_nonneg x, sq_nonneg r]
    have h3 : ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ 3 * m * r ^ 2 :=
      calc ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖
          ≤ m / 2 * |(x - -r / 2) * (x - r / 2)| := h1
        _ ≤ m / 2 * (6 * r ^ 2) := by gcongr
        _ = 3 * m * r ^ 2 := by ring
    have h4 : 22 * r * (r * m) < (lam - 5 * r * m) * (2 * r) := by nlinarith
    nlinarith
  have hslice0 : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε),
      HasFDerivAt (fun y => f (x, y)) ((Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) 0 :=
    fun x hx => hD (x, 0) ⟨hx, h0Y⟩
  refine ⟨ridge_exists (Dφ := fun x y => (Df (x, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) hr
    (fun x hx => hconcX x (hUX hx)) (fun x hx y hy => hD (x, y) ⟨hUX hx, hy⟩)
    (fun x hx => hwlt x (hUX hx)), ?_⟩
  intro h hhU hconv
  have hball : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => (hhU x hx).1
  have hcrit : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := fun x hx => (hhU x hx).2
  -- L7 uniqueness: the pins are on the ridge
  have hMU : -r / 2 ∈ Ioo (-2 * r - ε) (2 * r + ε) := ⟨by linarith, by linarith⟩
  have hSU : r / 2 ∈ Ioo (-2 * r - ε) (2 * r + ε) := ⟨by linarith, by linarith⟩
  have hha : h (-r / 2) = 0 :=
    slice_critical_unique (hconcX _ (hUX hMU)) hδ0 (hball _ hMU) h0Y (hcrit _ hMU)
      (by simpa [hwa] using hslice0 _ (hUX hMU))
  have hhc : h (r / 2) = 0 :=
    slice_critical_unique (hconcX _ (hUX hSU)) hδ0 (hball _ hSU) h0Y (hcrit _ hSU)
      (by simpa [hwc] using hslice0 _ (hUX hSU))
  -- L8: the ridge is in the open ball on `[-2r, 2r]`
  have hint : ∀ x ∈ Icc (-2 * r) (2 * r), h x ∈ Metric.ball (0 : E) (2 * r) := by
    intro x hx
    have hxU := hIU hx
    have h8 := slice_ridge_bound (hconcX x (hUX hxU)) h0Y (hball x hxU) (hcrit x hxU)
      (hslice0 x (hUX hxU))
    have h9 := hwlt x (hUX hxU)
    rw [Metric.mem_ball, dist_zero_right]
    have : (lam - 5 * r * m) * ‖h x‖ < (lam - 5 * r * m) * (2 * r) := by linarith
    exact lt_of_mul_lt_mul_left this hδ0.le
  have hridgeX : ∀ x ∈ Icc (-2 * r) (2 * r),
      (x, h x) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => ⟨hUX (hIU hx), hball x (hIU hx)⟩
  -- L7 regularity: the transverse part of `DG` is `H`, negative definite
  have hDGH : ∀ x ∈ Icc (-2 * r) (2 * r),
      DG (x, h x) ∘L ContinuousLinearMap.inr ℝ ℝ E = H (x, h x) := by
    intro x hx
    have h1 : HasFDerivAt (fun y => (Df (x, y)).comp (ContinuousLinearMap.inr ℝ ℝ E))
        ((DG (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (h x) :=
      (hGs _ (hridgeX x hx)).hasFDerivAt.comp (h x) (hasFDerivAt_prodMk_right x (h x))
    exact h1.unique (hH _ (hridgeX x hx))
  have hnegR : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E,
      (DG (x, h x) ∘L ContinuousLinearMap.inr ℝ ℝ E) v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
    intro x hx v
    rw [hDGH x hx]
    exact hneg _ (hridgeX x hx) v
  have hG : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ᶠ v in 𝓝 (x, h x),
      HasFDerivAt (fun y => f (v.1, y)) ((Df v).comp (ContinuousLinearMap.inr ℝ ℝ E)) v.2 := by
    intro x hx
    have hopen : IsOpen (Ioo (-2 * r - ε) (2 * r + ε) ×ˢ Metric.ball (0 : E) (2 * r)) :=
      isOpen_Ioo.prod Metric.isOpen_ball
    filter_upwards [hopen.mem_nhds ⟨hIU hx, hint x hx⟩] with v hv
    exact hD v ⟨hUX hv.1, Metric.ball_subset_closedBall hv.2⟩
  have hstrict := ridge_differentiable (f := f)
    (G := fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG := fun x => DG (x, h x))
    (h := h) (U := Ioo (-2 * r - ε) (2 * r + ε)) hδ0 isOpen_Ioo hIU
    (fun x hx => hconcX x (hUX hx)) hball hcrit hint hG
    (fun x hx => hGs _ (hridgeX x hx)) hnegR
  -- L11
  have hDr : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt f (Df (x, h x)) (x, h x) :=
    fun x hx => hDf _ (hridgeX x hx)
  have hridge : ∀ x ∈ Icc (-2 * r) (2 * r),
      (Df (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 :=
    fun x hx => (slice_hasFDerivAt (hDr x hx)).unique (hcrit x (hIU hx))
  obtain ⟨hh, hg, -⟩ := ridge_inputs_of_joint hr hDr
    (fun x hx => (hstrict x hx).hasFDerivAt.hasDerivAt) hridge
  have hFa : Df (-r / 2, h (-r / 2)) (1, 0) = 0 := by simp [hha, hM]
  have hFc : Df (r / 2, h (r / 2)) (1, 0) = 0 := by simp [hhc, hS]
  have hcap : ∀ x ∈ Icc (-2 * r) (r / 2), x ∈ Ioo (-2 * r - ε) (2 * r + ε) :=
    fun x hx => ⟨by linarith [hx.1], by linarith [hx.2]⟩
  have key := ridge_capHyp_H1 (w := fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E))
    hr hg0 hG0cv hG0cc hG0a hG0c
    (fun p hp => hD p ⟨hcapX hp.1, hp.2⟩) (fun p hp => hH p ⟨hcapX hp.1, hp.2⟩)
    (fun x hx => hHx x (hcapX hx)) (fun x hx => hHxb x (hcapX hx))
    (fun p hp => hHr p ⟨hcapX hp.1, hp.2⟩) (fun p hp => hHrb p ⟨hcapX hp.1, hp.2⟩) hlam hH1
    hh hg hFa hFc hconv (fun x hx => hball x (hcap x hx)) (fun x hx => hcrit x (hcap x hx))
    hha hhc (fun x hx => hslice0 x (hcapX hx))
    (fun v => (hwcv' v).subset hcapX (convex_Icc _ _))
    (fun v => (hwcc' v).subset hcapX (convex_Icc _ _)) (by rw [hha, hhc]; exact hnorm)
  rw [hha, hhc] at key
  exact ⟨hha, hhc, key⟩


/-! ### L6, L9 and L12 (v1.9)

The last analytic input of the cap, L12's convexity `hconv`, is derived here. L6 bounds `w'` by
chord slopes of L5's convexity pair (no integral), L9 bounds `h'` through the differentiated ridge
equation, and L12 uses that `F' = D²f[(1, h'), (1, 0)]` is the maximum of `ζ ↦ D²f[(1, ζ), (1, ζ)]`,
so that no second derivative of `h` is needed. `ridge_capHyp_C3` composes them with
`ridge_capHyp_source`. -/

/-- A convex function with equal values at `a < c` has derivative `≥ 0` to the right of `c` and
`≤ 0` to the left of `a`. -/
theorem convex_deriv_sign {φ : ℝ → ℝ} {D : Set ℝ} {a c x d : ℝ} (hφ : ConvexOn ℝ D φ)
    (haD : a ∈ D) (hcD : c ∈ D) (hac : a < c) (heq : φ a = φ c) (hx : x ∈ D)
    (hd : HasDerivAt φ d x) : (c ≤ x → 0 ≤ d) ∧ (x ≤ a → d ≤ 0) := by
  have h0 : (φ c - φ a) / (c - a) = 0 := by rw [heq, sub_self, zero_div]
  constructor
  · intro hcx
    have hax : a < x := hac.trans_le hcx
    have h1 : (φ c - φ a) / (c - a) ≤ (φ x - φ a) / (x - a) :=
      hφ.secant_mono haD hcD hx hac.ne' hax.ne' hcx
    have h2 := hφ.slope_le_of_hasDerivAt haD hx hax hd
    rw [slope_def_field] at h2
    linarith
  · intro hxa
    rcases hxa.eq_or_lt with h | h
    · subst h
      have h2 := hφ.le_slope_of_hasDerivAt hx hcD hac hd
      rw [slope_def_field] at h2
      linarith
    · have h1 : (φ a - φ x) / (a - x) ≤ (φ c - φ a) / (c - a) :=
        hφ.slope_mono_adjacent hx hcD h hac
      have h2 := hφ.le_slope_of_hasDerivAt hx haD h hd
      rw [slope_def_field] at h2
      linarith

/-- **L6 (scalar).** If `u + (m/2)t²` is convex and `u - (m/2)t²` concave on `D` (weakly
`|u''| ≤ m`), `u(∓r/2) = 0` and `u` has derivative `d` at `x ∈ D`, then `|d| ≤ m · max(|x|, r/2)`.
Outside the pins the chord through them has slope `0`; between them the chord to the nearer pin and
L5's bound on `u(x)` give `mr/2`. -/
theorem two_node_deriv_bound {u : ℝ → ℝ} {D : Set ℝ} {r m x d : ℝ} (hr : 0 < r) (hm : 0 ≤ m)
    (hcv : ConvexOn ℝ D (fun t => u t + m / 2 * t ^ 2))
    (hcc : ConcaveOn ℝ D (fun t => u t - m / 2 * t ^ 2))
    (haD : -r / 2 ∈ D) (hcD : r / 2 ∈ D) (ha : u (-r / 2) = 0) (hc : u (r / 2) = 0)
    (hx : x ∈ D) (hd : HasDerivAt u d x) : |d| ≤ m * max |x| (r / 2) := by
  have hac : -r / 2 < r / 2 := by linarith
  have hq : HasDerivAt (fun t : ℝ => m / 2 * t ^ 2) (m * x) x := by
    convert (hasDerivAt_pow 2 x).const_mul (m / 2) using 1
    push_cast
    ring
  have hg : HasDerivAt (fun t => u t + m / 2 * t ^ 2) (d + m * x) x := hd.add hq
  have hcv' : ConvexOn ℝ D (fun t => -u t + m / 2 * t ^ 2) :=
    hcc.neg.congr fun t _ => by simp only [Pi.neg_apply]; ring
  have hk : HasDerivAt (fun t => -u t + m / 2 * t ^ 2) (-d + m * x) x := hd.neg.add hq
  have hgeq : (fun t => u t + m / 2 * t ^ 2) (-r / 2) = (fun t => u t + m / 2 * t ^ 2) (r / 2) := by
    simp only [ha, hc]; ring
  have hkeq : (fun t => -u t + m / 2 * t ^ 2) (-r / 2) =
      (fun t => -u t + m / 2 * t ^ 2) (r / 2) := by
    simp only [ha, hc]; ring
  obtain ⟨g1, g2⟩ := convex_deriv_sign hcv haD hcD hac hgeq hx hg
  obtain ⟨k1, k2⟩ := convex_deriv_sign hcv' haD hcD hac hkeq hx hk
  rcases le_or_gt (r / 2) x with hcx | hxc
  · have h1 := g1 hcx
    have h2 := k1 hcx
    have hxpos : 0 ≤ x := by linarith
    rw [abs_of_nonneg hxpos]
    calc |d| ≤ m * x := abs_le.mpr ⟨by linarith, by linarith⟩
      _ ≤ m * max x (r / 2) := mul_le_mul_of_nonneg_left (le_max_left _ _) hm
  rcases le_or_gt x (-r / 2) with hxa | hax
  · have h1 := g2 hxa
    have h2 := k2 hxa
    have hxneg : x ≤ 0 := by linarith
    rw [abs_of_nonpos hxneg]
    calc |d| ≤ m * -x := abs_le.mpr ⟨by linarith, by linarith⟩
      _ ≤ m * max (-x) (r / 2) := mul_le_mul_of_nonneg_left (le_max_left _ _) hm
  · -- between the pins
    have hu := two_node_bound hcv hcc haD hcD hac ha hc hx
    have hP : (x - -r / 2) * (x - r / 2) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
    rw [abs_of_nonpos hP] at hu
    have hcx : 0 < r / 2 - x := by linarith
    have hs1 := hcv.le_slope_of_hasDerivAt hx hcD hxc hg
    have hs2 := hcv'.le_slope_of_hasDerivAt hx hcD hxc hk
    rw [slope_def_field, le_div_iff₀ hcx] at hs1 hs2
    simp only [hc] at hs1 hs2
    have hu1 := (abs_le.mp hu).1
    have hu2 := (abs_le.mp hu).2
    have key1 : d * (r / 2 - x) ≤ m / 2 * r * (r / 2 - x) := by nlinarith
    have key2 : -d * (r / 2 - x) ≤ m / 2 * r * (r / 2 - x) := by nlinarith
    have hd1 := le_of_mul_le_mul_right key1 hcx
    have hd2 := le_of_mul_le_mul_right key2 hcx
    calc |d| ≤ m / 2 * r := abs_le.mpr ⟨by linarith, hd1⟩
      _ = m * (r / 2) := by ring
      _ ≤ m * max |x| (r / 2) := mul_le_mul_of_nonneg_left (le_max_right _ _) hm


/-- **L6 (CAP (7)).** For the slice derivatives `w t` at `y = 0` with derivative `w' x` at `x`, under
L5's hypotheses: `‖w' x‖ ≤ m · max(|x|, r/2)`; on `[-2r, 2r]` this is SP's `‖w'‖ ≤ 2mr`. -/
theorem slice_gradient_deriv_bound {w : ℝ → E →L[ℝ] ℝ} {w' : E →L[ℝ] ℝ} {D : Set ℝ} {r m x : ℝ}
    (hr : 0 < r) (hm : 0 ≤ m)
    (hcv : ∀ v : E, ConvexOn ℝ D (fun t => w t v + m * ‖v‖ / 2 * t ^ 2))
    (hcc : ∀ v : E, ConcaveOn ℝ D (fun t => w t v - m * ‖v‖ / 2 * t ^ 2))
    (haD : -r / 2 ∈ D) (hcD : r / 2 ∈ D) (ha : w (-r / 2) = 0) (hc : w (r / 2) = 0)
    (hx : x ∈ D) (hd : HasDerivAt w w' x) : ‖w'‖ ≤ m * max |x| (r / 2) := by
  refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun v => ?_
  have hdv : HasDerivAt (fun t => w t v) (w' v) x := by
    simpa using hd.clm_apply (hasDerivAt_const x v)
  have h := two_node_deriv_bound (u := fun t => w t v) (m := m * ‖v‖) hr (by positivity) (hcv v)
    (hcc v) haD hcD (by simp [ha]) (by simp [hc]) hx hdv
  rw [Real.norm_eq_abs]
  calc |w' v| ≤ m * ‖v‖ * max |x| (r / 2) := h
    _ = m * max |x| (r / 2) * ‖v‖ := by ring

/-- **L9 and the Schur maximum.** Let `Q` be a bilinear form on `ℝ × E` with `δ`-negative definite
transverse part, and let `η` satisfy the differentiated ridge equation `Q (1, η) (0, ·) = 0`. Then
`δ ‖η‖ ≤ ‖Q (1, 0) (0, ·)‖` (L9: `‖h'‖ ≤ ‖∂_x ∇_y f‖ / δ`), `Q (1, η) (1, η) = Q (1, η) (1, 0)`, and,
if `Q` is symmetric, `Q (1, ζ) (1, ζ) ≤ Q (1, η) (1, η)` for every `ζ`: the Schur complement
`F' = f_xx + f_xy[h']` is the maximum of `ζ ↦ D²f[(1, ζ), (1, ζ)]`. -/
theorem schur_max {Q : ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ} {δ : ℝ} {η : E} (hδ : 0 < δ)
    (hneg : ∀ v : E, Q (0, v) (0, v) ≤ -δ * ‖v‖ ^ 2)
    (hridge : (Q (1, η)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0) :
    δ * ‖η‖ ≤ ‖(Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ∧
      Q (1, η) (1, η) = Q (1, η) (1, 0) ∧
      ((∀ v w, Q v w = Q w v) → ∀ ζ : E, Q (1, ζ) (1, ζ) ≤ Q (1, η) (1, η)) := by
  have h0 : ∀ v : E, Q (1, η) (0, v) = 0 := fun v => by
    simpa using congrArg (fun L : E →L[ℝ] ℝ => L v) hridge
  have hsplit : ∀ ζ : E, ((1 : ℝ), ζ) = ((1 : ℝ), (0 : E)) + ((0 : ℝ), ζ) := fun ζ => by simp
  refine ⟨?_, ?_, ?_⟩
  · have h1 : Q (1, 0) (0, η) + Q (0, η) (0, η) = 0 := by
      have := h0 η
      rw [hsplit η, map_add, add_apply] at this
      exact this
    have h2 : Q (1, 0) (0, η) ≤ ‖(Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ * ‖η‖ := by
      have := ((Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)).le_opNorm η
      have h3 : ((Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) η = Q (1, 0) (0, η) := by simp
      rw [h3] at this
      exact (le_abs_self _).trans (by simpa [Real.norm_eq_abs] using this)
    have h4 := hneg η
    have h5 : δ * ‖η‖ * ‖η‖ ≤ ‖(Q (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ * ‖η‖ := by
      nlinarith
    rcases (norm_nonneg η).eq_or_lt with h | h
    · rw [← h, mul_zero]; exact norm_nonneg _
    · exact le_of_mul_le_mul_right h5 h
  · have e : Q (1, η) (1, η) = Q (1, η) ((1, 0) + (0, η)) := by rw [← hsplit η]
    rw [e, map_add, h0 η, add_zero]
  · intro hsym ζ
    have hz : ((1 : ℝ), ζ) = ((1 : ℝ), η) + ((0 : ℝ), ζ - η) := by simp
    have h1 := h0 (ζ - η)
    have h2 : Q (0, ζ - η) (1, η) = 0 := by rw [hsym]; exact h1
    have h3 := hneg (ζ - η)
    rw [hz, map_add, map_add, add_apply, add_apply, h1, h2]
    nlinarith [sq_nonneg ‖ζ - η‖]

/-- **The correction terms of L12.** Let `T a b v` be the third derivative of `f` (the derivative of
`D²f[a, b]` in direction `v`). If every block other than `f_xxx` has norm at most `m`, in the combined
form below, then `T (1, b) (1, b) (1, a) ≥ f_xxx - m((1 + ‖b‖)²(1 + ‖a‖) - 1)`. With `‖a‖, ‖b‖ ≤ u`
this is SP's adverse bound `m u (3 + 3u + u²)`. -/
theorem trilinear_lower {T : ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {m : ℝ}
    (hT : ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (a b : E) :
    T (1, 0) (1, 0) (1, 0) - m * ((1 + ‖b‖) * (1 + ‖b‖) * (1 + ‖a‖) - 1) ≤ T (1, b) (1, b) (1, a) := by
  have h := (abs_le.mp (hT 1 1 1 b b a)).1
  simp only [abs_one, one_mul, mul_one] at h
  linarith

/-- **SP's constants close L12.** With `m ≥ 2`, `δ > rm(8m - 5)` and `u = 2s + 2s²`, `s = mr/δ`
(L9's bound on `‖h'‖`): `7/4 - m((1 + u)³ - 1) ≥ 1/4`. SP's exact value at `k = 11` is
`7/4 - 2554128/1771561 = 0.308… > 1/4`. -/
theorem l12_constants {r m δ : ℝ} (hr : 0 < r) (hm : 2 ≤ m) (hδ : r * m * (8 * m - 5) < δ) :
    1 / 4 ≤ 7 / 4 - m * ((1 + (2 * (m * r / δ) + 2 * (m * r / δ) ^ 2)) ^ 3 - 1) := by
  have hrm : 0 < r * m := mul_pos hr (by linarith)
  have h11 : 11 * (r * m) ≤ r * m * (8 * m - 5) := by nlinarith
  have hδ0 : 0 < δ := by linarith
  set s := m * r / δ with hs
  have hs0 : 0 ≤ s := by positivity
  have hs1 : s ≤ 1 / 11 := by
    rw [hs, div_le_iff₀ hδ0]; nlinarith
  have hms : m * s ≤ 2 / 11 := by
    have : m * s = m * m * r / δ := by rw [hs]; ring
    rw [this, div_le_iff₀ hδ0]
    nlinarith [mul_nonneg (mul_nonneg hr.le (by linarith : (0 : ℝ) ≤ m)) (by linarith : (0 : ℝ) ≤ m - 2)]
  set u := 2 * s + 2 * s ^ 2 with hu
  have hu0 : 0 ≤ u := by positivity
  have hu1 : u ≤ 24 / 121 := by rw [hu]; nlinarith
  have hmu : m * u ≤ 48 / 121 := by
    have : m * u = 2 * (m * s) * (1 + s) := by rw [hu]; ring
    rw [this]
    have h1 : 1 + s ≤ 12 / 11 := by linarith
    have h2 : 0 ≤ m * s := by positivity
    nlinarith
  have hexp : m * ((1 + u) ^ 3 - 1) = (m * u) * (3 + 3 * u + u ^ 2) := by ring
  rw [hexp]
  have h3 : 3 + 3 * u + u ^ 2 ≤ 3 + 3 * (24 / 121) + (24 / 121) ^ 2 := by nlinarith
  have h4 : (m * u) * (3 + 3 * u + u ^ 2) ≤ (48 / 121) * (3 + 3 * (24 / 121) + (24 / 121) ^ 2) :=
    mul_le_mul hmu h3 (by positivity) (by norm_num)
  norm_num at h4 ⊢
  linarith

/-- **The transverse Hessian is the transverse block of `D²f`.** If `y ↦ ∂_y f (x, y)` has
derivative `H` at `y₀` and `Df` has derivative `D2f` at `(x, y₀)`, then
`H v w = D²f (x, y₀) [(0, v), (0, w)]`. -/
theorem transverse_hessian_eq {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {H : E →L[ℝ] E →L[ℝ] ℝ} {x : ℝ} {y₀ : E}
    (hH : HasFDerivAt (fun y => (Df (x, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) H y₀)
    (hD2 : HasFDerivAt Df D2f (x, y₀)) (v w : E) : H v w = D2f (0, v) (0, w) := by
  have h1 := hH.clm_apply (hasFDerivAt_const w y₀)
  have h2 := (hD2.comp y₀ (hasFDerivAt_prodMk_right x y₀)).clm_apply
    (hasFDerivAt_const ((0 : ℝ), w) y₀)
  have h3 := h1.unique h2
  have := congrArg (fun L : E →L[ℝ] ℝ => L v) h3
  simpa using this

/-- **The differentiated ridge equation.** If `∂_y f (t, h t) = 0` for `t` near `x`, `h` has
derivative `h'` at `x` and `Df` has derivative `D2f` at `(x, h x)`, then
`D²f (x, h x) [(1, h'), (0, ·)] = 0`, i.e. `∂_x ∇_y f + D_y² f · h' = 0` (L11's first identity). -/
theorem ridge_second_identity {Df : ℝ × E → ℝ × E →L[ℝ] ℝ}
    {D2f : ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ} {h : ℝ → E} {h' : E} {x : ℝ}
    (hridge : ∀ᶠ t in 𝓝 x, (Df (t, h t)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0)
    (hh : HasDerivAt h h' x) (hD2 : HasFDerivAt Df D2f (x, h x)) :
    (D2f (1, h')).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by
  have hp : HasDerivAt (fun t => (t, h t)) ((1 : ℝ), h') x := (hasDerivAt_id x).prodMk hh
  have h1 : HasDerivAt (fun t => Df (t, h t)) (D2f (1, h')) x := hD2.comp_hasDerivAt x hp
  have h2 : HasDerivAt (fun t => (Df (t, h t)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      ((D2f (1, h')).comp (ContinuousLinearMap.inr ℝ ℝ E)) x := by
    have := h1.clm_comp (hasDerivAt_const x (ContinuousLinearMap.inr ℝ ℝ E))
    simpa using this
  have h3 : HasDerivAt (fun t => (Df (t, h t)).comp (ContinuousLinearMap.inr ℝ ℝ E)) 0 x :=
    (hasDerivAt_const x (0 : E →L[ℝ] ℝ)).congr_of_eventuallyEq hridge
  exact h2.unique h3

/-- **L12 (CAP (11)–(12)): the convexity input `hconv`.** Along the ridge, let `Df`, `D2f` be the
first two derivatives of `f`, with `D2f` symmetric and its transverse part `δ`-negative definite,
`‖h'‖ ≤ u`, and for every `‖b‖ ≤ u` let `s ↦ D²f(s, h s)[(1, b), (1, b)]` have derivative `≥ 1/4`
(SP: `f_xxx ≥ 7/4` minus the adverse part `m u (3 + 3u + u²)`; see `trilinear_lower`). Then
`F - (x + r/2)(x - r/2)/8`, `F x = ∂_x f (x, h x)`, is convex on `[-2r, 2r]`. No second derivative of
`h` is used: `F' = D²f[(1, h'), (1, 0)]` is the maximum of `ζ ↦ D²f[(1, ζ), (1, ζ)]` (`schur_max`),
so `F'(x₂) - F'(x₁) ≥ D²f(x₂)[(1, η), (1, η)] - D²f(x₁)[(1, η), (1, η)] ≥ (x₂ - x₁)/4` with
`η = h'(x₁)`. -/
theorem ridge_hconv {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {Φ : ℝ → E → ℝ} {h h' : ℝ → E} {U : Set ℝ} {r δ u : ℝ} (hδ : 0 < δ) (hU : IsOpen U)
    (hIU : Icc (-2 * r) (2 * r) ⊆ U)
    (hridge : ∀ x ∈ U, (Df (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0)
    (hh : ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt h (h' x) x)
    (hD2 : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt Df (D2f (x, h x)) (x, h x))
    (hsym : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v w, D2f (x, h x) v w = D2f (x, h x) w v)
    (hneg : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E, D2f (x, h x) (0, v) (0, v) ≤ -δ * ‖v‖ ^ 2)
    (hu : ∀ x ∈ Icc (-2 * r) (2 * r), ‖h' x‖ ≤ u)
    (hΦ : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ b : E,
      HasDerivAt (fun s => D2f (s, h s) (1, b) (1, b)) (Φ x b) x)
    (hΦb : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ b : E, ‖b‖ ≤ u → 1 / 4 ≤ Φ x b) :
    ConvexOn ℝ (Icc (-2 * r) (2 * r))
      (fun x => Df (x, h x) (1, 0) - (x + r / 2) * (x - r / 2) / 8) := by
  have hid : ∀ x ∈ Icc (-2 * r) (2 * r),
      (D2f (x, h x) (1, h' x)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := fun x hx =>
    ridge_second_identity (Filter.eventually_of_mem (hU.mem_nhds (hIU hx)) hridge) (hh x hx)
      (hD2 x hx)
  have hF : ∀ x ∈ Icc (-2 * r) (2 * r),
      HasDerivAt (fun t => Df (t, h t) (1, 0) - (t + r / 2) * (t - r / 2) / 8)
        (D2f (x, h x) (1, h' x) (1, 0) - x / 4) x := by
    intro x hx
    have hp : HasDerivAt (fun t => (t, h t)) ((1 : ℝ), h' x) x := (hasDerivAt_id x).prodMk (hh x hx)
    have h1 : HasDerivAt (fun t => Df (t, h t)) (D2f (x, h x) (1, h' x)) x :=
      (hD2 x hx).comp_hasDerivAt x hp
    have h2 : HasDerivAt (fun t => Df (t, h t) (1, 0)) (D2f (x, h x) (1, h' x) (1, 0)) x := by
      simpa using h1.clm_apply (hasDerivAt_const x ((1 : ℝ), (0 : E)))
    have h3 : HasDerivAt (fun t : ℝ => (t + r / 2) * (t - r / 2) / 8) (x / 4) x := by
      have := ((hasDerivAt_pow 2 x).div_const 8).sub_const (r ^ 2 / 32)
      convert this using 1
      · funext t
        ring
      · push_cast
        ring
    exact h2.sub h3
  have hmono : MonotoneOn (fun x => D2f (x, h x) (1, h' x) (1, 0) - x / 4)
      (Icc (-2 * r) (2 * r)) := by
    intro x1 hx1 x2 hx2 hle
    have hφ : ∀ t ∈ Icc (-2 * r) (2 * r),
        HasDerivAt (fun s => D2f (s, h s) (1, h' x1) (1, h' x1) - s / 4)
          (Φ t (h' x1) - 1 / 4) t := fun t ht =>
      (hΦ t ht (h' x1)).sub ((hasDerivAt_id t).div_const 4)
    have hφmono : MonotoneOn (fun s => D2f (s, h s) (1, h' x1) (1, h' x1) - s / 4)
        (Icc (-2 * r) (2 * r)) := by
      refine monotoneOn_of_deriv_nonneg (convex_Icc _ _)
        (fun t ht => (hφ t ht).continuousAt.continuousWithinAt)
        (fun t ht => (hφ t (interior_subset ht)).differentiableAt.differentiableWithinAt) ?_
      intro t ht
      rw [(hφ t (interior_subset ht)).deriv]
      have := hΦb t (interior_subset ht) (h' x1) (hu x1 hx1)
      linarith
    have h12 := hφmono hx1 hx2 hle
    obtain ⟨-, he1, -⟩ := schur_max hδ (hneg x1 hx1) (hid x1 hx1)
    obtain ⟨-, he2, hmax2⟩ := schur_max hδ (hneg x2 hx2) (hid x2 hx2)
    have hm2 := hmax2 (hsym x2 hx2) (h' x1)
    simp only at h12 ⊢
    rw [← he1, ← he2]
    linarith
  refine MonotoneOn.convexOn_of_deriv (convex_Icc _ _)
    (fun x hx => (hF x hx).continuousAt.continuousWithinAt)
    (fun x hx => (hF x (interior_subset hx)).differentiableAt.differentiableWithinAt) ?_
  intro x1 hx1 x2 hx2 hle
  rw [(hF x1 (interior_subset hx1)).deriv, (hF x2 (interior_subset hx2)).deriv]
  exact hmono (interior_subset hx1) (interior_subset hx2) hle


/-- **L3 in `y` for `∂_x ∇_y f`.** If along the segment `s ↦ (x, s • y)` the third derivatives
satisfy `|T (1, 0) (0, v) (0, y)| ≤ m ‖v‖ ‖y‖` (the block `f_xyy`), then
`‖∂_x ∇_y f (x, y) - ∂_x ∇_y f (x, 0)‖ ≤ m ‖y‖`. -/
theorem dx_grad_increment {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {x m : ℝ} {y : E} (hm : 0 ≤ m)
    (hD3 : ∀ s ∈ Icc (0 : ℝ) 1, ∀ a b : ℝ × E,
      HasFDerivAt (fun q => D2f q a b) (T3 (x, s • y) a b) (x, s • y))
    (hT : ∀ s ∈ Icc (0 : ℝ) 1, ∀ v : E, |T3 (x, s • y) (1, 0) (0, v) (0, y)| ≤ m * ‖v‖ * ‖y‖) :
    ‖(D2f (x, y) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) -
      (D2f (x, 0) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ m * ‖y‖ := by
  refine ContinuousLinearMap.opNorm_le_bound _ (by positivity) fun v => ?_
  have hφ : ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => D2f (x, s • y) (1, 0) (0, v))
      (T3 (x, s • y) (1, 0) (0, v) (0, y)) s := by
    intro s hs
    have hp : HasDerivAt (fun s : ℝ => (x, s • y)) ((0 : ℝ), y) s := by
      simpa using (hasDerivAt_const s x).prodMk ((hasDerivAt_id s).smul_const y)
    exact (hD3 s hs (1, 0) (0, v)).comp_hasDerivAt s hp
  have hφb : ∀ s ∈ Icc (0 : ℝ) 1, ‖T3 (x, s • y) (1, 0) (0, v) (0, y)‖ ≤ m * ‖v‖ * ‖y‖ :=
    fun s hs => by rw [Real.norm_eq_abs]; exact hT s hs v
  have := (convex_Icc (0 : ℝ) 1).norm_image_sub_le_of_norm_hasDerivWithin_le
    (fun s hs => (hφ s hs).hasDerivWithinAt) hφb (left_mem_Icc.mpr zero_le_one)
    (right_mem_Icc.mpr zero_le_one)
  simp only [one_smul, zero_smul, sub_zero, norm_one, mul_one] at this
  simp only [sub_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.inr_apply]
  calc ‖(D2f (x, y)) (1, 0) (0, v) - (D2f (x, 0)) (1, 0) (0, v)‖ ≤ m * ‖v‖ * ‖y‖ := this
    _ = m * ‖y‖ * ‖v‖ := by ring

/-- **L9's arithmetic.** From `δ ‖h'‖ ≤ ‖∂_x∇_y f(x, h)‖ ≤ 2mr + m‖h‖` and `δ ‖h‖ ≤ 2mr²`:
`‖h'‖ ≤ 2s + 2s²` with `s = mr/δ` (SP: `u = 2/k + 2/k²`). -/
theorem l9_arith {δ m r a b c : ℝ} (hδ : 0 < δ) (hm : 0 ≤ m) (ha : δ * a ≤ c)
    (hc : c ≤ 2 * m * r + m * b) (hb : δ * b ≤ 2 * m * r ^ 2) :
    a ≤ 2 * (m * r / δ) + 2 * (m * r / δ) ^ 2 := by
  have key : δ * δ * a ≤ 2 * m * r * δ + 2 * m * m * r ^ 2 := by nlinarith
  have hrhs : 2 * (m * r / δ) + 2 * (m * r / δ) ^ 2 =
      (2 * m * r * δ + 2 * m * m * r ^ 2) / (δ * δ) := by
    field_simp
  rw [hrhs, le_div_iff₀ (by positivity)]
  linarith

/-- **L12's bound at one point.** With the combined block bound, `f_xxx ≥ 7/4`, `‖a‖, ‖b‖ ≤ u`
and `7/4 - m((1 + u)³ - 1) ≥ 1/4` (`l12_constants`): `T (1, b) (1, b) (1, a) ≥ 1/4`. -/
theorem l12_lower {T : ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {m u : ℝ} {a b : E} (hm : 0 ≤ m)
    (hT : ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hxxx : 7 / 4 ≤ T (1, 0) (1, 0) (1, 0)) (hc : 1 / 4 ≤ 7 / 4 - m * ((1 + u) ^ 3 - 1))
    (ha : ‖a‖ ≤ u) (hb : ‖b‖ ≤ u) : 1 / 4 ≤ T (1, b) (1, b) (1, a) := by
  have h1 := trilinear_lower hT a b
  have hnb := norm_nonneg b
  have hna := norm_nonneg a
  have hb1 : 1 + ‖b‖ ≤ 1 + u := by linarith
  have ha1 : 1 + ‖a‖ ≤ 1 + u := by linarith
  have h4 : (1 + ‖b‖) * (1 + ‖b‖) * (1 + ‖a‖) ≤ (1 + u) ^ 3 := by
    have e : (1 + u) ^ 3 = (1 + u) * (1 + u) * (1 + u) := by ring
    rw [e]
    exact mul_le_mul (mul_le_mul hb1 hb1 (by linarith) (by linarith)) ha1 (by linarith)
      (mul_nonneg (by linarith) (by linarith))
  have h5 := mul_le_mul_of_nonneg_left (sub_le_sub_right h4 1) hm
  linarith

/-- **The ridge facts from SP's source data.** Under the hypotheses of `ridge_capHyp_source`, for a
ridge `h` on the collar: `m ≥ 2`, `δ = λ - 5rm > rm(8m - 5)`, `h x` lies in the open ball and
`δ ‖h x‖ ≤ 2mr²` on `[-2r, 2r]` (L5 and L8), `h` is differentiable there (L7), and the transverse
Hessian along the ridge is `δ`-negative definite (L4). -/
theorem ridge_source_facts [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hDf : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hGs : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) - m / 2 * t ^ 2))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) {h : ℝ → E}
    (hhU : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) :
    2 ≤ m ∧ r * m * (8 * m - 5) < lam - 5 * r * m ∧
      (∀ x ∈ Icc (-2 * r) (2 * r), h x ∈ Metric.ball (0 : E) (2 * r)) ∧
      (∀ x ∈ Icc (-2 * r) (2 * r), (lam - 5 * r * m) * ‖h x‖ ≤ 2 * m * r ^ 2) ∧
      (∃ h' : ℝ → E, ∀ x ∈ Icc (-2 * r) (2 * r), HasDerivAt h (h' x) x) ∧
      (∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E, H (x, h x) v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2) := by
  have hUX : Ioo (-2 * r - ε) (2 * r + ε) ⊆ Icc (-2 * r - ε) (2 * r + ε) := Ioo_subset_Icc_self
  have hcapX : Icc (-2 * r) (r / 2) ⊆ Icc (-2 * r - ε) (2 * r + ε) :=
    Icc_subset_Icc (by linarith) (by linarith)
  have hIU : Icc (-2 * r) (2 * r) ⊆ Ioo (-2 * r - ε) (2 * r + ε) :=
    fun t ht => ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  -- the slice derivatives are the transverse parts of `Df`
  have hD : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => f (p.1, y)) ((Df p).comp (ContinuousLinearMap.inr ℝ ℝ E)) p.2 := by
    rintro ⟨x, y⟩ hp
    exact slice_hasFDerivAt (hDf (x, y) hp)
  -- L1–L2: `m ≥ 2`
  have hg0 : ∀ x ∈ Icc (-r / 2) (r / 2),
      HasDerivAt (fun x => f (x, 0)) (Df (x, 0) (1, 0)) x := by
    intro x hx
    have hx' : (x, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
      ⟨⟨by linarith [hx.1], by linarith [hx.2]⟩, h0Y⟩
    have hp : HasDerivAt (fun t : ℝ => (t, (0 : E))) ((1 : ℝ), (0 : E)) x :=
      (hasDerivAt_id x).prodMk (hasDerivAt_const x (0 : E))
    exact (hDf _ hx').comp_hasDerivAt x hp
  have hG0a : Df (-r / 2, 0) (1, 0) = 0 := by simp [hM]
  have hG0c : Df (r / 2, 0) (1, 0) = 0 := by simp [hS]
  have hm : 2 ≤ m := hermite_m_ge_two (g := fun x => f (x, 0)) hr hg0 hG0cv hG0cc hG0a hG0c hnorm
  have hm0 : 0 ≤ m := by linarith
  -- L3–L4 on the full domain
  have hlip := hessian_increment_D hr hε hHx hHxb hHr hHrb
  have hneg := transverse_hessian_le_D hm0 hε hεr hlip hlam
  have hconcX := slices_strongConcave_D hm0 hε hεr hD hH hlip hlam
  have h11 : 11 * (r * m) < lam - 5 * r * m := by
    nlinarith [mul_nonneg (mul_nonneg hr.le hm0) (sub_nonneg.2 hm)]
  have hrm : 0 < r * m := mul_pos hr (by linarith)
  have hδ0 : 0 < lam - 5 * r * m := by linarith
  -- L5 on the full domain: `‖w x‖ ≤ 3mr² < 2rδ`
  have hwv : ∀ (t : ℝ) (v : E),
      ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v = Df (t, 0) (0, v) := fun t v => by simp
  have hwcv' : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v + m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcv v).congr fun t _ => by simp only [hwv]
  have hwcc' : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v - m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcc v).congr fun t _ => by simp only [hwv]
  have hwa : (Df (-r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hM]
  have hwc : (Df (r / 2, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := by simp [hS]
  have hwlt : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε),
      ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ < (lam - 5 * r * m) * (2 * r) := by
    intro x hx
    have h1 := slice_gradient_bound (w := fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      hm0 hwcv' hwcc' (show -r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩)
      (show r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩) (by linarith)
      hwa hwc hx
    have h2 : |(x - -r / 2) * (x - r / 2)| ≤ 6 * r ^ 2 := by
      have hxx : 0 ≤ (x - (-2 * r - ε)) * (2 * r + ε - x) :=
        mul_nonneg (by linarith [hx.1]) (by linarith [hx.2])
      have hee : 0 ≤ (r / 2 - ε) * (9 * r / 2 + ε) :=
        mul_nonneg (by linarith) (by linarith)
      rw [abs_le]; constructor <;> nlinarith [sq_nonneg x, sq_nonneg r]
    have h3 : ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ 3 * m * r ^ 2 :=
      calc ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖
          ≤ m / 2 * |(x - -r / 2) * (x - r / 2)| := h1
        _ ≤ m / 2 * (6 * r ^ 2) := by gcongr
        _ = 3 * m * r ^ 2 := by ring
    have h4 : 22 * r * (r * m) < (lam - 5 * r * m) * (2 * r) := by nlinarith
    nlinarith
  have hslice0 : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε),
      HasFDerivAt (fun y => f (x, y)) ((Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) 0 :=
    fun x hx => hD (x, 0) ⟨hx, h0Y⟩
  have hball : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => (hhU x hx).1
  have hcrit : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
      HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x) := fun x hx => (hhU x hx).2
  -- L8: the ridge is in the open ball on `[-2r, 2r]`
  have hint : ∀ x ∈ Icc (-2 * r) (2 * r), h x ∈ Metric.ball (0 : E) (2 * r) := by
    intro x hx
    have hxU := hIU hx
    have h8 := slice_ridge_bound (hconcX x (hUX hxU)) h0Y (hball x hxU) (hcrit x hxU)
      (hslice0 x (hUX hxU))
    have h9 := hwlt x (hUX hxU)
    rw [Metric.mem_ball, dist_zero_right]
    have : (lam - 5 * r * m) * ‖h x‖ < (lam - 5 * r * m) * (2 * r) := by linarith
    exact lt_of_mul_lt_mul_left this hδ0.le
  have hridgeX : ∀ x ∈ Icc (-2 * r) (2 * r),
      (x, h x) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => ⟨hUX (hIU hx), hball x (hIU hx)⟩
  -- L7 regularity: the transverse part of `DG` is `H`, negative definite
  have hDGH : ∀ x ∈ Icc (-2 * r) (2 * r),
      DG (x, h x) ∘L ContinuousLinearMap.inr ℝ ℝ E = H (x, h x) := by
    intro x hx
    have h1 : HasFDerivAt (fun y => (Df (x, y)).comp (ContinuousLinearMap.inr ℝ ℝ E))
        ((DG (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (h x) :=
      (hGs _ (hridgeX x hx)).hasFDerivAt.comp (h x) (hasFDerivAt_prodMk_right x (h x))
    exact h1.unique (hH _ (hridgeX x hx))
  have hnegR : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E,
      (DG (x, h x) ∘L ContinuousLinearMap.inr ℝ ℝ E) v v ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
    intro x hx v
    rw [hDGH x hx]
    exact hneg _ (hridgeX x hx) v
  have hG : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ᶠ v in 𝓝 (x, h x),
      HasFDerivAt (fun y => f (v.1, y)) ((Df v).comp (ContinuousLinearMap.inr ℝ ℝ E)) v.2 := by
    intro x hx
    have hopen : IsOpen (Ioo (-2 * r - ε) (2 * r + ε) ×ˢ Metric.ball (0 : E) (2 * r)) :=
      isOpen_Ioo.prod Metric.isOpen_ball
    filter_upwards [hopen.mem_nhds ⟨hIU hx, hint x hx⟩] with v hv
    exact hD v ⟨hUX hv.1, Metric.ball_subset_closedBall hv.2⟩
  have hstrict := ridge_differentiable (f := f)
    (G := fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG := fun x => DG (x, h x))
    (h := h) (U := Ioo (-2 * r - ε) (2 * r + ε)) hδ0 isOpen_Ioo hIU
    (fun x hx => hconcX x (hUX hx)) hball hcrit hint hG
    (fun x hx => hGs _ (hridgeX x hx)) hnegR
  have hmemA : -r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) := ⟨by linarith, by linarith⟩
  have hmemC : r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) := ⟨by linarith, by linarith⟩
  have hw2 : ∀ x ∈ Icc (-2 * r) (2 * r),
      ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ 2 * m * r ^ 2 := by
    intro x hx
    have h1 := slice_gradient_bound
      (w := fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) hm0 hwcv' hwcc' hmemA hmemC
      (by linarith) hwa hwc (hUX (hIU hx))
    have h2 : |(x - -r / 2) * (x - r / 2)| ≤ 4 * r ^ 2 := by
      have hxx : 0 ≤ (x - -2 * r) * (2 * r - x) :=
        mul_nonneg (by linarith [hx.1]) (by linarith [hx.2])
      rw [abs_le]; constructor <;> nlinarith [sq_nonneg x, sq_nonneg r]
    calc ‖(Df (x, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖
        ≤ m / 2 * |(x - -r / 2) * (x - r / 2)| := h1
      _ ≤ m / 2 * (4 * r ^ 2) := by gcongr
      _ = 2 * m * r ^ 2 := by ring
  refine ⟨hm, by nlinarith, hint, fun x hx =>
    (slice_ridge_bound (hconcX x (hUX (hIU hx))) h0Y (hball x (hIU hx)) (hcrit x (hIU hx))
      (hslice0 x (hUX (hIU hx)))).trans (hw2 x hx),
    ⟨_, fun x hx => (hstrict x hx).hasFDerivAt.hasDerivAt⟩,
    fun x hx v => hneg _ (hridgeX x hx) v⟩


/-- **The cap from SP's `C³` source data (L1–L20 composed; L12 proved).** As
`ridge_capHyp_source`, with L12's `hconv` derived instead of assumed. Additionally assume, on the
collar: a second derivative `D2f` of `f`, third derivatives `T3 p a b` of `p ↦ D²f p [a, b]`, every
third-derivative block other than `f_xxx` of norm at most `m` (combined form), and `f_xxx ≥ 7/4` on
`D = [-2r, 2r] × B̄(0, 2r)` (SP's L4, from L2 and (H2), on SP's domain). Then the ridge exists, and every ridge has `h(∓r/2) = 0` and yields
(C1)–(C3) and (C2⁺) on the cap. Inside: L5 gives `‖w‖ ≤ 2mr²` on `[-2r, 2r]`, L8 `‖h‖ ≤ 2mr²/δ`, L6
`‖w'‖ ≤ 2mr`, L3 in `y` `‖∂_x∇_y f(x, h)‖ ≤ ‖w'‖ + m‖h‖`, L9 `‖h'‖ ≤ 2s + 2s²` with `s = mr/δ`,
`trilinear_lower` and `l12_constants` give `F'' ≥ 1/4` in the form of `ridge_hconv`. -/
theorem ridge_capHyp_C3 [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ}
    {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hDf : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hD2 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt Df (D2f p) p)
    (hD3 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 p a b) p)
    (hT : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hxxx : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      7 / 4 ≤ T3 p (1, 0) (1, 0) (1, 0))
    (hGs : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) - m / 2 * t ^ 2))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  obtain ⟨hex, hcapS⟩ := ridge_capHyp_source hr hε hεr hDf hGs hH hHx hHxb hHr hHrb hlam hH1 hM hS
    hG0cv hG0cc hwcv hwcc hnorm
  refine ⟨hex, fun h hhU => hcapS h hhU ?_⟩
  obtain ⟨hm, hδ, hint, hh8, ⟨h', hhd⟩, hnegH⟩ := ridge_source_facts hr hε hεr hDf hGs hH hHx hHxb
    hHr hHrb hlam hH1 hM hS hG0cv hG0cc hwcv hwcc hnorm hhU
  have hm0 : 0 ≤ m := by linarith
  have hδ0 : 0 < lam - 5 * r * m := by
    have : 0 < r * m * (8 * m - 5) := mul_pos (mul_pos hr (by linarith)) (by linarith)
    linarith
  have hUX : Ioo (-2 * r - ε) (2 * r + ε) ⊆ Icc (-2 * r - ε) (2 * r + ε) := Ioo_subset_Icc_self
  have hIU : Icc (-2 * r) (2 * r) ⊆ Ioo (-2 * r - ε) (2 * r + ε) :=
    fun t ht => ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  have hball : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => (hhU x hx).1
  have hridgeX : ∀ x ∈ Icc (-2 * r) (2 * r),
      (x, h x) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun x hx => ⟨hUX (hIU hx), hball x (hIU hx)⟩
  have hridgeU : ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
      (Df (x, h x)).comp (ContinuousLinearMap.inr ℝ ℝ E) = 0 := fun x hx =>
    (slice_hasFDerivAt (hDf _ ⟨hUX hx, hball x hx⟩)).unique (hhU x hx).2
  have hD2r : ∀ x ∈ Icc (-2 * r) (2 * r), HasFDerivAt Df (D2f (x, h x)) (x, h x) :=
    fun x hx => hD2 _ (hridgeX x hx)
  have hsym : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v w, D2f (x, h x) v w = D2f (x, h x) w v := by
    intro x hx v w
    have hopen : IsOpen (Ioo (-2 * r - ε) (2 * r + ε) ×ˢ Metric.ball (0 : E) (2 * r)) :=
      isOpen_Ioo.prod Metric.isOpen_ball
    have hev : ∀ᶠ q in 𝓝 (x, h x), HasFDerivAt f (Df q) q := by
      filter_upwards [hopen.mem_nhds ⟨hIU hx, hint x hx⟩] with q hq
      exact hDf q ⟨hUX hq.1, Metric.ball_subset_closedBall hq.2⟩
    exact second_derivative_symmetric_of_eventually hev (hD2r x hx) v w
  have hnegD : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ v : E,
      D2f (x, h x) (0, v) (0, v) ≤ -(lam - 5 * r * m) * ‖v‖ ^ 2 := by
    intro x hx v
    rw [← transverse_hessian_eq (hH _ (hridgeX x hx)) (hD2r x hx) v v]
    exact hnegH x hx v
  -- L6: `‖w' x‖ ≤ 2mr`
  have hwcv' : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v + m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcv v).congr fun t _ => by simp
  have hwcc' : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => ((Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) v - m * ‖v‖ / 2 * t ^ 2) :=
    fun v => (hwcc v).congr fun t _ => by simp
  have hw6 : ∀ x ∈ Icc (-2 * r) (2 * r),
      ‖(D2f (x, 0) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ 2 * m * r := by
    intro x hx
    have hp : HasDerivAt (fun t : ℝ => (t, (0 : E))) ((1 : ℝ), (0 : E)) x :=
      (hasDerivAt_id x).prodMk (hasDerivAt_const x (0 : E))
    have h1 : HasDerivAt (fun t => Df (t, 0)) (D2f (x, 0) (1, 0)) x :=
      (hD2 _ ⟨hUX (hIU hx), h0Y⟩).comp_hasDerivAt x hp
    have hd : HasDerivAt (fun t => (Df (t, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E))
        ((D2f (x, 0) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)) x := by
      simpa using h1.clm_comp (hasDerivAt_const x (ContinuousLinearMap.inr ℝ ℝ E))
    have h2 := slice_gradient_deriv_bound hr hm0 hwcv' hwcc'
      (show -r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩)
      (show r / 2 ∈ Icc (-2 * r - ε) (2 * r + ε) from ⟨by linarith, by linarith⟩)
      (by simp [hM]) (by simp [hS]) (hUX (hIU hx)) hd
    have h3 : max |x| (r / 2) ≤ 2 * r :=
      max_le (abs_le.mpr ⟨by linarith [hx.1], by linarith [hx.2]⟩) (by linarith)
    calc ‖(D2f (x, 0) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ m * max |x| (r / 2) := h2
      _ ≤ m * (2 * r) := mul_le_mul_of_nonneg_left h3 hm0
      _ = 2 * m * r := by ring
  -- L3 in `y`, then L9
  have hu : ∀ x ∈ Icc (-2 * r) (2 * r), ‖h' x‖ ≤
      2 * (m * r / (lam - 5 * r * m)) + 2 * (m * r / (lam - 5 * r * m)) ^ 2 := by
    intro x hx
    have hseg : ∀ s ∈ Icc (0 : ℝ) 1,
        (x, s • h x) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) := by
      intro s hs
      have hhY : ‖h x‖ ≤ 2 * r := by
        simpa [Metric.mem_closedBall, dist_zero_right] using hball x (hIU hx)
      refine ⟨hUX (hIU hx), ?_⟩
      rw [Metric.mem_closedBall, dist_zero_right, norm_smul, Real.norm_eq_abs, abs_of_nonneg hs.1]
      nlinarith [hs.2, norm_nonneg (h x)]
    have hinc := dx_grad_increment (D2f := D2f) (T3 := T3) (x := x) (y := h x) hm0
      (fun s hs => hD3 _ (hseg s hs))
      (fun s hs v => by
        have := hT _ (hseg s hs) 1 0 0 0 v (h x)
        simpa [mul_assoc] using this)
    have hBn : ‖(D2f (x, h x) (1, 0)).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤
        2 * m * r + m * ‖h x‖ :=
      (norm_le_norm_add_norm_sub' _ _).trans (add_le_add (hw6 x hx) hinc)
    have hid := ridge_second_identity
      (Filter.eventually_of_mem (isOpen_Ioo.mem_nhds (hIU hx)) hridgeU) (hhd x hx) (hD2r x hx)
    obtain ⟨h9, -, -⟩ := schur_max hδ0 (hnegD x hx) hid
    exact l9_arith hδ0 hm0 h9 hBn (hh8 x hx)
  -- L12
  have hc := l12_constants hr hm hδ
  have hΦ : ∀ x ∈ Icc (-2 * r) (2 * r), ∀ b : E,
      HasDerivAt (fun s => D2f (s, h s) (1, b) (1, b)) (T3 (x, h x) (1, b) (1, b) (1, h' x)) x := by
    intro x hx b
    have hp : HasDerivAt (fun s => (s, h s)) ((1 : ℝ), h' x) x := (hasDerivAt_id x).prodMk (hhd x hx)
    exact (hD3 _ (hridgeX x hx) (1, b) (1, b)).comp_hasDerivAt x hp
  exact ridge_hconv (Φ := fun x b => T3 (x, h x) (1, b) (1, b) (1, h' x)) hδ0 isOpen_Ioo hIU
    hridgeU hhd hD2r hsym hnegD hu hΦ
    (fun x hx b hb => l12_lower hm0 (hT _ (hridgeX x hx)) (hxxx _ ⟨hx, hball x (hIU hx)⟩) hc
      (hu x hx) hb)


/-! ### L2's second statement and (H2) (v2.0)

SP's L4 lower bound `f_xxx ≥ 7/4` was the last analytic hypothesis of `ridge_capHyp_C3`. It follows
from the normalization (some point of the pin interval has `f_xxx ≥ 2`, by a one-sided Hermite bound)
and (H2) `rn ≤ 1/20` with a bound `n` on the derivative of `f_xxx`. `ridge_capHyp_C4` takes SP's
hypotheses only. -/

/-- **L1, one-sided.** As `hermite_gap_le`, from the concave half alone (`G'' ≤ m` weakly):
`g(a) - g(c) ≤ m (c - a)³/12`. -/
theorem hermite_gap_le_concave {g G : ℝ → ℝ} {a c m : ℝ} (hac : a < c)
    (hg : ∀ x ∈ Icc a c, HasDerivAt g (G x) x)
    (hcc : ConcaveOn ℝ (Icc a c) (fun t => G t - m / 2 * t ^ 2))
    (hGa : G a = 0) (hGc : G c = 0) :
    g a - g c ≤ m * (c - a) ^ 3 / 12 := by
  have hpoly : ∀ x, HasDerivAt (fun t => (c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
      ((x - a) * (c - x)) x := by
    intro x
    have h1 : HasDerivAt (fun t => t - a) 1 x := (hasDerivAt_id x).sub_const a
    have h2 := ((h1.pow 2).const_mul (c - a)).div_const 2
    have h3 := (h1.pow 3).div_const 3
    convert h2.sub h3 using 1
    push_cast
    ring
  set k : ℝ → ℝ := fun t => g t + m / 2 * ((c - a) * (t - a) ^ 2 / 2 - (t - a) ^ 3 / 3)
    with hk
  have hkd : ∀ x ∈ Icc a c, HasDerivAt k (G x + m / 2 * ((x - a) * (c - x))) x :=
    fun x hx => (hg x hx).add ((hpoly x).const_mul (m / 2))
  have hcv' : ConvexOn ℝ (Icc a c) (fun t => -G t + m / 2 * t ^ 2) :=
    hcc.neg.congr fun t _ => by simp only [Pi.neg_apply]; ring
  have hχ := convexOn_add_node (a := a) (c := c) hcv'
  have hmono : MonotoneOn k (Icc a c) := by
    refine monotoneOn_of_hasDerivWithinAt_nonneg (convex_Icc a c)
      (fun x hx => (hkd x hx).continuousAt.continuousWithinAt)
      (fun x hx => (hkd x (interior_subset hx)).hasDerivWithinAt) (fun x hx => ?_)
    have hx := interior_subset hx
    have h0 := convex_zeros_between hχ (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac.le
      (by rw [hGa]; ring) (by rw [hGc]; ring) hx
    nlinarith
  have hkac := hmono (left_mem_Icc.mpr hac.le) (right_mem_Icc.mpr hac.le) hac.le
  simp only [hk] at hkac
  have e0 : m / 2 * ((c - a) * (a - a) ^ 2 / 2 - (a - a) ^ 3 / 3) = 0 := by ring
  have e1 : m / 2 * ((c - a) * (c - a) ^ 2 / 2 - (c - a) ^ 3 / 3) = m * (c - a) ^ 3 / 12 := by
    ring
  linarith

/-- **L2, second statement.** Under SP's normalization `g(-r/2) - g(r/2) = r³/6` with critical pins,
if `g''' = φ` is continuous on the pin interval, then `φ(x₀) ≥ 2` for some `x₀` there (SP: the
`φ`-weighted average of `f_xxx` is `2`). -/
theorem third_deriv_exists_two {g G G' φ : ℝ → ℝ} {r : ℝ} (hr : 0 < r)
    (hg : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt g (G x) x)
    (hG : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt G (G' x) x)
    (hG' : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt G' (φ x) x)
    (hφ : ContinuousOn φ (Icc (-r / 2) (r / 2)))
    (hGa : G (-r / 2) = 0) (hGc : G (r / 2) = 0) (hgap : g (-r / 2) - g (r / 2) = r ^ 3 / 6) :
    ∃ x₀ ∈ Icc (-r / 2) (r / 2), 2 ≤ φ x₀ := by
  by_contra hcon
  simp only [not_exists, not_and, not_le] at hcon
  have hne : (Icc (-r / 2) (r / 2)).Nonempty := ⟨-r / 2, left_mem_Icc.mpr (by linarith)⟩
  obtain ⟨x₁, hx₁, hmax⟩ := isCompact_Icc.exists_isMaxOn hne hφ
  set M := φ x₁ with hM
  have hcc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => G t - M / 2 * t ^ 2) := by
    refine concaveOn_of_hasDerivWithinAt2_nonpos (convex_Icc _ _) (f' := fun t => G' t - M * t)
      (f'' := fun t => φ t - M)
      (fun t ht => ((hG t ht).sub ((hasDerivAt_pow 2 t).const_mul (M / 2))).continuousAt.continuousWithinAt)
      (fun t ht => ?_) (fun t ht => ?_) (fun t ht => ?_)
    · have := (hG t (interior_subset ht)).sub ((hasDerivAt_pow 2 t).const_mul (M / 2))
      convert this.hasDerivWithinAt using 1
      push_cast
      ring
    · have := (hG' t (interior_subset ht)).sub ((hasDerivAt_id' t).const_mul M)
      convert this.hasDerivWithinAt using 1
      ring
    · have h' : φ t ≤ φ x₁ := hmax (interior_subset ht)
      show φ t - M ≤ 0
      linarith
  have h1 := hermite_gap_le_concave (by linarith) hg hcc hGa hGc
  have hcr : r / 2 - -r / 2 = r := by ring
  rw [hgap, hcr] at h1
  have h2 := hcon x₁ hx₁
  have hr3 : 0 < r ^ 3 := by positivity
  nlinarith

/-- **L4 (`f_xxx ≥ 7/4`).** Let `ψ = f_xxx` have derivative `Dψ` on SP's domain
`D = [-2r, 2r] × B̄(0, 2r)`, with the two fourth-order blocks bounded by `n = M_4` there:
`|∂_x ψ| = |Dψ (1, 0)| ≤ n` and `‖D_y ψ‖ = ‖Dψ ∘ inr‖ ≤ n`. If `ψ(x₀, 0) ≥ 2` at a point of the pin
interval and (H2) `rn ≤ 1/20`, then `ψ ≥ 7/4` on `D` (the proof bounds `ψ ≥ 2 - 9rn/2`; the
statement exports `7/4`). As in SP's L3, move first in `x`
along `y = 0` (`|Δx| ≤ 5r/2`), then along the ray to `y` (`‖Δy‖ ≤ 2r`); SP rounds `9r/2` up to `5r`. -/
theorem fxxx_lower {ψ : ℝ × E → ℝ} {Dψ : ℝ × E → ℝ × E →L[ℝ] ℝ} {r n x₀ : ℝ} (hr : 0 < r)
    (hψ : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt ψ (Dψ p) p)
    (hnx : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), |Dψ p (1, 0)| ≤ n)
    (hny : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ‖(Dψ p).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ n)
    (hx₀ : x₀ ∈ Icc (-r / 2) (r / 2)) (h2 : 2 ≤ ψ (x₀, 0)) (hH2 : r * n ≤ 1 / 20) :
    ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), 7 / 4 ≤ ψ p := by
  rintro ⟨x, y⟩ ⟨hx, hy⟩
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  have hyn : ‖y‖ ≤ 2 * r := by simpa [Metric.mem_closedBall, dist_zero_right] using hy
  have hn0 : 0 ≤ n := (abs_nonneg _).trans (hnx (x, y) ⟨hx, hy⟩)
  -- along `y = 0`
  have hx₀D : x₀ ∈ Icc (-2 * r) (2 * r) := ⟨by linarith [hx₀.1], by linarith [hx₀.2]⟩
  have hd1 : ∀ t ∈ Icc (-2 * r) (2 * r),
      HasDerivAt (fun t : ℝ => ψ (t, 0)) (Dψ (t, 0) (1, 0)) t := by
    intro t ht
    have hp : HasDerivAt (fun s : ℝ => (s, (0 : E))) ((1 : ℝ), (0 : E)) t :=
      (hasDerivAt_id t).prodMk (hasDerivAt_const t (0 : E))
    exact (hψ (t, 0) ⟨ht, h0Y⟩).comp_hasDerivAt t hp
  have h1 := (convex_Icc (-2 * r) (2 * r)).norm_image_sub_le_of_norm_hasDerivWithin_le
    (f := fun t : ℝ => ψ (t, 0)) (fun t ht => (hd1 t ht).hasDerivWithinAt)
    (fun t ht => by rw [Real.norm_eq_abs]; exact hnx (t, 0) ⟨ht, h0Y⟩) hx₀D hx
  -- along the ray to `y`
  have hseg : ∀ s ∈ Icc (0 : ℝ) 1, (x, s • y) ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) := by
    intro s hs
    refine ⟨hx, ?_⟩
    rw [Metric.mem_closedBall, dist_zero_right, norm_smul, Real.norm_eq_abs, abs_of_nonneg hs.1]
    nlinarith [hs.2, norm_nonneg y]
  have hd2 : ∀ s ∈ Icc (0 : ℝ) 1,
      HasDerivAt (fun s : ℝ => ψ (x, s • y)) (Dψ (x, s • y) (0, y)) s := by
    intro s hs
    have hp : HasDerivAt (fun s : ℝ => (x, s • y)) ((0 : ℝ), y) s := by
      simpa using (hasDerivAt_const s x).prodMk ((hasDerivAt_id s).smul_const y)
    exact (hψ _ (hseg s hs)).comp_hasDerivAt s hp
  have hb2 : ∀ s ∈ Icc (0 : ℝ) 1, ‖Dψ (x, s • y) (0, y)‖ ≤ n * ‖y‖ := by
    intro s hs
    have := ((Dψ (x, s • y)).comp (ContinuousLinearMap.inr ℝ ℝ E)).le_opNorm y
    simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.inr_apply] at this
    exact this.trans (mul_le_mul_of_nonneg_right (hny _ (hseg s hs)) (norm_nonneg y))
  have h2' := (convex_Icc (0 : ℝ) 1).norm_image_sub_le_of_norm_hasDerivWithin_le
    (f := fun s : ℝ => ψ (x, s • y)) (fun s hs => (hd2 s hs).hasDerivWithinAt) hb2
    (left_mem_Icc.mpr zero_le_one) (right_mem_Icc.mpr zero_le_one)
  simp only [one_smul, zero_smul, sub_zero, norm_one, mul_one] at h2'
  have h1' : |ψ (x, 0) - ψ (x₀, 0)| ≤ n * |x - x₀| := by simpa [Real.norm_eq_abs] using h1
  rw [Real.norm_eq_abs] at h2'
  have hdx : |x - x₀| ≤ 5 * r / 2 := by
    rw [abs_le]; constructor <;> linarith [hx.1, hx.2, hx₀.1, hx₀.2]
  have e1 := (abs_le.mp h1').1
  have e2 := (abs_le.mp h2').1
  have h5 : n * |x - x₀| ≤ n * (5 * r / 2) := mul_le_mul_of_nonneg_left hdx hn0
  have h6 : n * ‖y‖ ≤ n * (2 * r) := mul_le_mul_of_nonneg_left hyn hn0
  nlinarith

/-- **The cap from SP's hypotheses (H1), (H2) and the derivative bounds (L1–L20 composed).** As
`ridge_capHyp_C3`, with `f_xxx ≥ 7/4` derived instead of assumed: from a derivative `Φ4` of `f_xxx`
on `D = [-2r, 2r] × B̄(0, 2r)` whose two blocks `∂_x f_xxx` and `D_y f_xxx` are bounded by
`n = M_4` there, as in SP, and (H2) `rn ≤ 1/20`. L2's second statement gives `f_xxx(x₀, 0) ≥ 2` on
the pin interval (`third_deriv_exists_two`), and `fxxx_lower` gives `f_xxx ≥ 7/4` on `D`. -/
theorem ridge_capHyp_C4 [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {Φ4 : ℝ × E → ℝ × E →L[ℝ] ℝ} {n : ℝ}
    {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hDf : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hD2 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt Df (D2f p) p)
    (hD3 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 p a b) p)
    (hT : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hD4 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun q => T3 q (1, 0) (1, 0) (1, 0)) (Φ4 p) p)
    (hn4x : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), |Φ4 p (1, 0)| ≤ n)
    (hn4y : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ‖(Φ4 p).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ n)
    (hH2 : r * n ≤ 1 / 20)
    (hGs : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hG0cv : ConvexOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) + m / 2 * t ^ 2))
    (hG0cc : ConcaveOn ℝ (Icc (-r / 2) (r / 2)) (fun t => Df (t, 0) (1, 0) - m / 2 * t ^ 2))
    (hwcv : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2))
    (hwcc : ∀ v : E, ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
      (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2))
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  have hmem : ∀ t ∈ Icc (-r / 2) (r / 2), (t, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) := fun t ht =>
    ⟨⟨by linarith [ht.1], by linarith [ht.2]⟩, Metric.mem_closedBall_self (by positivity)⟩
  have hp : ∀ t : ℝ, HasDerivAt (fun s : ℝ => (s, (0 : E))) ((1 : ℝ), (0 : E)) t := fun t =>
    (hasDerivAt_id t).prodMk (hasDerivAt_const t (0 : E))
  have hg : ∀ x ∈ Icc (-r / 2) (r / 2), HasDerivAt (fun x => f (x, 0)) (Df (x, 0) (1, 0)) x := by
    intro x hx
    exact (hDf _ (hmem x hx)).comp_hasDerivAt x (hp x)
  have hG : ∀ x ∈ Icc (-r / 2) (r / 2),
      HasDerivAt (fun t => Df (t, 0) (1, 0)) (D2f (x, 0) (1, 0) (1, 0)) x := by
    intro x hx
    have h1 : HasDerivAt (fun t => Df (t, 0)) (D2f (x, 0) (1, 0)) x :=
      (hD2 _ (hmem x hx)).comp_hasDerivAt x (hp x)
    simpa using h1.clm_apply (hasDerivAt_const x ((1 : ℝ), (0 : E)))
  have hG' : ∀ x ∈ Icc (-r / 2) (r / 2),
      HasDerivAt (fun t => D2f (t, 0) (1, 0) (1, 0)) (T3 (x, 0) (1, 0) (1, 0) (1, 0)) x := by
    intro x hx
    exact (hD3 _ (hmem x hx) (1, 0) (1, 0)).comp_hasDerivAt x (hp x)
  have hc : Continuous (fun s : ℝ => (s, (0 : E))) := continuous_id.prodMk continuous_const
  have hmemD : ∀ t ∈ Icc (-r / 2) (r / 2), (t, (0 : E)) ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) := fun t ht =>
    ⟨⟨by linarith [ht.1], by linarith [ht.2]⟩, Metric.mem_closedBall_self (by positivity)⟩
  have hφc : ContinuousOn (fun t => T3 (t, 0) (1, 0) (1, 0) (1, 0)) (Icc (-r / 2) (r / 2)) := by
    intro t ht
    have h1 : ContinuousAt (fun q => T3 q (1, 0) (1, 0) (1, 0)) (t, (0 : E)) :=
      (hD4 _ (hmemD t ht)).continuousAt
    have h2 : ContinuousAt (fun s : ℝ => T3 (s, 0) (1, 0) (1, 0) (1, 0)) t :=
      ContinuousAt.comp (g := fun q => T3 q (1, 0) (1, 0) (1, 0)) (f := fun s : ℝ => (s, (0 : E)))
        h1 hc.continuousAt
    exact h2.continuousWithinAt
  have hGa : Df (-r / 2, 0) (1, 0) = 0 := by simp [hM]
  have hGc : Df (r / 2, 0) (1, 0) = 0 := by simp [hS]
  have hxxx : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), 7 / 4 ≤ T3 p (1, 0) (1, 0) (1, 0) := by
    obtain ⟨x₀, hx₀, h2⟩ := third_deriv_exists_two hr hg hG hG' hφc hGa hGc hnorm
    exact fxxx_lower (ψ := fun q => T3 q (1, 0) (1, 0) (1, 0)) hr hD4 hn4x hn4y hx₀ h2 hH2
  exact ridge_capHyp_C3 hr hε hεr hDf hD2 hD3 hT hxxx hGs hH hHx hHxb hHr hHrb hlam hH1 hM hS
    hG0cv hG0cc hwcv hwcc hnorm


/-! ### The derivative bounds from the block bound, and SP's bounds on `D` (v2.1)

Up to v2.0 the capstones took six derivative bounds besides the block bound `hT`:
- `hHxb` and `hHrb`, the inputs of L3;
- `hwcv` and `hwcc`, the envelope inputs of L5–L6;
- `hG0cv` and `hG0cc`, the envelope inputs of L1–L2.

All six follow from `hT` and from `|f_xxx| ≤ m` on the pin segment, and `ridge_capHyp_C5` derives
them. Up to v2.0 the bounds `m` were also assumed on a collar `[−2r − ε, 2r + ε]` of SP's `D`.
`ridge_capHyp_D` assumes them on `D` only, through the continuity step `block_bound_collar` (I3). -/

/-- **L3's `x`-input from the block bound.** Let `y ↦ ∂_y f (t, y)` have derivative `H (t, 0)` at `0`
and `Df` have derivative `D2f (t, 0)` at `(t, 0)` for every `t ∈ J`, where `J` is uniquely
differentiable at `x ∈ J`. If `t ↦ H (t, 0)` has derivative `Hx'` at `x`, then
`Hx' v w = T3 (x, 0) [(0, v), (0, w), (1, 0)]`, so `‖Hx'‖ ≤ m` when that block is at most
`m ‖v‖ ‖w‖`. This derives v1.6's hypothesis `hHxb`. -/
theorem hessian_x_bound {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx' : E →L[ℝ] E →L[ℝ] ℝ} {J : Set ℝ} {m x : ℝ}
    (hJ : UniqueDiffWithinAt ℝ J x) (hxJ : x ∈ J) (hm : 0 ≤ m)
    (hH : ∀ t ∈ J, HasFDerivAt (fun y => (Df (t, y)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      (H (t, 0)) 0)
    (hD2 : ∀ t ∈ J, HasFDerivAt Df (D2f (t, 0)) (t, 0))
    (hD3 : ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 (x, 0) a b) (x, 0))
    (hT : ∀ v w : E, |T3 (x, 0) (0, v) (0, w) (1, 0)| ≤ m * (‖v‖ * ‖w‖))
    (hHx : HasDerivAt (fun t => H (t, 0)) Hx' x) : ‖Hx'‖ ≤ m := by
  have hp : HasDerivAt (fun t : ℝ => (t, (0 : E))) ((1 : ℝ), (0 : E)) x :=
    (hasDerivAt_id x).prodMk (hasDerivAt_const x (0 : E))
  have key : ∀ v w : E, Hx' v w = T3 (x, 0) (0, v) (0, w) (1, 0) := by
    intro v w
    have h1 : HasDerivAt (fun t => H (t, 0) v w) (Hx' v w) x := by
      have := (hHx.clm_apply (hasDerivAt_const x v)).clm_apply (hasDerivAt_const x w)
      simpa using this
    have h2 : HasDerivAt (fun t => D2f (t, 0) (0, v) (0, w)) (T3 (x, 0) (0, v) (0, w) (1, 0)) x :=
      HasFDerivAt.comp_hasDerivAt (l := fun q => D2f q (0, v) (0, w)) x (hD3 (0, v) (0, w)) hp
    have h3 : HasDerivWithinAt (fun t => D2f (t, 0) (0, v) (0, w)) (Hx' v w) J x :=
      h1.hasDerivWithinAt.congr_of_mem (fun t ht => (transverse_hessian_eq (hH t ht) (hD2 t ht) v w).symm) hxJ
    exact hJ.eq_deriv _ h3 h2.hasDerivWithinAt
  refine ContinuousLinearMap.opNorm_le_bound _ hm fun v => ?_
  refine ContinuousLinearMap.opNorm_le_bound _ (mul_nonneg hm (norm_nonneg v)) fun w => ?_
  rw [key, Real.norm_eq_abs, mul_assoc]
  exact hT v w

/-- **L3's radial input from the block bound.** Along the radius `τ ↦ (x, τ y)`, `τ ∈ [0, 1]`, if
`τ ↦ H (x, τ y)` has derivative `Hr'` at `s`, then `Hr' v w = T3 (x, s y) [(0, v), (0, w), (0, y)]`,
so `‖Hr'‖ ≤ m ‖y‖` when that block is at most `m ‖v‖ ‖w‖ ‖y‖`. The identification uses derivatives
within `[0, 1]`, so the endpoint `τ = 1` may lie on the sphere `‖y‖ = 2r`. This derives v1.6's
hypothesis `hHrb`. -/
theorem hessian_r_bound {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hr' : E →L[ℝ] E →L[ℝ] ℝ} {x m s : ℝ} {y : E}
    (hm : 0 ≤ m) (hs : s ∈ Icc (0 : ℝ) 1)
    (hH : ∀ τ ∈ Icc (0 : ℝ) 1, HasFDerivAt (fun z => (Df (x, z)).comp (ContinuousLinearMap.inr ℝ ℝ E))
      (H (x, τ • y)) (τ • y))
    (hD2 : ∀ τ ∈ Icc (0 : ℝ) 1, HasFDerivAt Df (D2f (x, τ • y)) (x, τ • y))
    (hD3 : ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 (x, s • y) a b) (x, s • y))
    (hT : ∀ v w : E, |T3 (x, s • y) (0, v) (0, w) (0, y)| ≤ m * (‖v‖ * ‖w‖ * ‖y‖))
    (hHr : HasDerivAt (fun τ : ℝ => H (x, τ • y)) Hr' s) : ‖Hr'‖ ≤ m * ‖y‖ := by
  have hp : HasDerivAt (fun τ : ℝ => (x, τ • y)) ((0 : ℝ), y) s := by
    have := (hasDerivAt_const s x).prodMk ((hasDerivAt_id s).smul_const y)
    simpa using this
  have key : ∀ v w : E, Hr' v w = T3 (x, s • y) (0, v) (0, w) (0, y) := by
    intro v w
    have h1 : HasDerivAt (fun τ => H (x, τ • y) v w) (Hr' v w) s := by
      have := (hHr.clm_apply (hasDerivAt_const s v)).clm_apply (hasDerivAt_const s w)
      simpa using this
    have h2 : HasDerivAt (fun τ : ℝ => D2f (x, τ • y) (0, v) (0, w))
        (T3 (x, s • y) (0, v) (0, w) (0, y)) s :=
      HasFDerivAt.comp_hasDerivAt (l := fun q => D2f q (0, v) (0, w)) s (hD3 (0, v) (0, w)) hp
    have h3 : HasDerivWithinAt (fun τ : ℝ => D2f (x, τ • y) (0, v) (0, w)) (Hr' v w) (Icc 0 1) s :=
      h1.hasDerivWithinAt.congr_of_mem
        (fun τ hτ => (transverse_hessian_eq (hH τ hτ) (hD2 τ hτ) v w).symm) hs
    exact (uniqueDiffOn_Icc zero_lt_one s hs).eq_deriv _ h3 h2.hasDerivWithinAt
  have hmy : 0 ≤ m * ‖y‖ := mul_nonneg hm (norm_nonneg y)
  refine ContinuousLinearMap.opNorm_le_bound _ hmy fun v => ?_
  refine ContinuousLinearMap.opNorm_le_bound _ (mul_nonneg hmy (norm_nonneg v)) fun w => ?_
  rw [key, Real.norm_eq_abs]
  calc |T3 (x, s • y) (0, v) (0, w) (0, y)| ≤ m * (‖v‖ * ‖w‖ * ‖y‖) := hT v w
    _ = m * ‖y‖ * ‖v‖ * ‖w‖ := by ring

/-- **The envelope inputs of L1 and L5–L6 from a second-derivative bound.** If the second derivative
`T3 (t, 0) [(1, 0), u, (1, 0)]` of `t ↦ Df (t, 0) u` has absolute value at most `B` on `[a, c]`, then
`Df (·, 0) u + (B/2) t²` is convex and `Df (·, 0) u − (B/2) t²` is concave there. With `u = (0, v)`
and `B = m‖v‖` (from the block bound) this gives `hwcv` and `hwcc`. With `u = (1, 0)` and `B = m`
(SP: `|f_xxx| ≤ M₃`) it gives `hG0cv` and `hG0cc`. -/
theorem axis_convexConcave {Df : ℝ × E → ℝ × E →L[ℝ] ℝ}
    {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {a c B : ℝ} {u : ℝ × E}
    (hD2 : ∀ t ∈ Icc a c, HasFDerivAt Df (D2f (t, 0)) (t, 0))
    (hD3 : ∀ t ∈ Icc a c, HasFDerivAt (fun q => D2f q (1, 0) u) (T3 (t, 0) (1, 0) u) (t, 0))
    (hB : ∀ t ∈ Icc a c, |T3 (t, 0) (1, 0) u (1, 0)| ≤ B) :
    ConvexOn ℝ (Icc a c) (fun t => Df (t, 0) u + B / 2 * t ^ 2) ∧
      ConcaveOn ℝ (Icc a c) (fun t => Df (t, 0) u - B / 2 * t ^ 2) := by
  have hp : ∀ t : ℝ, HasDerivAt (fun s : ℝ => (s, (0 : E))) ((1 : ℝ), (0 : E)) t := fun t =>
    (hasDerivAt_id t).prodMk (hasDerivAt_const t (0 : E))
  have hg : ∀ t ∈ Icc a c, HasDerivAt (fun s => Df (s, 0) u) (D2f (t, 0) (1, 0) u) t := by
    intro t ht
    have h1 : HasDerivAt (fun s => Df (s, 0)) (D2f (t, 0) (1, 0)) t :=
      (hD2 t ht).comp_hasDerivAt t (hp t)
    simpa using h1.clm_apply (hasDerivAt_const t u)
  have hg' : ∀ t ∈ Icc a c, HasDerivAt (fun s => D2f (s, 0) (1, 0) u) (T3 (t, 0) (1, 0) u (1, 0)) t :=
    fun t ht => HasFDerivAt.comp_hasDerivAt (l := fun q => D2f q (1, 0) u) t (hD3 t ht) (hp t)
  have hcont : ContinuousOn (fun s => Df (s, 0) u) (Icc a c) :=
    fun t ht => (hg t ht).continuousAt.continuousWithinAt
  have hsq : ∀ t : ℝ, HasDerivAt (fun s : ℝ => B / 2 * s ^ 2) (B * t) t := by
    intro t
    have := (hasDerivAt_pow 2 t).const_mul (B / 2)
    convert this using 1
    ring
  have hlin : ∀ t : ℝ, HasDerivAt (fun s : ℝ => B * s) B t := by
    intro t
    simpa using (hasDerivAt_id t).const_mul B
  have hIo : ∀ t ∈ interior (Icc a c), t ∈ Icc a c := fun t ht => interior_subset ht
  constructor
  · refine convexOn_of_hasDerivWithinAt2_nonneg (convex_Icc a c)
      (f' := fun t => D2f (t, 0) (1, 0) u + B * t) (f'' := fun t => T3 (t, 0) (1, 0) u (1, 0) + B)
      (hcont.add (continuousOn_const.mul (continuousOn_id.pow 2))) ?_ ?_ ?_
    · intro t ht
      exact ((hg t (hIo t ht)).add (hsq t)).hasDerivWithinAt
    · intro t ht
      exact ((hg' t (hIo t ht)).add (hlin t)).hasDerivWithinAt
    · intro t ht
      have := hB t (hIo t ht)
      have := neg_abs_le (T3 (t, 0) (1, 0) u (1, 0))
      show 0 ≤ T3 (t, 0) (1, 0) u (1, 0) + B
      linarith
  · refine concaveOn_of_hasDerivWithinAt2_nonpos (convex_Icc a c)
      (f' := fun t => D2f (t, 0) (1, 0) u - B * t) (f'' := fun t => T3 (t, 0) (1, 0) u (1, 0) - B)
      (hcont.sub (continuousOn_const.mul (continuousOn_id.pow 2))) ?_ ?_ ?_
    · intro t ht
      exact ((hg t (hIo t ht)).sub (hsq t)).hasDerivWithinAt
    · intro t ht
      exact ((hg' t (hIo t ht)).sub (hlin t)).hasDerivWithinAt
    · intro t ht
      have := hB t (hIo t ht)
      have := le_abs_self (T3 (t, 0) (1, 0) u (1, 0))
      show T3 (t, 0) (1, 0) u (1, 0) - B ≤ 0
      linarith

/-- The block weight `∏ (|sᵢ| + ‖yᵢ‖) − ∏ |sᵢ|` is nonnegative. -/
theorem block_nonneg {F : Type*} [SeminormedAddCommGroup F] (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : F) :
    0 ≤ (|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃| := by
  have a1 : |s₁| ≤ |s₁| + ‖y₁‖ := le_add_of_nonneg_right (norm_nonneg _)
  have a2 : |s₂| ≤ |s₂| + ‖y₂‖ := le_add_of_nonneg_right (norm_nonneg _)
  have a3 : |s₃| ≤ |s₃| + ‖y₃‖ := le_add_of_nonneg_right (norm_nonneg _)
  have := mul_le_mul (mul_le_mul a1 a2 (abs_nonneg _) (by positivity)) a3 (abs_nonneg _)
    (by positivity)
  linarith

/-- **The continuity step of I3: the block bound from `D` to a collar.** Assume the block bound holds
with `m` on SP's `D = [−2r, 2r] × B̄(0, 2r)`. Assume also that on the collar
`[−2r − ε₀, 2r + ε₀] × B̄(0, 2r)` the `x`-derivative `T4` of `T3` exists and satisfies the same block
bound with some `K ≥ 0` (SP: `f ∈ C⁴` near `D`, so `‖D⁴ f‖` is bounded on a compact collar). Then the
block bound holds with `m + Kε` on the collar of width `ε ≤ ε₀`. Each collar point `(x, y)` is
within `ε` of a point `(x₀, y) ∈ D`, and the mean value inequality along `x` gives the rest. -/
theorem block_bound_collar {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ}
    {T4 : ℝ × E → ℝ × E → ℝ × E → ℝ × E → ℝ} {r m K ε₀ ε : ℝ} (hr : 0 < r) (hK : 0 ≤ K)
    (hε : 0 ≤ ε) (hεε : ε ≤ ε₀)
    (hT : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hD4x : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ a b c : ℝ × E, HasDerivAt (fun t => T3 (t, p.2) a b c) (T4 p a b c) p.1)
    (hT4 : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T4 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T4 p (1, 0) (1, 0) (1, 0)| ≤
        K * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|)) :
    ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        (m + K * ε) * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
  rintro ⟨x, y⟩ ⟨hx, hy⟩ s₁ s₂ s₃ y₁ y₂ y₃
  have hX := block_nonneg s₁ s₂ s₃ y₁ y₂ y₃
  set X := (|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃| with hXdef
  obtain ⟨x₀, hx₀, hxx₀⟩ : ∃ x₀ ∈ Icc (-2 * r) (2 * r), |x - x₀| ≤ ε := by
    by_cases h1 : x ≤ -2 * r
    · refine ⟨-2 * r, ⟨le_rfl, ?_⟩, ?_⟩
      · linarith [hx.1]
      · rw [abs_le]; constructor <;> linarith [hx.1]
    · by_cases h2 : 2 * r ≤ x
      · refine ⟨2 * r, ⟨?_, le_rfl⟩, ?_⟩
        · linarith
        · rw [abs_le]; constructor <;> linarith [hx.2]
      · exact ⟨x, ⟨by linarith, by linarith⟩, by simp [hε]⟩
  set g : ℝ → ℝ := fun t => T3 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃) -
    s₁ * s₂ * s₃ * T3 (t, y) (1, 0) (1, 0) (1, 0) with hg
  have hderiv : ∀ t ∈ Icc (-2 * r - ε₀) (2 * r + ε₀), HasDerivWithinAt g
      (T4 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T4 (t, y) (1, 0) (1, 0) (1, 0))
      (Icc (-2 * r - ε₀) (2 * r + ε₀)) t := by
    intro t ht
    have hp : (t, y) ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r) := ⟨ht, hy⟩
    have h1 : HasDerivAt (fun t => T3 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃))
        (T4 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃)) t := hD4x _ hp _ _ _
    have h2 : HasDerivAt (fun t => T3 (t, y) (1, 0) (1, 0) (1, 0)) (T4 (t, y) (1, 0) (1, 0) (1, 0)) t :=
      hD4x _ hp _ _ _
    exact (h1.sub (h2.const_mul (s₁ * s₂ * s₃))).hasDerivWithinAt
  have hbound : ∀ t ∈ Icc (-2 * r - ε₀) (2 * r + ε₀),
      ‖T4 (t, y) (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T4 (t, y) (1, 0) (1, 0) (1, 0)‖ ≤ K * X :=
    fun t ht => by rw [Real.norm_eq_abs]; exact hT4 (t, y) ⟨ht, hy⟩ _ _ _ _ _ _
  have hxS : x ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) := ⟨by linarith [hx.1], by linarith [hx.2]⟩
  have hx₀S : x₀ ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) :=
    ⟨by linarith [hx₀.1, hε, hεε], by linarith [hx₀.2, hε, hεε]⟩
  have hmvt := (convex_Icc (-2 * r - ε₀) (2 * r + ε₀)).norm_image_sub_le_of_norm_hasDerivWithin_le
    hderiv hbound hx₀S hxS
  rw [Real.norm_eq_abs, Real.norm_eq_abs] at hmvt
  have h0 : |g x₀| ≤ m * X := hT (x₀, y) ⟨hx₀, hy⟩ s₁ s₂ s₃ y₁ y₂ y₃
  have hKX : 0 ≤ K * X := mul_nonneg hK hX
  have h3 : K * X * |x - x₀| ≤ K * X * ε := mul_le_mul_of_nonneg_left hxx₀ hKX
  have h4 := abs_sub_abs_le_abs_sub (g x) (g x₀)
  show |g x| ≤ (m + K * ε) * X
  nlinarith

/-- **The cap from the block bound alone.** `ridge_capHyp_C4` with its six derivative-bound
hypotheses derived. `hHxb`, `hHrb`, `hwcv` and `hwcc` come from the block bound `hT`
(`hessian_x_bound`, `hessian_r_bound`, `axis_convexConcave`). `hG0cv` and `hG0cc` come from
`|f_xxx| ≤ m` on the pin segment (`hTe`, SP: `|f_xxx| ≤ M₃`). The conclusion is unchanged. -/
theorem ridge_capHyp_C5 [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {Φ4 : ℝ × E → ℝ × E →L[ℝ] ℝ} {n : ℝ}
    {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m lam ε : ℝ} (hr : 0 < r) (hε : 0 < ε)
    (hεr : ε ≤ r / 2)
    (hDf : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hD2 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt Df (D2f p) p)
    (hD3 : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ a b : ℝ × E, HasFDerivAt (fun q => D2f q a b) (T3 p a b) p)
    (hT : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hTe : ∀ x ∈ Icc (-r / 2) (r / 2), |T3 (x, 0) (1, 0) (1, 0) (1, 0)| ≤ m)
    (hD4 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun q => T3 q (1, 0) (1, 0) (1, 0)) (Φ4 p) p)
    (hn4x : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), |Φ4 p (1, 0)| ≤ n)
    (hn4y : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ‖(Φ4 p).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ n)
    (hH2 : r * n ≤ 1 / 20)
    (hGs : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHr : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  have hm0 : 0 ≤ m := (abs_nonneg _).trans (hTe 0 ⟨by linarith, by linarith⟩)
  have h0Y : (0 : E) ∈ Metric.closedBall (0 : E) (2 * r) := Metric.mem_closedBall_self (by positivity)
  have hax : ∀ t ∈ Icc (-2 * r - ε) (2 * r + ε),
      (t, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun t ht => ⟨ht, h0Y⟩
  have hpin : ∀ t ∈ Icc (-r / 2) (r / 2),
      (t, (0 : E)) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun t ht => hax t ⟨by linarith [ht.1], by linarith [ht.2]⟩
  -- `∂_x D_y² f`
  have hHxb : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), ‖Hx x‖ ≤ m := by
    intro x hx
    refine hessian_x_bound (J := Icc (-2 * r - ε) (2 * r + ε))
      (uniqueDiffOn_Icc (by linarith) x hx) hx hm0 (fun t ht => hH _ (hax t ht))
      (fun t ht => hD2 _ (hax t ht)) (hD3 _ (hax x hx)) (fun v w => ?_) (hHx x hx)
    have := hT _ (hax x hx) 0 0 1 v w 0
    simpa using this
  -- `∂_ρ D_y² f`
  have hHrb : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, ‖Hr p s‖ ≤ m * ‖p.2‖ := by
    rintro ⟨x, y⟩ hp s hs
    have hyb : ‖y‖ ≤ 2 * r := by simpa [Metric.mem_closedBall, dist_zero_right] using hp.2
    have hseg : ∀ τ ∈ Icc (0 : ℝ) 1,
        (x, τ • y) ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r) := by
      intro τ hτ
      refine ⟨hp.1, ?_⟩
      rw [Metric.mem_closedBall, dist_zero_right, norm_smul, Real.norm_eq_abs, abs_of_nonneg hτ.1]
      nlinarith [hτ.2, norm_nonneg y]
    refine hessian_r_bound hm0 hs (fun τ hτ => hH _ (hseg τ hτ)) (fun τ hτ => hD2 _ (hseg τ hτ))
      (hD3 _ (hseg s hs)) (fun v w => ?_) (hHr _ hp s hs)
    have := hT _ (hseg s hs) 0 0 0 v w y
    simpa using this
  -- `w = ∂_y f (·, 0)` and `G0 = ∂_x f (·, 0)`
  have hw : ∀ v : E, ConvexOn ℝ (Icc (-2 * r - ε) (2 * r + ε))
        (fun t => Df (t, 0) (0, v) + m * ‖v‖ / 2 * t ^ 2) ∧
      ConcaveOn ℝ (Icc (-2 * r - ε) (2 * r + ε)) (fun t => Df (t, 0) (0, v) - m * ‖v‖ / 2 * t ^ 2) :=
    fun v => axis_convexConcave (fun t ht => hD2 _ (hax t ht))
      (fun t ht => hD3 _ (hax t ht) (1, 0) (0, v))
      (fun t ht => by have := hT _ (hax t ht) 1 0 1 0 v 0; simpa using this)
  have hG0 := axis_convexConcave (u := ((1 : ℝ), (0 : E))) (fun t ht => hD2 _ (hpin t ht))
    (fun t ht => hD3 _ (hpin t ht) (1, 0) (1, 0)) hTe
  exact ridge_capHyp_C4 hr hε hεr hDf hD2 hD3 hT hD4 hn4x hn4y hH2 hGs hH hHx hHxb hHr hHrb hlam
    hH1 hM hS hG0.1 hG0.2 (fun v => (hw v).1) (fun v => (hw v).2) hnorm

/-- **The cap from SP's bounds on `D` (the continuity step of I3).** The quantitative hypotheses are
assumed on SP's `D` only:
- the block bound `m` and `|f_xxx| ≤ m` on the pin segment;
- SP's `M₄` blocks `n`;
- (H1) and (H2).

On a collar of some width `ε₀ > 0`, only qualitative data are assumed: the derivatives, the strict
derivative of `∂_y f`, and some bound `K ≥ 0` on the `x`-derivative of `T3` in block form (SP:
`f ∈ C⁴` near `D`). Because (H1) is strict, some `ε ∈ (0, ε₀]` keeps `8r(m + Kε)² < λ`.
`ridge_capHyp_C5` then applies with `m + Kε` (`block_bound_collar`). -/
theorem ridge_capHyp_D [FiniteDimensional ℝ E] {f : ℝ × E → ℝ}
    {Df : ℝ × E → ℝ × E →L[ℝ] ℝ} {D2f : ℝ × E → ℝ × E →L[ℝ] ℝ × E →L[ℝ] ℝ}
    {T3 : ℝ × E → ℝ × E → ℝ × E → ℝ × E →L[ℝ] ℝ} {T4 : ℝ × E → ℝ × E → ℝ × E → ℝ × E → ℝ}
    {Φ4 : ℝ × E → ℝ × E →L[ℝ] ℝ} {n : ℝ}
    {DG : ℝ × E → ℝ × E →L[ℝ] E →L[ℝ] ℝ}
    {H : ℝ × E → E →L[ℝ] E →L[ℝ] ℝ} {Hx : ℝ → E →L[ℝ] E →L[ℝ] ℝ}
    {Hr : ℝ × E → ℝ → E →L[ℝ] E →L[ℝ] ℝ} {r m K lam ε₀ : ℝ} (hr : 0 < r) (hε₀ : 0 < ε₀)
    (hK : 0 ≤ K)
    (hDf : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt f (Df p) p)
    (hD2 : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r), HasFDerivAt Df (D2f p) p)
    (hD3 : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ a b : ℝ × E,
      HasFDerivAt (fun q => D2f q a b) (T3 p a b) p)
    (hD4x : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ a b c : ℝ × E, HasDerivAt (fun t => T3 (t, p.2) a b c) (T4 p a b c) p.1)
    (hT4 : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T4 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T4 p (1, 0) (1, 0) (1, 0)| ≤
        K * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hT : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |T3 p (s₁, y₁) (s₂, y₂) (s₃, y₃) - s₁ * s₂ * s₃ * T3 p (1, 0) (1, 0) (1, 0)| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hTe : ∀ x ∈ Icc (-r / 2) (r / 2), |T3 (x, 0) (1, 0) (1, 0) (1, 0)| ≤ m)
    (hD4 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun q => T3 q (1, 0) (1, 0) (1, 0)) (Φ4 p) p)
    (hn4x : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), |Φ4 p (1, 0)| ≤ n)
    (hn4y : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ‖(Φ4 p).comp (ContinuousLinearMap.inr ℝ ℝ E)‖ ≤ n)
    (hH2 : r * n ≤ 1 / 20)
    (hGs : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasStrictFDerivAt (fun q => (Df q).comp (ContinuousLinearMap.inr ℝ ℝ E)) (DG p) p)
    (hH : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      HasFDerivAt (fun y => (Df (p.1, y)).comp (ContinuousLinearMap.inr ℝ ℝ E)) (H p) p.2)
    (hHx : ∀ x ∈ Icc (-2 * r - ε₀) (2 * r + ε₀), HasDerivAt (fun t => H (t, 0)) (Hx x) x)
    (hHr : ∀ p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ s ∈ Icc (0 : ℝ) 1, HasDerivAt (fun s : ℝ => H (p.1, s • p.2)) (Hr p s) s)
    (hlam : ∀ v : E, H (-r / 2, 0) v v ≤ -lam * ‖v‖ ^ 2) (hH1 : 8 * r * m ^ 2 < lam)
    (hM : Df (-r / 2, 0) = 0) (hS : Df (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    ∃ ε, 0 < ε ∧ ε ≤ ε₀ ∧
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  have hev : ∀ᶠ ε in 𝓝[>] (0 : ℝ), 0 < ε ∧ ε ≤ ε₀ ∧ ε ≤ r / 2 ∧ 8 * r * (m + K * ε) ^ 2 < lam := by
    have hc : ContinuousAt (fun ε : ℝ => 8 * r * (m + K * ε) ^ 2) 0 := by fun_prop
    have h1 : ∀ᶠ ε in 𝓝 (0 : ℝ), 8 * r * (m + K * ε) ^ 2 < lam :=
      hc.eventually_lt continuousAt_const (by simpa using hH1)
    have h2 : ∀ᶠ ε in 𝓝 (0 : ℝ), ε < ε₀ := eventually_lt_nhds hε₀
    have h3 : ∀ᶠ ε in 𝓝 (0 : ℝ), ε < r / 2 := eventually_lt_nhds (by linarith)
    filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds h1, nhdsWithin_le_nhds h2,
      nhdsWithin_le_nhds h3] with ε hε h1 h2 h3
    exact ⟨hε, h2.le, h3.le, h1⟩
  obtain ⟨ε, hε, hεε, hεr, hH1'⟩ := hev.exists
  have hsub : ∀ p ∈ Icc (-2 * r - ε) (2 * r + ε) ×ˢ Metric.closedBall (0 : E) (2 * r),
      p ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r) :=
    fun p hp => ⟨⟨by linarith [hp.1.1], by linarith [hp.1.2]⟩, hp.2⟩
  have hsubI : ∀ x ∈ Icc (-2 * r - ε) (2 * r + ε), x ∈ Icc (-2 * r - ε₀) (2 * r + ε₀) :=
    fun x hx => ⟨by linarith [hx.1], by linarith [hx.2]⟩
  have hmm : m ≤ m + K * ε := le_add_of_nonneg_right (mul_nonneg hK hε.le)
  exact ⟨ε, hε, hεε, ridge_capHyp_C5 (m := m + K * ε) hr hε hεr (fun p hp => hDf p (hsub p hp))
    (fun p hp => hD2 p (hsub p hp)) (fun p hp => hD3 p (hsub p hp))
    (block_bound_collar hr hK hε.le hεε hT hD4x hT4) (fun x hx => (hTe x hx).trans hmm)
    hD4 hn4x hn4y hH2 (fun p hp => hGs p (hsub p hp)) (fun p hp => hH p (hsub p hp))
    (fun x hx => hHx x (hsubI x hx)) (fun p hp => hHr p (hsub p hp)) hlam hH1' hM hS hnorm⟩


/-! ### SP's `f ∈ C⁴` near `D` supplies the collar data (v2.2)

`block_of_norm` turns a bound on the norm of a trilinear form into the block bound, and
`collar_subset_open` finds a closed collar inside an open neighbourhood of `D`. `ridge_capHyp_E`
takes `f ∈ C⁴` on an open `U ⊇ D` and SP's quantitative bounds on `D`, written with the iterated
derivatives `iteratedFDeriv ℝ k f`. It builds every derivative datum of `ridge_capHyp_D` from `f`,
and it takes the collar width from `collar_subset_open` and `K` from the continuity of `D⁴f` on the
compact collar. -/

/-- **The block bound from the norm.** A trilinear form `B` on `ℝ × E` with `‖B‖ ≤ K` satisfies
the block bound with constant `K`. For `a = (s₁, y₁)`, `b = (s₂, y₂)`, `c = (s₃, y₃)` and
`e = (1, 0)`:
`|B(a, b, c) - s₁s₂s₃ B(e, e, e)| ≤ K((|s₁| + ‖y₁‖)(|s₂| + ‖y₂‖)(|s₃| + ‖y₃‖) - |s₁||s₂||s₃|)`.
Replacing one slot at a time by `sᵢ e` leaves three terms, and each has a transverse argument
`(0, yᵢ)`. -/
theorem block_of_norm {B : ContinuousMultilinearMap ℝ (fun _ : Fin 3 => ℝ × E) ℝ} {K : ℝ}
    (hB : ‖B‖ ≤ K) (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E) :
    |B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![(1, 0), (1, 0), (1, 0)]| ≤
      K * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
  have hK : 0 ≤ K := (norm_nonneg _).trans hB
  have hb : ∀ a b c : ℝ × E, |B ![a, b, c]| ≤ K * ‖a‖ * ‖b‖ * ‖c‖ := by
    intro a b c
    have h := B.le_opNorm ![a, b, c]
    simp only [Fin.prod_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_two, Matrix.tail_cons, Matrix.head_cons] at h
    rw [← Real.norm_eq_abs]
    calc ‖B ![a, b, c]‖ ≤ ‖B‖ * (‖a‖ * ‖b‖ * ‖c‖) := h
      _ ≤ K * (‖a‖ * ‖b‖ * ‖c‖) := by gcongr
      _ = K * ‖a‖ * ‖b‖ * ‖c‖ := by ring
  have h1 : ∀ x x' b c : ℝ × E, B ![x, b, c] - B ![x', b, c] = B ![x - x', b, c] := by
    intro x x' b c
    have hu : ∀ z, Function.update ![x, b, c] 0 z = ![z, b, c] := fun z => by
      funext i; fin_cases i <;> simp
    have := B.map_update_sub ![x, b, c] 0 x x'
    rw [hu, hu, hu] at this
    exact this.symm
  have h2 : ∀ a x x' c : ℝ × E, B ![a, x, c] - B ![a, x', c] = B ![a, x - x', c] := by
    intro a x x' c
    have hu : ∀ z, Function.update ![a, x, c] 1 z = ![a, z, c] := fun z => by
      funext i; fin_cases i <;> simp
    have := B.map_update_sub ![a, x, c] 1 x x'
    rw [hu, hu, hu] at this
    exact this.symm
  have h3 : ∀ a b x x' : ℝ × E, B ![a, b, x] - B ![a, b, x'] = B ![a, b, x - x'] := by
    intro a b x x'
    have hu : ∀ z, Function.update ![a, b, x] 2 z = ![a, b, z] := fun z => by
      funext i; fin_cases i <;> simp
    have := B.map_update_sub ![a, b, x] 2 x x'
    rw [hu, hu, hu] at this
    exact this.symm
  set e : ℝ × E := ((1 : ℝ), (0 : E)) with he_def
  have hsm : B ![s₁ • e, s₂ • e, s₃ • e] = s₁ * s₂ * s₃ * B ![e, e, e] := by
    have := B.map_smul_univ ![s₁, s₂, s₃] ![e, e, e]
    have hv : (fun i => (![s₁, s₂, s₃] : Fin 3 → ℝ) i • (![e, e, e] : Fin 3 → ℝ × E) i) =
        ![s₁ • e, s₂ • e, s₃ • e] := by
      funext i; fin_cases i <;> simp
    rw [hv, Fin.prod_univ_three] at this
    simpa [smul_eq_mul, mul_assoc] using this
  have hd : ∀ (s : ℝ) (y : E), ((s, y) : ℝ × E) - s • e = ((0 : ℝ), y) := by
    intro s y; ext <;> simp [he_def]
  have hn : ∀ (s : ℝ) (y : E), ‖((s, y) : ℝ × E)‖ ≤ |s| + ‖y‖ := by
    intro s y
    rw [Prod.norm_def, Real.norm_eq_abs]
    exact max_le_add_of_nonneg (abs_nonneg _) (norm_nonneg _)
  have hn0 : ∀ y : E, ‖((0 : ℝ), y)‖ = ‖y‖ := by intro y; simp [Prod.norm_def]
  have hne : ∀ s : ℝ, ‖s • e‖ = |s| := by
    intro s; rw [norm_smul, Real.norm_eq_abs]; simp [he_def, Prod.norm_def]
  have tel : B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![e, e, e] =
      B ![((0 : ℝ), y₁), (s₂, y₂), (s₃, y₃)] + B ![s₁ • e, ((0 : ℝ), y₂), (s₃, y₃)] +
        B ![s₁ • e, s₂ • e, ((0 : ℝ), y₃)] := by
    rw [← hsm, ← hd s₁ y₁, ← hd s₂ y₂, ← hd s₃ y₃, ← h1, ← h2, ← h3]
    ring
  rw [tel]
  have k1 : |B ![((0 : ℝ), y₁), (s₂, y₂), (s₃, y₃)]| ≤
      K * ‖y₁‖ * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) := by
    refine (hb _ _ _).trans ?_
    rw [hn0]
    gcongr
    · exact hn _ _
    · exact hn _ _
  have k2 : |B ![s₁ • e, ((0 : ℝ), y₂), (s₃, y₃)]| ≤ K * |s₁| * ‖y₂‖ * (|s₃| + ‖y₃‖) := by
    refine (hb _ _ _).trans ?_
    rw [hn0, hne]
    gcongr
    exact hn _ _
  have k3 : |B ![s₁ • e, s₂ • e, ((0 : ℝ), y₃)]| ≤ K * |s₁| * |s₂| * ‖y₃‖ := by
    refine (hb _ _ _).trans ?_
    rw [hn0, hne, hne]
  calc _ ≤ _ := abs_add_three _ _ _
    _ ≤ K * ‖y₁‖ * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) + K * |s₁| * ‖y₂‖ * (|s₃| + ‖y₃‖) +
        K * |s₁| * |s₂| * ‖y₃‖ := by gcongr
    _ = _ := by ring

omit [NormedSpace ℝ E] in
/-- **A collar inside a neighbourhood of `D`.** If an open `U` contains
`D = [-2r, 2r] × B̄(0, 2r)` and `E` is proper, then some closed collar
`[-2r - ε₀, 2r + ε₀] × B̄(0, 2r)` with `ε₀ > 0` lies in `U`: `D` is compact, so a closed
thickening of `D` lies in `U`. -/
theorem collar_subset_open [ProperSpace E] {U : Set (ℝ × E)} {r : ℝ} (hr : 0 < r) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U) :
    ∃ ε₀ > 0, Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U := by
  obtain ⟨δ, hδ, hsub⟩ :=
    (isCompact_Icc.prod (isCompact_closedBall (0 : E) (2 * r))).exists_cthickening_subset_open hU hDU
  refine ⟨δ, hδ, ?_⟩
  rintro ⟨x, y⟩ ⟨hx, hy⟩
  apply hsub
  obtain ⟨x₀, hx₀, hxx₀⟩ : ∃ x₀ ∈ Icc (-2 * r) (2 * r), |x - x₀| ≤ δ := by
    by_cases h1 : x ≤ -2 * r
    · refine ⟨-2 * r, ⟨le_rfl, ?_⟩, ?_⟩
      · linarith [hx.1]
      · rw [abs_le]; constructor <;> linarith [hx.1]
    · by_cases h2 : 2 * r ≤ x
      · refine ⟨2 * r, ⟨?_, le_rfl⟩, ?_⟩
        · linarith
        · rw [abs_le]; constructor <;> linarith [hx.2]
      · exact ⟨x, ⟨by linarith, by linarith⟩, by simp [hδ.le]⟩
  refine Metric.mem_cthickening_of_dist_le (x, y) (x₀, y) δ _ ⟨hx₀, hy⟩ ?_
  rw [Prod.dist_eq, dist_self, Real.dist_eq]
  exact max_le hxx₀ hδ.le

/-- **The cap from `f ∈ C⁴` near `D` and SP's bounds on `D` (I3's qualitative part).** Assume
that `f` is `C⁴` on an open `U ⊇ D` (SP: `f ∈ C⁴` near `D`). Assume also SP's quantitative
hypotheses on `D`, written with `D^k f(p)[v₁, …, v_k] = iteratedFDeriv ℝ k f p ![v₁, …, v_k]` and
`e = (1, 0)`:
- the block bound `m` on `D³f`, and `|f_xxx| ≤ m` on the pin segment;
- SP's `M₄` blocks `n`: `|D⁴f[e, e, e, e]| ≤ n` and `|D⁴f[(0, v), e, e, e]| ≤ n‖v‖`;
- (H1): `D²f(M)[(0, v), (0, v)] ≤ -λ‖v‖²` with `8rm² < λ`; and (H2);
- `M = (-r/2, 0)` and `S = (r/2, 0)` are critical, and `f(M) - f(S) = r³/6`.

Then `ridge_capHyp_D` applies. Its collar width comes from `collar_subset_open`, and its derivative
data are the Fréchet derivatives of `f` (`T3`, `T4` and `Φ4` are slices of `D³f` and `D⁴f`). Its
`K` bounds `‖D⁴f‖` on the compact collar. No derivative datum and no bound off `D` is assumed. -/
theorem ridge_capHyp_E [FiniteDimensional ℝ E] {f : ℝ × E → ℝ} {U : Set (ℝ × E)}
    {r m n lam : ℝ} (hr : 0 < r) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 f U)
    (hT : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |iteratedFDeriv ℝ 3 f p ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] -
          s₁ * s₂ * s₃ * iteratedFDeriv ℝ 3 f p ![(1, 0), (1, 0), (1, 0)]| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hTe : ∀ x ∈ Icc (-r / 2) (r / 2), |iteratedFDeriv ℝ 3 f (x, 0) ![(1, 0), (1, 0), (1, 0)]| ≤ m)
    (hn4x : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      |iteratedFDeriv ℝ 4 f p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ n)
    (hn4y : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      |iteratedFDeriv ℝ 4 f p ![(0, v), (1, 0), (1, 0), (1, 0)]| ≤ n * ‖v‖)
    (hH2 : r * n ≤ 1 / 20)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 f (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 8 * r * m ^ 2 < lam)
    (hM : fderiv ℝ f (-r / 2, 0) = 0) (hS : fderiv ℝ f (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = r ^ 3 / 6) :
    ∃ ε, 0 < ε ∧
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  obtain ⟨ε₀, hε₀, hcU⟩ := collar_subset_open hr hU hDU
  have hcpt : IsCompact (Icc (-2 * r - ε₀) (2 * r + ε₀) ×ˢ Metric.closedBall (0 : E) (2 * r)) :=
    isCompact_Icc.prod (isCompact_closedBall _ _)
  -- regularity of the derivatives on `U`
  have hf1 : ContDiffOn ℝ 3 (fderiv ℝ f) U := hf.fderiv_of_isOpen hU (by norm_num)
  have hf2 : ContDiffOn ℝ 2 (fderiv ℝ (fderiv ℝ f)) U := hf1.fderiv_of_isOpen hU (by norm_num)
  have dDf : ∀ p ∈ U, HasFDerivAt f (fderiv ℝ f p) p := fun p hp =>
    ((hf.differentiableOn (by norm_num)).differentiableAt (hU.mem_nhds hp)).hasFDerivAt
  have dD2 : ∀ p ∈ U, HasFDerivAt (fderiv ℝ f) (fderiv ℝ (fderiv ℝ f) p) p := fun p hp =>
    ((hf1.differentiableOn (by norm_num)).differentiableAt (hU.mem_nhds hp)).hasFDerivAt
  have dD3 : ∀ p ∈ U, HasFDerivAt (fderiv ℝ (fderiv ℝ f)) (fderiv ℝ (fderiv ℝ (fderiv ℝ f)) p) p :=
    fun p hp => ((hf2.differentiableOn (by norm_num)).differentiableAt (hU.mem_nhds hp)).hasFDerivAt
  have sD2 : ∀ p ∈ U, HasStrictFDerivAt (fderiv ℝ f) (fderiv ℝ (fderiv ℝ f) p) p := fun p hp =>
    (hf1.contDiffAt (hU.mem_nhds hp)).hasStrictFDerivAt (by norm_num)
  have dI2 : ∀ p ∈ U, HasFDerivAt (iteratedFDeriv ℝ 2 f) (fderiv ℝ (iteratedFDeriv ℝ 2 f) p) p :=
    fun p hp => ((hf.contDiffAt (hU.mem_nhds hp)).differentiableAt_iteratedFDeriv
      (by norm_num)).hasFDerivAt
  have dI3 : ∀ p ∈ U, HasFDerivAt (iteratedFDeriv ℝ 3 f) (fderiv ℝ (iteratedFDeriv ℝ 3 f) p) p :=
    fun p hp => ((hf.contDiffAt (hU.mem_nhds hp)).differentiableAt_iteratedFDeriv
      (by norm_num)).hasFDerivAt
  have hc4 : ContinuousOn (iteratedFDeriv ℝ 4 f) U :=
    (hf.continuousOn_iteratedFDerivWithin le_rfl hU.uniqueDiffOn).congr
      (fun x hx => (iteratedFDerivWithin_of_isOpen 4 hU hx).symm)
  -- the bound `K` on `D⁴f` over the compact collar
  obtain ⟨C, hC⟩ := hcpt.exists_bound_of_continuousOn (hc4.mono hcU)
  have hD2 : ∀ (p : ℝ × E) (a b : ℝ × E),
      fderiv ℝ (fderiv ℝ f) p a b = iteratedFDeriv ℝ 2 f p ![a, b] := by
    intro p a b; rw [iteratedFDeriv_two_apply]; rfl
  have he : ‖((1 : ℝ), (0 : E))‖ = 1 := by simp [Prod.norm_def]
  have hpath : ∀ (x : ℝ) (y : E), HasDerivAt (fun t : ℝ => ((t, y) : ℝ × E)) ((1 : ℝ), (0 : E)) x :=
    fun x y => (hasDerivAt_id x).prodMk (hasDerivAt_const x y)
  set Ψ : ((ℝ × E) →L[ℝ] ℝ) →L[ℝ] (E →L[ℝ] ℝ) :=
    ContinuousLinearMap.precomp ℝ (ContinuousLinearMap.inr ℝ ℝ E) with hΨ
  set Ξ : ((ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ) →L[ℝ] (E →L[ℝ] E →L[ℝ] ℝ) :=
    (ContinuousLinearMap.postcomp E Ψ).comp
      (ContinuousLinearMap.precomp ((ℝ × E) →L[ℝ] ℝ) (ContinuousLinearMap.inr ℝ ℝ E)) with hΞ
  have hn0 : 0 ≤ n := (abs_nonneg _).trans (hn4x (0, 0) ⟨⟨by linarith, by linarith⟩, by
    simp only [Metric.mem_closedBall, dist_self]; linarith⟩)
  have key := ridge_capHyp_D (E := E) (f := f) (Df := fderiv ℝ f) (D2f := fderiv ℝ (fderiv ℝ f))
    (T3 := fun p a b => (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => ℝ × E) ℝ ![a, b]).comp
      (fderiv ℝ (iteratedFDeriv ℝ 2 f) p))
    (T4 := fun p a b c => iteratedFDeriv ℝ 4 f p ![(1, 0), c, a, b])
    (Φ4 := fun p => (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 3 => ℝ × E) ℝ
      ![(1, 0), (1, 0), (1, 0)]).comp (fderiv ℝ (iteratedFDeriv ℝ 3 f) p)) (n := n)
    (DG := fun p => Ψ.comp (fderiv ℝ (fderiv ℝ f) p))
    (H := fun p => Ξ (fderiv ℝ (fderiv ℝ f) p))
    (Hx := fun x => Ξ (fderiv ℝ (fderiv ℝ (fderiv ℝ f)) (x, 0) (1, 0)))
    (Hr := fun p s => Ξ (fderiv ℝ (fderiv ℝ (fderiv ℝ f)) (p.1, s • p.2) (0, p.2)))
    (r := r) (m := m) (K := max C 0) (lam := lam) (ε₀ := ε₀) hr hε₀ (le_max_right _ _)
    (fun p hp => dDf p (hcU hp)) (fun p hp => dD2 p (hcU hp))
    (fun p hp a b => by
      have h := (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => ℝ × E) ℝ ![a, b]).hasFDerivAt.comp
        p (dI2 p (hcU hp))
      have hfun : (fun q => fderiv ℝ (fderiv ℝ f) q a b) =
          (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => ℝ × E) ℝ ![a, b]) ∘
            iteratedFDeriv ℝ 2 f := funext fun q => hD2 q a b
      rw [hfun]; exact h)
    (fun p hp a b c => by
      have h := (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 3 => ℝ × E) ℝ ![c, a, b]).hasFDerivAt.comp_hasDerivAt
        p.1 ((dI3 (p.1, p.2) (hcU hp)).comp_hasDerivAt p.1 (hpath p.1 p.2))
      exact h)
    (fun p hp s₁ s₂ s₃ y₁ y₂ y₃ => by
      have hB : ‖(iteratedFDeriv ℝ 4 f p).curryLeft ((1 : ℝ), (0 : E))‖ ≤ max C 0 := by
        refine ((iteratedFDeriv ℝ 4 f p).curryLeft.le_opNorm _).trans ?_
        rw [ContinuousMultilinearMap.curryLeft_norm, he, mul_one]
        exact (hC p hp).trans (le_max_left _ _)
      have h := block_of_norm hB s₃ s₁ s₂ y₃ y₁ y₂
      show |iteratedFDeriv ℝ 4 f p ![(1, 0), (s₃, y₃), (s₁, y₁), (s₂, y₂)] -
        s₁ * s₂ * s₃ * iteratedFDeriv ℝ 4 f p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ _
      rw [show s₁ * s₂ * s₃ = s₃ * s₁ * s₂ by ring]
      refine h.trans (le_of_eq ?_)
      ring)
    (fun p hp s₁ s₂ s₃ y₁ y₂ y₃ => by
      have h := hT p hp s₃ s₁ s₂ y₃ y₁ y₂
      show |iteratedFDeriv ℝ 3 f p ![(s₃, y₃), (s₁, y₁), (s₂, y₂)] -
        s₁ * s₂ * s₃ * iteratedFDeriv ℝ 3 f p ![(1, 0), (1, 0), (1, 0)]| ≤ _
      rw [show s₁ * s₂ * s₃ = s₃ * s₁ * s₂ by ring]
      refine h.trans (le_of_eq ?_)
      ring)
    (fun x hx => hTe x hx)
    (fun p hp => (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 3 => ℝ × E) ℝ
      ![(1, 0), (1, 0), (1, 0)]).hasFDerivAt.comp p (dI3 p (hDU hp)))
    (fun p hp => hn4x p hp)
    (fun p hp => ContinuousLinearMap.opNorm_le_bound _ hn0 fun v => by
      rw [Real.norm_eq_abs]; exact hn4y p hp v)
    hH2
    (fun p hp => Ψ.hasStrictFDerivAt.comp p (sD2 p (hcU hp)))
    (fun p hp => Ψ.hasFDerivAt.comp p.2 ((dD2 (p.1, p.2) (hcU hp)).comp p.2
      (hasFDerivAt_prodMk_right p.1 p.2)))
    (fun x hx => by
      have hx' : ((x, 0) : ℝ × E) ∈ U := hcU ⟨hx, by
        simp only [Metric.mem_closedBall, dist_self]; linarith⟩
      have hΞ : HasFDerivAt Ξ Ξ (fderiv ℝ (fderiv ℝ f) (x, 0)) :=
        ContinuousLinearMap.hasFDerivAt (E := (ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ)
          (F := E →L[ℝ] E →L[ℝ] ℝ) _
      exact HasFDerivAt.comp_hasDerivAt (l := Ξ)
        (f := fun t => fderiv ℝ (fderiv ℝ f) (t, 0)) x hΞ (HasFDerivAt.comp_hasDerivAt
          (l := fderiv ℝ (fderiv ℝ f)) (f := fun t : ℝ => ((t, (0 : E)) : ℝ × E)) x (dD3 (x, 0) hx')
          (hpath x 0)))
    (fun p hp s hs => by
      have hmem : ((p.1, s • p.2) : ℝ × E) ∈ U := hcU ⟨hp.1, by
        have := hp.2
        rw [Metric.mem_closedBall, dist_zero_right] at this ⊢
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hs.1]
        nlinarith [hs.2, norm_nonneg p.2]⟩
      have hp' : HasDerivAt (fun s : ℝ => ((p.1, s • p.2) : ℝ × E)) ((0 : ℝ), p.2) s := by
        have := (hasDerivAt_const s p.1).prodMk ((hasDerivAt_id s).smul_const p.2)
        simpa using this
      have hΞ : HasFDerivAt Ξ Ξ (fderiv ℝ (fderiv ℝ f) (p.1, s • p.2)) :=
        ContinuousLinearMap.hasFDerivAt (E := (ℝ × E) →L[ℝ] (ℝ × E) →L[ℝ] ℝ)
          (F := E →L[ℝ] E →L[ℝ] ℝ) _
      exact HasFDerivAt.comp_hasDerivAt (l := Ξ)
        (f := fun s => fderiv ℝ (fderiv ℝ f) (p.1, s • p.2)) s hΞ (HasFDerivAt.comp_hasDerivAt
          (l := fderiv ℝ (fderiv ℝ f)) (f := fun s : ℝ => ((p.1, s • p.2) : ℝ × E)) s (dD3 _ hmem) hp'))
    (fun v => by
      have h := hlam v
      rw [← hD2] at h
      exact h)
    hH1 hM hS hnorm
  obtain ⟨ε, hε, -, hrest⟩ := key
  exact ⟨ε, hε, hrest⟩

end Slice

/-! ### A concrete instance and the necessity of each hypothesis

On `ℝ`, `toyF x = max (-|x|) (x - 2)` has a local maximum `M = 0` at height `b = 0`, a
separating local minimum `S = 1` at height `s = -1`, and an older point `z = 3` at height `1`.
The cap is `[-2, 1]`. The instance shows that `CapHyp` is satisfiable (the theorems are not
vacuous), and the three counterexamples show that dropping (C1), (C2) or (C3) breaks the
maximin conclusion. -/

/-- The toy height function. -/
noncomputable def toyF (x : ℝ) : ℝ := max (-|x|) (x - 2)

/-- The toy ridge, `t ↦ 3t` from `0` to `3`. -/
def toyRidge : Path (0 : ℝ) 3 where
  toFun t := 3 * (t : ℝ)
  continuous_toFun := continuous_const.mul continuous_subtype_val
  source' := by simp
  target' := by simp

theorem toy_ridge_ge (t : I) : (-1 : ℝ) ≤ toyF (toyRidge t) := by
  have ht0 : (0 : ℝ) ≤ t := t.2.1
  show -1 ≤ max (-|3 * (t : ℝ)|) (3 * (t : ℝ) - 2)
  rcases le_or_gt (3 * (t : ℝ)) 1 with h | h
  · refine le_max_of_le_left ?_
    rw [abs_of_nonneg (by linarith)]
    linarith
  · exact le_max_of_le_right (by linarith)

theorem toy_frontier : frontier (Icc (-2 : ℝ) 1) = {-2, 1} := frontier_Icc (by norm_num)

/-- The toy satisfies (C1)–(C3) with `C = [-2, 1]`, `M = 0`, `z = 3`, `b = 0`, `s = -1`. -/
theorem toy_capHyp : CapHyp toyF (Icc (-2 : ℝ) 1) 0 3 0 (-1) where
  mem := ⟨by norm_num, by norm_num⟩
  ceiling x hx := by
    obtain ⟨_, hx2⟩ := hx
    show max (-|x|) (x - 2) ≤ 0
    exact max_le (by linarith [abs_nonneg x]) (by linarith)
  frontier_le x hx := by
    rw [toy_frontier] at hx
    rcases hx with rfl | rfl
    · show max (-|(-2 : ℝ)|) ((-2 : ℝ) - 2) ≤ -1
      norm_num [abs_of_neg]
    · show max (-|(1 : ℝ)|) ((1 : ℝ) - 2) ≤ -1
      norm_num
  older := by
    show (0 : ℝ) < max (-|(3 : ℝ)|) ((3 : ℝ) - 2)
    norm_num
  ridge := ⟨toyRidge, toy_ridge_ge⟩

/-- In the toy, `S = 1` is the only frontier point at height `-1` (C2⁺). -/
theorem toy_saddle_strict : ∀ x ∈ frontier (Icc (-2 : ℝ) 1), x ≠ 1 → toyF x < -1 := by
  intro x hx hne
  rw [toy_frontier] at hx
  rcases hx with rfl | h
  · show max (-|(-2 : ℝ)|) ((-2 : ℝ) - 2) < -1
    norm_num [abs_of_neg]
  · exact absurd h hne

/-- In the toy, removing `S = 1` from `{f ≥ -1}` disconnects `M = 0` from every older point. The
cut is not vacuous: `M` lies in its own component. -/
theorem toy_saddle_cut :
    (0 : ℝ) ∈ connectedComponentIn ({x | (-1 : ℝ) ≤ toyF x} \ {1}) 0 ∧
      ∀ w ∈ connectedComponentIn ({x | (-1 : ℝ) ≤ toyF x} \ {1}) 0, toyF w ≤ 0 :=
  toy_capHyp.saddle_cut toy_saddle_strict (by norm_num)

/-- The toy's elder death level is `-1`. -/
theorem toy_death_level : IsGreatest (CapHyp.joinedLevels toyF 0 0) (-1) :=
  toy_capHyp.elder_death_level

theorem toy_f_M : toyF 0 = 0 := by
  show max (-|(0 : ℝ)|) ((0 : ℝ) - 2) = 0
  norm_num

theorem toy_f_S : toyF 1 = -1 := by
  show max (-|(1 : ℝ)|) ((1 : ℝ) - 2) = -1
  norm_num

/-- Necessity of (C2). At the level `s = -3/2`, (C1) and (C3) hold, (C2) fails (the frontier point
`1` has height `-1 > -3/2`), and the maximin level is not `-3/2`. -/
theorem toy_needs_C2 :
    (∀ x ∈ Icc (-2 : ℝ) 1, toyF x ≤ 0) ∧ (0 : ℝ) < toyF 3 ∧
    (∃ γ : Path (0 : ℝ) 3, ∀ t, (-3 / 2 : ℝ) ≤ toyF (γ t)) ∧
    ¬ (∀ x ∈ frontier (Icc (-2 : ℝ) 1), toyF x ≤ -3 / 2) ∧
    ¬ IsGreatest (connectionLevels toyF 0 0) (-3 / 2) := by
  refine ⟨toy_capHyp.ceiling, toy_capHyp.older,
    ⟨toyRidge, fun t => le_trans (by norm_num) (toy_ridge_ge t)⟩, fun h => ?_, ?_⟩
  · have h1 := h 1 (by rw [toy_frontier]; simp)
    rw [toy_f_S] at h1
    norm_num at h1
  · rintro ⟨-, hub⟩
    have hmem : (-1 : ℝ) ∈ connectionLevels toyF 0 0 :=
      ⟨3, toyRidge, toy_capHyp.older, toy_ridge_ge⟩
    have := hub hmem
    norm_num at this

/-- Necessity of (C1). At the ceiling `b = -1/2`, (C2) and (C3) hold, (C1) fails (`f M = 0 > -1/2`),
and the constant path at `M` shows that the maximin level is not `-1`. -/
theorem toy_needs_C1 :
    ¬ (∀ x ∈ Icc (-2 : ℝ) 1, toyF x ≤ -1 / 2) ∧
    (∀ x ∈ frontier (Icc (-2 : ℝ) 1), toyF x ≤ -1) ∧ (-1 / 2 : ℝ) < toyF 3 ∧
    (∃ γ : Path (0 : ℝ) 3, ∀ t, (-1 : ℝ) ≤ toyF (γ t)) ∧
    ¬ IsGreatest (connectionLevels toyF 0 (-1 / 2)) (-1) := by
  refine ⟨fun h => ?_, toy_capHyp.frontier_le, lt_trans (by norm_num) toy_capHyp.older,
    toy_capHyp.ridge, ?_⟩
  · have h1 := h 0 ⟨by norm_num, by norm_num⟩
    rw [toy_f_M] at h1
    norm_num at h1
  · rintro ⟨-, hub⟩
    have hmem : (0 : ℝ) ∈ connectionLevels toyF 0 (-1 / 2) :=
      ⟨0, Path.refl 0, by rw [toy_f_M]; norm_num, fun t => by simp [toy_f_M]⟩
    have := hub hmem
    norm_num at this

/-- Necessity of (C3). At the level `s = 5`, (C1) and (C2) hold and `z` is older, but no ridge stays
at height `≥ 5` (it starts at `f M = 0`), and the maximin level is not `5`: the barrier at the true
level `-1` excludes it. -/
theorem toy_needs_C3 :
    (∀ x ∈ Icc (-2 : ℝ) 1, toyF x ≤ 0) ∧
    (∀ x ∈ frontier (Icc (-2 : ℝ) 1), toyF x ≤ 5) ∧ (0 : ℝ) < toyF 3 ∧
    ¬ (∃ γ : Path (0 : ℝ) 3, ∀ t, (5 : ℝ) ≤ toyF (γ t)) ∧
    ¬ IsGreatest (connectionLevels toyF 0 0) 5 := by
  refine ⟨toy_capHyp.ceiling, fun x hx => le_trans (toy_capHyp.frontier_le x hx) (by norm_num),
    toy_capHyp.older, ?_, ?_⟩
  · rintro ⟨γ, hγ⟩
    have h0 := hγ 0
    rw [Path.source, toy_f_M] at h0
    norm_num at h0
  · rintro ⟨hmem, -⟩
    have h1 := toy_capHyp.maximin_isGreatest.2 hmem
    norm_num at h1

/-- The ridge toy: SP's normalization at `r = 1` with one transverse coordinate,
`f (x, y) = x³/3 - x/4 - y²`, ridge `h = 0`, ridge derivative `F x = x² - 1/4`. -/
noncomputable def ridgeToyF (p : ℝ × ℝ) : ℝ := p.1 ^ 3 / 3 - p.1 / 4 - p.2 ^ 2

/-- The ridge hypotheses of `ridge_capHyp` are jointly satisfiable. Here the gap is
`g(-1/2) - g(1/2) = 1/6`, which is SP's normalization `r³/6`. -/
theorem ridgeToy_capHyp :
    CapHyp ridgeToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
      (2 * 1, 0) (ridgeToyF (-1 / 2, 0)) (ridgeToyF (1 / 2, 0)) := by
  refine ridge_capHyp (E := ℝ) (f := ridgeToyF) (h := fun _ => (0 : ℝ))
    (F := fun x => x ^ 2 - 1 / 4) (r := 1) one_pos continuousOn_const ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro x _
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext y
      simp only [ridgeToyF, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  · norm_num
  · norm_num
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · rintro ⟨x, y⟩ -
    simp only [ridgeToyF]
    nlinarith [sq_nonneg y]
  · rintro ⟨x, y⟩ ⟨⟨hx1, hx2⟩, hy⟩
    rw [mem_sphere_zero_iff_norm, Real.norm_eq_abs] at hy
    have hy2 : y ^ 2 = 4 := by rw [← sq_abs, hy]; norm_num
    simp only [ridgeToyF, hy2]
    nlinarith [mul_nonneg (sub_nonneg.mpr hx2) (sq_nonneg x),
      mul_nonneg (by linarith : (0 : ℝ) ≤ x + 2) (by linarith : (0 : ℝ) ≤ 2 - x)]
  · simp
  · simp only [ridgeToyF]
    norm_num

/-- In the ridge toy, `S = (1/2, 0)` is the only frontier point at height `s` (C2⁺). -/
theorem ridgeToy_saddle_strict :
    ∀ x ∈ frontier (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)),
      x ≠ (1 / 2, 0) → ridgeToyF x < ridgeToyF (1 / 2, 0) := by
  refine ridge_saddle_strict (E := ℝ) (f := ridgeToyF) (h := fun _ => (0 : ℝ))
    (F := fun x => x ^ 2 - 1 / 4) (r := 1) one_pos ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro x _
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext y
      simp only [ridgeToyF, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  · norm_num
  · norm_num
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · rintro ⟨x, y⟩ -
    simp only [ridgeToyF]
    nlinarith [sq_nonneg y]
  · rintro ⟨x, y⟩ - hy
    have hy' : y ≠ 0 := hy
    simp only [ridgeToyF]
    nlinarith [sq_pos_of_ne_zero hy']
  · rintro ⟨x, y⟩ ⟨⟨hx1, hx2⟩, hy⟩
    rw [mem_sphere_zero_iff_norm, Real.norm_eq_abs] at hy
    have hy2 : y ^ 2 = 4 := by rw [← sq_abs, hy]; norm_num
    simp only [ridgeToyF, hy2]
    nlinarith [mul_nonneg (sub_nonneg.mpr hx2) (sq_nonneg x),
      mul_nonneg (by linarith : (0 : ℝ) ≤ x + 2) (by linarith : (0 : ℝ) ≤ 2 - x)]
  · simp only [ridgeToyF]
    norm_num

/-! ### The ridge toy through the slice route -/

/-- The ridge toy's joint derivative along the ridge `y = 0` is `D x = (x² - 1/4) · dx`. Its
transverse part vanishes (the ridge equation), and `D x (1, 0) = x² - 1/4 = F x`. -/
theorem ridgeToy_joint (x : ℝ) :
    HasFDerivAt ridgeToyF ((x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ) (x, 0) ∧
      ((x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ) =
        0 ∧
      ((x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0) = x ^ 2 - 1 / 4 := by
  refine ⟨?_, by ext; simp, by simp⟩
  have hu : HasDerivAt (fun t : ℝ => t ^ 3 / 3 - t / 4) (x ^ 2 - 1 / 4) x := by
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext t; simp
    · push_cast; ring
  have hv : HasDerivAt (fun t : ℝ => t ^ 2) 0 (0 : ℝ) := by
    simpa using hasDerivAt_pow 2 (0 : ℝ)
  have h1 := hu.comp_hasFDerivAt ((x, (0 : ℝ)) : ℝ × ℝ) (hasFDerivAt_fst (𝕜 := ℝ))
  have h2 := hv.comp_hasFDerivAt ((x, (0 : ℝ)) : ℝ × ℝ) (hasFDerivAt_snd (𝕜 := ℝ))
  convert h1.sub h2 using 1
  · funext p; simp [ridgeToyF]
  · simp

/-- **The slice route is jointly satisfiable.** The ridge toy (`r = 1`, `h = 0`) meets every
input of `ridge_inputs_of_joint` and `ridge_capHyp_of_slices`, with `δ = 2` (each slice is
`-y²` plus a constant), `w = 0`, `ω = 0`, and the curvature condition
`2δ(b - s) = 2/3 < 16 = (2rδ - ω)²`. The combined theorems re-derive (C1)–(C3) and (C2⁺). -/
theorem ridgeToy_slices :
    CapHyp ridgeToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (ridgeToyF (-1 / 2, 0)) (ridgeToyF (1 / 2, 0)) ∧
      ∀ x ∈ frontier (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)),
        x ≠ (1 / 2, 0) → ridgeToyF x < ridgeToyF (1 / 2, 0) := by
  obtain ⟨hh, hg, hcrit⟩ := ridge_inputs_of_joint (E := ℝ) (f := ridgeToyF)
    (h := fun _ => (0 : ℝ)) (h' := fun _ => (0 : ℝ))
    (D := fun x => (x ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ) (r := 1) one_pos
    (fun x _ => (ridgeToy_joint x).1) (fun x _ => hasDerivAt_const x (0 : ℝ))
    (fun x _ => (ridgeToy_joint x).2.1)
  refine ridge_capHyp_of_slices (E := ℝ) (f := ridgeToyF) (h := fun _ => (0 : ℝ))
    (w := fun _ => 0) (r := 1) (δ := 2) (ω := 0) one_pos two_pos hh hg ?_ ?_ ?_ ?_ ?_
    hcrit (fun x hx => hcrit x hx) ?_ ?_ ?_ ?_
  · simp only [(ridgeToy_joint _).2.2]
    norm_num
  · simp only [(ridgeToy_joint _).2.2]
    norm_num
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [(ridgeToy_joint _).2.2, smul_eq_mul]
    ring
  · intro x _
    rw [strongConcaveOn_iff_convex]
    refine (concaveOn_const (x ^ 3 / 3 - x / 4) (convex_closedBall (0 : ℝ) (2 * 1))).congr ?_
    intro y _
    simp only [ridgeToyF, Real.norm_eq_abs, sq_abs]
    ring
  · intro x _
    simp
  · intro x _
    simp
  · norm_num
  · simp only [ridgeToyF]
    norm_num
  · simp only [ridgeToyF]
    norm_num


/-- A field that meets SP's normalized hypotheses: `f (x, y) = x³/3 - x/4 - 12 y²`, so `m = 2`
(`f_xxx = 2`), `δ = 24 > r m (8m - 5) = 22` at `r = 1`, and gap exactly `r³/6`. -/
noncomputable def spToyF (p : ℝ × ℝ) : ℝ := p.1 ^ 3 / 3 - p.1 / 4 - 12 * p.2 ^ 2

/-- **`ridge_capHyp_sp` is not vacuous.** `spToyF` meets every hypothesis, with `h = 0`, `w = 0`,
`F = x² - 1/4`, `m = 2` and `δ = 24`. -/
theorem spToy_capHyp :
    CapHyp spToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (spToyF (-1 / 2, 0)) (spToyF (1 / 2, 0)) ∧
      ∀ x ∈ frontier (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)),
        x ≠ (1 / 2, 0) → spToyF x < spToyF (1 / 2, 0) := by
  have hslice : ∀ x : ℝ, HasFDerivAt (fun y : ℝ => spToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    rw [hasFDerivAt_iff_hasDerivAt]
    have h1 := ((hasDerivAt_pow 2 (0 : ℝ)).const_mul 12).const_sub (x ^ 3 / 3 - x / 4)
    convert h1 using 1
    · funext y
      simp only [spToyF]
    · simp
  refine ridge_capHyp_sp (E := ℝ) (f := spToyF) (h := fun _ => (0 : ℝ))
    (F := fun x => x ^ 2 - 1 / 4) (w := fun _ => 0) (r := 1) (m := 2) (δ := 24) one_pos le_rfl
    (by norm_num) continuousOn_const ?_ ?_ ?_ ?_ ?_ ?_ (fun x _ => hslice x) rfl rfl
    (fun x _ => hslice x) ?_ ?_ ?_
  · intro x _
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext y
      simp only [spToyF, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  · norm_num
  · norm_num
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · intro x _
    rw [strongConcaveOn_iff_convex]
    refine (concaveOn_const (x ^ 3 / 3 - x / 4) (convex_closedBall (0 : ℝ) (2 * 1))).congr ?_
    intro y _
    simp only [spToyF, Real.norm_eq_abs, sq_abs]
    ring
  · intro x _
    simp
  · intro v
    have hc : ConvexOn ℝ univ (fun t : ℝ => ‖v‖ • t ^ 2) :=
      (even_two.convexOn_pow).smul (norm_nonneg v)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul, zero_apply]
    ring
  · intro v
    have hc : ConcaveOn ℝ univ (fun t : ℝ => -(‖v‖ • t ^ 2)) :=
      ((even_two.convexOn_pow).smul (norm_nonneg v)).neg
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul, zero_apply]
    ring
  · simp only [spToyF]
    norm_num


/-- **`ridge_capHyp_normalized` is not vacuous.** `spToyF` meets every hypothesis, with
`G0 = x² - 1/4` (so `|G0''| = 2 = m`) and gap exactly `1/6`. -/
theorem spToy_normalized :
    CapHyp spToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (spToyF (-1 / 2, 0)) (spToyF (1 / 2, 0)) ∧
      ∀ x ∈ frontier (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)),
        x ≠ (1 / 2, 0) → spToyF x < spToyF (1 / 2, 0) := by
  have hslice : ∀ x : ℝ, HasFDerivAt (fun y : ℝ => spToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    rw [hasFDerivAt_iff_hasDerivAt]
    have h1 := ((hasDerivAt_pow 2 (0 : ℝ)).const_mul 12).const_sub (x ^ 3 / 3 - x / 4)
    convert h1 using 1
    · funext y
      simp only [spToyF]
    · simp
  have hgx : ∀ x : ℝ, HasDerivAt (fun x => spToyF (x, 0)) (x ^ 2 - 1 / 4) x := by
    intro x
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext y
      simp only [spToyF, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  have hq : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
    ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
  refine ridge_capHyp_normalized (E := ℝ) (f := spToyF) (h := fun _ => (0 : ℝ))
    (F := fun x => x ^ 2 - 1 / 4) (G0 := fun x => x ^ 2 - 1 / 4) (w := fun _ => 0) (r := 1)
    (m := 2) (δ := 24) one_pos (fun x _ => hgx x) ?_ ?_ (by norm_num) (by norm_num)
    (by norm_num) continuousOn_const (fun x _ => hgx x) (by norm_num) (by norm_num) ?_ ?_
    (fun x _ => by simp) (fun x _ => hslice x) rfl rfl (fun x _ => hslice x) ?_ ?_ ?_
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (2 : ℝ) • x ^ 2 + (-1 / 4 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · refine (concaveOn_const (-1 / 4 : ℝ) (convex_Icc _ _)).congr ?_
    intro x _
    ring
  · refine (hq.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · intro x _
    rw [strongConcaveOn_iff_convex]
    refine (concaveOn_const (x ^ 3 / 3 - x / 4) (convex_closedBall (0 : ℝ) (2 * 1))).congr ?_
    intro y _
    simp only [spToyF, Real.norm_eq_abs, sq_abs]
    ring
  · intro v
    have hc : ConvexOn ℝ univ (fun t : ℝ => ‖v‖ • t ^ 2) :=
      (even_two.convexOn_pow).smul (norm_nonneg v)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul, zero_apply]
    ring
  · intro v
    have hc : ConcaveOn ℝ univ (fun t : ℝ => -(‖v‖ • t ^ 2)) :=
      ((even_two.convexOn_pow).smul (norm_nonneg v)).neg
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul, zero_apply]
    ring
  · simp only [spToyF]
    norm_num


/-- A field meeting (H1): `f (x, y) = x³/3 - x/4 - 20 y²`, so `λ = 40 > 8 r m² = 32` at `r = 1`,
`m = 2`, and gap exactly `r³/6`. -/
noncomputable def h1ToyF (p : ℝ × ℝ) : ℝ := p.1 ^ 3 / 3 - p.1 / 4 - 20 * p.2 ^ 2

/-- **`ridge_capHyp_H1` is not vacuous.** `h1ToyF` meets every hypothesis, with a constant
transverse Hessian `-40`, `h = 0`, `w = 0`, `G0 = F = x² - 1/4`, `m = 2` and `λ = 40`. -/
theorem h1Toy_capHyp :
    CapHyp h1ToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (h1ToyF (-1 / 2, 0)) (h1ToyF (1 / 2, 0)) ∧
      ∀ x ∈ frontier (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)),
        x ≠ (1 / 2, 0) → h1ToyF x < h1ToyF (1 / 2, 0) := by
  have hslope : ∀ x y : ℝ, HasFDerivAt (fun y : ℝ => h1ToyF (x, y))
      (ContinuousLinearMap.mul ℝ ℝ (-40 * y)) y := by
    intro x y
    rw [hasFDerivAt_iff_hasDerivAt]
    have h1 := ((hasDerivAt_pow 2 y).const_mul 20).const_sub (x ^ 3 / 3 - x / 4)
    convert h1 using 1
    · funext t
      simp only [h1ToyF]
    · simp
      ring
  have hslice : ∀ x : ℝ, HasFDerivAt (fun y : ℝ => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    simpa using hslope x 0
  have hhess : ∀ y : ℝ, HasFDerivAt (fun y : ℝ => ContinuousLinearMap.mul ℝ ℝ (-40 * y))
      ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) y := by
    intro y
    convert ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ).hasFDerivAt (x := y) using 1
    funext t
    ext
    simp
  have hgx : ∀ x : ℝ, HasDerivAt (fun x => h1ToyF (x, 0)) (x ^ 2 - 1 / 4) x := by
    intro x
    have h1 := ((hasDerivAt_pow 3 x).div_const 3).sub ((hasDerivAt_id x).div_const 4)
    convert h1 using 1
    · funext y
      simp only [h1ToyF, id, Pi.sub_apply]
      ring
    · push_cast
      ring
  have hq : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
    ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
  refine ridge_capHyp_H1 (E := ℝ) (f := h1ToyF) (h := fun _ => (0 : ℝ))
    (F := fun x => x ^ 2 - 1 / 4) (G0 := fun x => x ^ 2 - 1 / 4) (w := fun _ => 0)
    (Dφ := fun p => ContinuousLinearMap.mul ℝ ℝ (-40 * p.2))
    (H := fun _ => (-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) (Hx := fun _ => 0) (Hr := fun _ _ => 0)
    (r := 1) (m := 2) (lam := 40)
    one_pos (fun x _ => hgx x) ?_ ?_ (by norm_num) (by norm_num)
    (fun p _ => hslope p.1 p.2) (fun p _ => hhess p.2)
    (fun x _ => hasDerivAt_const x ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun x _ => by simp only [ContinuousLinearMap.opNorm_zero]; norm_num)
    (fun p _ s _ => hasDerivAt_const s ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun p _ s _ => by simp only [ContinuousLinearMap.opNorm_zero]; positivity) ?_ (by norm_num)
    continuousOn_const (fun x _ => hgx x) (by norm_num) (by norm_num) ?_
    (fun x _ => by simp) (fun x _ => hslice x) rfl rfl (fun x _ => hslice x) ?_ ?_ ?_
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (2 : ℝ) • x ^ 2 + (-1 / 4 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · refine (concaveOn_const (-1 / 4 : ℝ) (convex_Icc _ _)).congr ?_
    intro x _
    ring
  · intro v
    simp only [smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul,
      Real.norm_eq_abs, sq_abs]
    nlinarith [sq_nonneg v]
  · refine (hq.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    ring
  · intro v
    have hc : ConvexOn ℝ univ (fun t : ℝ => ‖v‖ • t ^ 2) :=
      (even_two.convexOn_pow).smul (norm_nonneg v)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul, zero_apply]
    ring
  · intro v
    have hc : ConcaveOn ℝ univ (fun t : ℝ => -(‖v‖ • t ^ 2)) :=
      ((even_two.convexOn_pow).smul (norm_nonneg v)).neg
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul, zero_apply]
    ring
  · simp only [h1ToyF]
    norm_num


/-- **`ridge_differentiable` is not vacuous.** For `h1ToyF`, with `h = 0`, transverse gradient
`G (x, y) = -40 y`, its constant derivative `DG = -40 · snd`, `δ = 40` and `U = (-3, 3)`, the
statement lists every hypothesis of `ridge_differentiable` at this instance, and its last
conjunct is that theorem's conclusion, obtained by applying it. -/
theorem h1Toy_ridge_differentiable :
    IsOpen (Ioo (-3 : ℝ) 3) ∧ Icc (-2 * (1 : ℝ)) (2 * 1) ⊆ Ioo (-3) 3 ∧
      (∀ x ∈ Ioo (-3 : ℝ) 3,
        StrongConcaveOn (Metric.closedBall (0 : ℝ) (2 * 1)) 40 (fun y => h1ToyF (x, y))) ∧
      (∀ x ∈ Ioo (-3 : ℝ) 3, (0 : ℝ) ∈ Metric.closedBall (0 : ℝ) (2 * 1)) ∧
      (∀ x ∈ Ioo (-3 : ℝ) 3, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0) ∧
      (∀ x ∈ Icc (-2 * (1 : ℝ)) (2 * 1), (0 : ℝ) ∈ Metric.ball (0 : ℝ) (2 * 1)) ∧
      (∀ x ∈ Icc (-2 * (1 : ℝ)) (2 * 1), ∀ᶠ v in nhds (x, (0 : ℝ)),
        HasFDerivAt (fun y => h1ToyF (v.1, y)) (ContinuousLinearMap.mul ℝ ℝ (-40 * v.2)) v.2) ∧
      (∀ x ∈ Icc (-2 * (1 : ℝ)) (2 * 1),
        HasStrictFDerivAt (fun p : ℝ × ℝ => ContinuousLinearMap.mul ℝ ℝ (-40 * p.2))
          (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) (x, 0)) ∧
      (∀ v : ℝ, ((((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) ∘L
        ContinuousLinearMap.inr ℝ ℝ ℝ) v v ≤ -40 * ‖v‖ ^ 2) ∧
      ∀ x ∈ Icc (-2 * (1 : ℝ)) (2 * 1), HasStrictFDerivAt (fun _ : ℝ => (0 : ℝ))
        (-((((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) ∘L
            ContinuousLinearMap.inr ℝ ℝ ℝ).inverse ∘L
          ((((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) ∘L
            ContinuousLinearMap.inl ℝ ℝ ℝ)) x := by
  have hslope : ∀ x y : ℝ, HasFDerivAt (fun y : ℝ => h1ToyF (x, y))
      (ContinuousLinearMap.mul ℝ ℝ (-40 * y)) y := by
    intro x y
    rw [hasFDerivAt_iff_hasDerivAt]
    have h1 := ((hasDerivAt_pow 2 y).const_mul 20).const_sub (x ^ 3 / 3 - x / 4)
    convert h1 using 1
    · funext t
      simp only [h1ToyF]
    · simp
      ring
  have hGs : ∀ p : ℝ × ℝ, HasStrictFDerivAt (fun p : ℝ × ℝ => ContinuousLinearMap.mul ℝ ℝ (-40 * p.2))
      (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    convert (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L
      ContinuousLinearMap.snd ℝ ℝ ℝ).hasStrictFDerivAt (x := p) using 1
    funext q
    ext
    simp
  have hIU : Icc (-2 * (1 : ℝ)) (2 * 1) ⊆ Ioo (-3) 3 :=
    fun t ht => ⟨by linarith [ht.1], by linarith [ht.2]⟩
  have hconc : ∀ x ∈ Ioo (-3 : ℝ) 3,
      StrongConcaveOn (Metric.closedBall (0 : ℝ) (2 * 1)) 40 (fun y => h1ToyF (x, y)) := by
    intro t _
    rw [strongConcaveOn_iff_convex]
    refine (concaveOn_const (t ^ 3 / 3 - t / 4) (convex_closedBall (0 : ℝ) (2 * 1))).congr ?_
    intro y _
    simp only [h1ToyF, Real.norm_eq_abs, sq_abs]
    ring
  have hball : ∀ x ∈ Ioo (-3 : ℝ) 3, (0 : ℝ) ∈ Metric.closedBall (0 : ℝ) (2 * 1) :=
    fun t _ => by simp
  have hcrit : ∀ x ∈ Ioo (-3 : ℝ) 3, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 :=
    fun t _ => by simpa using hslope t 0
  have hint : ∀ x ∈ Icc (-2 * (1 : ℝ)) (2 * 1), (0 : ℝ) ∈ Metric.ball (0 : ℝ) (2 * 1) :=
    fun t _ => by simp
  have hG : ∀ x ∈ Icc (-2 * (1 : ℝ)) (2 * 1), ∀ᶠ v in nhds (x, (0 : ℝ)),
      HasFDerivAt (fun y => h1ToyF (v.1, y)) (ContinuousLinearMap.mul ℝ ℝ (-40 * v.2)) v.2 :=
    fun t _ => Filter.Eventually.of_forall (fun v => hslope v.1 v.2)
  have hneg : ∀ v : ℝ, ((((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L
      ContinuousLinearMap.snd ℝ ℝ ℝ) ∘L ContinuousLinearMap.inr ℝ ℝ ℝ) v v ≤ -40 * ‖v‖ ^ 2 := by
    intro v
    simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.inr_apply,
      ContinuousLinearMap.coe_snd', smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul,
      Real.norm_eq_abs, sq_abs]
    nlinarith [sq_nonneg v]
  refine ⟨isOpen_Ioo, hIU, hconc, hball, hcrit, hint, hG, fun x _ => hGs (x, 0), hneg, ?_⟩
  exact ridge_differentiable (E := ℝ) (f := h1ToyF)
    (G := fun p => ContinuousLinearMap.mul ℝ ℝ (-40 * p.2))
    (DG := fun _ => ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ)
    (h := fun _ => (0 : ℝ)) (U := Ioo (-3) 3) (r := 1) (δ := 40) (by norm_num) isOpen_Ioo hIU
    hconc hball hcrit hint hG (fun x _ => hGs (x, 0)) (fun _ _ => hneg)

/-- **`ridge_capHyp_source` is not vacuous.** `h1ToyF` meets every source hypothesis on the full
collar `[-5/2, 5/2] × B̄(0, 2)` (`ε = 1/2`): joint derivative `(x² - 1/4, -40y)`, transverse gradient `-40y`
with derivative `-40 · snd`, constant transverse Hessian `-40`, `m = 2`, `λ = 40 > 32`, pins
`(∓1/2, 0)` and gap `1/6`. The theorem then gives the ridge, and for `h = 0`, which meets L12
(`F = x² - 1/4`), the cap with `M = (-1/2, 0)`. -/
theorem h1Toy_source :
    (∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - 1 / 2 : ℝ) (2 * 1 + 1 / 2), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp h1ToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (h1ToyF (-1 / 2, 0)) (h1ToyF (1 / 2, 0)) := by
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt h1ToyF ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have hf : HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
      hasFDerivAt_fst
    have hs : HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      hasFDerivAt_snd
    have h1 := HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow hf 3) (1 / 3)) (HasFDerivAt.const_mul hf (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow hs 2) 20)
    convert h1 using 1
    · funext q
      simp only [h1ToyF, Pi.sub_apply]
      ring
    · ext <;> simp; ring
  have hGs : ∀ p : ℝ × ℝ, HasStrictFDerivAt (fun q : ℝ × ℝ => ((q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp
        (ContinuousLinearMap.inr ℝ ℝ ℝ))
      (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    convert (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L
      ContinuousLinearMap.snd ℝ ℝ ℝ).hasStrictFDerivAt (x := p) using 1
    funext q
    ext
    simp
  have hH : ∀ p : ℝ × ℝ, HasFDerivAt (fun y => ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * y) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ))
      ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) p.2 := by
    intro p
    convert ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ).hasFDerivAt (x := p.2) using 1
    funext y
    ext
    simp
  have hsrc := ridge_capHyp_source (E := ℝ) (f := h1ToyF)
    (Df := fun p => (p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
    (DG := fun _ => ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ)
    (H := fun _ => (-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) (Hx := fun _ => 0)
    (Hr := fun _ _ => 0) (r := 1) (m := 2) (lam := 40) (ε := 1 / 2) one_pos (by norm_num)
    (by norm_num)
    (fun p _ => hDf p) (fun p _ => hGs p) (fun p _ => hH p)
    (fun x _ => hasDerivAt_const x ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun x _ => by simp only [ContinuousLinearMap.opNorm_zero]; norm_num)
    (fun p _ s _ => hasDerivAt_const s ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun p _ s _ => by simp only [ContinuousLinearMap.opNorm_zero]; positivity)
    (fun v => by
      simp only [smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num) (by ext <;> norm_num) (by ext <;> norm_num) ?_ ?_ ?_ ?_ (by simp only [h1ToyF]; norm_num)
  · obtain ⟨hex, hcap⟩ := hsrc
    have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
      intro x
      have := slice_hasFDerivAt (hDf (x, 0))
      convert this using 1
      ext
      simp
    have hq : ConvexOn ℝ univ (fun x : ℝ => (7 / 8 : ℝ) • x ^ 2 + (-7 / 32 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
      ((hq.subset (subset_univ _) (convex_Icc _ _)).congr fun x _ => by
        simp only [smul_eq_mul]
        simp
        ring)
    exact ⟨hex, hC⟩
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (2 : ℝ) • x ^ 2 + (-1 / 4 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    simp
    ring
  · refine (concaveOn_const (-1 / 4 : ℝ) (convex_Icc _ _)).congr ?_
    intro x _
    simp
    ring
  · intro v
    have hc : ConvexOn ℝ univ (fun t : ℝ => ‖v‖ • t ^ 2) :=
      (even_two.convexOn_pow).smul (norm_nonneg v)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul]
    simp
  · intro v
    have hc : ConcaveOn ℝ univ (fun t : ℝ => -(‖v‖ • t ^ 2)) :=
      ((even_two.convexOn_pow).smul (norm_nonneg v)).neg
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul]
    simp

/-- **`ridge_capHyp_C3` is not vacuous.** `h1ToyF` meets every hypothesis of `ridge_capHyp_C3`
on the collar `ε = 1/2`: besides the data of `h1Toy_source`, second derivative
`D²f = diag(2x, -40)`, third derivatives `T a b = 2 a₁ b₁ · fst` (only `f_xxx = 2 ≥ 7/4` is nonzero,
so every other block vanishes). The theorem then gives the ridge and the cap with `M = (-1/2, 0)`,
with L12 derived rather than supplied. -/
theorem h1Toy_C3 :
    (∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - 1 / 2 : ℝ) (2 * 1 + 1 / 2), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp h1ToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (h1ToyF (-1 / 2, 0)) (h1ToyF (1 / 2, 0)) := by
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt h1ToyF ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have hf : HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
      hasFDerivAt_fst
    have hs : HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      hasFDerivAt_snd
    have h1 := HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow hf 3) (1 / 3)) (HasFDerivAt.const_mul hf (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow hs 2) 20)
    convert h1 using 1
    · funext q
      simp only [h1ToyF, Pi.sub_apply]
      ring
    · ext <;> simp; ring
  have hGs : ∀ p : ℝ × ℝ, HasStrictFDerivAt (fun q : ℝ × ℝ => ((q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp
        (ContinuousLinearMap.inr ℝ ℝ ℝ))
      (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    convert (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L
      ContinuousLinearMap.snd ℝ ℝ ℝ).hasStrictFDerivAt (x := p) using 1
    funext q
    ext
    simp
  have hH : ∀ p : ℝ × ℝ, HasFDerivAt (fun y => ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * y) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ))
      ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) p.2 := by
    intro p
    convert ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ).hasFDerivAt (x := p.2) using 1
    funext y
    ext
    simp
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hD2 : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) p := by
    intro p
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4) ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ)
        p := by
      have := (HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)
      convert this using 1
      ext <;> simp
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2) ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      HasFDerivAt.const_mul (hs p) (-40)
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hD3 : ∀ p : ℝ × ℝ, ∀ a b : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ =>
      ((((2 * q.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) a) b)
      ((2 * a.1 * b.1) • ContinuousLinearMap.fst ℝ ℝ ℝ) p := by
    intro p a b
    have := (HasFDerivAt.const_mul (hf p) (2 * a.1 * b.1)).add_const (-40 * a.2 * b.2)
    convert this using 1
    funext q
    simp
    ring
  have hT : ∀ p : ℝ × ℝ, ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : ℝ),
      |((2 * (s₁, y₁).1 * (s₂, y₂).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) (s₃, y₃) -
          s₁ * s₂ * s₃ * ((2 * ((1 : ℝ), (0 : ℝ)).1 * ((1 : ℝ), (0 : ℝ)).1) •
            ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0)| ≤
        2 * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
    intro p s₁ s₂ s₃ y₁ y₂ y₃
    have h0 : ((2 * (s₁, y₁).1 * (s₂, y₂).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) (s₃, y₃) -
        s₁ * s₂ * s₃ * ((2 * ((1 : ℝ), (0 : ℝ)).1 * ((1 : ℝ), (0 : ℝ)).1) •
          ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0) = 0 := by
      simp
      ring
    rw [h0, abs_zero]
    have a1 : |s₁| ≤ |s₁| + ‖y₁‖ := le_add_of_nonneg_right (norm_nonneg _)
    have a2 : |s₂| ≤ |s₂| + ‖y₂‖ := le_add_of_nonneg_right (norm_nonneg _)
    have a3 : |s₃| ≤ |s₃| + ‖y₃‖ := le_add_of_nonneg_right (norm_nonneg _)
    have := mul_le_mul (mul_le_mul a1 a2 (abs_nonneg _) (by positivity)) a3 (abs_nonneg _)
      (by positivity)
    linarith
  have hsrc := ridge_capHyp_C3 (E := ℝ) (f := h1ToyF)
    (Df := fun p => (p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
    (D2f := fun p => ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight
        (ContinuousLinearMap.fst ℝ ℝ ℝ) +
      ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ))
    (T3 := fun _ a b => (2 * a.1 * b.1) • ContinuousLinearMap.fst ℝ ℝ ℝ)
    (DG := fun _ => ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ)
    (H := fun _ => (-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) (Hx := fun _ => 0)
    (Hr := fun _ _ => 0) (r := 1) (m := 2) (lam := 40) (ε := 1 / 2) one_pos (by norm_num)
    (by norm_num) (fun p _ => hDf p) (fun p _ => hD2 p) (fun p _ a b => hD3 p a b)
    (fun p _ => hT p) (fun p _ => by norm_num) (fun p _ => hGs p) (fun p _ => hH p)
    (fun x _ => hasDerivAt_const x ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun x _ => by simp only [ContinuousLinearMap.opNorm_zero]; norm_num)
    (fun p _ s _ => hasDerivAt_const s ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun p _ s _ => by simp only [ContinuousLinearMap.opNorm_zero]; positivity)
    (fun v => by
      simp only [smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num) (by ext <;> norm_num) (by ext <;> norm_num) ?_ ?_ ?_ ?_
    (by simp only [h1ToyF]; norm_num)
  · obtain ⟨hex, hcap⟩ := hsrc
    have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
      intro x
      have := slice_hasFDerivAt (hDf (x, 0))
      convert this using 1
      ext
      simp
    obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
    exact ⟨hex, hC⟩
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (2 : ℝ) • x ^ 2 + (-1 / 4 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    simp
    ring
  · refine (concaveOn_const (-1 / 4 : ℝ) (convex_Icc _ _)).congr ?_
    intro x _
    simp
    ring
  · intro v
    have hc : ConvexOn ℝ univ (fun t : ℝ => ‖v‖ • t ^ 2) :=
      (even_two.convexOn_pow).smul (norm_nonneg v)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul]
    simp
  · intro v
    have hc : ConcaveOn ℝ univ (fun t : ℝ => -(‖v‖ • t ^ 2)) :=
      ((even_two.convexOn_pow).smul (norm_nonneg v)).neg
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul]
    simp

/-- **`ridge_capHyp_C4` is not vacuous.** `h1ToyF` meets every hypothesis of `ridge_capHyp_C4`
on the collar `ε = 1/2`: the data of `h1Toy_C3`, with `f_xxx = 2` constant, so its derivative is `0`,
both fourth-order blocks vanish and (H2) holds with `n = 0`. The theorem gives the ridge and the cap with `M = (-1/2, 0)`, with
neither L12 nor `f_xxx ≥ 7/4` supplied. -/
theorem h1Toy_C4 :
    (∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - 1 / 2 : ℝ) (2 * 1 + 1 / 2), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp h1ToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (h1ToyF (-1 / 2, 0)) (h1ToyF (1 / 2, 0)) := by
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt h1ToyF ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have hf : HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
      hasFDerivAt_fst
    have hs : HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      hasFDerivAt_snd
    have h1 := HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow hf 3) (1 / 3)) (HasFDerivAt.const_mul hf (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow hs 2) 20)
    convert h1 using 1
    · funext q
      simp only [h1ToyF, Pi.sub_apply]
      ring
    · ext <;> simp; ring
  have hGs : ∀ p : ℝ × ℝ, HasStrictFDerivAt (fun q : ℝ × ℝ => ((q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp
        (ContinuousLinearMap.inr ℝ ℝ ℝ))
      (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    convert (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L
      ContinuousLinearMap.snd ℝ ℝ ℝ).hasStrictFDerivAt (x := p) using 1
    funext q
    ext
    simp
  have hH : ∀ p : ℝ × ℝ, HasFDerivAt (fun y => ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * y) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ))
      ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) p.2 := by
    intro p
    convert ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ).hasFDerivAt (x := p.2) using 1
    funext y
    ext
    simp
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hD2 : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) p := by
    intro p
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4) ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ)
        p := by
      have := (HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)
      convert this using 1
      ext <;> simp
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2) ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      HasFDerivAt.const_mul (hs p) (-40)
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hD3 : ∀ p : ℝ × ℝ, ∀ a b : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ =>
      ((((2 * q.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) a) b)
      ((2 * a.1 * b.1) • ContinuousLinearMap.fst ℝ ℝ ℝ) p := by
    intro p a b
    have := (HasFDerivAt.const_mul (hf p) (2 * a.1 * b.1)).add_const (-40 * a.2 * b.2)
    convert this using 1
    funext q
    simp
    ring
  have hT : ∀ p : ℝ × ℝ, ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : ℝ),
      |((2 * (s₁, y₁).1 * (s₂, y₂).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) (s₃, y₃) -
          s₁ * s₂ * s₃ * ((2 * ((1 : ℝ), (0 : ℝ)).1 * ((1 : ℝ), (0 : ℝ)).1) •
            ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0)| ≤
        2 * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
    intro p s₁ s₂ s₃ y₁ y₂ y₃
    have h0 : ((2 * (s₁, y₁).1 * (s₂, y₂).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) (s₃, y₃) -
        s₁ * s₂ * s₃ * ((2 * ((1 : ℝ), (0 : ℝ)).1 * ((1 : ℝ), (0 : ℝ)).1) •
          ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0) = 0 := by
      simp
      ring
    rw [h0, abs_zero]
    have a1 : |s₁| ≤ |s₁| + ‖y₁‖ := le_add_of_nonneg_right (norm_nonneg _)
    have a2 : |s₂| ≤ |s₂| + ‖y₂‖ := le_add_of_nonneg_right (norm_nonneg _)
    have a3 : |s₃| ≤ |s₃| + ‖y₃‖ := le_add_of_nonneg_right (norm_nonneg _)
    have := mul_le_mul (mul_le_mul a1 a2 (abs_nonneg _) (by positivity)) a3 (abs_nonneg _)
      (by positivity)
    linarith
  have hsrc := ridge_capHyp_C4 (E := ℝ) (f := h1ToyF)
    (Df := fun p => (p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
    (D2f := fun p => ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight
        (ContinuousLinearMap.fst ℝ ℝ ℝ) +
      ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ))
    (T3 := fun _ a b => (2 * a.1 * b.1) • ContinuousLinearMap.fst ℝ ℝ ℝ) (Φ4 := fun _ => 0) (n := 0)
    (DG := fun _ => ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ)
    (H := fun _ => (-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) (Hx := fun _ => 0)
    (Hr := fun _ _ => 0) (r := 1) (m := 2) (lam := 40) (ε := 1 / 2) one_pos (by norm_num)
    (by norm_num) (fun p _ => hDf p) (fun p _ => hD2 p) (fun p _ a b => hD3 p a b)
    (fun p _ => hT p) (fun p _ => hasFDerivAt_const _ p) (fun p _ => by simp) (fun p _ => by simp)
    (by norm_num)
    (fun p _ => hGs p) (fun p _ => hH p)
    (fun x _ => hasDerivAt_const x ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun x _ => by simp only [ContinuousLinearMap.opNorm_zero]; norm_num)
    (fun p _ s _ => hasDerivAt_const s ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun p _ s _ => by simp only [ContinuousLinearMap.opNorm_zero]; positivity)
    (fun v => by
      simp only [smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num) (by ext <;> norm_num) (by ext <;> norm_num) ?_ ?_ ?_ ?_
    (by simp only [h1ToyF]; norm_num)
  · obtain ⟨hex, hcap⟩ := hsrc
    have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
      intro x
      have := slice_hasFDerivAt (hDf (x, 0))
      convert this using 1
      ext
      simp
    obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
    exact ⟨hex, hC⟩
  · have hc : ConvexOn ℝ univ (fun x : ℝ => (2 : ℝ) • x ^ 2 + (-1 / 4 : ℝ)) :=
      ((even_two.convexOn_pow).smul (by norm_num)).add (convexOn_const _ convex_univ)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro x _
    simp only [smul_eq_mul]
    simp
    ring
  · refine (concaveOn_const (-1 / 4 : ℝ) (convex_Icc _ _)).congr ?_
    intro x _
    simp
    ring
  · intro v
    have hc : ConvexOn ℝ univ (fun t : ℝ => ‖v‖ • t ^ 2) :=
      (even_two.convexOn_pow).smul (norm_nonneg v)
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul]
    simp
  · intro v
    have hc : ConcaveOn ℝ univ (fun t : ℝ => -(‖v‖ • t ^ 2)) :=
      ((even_two.convexOn_pow).smul (norm_nonneg v)).neg
    refine (hc.subset (subset_univ _) (convex_Icc _ _)).congr ?_
    intro t _
    simp only [smul_eq_mul]
    simp

/-- **`ridge_capHyp_D` is not vacuous.** `h1ToyF` meets every hypothesis of `ridge_capHyp_D` with
`ε₀ = 1/2`. `T3` is constant, so `T4 = 0` and `K = 0`, and `|f_xxx| = 2 = m` on the pin segment.
The theorem chooses the collar width itself. It gives the ridge on `(−2 − ε, 2 + ε)` for some
`ε > 0` and the cap with `M = (−1/2, 0)`, from hypotheses on `D` and qualitative collar data only. -/
theorem h1Toy_D :
    (∃ ε > 0, ∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - ε : ℝ) (2 * 1 + ε), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp h1ToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (h1ToyF (-1 / 2, 0)) (h1ToyF (1 / 2, 0)) := by
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt h1ToyF ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have hf : HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
      hasFDerivAt_fst
    have hs : HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      hasFDerivAt_snd
    have h1 := HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow hf 3) (1 / 3)) (HasFDerivAt.const_mul hf (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow hs 2) 20)
    convert h1 using 1
    · funext q
      simp only [h1ToyF, Pi.sub_apply]
      ring
    · ext <;> simp; ring
  have hGs : ∀ p : ℝ × ℝ, HasStrictFDerivAt (fun q : ℝ × ℝ => ((q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp
        (ContinuousLinearMap.inr ℝ ℝ ℝ))
      (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    convert (((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L
      ContinuousLinearMap.snd ℝ ℝ ℝ).hasStrictFDerivAt (x := p) using 1
    funext q
    ext
    simp
  have hH : ∀ p : ℝ × ℝ, HasFDerivAt (fun y => ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * y) • ContinuousLinearMap.snd ℝ ℝ ℝ).comp (ContinuousLinearMap.inr ℝ ℝ ℝ))
      ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) p.2 := by
    intro p
    convert ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ).hasFDerivAt (x := p.2) using 1
    funext y
    ext
    simp
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hD2 : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) p := by
    intro p
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4) ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ)
        p := by
      have := (HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)
      convert this using 1
      ext <;> simp
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2) ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      HasFDerivAt.const_mul (hs p) (-40)
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hD3 : ∀ p : ℝ × ℝ, ∀ a b : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ =>
      ((((2 * q.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) a) b)
      ((2 * a.1 * b.1) • ContinuousLinearMap.fst ℝ ℝ ℝ) p := by
    intro p a b
    have := (HasFDerivAt.const_mul (hf p) (2 * a.1 * b.1)).add_const (-40 * a.2 * b.2)
    convert this using 1
    funext q
    simp
    ring
  have hT : ∀ p : ℝ × ℝ, ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : ℝ),
      |((2 * (s₁, y₁).1 * (s₂, y₂).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) (s₃, y₃) -
          s₁ * s₂ * s₃ * ((2 * ((1 : ℝ), (0 : ℝ)).1 * ((1 : ℝ), (0 : ℝ)).1) •
            ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0)| ≤
        2 * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
    intro p s₁ s₂ s₃ y₁ y₂ y₃
    have h0 : ((2 * (s₁, y₁).1 * (s₂, y₂).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) (s₃, y₃) -
        s₁ * s₂ * s₃ * ((2 * ((1 : ℝ), (0 : ℝ)).1 * ((1 : ℝ), (0 : ℝ)).1) •
          ContinuousLinearMap.fst ℝ ℝ ℝ) (1, 0) = 0 := by
      simp
      ring
    rw [h0, abs_zero]
    have a1 : |s₁| ≤ |s₁| + ‖y₁‖ := le_add_of_nonneg_right (norm_nonneg _)
    have a2 : |s₂| ≤ |s₂| + ‖y₂‖ := le_add_of_nonneg_right (norm_nonneg _)
    have a3 : |s₃| ≤ |s₃| + ‖y₃‖ := le_add_of_nonneg_right (norm_nonneg _)
    have := mul_le_mul (mul_le_mul a1 a2 (abs_nonneg _) (by positivity)) a3 (abs_nonneg _)
      (by positivity)
    linarith
  have hsrc := ridge_capHyp_D (E := ℝ) (f := h1ToyF)
    (Df := fun p => (p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
    (D2f := fun p => ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight
        (ContinuousLinearMap.fst ℝ ℝ ℝ) +
      ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ))
    (T3 := fun _ a b => (2 * a.1 * b.1) • ContinuousLinearMap.fst ℝ ℝ ℝ)
    (T4 := fun _ _ _ _ => 0) (Φ4 := fun _ => 0) (n := 0)
    (DG := fun _ => ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) ∘L ContinuousLinearMap.snd ℝ ℝ ℝ)
    (H := fun _ => (-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ) (Hx := fun _ => 0)
    (Hr := fun _ _ => 0) (r := 1) (m := 2) (K := 0) (lam := 40) (ε₀ := 1 / 2) one_pos
    (by norm_num) le_rfl
    (fun p _ => hDf p) (fun p _ => hD2 p) (fun p _ a b => hD3 p a b)
    (fun p _ a b c => hasDerivAt_const p.1 _) (fun p _ s₁ s₂ s₃ y₁ y₂ y₃ => by simp)
    (fun p _ => hT p) (fun x _ => by simp) (fun p _ => hasFDerivAt_const _ p) (fun p _ => by simp)
    (fun p _ => by simp) (by norm_num)
    (fun p _ => hGs p) (fun p _ => hH p)
    (fun x _ => hasDerivAt_const x ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun p _ s _ => hasDerivAt_const s ((-40 : ℝ) • ContinuousLinearMap.mul ℝ ℝ))
    (fun v => by
      simp only [smul_apply, ContinuousLinearMap.mul_apply', smul_eq_mul, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num) (by ext <;> norm_num) (by ext <;> norm_num)
    (by simp only [h1ToyF]; norm_num)
  obtain ⟨ε, hε, -, hex, hcap⟩ := hsrc
  have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    have := slice_hasFDerivAt (hDf (x, 0))
    convert this using 1
    ext
    simp
  obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
  exact ⟨⟨ε, hε, hex⟩, hC⟩


/-- **`ridge_capHyp_E` is not vacuous.** `h1ToyF` is a polynomial, so it is `C⁴` on `U = ℝ²`.
Its third derivative is `D³f[a, b, c] = 2 a.1 b.1 c.1` and its fourth derivative vanishes, so
`m = 2` and `n = 0`; `D²f[(0, v), (0, v)] = -40 v²` gives `λ = 40`. From these hypotheses on `D`
alone, the theorem gives the ridge on `(-2 - ε, 2 + ε)` for some `ε > 0` and the cap with
`M = (-1/2, 0)`. -/
theorem h1Toy_E :
    (∃ ε > 0, ∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - ε : ℝ) (2 * 1 + ε), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp h1ToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (h1ToyF (-1 / 2, 0)) (h1ToyF (1 / 2, 0)) := by
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt h1ToyF ((p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have h1 := HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hf p) 3) (1 / 3))
      (HasFDerivAt.const_mul (hf p) (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hs p) 2) 20)
    convert h1 using 1
    · funext q
      simp only [h1ToyF, Pi.sub_apply]
      ring
    · ext <;> simp; ring
  have hD2 : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) p := by
    intro p
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4) ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ)
        p := by
      have := (HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)
      convert this using 1
      ext <;> simp
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2) ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      HasFDerivAt.const_mul (hs p) (-40)
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hcd : ContDiff ℝ 4 h1ToyF := by unfold h1ToyF; fun_prop
  have hfd : fderiv ℝ h1ToyF = fun p => (p.1 ^ 2 - 1 / 4) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ := funext fun p => (hDf p).fderiv
  have h2 : ∀ (y : ℝ × ℝ) (m : Fin 2 → ℝ × ℝ),
      iteratedFDeriv ℝ 2 h1ToyF y m = 2 * y.1 * (m 0).1 * (m 1).1 - 40 * (m 0).2 * (m 1).2 := by
    intro y m
    rw [iteratedFDeriv_two_apply, hfd, (hD2 y).fderiv]
    simp
    ring
  have h3 : ∀ (x : ℝ × ℝ) (m : Fin 3 → ℝ × ℝ),
      iteratedFDeriv ℝ 3 h1ToyF x m = 2 * (m 0).1 * (m 1).1 * (m 2).1 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 2 h1ToyF y (Fin.tail m))
        ((2 * (m 1).1 * (m 2).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) x := by
      have := ((hf x).const_mul (2 * (m 1).1 * (m 2).1)).sub_const (40 * (m 1).2 * (m 2).2)
      convert this using 1
      funext y; rw [h2]; simp [Fin.tail]; ring
    rw [hg.fderiv]
    simp
    ring
  have h4 : ∀ (x : ℝ × ℝ) (m : Fin 4 → ℝ × ℝ), iteratedFDeriv ℝ 4 h1ToyF x m = 0 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : (fun y => iteratedFDeriv ℝ 3 h1ToyF y (Fin.tail m)) =
        fun _ => 2 * (m 1).1 * (m 2).1 * (m 3).1 := by
      funext y; rw [h3]; simp [Fin.tail]
    rw [hg]
    simp
  have hsrc := ridge_capHyp_E (E := ℝ) (f := h1ToyF) (U := univ) (r := 1) (m := 2) (n := 0)
    (lam := 40) one_pos isOpen_univ (subset_univ _) hcd.contDiffOn
    (fun p _ s₁ s₂ s₃ y₁ y₂ y₃ => by
      rw [h3, h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons]
      have h0 : 2 * s₁ * s₂ * s₃ - s₁ * s₂ * s₃ * (2 * 1 * 1 * 1) = 0 := by ring
      rw [h0, abs_zero]
      exact mul_nonneg (by norm_num) (block_nonneg s₁ s₂ s₃ y₁ y₂ y₃))
    (fun x _ => by rw [h3]; simp)
    (fun p _ => by rw [h4]; simp)
    (fun p _ v => by rw [h4]; simp)
    (by norm_num)
    (fun v => by
      rw [h2]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by simp only [h1ToyF]; norm_num)
  obtain ⟨ε, hε, hex, hcap⟩ := hsrc
  have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => h1ToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    have := slice_hasFDerivAt (hDf (x, 0))
    convert this using 1
    ext
    simp
  obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
  exact ⟨⟨ε, hε, hex⟩, hC⟩


/-- A pin-preserving quartic perturbation of `h1ToyF`: `q(x) = (x² - 1/4)²/480` vanishes to second
order at both pins, so they stay critical and the gap stays `1/6`, while `f_xxx = 2 + x/20` varies
and `D⁴f = 1/20 ≠ 0`. -/
noncomputable def quarticToyF (p : ℝ × ℝ) : ℝ :=
  p.1 ^ 3 / 3 - p.1 / 4 - 20 * p.2 ^ 2 + (p.1 ^ 2 - 1 / 4) ^ 2 / 480

/-- **`ridge_capHyp_E` away from the degenerate face.** On `quarticToyF` the third derivative
`f_xxx = 2 + x/20` varies, the fourth derivative `D⁴f = 1/20` is nonzero, and (H2) holds with
equality, `rn = 1/20` (`r = 1`, `n = 1/20`); `|f_xxx| ≤ 81/40 = m` on the pin segment, and (H1) holds
as `8m² = 32.805 < 40 = λ`. So `fxxx_lower`'s estimate is used with `n > 0` and the collar bound
`K` is positive, unlike `h1Toy_E` (where `D⁴f = 0`). The theorem gives the ridge on
`(-2 - ε, 2 + ε)` for some `ε > 0` and the cap with `M = (-1/2, 0)`. -/
theorem quarticToy_E :
    (∃ ε > 0, ∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - ε : ℝ) (2 * 1 + ε), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => quarticToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp quarticToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (quarticToyF (-1 / 2, 0)) (quarticToyF (1 / 2, 0)) := by
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt quarticToyF
      ((p.1 ^ 2 - 1 / 4 + (4 * p.1 ^ 3 - p.1) / 480) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have h1 := HasFDerivAt.add (HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hf p) 3) (1 / 3))
      (HasFDerivAt.const_mul (hf p) (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hs p) 2) 20))
      (HasFDerivAt.const_mul (HasFDerivAt.pow ((HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)) 2)
        (1 / 480))
    convert h1 using 1
    · funext q
      simp only [quarticToyF, Pi.sub_apply, Pi.add_apply]
      ring
    · ext <;> simp <;> ring
  have hD2 : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4 + (4 * q.1 ^ 3 - q.1) / 480) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * p.1 + (12 * p.1 ^ 2 - 1) / 480) • ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) p := by
    intro p
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4 + (4 * q.1 ^ 3 - q.1) / 480)
        ((2 * p.1 + (12 * p.1 ^ 2 - 1) / 480) • ContinuousLinearMap.fst ℝ ℝ ℝ) p := by
      have := ((HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)).add
        (HasFDerivAt.const_mul ((HasFDerivAt.const_mul (HasFDerivAt.pow (hf p) 3) 4).sub (hf p))
          (1 / 480))
      convert this using 1 <;> first | (funext q; simp; ring) | (ext <;> simp; ring)
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2) ((-40 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
      HasFDerivAt.const_mul (hs p) (-40)
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hcd : ContDiff ℝ 4 quarticToyF := by unfold quarticToyF; fun_prop
  have hfd : fderiv ℝ quarticToyF = fun p => (p.1 ^ 2 - 1 / 4 + (4 * p.1 ^ 3 - p.1) / 480) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ :=
    funext fun p => (hDf p).fderiv
  have h2 : ∀ (y : ℝ × ℝ) (m : Fin 2 → ℝ × ℝ), iteratedFDeriv ℝ 2 quarticToyF y m =
      (2 * y.1 + (12 * y.1 ^ 2 - 1) / 480) * (m 0).1 * (m 1).1 - 40 * (m 0).2 * (m 1).2 := by
    intro y m
    rw [iteratedFDeriv_two_apply, hfd, (hD2 y).fderiv]
    simp
    ring
  have h3 : ∀ (x : ℝ × ℝ) (m : Fin 3 → ℝ × ℝ),
      iteratedFDeriv ℝ 3 quarticToyF x m = (2 + x.1 / 20) * (m 0).1 * (m 1).1 * (m 2).1 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 2 quarticToyF y (Fin.tail m))
        (((2 + x.1 / 20) * (m 1).1 * (m 2).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) x := by
      have hp : HasFDerivAt (fun y : ℝ × ℝ => 2 * y.1 + (12 * y.1 ^ 2 - 1) / 480)
          ((2 + x.1 / 20) • ContinuousLinearMap.fst ℝ ℝ ℝ) x := by
        have := (HasFDerivAt.const_mul (hf x) 2).add
          (HasFDerivAt.const_mul ((HasFDerivAt.const_mul (HasFDerivAt.pow (hf x) 2) 12).sub_const 1)
            (1 / 480))
        convert this using 1 <;> first | (funext q; simp; ring) | (ext <;> simp; ring)
      have := (hp.mul_const ((m 1).1 * (m 2).1)).sub_const (40 * (m 1).2 * (m 2).2)
      convert this using 1
      · funext y; rw [h2]; simp [Fin.tail]; ring
      · (ext <;> simp); ring
    rw [hg.fderiv]
    simp
    ring
  have h4 : ∀ (x : ℝ × ℝ) (m : Fin 4 → ℝ × ℝ),
      iteratedFDeriv ℝ 4 quarticToyF x m = (1 / 20) * (m 0).1 * (m 1).1 * (m 2).1 * (m 3).1 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 3 quarticToyF y (Fin.tail m))
        ((1 / 20 * (m 1).1 * (m 2).1 * (m 3).1) • ContinuousLinearMap.fst ℝ ℝ ℝ) x := by
      have := (((hf x).const_mul (1 / 20)).const_add 2).mul_const ((m 1).1 * (m 2).1 * (m 3).1)
      convert this using 1
      · funext y; rw [h3]; simp [Fin.tail]; ring
      · (ext <;> simp); ring
    rw [hg.fderiv]
    simp
    ring
  have hsrc := ridge_capHyp_E (E := ℝ) (f := quarticToyF) (U := univ) (r := 1) (m := 81 / 40)
    (n := 1 / 20) (lam := 40) one_pos isOpen_univ (subset_univ _) hcd.contDiffOn
    (fun p _ s₁ s₂ s₃ y₁ y₂ y₃ => by
      rw [h3, h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons]
      have h0 : (2 + p.1 / 20) * s₁ * s₂ * s₃ - s₁ * s₂ * s₃ * ((2 + p.1 / 20) * 1 * 1 * 1) = 0 := by
        ring
      rw [h0, abs_zero]
      exact mul_nonneg (by norm_num) (block_nonneg s₁ s₂ s₃ y₁ y₂ y₃))
    (fun x hx => by
      rw [h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons, mul_one]
      rw [abs_le]
      constructor <;> linarith [hx.1, hx.2])
    (fun p _ => by rw [h4]; simp)
    (fun p _ v => by rw [h4]; simp)
    (by norm_num)
    (fun v => by
      rw [h2]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by simp only [quarticToyF]; norm_num)
  obtain ⟨ε, hε, hex, hcap⟩ := hsrc
  have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => quarticToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    have := slice_hasFDerivAt (hDf (x, 0))
    convert this using 1
    ext
    simp
  obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
  exact ⟨⟨ε, hε, hex⟩, hC⟩

/-! ### v2.4: the cap on P §7's event `G_r`, for every pin gap `k > 0`

P §7 (`imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`) uses the deterministic
cap on `G_r = {λ_min(−A_M) > [4/(3k)] r M3², r M4 ≤ 3k/10}` with the pin gap `f(M) − f(S) = k r³`.
`ridge_capHyp_E` is the case `k = 1/6`. A positive factor `f ↦ c f` multiplies every derivative
bound and the transverse curvature by `c`, and changes neither the critical points nor the cap. -/

namespace CapHyp

variable {f : X → ℝ} {C : Set X} {M z : X} {b s : ℝ}

/-- A positive factor. A cap for `c f` at the levels `c b`, `c s` is a cap for `f` at `b`, `s`. -/
theorem of_const_mul {c : ℝ} (hc : 0 < c)
    (H : CapHyp (fun x => c * f x) C M z (c * b) (c * s)) : CapHyp f C M z b s where
  mem := H.mem
  ceiling x hx := le_of_mul_le_mul_left (H.ceiling x hx) hc
  frontier_le x hx := le_of_mul_le_mul_left (H.frontier_le x hx) hc
  older := lt_of_mul_lt_mul_left H.older hc.le
  ridge := by
    obtain ⟨γ, hγ⟩ := H.ridge
    exact ⟨γ, fun t => le_of_mul_le_mul_left (hγ t) hc⟩

end CapHyp

section GapK

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A nonzero factor does not move the points where the derivative vanishes. -/
theorem hasFDerivAt_zero_const_mul_iff {φ : E → ℝ} {c : ℝ} (hc : c ≠ 0) {y : E} :
    HasFDerivAt (fun y => c * φ y) (0 : E →L[ℝ] ℝ) y ↔ HasFDerivAt φ (0 : E →L[ℝ] ℝ) y := by
  constructor
  · intro h
    have h' := h.const_mul c⁻¹
    rw [smul_zero] at h'
    exact h'.congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun y => (inv_mul_cancel_left₀ hc (φ y)).symm)
  · intro h
    have h' := h.const_mul c
    rwa [smul_zero] at h'

/-- On an open set where `f ∈ C⁴`, the derivatives of order `≤ 4` of `c f` are `c` times those of
`f`. -/
theorem iteratedFDeriv_const_mul_of_isOpen {f : ℝ × E → ℝ} {U : Set (ℝ × E)} (hU : IsOpen U)
    (hf : ContDiffOn ℝ 4 f U) (c : ℝ) {i : ℕ} (hi : i ≤ 4) {p : ℝ × E} (hp : p ∈ U) :
    iteratedFDeriv ℝ i (fun q => c * f q) p = c • iteratedFDeriv ℝ i f p := by
  have hfa : ContDiffAt ℝ i f p :=
    (hf.contDiffAt (hU.mem_nhds hp)).of_le (by exact_mod_cast hi)
  exact iteratedFDeriv_const_smul_apply' (a := c) hfa

/-- **The cap for every gap `k > 0`** (P §7's event `G_r`). With the pin gap
`f(M) − f(S) = k r³`, the hypotheses `(4/(3k)) r m² < λ` and `r n ≤ 3k/10` give the conclusion of
`ridge_capHyp_E`. That theorem is the case `k = 1/6`; the proof applies it to `f / (6k)`. -/
theorem ridge_capHyp_K [FiniteDimensional ℝ E] {f : ℝ × E → ℝ} {U : Set (ℝ × E)}
    {r k m n lam : ℝ} (hr : 0 < r) (hk : 0 < k) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 f U)
    (hT : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ∀ (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E),
      |iteratedFDeriv ℝ 3 f p ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] -
          s₁ * s₂ * s₃ * iteratedFDeriv ℝ 3 f p ![(1, 0), (1, 0), (1, 0)]| ≤
        m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|))
    (hTe : ∀ x ∈ Icc (-r / 2) (r / 2), |iteratedFDeriv ℝ 3 f (x, 0) ![(1, 0), (1, 0), (1, 0)]| ≤ m)
    (hn4x : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      |iteratedFDeriv ℝ 4 f p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ n)
    (hn4y : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      |iteratedFDeriv ℝ 4 f p ![(0, v), (1, 0), (1, 0), (1, 0)]| ≤ n * ‖v‖)
    (hH2 : r * n ≤ 3 * k / 10)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 f (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 4 / (3 * k) * r * m ^ 2 < lam)
    (hM : fderiv ℝ f (-r / 2, 0) = 0) (hS : fderiv ℝ f (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = k * r ^ 3) :
    ∃ ε, 0 < ε ∧
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  obtain ⟨c, hc_def⟩ : ∃ c : ℝ, c = 1 / (6 * k) := ⟨_, rfl⟩
  have hk0 : k ≠ 0 := hk.ne'
  have hc : 0 < c := by rw [hc_def]; positivity
  have hD : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), p ∈ U :=
    fun p hp => hDU hp
  have hseg : ∀ x ∈ Icc (-r / 2) (r / 2), ((x, 0) : ℝ × E) ∈ U := by
    intro x hx
    apply hDU
    refine ⟨⟨by linarith [hx.1], by linarith [hx.2]⟩, ?_⟩
    simp only [Metric.mem_closedBall, dist_self]
    linarith
  have hsc : ∀ {i : ℕ}, i ≤ 4 → ∀ p ∈ U,
      iteratedFDeriv ℝ i (fun q => c * f q) p = c • iteratedFDeriv ℝ i f p :=
    fun hi p hp => iteratedFDeriv_const_mul_of_isOpen hU hf c hi hp
  have hdiff : ∀ p ∈ U, DifferentiableAt ℝ f p := fun p hp =>
    (hf.contDiffAt (hU.mem_nhds hp)).differentiableAt (by norm_num)
  have hMU : ((-r / 2, 0) : ℝ × E) ∈ U := hseg _ ⟨le_refl _, by linarith⟩
  have hSU : ((r / 2, 0) : ℝ × E) ∈ U := hseg _ ⟨by linarith, le_refl _⟩
  have key := ridge_capHyp_E (E := E) (f := fun q => c * f q) (U := U) (m := c * m)
    (n := c * n) (lam := c * lam) hr hU hDU (contDiffOn_const.mul hf)
    (fun p hp s₁ s₂ s₃ y₁ y₂ y₃ => by
      rw [hsc (by norm_num) p (hD p hp)]
      simp only [smul_apply, smul_eq_mul]
      have h := hT p hp s₁ s₂ s₃ y₁ y₂ y₃
      generalize iteratedFDeriv ℝ 3 f p ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] = A at h ⊢
      generalize iteratedFDeriv ℝ 3 f p ![(1, 0), (1, 0), (1, 0)] = B at h ⊢
      rw [show c * A - s₁ * s₂ * s₃ * (c * B) = c * (A - s₁ * s₂ * s₃ * B) by ring, abs_mul,
        abs_of_pos hc, mul_assoc c m]
      exact mul_le_mul_of_nonneg_left h hc.le)
    (fun x hx => by
      rw [hsc (by norm_num) _ (hseg x hx)]
      simp only [smul_apply, smul_eq_mul]
      rw [abs_mul, abs_of_pos hc]
      exact mul_le_mul_of_nonneg_left (hTe x hx) hc.le)
    (fun p hp => by
      rw [hsc (by norm_num) p (hD p hp)]
      simp only [smul_apply, smul_eq_mul]
      rw [abs_mul, abs_of_pos hc]
      exact mul_le_mul_of_nonneg_left (hn4x p hp) hc.le)
    (fun p hp v => by
      rw [hsc (by norm_num) p (hD p hp)]
      simp only [smul_apply, smul_eq_mul]
      rw [abs_mul, abs_of_pos hc, mul_assoc]
      exact mul_le_mul_of_nonneg_left (hn4y p hp v) hc.le)
    (by
      rw [hc_def, show r * (1 / (6 * k) * n) = r * n / (6 * k) by ring,
        div_le_iff₀ (by positivity)]
      linarith)
    (fun v => by
      rw [hsc (by norm_num) _ hMU]
      simp only [smul_apply, smul_eq_mul]
      calc c * iteratedFDeriv ℝ 2 f (-r / 2, 0) ![(0, v), (0, v)]
          ≤ c * (-lam * ‖v‖ ^ 2) := mul_le_mul_of_nonneg_left (hlam v) hc.le
        _ = -(c * lam) * ‖v‖ ^ 2 := by ring)
    (by
      have h8 : 8 * r * (c * m) ^ 2 = c * (4 / (3 * k) * r * m ^ 2) := by
        rw [hc_def]; field_simp; ring
      rw [h8]
      exact mul_lt_mul_of_pos_left hH1 hc)
    (by rw [fderiv_const_mul (hdiff _ hMU) c, hM, smul_zero])
    (by rw [fderiv_const_mul (hdiff _ hSU) c, hS, smul_zero])
    (by
      show c * f (-r / 2, 0) - c * f (r / 2, 0) = r ^ 3 / 6
      rw [← mul_sub, hnorm, hc_def]
      field_simp)
  obtain ⟨ε, hε, hex, hcap⟩ := key
  refine ⟨ε, hε, ?_, ?_⟩
  · obtain ⟨h, hh⟩ := hex
    exact ⟨h, fun x hx => ⟨(hh x hx).1, (hasFDerivAt_zero_const_mul_iff hc.ne').1 (hh x hx).2⟩⟩
  · intro h hh
    obtain ⟨h1, h2, hC, hfr⟩ := hcap h fun x hx =>
      ⟨(hh x hx).1, (hasFDerivAt_zero_const_mul_iff hc.ne').2 (hh x hx).2⟩
    exact ⟨h1, h2, hC.of_const_mul hc, fun x hx hne => lt_of_mul_lt_mul_left (hfr x hx hne) hc.le⟩

/-- **P §7's `G_r` with operator norms.** If `‖D³f‖ ≤ M3` and `‖D⁴f‖ ≤ M4` on
`D = [−2r, 2r] × B̄(0, 2r)`, then `(4/(3k)) r M3² < λ` and `r M4 ≤ 3k/10` give the cap. Each
partial-block bound used by `ridge_capHyp_K` is at most the operator norm (`block_of_norm`). -/
theorem ridge_capHyp_norm_K [FiniteDimensional ℝ E] {f : ℝ × E → ℝ} {U : Set (ℝ × E)}
    {r k M3 M4 lam : ℝ} (hr : 0 < r) (hk : 0 < k) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 f U)
    (h3 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖iteratedFDeriv ℝ 3 f p‖ ≤ M3)
    (h4 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      ‖iteratedFDeriv ℝ 4 f p‖ ≤ M4)
    (hH2 : r * M4 ≤ 3 * k / 10)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 f (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 4 / (3 * k) * r * M3 ^ 2 < lam)
    (hM : fderiv ℝ f (-r / 2, 0) = 0) (hS : fderiv ℝ f (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = k * r ^ 3) :
    ∃ ε, 0 < ε ∧
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  have he : ‖((1 : ℝ), (0 : E))‖ = 1 := by simp [Prod.norm_def]
  have hv : ∀ v : E, ‖((0 : ℝ), v)‖ = ‖v‖ := fun v => by simp [Prod.norm_def]
  refine ridge_capHyp_K hr hk hU hDU hf
    (fun p hp s₁ s₂ s₃ y₁ y₂ y₃ => block_of_norm (h3 p hp) s₁ s₂ s₃ y₁ y₂ y₃)
    (fun x hx => ?_) (fun p hp => ?_) (fun p hp v => ?_) hH2 hlam hH1 hM hS hnorm
  · have hp : ((x, 0) : ℝ × E) ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) := by
      refine ⟨⟨by linarith [hx.1], by linarith [hx.2]⟩, ?_⟩
      simp only [Metric.mem_closedBall, dist_self]
      linarith
    have h := (iteratedFDeriv ℝ 3 f (x, 0)).le_opNorm ![(1, 0), (1, 0), (1, 0)]
    simp only [Fin.prod_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_two, Matrix.tail_cons, Matrix.head_cons, he, mul_one] at h
    rw [← Real.norm_eq_abs]
    exact h.trans (h3 _ hp)
  · have h := (iteratedFDeriv ℝ 4 f p).le_opNorm ![(1, 0), (1, 0), (1, 0), (1, 0)]
    simp only [Fin.prod_univ_four, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_two, Matrix.cons_val_three, Matrix.tail_cons, Matrix.head_cons, he,
      mul_one] at h
    rw [← Real.norm_eq_abs]
    exact h.trans (h4 p hp)
  · have h := (iteratedFDeriv ℝ 4 f p).le_opNorm ![(0, v), (1, 0), (1, 0), (1, 0)]
    simp only [Fin.prod_univ_four, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val_two, Matrix.cons_val_three, Matrix.tail_cons, Matrix.head_cons, he, hv,
      mul_one] at h
    rw [← Real.norm_eq_abs]
    exact h.trans (mul_le_mul_of_nonneg_right (h4 p hp) (norm_nonneg v))

end GapK

/-! ### v2.5: CAP (1) in its own partial-block norms

CAP §1 (`imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md`) states the deterministic
theorem with `M_j = max_{a+c=j} sup_D ‖∂_x^a D_y^c f‖_op` (`c` transverse Euclidean unit vectors,
chosen independently) and `λ = λ_min(−D_y² f(M))`, under (1): `λ > (4/(3κ)) r M_3²`,
`r M_4 ≤ 3κ/10`. `ridge_capHyp_K` takes the third derivative as one block inequality `hT`.
`block_of_partial` derives `hT` from CAP's partial blocks with `c = 1, 2, 3`, using the symmetry of
`D³f`, which holds for `f ∈ C³` (`iteratedFDeriv_three_swap₁₂`, `iteratedFDeriv_three_swap₂₃`).
`ridge_capHyp_blocks` then states the cap under CAP's hypotheses as written. -/

section PartialBlock

open Filter Topology

variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- The third derivative of a `C³` function is symmetric in its first two slots: here
`D³f[a, b, c] = D²(Df)[a, b] c`, and `Df` has a symmetric second derivative. -/
theorem iteratedFDeriv_three_swap₁₂ {f : F → ℝ} {p : F} (hf : ContDiffAt ℝ 3 f p) (a b c : F) :
    iteratedFDeriv ℝ 3 f p ![a, b, c] = iteratedFDeriv ℝ 3 f p ![b, a, c] := by
  have h2 : ContDiffAt ℝ 2 (fderiv ℝ f) p := hf.fderiv_right (by norm_num)
  have hs : IsSymmSndFDerivAt ℝ (fderiv ℝ f) p := h2.isSymmSndFDerivAt (by simp)
  have hr : ∀ x y : F, iteratedFDeriv ℝ 3 f p ![x, y, c] =
      iteratedFDeriv ℝ 2 (fderiv ℝ f) p ![x, y] c := by
    intro x y
    rw [iteratedFDeriv_succ_apply_right]
    have e : Fin.init ![x, y, c] = ![x, y] := by ext i; fin_cases i <;> rfl
    rw [e]
    rfl
  rw [hr, hr, IsSymmSndFDerivAt.iteratedFDeriv_cons (hf := hs)]

/-- The third derivative of a `C³` function is symmetric in its last two slots: here
`D³f[a, b, c] = (D(D²f)[a])[b, c]`, and `D²f` is symmetric on a neighbourhood of `p`, so its
derivative at `p` is symmetric too. -/
theorem iteratedFDeriv_three_swap₂₃ {f : F → ℝ} {p : F} (hf : ContDiffAt ℝ 3 f p) (a b c : F) :
    iteratedFDeriv ℝ 3 f p ![a, b, c] = iteratedFDeriv ℝ 3 f p ![a, c, b] := by
  have hev : (fun q => iteratedFDeriv ℝ 2 f q ![b, c]) =ᶠ[𝓝 p]
      (fun q => iteratedFDeriv ℝ 2 f q ![c, b]) := by
    filter_upwards [hf.eventually (by simp)] with q hq
    have hq2 : ContDiffAt ℝ 2 f q := hq.of_le (by norm_num)
    exact IsSymmSndFDerivAt.iteratedFDeriv_cons (hf := hq2.isSymmSndFDerivAt (by simp))
  have hd : DifferentiableAt ℝ (iteratedFDeriv ℝ 2 f) p :=
    (hf.iteratedFDeriv_right (i := 2) (m := 1) (by norm_num)).differentiableAt (by norm_num)
  have key : ∀ v : Fin 2 → F, fderiv ℝ (fun q => iteratedFDeriv ℝ 2 f q v) p a =
      fderiv ℝ (iteratedFDeriv ℝ 2 f) p a v := by
    intro v
    have := ((ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => F) ℝ v).hasFDerivAt.comp p
      hd.hasFDerivAt).fderiv
    rw [show (fun q => iteratedFDeriv ℝ 2 f q v) =
      (ContinuousMultilinearMap.apply ℝ (fun _ : Fin 2 => F) ℝ v) ∘ iteratedFDeriv ℝ 2 f from rfl,
      this]
    rfl
  rw [iteratedFDeriv_succ_apply_left, iteratedFDeriv_succ_apply_left]
  have t1 : Fin.tail ![a, b, c] = ![b, c] := by ext i; fin_cases i <;> rfl
  have t2 : Fin.tail ![a, c, b] = ![c, b] := by ext i; fin_cases i <;> rfl
  simp only [Matrix.cons_val_zero, t1, t2]
  rw [← key, ← key, hev.fderiv_eq]

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **CAP's partial blocks give the block inequality.** For a symmetric trilinear form `B` on
`ℝ × E`, bounds by `m` on the three partial blocks with `c = 1, 2, 3` transverse slots, namely
`|B[(0, y), e, e]| ≤ m‖y‖`, `|B[(0, y₁), (0, y₂), e]| ≤ m‖y₁‖‖y₂‖` and
`|B[(0, y₁), (0, y₂), (0, y₃)]| ≤ m‖y₁‖‖y₂‖‖y₃‖` with `e = (1, 0)`, give the block inequality `hT` of
`ridge_capHyp_E` and `ridge_capHyp_K`. The proof expands `B[(s₁, y₁), (s₂, y₂), (s₃, y₃)]`
multilinearly: the seven mixed terms are partial blocks in some slot order, symmetry brings each to
the canonical order, and their bounds add up to `m(Π(|sᵢ| + ‖yᵢ‖) − Π|sᵢ|)`. No sign of `m` is
assumed. -/
theorem block_of_partial {B : ContinuousMultilinearMap ℝ (fun _ : Fin 3 => ℝ × E) ℝ} {m : ℝ}
    (hs₁₂ : ∀ a b c : ℝ × E, B ![a, b, c] = B ![b, a, c])
    (hs₂₃ : ∀ a b c : ℝ × E, B ![a, b, c] = B ![a, c, b])
    (k₁ : ∀ y : E, |B ![((0 : ℝ), y), (1, 0), (1, 0)]| ≤ m * ‖y‖)
    (k₂ : ∀ y₁ y₂ : E, |B ![((0 : ℝ), y₁), (0, y₂), (1, 0)]| ≤ m * ‖y₁‖ * ‖y₂‖)
    (k₃ : ∀ y₁ y₂ y₃ : E, |B ![((0 : ℝ), y₁), (0, y₂), (0, y₃)]| ≤ m * ‖y₁‖ * ‖y₂‖ * ‖y₃‖)
    (s₁ s₂ s₃ : ℝ) (y₁ y₂ y₃ : E) :
    |B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![(1, 0), (1, 0), (1, 0)]| ≤
      m * ((|s₁| + ‖y₁‖) * (|s₂| + ‖y₂‖) * (|s₃| + ‖y₃‖) - |s₁| * |s₂| * |s₃|) := by
  have hu0 : ∀ x b c z : ℝ × E, Function.update ![x, b, c] 0 z = ![z, b, c] := fun _ _ _ _ => by
    funext i; fin_cases i <;> simp
  have hu1 : ∀ a x c z : ℝ × E, Function.update ![a, x, c] 1 z = ![a, z, c] := fun _ _ _ _ => by
    funext i; fin_cases i <;> simp
  have hu2 : ∀ a b x z : ℝ × E, Function.update ![a, b, x] 2 z = ![a, b, z] := fun _ _ _ _ => by
    funext i; fin_cases i <;> simp
  have a₁ : ∀ x x' b c : ℝ × E, B ![x + x', b, c] = B ![x, b, c] + B ![x', b, c] := by
    intro x x' b c
    have := B.map_update_add ![x, b, c] 0 x x'
    rwa [hu0, hu0, hu0] at this
  have a₂ : ∀ a x x' c : ℝ × E, B ![a, x + x', c] = B ![a, x, c] + B ![a, x', c] := by
    intro a x x' c
    have := B.map_update_add ![a, x, c] 1 x x'
    rwa [hu1, hu1, hu1] at this
  have a₃ : ∀ a b x x' : ℝ × E, B ![a, b, x + x'] = B ![a, b, x] + B ![a, b, x'] := by
    intro a b x x'
    have := B.map_update_add ![a, b, x] 2 x x'
    rwa [hu2, hu2, hu2] at this
  have m₁ : ∀ (t : ℝ) (x b c : ℝ × E), B ![t • x, b, c] = t * B ![x, b, c] := by
    intro t x b c
    have := B.map_update_smul ![x, b, c] 0 t x
    rwa [hu0, hu0, smul_eq_mul] at this
  have m₂ : ∀ (t : ℝ) (a x c : ℝ × E), B ![a, t • x, c] = t * B ![a, x, c] := by
    intro t a x c
    have := B.map_update_smul ![a, x, c] 1 t x
    rwa [hu1, hu1, smul_eq_mul] at this
  have m₃ : ∀ (t : ℝ) (a b x : ℝ × E), B ![a, b, t • x] = t * B ![a, b, x] := by
    intro t a b x
    have := B.map_update_smul ![a, b, x] 2 t x
    rwa [hu2, hu2, smul_eq_mul] at this
  set e : ℝ × E := ((1 : ℝ), (0 : E)) with he_def
  have hsp : ∀ (s : ℝ) (y : E), ((s, y) : ℝ × E) = s • e + ((0 : ℝ), y) := by
    intro s y; ext <;> simp [he_def]
  set P₁ := |s₁| + ‖y₁‖
  set P₂ := |s₂| + ‖y₂‖
  set P₃ := |s₃| + ‖y₃‖
  -- the multilinear expansion
  have expand : B ![(s₁, y₁), (s₂, y₂), (s₃, y₃)] - s₁ * s₂ * s₃ * B ![e, e, e] =
      (s₂ * s₃ * B ![((0 : ℝ), y₁), e, e] + s₂ * B ![((0 : ℝ), y₁), e, (0, y₃)] +
        s₃ * B ![((0 : ℝ), y₁), (0, y₂), e] + B ![((0 : ℝ), y₁), (0, y₂), (0, y₃)]) +
      (s₁ * s₃ * B ![e, ((0 : ℝ), y₂), e] + s₁ * B ![e, ((0 : ℝ), y₂), (0, y₃)]) +
      s₁ * s₂ * B ![e, e, ((0 : ℝ), y₃)] := by
    rw [hsp s₁ y₁, hsp s₂ y₂, hsp s₃ y₃]
    simp only [a₁, a₂, a₃, m₁, m₂, m₃]
    ring
  rw [he_def] at expand ⊢
  rw [expand]
  -- the seven partial blocks, brought to the canonical order by symmetry
  have b₁ := k₁ y₁
  have b₂ : |B ![((0 : ℝ), y₁), (1, 0), (0, y₃)]| ≤ m * ‖y₁‖ * ‖y₃‖ := by
    rw [hs₂₃]; exact k₂ y₁ y₃
  have b₃ := k₂ y₁ y₂
  have b₄ := k₃ y₁ y₂ y₃
  have b₅ : |B ![(1, 0), ((0 : ℝ), y₂), (1, 0)]| ≤ m * ‖y₂‖ := by
    rw [hs₁₂]; exact k₁ y₂
  have b₆ : |B ![(1, 0), ((0 : ℝ), y₂), (0, y₃)]| ≤ m * ‖y₂‖ * ‖y₃‖ := by
    rw [hs₁₂, hs₂₃]; exact k₂ y₂ y₃
  have b₇ : |B ![(1, 0), (1, 0), ((0 : ℝ), y₃)]| ≤ m * ‖y₃‖ := by
    rw [hs₂₃, hs₁₂]; exact k₁ y₃
  have hP : m * (P₁ * P₂ * P₃ - |s₁| * |s₂| * |s₃|) =
      (|s₂| * |s₃| * (m * ‖y₁‖) + |s₂| * (m * ‖y₁‖ * ‖y₃‖) + |s₃| * (m * ‖y₁‖ * ‖y₂‖) +
        m * ‖y₁‖ * ‖y₂‖ * ‖y₃‖) +
      (|s₁| * |s₃| * (m * ‖y₂‖) + |s₁| * (m * ‖y₂‖ * ‖y₃‖)) + |s₁| * |s₂| * (m * ‖y₃‖) := by
    simp only [P₁, P₂, P₃]; ring
  rw [hP]
  refine (abs_add_le _ _).trans (add_le_add ((abs_add_le _ _).trans (add_le_add
    ((abs_add_le _ _).trans (add_le_add ((abs_add_le _ _).trans (add_le_add
      ((abs_add_le _ _).trans (add_le_add ?_ ?_)) ?_)) ?_)) ((abs_add_le _ _).trans
      (add_le_add ?_ ?_)))) ?_)
  · rw [abs_mul, abs_mul]; exact mul_le_mul_of_nonneg_left b₁ (by positivity)
  · rw [abs_mul]; exact mul_le_mul_of_nonneg_left b₂ (abs_nonneg _)
  · rw [abs_mul]; exact mul_le_mul_of_nonneg_left b₃ (abs_nonneg _)
  · exact b₄
  · rw [abs_mul, abs_mul]; exact mul_le_mul_of_nonneg_left b₅ (by positivity)
  · rw [abs_mul]; exact mul_le_mul_of_nonneg_left b₆ (abs_nonneg _)
  · rw [abs_mul, abs_mul]; exact mul_le_mul_of_nonneg_left b₇ (by positivity)

/-- **The cap under CAP (1) as written.** `f ∈ C⁴` on an open `U ⊇ D`; CAP's partial-block bounds
on `D` with `M3` for `c = 1, 2, 3` transverse slots, and for `c = 0` on the pin segment only (CAP
bounds it on all of `D`); the two partial blocks of the fourth derivative that the proof uses,
`c = 0` and `c = 1`, with `M4`; the transverse curvature `λ` at `M`; and (1),
`(4/(3k)) r M3² < λ` and `r M4 ≤ 3k/10`, for the pin gap `k r³`. The conclusion is that of
`ridge_capHyp_K`. -/
theorem ridge_capHyp_blocks [FiniteDimensional ℝ E] {f : ℝ × E → ℝ} {U : Set (ℝ × E)}
    {r k M3 M4 lam : ℝ} (hr : 0 < r) (hk : 0 < k) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 f U)
    (h30 : ∀ x ∈ Icc (-r / 2) (r / 2), |iteratedFDeriv ℝ 3 f (x, 0) ![(1, 0), (1, 0), (1, 0)]| ≤ M3)
    (h31 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y : E,
      |iteratedFDeriv ℝ 3 f p ![(0, y), (1, 0), (1, 0)]| ≤ M3 * ‖y‖)
    (h32 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ : E,
      |iteratedFDeriv ℝ 3 f p ![(0, y₁), (0, y₂), (1, 0)]| ≤ M3 * ‖y₁‖ * ‖y₂‖)
    (h33 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ y₃ : E,
      |iteratedFDeriv ℝ 3 f p ![(0, y₁), (0, y₂), (0, y₃)]| ≤ M3 * ‖y₁‖ * ‖y₂‖ * ‖y₃‖)
    (h40 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      |iteratedFDeriv ℝ 4 f p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ M4)
    (h41 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      |iteratedFDeriv ℝ 4 f p ![(0, v), (1, 0), (1, 0), (1, 0)]| ≤ M4 * ‖v‖)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 f (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 4 / (3 * k) * r * M3 ^ 2 < lam) (hH2 : r * M4 ≤ 3 * k / 10)
    (hM : fderiv ℝ f (-r / 2, 0) = 0) (hS : fderiv ℝ f (r / 2, 0) = 0)
    (hnorm : f (-r / 2, 0) - f (r / 2, 0) = k * r ^ 3) :
    ∃ ε, 0 < ε ∧
    (∃ h : ℝ → E, ∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε), h x ∈ Metric.ball (0 : E) (2 * r) ∧
        HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) ∧
      ∀ h : ℝ → E, (∀ x ∈ Ioo (-2 * r - ε) (2 * r + ε),
          h x ∈ Metric.closedBall (0 : E) (2 * r) ∧
            HasFDerivAt (fun y => f (x, y)) (0 : E →L[ℝ] ℝ) (h x)) →
        h (-r / 2) = 0 ∧ h (r / 2) = 0 ∧
          CapHyp f (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) (-r / 2, 0)
            (2 * r, h (2 * r)) (f (-r / 2, 0)) (f (r / 2, 0)) ∧
          ∀ x ∈ frontier (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)),
            x ≠ (r / 2, 0) → f x < f (r / 2, 0) := by
  refine ridge_capHyp_K hr hk hU hDU hf (fun p hp s₁ s₂ s₃ y₁ y₂ y₃ => ?_) h30 h40 h41 hH2 hlam
    hH1 hM hS hnorm
  have hp3 : ContDiffAt ℝ 3 f p := (hf.contDiffAt (hU.mem_nhds (hDU hp))).of_le (by norm_num)
  exact block_of_partial (iteratedFDeriv_three_swap₁₂ hp3) (iteratedFDeriv_three_swap₂₃ hp3)
    (h31 p hp) (h32 p hp) (h33 p hp) s₁ s₂ s₃ y₁ y₂ y₃

end PartialBlock

/-- A toy with a nonzero mixed block. `x³/3 − x/4 − 20y²` plus `xy²`: the pins and the gap `1/6` are
unchanged, and `∂_x D_y² f = 2`. -/
noncomputable def mixedToyF (p : ℝ × ℝ) : ℝ :=
  p.1 ^ 3 / 3 - p.1 / 4 - 20 * p.2 ^ 2 + p.1 * p.2 ^ 2

/-- **`ridge_capHyp_blocks` with a nonzero mixed block.** On `mixedToyF` the partial blocks of
`D³f` are `2` (`c = 0`), `0` (`c = 1`), `2` (`c = 2`, the mixed block `∂_x D_y² f`) and `0`
(`c = 3`), so `M3 = 2`; `D⁴f = 0`, so `M4 = 0`; and `λ = 41`. CAP (1) holds with `k = 1/6`, `r = 1`:
`8 M3² = 32 < 41` and `r M4 = 0 ≤ 1/20`. Unlike `h1Toy_E` and `quarticToy_E`, the block inequality
`hT` is not trivial here, since the mixed terms of `D³f` do not vanish. -/
theorem mixedToy_blocks :
    (∃ ε > 0, ∃ h : ℝ → ℝ, ∀ x ∈ Ioo (-2 * 1 - ε : ℝ) (2 * 1 + ε), h x ∈ Metric.ball (0 : ℝ) (2 * 1) ∧
        HasFDerivAt (fun y => mixedToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) (h x)) ∧
      CapHyp mixedToyF (Icc (-2 * 1) (1 / 2) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1)) (-1 / 2, 0)
        (2 * 1, 0) (mixedToyF (-1 / 2, 0)) (mixedToyF (1 / 2, 0)) := by
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hDf : ∀ p : ℝ × ℝ, HasFDerivAt mixedToyF
      ((p.1 ^ 2 - 1 / 4 + p.2 ^ 2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-40 * p.2 + 2 * p.1 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
    intro p
    have h1 := HasFDerivAt.add (HasFDerivAt.sub (HasFDerivAt.sub
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hf p) 3) (1 / 3))
      (HasFDerivAt.const_mul (hf p) (1 / 4)))
      (HasFDerivAt.const_mul (HasFDerivAt.pow (hs p) 2) 20))
      (HasFDerivAt.mul (hf p) (HasFDerivAt.pow (hs p) 2))
    convert h1 using 1
    · funext q
      simp only [mixedToyF, Pi.sub_apply, Pi.add_apply, Pi.mul_apply]
      ring
    · ext <;> simp
      ring
  have hD2 : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4 + q.2 ^ 2) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2 + 2 * q.1 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ + (2 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((2 * p.2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
          (-40 + 2 * p.1) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.snd ℝ ℝ ℝ)) p := by
    intro p
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4 + q.2 ^ 2)
        ((2 * p.1) • ContinuousLinearMap.fst ℝ ℝ ℝ + (2 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
      have := ((HasFDerivAt.pow (hf p) 2).sub_const (1 / 4)).add (HasFDerivAt.pow (hs p) 2)
      convert this using 1
      ext <;> simp
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2 + 2 * q.1 * q.2)
        ((2 * p.2) • ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 + 2 * p.1) • ContinuousLinearMap.snd ℝ ℝ ℝ)
        p := by
      have := (HasFDerivAt.const_mul (hs p) (-40)).add
        ((HasFDerivAt.const_mul (hf p) 2).mul (hs p))
      convert this using 1
      ext <;> simp
      ring
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hcd : ContDiff ℝ 4 mixedToyF := by unfold mixedToyF; fun_prop
  have hfd : fderiv ℝ mixedToyF = fun p => (p.1 ^ 2 - 1 / 4 + p.2 ^ 2) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * p.2 + 2 * p.1 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ :=
    funext fun p => (hDf p).fderiv
  have h2 : ∀ (y : ℝ × ℝ) (m : Fin 2 → ℝ × ℝ), iteratedFDeriv ℝ 2 mixedToyF y m =
      2 * y.1 * (m 0).1 * (m 1).1 + 2 * y.2 * ((m 0).1 * (m 1).2 + (m 0).2 * (m 1).1) +
        (-40 + 2 * y.1) * (m 0).2 * (m 1).2 := by
    intro y m
    rw [iteratedFDeriv_two_apply, hfd, (hD2 y).fderiv]
    simp
    ring
  have h3 : ∀ (x : ℝ × ℝ) (m : Fin 3 → ℝ × ℝ), iteratedFDeriv ℝ 3 mixedToyF x m =
      2 * (m 0).1 * (m 1).1 * (m 2).1 + 2 * (m 0).2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1) +
        2 * (m 0).1 * (m 1).2 * (m 2).2 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 2 mixedToyF y (Fin.tail m))
        ((2 * (m 1).1 * (m 2).1 + 2 * (m 1).2 * (m 2).2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
          (2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1)) • ContinuousLinearMap.snd ℝ ℝ ℝ) x := by
      have := (((2 * (m 1).1 * (m 2).1 + 2 * (m 1).2 * (m 2).2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
          (2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1)) •
            ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt (x := x)).add_const (-40 * (m 1).2 * (m 2).2)
      convert this using 1
      funext y; rw [h2]; simp [Fin.tail]; ring
    rw [hg.fderiv]
    simp
    ring
  have h4 : ∀ (x : ℝ × ℝ) (m : Fin 4 → ℝ × ℝ), iteratedFDeriv ℝ 4 mixedToyF x m = 0 := by
    intro x m
    rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
      (hcd.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
    have hc : (fun y => iteratedFDeriv ℝ 3 mixedToyF y (Fin.tail m)) =
        fun _ => iteratedFDeriv ℝ 3 mixedToyF x (Fin.tail m) := by
      funext y; rw [h3, h3]
    have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 3 mixedToyF y (Fin.tail m))
        (0 : ℝ × ℝ →L[ℝ] ℝ) x := by
      rw [hc]; exact hasFDerivAt_const _ _
    rw [hg.fderiv]
    simp
  have hsrc := ridge_capHyp_blocks (E := ℝ) (f := mixedToyF) (U := univ) (r := 1) (k := 1 / 6)
    (M3 := 2) (M4 := 0) (lam := 41) one_pos (by norm_num) isOpen_univ (subset_univ _)
    hcd.contDiffOn
    (fun x _ => by rw [h3]; simp)
    (fun p _ y => by rw [h3]; simp)
    (fun p _ y₁ y₂ => by
      rw [h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons, Real.norm_eq_abs]
      rw [show (2 * (0 : ℝ) * 0 * 1 + 2 * y₁ * (0 * 0 + y₂ * 1) + 2 * 0 * y₂ * 0) = 2 * (y₁ * y₂) by
        ring, abs_mul, abs_mul]
      norm_num [mul_assoc])
    (fun p _ y₁ y₂ y₃ => by
      rw [h3]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
        Matrix.head_cons, Real.norm_eq_abs]
      rw [show (2 * (0 : ℝ) * 0 * 0 + 2 * y₁ * (0 * y₃ + y₂ * 0) + 2 * 0 * y₂ * y₃) = 0 by ring,
        abs_zero]
      positivity)
    (fun p _ => by rw [h4]; simp)
    (fun p _ v => by rw [h4]; simp)
    (fun v => by
      rw [h2]
      simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Real.norm_eq_abs, sq_abs]
      nlinarith [sq_nonneg v])
    (by norm_num) (by norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by rw [(hDf _).fderiv]; ext <;> norm_num)
    (by simp only [mixedToyF]; norm_num)
  obtain ⟨ε, hε, hex, hcap⟩ := hsrc
  have hcrit0 : ∀ x : ℝ, HasFDerivAt (fun y => mixedToyF (x, y)) (0 : ℝ →L[ℝ] ℝ) 0 := by
    intro x
    have := slice_hasFDerivAt (hDf (x, 0))
    convert this using 1
    ext
    simp
  obtain ⟨-, -, hC, -⟩ := hcap (fun _ => 0) (fun x _ => ⟨by simp, hcrit0 x⟩)
  exact ⟨⟨ε, hε, hex⟩, hC⟩

/-! ### v2.6: the cap and the elder death level on the torus, under CAP (1)

P §7 applies the cap theorem to the periodic lift of the field on `(ℝ/Lℤ)^d`, in a contact frame
whose first axis joins the pins. `torus_capHyp_blocks` composes `ridge_capHyp_blocks` (v2.5) with
`CapHyp.toTorus` and `CapHyp.map_saddle` for any frame `φ : ℝ × E ≃ₜ ℝ^d`. Then
`torus_elder_death_level_blocks` adds `elder_death_level_peak` on the torus, which is compact
(`L > 0`) and locally connected (`torus_locallyConnectedSpace`). `torusToy_elder` instantiates it
on `(ℝ/16ℤ)²`. -/

/-- The circle `ℝ/Lℤ` is locally connected: its topology is coinduced by the covering map from
`ℝ`, and local connectedness passes to coinduced topologies. -/
theorem addCircle_locallyConnectedSpace (L : ℝ) : LocallyConnectedSpace (AddCircle L) :=
  (QuotientAddGroup.isQuotientMap_mk (AddSubgroup.zmultiples L)).isCoinducing.locallyConnectedSpace

/-- The torus `(ℝ/Lℤ)^d` is locally connected: a finite product of locally connected spaces. -/
theorem torus_locallyConnectedSpace (L : ℝ) (d : ℕ) :
    LocallyConnectedSpace (Fin d → AddCircle L) :=
  haveI := addCircle_locallyConnectedSpace L
  Pi.locallyConnectedSpace_of_finite_not_preconnectedSpace (Set.toFinite _)

section Torus

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **The cap on the torus under CAP (1).** Let `φ : ℝ × E ≃ₜ ℝ^d` be any frame and let
`g = f ∘ torusCover L d ∘ φ` be the lift of a function `f` on the torus `(ℝ/Lℤ)^d`. If `g` meets
the hypotheses of `ridge_capHyp_blocks` (CAP §1's (1) with its partial-block norms, in the frame
`φ`), then `f` satisfies (C1)–(C3) on the image of the cap in the torus, with `M` and `S` the images
of the pins, and (C2⁺): every other frontier point of the image is strictly below `f(S)`. -/
theorem torus_capHyp_blocks [FiniteDimensional ℝ E] {L : ℝ} {d : ℕ}
    (φ : ℝ × E ≃ₜ (Fin d → ℝ)) {f : (Fin d → AddCircle L) → ℝ} {g : ℝ × E → ℝ}
    (hg : ∀ p, g p = f (torusCover L d (φ p))) {U : Set (ℝ × E)}
    {r k M3 M4 lam : ℝ} (hr : 0 < r) (hk : 0 < k) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 g U)
    (h30 : ∀ x ∈ Icc (-r / 2) (r / 2), |iteratedFDeriv ℝ 3 g (x, 0) ![(1, 0), (1, 0), (1, 0)]| ≤ M3)
    (h31 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y), (1, 0), (1, 0)]| ≤ M3 * ‖y‖)
    (h32 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y₁), (0, y₂), (1, 0)]| ≤ M3 * ‖y₁‖ * ‖y₂‖)
    (h33 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ y₃ : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y₁), (0, y₂), (0, y₃)]| ≤ M3 * ‖y₁‖ * ‖y₂‖ * ‖y₃‖)
    (h40 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      |iteratedFDeriv ℝ 4 g p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ M4)
    (h41 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      |iteratedFDeriv ℝ 4 g p ![(0, v), (1, 0), (1, 0), (1, 0)]| ≤ M4 * ‖v‖)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 g (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 4 / (3 * k) * r * M3 ^ 2 < lam) (hH2 : r * M4 ≤ 3 * k / 10)
    (hM : fderiv ℝ g (-r / 2, 0) = 0) (hS : fderiv ℝ g (r / 2, 0) = 0)
    (hnorm : g (-r / 2, 0) - g (r / 2, 0) = k * r ^ 3) :
    ∃ z, CapHyp f ((torusCover L d ∘ φ) '' (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)))
        (torusCover L d (φ (-r / 2, 0))) z (f (torusCover L d (φ (-r / 2, 0))))
        (f (torusCover L d (φ (r / 2, 0)))) ∧
      ∀ x ∈ frontier ((torusCover L d ∘ φ) ''
          (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r))),
        x ≠ torusCover L d (φ (r / 2, 0)) → f x < f (torusCover L d (φ (r / 2, 0))) := by
  obtain ⟨ε, -, ⟨h0, hh0⟩, hall⟩ := ridge_capHyp_blocks hr hk hU hDU hf h30 h31 h32 h33 h40 h41
    hlam hH1 hH2 hM hS hnorm
  obtain ⟨-, -, hC, hSf⟩ := hall h0 fun x hx => ⟨Metric.ball_subset_closedBall (hh0 x hx).1,
    (hh0 x hx).2⟩
  have hgf : g = f ∘ (torusCover L d ∘ φ) := funext hg
  subst hgf
  have hK : IsCompact (Icc (-2 * r) (r / 2) ×ˢ Metric.closedBall (0 : E) (2 * r)) :=
    isCompact_Icc.prod (isCompact_closedBall 0 (2 * r))
  exact ⟨_, hC.toTorus φ hK, CapHyp.map_saddle ((torusCover_continuous L d).comp φ.continuous)
    ((torusCover_isOpenMap L d).comp φ.isOpenMap) hK hSf⟩

/-- **The elder death level on the torus under CAP (1).** For `L > 0` and a continuous `f` on
`(ℝ/Lℤ)^d` whose lift in the frame `φ` meets CAP §1's (1), the largest level at which the
component of the image of `M` contains a strictly higher local maximum is `f` at the image of `S`.
This is `elder_death_level_peak` on the torus, which is compact and locally connected. -/
theorem torus_elder_death_level_blocks [FiniteDimensional ℝ E] {L : ℝ} [Fact (0 < L)] {d : ℕ}
    (φ : ℝ × E ≃ₜ (Fin d → ℝ)) {f : (Fin d → AddCircle L) → ℝ} (hfc : Continuous f)
    {g : ℝ × E → ℝ} (hg : ∀ p, g p = f (torusCover L d (φ p))) {U : Set (ℝ × E)}
    {r k M3 M4 lam : ℝ} (hr : 0 < r) (hk : 0 < k) (hU : IsOpen U)
    (hDU : Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r) ⊆ U)
    (hf : ContDiffOn ℝ 4 g U)
    (h30 : ∀ x ∈ Icc (-r / 2) (r / 2), |iteratedFDeriv ℝ 3 g (x, 0) ![(1, 0), (1, 0), (1, 0)]| ≤ M3)
    (h31 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y), (1, 0), (1, 0)]| ≤ M3 * ‖y‖)
    (h32 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y₁), (0, y₂), (1, 0)]| ≤ M3 * ‖y₁‖ * ‖y₂‖)
    (h33 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ y₁ y₂ y₃ : E,
      |iteratedFDeriv ℝ 3 g p ![(0, y₁), (0, y₂), (0, y₃)]| ≤ M3 * ‖y₁‖ * ‖y₂‖ * ‖y₃‖)
    (h40 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r),
      |iteratedFDeriv ℝ 4 g p ![(1, 0), (1, 0), (1, 0), (1, 0)]| ≤ M4)
    (h41 : ∀ p ∈ Icc (-2 * r) (2 * r) ×ˢ Metric.closedBall (0 : E) (2 * r), ∀ v : E,
      |iteratedFDeriv ℝ 4 g p ![(0, v), (1, 0), (1, 0), (1, 0)]| ≤ M4 * ‖v‖)
    (hlam : ∀ v : E, iteratedFDeriv ℝ 2 g (-r / 2, 0) ![(0, v), (0, v)] ≤ -lam * ‖v‖ ^ 2)
    (hH1 : 4 / (3 * k) * r * M3 ^ 2 < lam) (hH2 : r * M4 ≤ 3 * k / 10)
    (hM : fderiv ℝ g (-r / 2, 0) = 0) (hS : fderiv ℝ g (r / 2, 0) = 0)
    (hnorm : g (-r / 2, 0) - g (r / 2, 0) = k * r ^ 3) :
    IsGreatest (CapHyp.olderPeakLevels f (torusCover L d (φ (-r / 2, 0))))
      (f (torusCover L d (φ (r / 2, 0)))) := by
  obtain ⟨z, H, -⟩ := torus_capHyp_blocks φ hg hr hk hU hDU hf h30 h31 h32 h33 h40 h41 hlam hH1
    hH2 hM hS hnorm
  have := torus_locallyConnectedSpace L d
  exact H.elder_death_level_peak hfc rfl

end Torus

/-- A continuous chart of the circle near `0`: `circleChart L θ = ‖θ + L/4‖ − L/4`, which equals
`x` at the image of every `x` with `|x| ≤ L/4` (`circleChart_coe`). -/
noncomputable def circleChart (L : ℝ) (θ : AddCircle L) : ℝ :=
  ‖θ + ((L / 4 : ℝ) : AddCircle L)‖ - L / 4

/-- `circleChart` is continuous. -/
theorem circleChart_continuous (L : ℝ) : Continuous (circleChart L) :=
  (continuous_norm.comp (continuous_id.add continuous_const)).sub continuous_const

/-- `circleChart L` inverts the covering map on `[−L/4, L/4]`. -/
theorem circleChart_coe {L : ℝ} (hL : 0 < L) {x : ℝ} (hx : |x| ≤ L / 4) :
    circleChart L (x : AddCircle L) = x := by
  have h1 := (abs_le.mp hx).1
  have h2 := (abs_le.mp hx).2
  have hn : ‖((x + L / 4 : ℝ) : AddCircle L)‖ = |x + L / 4| :=
    (AddCircle.norm_coe_eq_abs_iff L hL.ne').mpr (by
      rw [abs_of_nonneg (by linarith), abs_of_pos hL]; linarith)
  unfold circleChart
  rw [← AddCircle.coe_add, hn, abs_of_nonneg (by linarith)]
  ring

/-- The derivative of `mixedToyF` (v2.5's toy), stated as a lemma for reuse. -/
theorem mixedToy_hasFDerivAt (p : ℝ × ℝ) : HasFDerivAt mixedToyF
    ((p.1 ^ 2 - 1 / 4 + p.2 ^ 2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      (-40 * p.2 + 2 * p.1 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) p := by
  have hf : HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p := hasFDerivAt_fst
  have hs : HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p := hasFDerivAt_snd
  have h1 := HasFDerivAt.add (HasFDerivAt.sub (HasFDerivAt.sub
    (HasFDerivAt.const_mul (HasFDerivAt.pow hf 3) (1 / 3))
    (HasFDerivAt.const_mul hf (1 / 4)))
    (HasFDerivAt.const_mul (HasFDerivAt.pow hs 2) 20))
    (HasFDerivAt.mul hf (HasFDerivAt.pow hs 2))
  convert h1 using 1
  · funext q
    simp only [mixedToyF, Pi.sub_apply, Pi.add_apply, Pi.mul_apply]
    ring
  · ext <;> simp
    ring

/-- `mixedToyF` is `C⁴`. -/
theorem mixedToy_contDiff : ContDiff ℝ 4 mixedToyF := by unfold mixedToyF; fun_prop

/-- The second derivative of `mixedToyF`. -/
theorem mixedToy_iteratedFDeriv_two (y : ℝ × ℝ) (m : Fin 2 → ℝ × ℝ) :
    iteratedFDeriv ℝ 2 mixedToyF y m =
      2 * y.1 * (m 0).1 * (m 1).1 + 2 * y.2 * ((m 0).1 * (m 1).2 + (m 0).2 * (m 1).1) +
        (-40 + 2 * y.1) * (m 0).2 * (m 1).2 := by
  have hf : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.1) (ContinuousLinearMap.fst ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_fst
  have hs : ∀ p : ℝ × ℝ, HasFDerivAt (fun q : ℝ × ℝ => q.2) (ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
    fun p => hasFDerivAt_snd
  have hD2 : HasFDerivAt (fun q : ℝ × ℝ => (q.1 ^ 2 - 1 / 4 + q.2 ^ 2) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * q.2 + 2 * q.1 * q.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (((2 * y.1) • ContinuousLinearMap.fst ℝ ℝ ℝ + (2 * y.2) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        ((2 * y.2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
          (-40 + 2 * y.1) • ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
          (ContinuousLinearMap.snd ℝ ℝ ℝ)) y := by
    have h1 : HasFDerivAt (fun q : ℝ × ℝ => q.1 ^ 2 - 1 / 4 + q.2 ^ 2)
        ((2 * y.1) • ContinuousLinearMap.fst ℝ ℝ ℝ + (2 * y.2) • ContinuousLinearMap.snd ℝ ℝ ℝ) y := by
      have := ((HasFDerivAt.pow (hf y) 2).sub_const (1 / 4)).add (HasFDerivAt.pow (hs y) 2)
      convert this using 1
      ext <;> simp
    have h2 : HasFDerivAt (fun q : ℝ × ℝ => -40 * q.2 + 2 * q.1 * q.2)
        ((2 * y.2) • ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 + 2 * y.1) • ContinuousLinearMap.snd ℝ ℝ ℝ)
        y := by
      have := (HasFDerivAt.const_mul (hs y) (-40)).add
        ((HasFDerivAt.const_mul (hf y) 2).mul (hs y))
      convert this using 1
      ext <;> simp
      ring
    exact (h1.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
      (h2.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))
  have hfd : fderiv ℝ mixedToyF = fun p => (p.1 ^ 2 - 1 / 4 + p.2 ^ 2) •
      ContinuousLinearMap.fst ℝ ℝ ℝ + (-40 * p.2 + 2 * p.1 * p.2) • ContinuousLinearMap.snd ℝ ℝ ℝ :=
    funext fun p => (mixedToy_hasFDerivAt p).fderiv
  rw [iteratedFDeriv_two_apply, hfd, hD2.fderiv]
  simp
  ring

/-- The third derivative of `mixedToyF`: `2a₁b₁c₁ + 2a₂(b₁c₂ + b₂c₁) + 2a₁b₂c₂`. -/
theorem mixedToy_iteratedFDeriv_three (x : ℝ × ℝ) (m : Fin 3 → ℝ × ℝ) :
    iteratedFDeriv ℝ 3 mixedToyF x m =
      2 * (m 0).1 * (m 1).1 * (m 2).1 + 2 * (m 0).2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1) +
        2 * (m 0).1 * (m 1).2 * (m 2).2 := by
  rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
    (mixedToy_contDiff.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
  have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 2 mixedToyF y (Fin.tail m))
      ((2 * (m 1).1 * (m 2).1 + 2 * (m 1).2 * (m 2).2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1)) • ContinuousLinearMap.snd ℝ ℝ ℝ) x := by
    have := (((2 * (m 1).1 * (m 2).1 + 2 * (m 1).2 * (m 2).2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (2 * ((m 1).1 * (m 2).2 + (m 1).2 * (m 2).1)) •
          ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt (x := x)).add_const (-40 * (m 1).2 * (m 2).2)
    convert this using 1
    funext y; rw [mixedToy_iteratedFDeriv_two]; simp [Fin.tail]; ring
  rw [hg.fderiv]
  simp
  ring

/-- The fourth derivative of `mixedToyF` vanishes. -/
theorem mixedToy_iteratedFDeriv_four (x : ℝ × ℝ) (m : Fin 4 → ℝ × ℝ) :
    iteratedFDeriv ℝ 4 mixedToyF x m = 0 := by
  rw [DifferentiableAt.iteratedFDeriv_succ_apply_left'
    (mixedToy_contDiff.contDiffAt.differentiableAt_iteratedFDeriv (by norm_num))]
  have hc : (fun y => iteratedFDeriv ℝ 3 mixedToyF y (Fin.tail m)) =
      fun _ => iteratedFDeriv ℝ 3 mixedToyF x (Fin.tail m) := by
    funext y; rw [mixedToy_iteratedFDeriv_three, mixedToy_iteratedFDeriv_three]
  have hg : HasFDerivAt (fun y => iteratedFDeriv ℝ 3 mixedToyF y (Fin.tail m))
      (0 : ℝ × ℝ →L[ℝ] ℝ) x := by
    rw [hc]; exact hasFDerivAt_const _ _
  rw [hg.fderiv]
  simp

/-- A function on the torus `(ℝ/16ℤ)²` whose lift equals `mixedToyF` on `(−4, 4)²`:
`mixedToyF` read through `circleChart` in each coordinate. It is continuous. -/
noncomputable def torusToyF (θ : Fin 2 → AddCircle (16 : ℝ)) : ℝ :=
  mixedToyF (circleChart 16 (θ 0), circleChart 16 (θ 1))

/-- **A concrete instance of the elder death level on the torus.** On `(ℝ/16ℤ)²`, with the standard
frame `ℝ × ℝ ≃ₜ ℝ²`, the lift of `torusToyF` agrees with `mixedToyF` near the cap, so it meets
CAP (1) with `r = 1`, `k = 1/6`, `M3 = 2`, `M4 = 0` and `λ = 41`. `torus_elder_death_level_blocks`
then gives the death level of the class born at the image of `M = (−1/2, 0)`: it is the value at
the image of `S = (1/2, 0)`. This is the first concrete instance of the older-peak theorems
(`older_peak`, `elder_death_level_peak`), whose compactness hypothesis fails on the earlier toys'
domain `ℝ`. -/
theorem torusToy_elder :
    IsGreatest (CapHyp.olderPeakLevels torusToyF
        (torusCover 16 2 ((Homeomorph.finTwoArrow (X := ℝ)).symm (-1 / 2, 0))))
      (torusToyF (torusCover 16 2 ((Homeomorph.finTwoArrow (X := ℝ)).symm (1 / 2, 0)))) := by
  have : Fact ((0 : ℝ) < 16) := ⟨by norm_num⟩
  set φ : ℝ × ℝ ≃ₜ (Fin 2 → ℝ) := (Homeomorph.finTwoArrow (X := ℝ)).symm with hφ
  set U : Set (ℝ × ℝ) := Ioo (-4) 4 ×ˢ Ioo (-4) 4 with hU_def
  have hU : IsOpen U := isOpen_Ioo.prod isOpen_Ioo
  have hGU : ∀ p ∈ U, torusToyF (torusCover 16 2 (φ p)) = mixedToyF p := by
    rintro ⟨a, b⟩ ⟨ha, hb⟩
    have ha' : |a| ≤ 16 / 4 := by rw [abs_le]; constructor <;> linarith [ha.1, ha.2]
    have hb' : |b| ≤ 16 / 4 := by rw [abs_le]; constructor <;> linarith [hb.1, hb.2]
    simp only [torusToyF, torusCover, hφ, Homeomorph.finTwoArrow_symm_apply, Pi.map_apply]
    simp [circleChart_coe (by norm_num : (0 : ℝ) < 16) ha',
      circleChart_coe (by norm_num : (0 : ℝ) < 16) hb']
  have hev : ∀ p ∈ U, (fun q => torusToyF (torusCover 16 2 (φ q))) =ᶠ[nhds p] mixedToyF :=
    fun p hp => Filter.eventuallyEq_of_mem (hU.mem_nhds hp) hGU
  have hD : ∀ n p, p ∈ U → iteratedFDeriv ℝ n (fun q => torusToyF (torusCover 16 2 (φ q))) p =
      iteratedFDeriv ℝ n mixedToyF p :=
    fun n p hp => ((hev p hp).iteratedFDeriv ℝ n).eq_of_nhds
  have hDU : Icc (-2 * 1) (2 * 1) ×ˢ Metric.closedBall (0 : ℝ) (2 * 1) ⊆ U := by
    rintro ⟨a, b⟩ ⟨ha, hb⟩
    rw [Metric.mem_closedBall, Real.dist_eq, sub_zero, abs_le] at hb
    exact ⟨⟨by linarith [ha.1], by linarith [ha.2]⟩, ⟨by linarith [hb.1], by linarith [hb.2]⟩⟩
  have hseg : ∀ x ∈ Icc (-1 / 2 : ℝ) (1 / 2), ((x, 0) : ℝ × ℝ) ∈ U := fun x hx =>
    ⟨⟨by linarith [hx.1], by linarith [hx.2]⟩, ⟨by norm_num, by norm_num⟩⟩
  have hM : ((-1 / 2, 0) : ℝ × ℝ) ∈ U := hseg _ ⟨by norm_num, by norm_num⟩
  have hS : ((1 / 2, 0) : ℝ × ℝ) ∈ U := hseg _ ⟨by norm_num, by norm_num⟩
  refine torus_elder_death_level_blocks (E := ℝ) (L := 16) (d := 2) φ
    (mixedToy_contDiff.continuous.comp ((circleChart_continuous 16).comp (continuous_apply 0) |>.prodMk
      ((circleChart_continuous 16).comp (continuous_apply 1))))
    (g := fun q => torusToyF (torusCover 16 2 (φ q))) (fun p => rfl) (U := U)
    (r := 1) (k := 1 / 6) (M3 := 2) (M4 := 0) (lam := 41) one_pos (by norm_num) hU hDU
    (mixedToy_contDiff.contDiffOn.congr fun p hp => hGU p hp)
    (fun x hx => ?_) (fun p hp y => ?_) (fun p hp y₁ y₂ => ?_) (fun p hp y₁ y₂ y₃ => ?_)
    (fun p hp => ?_) (fun p hp v => ?_) (fun v => ?_) (by norm_num) (by norm_num) ?_ ?_ ?_
  · rw [hD 3 _ (hseg _ (by simpa using hx)), mixedToy_iteratedFDeriv_three]; simp
  · rw [hD 3 _ (hDU hp), mixedToy_iteratedFDeriv_three]; simp
  · rw [hD 3 _ (hDU hp), mixedToy_iteratedFDeriv_three]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
      Matrix.head_cons, Real.norm_eq_abs]
    rw [show (2 * (0 : ℝ) * 0 * 1 + 2 * y₁ * (0 * 0 + y₂ * 1) + 2 * 0 * y₂ * 0) = 2 * (y₁ * y₂) by
      ring, abs_mul, abs_mul]
    norm_num [mul_assoc]
  · rw [hD 3 _ (hDU hp), mixedToy_iteratedFDeriv_three]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.tail_cons,
      Matrix.head_cons, Real.norm_eq_abs]
    rw [show (2 * (0 : ℝ) * 0 * 0 + 2 * y₁ * (0 * y₃ + y₂ * 0) + 2 * 0 * y₂ * y₃) = 0 by ring,
      abs_zero]
    positivity
  · rw [hD 4 _ (hDU hp), mixedToy_iteratedFDeriv_four]; simp
  · rw [hD 4 _ (hDU hp), mixedToy_iteratedFDeriv_four]; simp
  · rw [hD 2 _ (by simpa using hM), mixedToy_iteratedFDeriv_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Real.norm_eq_abs, sq_abs]
    nlinarith [sq_nonneg v]
  · rw [(hev _ (by simpa using hM)).fderiv_eq, (mixedToy_hasFDerivAt _).fderiv]
    ext <;> norm_num
  · rw [(hev _ (by simpa using hS)).fderiv_eq, (mixedToy_hasFDerivAt _).fderiv]
    ext <;> norm_num
  · rw [hGU _ (by simpa using hM), hGU _ (by simpa using hS)]
    simp only [mixedToyF]; norm_num

end CapFirstExit
