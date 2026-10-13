/-
  CapFirstExit: the topological step from a separating cap to the global elder level.

  `CapHyp f C M z b s` bundles the conclusions that CAP §§2–5 derive from the local good event
  `G_r` (reviews/d1_cap_elder_partner_second_pass_claude_20260930/REVIEW.md, L16–L20):

  (C1) `M ∈ C` and `f ≤ b` on the cap `C`                                   (L16);
  (C2) `f ≤ s` on the whole frontier of `C`                                  (L17–L19);
  (C3) `b < f z`, and a path from `M` to `z` along which `f ≥ s`             (L17, L20).

  The saddle form adds (C2⁺): every frontier point other than `S` has `f < s`   (L19).

  Nothing is assumed about `f` outside `C` beyond the witness path in (C3). No metric, smoothness,
  Morse, compactness or probabilistic hypothesis is used, and the separation statements do not
  even use continuity of `f`. `X` is an arbitrary topological space; the torus is one instance.

  What this file does NOT formalize: the analytic inequalities L1–L20 that produce (C1)–(C3)
  from `G_r`, the Morse-theoretic identification of the merging critical point (L22, "distinct
  critical values"), the unstable-branch analysis (L23), and every probabilistic statement.
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
when `S` is the only frontier point at height `s` (C2⁺, L19). -/
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

/-- **`S` is a cut point at the death level.** Under (C2⁺), removing `S` from `{f ≥ s}` separates
`M` from every point strictly above `b`, while `{f ≥ s}` itself joins them
(`elder_merge_iff_closed` at `h = s`). -/
theorem saddle_cut (H : CapHyp f C M z b s) {S : X}
    (hS : ∀ x ∈ frontier C, x ≠ S → f x < s) {w : X}
    (hw : w ∈ connectedComponentIn ({x | s ≤ f x} \ {S}) M) : f w ≤ b := by
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

/-- The toy's elder death level is `-1`. -/
theorem toy_death_level : IsGreatest (CapHyp.joinedLevels toyF 0 0) (-1) :=
  toy_capHyp.elder_death_level

/-- Necessity of (C2). With (C1) and (C3) holding at the wrong level `s = -3/2` (the frontier
point `1` has height `-1 > -3/2`), the maximin level is not `-3/2`. -/
theorem toy_needs_C2 : ¬ IsGreatest (connectionLevels toyF 0 0) (-3 / 2) := by
  rintro ⟨-, hub⟩
  have hmem : (-1 : ℝ) ∈ connectionLevels toyF 0 0 :=
    ⟨3, toyRidge, toy_capHyp.older, toy_ridge_ge⟩
  have := hub hmem
  norm_num at this

/-- Necessity of (C1). At the wrong ceiling `b = -1/2` (so `f M = 0 > b` and (C1) fails), the
constant path at `M` already reaches above `b`, so the maximin level is not `-1`. -/
theorem toy_needs_C1 : ¬ IsGreatest (connectionLevels toyF 0 (-1 / 2)) (-1) := by
  rintro ⟨-, hub⟩
  have hM : toyF 0 = 0 := by
    show max (-|(0 : ℝ)|) ((0 : ℝ) - 2) = 0
    norm_num
  have hmem : (0 : ℝ) ∈ connectionLevels toyF 0 (-1 / 2) :=
    ⟨0, Path.refl 0, by rw [hM]; norm_num, fun t => by simp [hM]⟩
  have := hub hmem
  norm_num at this

/-- Necessity of (C3). At the wrong level `s = 5` (no ridge stays at height `≥ 5`, though (C1) and
(C2) hold), the maximin level is not `5`: the barrier at the true level `-1` excludes it. -/
theorem toy_needs_C3 : ¬ IsGreatest (connectionLevels toyF 0 0) 5 := by
  rintro ⟨hmem, -⟩
  have h1 := toy_capHyp.maximin_isGreatest.2 hmem
  norm_num at h1

end CapFirstExit
