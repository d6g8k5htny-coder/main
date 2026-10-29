import Mathlib

/-!
# C263 total strict-clearance cut for continuous Euclidean heights

Proves: total essentiality, strict-clearance set and real death value with an
explicit empty-set value zero; an unconditional bound for clearance members;
nonemptiness for continuous nonessential inputs; strict witness improvement;
the exact strict supremum cut, guarded death bound and cut uniqueness.

Method: the starting endpoint bounds every clearance; a straight path and the
attained minimum on its compact unit-interval parameter supply a lower witness
and a uniform strict margin; conditional real supremum laws identify the cut.

Does not establish: an inhabitant of the separately owned Field or Lifetime
record, an elder relation, Morse endpoint attainment, ordinary H0 bar pairing,
finite-bar counts, a Gaussian field or weighted law, or scientific acceptance.
The scalar zero on an empty set is a default, not an essential-bar death.

Authored by OpenAI/Codex c262_lean_source_review, now acting as C263 implementer,
delegated for Dylan Roy. Source/design-exposed; organizational independence 0;
no human review. Consumes C260 prose SHA-256
d2462639df32364e6fc8b0fea0703106c668cedb0aa7bb00e139aa81561bd845.
Target pin: Lean 4.34.1 / Mathlib d13f23b723b8a846827a245b89c10fc7d3f11612.
Elaboration, kernel audit and executed controls are recorded in separate
source-bound receipts; this text makes no execution claim.

Compile fix at the pin (Anthropic/Claude lane, companion owner; handoff bytes
SHA-256 d74dc8b2c25eff5b90ce7133e51a0888c7885cae24ca6d5e648bc514089637a8 did not
elaborate with `-DwarningAsError=true`): in `superlevelBar_death_eq_sSup_of_nonempty`
the simp lemma `if_pos` (deprecated in Lean core since 2026-07-21) is replaced by
its successor `ite_eq_left`, the same statement; no other byte changes.
-/

noncomputable section

namespace TorusFiltration

/-- Proves: the ambient type is real Euclidean space in the stated dimension.
Method: a type abbreviation. Does not establish: a positive-dimensional or
periodic quotient-space hypothesis. -/
abbrev superlevelBar_E (d : ℕ) := EuclideanSpace ℝ (Fin d)

/-- Proves: the essentiality datum is total. Method: global order comparison.
Does not establish: an actual persistence interpretation off the guarded locus. -/
def superlevelBar_isEssential {d : ℕ} (g : superlevelBar_E d → ℝ)
    (M : superlevelBar_E d) : Prop :=
  ∀ y, g y ≤ g M

/-- Proves: a total strict-clearance set with the source's existential endpoint.
Method: quantify over actual paths and every parameter, including endpoints.
Does not establish: a path attaining the supremum with closed inequalities. -/
def superlevelBar_clearanceSet {d : ℕ} (g : superlevelBar_E d → ℝ)
    (M : superlevelBar_E d) : Set ℝ :=
  {h | ∃ y, g M < g y ∧ ∃ γ : Path M y, ∀ t : unitInterval, h < g (γ t)}

/-- Proves: the death datum is a total real number with explicit empty default.
Method: nonempty conditional supremum; boundedness is proved below.
Does not establish: a finite death for an essential class. -/
def superlevelBar_deathHeight {d : ℕ} (g : superlevelBar_E d → ℝ)
    (M : superlevelBar_E d) : ℝ := by
  classical
  exact if (superlevelBar_clearanceSet g M).Nonempty then
    sSup (superlevelBar_clearanceSet g M) else 0

/-- Proves: essentiality is exactly global maximality for these total data.
Method: unfold the defining proposition. Does not establish: agreement with
an independently defined essential persistence class. -/
theorem superlevelBar_essential_iff {d : ℕ} (g : superlevelBar_E d → ℝ)
    (M : superlevelBar_E d) :
    superlevelBar_isEssential g M ↔ ∀ y, g y ≤ g M := Iff.rfl

/-- Proves: an unconditional strict bound on each clearance member.
Method: evaluate its path at zero. Does not establish: an unconditional bound
on the default death value when the clearance set is empty. -/
theorem superlevelBar_mem_lt_start {d : ℕ} (g : superlevelBar_E d → ℝ)
    (M : superlevelBar_E d) {h : ℝ} (hh : h ∈ superlevelBar_clearanceSet g M) :
    h < g M := by
  obtain ⟨y, hy, γ, hγ⟩ := hh
  have hzero := hγ (0 : unitInterval)
  simpa only [γ.source] using hzero

/-- Proves: the clearance set is bounded above, including when empty.
Method: the initial height bounds each member. Does not establish: that the
empty-set default death value obeys the same bound. -/
theorem superlevelBar_clearance_bddAbove {d : ℕ} (g : superlevelBar_E d → ℝ)
    (M : superlevelBar_E d) : BddAbove (superlevelBar_clearanceSet g M) := by
  exact ⟨g M, fun h hh => (superlevelBar_mem_lt_start g M hh).le⟩

/-- Proves: essentiality forces the clearance set to be empty.
Method: contradict the required strictly higher endpoint.
Does not establish: a converse for arbitrary discontinuous heights. -/
theorem superlevelBar_clearance_empty_of_essential {d : ℕ}
    (g : superlevelBar_E d → ℝ) (M : superlevelBar_E d)
    (hM : superlevelBar_isEssential g M) : superlevelBar_clearanceSet g M = ∅ := by
  ext h
  constructor
  · rintro ⟨y, hy, γ, hγ⟩
    exact False.elim ((not_lt_of_ge (hM y)) hy)
  · intro hh
    exact False.elim hh

/-- Proves: an empty clearance set selects the declared real value zero.
Method: simplify the conditional definition. Does not establish: a geometric
death event or finite persistence interval at that zero. -/
theorem superlevelBar_death_eq_zero_of_empty {d : ℕ}
    (g : superlevelBar_E d → ℝ) (M : superlevelBar_E d)
    (he : superlevelBar_clearanceSet g M = ∅) : superlevelBar_deathHeight g M = 0 := by
  classical
  simp [superlevelBar_deathHeight, he]

/-- Proves: essential inputs have the declared zero default.
Method: essentiality implies empty clearance, then use the empty branch.
Does not establish: an essential-bar death height. -/
theorem superlevelBar_death_eq_zero_of_essential {d : ℕ}
    (g : superlevelBar_E d → ℝ) (M : superlevelBar_E d)
    (hM : superlevelBar_isEssential g M) : superlevelBar_deathHeight g M = 0 := by
  exact superlevelBar_death_eq_zero_of_empty g M
    (superlevelBar_clearance_empty_of_essential g M hM)

/-- Proves: nonempty clearance selects its real supremum in the total data.
Method: simplify the nonempty conditional branch. Does not establish:
nonemptiness without a witness or a regularity hypothesis. -/
theorem superlevelBar_death_eq_sSup_of_nonempty {d : ℕ}
    (g : superlevelBar_E d → ℝ) (M : superlevelBar_E d)
    (hne : (superlevelBar_clearanceSet g M).Nonempty) :
    superlevelBar_deathHeight g M = sSup (superlevelBar_clearanceSet g M) := by
  classical
  simp only [superlevelBar_deathHeight, ite_eq_left hne]

/-- Proves: a continuous height attains a minimum along each actual path.
Method: compact unit interval and the extreme-value theorem.
Does not establish: a minimum over all paths or an optimal path. -/
theorem superlevelBar_path_minimum {d : ℕ} (g : superlevelBar_E d → ℝ)
    (hg : Continuous g) {M y : superlevelBar_E d} (γ : Path M y) :
    ∃ t₀ : unitInterval, ∀ t : unitInterval, g (γ t₀) ≤ g (γ t) := by
  obtain ⟨t₀, ht₀, hmin⟩ :=
    (isCompact_univ : IsCompact (Set.univ : Set unitInterval)).exists_isMinOn
      (show (Set.univ : Set unitInterval).Nonempty from ⟨0, Set.mem_univ _⟩)
      (hg.comp γ.continuous).continuousOn
  exact ⟨t₀, fun t => hmin (Set.mem_univ t)⟩

/-- Proves: nonessential continuous inputs have a nonempty clearance set.
Method: a higher endpoint, Path.segment, and minimum minus one.
Does not establish: compactness of the whole Euclidean domain. -/
theorem superlevelBar_clearance_nonempty {d : ℕ} (g : superlevelBar_E d → ℝ)
    (hg : Continuous g) (M : superlevelBar_E d)
    (hM : ¬ superlevelBar_isEssential g M) :
    (superlevelBar_clearanceSet g M).Nonempty := by
  classical
  have hhigh : ∃ y, g M < g y := by
    by_contra hn
    apply hM
    intro y
    exact le_of_not_gt (fun hy => hn ⟨y, hy⟩)
  obtain ⟨y, hy⟩ := hhigh
  let γ : Path M y := Path.segment M y
  obtain ⟨t₀, hmin⟩ := superlevelBar_path_minimum g hg γ
  refine ⟨g (γ t₀) - 1, y, hy, γ, ?_⟩
  intro t
  have ht := hmin t
  linarith

/-- Proves: for continuous heights, clearance is nonempty exactly off the
essential locus. Method: the higher witness endpoint and the straight-path
construction. Does not establish: this equivalence without continuity. -/
theorem superlevelBar_clearance_nonempty_iff {d : ℕ}
    (g : superlevelBar_E d → ℝ) (hg : Continuous g) (M : superlevelBar_E d) :
    (superlevelBar_clearanceSet g M).Nonempty ↔ ¬ superlevelBar_isEssential g M := by
  constructor
  · rintro ⟨h, y, hy, γ, hγ⟩ hM
    exact (not_lt_of_ge (hM y)) hy
  · exact superlevelBar_clearance_nonempty g hg M

/-- Proves: any smaller threshold than an existing clearance is a clearance.
Method: reuse the same endpoint and path and compose strict inequalities.
Does not establish: strict improvement to a larger threshold. -/
theorem superlevelBar_clearance_downward {d : ℕ} (g : superlevelBar_E d → ℝ)
    (M : superlevelBar_E d) {h c : ℝ} (hhc : h < c)
    (hc : c ∈ superlevelBar_clearanceSet g M) : h ∈ superlevelBar_clearanceSet g M := by
  obtain ⟨y, hy, γ, hγ⟩ := hc
  exact ⟨y, hy, γ, fun t => hhc.trans (hγ t)⟩

/-- Proves: each strict witness has a larger strict witness on its same path.
Method: attained minimum and a midpoint below it.
Does not establish: a witness with clearance strictly above the supremum. -/
theorem superlevelBar_clearance_improve {d : ℕ} (g : superlevelBar_E d → ℝ)
    (hg : Continuous g) (M : superlevelBar_E d) {h : ℝ}
    (hh : h ∈ superlevelBar_clearanceSet g M) :
    ∃ c : ℝ, h < c ∧ c ∈ superlevelBar_clearanceSet g M := by
  obtain ⟨y, hy, γ, hγ⟩ := hh
  obtain ⟨t₀, hmin⟩ := superlevelBar_path_minimum g hg γ
  have hmargin := hγ t₀
  refine ⟨(h + g (γ t₀)) / 2, ?_, y, hy, γ, ?_⟩
  · linarith
  · intro t
    have ht := hmin t
    linarith

/-- Proves: the exact strict cut for continuous nonessential inputs.
Method: least-upper-bound property, downward closure and strict improvement.
Does not establish: closed-threshold attainment at the death height. -/
theorem superlevelBar_strict_cut {d : ℕ} (g : superlevelBar_E d → ℝ)
    (hg : Continuous g) (M : superlevelBar_E d)
    (hM : ¬ superlevelBar_isEssential g M) (h : ℝ) :
    h < superlevelBar_deathHeight g M ↔ h ∈ superlevelBar_clearanceSet g M := by
  classical
  have hne := superlevelBar_clearance_nonempty g hg M hM
  rw [superlevelBar_death_eq_sSup_of_nonempty g M hne]
  constructor
  · intro hlt
    have hex : ∃ c ∈ superlevelBar_clearanceSet g M, h < c := by
      by_contra hn
      have hub : ∀ c ∈ superlevelBar_clearanceSet g M, c ≤ h := by
        intro c hc
        exact le_of_not_gt (fun hhc => hn ⟨c, hc, hhc⟩)
      exact (not_le_of_gt hlt) (csSup_le hne hub)
    obtain ⟨c, hc, hhc⟩ := hex
    exact superlevelBar_clearance_downward g M hhc hc
  · intro hh
    obtain ⟨c, hhc, hc⟩ := superlevelBar_clearance_improve g hg M hh
    exact hhc.trans_le (le_csSup (superlevelBar_clearance_bddAbove g M) hc)

/-- Proves: the death value is at most the initial height when clearance is
nonempty. Method: the supremum is at most this upper bound.
Does not establish: the bound on the empty default branch. -/
theorem superlevelBar_death_le_start_of_nonempty {d : ℕ}
    (g : superlevelBar_E d → ℝ) (M : superlevelBar_E d)
    (hne : (superlevelBar_clearanceSet g M).Nonempty) :
    superlevelBar_deathHeight g M ≤ g M := by
  rw [superlevelBar_death_eq_sSup_of_nonempty g M hne]
  exact csSup_le hne (fun h hh => (superlevelBar_mem_lt_start g M hh).le)

/-- Proves: the death upper bound for continuous nonessential inputs.
Method: establish nonempty clearance and apply its guarded supremum bound.
Does not establish: a strict inequality or an attained optimal path. -/
theorem superlevelBar_death_le_start {d : ℕ} (g : superlevelBar_E d → ℝ)
    (hg : Continuous g) (M : superlevelBar_E d)
    (hM : ¬ superlevelBar_isEssential g M) : superlevelBar_deathHeight g M ≤ g M := by
  exact superlevelBar_death_le_start_of_nonempty g M
    (superlevelBar_clearance_nonempty g hg M hM)

/-- Proves: the clearance set is the open lower interval below death on the
continuous nonessential locus. Method: extensionality and the strict cut.
Does not establish: a closed-threshold path at the excluded endpoint. -/
theorem superlevelBar_clearance_eq_Iio {d : ℕ} (g : superlevelBar_E d → ℝ)
    (hg : Continuous g) (M : superlevelBar_E d)
    (hM : ¬ superlevelBar_isEssential g M) :
    superlevelBar_clearanceSet g M = Set.Iio (superlevelBar_deathHeight g M) := by
  ext h
  exact (superlevelBar_strict_cut g hg M hM h).symm

/-- Proves: death is not a strict clearance for continuous nonessential inputs.
Method: the cut would otherwise imply that death is strictly below itself.
Does not establish: failure or success of closed-threshold attainment. -/
theorem superlevelBar_death_not_mem_clearance {d : ℕ}
    (g : superlevelBar_E d → ℝ) (hg : Continuous g) (M : superlevelBar_E d)
    (hM : ¬ superlevelBar_isEssential g M) :
    superlevelBar_deathHeight g M ∉ superlevelBar_clearanceSet g M := by
  intro hh
  have hlt := (superlevelBar_strict_cut g hg M hM (superlevelBar_deathHeight g M)).2 hh
  exact (lt_irrefl _) hlt

/-- Proves: uniqueness of a real number with this strict cut on the valid locus.
Method: evaluate either cut at the competing endpoint.
Does not establish: equality of arbitrary total interface records off that locus. -/
theorem superlevelBar_cut_unique {d : ℕ} (g : superlevelBar_E d → ℝ)
    (hg : Continuous g) (M : superlevelBar_E d)
    (hM : ¬ superlevelBar_isEssential g M) (a : ℝ)
    (ha : ∀ h : ℝ, h < a ↔ h ∈ superlevelBar_clearanceSet g M) :
    a = superlevelBar_deathHeight g M := by
  apply le_antisymm
  · by_contra hn
    have hlt : superlevelBar_deathHeight g M < a := lt_of_not_ge hn
    exact superlevelBar_death_not_mem_clearance g hg M hM ((ha _).1 hlt)
  · by_contra hn
    have hlt : a < superlevelBar_deathHeight g M := lt_of_not_ge hn
    have hc := (superlevelBar_strict_cut g hg M hM a).1 hlt
    exact (lt_irrefl a) ((ha a).2 hc)

/-- Proves: any constant height selects the zero default at every point.
Method: every point is globally maximal, hence essential in these data.
Does not establish: any finite ordinary bar for that constant height. -/
theorem superlevelBar_death_const (d : ℕ) (b : ℝ) (M : superlevelBar_E d) :
    superlevelBar_deathHeight (fun _ : superlevelBar_E d => b) M = 0 := by
  apply superlevelBar_death_eq_zero_of_essential
  intro y
  exact le_rfl

/-- Proves: the essential default forbids an unconditional death/start bound.
Method: constant height minus one has empty clearance and death zero.
Does not establish: any finite ordinary bar on a constant field. -/
theorem superlevelBar_control_negative_essential (d : ℕ) (M : superlevelBar_E d) :
    superlevelBar_deathHeight (fun _ : superlevelBar_E d => (-1 : ℝ)) M = 0 ∧
      ¬ superlevelBar_deathHeight (fun _ : superlevelBar_E d => (-1 : ℝ)) M ≤ (-1 : ℝ) := by
  rw [superlevelBar_death_const d (-1) M]
  norm_num

end TorusFiltration
