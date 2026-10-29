import Mathlib
import SuperlevelBar
import Specifications.Field

/-!
# C263 adapter: an inhabitant of the Field superlevel-bar interface

Proves: the C263 strict-clearance data (`superlevelBar_deathHeight`,
`superlevelBar_isEssential`) together with the elder clause's own right-hand side
inhabit `UniversalLaw.Spec.Field.SuperlevelBarInterface d L` for every `d` and `L`;
on smooth `L`-periodic heights every inhabitant agrees with this one on essentiality,
on the death height at nonessential points and, on the Morse distinct-value locus, on
the elder relation; and the interface leaves the death height at essential points free
(two inhabitants differ there).

Method: `IsSmooth` at `n = 0` gives continuity, so `superlevelBar_strict_cut` is the
maximin clause and `superlevelBar_cut_unique` forces agreement of any other death
height; `essential_iff` is `superlevelBar_essential_iff`; `ElderPartner` is defined as
the right-hand side of `elderPartner_iff`, so that clause holds by `Iff.rfl`.

Does not establish: an identification of `ElderPartner` with a persistence-module elder
rule, the source's elder event or its Borel measurability (the clause holds by
definition, not as a theorem about persistence); any use of periodicity or of `L` in the
construction (the uniqueness theorems take the periodicity hypothesis only to apply an
arbitrary inhabitant's clauses, and are false without it); a torus quotient; Morse
genericity; any meaning of the value `0` that the C263 default
gives at essential points; anything about the separate
`UniversalLaw.Spec.Lifetime.SuperlevelBarInterface` (not inhabited here) or about any
Lifetime statement; scientific acceptance. The statement packet's own text "no
inhabitant is constructed" is about that packet and stays true of it; nothing in it
changes. No claim-level label moves; alignment stays PENDING_INDEPENDENT_REVIEW.

Authored by the Anthropic/Claude lane (owner of this companion's gate), delegated AI
work for Dylan Roy, under the adapter route of Math- #428 comments 6112932068 and
6124097060; organizational-independence credit 0; no human review. It consumes the
C263 definitions, whose FORMAL HOLD extends to this module. The interface is
`specifications/lean_statements_20261010/Specifications/Field.lean`, blob
cc40e9d19446b8a150401c574d3cfd56177f195f, compiled in the same replay.
Target pin: Lean 4.34.1 / Mathlib d13f23b723b8a846827a245b89c10fc7d3f11612.
Elaboration, kernel audit and executed controls are recorded in separate source-bound
receipts; this text makes no execution claim.
-/

noncomputable section

open UniversalLaw.Spec.Field

namespace TorusFiltration

/-- Proves: a smooth height in the Field sense is continuous.
Method: the `n = 0` instance of `IsSmooth` and `ContDiff.continuous`.
Does not establish: any regularity beyond continuity. -/
theorem superlevelBarField_continuous {d : ℕ} {g : E d → ℝ} (hg : IsSmooth d g) :
    Continuous g :=
  (hg 0).continuous

/-- Proves: the C263 data inhabit the Field interface for every dimension and period.
Method: death height and essentiality are the C263 definitions, the maximin clause is
`superlevelBar_strict_cut`, and the elder relation is the right-hand side of its clause.
Does not establish: a persistence-module elder rule; `elderPartner_iff` holds by
definition, and periodicity and `L` are not used. -/
def superlevelBarField_interface (d : ℕ) (L : ℝ) : SuperlevelBarInterface d L where
  deathHeight := superlevelBar_deathHeight
  IsEssential := superlevelBar_isEssential
  ElderPartner := fun g M S =>
    IsNondegMax d g M ∧ IsIndexSaddle d g S ∧ ¬ superlevelBar_isEssential g M ∧
      superlevelBar_deathHeight g M = g S
  essential_iff := fun g _ _ M => superlevelBar_essential_iff g M
  deathHeight_maximin := fun g hs _ M hM h =>
    superlevelBar_strict_cut g (superlevelBarField_continuous hs) M hM h
  elderPartner_iff := fun _ _ _ _ _ _ => Iff.rfl

/-- Proves: the Field superlevel-bar interface is inhabited for every `d` and `L`.
Method: the record above. Does not establish: that any inhabitant is the source's
superlevel H0 object. -/
theorem superlevelBarField_nonempty (d : ℕ) (L : ℝ) :
    Nonempty (SuperlevelBarInterface d L) :=
  ⟨superlevelBarField_interface d L⟩

/-- Proves: the death height of the record is the C263 death height.
Method: `rfl`. Does not establish: a finite death for an essential point. -/
theorem superlevelBarField_deathHeight_eq {d : ℕ} {L : ℝ} (g : E d → ℝ) (M : E d) :
    (superlevelBarField_interface d L).deathHeight g M = superlevelBar_deathHeight g M :=
  rfl

/-- Proves: essentiality in the record is global maximality.
Method: `Iff.rfl`. Does not establish: agreement with a persistence essential class. -/
theorem superlevelBarField_isEssential_iff {d : ℕ} {L : ℝ} (g : E d → ℝ) (M : E d) :
    (superlevelBarField_interface d L).IsEssential g M ↔ ∀ y, g y ≤ g M :=
  Iff.rfl

/-- Proves: the elder relation of the record is, for every height, the right-hand side
of the interface clause. Method: `Iff.rfl`, since it is defined that way.
Does not establish: that this relation is ordinary elder death in any persistence sense. -/
theorem superlevelBarField_elderPartner_iff {d : ℕ} {L : ℝ} (g : E d → ℝ) (M S : E d) :
    (superlevelBarField_interface d L).ElderPartner g M S ↔
      (IsNondegMax d g M ∧ IsIndexSaddle d g S ∧ ¬ superlevelBar_isEssential g M ∧
        superlevelBar_deathHeight g M = g S) :=
  Iff.rfl

/-- Proves: on smooth periodic heights every inhabitant's essentiality is global maximality.
Method: its own `essential_iff` clause. Does not establish: anything off that locus. -/
theorem superlevelBarField_essential_unique {d : ℕ} {L : ℝ} (I : SuperlevelBarInterface d L)
    (g : E d → ℝ) (hs : IsSmooth d g) (hp : IsLPeriodic d L g) (M : E d) :
    I.IsEssential g M ↔ superlevelBar_isEssential g M :=
  I.essential_iff g hs hp M

/-- Proves: on smooth periodic heights every inhabitant's death height at a nonessential
point is the C263 death height. Method: its maximin clause is a strict cut for the
clearance set, and `superlevelBar_cut_unique` leaves one such number.
Does not establish: agreement at essential points (see the control below). -/
theorem superlevelBarField_deathHeight_unique {d : ℕ} {L : ℝ} (I : SuperlevelBarInterface d L)
    (g : E d → ℝ) (hs : IsSmooth d g) (hp : IsLPeriodic d L g) (M : E d)
    (hM : ¬ I.IsEssential g M) :
    I.deathHeight g M = superlevelBar_deathHeight g M := by
  have hM' : ¬ superlevelBar_isEssential g M :=
    fun h => hM ((superlevelBarField_essential_unique I g hs hp M).2 h)
  exact superlevelBar_cut_unique g (superlevelBarField_continuous hs) M hM' _
    (fun h => I.deathHeight_maximin g hs hp M hM h)

/-- Proves: on the Morse distinct-value locus of smooth periodic heights every inhabitant's
elder relation is the record's. Method: both are their clause's right-hand side, whose
essentiality and death-height terms agree by the two uniqueness theorems above.
Does not establish: agreement off the Morse distinct-value locus, where the interface
constrains no elder relation. -/
theorem superlevelBarField_elderPartner_unique {d : ℕ} {L : ℝ} (I : SuperlevelBarInterface d L)
    (g : E d → ℝ) (hs : IsSmooth d g) (hp : IsLPeriodic d L g) (hm : IsMorseDistinct d L g)
    (M S : E d) :
    I.ElderPartner g M S ↔ (superlevelBarField_interface d L).ElderPartner g M S := by
  rw [I.elderPartner_iff g hs hp hm M S]
  constructor
  · rintro ⟨hmax, hsad, hne, hdeath⟩
    exact ⟨hmax, hsad, fun h => hne ((superlevelBarField_essential_unique I g hs hp M).2 h),
      (superlevelBarField_deathHeight_unique I g hs hp M hne).symm.trans hdeath⟩
  · rintro ⟨hmax, hsad, hne, hdeath⟩
    have hne' : ¬ I.IsEssential g M :=
      fun h => hne ((superlevelBarField_essential_unique I g hs hp M).1 h)
    exact ⟨hmax, hsad, hne', (superlevelBarField_deathHeight_unique I g hs hp M hne').trans hdeath⟩

/-- Proves: a second record, equal to the first except that its death height is the
constant `c` at essential points, also inhabits the interface. Method: off the essential
points it is the C263 death height, so the same strict cut applies; its elder relation is
again its clause's right-hand side. Does not establish: any preferred value at essential
points. -/
def superlevelBarField_shift (d : ℕ) (L c : ℝ) : SuperlevelBarInterface d L where
  deathHeight := fun g M => by
    classical
    exact if superlevelBar_isEssential g M then c else superlevelBar_deathHeight g M
  IsEssential := superlevelBar_isEssential
  ElderPartner := fun g M S =>
    IsNondegMax d g M ∧ IsIndexSaddle d g S ∧ ¬ superlevelBar_isEssential g M ∧
      (by classical
          exact if superlevelBar_isEssential g M then c else superlevelBar_deathHeight g M) = g S
  essential_iff := fun g _ _ M => superlevelBar_essential_iff g M
  deathHeight_maximin := fun g hs _ M hM h => by
    classical
    rw [ite_eq_right hM]
    exact superlevelBar_strict_cut g (superlevelBarField_continuous hs) M hM h
  elderPartner_iff := fun _ _ _ _ _ _ => Iff.rfl

/-- Proves: at a constant height the shifted record's death height is `c`.
Method: every point of a constant height is essential. Does not establish: a finite bar
for a constant height. -/
theorem superlevelBarField_shift_deathHeight_const (d : ℕ) (L c b : ℝ) (M : E d) :
    (superlevelBarField_shift d L c).deathHeight (fun _ => b) M = c := by
  classical
  show (if superlevelBar_isEssential (fun _ : E d => b) M then c
    else superlevelBar_deathHeight (fun _ : E d => b) M) = c
  have hM : superlevelBar_isEssential (fun _ : E d => b) M := fun _ => le_rfl
  rw [ite_eq_left hM]

/-- Proves: constant heights are smooth. Method: `contDiff_const`.
Does not establish: anything about nonconstant heights. -/
theorem superlevelBarField_const_smooth (d : ℕ) (b : ℝ) : IsSmooth d (fun _ : E d => b) :=
  fun _ => contDiff_const

/-- Proves: constant heights are `L`-periodic. Method: both sides are the constant.
Does not establish: anything about the lattice. -/
theorem superlevelBarField_const_periodic (d : ℕ) (L b : ℝ) :
    IsLPeriodic d L (fun _ : E d => b) :=
  fun _ _ => rfl

/-- Proves: the nonessential hypothesis of `superlevelBarField_deathHeight_unique` cannot
be dropped: at the smooth periodic constant height `0` every point is essential for both
records, the record's death height there is `0`, its shift by `1` has death height `1`,
and the two differ. Method: `superlevelBar_death_const` and the shifted value.
Does not establish: anything about nonessential points, where all inhabitants agree. -/
theorem superlevelBarField_control_essential_free (d : ℕ) (L : ℝ) (M : E d) :
    IsSmooth d (fun _ : E d => (0 : ℝ)) ∧ IsLPeriodic d L (fun _ : E d => (0 : ℝ)) ∧
      (superlevelBarField_interface d L).IsEssential (fun _ => (0 : ℝ)) M ∧
      (superlevelBarField_shift d L 1).IsEssential (fun _ => (0 : ℝ)) M ∧
      (superlevelBarField_interface d L).deathHeight (fun _ => (0 : ℝ)) M = 0 ∧
      (superlevelBarField_shift d L 1).deathHeight (fun _ => (0 : ℝ)) M = 1 ∧
      (superlevelBarField_interface d L).deathHeight (fun _ => (0 : ℝ)) M ≠
        (superlevelBarField_shift d L 1).deathHeight (fun _ => (0 : ℝ)) M := by
  have h0 : (superlevelBarField_interface d L).deathHeight (fun _ => (0 : ℝ)) M = 0 := by
    rw [superlevelBarField_deathHeight_eq, superlevelBar_death_const]
  have h1 : (superlevelBarField_shift d L 1).deathHeight (fun _ => (0 : ℝ)) M = 1 :=
    superlevelBarField_shift_deathHeight_const d L 1 0 M
  refine ⟨superlevelBarField_const_smooth d 0, superlevelBarField_const_periodic d L 0,
    fun _ => le_rfl, fun _ => le_rfl, h0, h1, ?_⟩
  rw [h0, h1]
  norm_num

end TorusFiltration
