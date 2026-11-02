import IntervalDecomposition
import Mathlib.CategoryTheory.ComposableArrows.Basic
import Mathlib.Algebra.Field.ZMod
import Mathlib.Tactic.NormNum

noncomputable section
open CategoryTheory ElementaryHistory
namespace EventFixturesSupplement

attribute [local simp] Fin.reduceFinMk

local instance : Fact (Nat.Prime 2) := ⟨by decide⟩

private def emptyBirth : Fin 0 → Fin 1 := Fin.elim0
private def birthA : Fin 1 → Fin 2 := Fin.castSucc
private def birthB : Fin 2 → Fin 3 := Fin.castSucc
private def mergeBA : Fin 3 → Fin 2 := fun i => if i = 0 then 0 else 1
private def mergeAG : Fin 2 → Fin 1 := fun _ => 0

private def nestedDiagram : Fin 6 ⥤ Type :=
  ComposableArrows.mk₅ (↾emptyBirth) (↾birthA) (↾birthB) (↾mergeBA) (↾mergeAG)

/-- The inputs are a functor and primitive transition witnesses, never parent or interval data. -/
def nested : History 5 where
  diagram := nestedDiagram
  finite t := by
    fin_cases t <;> dsimp [nestedDiagram] <;> infer_instance
  initial := by
    change IsEmpty (Fin 0)
    infer_instance
  event i := match i with
    | ⟨0, _⟩ => by
      change Event emptyBirth
      refine .birth 0 ?_ ?_ ?_
      · intro a
        exact Fin.elim0 a
      · rintro ⟨a, _⟩
        exact Fin.elim0 a
      · intro b
        fin_cases b
        exact Or.inl rfl
    | ⟨1, _⟩ => by
      change Event birthA
      refine .birth 1 ?_ ?_ ?_
      · exact Fin.castSucc_injective 1
      · rintro ⟨a, ha⟩
        fin_cases a
        norm_num [birthA] at ha
      · intro b
        fin_cases b
        · exact Or.inr ⟨0, rfl⟩
        · exact Or.inl rfl
    | ⟨2, _⟩ => by
      change Event birthB
      refine .birth 2 ?_ ?_ ?_
      · exact Fin.castSucc_injective 2
      · rintro ⟨a, ha⟩
        fin_cases a <;> norm_num [birthB] at ha
      · intro b
        fin_cases b
        · exact Or.inr ⟨0, rfl⟩
        · exact Or.inr ⟨1, rfl⟩
        · exact Or.inl rfl
    | ⟨3, _⟩ => by
      change Event mergeBA
      refine .merge 1 2 (by decide) ?_ ?_
      · intro b
        fin_cases b
        · exact ⟨0, rfl⟩
        · exact ⟨1, rfl⟩
      · intro a b
        fin_cases a <;> fin_cases b <;> decide
    | ⟨4, _⟩ => by
      change Event mergeAG
      refine .merge 0 1 (by decide) ?_ ?_
      · intro b
        fin_cases b
        exact ⟨0, rfl⟩
      · intro a b
        fin_cases a <;> fin_cases b <;> decide

def G : nested.BirthTag := ⟨0, by change True; trivial⟩
def A : nested.BirthTag := ⟨1, by change True; trivial⟩
def B : nested.BirthTag := ⟨2, by change True; trivial⟩

private theorem nested_cases (b : nested.BirthTag) : b = G ∨ b = A ∨ b = B := by
  rcases b with ⟨i, hi⟩
  fin_cases i
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)
  · exact Or.inr (Or.inr rfl)
  · exact False.elim hi
  · exact False.elim hi

private theorem nested_exists (P : nested.BirthTag → Prop) :
    (∃ b, P b) ↔ P G ∨ P A ∨ P B := by
  constructor
  · rintro ⟨b, hb⟩
    rcases nested_cases b with rfl | rfl | rfl
    · exact Or.inl hb
    · exact Or.inr (Or.inl hb)
    · exact Or.inr (Or.inr hb)
  · rintro (h | h | h)
    · exact ⟨G, h⟩
    · exact ⟨A, h⟩
    · exact ⟨B, h⟩

private theorem nested_lt (a b : nested.BirthTag) :
    a < b ↔ a.1.val < b.1.val := Iff.rfl

private theorem nested_eq (a b : nested.BirthTag) :
    a = b ↔ a.1.val = b.1.val := Subtype.ext_iff.trans Fin.ext_iff

private theorem point_G : nested.birthPoint G = (0 : Fin 1) := by rfl
private theorem point_A : nested.birthPoint A = (1 : Fin 2) := by rfl
private theorem point_B : nested.birthPoint B = (2 : Fin 3) := by rfl

private theorem loc_G_0 : nested.location 0 G = none := by
  rw [History.location, dite_eq_right (by change ¬ ((1 : Fin 6) ≤ 0); decide)]

private theorem loc_G_1 : nested.location 1 G = some (0 : Fin 1) := by
  rw [History.location, dite_eq_left (by change (1 : Fin 6) ≤ 1; decide)]
  rfl

private theorem loc_G_2 : nested.location 2 G = some (0 : Fin 2) := by
  rw [History.location, dite_eq_left (by change (1 : Fin 6) ≤ 2; decide)]
  rfl

private theorem loc_G_3 : nested.location 3 G = some (0 : Fin 3) := by
  rw [History.location, dite_eq_left (by change (1 : Fin 6) ≤ 3; decide)]
  rfl

private theorem loc_G_4 : nested.location 4 G = some (0 : Fin 2) := by
  rw [History.location, dite_eq_left (by change (1 : Fin 6) ≤ 4; decide)]
  rfl

private theorem loc_G_5 : nested.location 5 G = some (0 : Fin 1) := by
  rw [History.location, dite_eq_left (by change (1 : Fin 6) ≤ 5; decide)]
  rfl

private theorem loc_A_0 : nested.location 0 A = none := by
  rw [History.location, dite_eq_right (by change ¬ ((2 : Fin 6) ≤ 0); decide)]

private theorem loc_A_1 : nested.location 1 A = none := by
  rw [History.location, dite_eq_right (by change ¬ ((2 : Fin 6) ≤ 1); decide)]

private theorem loc_A_2 : nested.location 2 A = some (1 : Fin 2) := by
  rw [History.location, dite_eq_left (by change (2 : Fin 6) ≤ 2; decide)]
  rfl

private theorem loc_A_3 : nested.location 3 A = some (1 : Fin 3) := by
  rw [History.location, dite_eq_left (by change (2 : Fin 6) ≤ 3; decide)]
  rfl

private theorem loc_A_4 : nested.location 4 A = some (1 : Fin 2) := by
  rw [History.location, dite_eq_left (by change (2 : Fin 6) ≤ 4; decide)]
  rfl

private theorem loc_A_5 : nested.location 5 A = some (0 : Fin 1) := by
  rw [History.location, dite_eq_left (by change (2 : Fin 6) ≤ 5; decide)]
  rfl

private theorem loc_B_0 : nested.location 0 B = none := by
  rw [History.location, dite_eq_right (by change ¬ ((3 : Fin 6) ≤ 0); decide)]

private theorem loc_B_1 : nested.location 1 B = none := by
  rw [History.location, dite_eq_right (by change ¬ ((3 : Fin 6) ≤ 1); decide)]

private theorem loc_B_2 : nested.location 2 B = none := by
  rw [History.location, dite_eq_right (by change ¬ ((3 : Fin 6) ≤ 2); decide)]

private theorem loc_B_3 : nested.location 3 B = some (2 : Fin 3) := by
  rw [History.location, dite_eq_left (by change (3 : Fin 6) ≤ 3; decide)]
  rfl

private theorem loc_B_4 : nested.location 4 B = some (1 : Fin 2) := by
  rw [History.location, dite_eq_left (by change (3 : Fin 6) ≤ 4; decide)]
  rfl

private theorem loc_B_5 : nested.location 5 B = some (0 : Fin 1) := by
  rw [History.location, dite_eq_left (by change (3 : Fin 6) ≤ 5; decide)]
  rfl

private theorem older_B (t : Fin 6) : nested.OlderAt t B ↔ (4 : Fin 6) ≤ t := by
  fin_cases t <;> simp only [Fin.reduceFinMk, History.OlderAt, nested_exists, loc_G_0, loc_G_1, loc_G_2, loc_G_3, loc_G_4, loc_G_5, loc_A_0, loc_A_1, loc_A_2, loc_A_3, loc_A_4, loc_A_5, loc_B_0, loc_B_1, loc_B_2, loc_B_3, loc_B_4, loc_B_5] <;>
    simp only [nested_lt] <;>
    norm_num [G, A, B, History.bornStage, Fin.le_def, Fin.succ, -Fin.val_fin_lt]

  · change ¬ (some (0 : Fin 3) = some 2) ∧ ¬ (some (1 : Fin 3) = some 2)
    decide
  · change (some (0 : Fin 2) = some 1) ∨ (some (1 : Fin 2) = some 1)
    exact Or.inr rfl
  · rfl

private theorem older_A (t : Fin 6) : nested.OlderAt t A ↔ (5 : Fin 6) ≤ t := by
  fin_cases t <;> simp only [Fin.reduceFinMk, History.OlderAt, nested_exists, loc_G_0, loc_G_1, loc_G_2, loc_G_3, loc_G_4, loc_G_5, loc_A_0, loc_A_1, loc_A_2, loc_A_3, loc_A_4, loc_A_5, loc_B_0, loc_B_1, loc_B_2, loc_B_3, loc_B_4, loc_B_5] <;>
    simp only [nested_lt] <;>
    norm_num [G, A, B, History.bornStage, Fin.le_def, Fin.succ, -Fin.val_fin_lt]

  · change ¬ (some (0 : Fin 2) = some 1)
    decide
  · change ¬ (some (0 : Fin 3) = some 1)
    decide
  · change ¬ (some (0 : Fin 2) = some 1)
    decide
  · rfl

private theorem firstDeath_at {n : ℕ} (H : History n) (j : H.BirthTag)
    (t : Fin (n + 1)) (ht : H.OlderAt t j)
    (hmin : ∀ s, H.OlderAt s j → t ≤ s) : H.firstDeath j = some ⟨t, ht⟩ := by
  classical
  have hmem : t ∈ H.collisionStages j := by simp [History.collisionStages, ht]
  have hn : (H.collisionStages j).Nonempty := ⟨t, hmem⟩
  have hm : (H.collisionStages j).min' hn = t :=
    (Finset.min'_eq_iff (H.collisionStages j) hn t).2 ⟨hmem, fun s hs => hmin s (by
      simpa [History.collisionStages] using hs)⟩
  unfold History.firstDeath
  rw [dite_eq_left hn]
  congr 1
  exact Subtype.ext hm

private theorem firstDeath_B : nested.firstDeath B =
    some ⟨4, (older_B 4).2 (by decide)⟩ :=
  firstDeath_at nested B 4 ((older_B 4).2 (by decide)) (fun s hs => (older_B s).1 hs)

private theorem firstDeath_A : nested.firstDeath A =
    some ⟨5, (older_A 5).2 (by decide)⟩ :=
  firstDeath_at nested A 5 ((older_A 5).2 (by decide)) (fun s hs => (older_A s).1 hs)

-- These test declarations are proved from the concrete primitive history.
theorem parent_B_is_A : nested.parent B = some A := by
  classical
  have hs : Finset.univ.filter (fun i : nested.BirthTag =>
      i < B ∧ nested.location 4 i = nested.location 4 B) = {A} := by
    ext i
    rcases nested_cases i with rfl | rfl | rfl <;>
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        loc_G_4, loc_A_4, loc_B_4] <;> simp only [nested_lt, nested_eq] <;>
      norm_num [G, A, B, -Fin.val_fin_lt]
    · change ¬ (some (0 : Fin 2) = some 1)
      decide
    · rfl
  simp [History.parent, firstDeath_B, hs]

theorem parent_A_is_G : nested.parent A = some G := by
  classical
  have hs : Finset.univ.filter (fun i : nested.BirthTag =>
      i < A ∧ nested.location 5 i = nested.location 5 A) = {G} := by
    ext i
    rcases nested_cases i with rfl | rfl | rfl <;>
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton,
        loc_G_5, loc_A_5, loc_B_5] <;> simp only [nested_lt, nested_eq] <;>
      norm_num [G, A, B, -Fin.val_fin_lt]
    · rfl
  simp [History.parent, firstDeath_A, hs]

theorem death_B : nested.death B = 4 := by
  simp [History.death, firstDeath_B]

theorem death_A : nested.death A = 5 := by
  simp [History.death, firstDeath_A]

theorem root_essential : nested.parent G = none := by
  cases hp : nested.parent G with
  | none => rfl
  | some p =>
    have hl := nested.parent_strict hp
    change p.1.val < 0 at hl
    exact False.elim (Nat.not_lt_zero _ hl)

example : nested.parent B ≠ some G := by
  rw [parent_B_is_A]
  intro h
  have he := Option.some.inj h
  have hv := congrArg (fun b : nested.BirthTag => b.1.val) he
  norm_num [A, G] at hv

example : nested.death B < nested.death A := by
  rw [death_B, death_A]
  decide

example : nested.diagram.map (homOfLE (by decide : (3 : Fin 6) ≤ 5)) (2 : Fin 3) =
    (0 : Fin 1) := by
  rfl

example : nested.diagram.map (homOfLE (by decide : (3 : Fin 6) ≤ 4)) (2 : Fin 3) =
    (1 : Fin 2) := by
  rfl

private def aliveB : nested.Alive 3 :=
  ⟨B, by
    change nested.Active (nested.bornStage B) B
    exact nested.birth_active B⟩

private def aliveA : nested.Alive 4 := by
  refine ⟨A, ?_⟩
  constructor
  · change (2 : Fin 6) ≤ 4
    decide
  · rw [death_A]
    decide

theorem nested_B_generator : nested.fixedGenerator ℚ 3 aliveB =
    Finsupp.single (2 : Fin 3) 1 - Finsupp.single 1 1 := by
  classical
  have hp : nested.aliveParent 3 aliveB =
      some ⟨A, nested.parent_active parent_B_is_A aliveB.2⟩ :=
    Option.pmap_some' parent_B_is_A
  simp only [History.fixedGenerator, StrictParentBasis.difference, hp,
    Finsupp.mapDomain_sub, Finsupp.mapDomain_single]
  congr 1

theorem B_zero_at_death :
    Finsupp.lmapDomain ℚ ℚ (nested.diagram.map (homOfLE (by decide : (3 : Fin 6) ≤ 4)))
      (nested.fixedGenerator ℚ 3 aliveB) = 0 := by
  rw [nested_B_generator]
  change (Finsupp.lmapDomain ℚ ℚ mergeBA)
    (Finsupp.single (2 : Fin 3) 1 - Finsupp.single 1 1) = 0
  rw [map_sub]
  simp only [Finsupp.lmapDomain_apply, Finsupp.mapDomain_single]
  change Finsupp.single (1 : Fin 2) (1 : ℚ) - Finsupp.single 1 1 = 0
  exact sub_self _

theorem A_survives_B_death : nested.fixedGenerator ℚ 4 aliveA ≠ 0 := by
  rw [← nested.stateBasis_apply ℚ 4 aliveA]
  exact (nested.stateBasis ℚ 4).ne_zero aliveA

-- Birth is included, the first collision/death is excluded, and an unborn tag is absent.
example : nested.Active 3 B := aliveB.2
example : ¬ nested.Active 4 B := by simp [History.Active, death_B]
example : ¬ nested.Active 1 B := by
  intro h
  have hb := h.1
  norm_num [History.bornStage, B, Fin.le_def, Fin.succ] at hb

-- The wrong final-root difference does NOT disappear at B's actual first death.
example : Finsupp.lmapDomain ℚ ℚ
    (nested.diagram.map (homOfLE (by decide : (3 : Fin 6) ≤ 4)))
    (Finsupp.single (2 : Fin 3) 1 - Finsupp.single 0 1) ≠ 0 := by
  change (Finsupp.lmapDomain ℚ ℚ mergeBA)
    (Finsupp.single (2 : Fin 3) 1 - Finsupp.single 0 1) ≠ 0
  rw [map_sub]
  simp only [Finsupp.lmapDomain_apply, Finsupp.mapDomain_single]
  change Finsupp.single (1 : Fin 2) (1 : ℚ) - Finsupp.single 0 1 ≠ 0
  intro h
  have hc := congrArg (fun v : Fin 2 →₀ ℚ => v 1) h
  norm_num at hc

example (t s : Fin 6) (h : t ≤ s) (v : nested.Alive t →₀ ZMod 2) :
    Finsupp.lmapDomain (ZMod 2) (ZMod 2) (nested.diagram.map (homOfLE h))
      (nested.toComponents (ZMod 2) t v) =
      nested.toComponents (ZMod 2) s (nested.intervalMap (ZMod 2) h v) :=
  nested.coordinates_natural (ZMod 2) h v

example (t s r : Fin 6) (h : t ≤ s) (h' : s ≤ r) :
    nested.intervalMap (ZMod 2) (h.trans h') =
      (nested.intervalMap (ZMod 2) h').comp (nested.intervalMap (ZMod 2) h) :=
  nested.intervalMap_comp (ZMod 2) h h'

private def swapTwo : Fin 2 → Fin 2 := Equiv.swap 0 1
private def disconnectedDiagram : Fin 4 ⥤ Type :=
  ComposableArrows.mk₃ (↾emptyBirth) (↾birthA) (↾swapTwo)

/-- Two births followed by an actual permutation, so both terminal components are essential. -/
def disconnected : History 3 where
  diagram := disconnectedDiagram
  finite t := by
    fin_cases t <;> dsimp [disconnectedDiagram] <;> infer_instance
  initial := by
    change IsEmpty (Fin 0)
    infer_instance
  event i := match i with
    | ⟨0, _⟩ => by
      change Event emptyBirth
      exact nested.event 0
    | ⟨1, _⟩ => by
      change Event birthA
      exact nested.event 1
    | ⟨2, _⟩ => by
      change Event swapTwo
      exact .neutral (Equiv.swap 0 1).bijective

def disconnectedG : disconnected.BirthTag := ⟨0, by change True; trivial⟩
def disconnectedA : disconnected.BirthTag := ⟨1, by change True; trivial⟩

private theorem disconnected_cases (b : disconnected.BirthTag) :
    b = disconnectedG ∨ b = disconnectedA := by
  rcases b with ⟨i, hi⟩
  fin_cases i
  · exact Or.inl rfl
  · exact Or.inr rfl
  · exact False.elim hi

private theorem disconnected_exists (P : disconnected.BirthTag → Prop) :
    (∃ b, P b) ↔ P disconnectedG ∨ P disconnectedA := by
  constructor
  · rintro ⟨b, hb⟩
    rcases disconnected_cases b with rfl | rfl
    · exact Or.inl hb
    · exact Or.inr hb
  · rintro (h | h)
    · exact ⟨disconnectedG, h⟩
    · exact ⟨disconnectedA, h⟩

private theorem disconnected_point_G : disconnected.birthPoint disconnectedG = (0 : Fin 1) := by rfl
private theorem disconnected_point_A : disconnected.birthPoint disconnectedA = (1 : Fin 2) := by rfl

private theorem dloc_G_0 : disconnected.location 0 disconnectedG = none := by
  rw [History.location, dite_eq_right (by change ¬ ((1 : Fin 4) ≤ 0); decide)]

private theorem dloc_G_1 : disconnected.location 1 disconnectedG = some (0 : Fin 1) := by
  rw [History.location, dite_eq_left (by change (1 : Fin 4) ≤ 1; decide)]
  rfl

private theorem dloc_G_2 : disconnected.location 2 disconnectedG = some (0 : Fin 2) := by
  rw [History.location, dite_eq_left (by change (1 : Fin 4) ≤ 2; decide)]
  rfl

private theorem dloc_G_3 : disconnected.location 3 disconnectedG = some (1 : Fin 2) := by
  rw [History.location, dite_eq_left (by change (1 : Fin 4) ≤ 3; decide)]
  change some (Equiv.swap (0 : Fin 2) 1 0) = some (1 : Fin 2)
  simp

private theorem dloc_A_0 : disconnected.location 0 disconnectedA = none := by
  rw [History.location, dite_eq_right (by change ¬ ((2 : Fin 4) ≤ 0); decide)]

private theorem dloc_A_1 : disconnected.location 1 disconnectedA = none := by
  rw [History.location, dite_eq_right (by change ¬ ((2 : Fin 4) ≤ 1); decide)]

private theorem dloc_A_2 : disconnected.location 2 disconnectedA = some (1 : Fin 2) := by
  rw [History.location, dite_eq_left (by change (2 : Fin 4) ≤ 2; decide)]
  rfl

private theorem dloc_A_3 : disconnected.location 3 disconnectedA = some (0 : Fin 2) := by
  rw [History.location, dite_eq_left (by change (2 : Fin 4) ≤ 3; decide)]
  change some (Equiv.swap (0 : Fin 2) 1 1) = some (0 : Fin 2)
  simp

theorem disconnected_G_essential : disconnected.parent disconnectedG = none := by
  cases hp : disconnected.parent disconnectedG with
  | none => rfl
  | some p =>
    have hl := disconnected.parent_strict hp
    change p.1.val < 0 at hl
    exact False.elim (Nat.not_lt_zero _ hl)

theorem disconnected_A_essential : disconnected.parent disconnectedA = none := by
  cases hp : disconnected.parent disconnectedA with
  | none => rfl
  | some p =>
    have hpg : p = disconnectedG := by
      apply Subtype.ext
      apply Fin.ext
      have hl := disconnected.parent_strict hp
      change p.1.val < 1 at hl
      change p.1.val = 0
      omega
    subst p
    obtain ⟨d, hd, _, he⟩ := disconnected.parent_collision hp
    have hb := (disconnected.birth_active disconnectedA).2
    change 2 < disconnected.death disconnectedA at hb
    rw [hd] at hb
    have hd3 : d = (3 : Fin 4) := by
      apply Fin.ext
      have hi := d.isLt
      change d.val = 3
      omega
    subst d
    rw [dloc_G_3, dloc_A_3] at he
    exact False.elim ((by decide : (1 : Fin 2) ≠ 0) (Option.some.inj he))

example : disconnected.diagram.map (homOfLE (by decide : (2 : Fin 4) ≤ 3))
    (0 : Fin 2) = (1 : Fin 2) := by
  change Equiv.swap (0 : Fin 2) 1 0 = 1
  simp

example : disconnected.diagram.map (homOfLE (by decide : (2 : Fin 4) ≤ 3))
    (1 : Fin 2) = (0 : Fin 2) := by
  change Equiv.swap (0 : Fin 2) 1 1 = 0
  simp

example : disconnected.Alive (3 : Fin 4) ≃ Fin 2 := disconnected.aliveEquiv 3

/-- No events, no born tags, and the actual initial component set is empty. -/
def empty : History 0 where
  diagram := ComposableArrows.mk₀ (Fin 0)
  finite _ := by
    change Finite (Fin 0)
    infer_instance
  initial := by
    change IsEmpty (Fin 0)
    infer_instance
  event i := Fin.elim0 i

example : IsEmpty empty.BirthTag := ⟨fun b => Fin.elim0 b.1⟩
example : IsEmpty (empty.Component 0) := by
  change IsEmpty (Fin 0)
  infer_instance
example : empty.Alive 0 ≃ Fin 0 := empty.aliveEquiv 0

end EventFixturesSupplement

