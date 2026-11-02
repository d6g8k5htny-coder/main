import Mathlib.CategoryTheory.Category.Preorder
import Mathlib.CategoryTheory.Types.Basic
import Mathlib.Data.Finset.Max
import Mathlib.Order.Fin.Basic

noncomputable section
open CategoryTheory
universe u
namespace ElementaryHistory

/-- Primitive unlabelled component transitions. No elder, basis or interval data. -/
inductive Event {A B : Type u} (f : A → B) where
  | birth (fresh : B) (injective : Function.Injective f)
      (new : fresh ∉ Set.range f) (cover : ∀ b, b = fresh ∨ b ∈ Set.range f)
  | merge (left right : A) (distinct : left ≠ right) (surjective : Function.Surjective f)
      (fiber : ∀ a b, f a = f b ↔ a = b ∨
        (a = left ∧ b = right) ∨ (a = right ∧ b = left))
  | neutral (bijective : Function.Bijective f)

def Event.isBirth {A B : Type u} {f : A → B} : Event f → Prop
  | .birth .. => True
  | .merge .. => False
  | .neutral .. => False

def Event.birthPoint {A B : Type u} {f : A → B} (e : Event f) (h : e.isBirth) : B := by
  cases e with
  | birth fresh _ _ _ => exact fresh
  | merge _ _ _ _ _ => exact False.elim h
  | neutral _ => exact False.elim h

private theorem Event.birthPoint_eq_birth {A B : Type u} {f : A → B}
    {e : Event f} {fresh : B} {hi : Function.Injective f} {hn : fresh ∉ Set.range f}
    {hc : ∀ b, b = fresh ∨ b ∈ Set.range f} (he : e = .birth fresh hi hn hc)
    (h : e.isBirth) : e.birthPoint h = fresh := by
  cases he
  rfl

structure History (n : ℕ) where
  diagram : Fin (n + 1) ⥤ Type u
  finite : ∀ t, Finite (diagram.obj t)
  initial : IsEmpty (diagram.obj 0)
  event : ∀ i : Fin n, Event (diagram.map (homOfLE (show i.castSucc ≤ i.succ by
    simp only [Fin.le_def, Fin.val_castSucc, Fin.val_succ]; omega)))

namespace History

variable {n : ℕ} (H : History.{u} n)

abbrev Component (t : Fin (n + 1)) := H.diagram.obj t
instance (t : Fin (n + 1)) : Finite (H.Component t) := H.finite t

def step (i : Fin n) : H.Component i.castSucc → H.Component i.succ :=
  H.diagram.map (homOfLE (show i.castSucc ≤ i.succ by
    simp only [Fin.le_def, Fin.val_castSucc, Fin.val_succ]; omega))

def BirthTag := {i : Fin n // (H.event i).isBirth}
def bornStage (b : H.BirthTag) : Fin (n + 1) := b.1.succ
def BornAt (t : Fin (n + 1)) := {b : H.BirthTag // H.bornStage b ≤ t}

def birthPoint (b : H.BirthTag) : H.Component (H.bornStage b) :=
  (H.event b.1).birthPoint b.2

def origin (t : Fin (n + 1)) (b : H.BornAt t) : H.Component t :=
  H.diagram.map (homOfLE b.2) (H.birthPoint b.1)

theorem origin_map {t s : Fin (n + 1)} (h : t ≤ s) (b : H.BornAt t) :
    H.diagram.map (homOfLE h) (H.origin t b) =
      H.origin s ⟨b.1, b.2.trans h⟩ := by
  simpa only [origin, homOfLE_comp] using
    (H.diagram.map_comp_apply (homOfLE b.2) (homOfLE h) (H.birthPoint b.1)).symm

theorem origin_surjective (t : Fin (n + 1)) : Function.Surjective (H.origin t) := by
  classical
  induction t using Fin.induction with
  | zero =>
    intro x
    exact (H.initial.false x).elim
  | succ i ih =>
    have hs : i.castSucc ≤ i.succ := by
      simp only [Fin.le_def, Fin.val_castSucc, Fin.val_succ]
      omega
    have hold : ∀ x : H.Component i.castSucc,
        ∃ b : H.BornAt i.succ, H.origin i.succ b = H.step i x := by
      intro x
      obtain ⟨b, hb⟩ := ih x
      refine ⟨⟨b.1, b.2.trans hs⟩, ?_⟩
      rw [← H.origin_map hs b, hb]
      rfl
    cases he : H.event i with
    | birth fresh hinj hnew hcover =>
      intro y
      rcases hcover y with rfl | ⟨x, rfl⟩
      · let b : H.BirthTag := ⟨i, by rw [he]; trivial⟩
        refine ⟨⟨b, le_rfl⟩, ?_⟩
        have hp : H.birthPoint b = y := Event.birthPoint_eq_birth he b.2
        change (H.diagram.map (homOfLE (le_refl i.succ))) (H.birthPoint b) = y
        rw [hp]
        exact H.diagram.map_id_apply i.succ y
      · exact hold x
    | merge left right hne hsurj hfiber =>
      intro y
      obtain ⟨x, rfl⟩ := hsurj y
      exact hold x
    | neutral hbij =>
      intro y
      obtain ⟨x, rfl⟩ := hbij.surjective y
      exact hold x

end History
end ElementaryHistory
