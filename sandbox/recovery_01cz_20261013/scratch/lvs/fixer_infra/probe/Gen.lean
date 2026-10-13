namespace UniversalLaw.Spec.Field

/-- statement -/
def TwoAndTwo : Prop := (2 : Nat) + 2 = 4

/-- Type-valued structure with a Prop field -/
structure Iface (d : Nat) where
  x : Nat
  pos : 0 < x

/-- Prop-valued structure -/
structure PropIface (n : Nat) : Prop where
  pos : 0 < n
  even : n % 2 = 0

/-- extends -/
structure Ext (d : Nat) extends Iface d where
  y : Nat
  hy : y = x

/-- explicit constructor name -/
structure Named where
  build ::
  z : Nat

/-- auxiliary -/
noncomputable def helper (x : Nat) : Nat := x + 1

/-- abbrev auxiliary -/
abbrev E (d : Nat) := Fin d → Nat

-- escape families (should be flagged by the audit)
def proofAsDef : (2 : Nat) + 2 = 4 := rfl
abbrev proofAsAbbrev : True := trivial
def Untyped := (2 : Nat) + 2 = 4
def SortZero : Sort 0 := True
 def HiddenIndented : Prop := False
theorem smuggled : True := trivial
axiom hidden : False
opaque opq : Nat
def usesSorry : Prop := sorry
instance : Inhabited (Iface 1) := ⟨⟨1, Nat.one_pos⟩⟩

end UniversalLaw.Spec.Field
