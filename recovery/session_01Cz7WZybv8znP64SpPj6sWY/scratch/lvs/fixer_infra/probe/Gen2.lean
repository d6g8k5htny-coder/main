namespace UniversalLaw.Spec.Field
/-- parametrized Prop -/
def IsTwo (n : Nat) (h : 0 < n) : Prop := n = 2
/-- nested proof in a def body -/
def toFin3 (x : Nat) : Fin 3 := ⟨x % 3, Nat.mod_lt _ (by decide)⟩
/-- pattern matching def -/
def pm : Nat → Nat
  | 0 => 1
  | n + 1 => n
/-- structure with Prop field -/
structure Iface (d : Nat) where
  x : Nat
  pos : 0 < x
end UniversalLaw.Spec.Field
