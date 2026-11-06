module
public import Init

public inductive ReplayToy where
  | zero : ReplayToy
  | next : ReplayToy → ReplayToy

private theorem privateWitness : True := True.intro
public theorem publicWitness : True := privateWitness
public def publicData : ReplayToy := .next .zero
private def privateData : Nat := 7
public theorem publicDataLaw : (7 : Nat) = 7 := rfl
