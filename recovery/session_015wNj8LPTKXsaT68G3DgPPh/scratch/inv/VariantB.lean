import ResearchFormalCoreR1
import Lean
namespace ResearchFormalCoreR1
  private theorem hiddenAdmission : (1 : Nat) = 1 := by sorry
end ResearchFormalCoreR1
open Lean

def _inv.kind : ConstantInfo → String
  | .thmInfo _ => "theorem" | .defnInfo _ => "def" | .axiomInfo _ => "axiom" | .opaqueInfo _ => "opaque"
  | .inductInfo _ => "inductive" | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .quotInfo _ => "quot"

def _inv.main (env : Environment) (pkg : Name) (includeLocal : Bool) : IO Unit := do
  let mut consts : Array (Name × ConstantInfo × Name) := #[]
  for (c, ci) in env.constants.map₁.toList do
    if let some m := env.getModuleFor? c then
      if pkg.isPrefixOf m then
        consts := consts.push (c, ci, m)
  if includeLocal then
    for (c, ci) in env.constants.map₂.toList do
      unless (`_inv).isPrefixOf c do
        consts := consts.push (c, ci, `_local)
  let sorted := consts.qsort (fun a b => a.1.toString < b.1.toString)
  for (c, ci, m) in sorted do
    IO.println s!"INV|{_inv.kind ci}|{m}|{c}"
  let (_, s) := ((sorted.forM (fun x => CollectAxioms.collect x.1)).run env).run {}
  let axs := (s.axioms.qsort (fun a b => a.toString < b.toString)).toList.map toString
  IO.println s!"AXIOMS|[{", ".intercalate axs}]|{s.visited.size}"

#eval show CoreM Unit from do _inv.main (← getEnv) `ResearchFormalCoreR1 true
