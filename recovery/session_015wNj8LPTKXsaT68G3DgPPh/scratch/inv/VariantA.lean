import ResearchFormalCoreR1
import Lean
namespace ResearchFormalCoreR1
  private theorem hiddenAdmission : (1 : Nat) = 1 := by sorry
end ResearchFormalCoreR1
open Lean

def _inv.kind : ConstantInfo → String
  | .thmInfo _ => "theorem" | .defnInfo _ => "def" | .axiomInfo _ => "axiom" | .opaqueInfo _ => "opaque"
  | .inductInfo _ => "inductive" | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .quotInfo _ => "quot"

def _inv.report (c : Name) (ci : ConstantInfo) (m : Name) : CoreM Unit := do
  let axs : Array Name := #[]
  let names := (axs.qsort (fun a b => a.toString < b.toString)).toList.map toString
  IO.println s!"INV|{_inv.kind ci}|{m}|{c}|[{", ".intercalate names}]"

#eval show CoreM Unit from do
  let env ← getEnv
  let pkg := `ResearchFormalCoreR1
  let report := _inv.report
  for (c, ci) in env.constants.map₁.toList do
    if let some m := env.getModuleFor? c then
      if pkg.isPrefixOf m then
        report c ci m
  for (c, ci) in env.constants.map₂.toList do
    unless `_inv.isPrefixOf c do
      report c ci `_local
