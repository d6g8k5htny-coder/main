import ResearchFormalCoreR1
import Lean
namespace ResearchFormalCoreR1
  @[simp] theorem indentedAdmission : (1 : Nat) = 1 := rfl
end ResearchFormalCoreR1
open Lean

def _inv.kind : ConstantInfo → String
  | .thmInfo _ => "theorem" | .defnInfo _ => "def" | .axiomInfo _ => "axiom" | .opaqueInfo _ => "opaque"
  | .inductInfo _ => "inductive" | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .quotInfo _ => "quot"

def _inv.hex (s : String) : String :=
  s.toUTF8.foldl (fun acc b => acc ++ (if b.toNat < 16 then "0" else "") ++ String.ofList (Nat.toDigits 16 b.toNat)) ""

def _inv.join (a b : String) : String := if a.isEmpty then b else a ++ "/" ++ b

def _inv.enc : Name → String
  | .anonymous => ""
  | .str p s => _inv.join (_inv.enc p) ("s" ++ _inv.hex s)
  | .num p k => _inv.join (_inv.enc p) ("n" ++ toString k)

structure _inv.St where
  visited : NameSet := {}
  axioms : NameSet := {}

partial def _inv.collect (env : Environment) (c : Name) : StateM _inv.St Unit := do
  unless (← get).visited.contains c do
    modify fun s => { s with visited := s.visited.insert c }
    let collectExpr (e : Expr) : StateM _inv.St Unit := e.getUsedConstants.forM (_inv.collect env)
    match env.find? c with
    | some (.axiomInfo v)  => modify (fun s => { s with axioms := s.axioms.insert c }) *> collectExpr v.type
    | some (.defnInfo v)   => collectExpr v.type *> collectExpr v.value
    | some (.thmInfo v)    => collectExpr v.type *> collectExpr v.value
    | some (.opaqueInfo v) => collectExpr v.type *> collectExpr v.value
    | some (.quotInfo _)   => pure ()
    | some (.ctorInfo v)   => collectExpr v.type
    | some (.recInfo v)    => collectExpr v.type
    | some (.inductInfo v) => collectExpr v.type *> v.ctors.forM (_inv.collect env)
    | none                 => pure ()

def _inv.main (pkg : Name) (includeLocal : Bool) : CoreM Unit := do
  let env ← getEnv
  let header := env.header
  let names := header.moduleNames
  let mut consts : Array (Name × ConstantInfo × Name) := #[]
  for h : idx in [0:names.size] do
    let m := names[idx]
    if pkg.isPrefixOf m then
      let data := header.moduleData[idx]!
      IO.println s!"MODULE|{_inv.enc m}|{data.isModule}|{data.constNames.size}|{data.extraConstNames.size}"
      for c in data.constNames do
        match env.find? c with
        | some ci => consts := consts.push (c, ci, m)
        | none => throwError "constant {c} of module {m} is missing from the environment"
  if includeLocal then
    for (c, ci) in env.constants.map₂.toList do
      unless (`_inv).isPrefixOf c do
        consts := consts.push (c, ci, `_local)
  let sorted := consts.qsort (fun a b => _inv.enc a.1 < _inv.enc b.1)
  for (c, ci, m) in sorted do
    let origin := if (← findDeclarationRanges? c).isSome then "user" else "aux"
    IO.println s!"INV|{_inv.kind ci}|{origin}|{_inv.enc m}|{_inv.enc c}"
  let (_, s) := (sorted.forM (fun x => _inv.collect env x.1)).run {}
  let axs := (s.axioms.toArray.map _inv.enc).qsort (· < ·)
  IO.println s!"AXIOMS|{",".intercalate axs.toList}"
  IO.println s!"CLOSURE|{s.visited.size}"

#eval _inv.main `ResearchFormalCoreR1 true
