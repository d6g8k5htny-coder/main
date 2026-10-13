import ResearchFormalCoreR1
import Lean
open Lean

def _inv.kind : ConstantInfo → String
  | .thmInfo _ => "theorem" | .defnInfo _ => "def" | .axiomInfo _ => "axiom" | .opaqueInfo _ => "opaque"
  | .inductInfo _ => "inductive" | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .quotInfo _ => "quot"

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

def _inv.main (env : Environment) (pkg : Name) (includeLocal : Bool) : IO Unit := do
  let header := env.header
  let names := header.moduleNames
  let mut consts : Array (Name × ConstantInfo × Name) := #[]
  for h : idx in [0:names.size] do
    let m := names[idx]
    if pkg.isPrefixOf m then
      let data := header.moduleData[idx]!
      IO.println s!"MODULE|{m}|{data.isModule}|{data.constNames.size}|{data.extraConstNames.size}"
      for c in data.constNames do
        match env.find? c with
        | some ci => consts := consts.push (c, ci, m)
        | none => throw (IO.userError s!"constant {c} of module {m} is missing from the environment")
  if includeLocal then
    for (c, ci) in env.constants.map₂.toList do
      unless (`_inv).isPrefixOf c do
        consts := consts.push (c, ci, `_local)
  let sorted := consts.qsort (fun a b => a.1.toString < b.1.toString)
  for (c, ci, m) in sorted do
    IO.println s!"INV|{_inv.kind ci}|{m}|{c}"
  let (_, s) := (sorted.forM (fun x => _inv.collect env x.1)).run {}
  let axs := (s.axioms.toArray.qsort (fun a b => a.toString < b.toString)).toList.map toString
  IO.println s!"AXIOMS|[{", ".intercalate axs}]"
  IO.println s!"CLOSURE|{s.visited.size}"

#eval show CoreM Unit from do _inv.main (← getEnv) `ResearchFormalCoreR1 false
