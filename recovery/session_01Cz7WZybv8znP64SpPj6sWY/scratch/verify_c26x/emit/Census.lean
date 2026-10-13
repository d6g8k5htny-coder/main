import TorusFiltration
open Lean in
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← Lean.getEnv
  let modules : Array Lean.Name := #[`TorusTopology, `ClosedSuperlevel, `SuperlevelBar, `OrdinaryH0, `TorusFiltration]
  let idxs := modules.filterMap env.getModuleIdx?
  if idxs.size != modules.size then throwError "census: a companion module is not imported"
  let names := env.constants.fold (init := (#[] : Array Lean.Name)) fun acc n _ =>
    match env.getModuleIdxFor? n with
    | some i => if idxs.contains i then acc.push n else acc
    | none => acc
  for n in names.qsort (fun a b => a.toString < b.toString) do
    let ranged ← Lean.findDeclarationRanges? n
    IO.println s!"{n} {if n.isInternalDetail then "internal" else "user"} {if ranged.isSome then "range" else "generated"}"
  IO.println s!"CENSUS_END {names.size}"
