import TorusFiltration
open Lean in
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← Lean.getEnv
  let modules : Array Lean.Name := #[`TorusTopology, `ClosedSuperlevel, `TorusFiltration]
  let idxs := modules.filterMap env.getModuleIdx?
  if idxs.size != modules.size then throwError "module index missing"
  let names := env.constants.fold (init := (#[] : Array Lean.Name)) fun acc n _ =>
    match env.getModuleIdxFor? n with
    | some i => if idxs.contains i && !n.isInternalDetail then acc.push n else acc
    | none => acc
  for n in names.qsort (fun a b => a.toString < b.toString) do
    IO.println n
  IO.println s!"CENSUS_END {names.size}"
