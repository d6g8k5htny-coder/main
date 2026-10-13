import OrdinaryH0Controls
open Lean in
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← Lean.getEnv
  let some idx := env.getModuleIdx? `OrdinaryH0Controls | throwError "census: module not imported"
  let names := env.constants.fold (init := (#[] : Array Lean.Name)) fun acc n _ =>
    match env.getModuleIdxFor? n with
    | some i => if i == idx then acc.push n else acc
    | none => acc
  for n in names.qsort (fun a b => a.toString < b.toString) do
    let ranged ← Lean.findDeclarationRanges? n
    IO.println s!"{n} {if n.isInternalDetail then "internal" else "user"} {if ranged.isSome then "range" else "generated"}"
  IO.println s!"CENSUS_END {names.size}"
