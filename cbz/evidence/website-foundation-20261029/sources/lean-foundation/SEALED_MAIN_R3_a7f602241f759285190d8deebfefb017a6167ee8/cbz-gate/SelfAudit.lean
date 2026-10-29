import Lean
import CBZNonvacuity
open Lean Meta Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`CBZNonvacuity]
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut found : Array Name := #[]
  let mut count : Nat := 0
  for (moduleName, data) in env.header.moduleNames.zip env.header.moduleData do
    if modules.contains moduleName then
      found := found.push moduleName
      for ci in data.constants do
        let axioms ← collectAxioms ci.name
        let bad := axioms.filter (!allowed.contains ·)
        unless bad.isEmpty do
          throwError "CBZ_FORBIDDEN {ci.name} {bad}"
        let kind := match ci with
          | .thmInfo _ => "theorem"
          | .axiomInfo _ => "axiom"
          | .defnInfo _ => "definition"
          | _ => "generated"
        if kind == "axiom" then throwError "CBZ_FORBIDDEN project axiom {ci.name}"
        let (statement, premises) ← liftTermElabM do
          let statement := (← ppExpr ci.type).pretty
          let premises ← forallTelescope ci.type fun xs _ => do
            let mut ps : Array Json := #[]
            for x in xs do
              let decl ← x.fvarId!.getDecl
              if ← isProp decl.type then
                ps := ps.push <| Json.mkObj [("name",toJson decl.userName.toString),("type",toJson (← ppExpr decl.type).pretty)]
            pure ps
          pure (statement,premises)
        let row := Json.mkObj [("name",toJson ci.name.toString),("module",toJson moduleName.toString),("kind",toJson kind),("axioms",toJson (axioms.map toString)),("statement",toJson statement),("prop_premises",Json.arr premises)]
        logInfo m!"CBZ_RECORD {row.compress}"
        count := count + 1
  unless found.size == modules.size do
    throwError "CBZ_MISSING_MODULE {modules.filter (!found.contains ·)}"
  let summary := Json.mkObj [("count",toJson count),("modules",toJson (found.map toString))]
  logInfo m!"CBZ_END {summary.compress}"
