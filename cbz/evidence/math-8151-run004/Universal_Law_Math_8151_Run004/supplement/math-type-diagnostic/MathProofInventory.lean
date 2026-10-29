import Lean
import ResearchFormalCoreR1
open Lean Meta Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`ResearchFormalCoreR1, `ResearchFormalCoreR1.AlgebraV2, `ResearchFormalCoreR1.ProbabilityCompanionsV2, `ResearchFormalCoreR1.MeasureBridge, `ResearchFormalCoreR1.WeightedLaw, `ResearchFormalCoreR1.MomentTail, `ResearchFormalCoreR1.MomentGenerality, `ResearchFormalCoreR1.WeightPerturbation, `ResearchFormalCoreR1.D2Schur]
  let mut count : Nat := 0
  for (moduleName, data) in env.header.moduleNames.zip env.header.moduleData do
    if modules.contains moduleName then
      for ci in data.constants do
        if let .thmInfo _ := ci then
          let axioms ← collectAxioms ci.name
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
          let row := Json.mkObj [("declaration",toJson ci.name.toString),("module",toJson moduleName.toString),("statement",toJson statement),("prop_premise_binders",Json.arr premises),("axioms",toJson (axioms.map toString))]
          logInfo m!"CBZ_PROOF {row.compress}"
          count := count + 1
  logInfo m!"CBZ_PROOF_END {count}"
