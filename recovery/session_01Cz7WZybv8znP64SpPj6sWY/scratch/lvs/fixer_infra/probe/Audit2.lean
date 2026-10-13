import Lean
import Gen2
open Lean Meta Elab Command
run_cmd do
  let env ← getEnv
  for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
    unless modName == `Gen2 do continue
    for ci in data.constants do
      let n := ci.name
      let kind := match ci with
        | .axiomInfo _ => "axiomInfo" | .defnInfo _ => "defnInfo" | .thmInfo _ => "thmInfo"
        | .opaqueInfo _ => "opaqueInfo" | .quotInfo _ => "quotInfo" | .inductInfo _ => "inductInfo"
        | .ctorInfo _ => "ctorInfo" | .recInfo _ => "recInfo"
      let (former, proof) ← liftTermElabM do pure ((← isPropFormerType ci.type), (← isProp ci.type))
      let rng ← findDeclarationRanges? n
      logInfo m!"AUDIT {n} {kind} former={former} proof={proof} internal={n.isInternalDetail} range={rng.isSome} axioms={← collectAxioms n}"
  -- current-file constants via foldStage2
  let cur := env.constants.foldStage2 (fun acc _ ci => acc.push ci.name) (#[] : Array Name)
  logInfo m!"CURRENT {cur}"
def localThing : Prop := True
run_cmd do
  let env ← getEnv
  let cur := env.constants.foldStage2 (fun acc _ ci => acc.push ci.name) (#[] : Array Name)
  logInfo m!"CURRENT2 {cur}"
#check UniversalLaw.Spec.Field.IsTwo
#check UniversalLaw.Spec.Field.Iface
