import Lean
import Gen
open Lean Meta Elab Command

def targets : Array Name := #[`Gen]

run_cmd do
  let env ← getEnv
  let mut count := 0
  for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
    unless targets.contains modName do continue
    for ci in data.constants do
      let n := ci.name
      let kind := match ci with
        | .axiomInfo _ => "axiomInfo" | .defnInfo _ => "defnInfo" | .thmInfo _ => "thmInfo"
        | .opaqueInfo _ => "opaqueInfo" | .quotInfo _ => "quotInfo" | .inductInfo _ => "inductInfo"
        | .ctorInfo _ => "ctorInfo" | .recInfo _ => "recInfo"
      let (propConst, propType) ← liftTermElabM do
        let c := mkConst n (ci.levelParams.map mkLevelParam)
        let a ← isProp c
        let b ← isProp ci.type
        pure (a, b)
      let axs ← collectAxioms n
      let rng ← findDeclarationRanges? n
      let inst ← liftCoreM (isInstance n)
      logInfo m!"AUDIT {n} kind={kind} isProp(const)={propConst} isProp(type)={propType} proj={env.isProjectionFn n} auxRec={isAuxRecursor env n} noConf={isNoConfusion env n} internal={n.isInternalDetail} inst={inst} range={rng.isSome} axioms={axs}"
      count := count + 1
  logInfo m!"AUDIT END {count}"
