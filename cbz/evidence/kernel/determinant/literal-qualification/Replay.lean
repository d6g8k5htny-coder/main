import ResearchFormalCoreR1
import Lean
import Lean.Replay
open Lean

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


def _inv.checkProofClosure (modules : Array Name) : CoreM Unit := do
  if modules.isEmpty then throwError "candidate module list is empty"
  if modules.toList.eraseDups.length != modules.size then
    throwError "candidate module list contains duplicates"
  let env := (← getEnv).setExporting false
  let mut seeds : Array Name := #[]
  for modName in modules do
    let some idx := env.getModuleIdx? modName
      | throwError "missing candidate module: {modName}"
    let names := (env.constants.toList.filterMap fun (name, _) =>
      if env.getModuleIdxFor? name == some idx then some name else none).toArray.qsort Name.lt
    if names.isEmpty then throwError "empty candidate module: {modName}"
    IO.println s!"KERNEL_CLOSURE_MODULE {modName} COUNT {names.size}"
    for name in names do
      if (env.checked.get.find? name).isNone then
        throwError "candidate has no kernel declaration: {name}"
      IO.println s!"KERNEL_CLOSURE_SEED {modName} {name}"
      seeds := seeds.push name
  let (_, state) := (seeds.forM (_inv.collect env)).run {}
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  for ax in state.axioms.toArray do
    unless allowed.contains ax do
      throwError "forbidden proof-closure axiom: {ax}"
  let mut selected : Std.HashMap Name ConstantInfo := {}
  for name in state.visited.toArray do
    let some ci := env.find? name
      | throwError "missing transitive proof dependency: {name}"
    if ci.isUnsafe || ci.isPartial then
      throwError "proof closure contains unsafe/partial constant: {name}"
    selected := selected.insert name ci
  IO.println s!"KERNEL_CLOSURE_REPLAY_BEGIN SEEDS {seeds.size} CLOSURE {selected.size}"
  let fresh ← mkEmptyEnvironment 0
  let checked ← fresh.toKernelEnv.replay selected
  for name in seeds do
    if (checked.find? name).isNone then
      throwError "kernel replay omitted candidate constant: {name}"
  let axioms := (state.axioms.toArray.map toString).qsort (· < ·)
  IO.println s!"KERNEL_CLOSURE_STANDARD_ONLY_PASS SEEDS {seeds.size} CLOSURE {selected.size} AXIOMS [{", ".intercalate axioms.toList}]"

#eval _inv.checkProofClosure #[`ResearchFormalCoreR1.DeterminantMoment, `ResearchFormalCoreR1.GaussianJetMoments, `ResearchFormalCoreR1.AlgebraV2, `ResearchFormalCoreR1.EndpointGaussianMoments, `ResearchFormalCoreR1.EndpointNormalizerBridge, `ResearchFormalCoreR1.GaussianRegressionKernel, `ResearchFormalCoreR1.GaussianCoupling, `ResearchFormalCoreR1.GaussianScalarMoments, `ResearchFormalCoreR1.GaussianNormalizer, `ResearchFormalCoreR1.NormalizerFloor, `ResearchFormalCoreR1.GaussianNormalizerLimit, `ResearchFormalCoreR1.ProbabilityCompanionsV2, `ResearchFormalCoreR1.PalmReweighting]
