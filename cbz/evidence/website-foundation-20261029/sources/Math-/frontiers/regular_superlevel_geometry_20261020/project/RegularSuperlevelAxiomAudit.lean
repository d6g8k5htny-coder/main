import RegularSuperlevelLiteral
import Lean.Elab.Command
import Lean.Util.CollectAxioms

/-! Complete module-origin census for C297 and its exact Lifetime/peer-chart inputs.
Includes generated and private declarations. Only the standard axioms are allowed.
Command success is scoped execution evidence, not scientific promotion. -/

#print axioms TorusRegularSuperlevel.closed_superlevel_regular_basis
#print axioms TorusRegularSuperlevel.closed_superlevel_path_connected_basis
#print axioms TorusRegularSuperlevel.closed_superlevel_interior_basis
#check @TorusRegularSuperlevel.closed_superlevel_regular_basis
#check @TorusRegularSuperlevel.closed_superlevel_path_connected_basis
#check @TorusRegularSuperlevel.closed_superlevel_interior_basis

open Lean Elab Command in
run_cmd do
  let env := (← getEnv).setExporting false
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let modules : Array Name := #[`Lifetime, `RegularPointChart,
    `RegularSuperlevel, `RegularSuperlevelLiteral]
  let constants := env.constants.toList
  let mut allNames : Array Name := #[]
  for modName in modules do
    let some idx := env.getModuleIdx? modName
      | throwError "missing audited module: {modName}"
    let names := (constants.filterMap fun (name, _) =>
      if env.getModuleIdxFor? name == some idx then some name else none).toArray.qsort Name.lt
    if names.isEmpty then throwError "empty declaration inventory: {modName}"
    liftIO <| IO.eprintln s!"SUPERLEVEL_AUDIT_PROGRESS {modName} COUNT {names.size}"
    logInfo m!"SUPERLEVEL_AUDIT_MODULE {modName} COUNT {names.size}"
    for name in names do
      if (env.checked.get.find? name).isNone then
        throwError "missing kernel declaration: {name}"
      let axs ← Lean.collectAxioms name
      logInfo m!"SUPERLEVEL_AUDIT_DECL {name} AXIOMS {axs}"
      for ax in axs do
        unless allowed.contains ax do
          throwError "forbidden transitive axiom {ax} in {name}"
      allNames := allNames.push name
  let required : Array Name := #[
    `UniversalLaw.Spec.Lifetime.IsSmooth,
    `UniversalLaw.Spec.Lifetime.IsCriticalPoint,
    `UniversalLaw.Spec.Lifetime.regularPointDerivative_range_eq_top,
    `UniversalLaw.Spec.Lifetime.regularPointHasStrictFDerivAt,
    `UniversalLaw.Spec.Lifetime.regularPointChart,
    `UniversalLaw.Spec.Lifetime.regularPointChart_fst,
    `UniversalLaw.Spec.Lifetime.regularPointChart_self,
    `UniversalLaw.Spec.Lifetime.mem_regularPointChart_source,
    `UniversalLaw.Spec.Lifetime.mem_regularPointChart_target,
    `TorusRegularSuperlevel.path_connected_basis_of_openEmbedding,
    `TorusRegularSuperlevel.path_connected_basis_of_open,
    `TorusRegularSuperlevel.relative_chart_homeomorph,
    `TorusRegularSuperlevel.path_connected_basis_of_relative_chart,
    `TorusRegularSuperlevel.closed_superlevel_basis_of_chart,
    `TorusRegularSuperlevel.closed_superlevel_interior_basis,
    `TorusRegularSuperlevel.closed_superlevel_regular_basis,
    `TorusRegularSuperlevel.closed_superlevel_path_connected_basis]
  for name in required do
    unless allNames.contains name do
      throwError "required declaration omitted from audit: {name}"
  logInfo m!"SUPERLEVEL_AUDIT_STANDARD_ONLY_PASS COUNT {allNames.size}"
