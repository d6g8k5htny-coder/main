import SourceSmoothControls
import Lean

#print axioms I4Smooth.sourceField_smooth_ae
#print axioms I4SmoothCarrier.sampleInclusion_map
#print axioms I4SmoothCarrier.smoothLaw_isProbabilityMeasure
#print axioms I4SmoothCarrier.smoothSourceField_finiteVector_hasGaussianLaw
#print axioms I4SmoothCarrier.smoothTorusSourceField_smooth

-- Actual imported module identities select every environment constant, including
-- generated declarations and private helpers. Endpoint lines do not define coverage.
open Lean Elab Command in
run_cmd do
  let env := (← getEnv).setExporting false
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let modules : Array Name := #[
    `GaussianSeriesFoundation, `SpectralWeights, `SourceFourierSeries,
    `SourceC4, `SourceGaussianLaw, `SourceLattice, `SourceKernel,
    `SourceCovariance, `SourceTorus, `SourceTorusC4,
    `SourceSmooth, `SourceSmoothCarrier, `SourceSmoothControls]
  let some carrierIdx := env.getModuleIdx? `SourceSmoothCarrier
    | throwError "missing carrier module for private-helper guards"
  let mut allNames : Array Name := #[]
  for modName in modules do
    let some idx := env.getModuleIdx? modName
      | throwError "missing audited module: {modName}"
    let names := (env.constants.toList.filterMap fun (name, _) =>
      if env.getModuleIdxFor? name == some idx then some name else none).toArray.qsort Name.lt
    if names.isEmpty then
      throwError "empty declaration inventory for audited module: {modName}"
    logInfo m!"SMOOTH_AUDIT_MODULE {modName} COUNT {names.size}"
    for name in names do
      if (env.checked.get.find? name).isNone then
        throwError "missing kernel declaration: {name}"
      let axs ← Lean.collectAxioms name
      logInfo m!"SMOOTH_AUDIT_DECL {name} AXIOMS {axs}"
      for ax in axs do
        unless allowed.contains ax do
          throwError "forbidden transitive axiom {ax} in {name}"
      allNames := allNames.push name
  let required : Array Name := #[
    `I4Foundation.coefficientLaw,
    `I4SourceGaussian.source_finiteVector_hasGaussianLaw,
    `I4SourceCovariance.sourceField_variance_one,
    `I4Smooth.goodSample, `I4Smooth.goodSample_ae,
    `I4Smooth.sourceField_smooth_of_goodSample,
    `I4Smooth.goodSample_finite_change_iff,
    `I4SmoothCarrier.goodSet_measurable, `I4SmoothCarrier.sampleInclusion_map,
    `I4SmoothCarrier.smoothLaw_isProbabilityMeasure,
    `I4SmoothCarrier.observation_map_eq,
    `I4SmoothCarrier.smoothSourceField_finiteVector_hasGaussianLaw,
    `I4SmoothCarrier.smoothSourceField_covariance,
    `I4SmoothCarrier.smoothTorusSourceField_smooth,
    `I4SmoothControls.old_four_does_not_fill_all_orders,
    `I4SmoothControls.field_variance_one]
  for name in required do
    unless allNames.contains name do
      throwError "required source-law declaration omitted from audit: {name}"
  let privateHelpers : Array Name := #[
    `I4SmoothCarrier.orderAbsNN, `I4SmoothCarrier.orderAbsTotal,
    `I4SmoothCarrier.orderAbsTotal_ne_top_iff,
    `I4SmoothCarrier.orderAbsTotal_measurable]
  for helper in privateHelpers do
    unless allNames.any (fun name =>
        env.getModuleIdxFor? name == some carrierIdx &&
        Lean.privateToUserName? name == some helper) do
      throwError "required carrier-private helper omitted from audit: {helper}"
  logInfo m!"SMOOTH_AUDIT_STANDARD_ONLY_PASS COUNT {allNames.size}"
