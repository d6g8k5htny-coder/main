import SourceLifetimeField
import SourceLifetimeControls
import SourceSmoothControls
import Lean.Elab.Command
import Lean.Util.CollectAxioms

/-!
Complete module-origin declaration census, including private/generated helpers.
This audit is adapted from the unchanged source SourceSmoothAxiomAudit.
The emitted report is kernel/axiom evidence only after actual successful execution.
-/

#print axioms I4ThetaBridge.sourceSpectralKernel_eq_periodicCovariance
#print axioms I4LifetimeField.actualGaussianFieldInterface
#print axioms I4LifetimeField.actualGaussianFieldInterface_exists

open Lean Elab Command in
run_cmd do
  let env := (← getEnv).setExporting false
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let modules : Array Name := #[
    `GaussianSeriesFoundation, `SpectralWeights, `SourceFourierSeries,
    `SourceC4, `SourceGaussianLaw, `SourceLattice, `SourceKernel,
    `SourceCovariance, `SourceTorus, `SourceTorusC4,
    `SourceSmooth, `SourceSmoothCarrier, `SourceSmoothControls,
    `Lifetime, `SourceThetaBridge, `SourceLifetimeGeometry, `SourceLifetimeField,
    `SourceLifetimeControls]
  liftIO <| IO.eprintln "LIFETIME_AUDIT_PROGRESS START"
  -- Materialize the complete constant list once; coverage is unchanged.
  let constants := env.constants.toList
  let mut allNames : Array Name := #[]
  for modName in modules do
    let some idx := env.getModuleIdx? modName
      | throwError "missing audited module: {modName}"
    let names := (constants.filterMap fun (name, _) =>
      if env.getModuleIdxFor? name == some idx then some name else none).toArray.qsort Name.lt
    if names.isEmpty then
      throwError "empty declaration inventory for audited module: {modName}"
    liftIO <| IO.eprintln s!"LIFETIME_AUDIT_PROGRESS {modName} COUNT {names.size}"
    logInfo m!"LIFETIME_AUDIT_MODULE {modName} COUNT {names.size}"
    for name in names do
      if (env.checked.get.find? name).isNone then
        throwError "missing kernel declaration: {name}"
      let axs ← Lean.collectAxioms name
      logInfo m!"LIFETIME_AUDIT_DECL {name} AXIOMS {axs}"
      for ax in axs do
        unless allowed.contains ax do
          throwError "forbidden transitive axiom {ax} in {name}"
      allNames := allNames.push name
  let required : Array Name := #[
    `I4SmoothCarrier.sampleInclusion_map,
    `I4SmoothCarrier.smoothLaw_isProbabilityMeasure,
    `I4SmoothCarrier.smoothSourceField_finiteVector_hasGaussianLaw,
    `I4SmoothCarrier.smoothSourceField_covariance,
    `UniversalLaw.Spec.Lifetime.GaussianFieldInterface,
    `I4ThetaBridge.theta_summable,
    `I4ThetaBridge.theta_zero_pos,
    `I4ThetaBridge.thetaSum_eq_scaled_cos,
    `I4ThetaBridge.thetaSum_zero_eq,
    `I4ThetaBridge.sourceSpectralKernel_eq_periodicCovariance,
    `I4LifetimeField.coordinates,
    `I4LifetimeField.actualField,
    `I4LifetimeField.actualField_integral_zero,
    `I4LifetimeField.actualField_covariance,
    `I4LifetimeField.actualField_raw_second_moment,
    `I4LifetimeField.actualField_linearCombination_variance,
    `I4LifetimeField.actualField_linearCombination_map,
    `I4LifetimeField.actualGaussianFieldInterface,
    `I4LifetimeField.actualGaussianFieldInterface_exists,
    `I4LifetimeControls.literalCarrier,
    `I4LifetimeControls.literalLaw,
    `I4LifetimeControls.literalField,
    `I4LifetimeControls.dimensionZeroRawVariance,
    `I4LifetimeControls.emptyFamilyLaw,
    `I4LifetimeControls.cancelledLaw,
    `I4LifetimeControls.reinforcedLaw,
    `I4LifetimeControls.shiftedMeanOne]
  for name in required do
    unless allNames.contains name do
      throwError "required source-law declaration omitted from audit: {name}"
  logInfo m!"LIFETIME_AUDIT_STANDARD_ONLY_PASS COUNT {allNames.size}"
