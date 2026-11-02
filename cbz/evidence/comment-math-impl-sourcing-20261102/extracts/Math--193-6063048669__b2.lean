import PinDisintegration
#print axioms I4PinDisintegration.canonicalKernel
#print axioms I4PinDisintegration.pinJoint
#print axioms I4PinDisintegration.canonicalKernel_markov
#print axioms I4PinDisintegration.pinJoint_probability
#print axioms I4PinDisintegration.canonicalKernel_apply
#print axioms I4PinDisintegration.residual_pin_covariance
#print axioms I4PinDisintegration.residual_independent
#print axioms I4PinDisintegration.canonical_disintegration
#print axioms I4PinDisintegration.canonicalKernel_eq_condKernel_ae
#print axioms I4PinDisintegration.residual
#print axioms I4PinDisintegration.shift

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for suffix in ["I4PinDisintegration.series_family_gaussian",
      "I4PinDisintegration.joint_block_gaussian", "I4PinDisintegration.kernel_as_shift"] do
    let candidates := env.constants.toList.filter fun (name, _) =>
      name.toString.startsWith "_private.PinDisintegration." && name.toString.endsWith suffix
    match candidates with
    | [(name, _)] => elabCommand (← `(#print axioms $(mkIdent name):ident))
    | _ => throwError "Expected exactly one private helper ending in {suffix}"

