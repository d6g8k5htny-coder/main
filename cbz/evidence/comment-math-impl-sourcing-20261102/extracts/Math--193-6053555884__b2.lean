import SourceJetLaw

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for suffix in ["I4JetLaw.jet_eq_series_ae", "I4JetLaw.jet_aemeasurable"] do
    let candidates := env.constants.toList.filter fun (name, _) =>
      name.toString.startsWith "_private.SourceJetLaw." && name.toString.endsWith suffix
    match candidates with
    | [(name, _)] => elabCommand (← `(#print axioms $(mkIdent name):ident))
    | _ => throwError "Expected exactly one private helper ending in {suffix}"
#print axioms I4JetLaw.sourceJet_map_gaussian
#print axioms I4JetLaw.sourceJet_linearCombination_hasGaussianLaw
#print axioms I4JetLaw.sourceJet_finiteVector_hasGaussianLaw
#print axioms I4JetLaw.sourceJet_integral_zero
#print axioms I4JetLaw.sourceJet_covariance
#print axioms I4JetLaw.sourceJet_variance

