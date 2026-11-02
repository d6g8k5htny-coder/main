import SourceContactRank
#print axioms I4ContactRank.frameVector
#print axioms I4ContactRank.contactObservable
#print axioms I4ContactRank.contactObservable_variance_pos

-- Use literal environment Names, including numeric private-name components.
-- This is the same transitive collector used by #print axioms, without guessing parser aliases.
open Lean Elab Command in
run_cmd do
  let env := (← getEnv).setExporting false
  let targets := env.constants.toList.filter fun (n, c) =>
    n.toString.startsWith "_private.SourceContactRank." &&
      (c.isTheorem || n.toString.endsWith ".contactCoefficient" || n.toString.endsWith ".contactW")
  unless targets.length == 22 do
    throwError "expected 14 named private theorems, 2 private definitions and 6 generated proof/equation declarations; got {targets.length}"
  for (n, _) in targets do
    let axioms ← collectAxioms n
    logInfo m!"private declaration {n}: {axioms.qsort Name.lt}"
    unless axioms.all (fun a => a == `propext || a == `Classical.choice || a == `Quot.sound) do
      throwError "nonstandard axiom in {n}"

