import Lean
import Gen
open Lean Meta Elab Command

/-- Modules whose constants are audited; empty means the constants of this file under `auditPrefix`. -/
def auditModules : Array Name := #[`Gen]
/-- Namespace of the control declarations audited when `auditModules` is empty. -/
def auditPrefix : Name := `UniversalLaw.Spec.Control

/-! Every constant of the audited modules (or, when `auditModules` is empty, every constant of
this file under `auditPrefix`) is classified by Lean's own `ConstantInfo`, never by source text:
`structure`; `statement` (a definition whose type is `Prop` or `… → Prop`); `auxiliary` (any
other definition); `generated` (what `structure` and `def` elaboration add: projections,
recursors, `casesOn`, `noConfusion`, `injEq`/`inj`/`sizeOf_spec`, `_sizeOf_*`, `_flat_ctor`,
`match_*`, `_proof_*`; tolerated only when range-less, flagged by the environment, or an
internal name under its structure); or `REFUSED:<why>` for a theorem, axiom, opaque or quotient
constant, an inductive type that is not a structure, a constructor or recursor of a foreign
type, an instance, a definition whose type is a proposition (a proof), or any transitive axiom
outside propext, Classical.choice and Quot.sound. One `AUDIT <class> <name>` line per constant,
then `AUDIT END <count> <refused>`; any refusal makes the file fail. -/
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut constants : Array ConstantInfo := #[]
  if auditModules.isEmpty then
    constants := env.constants.foldStage2 (fun acc _ ci => if auditPrefix.isPrefixOf ci.name then acc.push ci else acc) #[]
  else
    let mut found : Array Name := #[]
    for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
      if auditModules.contains modName then
        found := found.push modName
        constants := constants ++ data.constants
    unless found.size == auditModules.size do
      throwError "AUDIT REFUSED: audited module not imported: {auditModules.filter (!found.contains ·)}"
  let structs : Array Name := constants.filterMap fun ci => match ci with
    | .inductInfo _ => if isStructure env ci.name then some ci.name else none
    | _ => none
  let underStructure (n : Name) : Bool := structs.any fun s => s != n && s.isPrefixOf n
  let generatedTheorem (n : Name) : Bool := structs.any fun s =>
    let c := (getStructureCtor env s).name
    n == c ++ `injEq || n == c ++ `inj || n == c ++ `sizeOf_spec
  let mut refused := 0
  for ci in constants do
    let n := ci.name
    let axioms ← collectAxioms n
    let bad := axioms.filter (!allowed.contains ·)
    let range ← findDeclarationRanges? n
    let inst ← liftCoreM (isInstance n)
    let (statement, proof) ← liftTermElabM do
      pure ((← isPropFormerType ci.type), (← isProp ci.type))
    let kind : String :=
      match ci with
        | .inductInfo _ => if structs.contains n then "structure" else "REFUSED:inductive"
        | .ctorInfo v => if structs.contains v.induct then "generated" else "REFUSED:constructor"
        | .recInfo _ => if underStructure n then "generated" else "REFUSED:recursor"
        | .axiomInfo _ => "REFUSED:axiom"
        | .opaqueInfo _ => "REFUSED:opaque"
        | .quotInfo _ => "REFUSED:quot"
        | .thmInfo _ =>
          if (underStructure n && env.isProjectionFn n) || (range.isNone && (generatedTheorem n || n.isInternalDetail))
          then "generated" else "REFUSED:theorem"
        | .defnInfo _ =>
          if (underStructure n && (env.isProjectionFn n || isAuxRecursor env n || isNoConfusion env n
                || n.isInternalDetail || range.isNone)) || (range.isNone && n.isInternalDetail) then "generated"
          else if inst then "REFUSED:instance"
          else if proof then "REFUSED:proof"
          else if statement then "statement"
          else "auxiliary"
    let verdict : String :=
      if kind.startsWith "REFUSED" || bad.isEmpty then kind
      else "REFUSED:axioms:" ++ ",".intercalate (bad.map toString).toList
    if verdict.startsWith "REFUSED" then refused := refused + 1
    logInfo m!"AUDIT {verdict} {n}"
  logInfo m!"AUDIT END {constants.size} {refused}"
  if refused > 0 then throwError "AUDIT REFUSED {refused} constant(s); see the AUDIT lines"
