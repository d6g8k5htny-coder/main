import Lean
open Lean Meta Elab Command

/-- Modules whose constants are audited; empty means the constants of this file under `auditPrefix`. -/
def auditModules : Array Name := #[]
/-- Namespace of the control declarations audited when `auditModules` is empty. -/
def auditPrefix : Name := `UniversalLaw.Proofs.Control
/-- The registered `def … : Prop` constants of the statement register. -/
def registeredProps : Array Name := #[`UniversalLaw.Spec.Demo.A, `UniversalLaw.Spec.Demo.B]

namespace UniversalLaw.Spec.Demo
def A : Prop := 1 + 1 = 2
def B (n : Nat) : Prop := n + 0 = n
end UniversalLaw.Spec.Demo

namespace UniversalLaw.Proofs.Control
theorem sorried : UniversalLaw.Spec.Demo.A := sorry
axiom hidden : False
theorem usesAxiom : UniversalLaw.Spec.Demo.A := hidden.elim
theorem unregistered : True := trivial
theorem smuggled (h : False) : UniversalLaw.Spec.Demo.A := h.elim
theorem implication : UniversalLaw.Spec.Demo.A → UniversalLaw.Spec.Demo.A := fun h => h
def proofAsDef : (2 : Nat) + 2 = 4 := rfl
opaque opq : Nat
unsafe def unsafeDef : Nat := 0
end UniversalLaw.Proofs.Control

namespace UniversalLaw.Proofs.Control
theorem good : UniversalLaw.Spec.Demo.A := rfl
theorem goodB (n : Nat) : UniversalLaw.Spec.Demo.B n := rfl
theorem narrowed (n : Nat) : UniversalLaw.Spec.Demo.B (n + 1) := rfl
theorem Aux.h : True := trivial
private theorem p : True := trivial
def d : Nat := 0
theorem boundVar (p : Prop) (hp : p) : p := hp
end UniversalLaw.Proofs.Control

/-! Every constant of the audited modules (or, when `auditModules` is empty, every constant of
this file under `auditPrefix`, private names included) is classified by Lean's own `ConstantInfo`
and type, never by source text. `theorem`: a theorem that is not private, not under an `Aux`
namespace and carries a declaration range, whose type after `forallTelescope` is a registered Prop
constant applied to exactly the telescope's variables (reported with that constant). `aux_theorem`:
a private theorem or one under `Aux`. `generated`: a range-less theorem (structure `injEq`,
`sizeOf_spec`, `_proof_*`), refused if its head is a registered Prop. `auxiliary`: definitions,
instances, inductive types, constructors and recursors whose type is not a proposition.
`REFUSED:<why>`: an axiom, opaque, quotient or unsafe constant; a definition whose type is a
proposition (a proof written as `def`); a non-aux theorem whose head is not registered (`type`) or
whose arguments are not exactly its own parameters (`premise`: an extra hypothesis or a narrowed
argument); any transitive axiom outside propext, Classical.choice and Quot.sound (`axioms`).
One `AUDIT <class> <name> [<head>]` line per constant, then `AUDIT END <count> <refused>
<theorems>`; any refusal makes the file fail. -/
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut constants : Array ConstantInfo := #[]
  if auditModules.isEmpty then
    constants := env.constants.foldStage2 (fun acc _ ci =>
      let user := (privateToUserName? ci.name).getD ci.name
      if auditPrefix.isPrefixOf user then acc.push ci else acc) #[]
  else
    let mut found : Array Name := #[]
    for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
      if auditModules.contains modName then
        found := found.push modName
        constants := constants ++ data.constants
    unless found.size == auditModules.size do
      throwError "AUDIT REFUSED: audited module not imported: {auditModules.filter (!found.contains ·)}"
  let mut refused := 0
  let mut theorems := 0
  for ci in constants do
    let n := ci.name
    let axioms ← collectAxioms n
    let bad := axioms.filter (!allowed.contains ·)
    let range ← findDeclarationRanges? n
    let user := (privateToUserName? n).getD n
    let isAux := isPrivateName n || user.components.contains `Aux
    let (head, exact) ← liftTermElabM do
      forallTelescope ci.type fun xs body => do
        let args := body.getAppArgs
        let exact := args.size == xs.size && args.all (fun a => xs.contains a) && xs.all (fun x => args.contains x)
        pure (body.getAppFn.constName?, exact)
    let headStr : String := match head with | some c => toString c | none => "nonconstant"
    let registeredHead := match head with | some c => registeredProps.contains c | none => false
    let isP ← liftTermElabM (isProp ci.type)
    let kind : String :=
      if ci.isUnsafe then "REFUSED:unsafe" else
      match ci with
        | .axiomInfo _ => "REFUSED:axiom"
        | .opaqueInfo _ => "REFUSED:opaque"
        | .quotInfo _ => "REFUSED:quot"
        | .inductInfo _ => "auxiliary"
        | .ctorInfo _ => "auxiliary"
        | .recInfo _ => "auxiliary"
        | .defnInfo _ => if isP then "REFUSED:proof_as_def" else "auxiliary"
        | .thmInfo _ =>
          if isAux then "aux_theorem"
          else if range.isNone then (if registeredHead then s!"REFUSED:generated_proof:{headStr}" else "generated")
          else if registeredHead then (if exact then "theorem" else s!"REFUSED:premise:{headStr}")
          else s!"REFUSED:type:{headStr}"
    let verdict : String :=
      if kind.startsWith "REFUSED" || bad.isEmpty then kind
      else "REFUSED:axioms:" ++ ",".intercalate (bad.map toString).toList
    if verdict.startsWith "REFUSED" then refused := refused + 1
    if verdict == "theorem" then
      theorems := theorems + 1
      logInfo m!"AUDIT theorem {n} {headStr}"
    else
      logInfo m!"AUDIT {verdict} {n}"
  logInfo m!"AUDIT END {constants.size} {refused} {theorems}"
  if refused > 0 then throwError "AUDIT REFUSED {refused} constant(s); see the AUDIT lines"
