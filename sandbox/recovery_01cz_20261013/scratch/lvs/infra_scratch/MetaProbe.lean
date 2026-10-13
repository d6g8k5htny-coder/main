import Lean
open Lean Meta Elab Command

namespace Probe.Reg
def A : Prop := 1 + 1 = 2
def B (n : Nat) : Prop := n = n
def C {X : Type} [Inhabited X] (x : X) : Prop := x = x
end Probe.Reg

namespace Probe.Proofs
theorem a : Probe.Reg.A := rfl
theorem b (n : Nat) : Probe.Reg.B n := rfl
theorem c {X : Type} [Inhabited X] (x : X) : Probe.Reg.C x := rfl
theorem smuggled (h : False) : Probe.Reg.A := h.elim
theorem narrowed (n : Nat) : Probe.Reg.B (n + 1) := rfl
theorem imp : Probe.Reg.A → Probe.Reg.A := fun h => h
theorem unregistered : True := trivial
private theorem priv : True := trivial
namespace Aux
theorem helper : True := trivial
end Aux
def d : Nat := 1
def proofAsDef : 1 = 1 := rfl
axiom ax : False
theorem usesAx : Probe.Reg.A := ax.elim
theorem sorried : Probe.Reg.A := sorry
end Probe.Proofs

def registered : Array Name := #[`Probe.Reg.A, `Probe.Reg.B, `Probe.Reg.C]

run_cmd do
  let env ← getEnv
  let pre : Name := `Probe.Proofs
  let consts := env.constants.foldStage2 (fun acc _ ci => if pre.isPrefixOf ci.name || (privateToUserName? ci.name).any (pre.isPrefixOf ·) then acc.push ci else acc) #[]
  for ci in consts do
    let n := ci.name
    let priv := isPrivateName n
    let range ← findDeclarationRanges? n
    let axioms ← collectAxioms n
    let (head, ok) ← liftTermElabM do
      forallTelescope ci.type fun xs body => do
        let fn := body.getAppFn
        let args := body.getAppArgs
        let okArgs := args.size == xs.size && args.all (fun a => xs.contains a) && xs.all (fun x => args.contains x)
        match fn.constName? with
        | some c => pure (toString c, okArgs)
        | none => pure ("?", false)
    let isP ← liftTermElabM (isProp ci.type)
    let kind := match ci with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "axiom" | .opaqueInfo _ => "opaque" | .inductInfo _ => "ind" | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .quotInfo _ => "quot"
    logInfo m!"{kind} {n} private={priv} range={range.isSome} head={head} okArgs={ok} isProp={isP} internal={n.isInternalDetail} axioms={axioms} unsafe={ci.isUnsafe}"
