import Lean
import CBZControl_native

/-! CBZ evidence only. The original 31 arithmetic statements are unchanged.
The four witnesses inhabit every admissible finite-domain premise. They do not
identify Gaussian moments, coefficients, persistence bars, or parent theorems. -/
namespace CBZNonvacuity

theorem pairing_domain_inhabited : ∃ q : Nat, q ∈ [0,1,2,3,4,5,6] := ⟨0, by decide⟩
theorem pairing_all_seven : (0:Nat) ∈ [0,1,2,3,4,5,6] ∧ (1:Nat) ∈ [0,1,2,3,4,5,6] ∧ (2:Nat) ∈ [0,1,2,3,4,5,6] ∧ (3:Nat) ∈ [0,1,2,3,4,5,6] ∧ (4:Nat) ∈ [0,1,2,3,4,5,6] ∧ (5:Nat) ∈ [0,1,2,3,4,5,6] ∧ (6:Nat) ∈ [0,1,2,3,4,5,6] := by decide
theorem dimensions_inhabited : ∃ d : Nat, d ∈ [2,3] := ⟨2, by decide⟩
theorem dimensions_both : (2:Nat) ∈ [2,3] ∧ (3:Nat) ∈ [2,3] := by decide

#print axioms pairing_domain_inhabited
#print axioms pairing_all_seven
#print axioms dimensions_inhabited
#print axioms dimensions_both
end CBZNonvacuity

open Lean Meta Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`CBZControl_native]
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut found : Array Name := #[]
  let mut count : Nat := 0
  for (moduleName, data) in env.header.moduleNames.zip env.header.moduleData do
    if modules.contains moduleName then
      found := found.push moduleName
      for ci in data.constants do
        let axioms ← collectAxioms ci.name
        let bad := axioms.filter (!allowed.contains ·)
        unless bad.isEmpty do
          throwError "CBZ_FORBIDDEN {ci.name} {bad}"
        let kind := match ci with
          | .thmInfo _ => "theorem"
          | .axiomInfo _ => "axiom"
          | .defnInfo _ => "definition"
          | _ => "generated"
        if kind == "axiom" then throwError "CBZ_FORBIDDEN project axiom {ci.name}"
        let (statement, premises) ← liftTermElabM do
          let statement := (← ppExpr ci.type).pretty
          let premises ← forallTelescope ci.type fun xs _ => do
            let mut ps : Array Json := #[]
            for x in xs do
              let decl ← x.fvarId!.getDecl
              if ← isProp decl.type then
                ps := ps.push <| Json.mkObj [("name",toJson decl.userName.toString),("type",toJson (← ppExpr decl.type).pretty)]
            pure ps
          pure (statement,premises)
        let row := Json.mkObj [("name",toJson ci.name.toString),("module",toJson moduleName.toString),("kind",toJson kind),("axioms",toJson (axioms.map toString)),("statement",toJson statement),("prop_premises",Json.arr premises)]
        logInfo m!"CBZ_RECORD {row.compress}"
        count := count + 1
  unless found.size == modules.size do
    throwError "CBZ_MISSING_MODULE {modules.filter (!found.contains ·)}"
  let summary := Json.mkObj [("count",toJson count),("modules",toJson (found.map toString))]
  logInfo m!"CBZ_END {summary.compress}"
