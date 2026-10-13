import ResearchFormalCoreR1
import Lean
open Lean

namespace ResearchFormalCoreR1
theorem _unregistered : True := True.intro
private theorem hiddenPriv : (1 : Nat) = 1 := rfl
theorem withMatch (n : Nat) : n = n := by
  cases n with
  | zero => rfl
  | succ k => rfl
def withEqns : Nat → Nat
  | 0 => 0
  | n+1 => withEqns n
end ResearchFormalCoreR1

#eval show CoreM Unit from do
  let env ← getEnv
  let pkg := `ResearchFormalCoreR1
  let header := env.header
  let names := header.moduleNames
  let mut withR := 0
  let mut withoutR := 0
  for h : idx in [0:names.size] do
    let m := names[idx]
    if pkg.isPrefixOf m then
      for c in header.moduleData[idx]!.constNames do
        let r ← findDeclarationRanges? c
        let kind := match env.find? c with
          | some (.thmInfo _) => "thm" | some (.defnInfo _) => "def" | some (.inductInfo _) => "ind"
          | some (.ctorInfo _) => "ctor" | some (.recInfo _) => "rec" | some (.opaqueInfo _) => "opq" | _ => "other"
        if r.isSome then withR := withR + 1 else withoutR := withoutR + 1
        if r.isNone then IO.println s!"NORANGE|{kind}|{c}"
  IO.println s!"package with-range={withR} without-range={withoutR}"
  for (c, ci) in env.constants.map₂.toList do
    let r ← findDeclarationRanges? c
    let kind := match ci with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | _ => "other"
    IO.println s!"LOCAL|{kind}|range={r.isSome}|{c}"
