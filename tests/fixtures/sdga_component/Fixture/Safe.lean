module
import Init
meta import Lean

/-!
Core-only genuine object populations for the SDGA Task A controls.
The earlier Type-valued Switch fixture is retained as actual compiler-extra
evidence. This Prop-valued fixture still requires two actual constructors and
a generated recursor; absence of compiler extras must be observed natively.
-/

namespace Fixture.Safe

public theorem good : True := True.intro

public def proof : True := True.intro

public def propValue : Prop := True

public def natValue : Nat := 7

private theorem hidden : True := True.intro

-- Ordinary private syntax mangles its name. Create this one checked declaration
-- under an explicitly unexported outer branch so the raw Name remains ordinary.
-- No object data is altered, and no unchecked declaration insertion is used.
set_option debug.skipKernelTC false in
set_option Elab.async false in
run_elab do
  let name := `Fixture.Safe.unexported
  let pending ← (← Lean.getEnv).addConstAsync name .defn (exportedKind? := none)
  Lean.setEnv pending.asyncEnv
  Lean.addDecl (.defnDecl {
    name := name
    levelParams := []
    type := Lean.mkConst `Nat
    value := Lean.mkNatLit 11
    hints := .abbrev
    safety := .safe
  })
  pending.commitCheckEnv (← Lean.getEnv)
  Lean.setEnv pending.mainEnv

public inductive Switch : Prop where
  | off : Switch
  | on : Switch

-- Proof-valued equation elaboration retains actual recursor/auxiliary CIs.
public def flip : Switch → Switch
  | .off => .on
  | .on => .off

end Fixture.Safe

namespace ForeignNamespace

-- Ownership follows the compiled module, not this unrelated namespace prefix.
public theorem visible : True := True.intro

end ForeignNamespace
