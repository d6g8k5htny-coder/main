module
import Init

/-!
Core-only, genuine compiled populations for the SDGA Task A controls.
The public spellings below are intentional raw Lean Names.
-/

namespace Fixture.Safe

public theorem good : True := True.intro

public def proof : True := True.intro

public def propValue : Prop := True

public def natValue : Nat := 7

private theorem hidden : True := True.intro

-- Module visibility keeps this raw name but omits it from the exported population.
def unexported : Nat := 11

public inductive Switch : Type where
  | off : Switch
  | on : Switch

-- Equation elaboration and the inductive create genuine additional object entries.
-- Controls obtains generated identities from actual ConstantInfo, never a guessed count.
public def flip : Switch → Switch
  | .off => .on
  | .on => .off

end Fixture.Safe

namespace ForeignNamespace

-- Ownership follows the compiled module, not this unrelated namespace prefix.
public theorem visible : True := True.intro

end ForeignNamespace
