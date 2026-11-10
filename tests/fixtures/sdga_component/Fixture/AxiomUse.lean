module
public import Fixture.BadDependency

namespace Fixture.AxiomUse

public def TypeAlias : Prop :=
  (fun (_ : True) => True) Fixture.BadDependency.BadProof

public def typeOnly : TypeAlias := True.intro
public def valueOnly : True := Fixture.BadDependency.BadProof
public opaque opaqueOnly : True := Fixture.BadDependency.BadProof

end Fixture.AxiomUse
