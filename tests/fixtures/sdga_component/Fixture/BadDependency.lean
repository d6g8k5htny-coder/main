module
public import Init

namespace Fixture.BadDependency

-- A genuine dependency axiom, never inserted into a test environment.
public axiom BadProof : True

end Fixture.BadDependency
