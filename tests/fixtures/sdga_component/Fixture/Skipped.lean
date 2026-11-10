module
import Init

namespace Fixture.Skipped

public unsafe def unsafeValue (n : Nat) : Nat := n

-- The compiler emits a genuine partial implementation CI under mkUnsafeRecName;
-- the public opaque wrapper itself is not falsely credited as isPartial.
public partial def partialValue (n : Nat) : Nat := partialValue n

end Fixture.Skipped
