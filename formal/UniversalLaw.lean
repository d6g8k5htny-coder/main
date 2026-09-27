import UniversalLaw.Side24.Ledger
import UniversalLaw.Side24.ConeMoments
import UniversalLaw.Side24.Endpoints

/-!
# Universal Law — formal layer (Layer 1)

Root module of the dependency-free Lean 4 library. Each imported module is
kernel-checked by `lake build`; `UniversalLaw/Audit.lean` prints the axioms
each registered declaration depends on, and `tools/formal_gate_check.py`
refuses any declaration that is registered as `kernel-checked` but uses an
axiom outside the allow-list, `sorryAx`, or `Lean.ofReduceBool`.
-/
