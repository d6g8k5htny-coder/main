/- QUARANTINED NEGATIVE FIXTURE: deliberately environment-hacked theorem.
   Never imported by production mathematical code or certified as closed. -/
module
import Lean
open Lean Elab Command

run_cmd do
  let decl := Declaration.thmDecl {
    name := `malformedPublic, levelParams := [], type := mkConst `False,
    value := mkConst `malformedPublic }
  -- Use the public declaration API so the exported module contains only its
  -- axiom presentation and the malformed theorem body lives in private data.
  liftCoreM <| withOptions (fun opts => debug.skipKernelTC.set opts true) <| Lean.addDecl decl
