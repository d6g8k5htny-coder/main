/- QUARANTINED NEGATIVE FIXTURE: deliberately environment-hacked private body.
   Never imported by production mathematical code or certified as closed. -/
module
import Lean
open Lean Elab Command

run_cmd do
  let name := mkPrivateName (← getEnv) `malformedPrivate
  let decl := Declaration.thmDecl {
    name, levelParams := [], type := mkConst `False, value := mkConst name }
  -- Proper private visibility is essential: exported-only replay must miss
  -- this defect, and therefore be killed by this negative fixture's test.
  liftCoreM <| withOptions (fun opts => debug.skipKernelTC.set opts true) <| Lean.addDecl decl
