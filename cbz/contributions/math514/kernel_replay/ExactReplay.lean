/-
CBZ exact-module scheduling adapter for Lean 4.34.1.
The kernel operation is the installed Apache-2.0 LeanChecker implementation,
Copyright (c) 2023 Kim Morrison; authors Kim Morrison, Sebastian Ullrich.
No replay logic is copied here. See SPEC.md and upstream provenance in reports.
-/
import LeanChecker

open Lean

/-- Invoked by the strict Python CLI in a fresh, resource-bounded process.
    Importing LeanChecker imports the checker implementation, never the target. -/
unsafe def cbzExactReplay : IO Unit := do
  let some raw ← IO.getEnv "CBZ_KERNEL_REPLAY_MODULE"
    | throw <| IO.userError "missing exact-module request"
  let fields := raw.splitOn "."
  unless !raw.isEmpty && fields.all (fun s => !s.isEmpty &&
      s.toList.all (fun c => c.isAlphanum || c == '_' || c == '\'')) do
    throw <| IO.userError "invalid exact module name"
  let target := raw.toName
  if target.isAnonymous then throw <| IO.userError "anonymous module forbidden"
  initSearchPath (← findSysroot)
  let actual ← IO.FS.realPath (← findOLean target)
  let some expected ← IO.getEnv "CBZ_KERNEL_REPLAY_OLEAN"
    | throw <| IO.userError "missing expected olean binding"
  unless actual.toString == expected do
    throw <| IO.userError s!"resolved olean mismatch: {actual} != {expected}"
  replayFromImports target
  IO.println s!"CBZ_KERNEL_REPLAY_OK {raw}"

-- Deliberately evaluated as a file, not `lean --run`: LeanChecker already
-- defines root `main`, whose prefix-expanding CLI must never be invoked here.
#eval cbzExactReplay
