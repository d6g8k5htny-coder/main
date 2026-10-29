import LeanChecker
open Lean
/-! Executes the pinned LeanChecker API for exactly one module, not its prefix.
This is ordinary named-module replay against imported dependencies, not --fresh.
The generated copy replaces `CBZMissingExactModule` with one validated Lean name. -/
unsafe def cbzExactReplay : IO Unit := do
  initSearchPath (← findSysroot)
  IO.println "CBZ_EXACT_REPLAY_START CBZMissingExactModule"
  replayFromImports `CBZMissingExactModule
  IO.println "CBZ_EXACT_REPLAY_PASS CBZMissingExactModule"
#eval cbzExactReplay
