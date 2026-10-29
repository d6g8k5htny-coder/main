import LeanChecker
open Lean
/-! Executes the pinned LeanChecker API for exactly one module, not its prefix.
This is ordinary named-module replay against imported dependencies, not --fresh.
The generated copy replaces `CBZ_EXACT_MODULE` with one validated Lean name. -/
unsafe def cbzExactReplay : IO Unit := do
  initSearchPath (← findSysroot)
  IO.println "CBZ_EXACT_REPLAY_START CBZ_EXACT_MODULE"
  replayFromImports `CBZ_EXACT_MODULE
  IO.println "CBZ_EXACT_REPLAY_PASS CBZ_EXACT_MODULE"
#eval cbzExactReplay
