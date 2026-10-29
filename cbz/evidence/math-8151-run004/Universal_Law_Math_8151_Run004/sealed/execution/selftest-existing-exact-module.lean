import LeanChecker
open Lean
/-! Executes the pinned LeanChecker API for exactly one module, not its prefix.
This is ordinary named-module replay against imported dependencies, not --fresh.
The generated copy replaces `UniversalLaw.Side24.Endpoints` with one validated Lean name. -/
unsafe def cbzExactReplay : IO Unit := do
  initSearchPath (← findSysroot)
  IO.println "CBZ_EXACT_REPLAY_START UniversalLaw.Side24.Endpoints"
  replayFromImports `UniversalLaw.Side24.Endpoints
  IO.println "CBZ_EXACT_REPLAY_PASS UniversalLaw.Side24.Endpoints"
#eval cbzExactReplay
