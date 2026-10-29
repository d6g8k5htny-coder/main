import LeanChecker
open Lean
/-! Executes the pinned LeanChecker API for exactly one module, not its prefix.
This is ordinary named-module replay against imported dependencies, not --fresh.
The generated copy replaces `ResearchFormalCoreR1.WeightedLaw` with one validated Lean name. -/
unsafe def cbzExactReplay : IO Unit := do
  initSearchPath (← findSysroot)
  IO.println "CBZ_EXACT_REPLAY_START ResearchFormalCoreR1.WeightedLaw"
  replayFromImports `ResearchFormalCoreR1.WeightedLaw
  IO.println "CBZ_EXACT_REPLAY_PASS ResearchFormalCoreR1.WeightedLaw"
#eval cbzExactReplay
