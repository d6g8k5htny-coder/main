import LeanChecker
open Lean
/-! Executes the pinned LeanChecker API for exactly one module, not its prefix.
This is ordinary named-module replay against imported dependencies, not --fresh.
The generated copy replaces `ResearchFormalCoreR1.D2Schur` with one validated Lean name. -/
unsafe def cbzExactReplay : IO Unit := do
  initSearchPath (← findSysroot)
  IO.println "CBZ_EXACT_REPLAY_START ResearchFormalCoreR1.D2Schur"
  replayFromImports `ResearchFormalCoreR1.D2Schur
  IO.println "CBZ_EXACT_REPLAY_PASS ResearchFormalCoreR1.D2Schur"
#eval cbzExactReplay
