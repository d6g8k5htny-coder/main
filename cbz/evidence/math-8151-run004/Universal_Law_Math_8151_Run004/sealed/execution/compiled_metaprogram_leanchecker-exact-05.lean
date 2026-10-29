import LeanChecker
open Lean
/-! Executes the pinned LeanChecker API for exactly one module, not its prefix.
This is ordinary named-module replay against imported dependencies, not --fresh.
The generated copy replaces `ResearchFormalCoreR1.MomentGenerality` with one validated Lean name. -/
unsafe def cbzExactReplay : IO Unit := do
  initSearchPath (← findSysroot)
  IO.println "CBZ_EXACT_REPLAY_START ResearchFormalCoreR1.MomentGenerality"
  replayFromImports `ResearchFormalCoreR1.MomentGenerality
  IO.println "CBZ_EXACT_REPLAY_PASS ResearchFormalCoreR1.MomentGenerality"
#eval cbzExactReplay
