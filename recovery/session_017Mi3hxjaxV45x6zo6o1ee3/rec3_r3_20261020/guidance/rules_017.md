# Standing rules learned in session 017 (local notes)
- 2026-10-14 20:33Z: Do NOT post merge-tree/composition data that touches PRs with live integration offers unless dot asks.
  Reason: C8's eligibility line (401/6176304839) counts merge and composition results as content exposure; my
  307/6177718797 triggered dot's reconciliation request on #404 (6177815623). Committed in 404/6177843592.
- dot: "No additional source review or composition experiment is requested" (404/6177815623).
- 2026-10-14 21:25Z (from C8 307/6178816289): on any PR that might later need an integrator, stay metadata-only
  (title, state, head/base, changed-file names, description, comments) unless dot allocates a review.
  Any Anthropic read of the diff, file contents or a merge/composition result ends C8's eligibility on that PR.
  New PRs currently clear of Anthropic content reads include main#408; keep them that way.

## Correction, 2026-10-20 (owner instruction in session, 2026-10-19 about 23:55Z)
- The 2026-10-14 entries above are kept unchanged as history. Both were C8-eligibility courtesies. The 20:33Z
  composition-data commitment (404/6177843592) and the 21:25Z metadata-only note (from C8's 307/6178816289)
  protected C8's chance to integrate PRs that an Anthropic content read would otherwise have excluded it from.
  dot confirmed on 2026-10-14 (307/6177962483) that the restriction is C8-specific, not provider-wide.
- Later I compressed them into "no integration unless dot allocates it" (task #282 and chat, 2026-10-19).
  That is NOT a rule, and it is withdrawn.
- Integration eligibility comes from three sources:
  (1) the actual assignment: an owner instruction such as Dylan's in-session assignment, a coordinator
      activation, or a self-pickup where the rules allow one;
  (2) the current project rules: OP-WORKFLOW-20260930, CONTRIBUTING, and owner decisions recorded on the board.
      OP-WORKFLOW says a self-started pickup does not authorize a protected integration, an occupied role,
      a private-access expansion or a security-sensitive change;
  (3) platform permissions: this session's permission mode, and any denial. A denied step is never retried
      by another tool, routed to another agent, or worked around.
- dot's allocation is needed only where (1) or (2) requires it for that PR.
