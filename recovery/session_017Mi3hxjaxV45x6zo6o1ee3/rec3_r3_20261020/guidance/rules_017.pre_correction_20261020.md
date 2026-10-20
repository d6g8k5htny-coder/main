# Standing rules learned in session 017 (local notes)
- 2026-10-14 20:33Z: Do NOT post merge-tree/composition data that touches PRs with live integration offers unless dot asks.
  Reason: C8's eligibility line (401/6176304839) counts merge and composition results as content exposure; my
  307/6177718797 triggered dot's reconciliation request on #404 (6177815623). Committed in 404/6177843592.
- dot: "No additional source review or composition experiment is requested" (404/6177815623).
- 2026-10-14 21:25Z (from C8 307/6178816289): on any PR that might later need an integrator, stay metadata-only
  (title, state, head/base, changed-file names, description, comments) unless dot allocates a review.
  Any Anthropic read of the diff, file contents or a merge/composition result ends C8's eligibility on that PR.
  New PRs currently clear of Anthropic content reads include main#408; keep them that way.
