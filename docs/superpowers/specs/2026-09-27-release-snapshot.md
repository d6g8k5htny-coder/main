# Release snapshot and integration design

Owner: Dylan Roy, explicit autonomous implementation and merge-management request of 2026-09-27. Scientific effect: NONE.

Use the existing main #95 and Math-/formal foundation. Add one read-only, manually reproducible evidence collector, not another formal registry, promotion engine, scheduler or agent loop. It records all ten currently accessible owner repositories, their default commit, every open same-repository PR head, current CI/review evidence, and exact downloadable source snapshots. A snapshot is as-of evidence, never a claim of current tip equality. Archive all tracked blobs at each pinned commit, verify Git blob identities, compute SHA-256 and record gitlinks without traversing them. No credentials enter source archives or reports. Restrict remote reads to the explicit public owner/repository allowlist. API pages must be exhausted or an error recorded; retrieval failure cannot become an empty queue. Snapshot errors produce a nonzero exit and a partial report.

The hosted workflow uses read-only permissions, immutable action refs, no persisted checkout credential and no privileged PR trigger. A separate job installs the existing pinned primary Lean/mathlib environment and replays its unchanged gate at the observed merged Math- commit. Actual execution is the installation evidence. No assertion that Dylan's computer has been installed or that optional research/prover tools were run.

Reconcile overlapping formal proposals by source-bound reviews in their own PRs. Do not modify peer branches, erase historical failures, self-approve or change mathematical verdicts. Merge decisions remain separate, exact-head operations after diff/review/check inspection; source changes stale prior reviews.
