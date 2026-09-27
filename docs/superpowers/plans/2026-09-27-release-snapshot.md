# Release snapshot implementation plan

Goal: reproducible, read-only repository/PR custody and primary toolchain replay.
Architecture: one stdlib collector plus isolated GitHub Actions jobs; existing formal gates retain their scoped evidence roles. No new scientific-status database.
Tech stack: Python 3, Git, GitHub REST GET, existing pinned Lean action.
Spec: ../specs/2026-09-27-release-snapshot.md

1. Write tests for immutable SHA/repo validation, complete pagination, unsafe paths, actual Git blob archive identity, tampering, gitlink exclusion, authenticated redirect refusal and nonempty output discipline. Observe RED, then implement tools/release_snapshot.py and run both Python modes.
2. Add .github/workflows/release-snapshot.yml: tests, snapshot, primary pinned replay, always-save evidence. Record exact runtime identities and failure limitations.
3. Create an isolated integration branch and PR; inspect hosted results. Download artifacts into the active container and verify archive/log digests. Use snapshots to inspect open PR diffs and current reviews, then merge only eligible changes using expected_head_sha.
4. Record every action/blocker in existing main #95 and per-PR conversations; publish permanent agent entry pointers without claiming acknowledgments not observed.

Review focus: pagination truncation, fork/URL substitution, Git attribute exclusions, source drift, unmaterialized gitlinks, missing API access, premature merge eligibility. Each must fail closed or remain explicitly incomplete. Owner preapproval permits inline execution; no subagent independence is represented.
