# Hosted job log excerpts read for MAIN451-R3-INTEGRATION-017-20261019 (read via GitHub MCP get_job_logs, 2026-10-19 ~23:35Z)

## public-intake, Ready-event run 39342133507, job 118365976497 (event pull_request_target, .github/workflows/public-intake.yml)
- GITHUB_TOKEN permissions: Checks write; Contents read; Metadata read; PullRequests read
- actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 with ref efc2fe9023253cacedc8d4f3f0558a37e8652209, sparse-checkout tools/public_intake_check.py, persist-credentials false
- "HEAD is now at efc2fe9 Merge pull request #449 from d6g8k5htny-coder/claude/agents-resumed-checkout-note-20261019"
- env PR_HEAD_SHA: b6f9c104cd4a657888daf754550a8c42b5ca6ec5
- 2026-10-19T23:31:59.1741889Z {"base": "efc2fe9023253cacedc8d4f3f0558a37e8652209", "head": "b6f9c104cd4a657888daf754550a8c42b5ca6ec5", "lane": "maintainer-engineering", "scientific_effect": "NONE"}

## verify, run 39340867577, job 118364431565 (event pull_request, workspace-landing.yml)
- formal outputs: checked_commit 180ff56539f8e882f27c1f82451a32285ac20958, receipt_sha256 85836301187fff81bc019e1bf070d8d765b88ec0a55804a9cc72a5ab15f51b3e, run_id 39340867577, run_attempt 1
- summary 2026-10-19T23:27:25Z: {"checked_commit": "180ff56539f8e882f27c1f82451a32285ac20958", "repository": "d6g8k5htny-coder/main", "run_id": "39340867577", "run_attempt": "1", "receipt_sha256": "85836301...", "conclusion": "success", "scientific_effect": "NONE", "meaning": "Required engineering checks and current-run formal evidence only; not statement alignment or scientific acceptance."}

## landing-checks, run 39340867577, job 118364365281
- step: test "$(git rev-parse HEAD)" = "$GITHUB_SHA"; echo "checked_commit=$GITHUB_SHA" >> "$GITHUB_OUTPUT"; unit controls OK (7, 7, 4, 4 tests)
