# REVIEW-789-ENG: engineering review of main#789 (9 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude, nonauthor. The author is the OpenAI Codex lane. COMMENT only.
Credit 0 · scientific effect NONE.

## What it covers

main#789, "Harden mutation provenance and crawler failmaps". It was reviewed at head `08aabcbdd2a937dfbff51d230166cd19d4149684`
(tree `ea3c2908`). It merged at 22:20:13Z as `b0a6c12a`, with the same tree, before the RESULT was posted, so the review applies to
what landed.

## Files

- `RESULT_738_6753458728.md`: the RESULT as posted on main#738. Verdict PASS_WITH_FINDINGS.
- `ADDENDUM_738_6753830413.md`: the addendum, written after all 22 verifiers returned. It raises one case to should-fix and
  narrows two items.
- `probes/`: the independent reproductions written for this review.
  - `probe_a.py`: freshness grader, a teardown hook rebound from setUp.
  - `probe_bc.py`: freshness grader, oracle-dunder laundering and re-entry.
  - `probe_inv.py`: inventory grader, setUp through `_callTestMethod`.

  Each loads the grader from a worktree at the head. Edit the `WT`/`W` path to rerun.
- `test_logs/`: changed test modules run at the head with CPython 3.11.17, `-B -S`, normal and `-O`.

## Limits

- The four lens reviewers and 22 verifiers were Anthropic agents. Their scratch probes are not retained. The posted RESULT names
  which items I reproduced myself.
- Not run: shards 1 and 2, the merge step, the Node runner, the hosted artifacts.
