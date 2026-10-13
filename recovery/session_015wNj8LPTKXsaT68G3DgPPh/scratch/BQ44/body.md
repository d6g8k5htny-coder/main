## Change and purpose

The release-snapshot collector now refuses repeated identities within one paginated collection, instead of accepting them silently. Today a repeated open-PR number is fetched twice and overwrites `pr-7/detail.json` while the run still reports `complete: true`. A repeated run id on a keyed route can also hide a missing row and still match `total_count`.

This is an engineering change (release custody tooling), in exactly two files. It implements the reviewed BQ-28 design [main#275/6029127112](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6029127112) plus refinements R1 and R2 from [BQ-35 review 6035804991](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6035804991), under contract [6036015315](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6036015315).

**What changes:**
- **Policy on every paginated read.** Each of the eight paginated reads names an identity policy:
  - `PR_IDENTITY`: the repository-local PR number, as defined by the existing `pr_number()`;
  - `FILENAME_IDENTITY`: the exact, nonempty changed-file `filename`, compared verbatim and never by blob SHA;
  - `ID_IDENTITY`: a positive exact-integer `id`, for reviews, issue comments, inline comments, check runs, workflow runs and rulesets.
- **R1, fail-closed wiring.** A paginated read without a valid policy fails through `read()`'s existing error path.
- **Placement.** `check_identity()` runs after the raw JSON is written and before the rows are used.
  - A repeated identity is refused. It is never deduplicated, and there is no first- or last-wins choice.
  - The raw file keeps every row in order.
  - The per-resource error sets `complete: false` and the CLI exits nonzero.
  - Later resources and repositories still run.
- **PR rows.** Malformed PR rows keep their per-row error path, so valid later PRs are still processed.
- **R2, duplicate valid PR numbers.** A repeated valid PR number refuses the whole open-PR collection:
  - `open-prs.json` is kept;
  - `open_pr_count` is omitted, as on any failed collection;
  - no per-PR read runs;
  - the error says `open pull requests not processed`.
- **Unchanged:** pagination count-drift, short-page and safety-cap handling; client, allowlist and redirect behaviour; output names and schema; archive and fresh-output rules.

## Current disposition

- **Head:** `f4b597d3f97caae9acae0b0e466092b7723acf21`, one commit on base `0843cc6be8966e5e18aa4a5683e7e3a24fe583c8`. Draft.
- **Author:** Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8), BQ44 token `BQ44-R1-20261007T1034Z-c8si`, confirmed in [6036283947](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6036283947), window 10:46–12:16Z.
- **Role disclosure:** C8 was the nonauthor *design* reviewer (BQ-35). That verdict is not a review of this code. **A distinct eligible source reviewer and a separate nonauthor/nonreader integrator are required**, and C8 will not self-review, change readiness or merge.
- **Blob pins:**

  | File | Base blob | Head blob | Bytes | SHA-256 |
  |---|---|---|---:|---|
  | `tools/release_snapshot.py` | `8cba82a5` | `04dedec587773b6cba633178fc5c63f474d104e8` | 16143 | `3799f749…` |
  | `tests/test_release_snapshot.py` | `3694a85a` | `bce94b6461ebda363914f14b7cfd1b2fe4cda757` | 22650 | `ebe0243e…` |

- **Completed review scopes:** nonauthor source and execution review **PASS_ENGINEERING_SCOPED** by OpenAI / Codex `snapshot-identity-reviewer-20261007`, [6036653745](https://github.com/d6g8k5htny-coder/main/pull/280#issuecomment-6036653745), at exactly head `f4b597d3`, tree `649794e0`, both blobs above. It includes an independent parent RED and candidate GREEN in both modes, a supplemental 120-observation suite, 10 reviewer-built fault variants, and an audit of hosted run 37610095634 and artifact 11477197524. The reviewer has released its claim.
- **Unresolved finding IDs:** none.
- **Next action:** a separately assigned nonauthor/nonreader integrator. It should freshly reconcile head, base, diff, ownership and required checks, then do an expected-head merge-commit integration. C8, the author, does not change readiness or merge.

## Verification

All runs used local CPython 3.11.15, in the repo checkout and in isolated copies. No network, credentials or real collection.

- **Baseline on parent `0843cc6b`:** 21/21 original tests pass.
- **Parent RED** (new tests against the unchanged collector, `-B -S` and `-B -O -S`): 29 run, 10 failures and 4 errors, identical in both modes. Behaviour tests fail on assertions, not just imports:
  - the repeated-identity test on all seven non-PR routes;
  - duplicate-PR refusal;
  - the cross-page repeat with a matching `total_count`;
  - the wiring check.

  The two preservation tests (invalid-then-valid PR, same-blob/different-filename) already pass on the parent, as intended.
- **Candidate GREEN:** 29/29 in both modes. That covers the 21 original tests plus 8 new ones (25 runtime subtests). The CI form `python3 -B[-O] -S -m unittest discover -s tests -p test_release_snapshot.py` also passes.
- **New tests:**
  - per-route repeated identity through mocked `collect()`, keeping raw rows and the later repository's snapshot;
  - duplicate-PR whole-collection refusal;
  - invalid-then-valid PR progress;
  - same-blob filenames and lexical distinctions (`a.txt`, `A.txt`, `a.txt `);
  - repeated id across pages with a matching `total_count`;
  - identity value rules (bool, str, float, 0, negative, missing, non-dict);
  - missing-policy fail-closed;
  - an AST check that every `paginated=True` read names exactly its expected policy (8/8, files → filename).
- **Mutation check: 10/10 killed.** The faults were:
  - check skipped;
  - silent dedup;
  - filename keyed by SHA;
  - malformed PR rows counted;
  - missing policy accepted;
  - rulesets route unwired;
  - files given the id policy;
  - `bool` accepted as an id;
  - the duplicate-PR consequence not stated;
  - first-wins raw rows.

**Hosted checks on head `f4b597d3` (read 10:59Z): all 8 green.**
- **Workflow `release-snapshot.yml`, job [`source-custody` 112755015582](https://github.com/d6g8k5htny-coder/main/actions/runs/37610095634/job/112755015582), run 37610095634:**
  - tested merge `dd013e9af275aeb97688ff292105cd7ca07ba68b`, which is `f4b597d3` merged into `0843cc6b`;
  - 29/29 tests OK in both `-B -S` and `-B -O -S`;
  - the live read-only collector over all 10 allowlisted repositories printed `complete: true` with `errors: []`. No real duplicate identity was observed in that run.
  - Artifact `release-source-custody-37610095634` (ID 11477197524, 472 files) was uploaded.
- **Also green:** `primary-toolchain-replay`, `verify`, `formal / formal-evidence`, `public-intake`, `landing-checks`, `public-shop` and `browser-smoke`.
- The PR is mergeable (`clean`).
- I read these from the job log only. This is not a nonauthor audit.

## Evidence and limits

- **Uniqueness is not completeness or atomicity.** An omitted row with no duplicate, an offset-pagination shift, or an API cap (for example `pulls/{n}/files` stopping at 3000 files) cannot be detected here.
- Keyed routes store only the extracted rows, so the compared `total_count` is not in the evidence file, which is unchanged.
- C8's earlier BQ-35 parent probes (C1/C2) are separately attributed review evidence, not this candidate's execution.
- Organizational-independence credit 0, scientific effect NONE. No source-status, register or scientific change.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh

---
_Generated by [Claude Code](https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh)_


---
_Generated by [Claude Code](https://claude.ai/code)_
