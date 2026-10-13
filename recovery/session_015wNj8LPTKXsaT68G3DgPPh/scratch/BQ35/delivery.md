## BQ-35 AGREE_ENGINEERING_SCOPED / RELEASE: BQ-28 snapshot-identity design (BQ35-R3-20261007T1003Z-c8rd)

Dylan Roy — delegated AI architecture/evidence maintenance. Actual performer Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). I did not write the design, the collector or its tests. Credit 0, scientific effect NONE.
- **Confirmation:** [6035761308](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6035761308), window 10:12–11:12Z; acknowledgement [6035668621](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6035668621).
- **Reuse check first:** the historical record 6031662345 holds no result (outcome UNKNOWN), and there is no later equivalent review, so nothing to reconcile.

**Verdict: AGREE_ENGINEERING_SCOPED** on the BQ-28 design [6029127112](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6029127112) (8609 B), against frozen main `cddbb7f6`. It may proceed to a separately assigned two-file author phase. Two non-blocking refinements (R1, R2) and the limits are below. This is not an implementation, a patch or a CI result.

### What I verified, statically and by bounded controls
- **Pins:** `tools/release_snapshot.py` is blob `8cba82a5…`, 14144 B, SHA-256 `9f12296f…`; `tests/test_release_snapshot.py` is blob `3694a85a…`, 12336 B, SHA-256 `0c4ef3b4…`. Both fetched natively and re-hashed; they match the design and the coordinator's current-main check.
- **Caller map is exact.** `pages()` is called only once, at line 209 inside `collect.read`. That is reached by exactly eight `paginated=True` sites, which match the design's table line for line:
  - 225 `/pulls?state=open`;
  - 247 `files`, 248 `reviews`, 249 issue `comments`, 250 inline `comments`;
  - 251 `check-runs`, keyed `check_runs`;
  - 253 `actions/runs`, keyed `workflow_runs`;
  - 259 `rulesets`.

  The four non-paginated reads (219, 223, 239, 252) are correctly outside the identity scope.
- **Placement is right.** `read()` already runs `pages|get → write_json → validate → return`, and any exception becomes a per-resource `SUMMARY.errors` entry and `None`, after which later resources and repositories continue. A post-write validator therefore keeps the raw rows, records `complete:false` and a nonzero CLI exit, and preserves cross-repository progress. A failure inside `pages()` still writes nothing, which is the existing behaviour, and the design correctly doesn't promise retained pages.
- **The PR special case is handled correctly.** Malformed rows still go to the unchanged `pr_number()` per-row error at lines 231–237 and are skipped, while valid later rows proceed. This matches the existing `pr-number` regression fixture, whose only row is malformed. Restricting uniqueness to numbers that pass `pr_number()` is necessary to keep that test green.
- **Key policy:**
  - `type(v) is int and v > 0` excludes `bool`, float and string, matching `pr_number()` and `total_count`.
  - Filenames are compared verbatim, never by blob, which matches the GitHub changed-files row semantics.
  - Keys reset per collection.
  - No first/last-wins and no silent dedup.
- **Bounded local controls.** C8-run in an isolated copy of the frozen collector, Python 3.11.15, `-B -S -I` and `-B -O -S -I`, with identical output in both modes. Probe SHA-256 `b2e85e86…`, fake client and local Git fixture only, no network. They show the current defect that the successor tests must turn RED→GREEN:
  - **C1:** in keyed `workflow_runs`, page 1 holds ids 1–100 and page 2 repeats id 100, with `total_count` 101. `pages()` accepts 101 rows with 100 unique ids, and id 101 is silently missing.
  - **C2:** two open-PR rows both `number: 7` give `complete: true` and `errors: 0`. The collector fetches `/pulls/7` twice, `pr-7/detail.json` is silently overwritten by the second fetch, and `pull_requests` gets two entries.

### Non-blocking refinements for the author phase
- **R1. Enforce the wiring structurally, not only by test enumeration.** Make the identity policy a required argument whenever `paginated=True` (no default; a missing policy raises inside `read()`). A new or bypassed paginated route then fails in the existing mocked-`collect()` harness rather than passing silently. Keep the design's test that records route→policy for exactly the eight suffix families, and that a filename route never uses a SHA key.
- **R2. Make the duplicate-PR consequence explicit in the test and the summary.** Refusing a collection with duplicate valid PR numbers returns `None`, so none of that repository's per-PR reads run, and `open_pr_count` is no longer recorded, though `open-prs.json` keeps all the rows. That is a defensible fail-closed choice, but it is broader than per-row refusal. State it in the guide or summary, and assert in the test that the raw `open-prs.json` still contains both duplicate rows and that the later repository still completes. Whether to refuse only the duplicated numbers instead is the author's or owner's choice; I don't require it.

### Limits, not tested
- Uniqueness is not completeness or atomicity, as the design itself says. Two cases it can't catch:
  - offset pagination can shift so that an omission happens with no duplicate;
  - GitHub's `pulls/{n}/files` stops at 3000 files, so a large PR can return exactly 3000 unique rows and pass.
- Keyed routes store only the extracted list, so the `total_count` that `pages()` compared isn't in the evidence file. That is acknowledged and unchanged.
- **Not done:** no implementation; no full 21-method suite; no real collection, network, credentials, archive or BQ29 duplication; no review of the release guide beyond a reference read.
- The controls are separately attributed C8 observations against the parent, not successor evidence.

**RELEASE:** BQ-35 is complete and released. C8 holds no claim. The integration queue is unchanged.

---
_Generated by [Claude Code](https://claude.ai/code)_
