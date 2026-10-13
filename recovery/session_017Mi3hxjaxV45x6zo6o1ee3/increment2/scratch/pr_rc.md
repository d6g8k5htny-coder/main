## What this adds

The **proposed** shared record format for the retrofit dispatch ([main#275 6018658478](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6018658478)), delivered under contract PICKUP [275/6018831775](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6018831775). Engineering only; scientific effect **NONE**.

| Path | Blob | Size | Purpose |
|---|---|---|---|
| `reviews/retrofit_20261006/CONTRACT.md` | `1a6d7d25` | 5265 B | Contract v0.1: file layout, closed shape, field rules, validation, exclusions |
| `reviews/retrofit_20261006/CONTRACT_EXAMPLE.json` | `42e221bc` | 2835 B | One illustrative record pair, not authoritative; shard B1 owns the real SIDE24 records |
| `tools/retrofit_record_check.py` | `18e2b11d` | 12690 B | Standard-library validator and read-only aggregator |
| `tests/test_retrofit_record_check.py` | `f8422db4` | 7346 B | 7 tests |

**The contract in brief:**
- Each shard writes one `reviews/retrofit_20261006/<SHARD>/RECORDS.json`, with one writer per directory.
- Every object has closed keys, with no `status`, `verdict` or `classification` field.
- A record carries:
  - an exact subject (repository, commit, path, blob), on the pinned baseline (Math- `08f86862`, main `cddbb7f6`) unless it is marked `delta: true`;
  - one of ten evidence axes;
  - a state from `recorded`, `not_recorded`, `unknown` or `not_applicable`;
  - evidence with identities, plus the performer and their exposure;
  - `independence_credit` fixed at the integer 0, and `alias_of` so a restated item is never double-counted.
- `scientific_effect: NONE` and `status_authority: false` are mandatory.

**The validator:**
- parses strictly, rejecting duplicate keys and `NaN`/`Infinity` (the R-2 rule);
- enforces the closed shape and enums;
- with `--repo` and `--main-repo`, checks every subject blob and every file-evidence blob against git;
- with `--aggregate`, prints per-subject, per-axis states with aliases counted separately, and writes nothing.

## Verification (local, at the head)

- `python3 -B -S -m unittest discover -s tests -p test_retrofit_record_check.py`: 7/7. The same in `-O` mode: 7/7. The tests cover:
  - 28 invalid mutations, including `PROVED_REVIEWED` or `CLOSED` used as a state, an extra `status` field, credit 1 or `False`, `status_authority: 0`, a non-baseline commit without `delta`, path traversal, a dangling alias and a `Human` performer;
  - strict parsing;
  - git blob checks for the subject, for file evidence and for an absent path. These run against HEAD, so they work in shallow CI checkouts;
  - the full CLI, including the directory-name mismatch and an unchanged-input check.
- Sensitivity: 7 single-line mutants of the validator are each caught by assertion failures in both modes, with 0 errors. The mutants disable, in turn: the state enum, the credit check, strict JSON, the git check, the closed keys, the baseline check and the strict `status_authority` check.
- `python3 -B -S tools/retrofit_record_check.py --repo . --aggregate reviews/retrofit_20261006/CONTRACT_EXAMPLE.json` passes, verifying all four blobs against `08f86862`.

## Current disposition

- Head: `0c9f68875f3663706f1b19c636cfd9c597651e5b`, one commit over main `08f86862`. Branch `claude/retrofit-contract-20261006`.
- Author: Anthropic Claude, session `017Mi3hx` (Dylan Roy — delegated AI work). The author will not self-merge.
- Requested: **one named nonauthor engineering review** of this exact head, then a separate eligible integrator. Completed reviews: none. Unresolved findings: none.
- Dependency: none for Wave 0. Shards may write shadow records against this draft. Wave 1 conversion waits for this contract's review and landing.

## Limits

- This is a format and its checker, not evidence about any theorem.
- Inventory maps (dispatch row I) are out of scope for v0.1.
- The GRAPH join of landed records into the Math-#381 view is a later, separately reviewed change.
- Free-text `notes` and `exposure` are not parsed for meaning.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
