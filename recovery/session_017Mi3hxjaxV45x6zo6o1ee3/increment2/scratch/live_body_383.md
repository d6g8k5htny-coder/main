## What this adds

The **proposed** shared record format for the retrofit dispatch ([main#275 6018658478](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6018658478)), delivered under contract PICKUP [275/6018831775](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6018831775). Engineering only; scientific effect **NONE**.

| Path | Blob | Size | Purpose |
|---|---|---|---|
| `reviews/retrofit_20261006/CONTRACT.md` | `8ce445c7` | 10071 B | Contract v0.1: file layout, closed shape, field rules, availability-only state, validation, exclusions |
| `reviews/retrofit_20261006/CONTRACT_EXAMPLE.json` | `42e221bc` | 2835 B | One illustrative record pair, not authoritative; shard B1 owns the real SIDE24 records |
| `tools/retrofit_record_check.py` | `b9538201` | 17439 B | Standard-library validator and read-only aggregator |
| `tests/test_retrofit_record_check.py` | `93d4d661` | 21847 B | 21 tests |

**The contract in brief:**
- Each shard writes one `reviews/retrofit_20261006/<SHARD>/RECORDS.json`, with one writer per directory.
- Every object has closed keys, with no `status`, `verdict` or `classification` field.
- A record carries:
  - an exact subject (repository, commit, path, blob), on the pinned baseline (Math- `08f86862`, main `cddbb7f6`) unless it is marked `delta: true`;
  - one of ten evidence axes;
  - a state from `recorded`, `not_recorded`, `unknown` or `not_applicable`;
  - evidence with identities, plus the performer and their exposure. A workflow run keeps its attempt, job, native run head, the commit its job actually checked out (or `null` when not inspected), purpose, native conclusion and expected conclusion. A review keeps its native commit. A comment carries no commit;
  - `independence_credit` fixed at the integer 0, and `alias_of`, which names a canonical record restating the same evidence, so a restated item is never double-counted.
- `state` records availability only. A failed run is still `recorded`, and nothing is collapsed into a pass.
- `scientific_effect: NONE` and `status_authority: false` are mandatory.

**The validator:**
- parses strictly, rejecting duplicate keys and `NaN`/`Infinity` (the R-2 rule);
- enforces the closed shape and enums;
- with `--repo` and `--main-repo`, checks every subject blob and every file-evidence blob against git, requiring a commit object and a blob;
- reports malformed values of any JSON type as violations and keeps reading later files;
- rejects the same input twice and two inputs for one shard;
- with `--aggregate`, prints per-subject (full identity), per-axis states with aliases counted separately, and writes nothing.

## Verification (local, at the head)

- `python3 -B -S -m unittest discover -s tests -p test_retrofit_record_check.py`: 21/21. The same in `-O` mode: 21/21. The tests cover:
  - closed vocabulary and keys, strict parsing, and every JSON type substituted at every path;
  - git object kinds, and blobs for the subject and file evidence, against HEAD (they work in shallow checkouts);
  - workflow-run and review bindings, using native ids from the #343, #183 and #250 histories;
  - canonical aliases, full-identity aggregation, and the real CLI (malformed file then valid file, repeated inputs, same shard).
- Against the originally reviewed validator, the new suite gives 19 failures and 13 errors.
- Sensitivity: 25 single-line mutants at `4837a5b3`, plus 3 for the run-identity rules at `5c2149f7`, are each caught in both modes.
- `python3 -B -S tools/retrofit_record_check.py --repo . --aggregate reviews/retrofit_20261006/CONTRACT_EXAMPLE.json` passes, verifying all four blobs against `08f86862`.

## Current disposition

- Head: `7676d9ce263e003b6544235f6d02bf7e9855236c`, five commits over main `08f86862`. Branch `claude/retrofit-contract-20261006`.
- Author: Anthropic Claude, session `017Mi3hx` (Dylan Roy — delegated AI work). The author will not self-merge.
- Review history (OpenAI `review_retrofit_record_contract_20261006T1500Z`, nonauthor, credit 0):
  - **AMEND** [5430470465](https://github.com/d6g8k5htny-coder/Math-/pull/383#pullrequestreview-5430470465) at `0c9f6887`, findings RC383-01 to 05.
  - Applied in `4837a5b` (with `cd3c575` adding the main#275 acceptance cases [6019011163](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6019011163) and [6019173460](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6019173460)).
  - **PASS_ENGINEERING_SCOPED** [5430740703](https://github.com/d6g8k5htny-coder/Math-/pull/383#pullrequestreview-5430740703) at `4837a5b3` for the delta `0c9f6887..4837a5b3`. It found no unresolved RC383 finding; all five threads are resolved. The earlier AMEND is kept as history.
- **Rollout note from the PASS:** the schema string stays `retrofit-records/v0.1`, but prelanding shadow records that used the earlier three-key `workflow_run` or `github_review` items now fail. Revalidate every shadow against this source rather than assuming compatibility from the string.
  - `5c2149f` splits `run_head_sha` from `checked_commit`, from the case in [6019566432](https://github.com/d6g8k5htny-coder/Math-/pull/383#issuecomment-6019566432); the same reviewer filed **AMEND RC383-06** [5430801433](https://github.com/d6g8k5htny-coder/Math-/pull/383#pullrequestreview-5430801433) at `4837a5b3` on that point, and `7676d9c` adds its exact wording. A second real example is in [6019690040](https://github.com/d6g8k5htny-coder/Math-/pull/383#issuecomment-6019690040).
  - **PASS_ENGINEERING_SCOPED** [5430856539](https://github.com/d6g8k5htny-coder/Math-/pull/383#pullrequestreview-5430856539) at `7676d9ce` for the delta `4837a5b3..7676d9ce`. RC383-06 is resolved, and RC383-01 to 05 keep their earlier resolution. I pushed `7676d9c` after the reviewer's pickup; I disclosed that in [6019659980](https://github.com/d6g8k5htny-coder/Math-/pull/383#issuecomment-6019659980), and the reviewer read the full extra delta.
- CI at `7676d9ce`: **8/8 green**, including `math-downstream-gates`.
- **Integration preflight:** delivered and released by OpenAI / GPT-6 Astra Pro in [6020602096](https://github.com/d6g8k5htny-coder/Math-/pull/383#issuecomment-6020602096). The tested synthetic merge `0a70dfa6` has the same tree as the head (`e0799ef9`). Hosted D5 job `112353071371` ran 251 tests OK in each mode, including all 21 contract tests. All five threads are resolved. That receipt binds test merge `0a70dfa6` over `08f86862`. Main has since advanced to `665f744a` (Math-#382 at `bf0d1b1d`, then Math-#386 at `665f744a`). Both touch only the witness checker and readiness files, which are disjoint from this PR's, so this branch is now 4 commits behind. Under the strict required check, the integrator must bring it up to date with a merge commit and reconcile CI on the new base. The reviewed source bytes do not change; the old CI results do not carry over to the new base. See [6020693346](https://github.com/d6g8k5htny-coder/Math-/pull/383#issuecomment-6020693346).
- **Next:** a separate eligible integrator, who must reconcile the live base, queue and current-commit checks. The author will not self-merge. Unresolved findings: none.
- Dependency: none for Wave 0. Shards may write shadow records against this draft. Wave 1 conversion waits for this contract's review and landing.

## Limits

- This is a format and its checker, not evidence about any theorem.
- Inventory maps (dispatch row I) are out of scope for v0.1.
- The GRAPH join of landed records into the Math-#381 view is a later, separately reviewed change.
- Free-text `notes` and `exposure` are not parsed for meaning.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3

---
_Generated by [Claude Code](https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3)_


---
_Generated by [Claude Code](https://claude.ai/code)_
