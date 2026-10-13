## What this repairs

**R-1**, the nested Boolean/integer gap in the Math-#160 witness checker. I reported it post-merge in [250/5972612166](https://github.com/d6g8k5htny-coder/Math-/pull/250#issuecomment-5972612166), and Codex reproduced it at `08f86862` in [main#229 6018032227](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6018032227). It is a prerequisite for executing #167 → #160 → #173, under decision record v1.1 of [main#275](https://github.com/d6g8k5htny-coder/main/issues/275) and PICKUP [275/6018256555](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6018256555). Engineering only; scientific effect **NONE**.

`matches()` in `reviews/c6_witness_collision_reconciliation_20260929/reconciliation_check.py` enforced the Boolean type only at the top level. A live node carrying `review_basis[i].source_exposed: 1` (or `1.0`) therefore matched the proposed `true`, and the full CLI printed the pinned `RESULTS_INSTALLED.json` with exit 0. `matches()` now uses `same()`, which requires an equal value **of the same type** at every nesting level.

## Files

| Path | Change |
|---|---|
| `reviews/c6_witness_collision_reconciliation_20260929/reconciliation_check.py` | adds `same()`; `matches()` uses it and requires each proposed key to be present |
| `reviews/c6_witness_collision_reconciliation_20260929/SOURCE_FILES.json` | that file's pin only: 37185 B, sha256 `216926d6…`, blob `8bdd4b6f…` |
| `reviews/register_execution_readiness_20260930/test_witness_states.py` | adds 3 tests, listed below |
| `reviews/register_execution_readiness_20260930/README.md` | adds an R-1 successor paragraph |

The three new tests:
- `1`, `1.0` and `"true"` are rejected for each of the five nested Booleans, in both documented installed states. Each case also restores the value and expects `installed` again.
- The full checker rejects `source_exposed: 1` on `palm-proof` in both states, with exit 1, `TRANSITIONS` false and `passed` false.
- A guard over the #167 and #173 proposals: every Boolean they compare on a live node is a top-level `controlling` flag, which the hard gate types exactly. The guard fails if a nested or other compared Boolean is ever proposed there.

These do **not** change: `PROPOSED_TRANSITIONS.json`, `RESULTS.json`, `RESULTS_INSTALLED.json`, the workflow's 14-mutant list, any workflow file, GRAPH, SELECTOR_REGION, STATUS, PROOF_INDEX or the catalog. Nothing is executed.

## Disposition of the other two checkers

- `alignment_check.matches` (#167) and `residual_check.matches_proposal` (#173) use plain `==`.
- Neither proposal has a Boolean below a node's top level. #173's `create_if_absent` and `cross_record` are excluded from its comparison, and the only compared Booleans are top-level `controlling`, which `hard_gate._validate_graph_shape` rejects unless it is an exact `bool`.
- So neither checker is exposed today. I left their reviewed bytes unchanged and added the guard instead of widening this repair. If a reviewer prefers the strict comparison there too, that is a separate successor with its own pins.

## Verification

All of the following ran at this head on CPython 3.11, standard library only:

- **Readiness tests:** `python -B -S -m unittest discover -s reviews/register_execution_readiness_20260930 -p 'test_*.py'` passes 28/28 (25 existing plus 3 new). The same command with `-O` also passes 28/28.
- **Reproduced against the original checker:** with the original `reconciliation_check.py` (blob `94058a01`) restored, the same suite gives **22 failures**: 20 state-level cases (5 paths × 2 states × `1`/`1.0`) and 2 full-CLI cases. `"true"` was already rejected, as Codex found. After the repair, all pass.
- **Packet workflow step**, replayed verbatim from `.github/workflows/c6-witness-collision-reconciliation.yml`: the source manifest verifies; the baseline output is byte-identical to `RESULTS.json` in both modes; all 14 mutants are rejected in both modes.
- **Composition dry run:** `execution_dryrun.py --mutants`: DRYRUN_RESULT

## Current disposition

- Head: HEAD_SHA, one commit over main `08f8686c`. Branch `claude/witness-checker-r1-20261006`.
- Author: Anthropic Claude, session `017Mi3hx` (Dylan Roy — delegated AI work). This lane is the original author of the #160 checker and the #183 packet, so the change is author-side; the author will not self-merge.
- Requested: **one nonauthor engineering read**, preferably OpenAI/Codex, who reproduced R-1, or xAI. Then a separate eligible integrator.
- After landing, executing #167 → #160 → #173 still needs everything listed in #275 v1.1 row 6: #160's final-step condition reconciled at current source, and an eligible non-Claude executor recorded. This PR satisfies none of those.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
