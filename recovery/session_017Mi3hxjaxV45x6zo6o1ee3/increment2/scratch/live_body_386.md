## What this repairs

**R-2**, which I recorded as an open author-side finding in [382/6018648978](https://github.com/d6g8k5htny-coder/Math-/pull/382#issuecomment-6018648978), after the R-1 repair [Math-#382](https://github.com/d6g8k5htny-coder/Math-/pull/382) landed at `bf0d1b1d`. Engineering only; scientific effect **NONE**.

`reviews/c6_witness_collision_reconciliation_20260929/reconciliation_check.py` parsed GRAPH, SELECTOR_REGION, the proposals and the D5 transitions with plain `json.loads`. That keeps the last of any duplicate key and accepts `NaN`, `Infinity` and `-Infinity`. Run against the R-1 checker, each of these inputs reached the pinned installed output with exit 0:
- a GRAPH text with a duplicate `classification` on the witness node (first `REFUTED`, then the real value);
- a SELECTOR_REGION text with a duplicate `regions` key;
- a non-finite constant in either file.

All ten JSON reads now go through one strict loader, `load_json`, with the rule of `hard_gate.load_json_strict`. A refused input prints `{"object": …, "passed": false, "input_error": …}` and exits 1.

## Files

| Path | Change |
|---|---|
| `reviews/c6_witness_collision_reconciliation_20260929/reconciliation_check.py` | adds `InputError` and `load_json`; all ten reads use it; `main()` reports a refusal as `passed: false` with an `input_error` |
| `reviews/c6_witness_collision_reconciliation_20260929/SOURCE_FILES.json` | that file's pin only: 37996 B, sha256 `f13c63fd…`, blob `3e95763f…` |
| `reviews/register_execution_readiness_20260930/test_witness_states.py` | `selector_fixture()` helper (the R-1 inline selector setup, unchanged in behavior), `run_text()`, and two tests (below) |
| `reviews/register_execution_readiness_20260930/README.md` | adds an R-2 successor paragraph |

The new tests:
- **Full CLI on five malformed inputs:** a duplicate GRAPH key, a duplicate SELECTOR key, `NaN` and `-Infinity` in GRAPH, and `Infinity` in SELECTOR. Each must exit 1 with `passed: false`, the right `input_error` reason, and not the pinned installed bytes. The same texts without the injection still give the pinned output.
- **Single parse site guard:** `json.loads(` appears exactly once in the checker, inside `load_json` with both hooks. GRAPH is read in several checks, so the full-CLI cases alone cannot see one loose call site; a mutant that loosens only the OBLIGATION read survives them and is caught by this guard.

These do **not** change: `PROPOSED_TRANSITIONS.json`, `RESULTS.json`, `RESULTS_INSTALLED.json`, the workflow's 14-mutant list, any workflow file, GRAPH, SELECTOR_REGION, STATUS, PROOF_INDEX or the catalog. Nothing is executed.

**Not taken here:** `hard_gate.load_selector_region` (hard_gate.py line 522) also parses SELECTOR_REGION with plain `json.loads`. It belongs to the downstream-gate package and its pinned manifest, so it is left to that package's owner.

## Verification

All of the following ran at this head on CPython 3.11, standard library only:

- **Readiness tests:** `python -B -S -m unittest discover -s reviews/register_execution_readiness_20260930 -p 'test_*.py'` passes 30/30 (28 existing plus 2 new). The same command with `-O` also passes 30/30.
- **Packet workflow step**, replayed verbatim from `.github/workflows/c6-witness-collision-reconciliation.yml`: the source manifest verifies with the new pin; the baseline output is byte-identical to `RESULTS.json` in both modes; all 14 mutants are rejected in both modes.
- **Composition dry run:** `execution_dryrun.py --mutants` gives `passed: true` with mutants run, 88 nodes / 131 edges, `gate_ok` true, three holds cleared, no new hold, and nothing written to the repository.
- `git diff --check` over the commit is clean.

- **Reproduced against the R-1 checker:** with the landed checker (`8bdd4b6f`) restored, all five full-CLI cases fail by assertion (it exits 0 on every one), and so does the parse-site guard (the checker has ten `json.loads(` calls).
- **Sensitivity:** four single-line mutants are each caught in both modes, with assertion failures only:
  - duplicate-key rule disabled;
  - constant rule disabled;
  - refusal exits 0;
  - the OBLIGATION read loosened (caught by the guard).

## Current disposition

- Head: `9938b5921ee6b800287b3b0eadad6c4e7bc2e7f0`, one commit over main `bf0d1b1d` (Math-#382). Branch `claude/witness-checker-r2-20261006`.
- Author: Anthropic Claude, session `017Mi3hx` (Dylan Roy — delegated AI work). This lane is the original author of this checker, so the change is author-side; the author will not self-merge.
- Completed review: **PASS_ENGINEERING_SCOPED** by OpenAI / GPT-5.6 Sol at `9938b592` ([5431531922](https://github.com/d6g8k5htny-coder/Math-/pull/386#pullrequestreview-5431531922)). It is nonauthor, with credit 0 and no amendment. It confirmed in hosted runs: readiness run 37495209447 (job 112377866784) and witness run 37495209511 (job 112377867222). A crossed later pickup released without a duplicate verdict, adding hosted evidence ([6020702339](https://github.com/d6g8k5htny-coder/Math-/pull/386#issuecomment-6020702339)). That evidence covers the general run 37495210200: its aggregate checkout and both producers' `checked_commit` are the test merge `daa2db7c`, with parents `bf0d1b1d` and `9938b592`. It also shows 30/30 readiness methods in each mode, and a 14+15+15 mutant composition with nothing written.
- Known limit, noted by the reviewer: malformed JSON syntax that is neither a duplicate key nor a non-finite constant still fails closed (exit 1), but as a `JSONDecodeError` traceback rather than a structured `input_error`. This is outside R-2's stated scope.
- CI at this head: **5/5 green** (16:32Z), including `math-downstream-gates`. Mergeable: clean. The branch is current with main `bf0d1b1d`.
- Next: a separate eligible integrator. The author will not self-merge.
- There are no ordering dependencies on Math-#381, #383 or main#276. Executing #167 → #160 → #173 still needs #160's final-step condition reconciled at current source and an eligible non-Claude executor recorded. This PR satisfies neither.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3

---
_Generated by [Claude Code](https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3)_


---
_Generated by [Claude Code](https://claude.ai/code)_
