**This makes the C8 lifetime-difference wrapper and the d = 3 Theorem A constants wrapper check why each checker child exits, not just whether it exits.** Before, both accepted their baselines on exit status 0 alone and each mutant on exit status 1 alone, with the children's output captured and thrown away. So an unrelated crash, a silent exit 1 or a failure for another reason counted as a rejection. Each now requires the exact verdict measured from its unchanged checker, as batches 1 ([#519](https://github.com/d6g8k5htny-coder/Math-/pull/519)), 2 ([#520](https://github.com/d6g8k5htny-coder/Math-/pull/520)) and 3 ([#544](https://github.com/d6g8k5htny-coder/Math-/pull/544)) do for the other five wrappers; this is the last batch of the seven. Packet bytes are unchanged. Scientific effect NONE.

## Current disposition — @@DATE@@

- **Head:** `@@HEAD@@`, one commit on Math- `main` `19a55750`, the merge of batch 3. Draft.
- **Author:** Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8), stage `C8-WRAPPERS-REPAIR-C8-20261026`, batch 4 ([#322/6545025569](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-6545025569)), under pickup [#322/6422210000](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-6422210000). Dylan Roy — delegated AI work. Dylan approved draft PRs for the seven `c8-*` wrappers in my session on 26 October.
- **Diagnosis credited:** 01Cz's ledger [#322/6388271177](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-6388271177), which classifies both wrappers as `status_only`; GPT-6's read-only audits [#322/6516675793](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-6516675793) (d3) and [#322/6516944036](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-6516944036) (lifetime, finding `MTW-C8-LIFETIME-WRAPPER-01`), which reproduced the false acceptance in disposable fixtures and prescribed extending this helper.
- **Reviews:** none yet.
- **Open findings:** none.
- **Next:** one nonauthor engineering review at the exact head (two workflows, the shared helper, its reference file and its test file), fresh hosted checks at this head, then a separate integrator. Any family may review. C8 takes neither seat.
- **Hosted at this head:** pending.

## Change (six files, one commit)

| File | Change |
|---|---|
| `tools/check_mode_replay.py` | Two inventory rows, in `PLANNED`. A `SERIAL` table restates the plan the planar Theorem A wrapper shares with these two (normal `--check-full`, optimized `--check`, then each mutant in normal mode, one child at a time) with each wrapper's timeouts and `--procs`: planar's fixed `2`, and for these two the wrapper's own `str(max(2, os.cpu_count() or 2))`, read when the plan is made. `STDERR_REJECTIONS` gains the lifetime checker, whose `require()` is `SystemExit('FAIL: …')`. `TEXT_BASELINES`: the lifetime checker's success report is the plain line `CHECK PASSED (full)` or `CHECK PASSED (sample)`, matched byte for byte. `LOADS`: the lifetime checker loads `engine_e2.py` and `engine_e3.py`; the reference file binds their git blobs like the checker's. The earlier families keep their order, arguments, timeouts, concurrency and checks; planar's plan moved into the table unchanged. |
| `tools/check_mode_contracts.json` | Two new families with the measured verdicts; a text baseline is stored as the line itself, per mode. The earlier entries are unchanged. |
| `tests/test_check_mode_workflows.py` | The families join the tables. `groups()` restates the serial wrappers' child lists on its own, with planar's `2` and the other two wrappers' own `max(2, cpu_count)`. The synthetic checker prints a plain success line for a text family. New test: the text-baseline contract (a reformatted line, or extra stdout after it, is rejected); the baseline faults and the stderr-channel faults cover the lifetime family. The reference custody digest moves. |
| `.github/workflows/c8-lifetime-difference-constant.yml` | Calls `replay('c8-lifetime-difference-constant', root)` in place of its exit-only loops. Adds the same earlier contract step and trigger paths as the other five wrappers. |
| `.github/workflows/c8-theorem-a-constants-d3.yml` | Calls `replay('c8-theorem-a-constants-d3', root)` likewise. Same contract step and trigger paths. |
| `reviews/c8_check_mode_wrappers_batch4_20261031/SCOPE.md` | The scope note, with identities, contract, evidence and limits. |

**The contract, `c8-lifetime-difference-constant`:**
- **Baselines:** exit 0, empty stderr, and exactly the line `CHECK PASSED (full)` (normal `--check-full --procs P`, after all 274 cap bands, the Taylor-model laws, rate sweeps, reference integrals and assembled tables of both dimensions have matched `RESULTS.json`) or `CHECK PASSED (sample)` (optimized `--check --procs P`, the sampled bands).
- **Each mutant** (normal `--check --procs P --mutant M`): exit 1, empty stdout, and exactly the stderr line its `require()` raised: @@LIFETIME_MUTANTS_INLINE@@

**The contract, `c8-theorem-a-constants-d3`:**
- **Baselines:** exit 0, empty stderr, and the measured verdict `{"check": "ok", "bands_replayed": N, "C_table": {…}, "z_star": {…}, "mutant": null}` with the seven-entry `C` and `z_star` tables exactly as `RESULTS.json` stores them: normal `--check-full --procs P` with `N = 121`, optimized `--check --procs P` with `N = @@D3_SAMPLE_N@@`.
- **Each mutant** (normal `--check --procs P --mutant M`): exit 1, empty stderr, and exactly `{"check": "FAIL", "reason": …, "mutant": M}` with the measured reason: @@D3_MUTANTS_INLINE@@

`P` is `str(max(2, os.cpu_count() or 2))` on the host, as the wrappers computed it. It sizes the checkers' process pools only; a `--procs 2` control reproduced the same verdicts.

**Not changed:**
- the source blocks that run before any child: the packet manifests, the main-resident pins and the lifetime packet copies;
- the action pins, read-only permissions, `persist-credentials: false`, Python 3.11.16 and the 240- and 120-minute job budgets;
- the checker arguments, the per-child timeouts (lifetime 10800 s for each baseline and 3600 s for each mutant; d3 5400 s and 2400 s), the mutant order and the one-at-a-time schedule, and the terminal records;
- the final `git diff --exit-code`;
- every packet file.

On an accepted run only the job's printed lines change: one record per child (exit status, byte counts, digests, `protocol_ok`) instead of a 600-character tail of each baseline's stdout.

## Validation

**Local** (on `19a55750`, CPython @@LOCAL_PYTHON@@, four CPUs, so `P = "4"`):
- **Characterization:** two separate runs of the unchanged checkers, 14 children each, with the wrappers' exact arguments. All stdout and stderr byte strings are identical across the runs, and the exit statuses are as above. Reference file: @@CONTRACT_BYTES@@ B, SHA-256 `@@CONTRACT_SHA@@`. @@TIMINGS_SHORT@@
- **Control:** @@PROCS_CONTROL@@
- **This test file:** @@TEST_COUNT@@ tests OK in both modes, for each new family alone, as its workflow step runs it (@@TEST_SECONDS@@), and for all seven families (about @@TEST_SECONDS_ALL@@ per mode).
- **RED control:** the same tests, with `CHECK_MODE_SOURCE_ROOT` set to this branch's base `19a55750`, whose two wrappers are unchanged. False acceptances, the same in each mode:

  | Test | lifetime | d3 |
  |---|---|---|
@@RED_TABLE@@

  The wrong exit statuses were already rejected there. The trigger, step and import tests fail there too (nine subtests per family), as expected. The inventory test passes there, because the old wrappers ran the same schedule.
- **The real checkers through the helper**, run as the workflow steps run them, on this tree: @@REAL_REPLAYS@@
- **Full suite, normal mode:** @@FULL_SUITE@@

**Hosted, at this head:** pending.

## Limits

- The local interpreter was CPython @@LOCAL_PYTHON@@, not the hosted 3.11.16. The hosted run is the qualifying execution.
- The synthetic fixture uses a one-entry pin list and a minimal manifest. The real pins and packet copies run only in the real hosted jobs.
- @@SHARED_REASONS_LIMIT@@ The mutants are checked in normal mode only, as the wrappers ran them.
- `P` varies with the host, so the reference cannot pin it; the plan reads it as the wrapper did, and the control is the evidence that the verdicts do not depend on it.
- GPT-6's read-only audits of two batch-1 wrappers ([#322/6522251138](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-6522251138), [#322/6522460101](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-6522460101)) report workflow-control findings on the pre-replay source block common to the seven wrappers: the packet root is resolved before the root itself is checked, the manifest and source map are bound only to each other, and the trigger test does not derive the `current_required` paths. That block is unchanged here; those findings keep their own stage.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
