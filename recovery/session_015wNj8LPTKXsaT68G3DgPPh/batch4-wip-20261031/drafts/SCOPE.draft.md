# C8 check-mode wrappers, batch 4: the lifetime-difference and d = 3 Theorem A wrappers

Dylan Roy — delegated AI engineering. Actual author: Anthropic Claude, session
`session_015wNj8LPTKXsaT68G3DgPPh` (C8), stage `C8-WRAPPERS-REPAIR-C8-20261026`,
batch 4, pickup Math-#322/6422210000 and its batch-4 line Math-#322/6545025569.
Dylan approved draft PRs for the seven `c8-*` wrappers around this session's
packets, in the author's session at about 16:45Z on 26 October 2026. Scientific
effect NONE; shared-account organizational credit 0. Author-side implementation
only: a separate nonauthor engineering review and a separate integrator are
required. No mathematical acceptance is asserted.

The branch is cut from Math- `main` `19a55750`, the merge of batch 3
(Math-#544), because it extends the same helper, reference file and test file.
This is the last of the four batches, and the seventh and sixth wrappers of the
seven. The scope notes of batches 1 to 3 are unchanged.

## Finding and exact source

01Cz's ledger (Math-#322/6388271177) classifies both wrappers as `status_only`,
and GPT-6's read-only audits of 30 October (Math-#322/6516675793 for d3 and
Math-#322/6516944036 for lifetime, finding `MTW-C8-LIFETIME-WRAPPER-01`)
reproduced the defect in disposable fixtures with synthetic children:

- `c8-lifetime-difference-constant` accepted its two baselines on exit status 0
  alone, echoing a tail of stdout and discarding stderr, and each mutant on
  exit status 1 alone, with its stdout and stderr captured and thrown away.
- `c8-theorem-a-constants-d3` did the same with its checker and five mutants.

So an unrelated crash, a silent exit 1 or a failure for another reason counted
as an intended rejection, and the baselines' verdicts were never compared. The
ledger read the workflow blobs `1f99cdab` and `8a1fdad2` at `d1f6ad28`; the
blobs below differ from those only in the `actions/setup-python` pin (`2c041de`,
24 October), so the classification still applies. Both audits prescribe extending the check-mode helper with
measured references and negative controls while preserving the invocation
matrix; both were released, and no repair claim existed when batch 4 was
picked up.

| Object | Identity |
|---|---|
| Math- `main`: branch base and characterization tree | `19a55750451b8016b50d1d20be6a18c021807ecb` |
| `c8-lifetime-difference-constant.yml` before | blob `ba3b9c099f03c189fc4eb40d8858f46ddb46b331`, 4,135 B |
| `c8-theorem-a-constants-d3.yml` before | blob `c315ed1283d4e1955874d8df529c1c00959f6405`, 3,761 B |
| `theorem_b.py` (unchanged) | blob `36b39dbd33ef2e6d277ce1c3820b3f0d6659d779`, 53,141 B |
| lifetime `engine_e2.py` (unchanged, loaded) | blob `ae279e9015483a666ece8deb2544cbc458a5e270`, 52,864 B |
| lifetime `engine_e3.py` (unchanged, loaded) | blob `267890232119b35206be7a8cc74d730de9a68644`, 76,209 B |
| lifetime `RESULTS.json` (unchanged) | 463,344 B, SHA-256 `77703dbdd7969ab62c179aa49391c6cddb545d37e0d59654476a09166a0b938f` |
| `theorem_a3.py` (unchanged) | blob `267890232119b35206be7a8cc74d730de9a68644`, 76,209 B |
| d3 `RESULTS.json` (unchanged) | 159,177 B, SHA-256 `9d9e889018b197ef4d997744d05bb1e7b84881abd29ae14874d452343d9b67c0` |
| packet trees (unchanged) | lifetime `4c33a54bb1b6408b30c3610432e900e68e9db69c`, d3 `b494502a28822f05d1635f6a35199a777aea2d14` |

As the lifetime `SOURCE_MAP.json` pins them, the two loaded files are
byte-identical packet copies: `engine_e2.py` of the planar Theorem A packet's
`theorem_a.py` (Math-#206) and `engine_e3.py` of the d = 3 packet's
`theorem_a3.py` (Math-#212), which is this batch's other checker. The torus
packet of batch 3 carries the same copies. The imported
`tools/legacy_json_replay.py` (blob `1aa66bb3`) and
`tests/test_legacy_workflow_triggers.py` (blob `c2e37a3a`) are the blobs every
local run below used.

## Repair and preservation

Six files: `tools/check_mode_replay.py`, `tools/check_mode_contracts.json`,
`tests/test_check_mode_workflows.py`, the two workflows, and this note.

- **`tools/check_mode_replay.py`:**
  - Two inventory rows, in `PLANNED`. A `SERIAL` table restates the plan that
    the planar Theorem A wrapper shares with these two: one child at a time,
    every band with `--check-full` in normal mode, the sampled `--check` in
    optimized mode, then each mutant in normal mode only as
    `--check --procs P --mutant M`, with each wrapper's timeouts (lifetime
    10800 s for a baseline and 3600 s for a mutant; d3 5400 s and 2400 s;
    planar 3600 s and 1800 s, as before). Planar passed `--procs 2`; these two
    wrappers passed `str(max(2, os.cpu_count() or 2))`, which the plan reads
    the same way when it is made, so each child gets exactly the arguments
    its wrapper gave it on that host.
  - `STDERR_REJECTIONS` gains the lifetime checker, whose `require()` is
    `raise SystemExit('FAIL: ' + msg)`: exit 1, nothing on stdout and one line
    on stderr, matched byte for byte.
  - `TEXT_BASELINES`: the lifetime checker's success report is the plain line
    `CHECK PASSED (full)` or `CHECK PASSED (sample)`, not JSON. Its baseline
    must match the measured line byte for byte, with exit 0 and empty stderr.
    The d3 checker reports JSON like the planar one: `{"check": "ok", ...,
    "mutant": null}` on success and `{"check": "FAIL", "reason": ...,
    "mutant": M}` from its `require()`.
  - `LOADS`: the lifetime checker loads `engine_e2.py` and `engine_e3.py`
    (`load_engine()`); the reference binds their git blobs like the checker's.
  - `_contract()` also requires `PLANNED` to be exactly the `SERIAL` and
    `PER_JOB_MODE` families, and the stderr and text families to be in the
    inventory. The batch-1 to batch-3 families keep their order, arguments,
    timeouts, concurrency and checks; planar's plan moved into the table
    unchanged and is covered by the same tests as before.
- **`tools/check_mode_contracts.json`:** two new families with the measured
  verdicts; a text baseline is stored as the line itself, per mode. The
  earlier families' entries are unchanged.
- **`tests/test_check_mode_workflows.py`:** the families join the tables.
  `groups()` restates the serial wrappers' child lists on its own, apart from
  the helper, with planar's `2` and the other two wrappers' own
  `str(max(2, os.cpu_count() or 2))`. The synthetic checker prints a plain
  success line for a text family and accepts exactly the arguments each wrapper
  passed in each mode. New: `test_text_baseline_contract` (a reformatted plain
  success line, or extra stdout after it, is rejected); the baseline faults
  (unexpected stderr, wrong value, wrong exit, missing newline, malformed) and
  the stderr-channel faults cover the lifetime family as well. The
  formatting-tolerance test skips a text baseline, which admits none. The
  reference custody digest moves with the new families.
- **The workflows:** each calls `replay(...)` in place of its exit-only child
  loops and gains the same earlier contract step and trigger paths as the other
  five wrappers. The unused imports of the removed loops are dropped.

Unchanged: the source block before any child (the manifest, flat regular files,
the `current_required` pins and, for lifetime, the two `packet_copy`
bindings); the action pins, read-only permissions, `persist-credentials:
false`, Python 3.11.16 and the 240- and 120-minute job budgets; the checker
arguments, the per-child timeouts, the mutant order and the one-at-a-time
schedule; the terminal records; the final `git diff --exit-code`; every packet
file.

No behavior changes on an accepted run except the job's printed lines: the
wrapper echoed the last 600 characters of each baseline's stdout; the helper
prints one record per child (family, mode, label, exit status, byte counts,
digests, `protocol_ok`). On a failed baseline the wrapper printed 400-character
tails of its streams; the helper prints the child's record and raises.

## The contract

The references are complete verdicts measured from the unchanged CLI, never
outputs learned from the invocation under test.

**`c8-lifetime-difference-constant`:**

- **Baselines:** exit 0, empty stderr, and exactly the measured line:
  - normal `--check-full --procs P`: `CHECK PASSED (full)`, after every one of
    the 274 cap bands of both dimensions, the Taylor-model laws, the rate
    sweeps, the reference integrals, the cap boxes and the assembled tables
    have matched `RESULTS.json`;
  - optimized `--check --procs P`: `CHECK PASSED (sample)`, the sampled bands.
- **Each mutant** (normal `--check --procs P --mutant M`): exit 1, empty stdout,
  and exactly the stderr line its `require()` raised.

| Mutant | Measured line |
|---|---|
@@LIFETIME_MUTANTS@@

@@LIFETIME_REASON_NOTE@@

**`c8-theorem-a-constants-d3`:**

- **Baselines:** exit 0, empty stderr, and the measured verdict
  `{"check": "ok", "bands_replayed": N, "C_table": {…}, "z_star": {…},
  "mutant": null}`, with all seven entries of the `C` table and the `z_star`
  table as `RESULTS.json` stores them:
  - normal `--check-full --procs P`: `N = 121`, every band;
  - optimized `--check --procs P`: `N = @@D3_SAMPLE_N@@`, the sampled bands.
- **Each mutant** (normal `--check --procs P --mutant M`): exit 1, empty stderr,
  and exactly the verdict its `require()` prints, `{"check": "FAIL", "reason": …,
  "mutant": M}`.

| Mutant | Measured reason |
|---|---|
@@D3_MUTANTS@@

@@D3_REASON_NOTE@@

`P` is `str(max(2, os.cpu_count() or 2))` on the host that runs the wrapper. It
sets only the size of the checkers' process pools; a control run below shows
that the verdicts do not depend on it.

## Execution and non-vacuous regressions

All of the following ran locally on `19a55750`, on CPython @@LOCAL_PYTHON@@ with
`os.cpu_count() == 4`, so `P = "4"`.

- **Characterization:** two separate runs of the unchanged checkers, with the
  wrappers' exact arguments, each child a separate process started from its
  packet directory. All 14 stdout and stderr byte strings of the second run
  equal the first run's, and the exit statuses are as above. The reference
  file built from them is @@CONTRACT_BYTES@@ B, SHA-256 `@@CONTRACT_SHA@@`. The
  two families ran side by side, which changes only the timing:
  @@TIMINGS@@
- **Control, `--procs 2`:** the sampled optimized `--check` of each checker
  with `--procs 2` printed @@PROCS_CONTROL@@.
- **The real checkers through the helper:** the hosted jobs at this head run
  them, and they are the qualifying execution. Local runs, as the workflow
  steps run them, are recorded in the pull request: @@REAL_REPLAYS@@
- **This test file:** @@TEST_COUNT@@ tests OK in both modes, for each new family
  alone, as its workflow step runs it (about @@TEST_SECONDS@@), and for all
  seven families (about @@TEST_SECONDS_ALL@@ per mode).
- **RED control:** the same tests, with `CHECK_MODE_SOURCE_ROOT` set to this
  branch's base `19a55750`, whose two wrappers are unchanged. In each mode:

  | Test | lifetime | d3 |
  |---|---|---|
@@RED_TABLE@@

  The wrong exit statuses were already rejected there. The trigger, step and
  import tests fail there too (nine subtests per family), as expected. The
  inventory test passes there, because the old wrappers ran the same schedule.
- **Full suite:** the hosted `full-suite` job at this head is the qualifying
  run; a local run is recorded in the pull request.

## Limits

- The local interpreter was CPython @@LOCAL_PYTHON@@, not the hosted 3.11.16,
  and the local host had four CPUs, as the standard hosted runner does. The
  hosted run is the qualifying execution.
- The synthetic fixture uses a one-entry pin list and a minimal manifest. The
  real pins and packet copies run only in the real hosted jobs.
- @@SHARED_REASONS_LIMIT@@ The mutants are checked in normal mode only, as the
  wrappers ran them.
- `P` varies with the host, so the reference cannot pin it; the plan reads it
  as the wrapper did, and the control above is the evidence that the verdicts
  are independent of it.
- GPT-6's later read-only audits of two batch-1 wrappers
  (Math-#322/6522251138 and 6522460101) report workflow-control findings on
  the pre-replay source block: a packet root that is resolved before the root
  itself is checked, a manifest and source map bound only to each other, and a
  trigger test that does not derive the `current_required` paths. That block
  is common to the seven wrappers and is unchanged by this batch, which is
  scoped to the verdict contract. Those findings keep their own stage.
