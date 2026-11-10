# Node-status v2 test-first source candidate

This document records the accepted T1 test and historical-fixture harness source
from plan `557e3d173316a55ffc25c8694a29d22c589297ba` and append
`366fda3c562e84d5267ee6017bf9c497c46f9dec`. Root authorized an unreferenced
candidate only after whole-source review and the final bounded fix closed.
The candidate must use fresh Main as its sole parent; its commit/tree and full
preservation guard are reported separately. The reserved branch name is
`codex/node-status-v2-test-first-r0-20261110`; this source record creates no
branch, pull request or workflow run.

The source cut observed on 10 November 2026 was Main
`8aaaa387d11dc21a0867b9cc41cc81743ca59975`, tree
`542e6ee34af4c99ba80c8b4d6acd2c6d5bc22f7c`, with 6,246 leaves in a complete
nontruncated recursive tree. Construction requires another current-ref check.
All 17 harness bodies were fetched and their Git blob identities, UTF-8
SHA-256 values, sizes and line counts independently recomputed in memory.
This is source verification; no Python, YAML or test execution occurred.

## Frozen source

| Path | Final Git blob |
| --- | --- |
| tests/test_node_status_live_v2.py | af0d51d8930d6b7ad4fa718dc0741bc6cae53146 |
| .github/workflows/node-status-v2-contract.yml | f77c11f9c16a574fc193ed9be0e2604f84f856ad |
| tests/node_status_legacy_v1_runner.py | 77e036b5e8e816cec67720318d61fb95d445bc47 |
| tests/test_node_status_live_check.py | 3591d1088f22148fea71e25f829ba27ae4524d08 |
| tests/test_node_status_live_cli.py | 4d0df19642d66ccd22eb008325c15d6b370ac7e8 |
| tests/fixtures/node_status_live_v1/bundle.json | 0723223a945bea1593b6e42cb82c957f64b5488b |

The other eleven additions are exact original payload copies under
`tests/fixtures/node_status_live_v1/payload/<original-path>.txt`. They total
1,072,811 bytes and reuse the original Git blobs. The
[complete line/oracle/payload trace](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/e943ac4b76d000ba2f83737bf1dd63e3b19f0bf8)
records every payload identity, all 12 new parent IDs and all 43 unchanged
legacy method IDs and original body locations. The 17 harness bodies total
1,246,504 bytes; all are ASCII, terminal-LF, mode 100644.

## Test-first scope and retained originals

The new test module imports `evaluate_fresh_records_v2` and `run_live_v2`
unconditionally from the existing live module at line 31. Those names remain
absent. The later automatic R0 must show the actual missing-name ImportError,
one loader FailedTest and zero of 12 behavioral bodies. Another error is not
absence evidence. Actual R0 and every behavioral outcome remain unobserved.

The existing outer classes retain all 23 helper and 20 CLI method names.
Each fixed wrapper starts a fresh same-CPython `-B[-O]-S` child and runs only
the identical original method with its original unittest lifecycle. The full
original modules and assertions stay byte-exact in the eleven-member bundle.
The harness retains original test and program stdout/stderr separately from
outer unittest stderr and captures the actual owned native-Git nine-input
program fixtures before cleanup. These are historical local TEST receipts;
they confer no authenticated HTTP or current-v2 evidence.

Shared bootstrap `ea5e25c0605bb9439a3275b207d86ad08918e988` remains byte-exact
and executes from one retained external file. Complete actual source captures
are retained once by SHA-256 with per-case exact references, while complete
before/after captures are still compared. All raw streams remain complete.
The 2 MiB readback cap remains 2,097,152 bytes. Missing, omitted or erroneous
material means HOLD; runtime and actual complete native material size remain
unobserved.

The final fix admits the temporary parent before any creation: literal
`/tmp` when RUNNER_TEMP is absent, otherwise the exact RUNNER_TEMP value,
through `_destination`. Empty, relative, aliased and checkout-contained
parents refuse. There is no `gettempdir` probe, TMPDIR selection or global
tempfile-cache mutation. Both callers still pass explicit admitted directories.
Only the runner and its two wrapper pins and three workflow pins changed in
the final 31-line delta; all other code, bootstrap and payload bytes stayed exact.

The separate workflow has normal and optimized CPython 3.11 jobs, fixed action
pins, 37 protected source pins and the existing 15-minute job limit. It retains
strict separate 23/20/12 outer streams and all inner evidence. It has no
manual-dispatch route or credential environment for the fixture code.

The new 12 parents freeze the closed seven-key manifest and finite profiles;
exact native Git source/input/mode/import binding; ancestor/orphan/historical
artifact controls with pure-v1 positives; refresh validation before observation
or writes; and the fixed native Math pinned-path/blob to local raw Graph,
current identity and complete embedded-data join. Coherent raw or commit
substitution against an unchanged native Graph must refuse. Synthetic API-shaped
fixture inputs traverse the real observer and supply no actual transport,
current-default freshness or scientific authority.

## Exact source evidence and review

- [Full 15-addition diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/dd838bbb3624ac562678d35ee2cba89af126cee0).
- [Helper loader diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/46a7e963044383516c8bc91c9ad8fcee3851893b) and [CLI loader diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/17d5b18e1e663aa1e650f76b39dbd674f5a3f06d).
- [Combined 17-path diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/ba798023a97bca25c0a028a9279c9bc141733987).
- [Final 31-line temporary-parent delta](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/b9a4486fc70580cd7f6e67b35f3034e015222562).
- Earlier bounded numeric/temp-admission and size-dedup deltas:
  `517c3bba57fd0939b9f15533c7f292fda6409863` and
  `368f2b99186e15eacdb8deeac549f135bba668f9`.

Root read the entire 17-file source and independently reconstructed all four
final blobs forward and inverse. Independent technical review by
`v2_newtest_source_review` closed at C0/I0/M0; Lean had read the entire original
legacy source and its 43 targets. The test/workflow author is
`v2_test_contract_source`; runner/wrappers and packet integration are by
`connected_inputs_refresh`. Organizational-independence credit is zero.

The candidate contains exactly the 17 harness paths, this document and a
minimal append to DECISIONS.md. It preserves every other Main leaf, the entire
old decision prefix, production tools and schemas, the four canonical records
and histories, projection, Graph source cut, all Acq/V1/G5/native controls and
owner state. It adds no production input manifest or record population.

Root's full candidate guard precedes publication. Later implementation requires
the actual R0 read and separate source authority. Actual nine-record adoption
and original Q9004/Q9005 current exercise remain later work. No current-v2,
full-Main, G5, CBZ or scientific qualification follows; scientific_effect NONE.

## Publication failure and bounded packaging correction

The preceding source record belongs to candidate
`0e81116ca8fcde518231a1d1c1ee616e21a58367`; it is retained unchanged.
Root subsequently passed its full tree guard and authorized only the fixed
branch and draft [PR #816](https://github.com/d6g8k5htny-coder/main/pull/816).
The automatic [run 42830214749/1](https://github.com/d6g8k5htny-coder/main/actions/runs/42830214749)
failed before creating any job. Its public annotation reports:

```text
(Line: 55, Col: 39): Unrecognized named-value: 'runner'. Located at position 1 within expression: runner.temp, (Line: 56, Col: 37): Unrecognized named-value: 'runner'. Located at position 1 within expression: runner.temp
```

This exact error was read from the public page at line 146 with
`mcp__codex_apps__search_service_web_run`. The
[small native-error receipt](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/3d26680d2f2e1afc3758d59be1d8d0b136cfb271)
retains the public URL, tool operation, source binding and limits, without
private or authenticated HTML. No normal/optimized target job or missing-import
R0 ran. The zero-job failure remains historical failure evidence.

GitHub's [context table](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability)
omits `runner` from job-level env and permits it in step env. The same lesson
had already been recorded in [PR #803's source correction](https://github.com/d6g8k5htny-coder/main/commit/db3066b4950d5b88583ea4fd6e88b026203e3928);
this review missed that known platform constraint. Separately, the original
preparation run scalar measures 27,799 ASCII characters, above the
[documented 21,000-character limit](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstepsrun).
That second constraint is measured source evidence; it was not the reported
native annotation.

Root authorized only the packaging correction below, pending source review.
The fixed workflow is
`e7ca903c6df0d8eea44cb3d3f19526dcde2ca2eb` (8,829 ASCII bytes, 174 LF).
New non-discovered utility `tests/node_status_v2_controller.py` is
`70146dc623661ad6275881c6fc564bcb7e0864fc` (27,043 ASCII bytes, 460 LF;
SHA-256 `a96987b44609220a64ae74ab62ddc3edf8f49af73dbb8eabf9abb8e417925af3`).
Its entire body equals the former f77 workflow's runtime SOURCE literal after
YAML indentation removal. All 37 protected pins, controller logic, oracles,
streams and 2 MiB cap are byte-exact.

The short preparation source joins the utility's fixed path, 100644 mode, blob,
length and SHA-256 to native HEAD/GITHUB_SHA, native Git object bytes and actual
working bytes before an exclusive external copy. It rejects aliases, executable
or nonregular files and capture drift. Each of the four unchanged invocation
steps receives the same two evidence variables in its valid step env. Only the
new utility path is added to both existing path filters; events, commands,
conditions and artifact scope are unchanged. All five run scalars now measure
3,123 / 131 / 128 / 132 / 131 characters, each within 21,000.

Existing automatic legacy contract job 129878243834 did print all 23 helper and
20 CLI outer successes in both modes at signed native merge
`a22d6c05314c6aae25eea5076751dec75ad7c08c`, tree `cd37442f`.
Its [entire original job log](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/ccda5c00a54f3a172146d691e06c5550b415d8ba)
is retained. Its old workflow uploads only four outer logs, so full 43 inner
and program readback remains unobserved and receives no omission credit.
Those outer results retain historical local TEST scope and confer no current
HTTP authority.

This correction changes no af0d test, 77e runner, 3591/4d0d wrapper, ea5e
bootstrap, bundle/payload, production tool/schema/record/projection/owner or
Graph source cut. It supplies no successful new workflow execution or R0.
Candidate construction, ref movement and another automatic run require the
separate source and candidate gates; no dispatch or rerun is part of this
source packet. Scientific effect remains NONE.
