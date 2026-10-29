# Strict replay mutation grader: source-condition review

Observed source: [main #705 candidate commit 30184a6c7337099f910809116523112394e6504b](https://github.com/d6g8k5htny-coder/main/blob/30184a6c7337099f910809116523112394e6504b/cbz/replay-qualification/replay_mutations_strict.py), Git blob `12be4b1a30106ce99049ad461c0a09381571ec69`. Companion test blob: `1ce6df399e7e907185fd580aa6817e4f7e4c47fb`. This is a **source-condition inspection**, not an executed counterexample or an observed false pass in any real workflow. The current-head checks were queued at review; no result is inferred.

## Unpinned baseline identity and outcome membership

At lines 40–62, `grade` requires 31 unique names, but with its default `names=None` it does not compare those names with an independently specified expected set. Lines 152–155 use that default for both baselines and compare their sets with each other; lines 163–164 compare each mutant's names with the corresponding baseline. At lines 57–61 the sole-failure check compares a failure tuple to the expected tuple but does not require that its test ID occur in `observation['names']`.

For example, from the source conditions alone, the following fabricated observations satisfy those checks (assuming the unchanged function is invoked; this example has not been executed):

```python
names = [f"foreign.Other.test_{i}" for i in range(31)]
base = dict(count=31, names=names, errors=[], skipped=[],
            expected_failures=[], unexpected_successes=[], failures=[])
expected = (
    "test_replay_tool.TranscriptTests.test_wrong_module_assignment",
    "AssertionError: ValueError not raised",
)
mutant = dict(base, failures=[list(expected)])
# Source analysis predicts grade(base) and grade(mutant, names, expected)
# both return normally, even though the failure ID is absent from names.
```

The ordinary workflow constructs observations from the pinned test source and the Python unittest runner. Under that trust assumption this fabricated record does **not** establish that an actual run can produce such inconsistency or that any historical result is false. A separately pinned full expected-ID set and adverse cases for failure membership would strengthen the grader's explicit source-level contract; those controls remain unimplemented and unexecuted at this review.

## Byte binding and limits

The top-level program checks fixed SHA-256 pins for the historical target and test bytes (lines 26–27 and 114–118), validates unique mutation anchors, and its worker compiles its checked target bytes and the test bytes after one exact loader substitution (lines 28–34 and 72–86). `--worker` accepts caller-provided digests and emits observations; it is an internal execution path, not a standalone certification.

The pinned historical test source [test_replay_tool.py](https://github.com/d6g8k5htny-coder/main/blob/30184a6c7337099f910809116523112394e6504b/cbz/evidence/kernel/additional/replay-tool/test_replay_tool.py#L230) executes `replay_tool.py` by filesystem path in two subprocess calls at lines 230–237. Those child executions are distinct from the in-memory loader and are not byte-bound against a transient path replacement by the worker's checks before and after the suite. Their command omits `-O` even when the parent suite is run optimized; the in-process validator tests still run in both parent modes. Trust in the runtime, filesystem and subprocess execution is explicitly documented by the candidate.

The candidate covers three declared Python validator-source mutations only. It does not repair the separate historical Lean self-test's diagnostic matching or normal-import behavior, establish general Layer A auditor soundness, authenticate historical execution, or satisfy any final federation gate.
