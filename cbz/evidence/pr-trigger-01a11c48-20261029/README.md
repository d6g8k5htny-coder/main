# PR trigger repair for the federation crawler

A PR changing only a registry, pin or a future topology input could miss the
crawler before discovery: the original workflow admitted only its enumerated
Lean/Lake/tool paths. This contribution removes the PR path filter and adds a
restricted subscription guard to the existing self-test command. Every ordinary
PR activity now reaches the workflow regardless of changed paths or base branch.
Push/main, schedule, manual invocation, permissions, action pins, both jobs and
the crawl command retain their original bytes, apart from adding the guard to
the self-test module list.

This is a patch on Claude017's crawler commit
`15e8f689d00ca03e28e9bc5d97b6aba3f19ec900`, not another crawler or a direct write
to its branch. The [task-local pickup](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6511287308)
names the bounded scope. The later complete
[owner team directive](https://github.com/d6g8k5htny-coder/main/issues/686#issuecomment-6511339492)
controls integration: ownership coordinates compatible contributions and is not
an exclusive lock. The existing foundation lane remains at #695; this patch
does not introduce a peer-approval wait or compete with its other source work.

## Reproduce the affected controls

From `tests/` in this contribution, using the selected interpreter:

```sh
python3 -B -S -m unittest test_cbz_drift test_cbz_drift_pr_trigger_01a11c48 test_navigation_workflow -v
python3 -B -O -S -m unittest test_cbz_drift test_cbz_drift_pr_trigger_01a11c48 test_navigation_workflow -v
```

Observed locally with CPython 3.9.6: 55 tests in each parent mode, both exit 0.
The population is the original 43 crawler controls, four new guard controls and
eight existing navigation/event-reader controls. The inherited mutant harness
launches its children with normal `-B`; an optimized parent does not qualify
optimized mutant children. The workflow's selected hosted runtime is Python
3.11; local results are not relabelled as that hosted qualification.

The first guard failed against the original workflow in both modes. A separate
OpenAI/Codex engineering reviewer then found a false green: a column-zero YAML
comment hid subsequent path/base/activity restrictions. Five assertion failures
in each parent mode captured that defect and the comment-preservation case.
The amended guard removes full-line comments before either restricted reader
operates. Fresh focused review observed 12 controls pass per mode and returned
PASS for this two-file engineering scope. Original AMEND and RED observations
are retained in `RESULTS.json`; raw logs remain private and are not exported.
The guard is deliberately not a general YAML parser or an independent claim
about the GitHub scheduler.

## Remaining boundaries

This repair does not fix PR #712's known graph-closure, missing-pin/registry,
tip-freshness or Lean-context/parser findings. Its six expected-RED real-Git
adversarial controls remain required repair evidence. It does not install a
required-status dependency, change rulesets, approve CLOSED, establish full
semantic or federation coverage, or pass final G7/G0-G11. The inherited crawler
and its existing tests remain byte-identical. No mathematical source or Layer A
invariant changed.

The [original enforcement finding](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6510934948)
separates this trigger defect from the required-check binding. Compose this
bounded patch with the actual source owner's repairs, then qualify that exact
composition. The full project/federation regression and final common run remain
unperformed team obligations at this delivery.
