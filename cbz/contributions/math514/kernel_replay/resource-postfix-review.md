# Focused review of the kernel-replay resource repair

Verdict: **PASS for the scoped P2 repair; no blocking findings.** This is a separate same-provider engineering review of the resource wrapper and its evidence, not provider-distinct scientific review or a new mathematical certification.

## Assessed generation and scope

Reviewed `resource-boundary-v2` in `/workspace/cbz-0/tools/kernel_replay`, including `kernel_replay.py`, the changed wrapper tests and mutation definitions, `README.md`, `RESOURCE_FIX.md`, and `evidence/g0-resource-applicable.json`. The assessed wrapper SHA-256 is `27da13a9455a62c2b192ff62f6222266cb483d323cf680f3d81d3b19b260b8ce`; driver SHA-256 is `24e55f5e99c83a46f9f908bbc90f22292a94305294e287d9b2eccbf24e8e6b66`; wrapper-test SHA-256 is `2ccdd956379bf3ddbc479a10604a10db7c26acee96c54c869c7a6bafc0f2dfeb`. The applicability manifest binds the remaining assessed sources and receipts.

The comparison baseline was the published `cd3a0021a16437567ceda8c9a0f0525677042de2` contribution, obtained with read-only `git show`. The original external P2 is preserved in `discovery/kernel-resource-review-native.json` and corresponds to [discussion 4416529549](https://github.com/d6g8k5htny-coder/main/pull/696#discussion_r4416529549).

No production or test source was changed during this review. No Git mutation, remote action, test execution, Lean compilation, real-fixture replay, or project mathematical gate was performed. Review activity consisted of source/evidence inspection, baseline comparisons, and metadata/count/hash checks on the newly supplied applicability manifest.

## Source assessment

- Memory is rejected before spawning unless it is an exact integer within the documented minimum and the smaller of the platform signed-byte ceiling and inherited finite hard `RLIMIT_AS`, converted down to MiB. Requested caps are not silently clamped. The timeout type, positive upper bound, and finite-value checks reject booleans, nonfinite floats, huge finite floats, and huge integers; the integer upper-bound comparison precedes `math.isfinite`, avoiding its huge-integer conversion overflow.
- Spawn and wait `OSError`, `SubprocessError`, `OverflowError`, and `ValueError` remain structured, non-green results. An actual preexec failure is represented by a retained passing regression receipt, in addition to injected exception coverage.
- Explicit `Popen` ownership removes the wrapper's implicit context-manager wait. Timeout and abnormal-wait paths signal the process group, attempt a child kill on other signaling failures, and use `wait(timeout=1.0)` followed only by nonblocking `poll` if needed. `ProcessLookupError` is handled as an exit race and still leads to bounded reaping. There is no wrapper cleanup `wait()` without a timeout.
- An unreaped PID is retained with cleanup diagnostics, forces `ERROR`, and stops subsequent module scheduling. Expected coverage stays non-green. The documented limitation is accurate: finite cleanup cannot force an OS-uninterruptible child to terminate.
- Only a normal completed wait sets `FINISHED`, and only `FINISHED` can become `PASS`. A spawn error, wait error, or timeout cannot become green even if cleanup returns zero and the output contains the success marker. Normal completion still requires exit zero and exactly one matching marker; nonzero exit with a marker remains fatal.

The normal child command, `cwd`, environment construction, checker precedence, resource installation, output-tail handling, marker predicate, module/part resolution and binding validation are unchanged. `ExactReplay.lean`, `run_tests.py`, real-Lean tests, and mutation-accounting tests are byte-identical to the published baseline. The production change therefore composes with the earlier normal-path and replay-engine evidence; this review does not claim that the new wrapper bytes ran those real-Lean tests again.

## Evidence assessment and reuse

Read the fresh receipts rather than repeating them:

- `g0-resource-wrapper-tests.json`: 30 passing tests, zero assertion failures, errors or skips; 17 original wrapper methods, 12 new resource methods, and one mutation-accounting method.
- `g0-resource-additional-tests.json`: two passing methods, covering oversized CLI exit/report behavior and the strengthened existing memory oracle. The latter prints actual soft/hard child `RLIMIT_AS` and checks both in the parent.
- `g0-resource-preexec-test.json`: one additional passing actual-preexec-failure method.

The distinct-ID union is 32 tests across 33 invocations, including 14 new resource methods. The repeated memory method is not counted twice. All 20 path/hash bindings in the new applicability manifest matched their files, and its fresh IDs/counts agree with the receipt logs.

`g0-resource-mutations.json` contains 19/19 current-wrapper mutant kills: ten original wrapper cases and nine resource cases. Each has actual assertion failures, zero test errors, zero skips and a failing test exit. Inspected traces address the intended defects, including oversized caps, inherited hard limits, boolean timeout, process-error result loss, unbounded cleanup, missing cleanup, erroneous success promotion and continuing after failed reaping. The retained `g0-resource-memory-mutant-refresh.json` establishes the memory kill with the strengthened parent oracle (`[-1, -1]` versus the requested finite soft/hard limits). The classifier continues to refuse errors, empty runs, skips and timeouts as kills.

The original 25-plus-2 test receipts, their 27-test applicability receipt, and the original 16-mutant receipt remain separate. The three `evidence/resource-baseline/` files match their historical hashes and are byte-equal to the published baseline. All original mutation cases remain defined. The nine real-Lean methods and six driver/import-trust mutants retain their completed applicable evidence because their relevant sources, invocation semantics and dependencies are unchanged. Thus 41 distinct applicable tests and 25 applicable mutation cases are a composition, not claims of fresh 41-test or 25-mutant runs. The clean red receipts are separately identified; the initial red probe's two harness errors are preserved as failed development evidence rather than passing verification.

This follows `/workspace/AGENTS.md`: completed applicable verification is reusable; a handoff alone is not a reason to repeat it. The existing successful source8151 Math-core gate with two threads remains established evidence. This repair neither downgrades that result nor repeats it. G2, source/build alignment, complete owned-module coverage, full CBZ acceptance, hosted checks and any provider-distinct process obligation remain separate from this engineering verdict.
