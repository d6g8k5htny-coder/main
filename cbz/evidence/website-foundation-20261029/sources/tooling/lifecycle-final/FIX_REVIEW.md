# Independent finite mutation-grader review

Result: PASS_SCOPED_ENGINEERING_AFTER_FIX at helper SHA-256 `1d441950e37a087ad97306d575892cb1b9723c04a3db1cc45a56a28a51798231`. This is a same-family engineering review with no scientific independence or theorem-certification credit.

## Important finding found and closed

The initial helper (`00e316ffde3cb33b7499d57141856ef0efa3bd304001148b8737c5674c7ee2ea`) replaced an async named test with an ordinary function. `IsolatedAsyncioTestCase` then failed to await the real body. A real async test that appended a body-entry marker and called `self.fail` was reported as a successful baseline with no body marker. Generator test bodies and ordinary functions returning coroutines had the same false-positive baseline. The three benign controls reproduced normally and under Python optimization; they did not change any original test or mathematical input.

The corrected helper explicitly rejects async/generator/async-generator named functions before execution (tools/cbz_mutation.py lines55–60), refuses every non-None body return, and closes returned coroutines without executing them (lines67–70). The independent three-case replay now reports neither positive baseline nor mutation credit for every unsupported case, in both modes, with no deferred-body execution. Five permanent regression cases cover these types. The original synchronous scanner/ref/baseline populations are unaffected; no historical actual-body failure is retracted by this finding.

## What was checked

Read the helper and all three integrations. The classifier records each expected named body entry, phase, AssertionError identity, actual result event and complete population. Fixture, dispatcher, teardown, cleanup, class-fixture, skip, error, expected-failure, unexpected-success and subtest-only outcomes cannot supply a kill. The original test assertions remain unchanged. Subtest diagnostics alone never supply mutation credit; a separate direct body assertion may still count when the full population and lifecycle are valid. Test identities and helper/test bytes are bound by the callers, and child optimization flags are explicitly propagated and checked by the scanner selftest.

Independent execution: 25/25 lifecycle tests normally and 25/25 optimized, plus the separate async/generator/returned-coroutine probes in both modes. All four commands exited0 and checked source bytes remained unchanged. Exact commands, log hashes and input identities are in REVIEW_VERIFICATION.json. Unsupported non-None result fixtures produce ordinary unittest DeprecationWarnings; these are visible expected negative-case diagnostics, not omitted failures.

No unresolved finding remains within this finite synchronous-unittest classification scope. This does not certify arbitrary hostile Python, async unittest support, all scanner semantics, mathematical statements, all project tests or federation CBZ-0. Source changes after the recorded hashes require fresh verification. The implementer's full guard-mutant reruns and helper source mutants are separate evidence, not claimed as independently executed by this review.
