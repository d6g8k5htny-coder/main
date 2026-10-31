# Source-custody final engineering review

PASS_SCOPED_ENGINEERING at exact source commit `5d25cadf4470c462af9c6e4f383b831ce2025320`. No actionable finding remains in the seven-file change. This is a same-family engineering review, not mathematical alignment or organizational independence.

The loaded producer/helper digests, actual manifest bytes and parsed argument, complete bound/unbound owned-file inventory, raw committed blobs, regular-file/executable modes, exact repository root and captured HEAD are checked. Raw Git reads disable replacement objects, ambient Git repository/index overrides and content filters. The postflight check uses the same initial commit and checks the head again within inspection. The separate package argument is now bound to the manifest-owned tree before any output mutation, addressing the earlier review gap. Prior successful receipt/logs are archived together before an initial valid-package custody failure can leave a current success. Generated build outputs and unrelated repository data remain explicitly outside owned-input custody.

Fresh CPython 3.11.16 execution from an exact Git archive of the frozen commit:

- `python3.11 -B -S -m unittest tests.test_formal_source_custody -v`: 22/22 pass; optimized variant also 22/22. These include nine reason-bound source mutants whose full controls and named assertion failures are enforced by the strict shared mutation helper.
- `python3.11 -B -S -m unittest tests.test_formal_gate.SyntheticControls -v`: 48/48 pass; optimized variant also 48/48. Source/head call ordering, invalid runtimes, changed source and refusal to publish receipts remain exercised.
- `python3.11 -B -S -m unittest tests.test_formal_workflow_shell -v`: 9/9 pass; optimized variant also 9/9. The hosted workflow includes custody controls before the dedicated final kernel run and receipt binding.
- Source-only gate passes with manifest SHA256 `5eb96030d92a6da917053310229355c8c8c314ce552cdb9127be0c64a883edd0`.

There were zero failures, errors or skips in the qualified runs. Counts above describe separate modules; nested mutation controls overlap other methods and are not added as distinct mathematical evidence. All seven frozen source files were byte-compared after execution against their Git objects. Exact source/log hashes and limits are in REVIEW.json.

The initial test launch accidentally used the in-progress merge worktree rather than the frozen archive. Its `normal.log` contains the expected unresolved-manifest JSON error and is retained as an unqualified invocation. It is excluded from frozen-source results; all qualified runs used the reconstructed exact archive.

No Lean build/kernel, hosted check, pending peer-evidence composition or federation-wide CBZ state is certified by this review. The contract is pre/post endpoint custody in a trusted runner, not continuous isolation from a malicious concurrent writer. The temporary source archive can be reconstructed with `git archive 5d25cadf4470c462af9c6e4f383b831ce2025320`; the permanent review evidence is the report, source hashes and raw logs.
