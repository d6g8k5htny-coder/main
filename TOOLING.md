# CBZ replay delivery tooling

The historical additional and determinant packages each contain the same
bounded replay helper: `cbz/evidence/kernel/{additional,determinant}/replay-tool`.
It generates a Lean closure-replay program and validates its transcript against
a caller-supplied census; it does not authenticate a caller-supplied execution.
It is not a global auditor. Maintenance owner: none, bounded evidence artifact.
It serves criteria 3 and 6 only within the explicitly supplied closure, and
criteria 9 and 11 for its documentation and reproducibility.

With Python 3.11.16, from either replay-tool directory, run
`python -B -S -m unittest -v test_replay_tool.py` and
`python -B -O -S -m unittest -v test_replay_tool.py`.
Run `python -B -S run_mutations.py --output /new/output-directory` to
exercise the three registered source mutations in both modes. Actual-Lean
controls require the Lean 4.34.1 runtime and collector described in the package
README; use its documented command once in normal mode and once with `-O`,
with separate new output directories. Verify `sha256sum -c SHA256SUMS` in each
package directory and each replay-tool directory. The hosted
`.github/workflows/cbz-replay-delivery.yml` executes these checks and retains
logs and reports. Current-head execution is pending until observed.

The historical `verification-v2` and `lean-verification-v3-*` reports are
historical evidence, not fresh verification. The Lean `final-normal` report
names a different self-test digest, so does not authenticate the delivered
self-test bytes. Historical Python mutation grading accepts unrelated assertion
failures, extra errors and a zero-test baseline, and its module loader can read
poisoned bytecode despite `-B`. The Lean self-test checks substrings, which
does not bind the entire transcript to the asserted reason. Source review
identified these issues; hosted counterexample execution is pending. Therefore
the helper and its historical self-tests are not admitted as an A8/G0
certifier. The historical source and reports remain unchanged.

The new `tests/test_cbz_replay_harness_adverse.py` exercises the actual
historical mutation `main()` with controlled subprocess results and tests a
timestamp/size-valid poisoned cache against the actual historical module loader.
Run `python -B -S -m unittest -v tests.test_cbz_replay_harness_adverse` and
the same command with `-O`. These are counterexample demonstrations: a pass
means the historical weakness was observed, not that it is fixed. The workflow
runs them and preserves logs in both modes. Owner: none. Their own execution
and the workflow are pending observation at the current head; they do not
certify Layer A. Fresh private bytecode prefix isolation in the historical
workflow assumes a trusted hosted runner and mitigates pre-existing caches only.

## Proposed strict mutation qualification

`cbz/replay-qualification/replay_mutations_strict.py` and
`test_replay_mutations_strict.py` have a **new separate** SHA256SUMS. They
do not replace or modify the four historical manifests. This candidate pins the
original source/test hashes, adapts only their loader in memory, executes the
verified source bytes, retains the adapted test hash and structured results,
and requires 31 unique tests, no extra failures/errors/skips, and the exact
guard-specific assertion in each Python mode. Its tests plant classifier
mutations and exercise unrelated assertions, foreign errors and poisoned
timestamp-valid source/test caches. Current-head verification is pending.
The candidate has not been admitted as a certifying tool.

With `dir="$PWD/cbz/evidence/kernel/additional/replay-tool"`, verify the new
manifest using `(cd cbz/replay-qualification && sha256sum -c SHA256SUMS)`.
Run its tests using
`CBZ_REPLAY_DIR="$dir" python -B -S cbz/replay-qualification/test_replay_mutations_strict.py -v`
and again with `-O`. Run the candidate itself with
`python -I -B -S cbz/replay-qualification/replay_mutations_strict.py --directory "$dir" --source-sha256 b97bee53a590651b83d68056c090ba70e8a055157e65d2bd3fcc5cd0219840b3 --test-sha256 a07eaab8b6fd40a911d313197fae688927ebb3e1d74644628e4888a7e6505fc0 --output /new/strict-output`.
Repeat for the determinant directory with a distinct output. The candidate
runs both Python modes internally. The workflow performs all of these.

Trust assumptions include the Python runtime, standard library, subprocess
results and filesystem; read/re-read hashing cannot rule out transient
replacement. This candidate does not fix the separate Lean self-test's
substring grader or normal-import behavior. Owner: none, bounded
self-contained candidate, serving criteria 6 and 9 if verified.
