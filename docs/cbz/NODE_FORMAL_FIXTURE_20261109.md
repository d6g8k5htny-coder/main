# Historical Atlas profile refusal control

The prepared Node regression at Main `24ad8a143862fb9255fc6e2e7bf08c12f989f71d`
ran 767 tests: 766 passed and one failed, with no skipped, cancelled or todo
tests. The failure was the assertion that the checked-in `current.json` must
remain unmapped. Its raw TAP, result, exact current snapshot and source files
are retained outside the source tree under `node-regression-24ad` and
`node-formal-fixture-fix/baseline-red` in the session evidence.

That assertion was introduced by `6edc839b57f0f06fc0748fd13cde820c7d49da42`.
Commit `2429c5fcd2eed458824a5cba71d09d7ef7be0e0f` later added an explicit
source bridge for the public-site `8d43` manifest and a positive test of the
same snapshot, claim and model. It left the older null assertion unchanged.
The two tests consequently required opposite results for identical input.

The three committed versions of `docs/site/current.json` found by following
that path through the ancestry of Main
`f1ef3a2babecca0f0d4eb35b018dc81d30e9d62c` use
manifest blob `8d43188691c695f66e233727322484da22662a58`, 17,411 bytes.
The latest snapshot has blob `a64fb6d9768ada12b9884adac854e0ccdd5a910e`,
962,639 bytes and SHA-256
`560d3d4e77369835c832dbb4fd0d12f4988744045b9cbc4c7978beddb6e9d20f`.
No genuine older unmapped snapshot was found in this bounded history. The
historical source is preserved; no invented older observation replaces it.

The successor control starts from the existing pinned
`tests/fixtures/atlas_formal_projection.json` (SHA-256
`7b36e2376f4010444108e81f316293ef3b3834d767e0900f68d684c760af2db1`).
It explicitly selects this historical projection profile:

- Commit: `b7deb198722582df191006e4f47238f92e6e543d`.
- Manifest blob: `8d43188691c695f66e233727322484da22662a58`.
- Length: 17,411 bytes.
- Manifest SHA-256: `1c86e1eed110df3ef283f3a6e402194f82f682ba06e55e3ad8fe2010f152df7e`.

This is a controlled profile fixture, not a claim that it is an unmapped
historical observation. The positive baseline must select the exact admitted
`8d43` bridge while retaining execution at
`3b39c42d6edcc98ae165270be346f0e31c57f81d` and the original `REJECTED`
alignment. The control then changes only the manifest blob identity and must
receive `null`. This retains a live refusal assertion rather than merely
inverting the obsolete expectation.

Production model, bridge data, mathematical sources, original execution and
alignment records are unchanged. A targeted guard-removal mutant must fail
the new refusal assertion rather than crash. Focused controls and the full
prepared Node regression in normal and optimized host Python must be recorded
against the committed candidate. These finite operational observations grant
no new scientific status, Python 3.11 cold-start result or final CBZ gate.

## Hosted prepared regression

The reviewed repair was composed after the actual Main #780 merge
`b1d36c80403acc96d257b9e71140a6cb83a595af` as local commit
`aa4a74bff613b58224ea7995dbb0fa48c62d38b0`. Its original evidence and native
recovery bundle remain external and unchanged. Local process/thread exhaustion
also prevented an ordinary rootless container shim; no PID 1 modification or
privilege workaround is used to obtain a new execution result.

The root-authorized follow-on adds `.github/workflows/cbz-node-regression.yml`.
Existing public-shop CI runs selected Node files, and the older cold-start
workflow's path filter omits this repaired Node test. The new workflow calls
the existing prepared helper over every tracked Node entry on ordinary pull
requests, Main pushes and manual dispatch, with read-only contents permission.
It uses an ordinary Ubuntu 24.04 runner, Python 3.11 and Node 24. Main has full
history under `main`; the exact public siblings are Math-
`f532c3e2d91a0c7679a37c733741660917a9b21b` and query-
`fb4c457b5861dc49696fd0a8e11c7c405056c05f`. The seven immutable historical
commits listed in `COLD_START.md` are fetched explicitly without moving HEAD.

The admitted `NodeRunnerTests` class runs in both Python modes; the unadmitted
mutation/control launcher is not invoked. The actual prepared regression then
runs sequentially in normal and optimized Python with fresh mode directories
under `RUNNER_TEMP`, outside every source root. The optimized attempt runs
after a normal failure unless the job is cancelled; a failed command still
fails the job. An always-run artifact step retains the original result files,
TAP and complete fixture trees, including failures. Source/helper/model bytes
are not edited by the workflow.

This small dedicated job was chosen over expanding the unrelated navigation
suite or invoking the unadmitted cold-start control harness. Local validation
is structural YAML parsing with the already installed parser; there are no
new mirror tests or package installations. Root publishes the candidate, and
its actual hosted job must pass before merge. No future hosted outcome is
inferred from the prior 767/767 local results or the intended guard-mutant
failure.
