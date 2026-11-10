# Pages publication and atomic-head observations

Observed 10 November 2026. These are separately timed observations and do not
establish CBZ-0, any mathematical result, scientific review or a successful
recovery. No receipt branch write is made by this record.

## Failed publication and current site

Main run 42747646763, source commit
`31bc4c1f5b66e92bfbf327366956944ea691289c`, completed with failure.
The full hosted build log `site-build-129605863089.log` is the original
GitHub job 129605863089 output, captured through the job-log API. Its restore
step exited 1 after public readback could not verify the prepared receipt;
the deploy job 129609918709 was skipped. The job also reported that no browser
artifact was present. Its controls and node-status refresh jobs succeeded.
The prior failed deploy attempts of run 42737966287 and their evidence remain
in the separately owned contribution archive; they are not replaced.

## Recovery candidate and custody

Main #801 corrected head `84f53afa766853b749e0ed127043d1954e70f7c3`,
tree `98a4032b79ada6a74cfe58c60717eaa819ae8817`, matches the locally tested
tree of `bac3d4a3e161f49ed31a24ac6e10ae766236e151`. Its hosted controls job
129609355495 succeeded on tested merge
`82a62e8629314511ce4076186325911436ba50f4`; its original full log is
`recovery-controls-129609355495.log`. The job uploaded artifact 13731748998,
reported SHA-256
`f1f54d21ac4a1eb97b9c167deb7f5b980553b415bea77f2ed20cfa206ca97f81`.
The artifact has not been independently unpacked in this observation. The
recovery job was skipped on the pull request; no receipt was modified.

Before that hosted run, the exact local workflow block ran 34 tests without
skips in each Python mode, plus 27 source and 10 workflow selected-oracle
mutations and three rejected subject/helper/lifecycle assertion origins per
mode. The initial recovery head missed the ancestor-rollback race; that actual
failing test and two unsuccessful intermediate mutation campaigns remain
historical observations in the contribution record. The corrected source uses
the fixed receipt ref, GitHub GraphQL `updateRefs`, exact `beforeOid` and
`force: false`. Authenticated GraphQL and Pages cancellation availability
and actual dispatch have not been established.

## Normal publisher defect

The separate normal publisher's non-force REST update has the same
ancestor-rollback race. The added test
`tests/test_current_site_publication.py::PublicationTests.test_rollback_between_read_and_ref_update_requires_exact_head`
actually failed locally in normal and optimized Python modes: the expected
`ValueError` was not raised. It models a verified single-parent ancestor
chain, a rollback between the guard and the write, and acceptance of the
descendant by a non-force update. The native local red commit was
`5843d89c53f9da7de7dae5df424f3e4a8218391f`; its exact tree
`68f3492aea322195c860fe73a3c8830367215842` was reconstructed from the
published Main base and is the tree of this draft's initial hosted head.
The red test remains an active defect, not a passing certification.

## Workspace and durable checkpoint

Workspace #69 merged as `d8d8b16af35b3f6d3d4b454ce98e26d7c3b81ab0`,
tree `0bb00bcefff98806513b2340faa07b0912a7ca3d`, matching its tested tree.
Its post-merge pins run 42749081900 and federation-contract run 42749081931
both completed successfully. The earlier source-specific hosted artifact and
its limited mutation controls are recorded in the contribution checkpoint;
these observations do not constitute a final same-pass federation gate.

The separately owned contribution supplement is version 26. The independently
read-back evidence archive is version 38, 9,725 ZIP members, SHA-256
`22cda645329c1c59e54fcd8619dc5e10ec0425ff12b4ded217f9092ac7840643`;
its 9,724 member hashes and CRCs and prior data retention were checked. The
shared original checkpoint and archive were not replaced. The local execution
environment became unavailable after the red test; this native Git evidence
keeps the source and hosted logs recoverable without relying on that path.

## Later hosted red and candidate failures

The exact original publisher test was also run in Main current-site controls
job `129610695154` at tested merge `ed5bc5d`; the normal run reported
229 tests and one failure, `ValueError not raised`. The full original job
log is `publisher-red-controls-129610695154.log`. Its optimized step was not
reached after that failure; the separate local optimized failure is recorded
above.

A later source candidate at head `a6a17b5a5442f031052b81bc9fe25276158db181`
was not accepted: its publisher controls job `129612202807` ran 40 normal
tests and reported one error because the synthetic no-retry client lacked a
token. Its stacked recovery controls job `129612202618` also failed and
uploaded partial evidence artifact `13733291004` (SHA-256
`cce0abe451fb97b47958f7b75579353afb17d96ea68676a7e0b26ed9ce17a939`).
The two complete hosted job logs are preserved alongside this file. The
recovery job's redirected internal output was not independently read at this
observation; its exact cause is not inferred. A later draft adds failure-log
printing and corrects the token fixture. New head tests and mutations must run
before any passing or integration claim.

## Stacked correction and hosted diagnostics

The publisher draft was retargeted to the #801 recovery branch at
`84f53afa766853b749e0ed127043d1954e70f7c3` to test the two related
interfaces together. Its first proposed repair is not qualified by the
failed jobs above. A following source revision supplies the missing
synthetic token and prints redirected recovery-control failures on error.
New hosted tests are required; changing the branch base or source is not
itself evidence of success. If #801 merges, retarget this draft to the new
Main tip and recheck its actual composition and required checks.

## Scoped hosted controls after fixture correction

At tested merge `d243e1a2c833e5039b9cd0d1ceb9aa5682c78e35`, tree
`dafca516877750f5e06603729e862ca9e7420b7a`, publisher controls job
`129613200124` completed successfully. Its full log
`publisher-cas-controls-129613200124.log` records 40 tests without skips
per mode, 11 clean-baseline/designated-oracle mutations, three rejected
unauthorized assertion origins and one rejected runtime error per mode. It
reported artifact `13733117677`, SHA-256
`8c84b2b1fb2a9ced407ac72e7994a973d54313bbcbb6509e171587c5780822b3`.
The artifact ZIP was not independently unpacked in this observation.

Stacked recovery controls job `129613199744` also completed successfully
on that tested merge; its full log is
`stacked-recovery-controls-129613199744.log`. It reported artifact
`13733396297`, SHA-256
`a8228247988dfad4d15a0ce345b7614249c654b54270cd786e5315ca87a9b182`.
The recovery-dispatch job was skipped on this pull request. These controls
do not establish authenticated ref mutation, Pages cancellation, public
recovery, scientific acceptance or the final federation gates.
