# Current status of this historical replay evidence

The adjacent delivery, registry, replay instructions and reports are preserved
as historical source-cut evidence. Their historical "verified" labels and
`-B` instructions are not a current admission of this helper as a sound CBZ
Layer A/G0 certifier. Four missing source/test files have been restored from
composite commit `82d9e480a51b946b3e74525ccf32139129d37d28`; their original
manifests and delivery metadata remain unchanged.

Source review found that the historical Python mutation grader accepts a
zero-test baseline, a named failure caused by an unrelated assertion, and a
named failure mixed with a foreign error. Its loader can read pre-existing
bytecode despite `-B`. The historical actual-Lean self-test uses substring
diagnostics and also imports the helper normally. Its finite reports do not
authenticate every import or make a federation-wide claim. The Lean
`lean-verification-final-normal` report identifies a different self-test
source digest; the v3 normal/optimized reports identify the delivered source.

The separate `cbz/replay-qualification` candidate uses source-only loading
and strict structured outcomes for the historical Python mutations, with
independent tests and a separate checksum manifest. It does not repair the Lean
self-test. Its current-head hosted verification is pending. The
`.github/workflows/cbz-replay-delivery.yml` workflow uses a fresh private
bytecode prefix for historical runs and preserves actual logs, but this
mitigation assumes a trusted runner and is not a source-level Lean fix.
See root `TOOLING.md`, `GOVERNANCE.md`, and `STATUS.md` for commands and
boundaries. No new mathematical conclusion is made by this notice.
