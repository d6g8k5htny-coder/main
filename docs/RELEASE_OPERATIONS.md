# Release operations and source custody

Use the [current workflow](../governance/OP-WORKFLOW-20260930.md) for task
ownership, proportionate verification and delivery. This page supplies the
details for source downloads and integration; read only the relevant section.
The [formal-verification guide](FORMAL_VERIFICATION.md) and
[required-check contract](FORMAL_REQUIRED_CHECKS.md) retain their evidence gates.

## Download and reproduce

`tools/release_snapshot.py --output <empty-directory>` uses Python 3.11+ and Git.
It reads only the ten explicit public repositories in its allowlist, exhausts API
pagination, and records default commits and open same-repository PR heads. Sources
are archived directly from Git blobs, including export-ignored files; every blob's
Git identity and SHA-256 are checked. Symlinks are archived as link-target bytes,
not followed. Submodule gitlinks are listed, not recursively fetched. No complete
Git history, Git LFS hydration, personal Drive content, model weights, credentials,
or installation on the owner's laptop is implied. API/download failures produce
an explicit partial report and nonzero exit, never an empty successful queue.

`SUMMARY.json`, per-commit manifests and `SHA256SUMS` bind the evidence. A snapshot
is an as-of observation, not a permanently current branch tip. Before extracting
an archive, validate paths and symlinks. Use isolated checkouts for executing code.

The `Release source custody and pinned toolchain replay` Actions workflow runs
on relevant PRs or manual dispatch. It has read-only permissions, immutable action
pins, no persisted checkout credential and no privileged PR trigger. Evidence is
retained for 90 days by the requested artifact policy; download important receipts
for durable custody before expiration. Repository-policy limits can shorten retention.

The separate primary replay job checks out Math- at
`867d9e34b60186ff46ab5b3ecf8ad5ba6a5cc8b4`, installs the existing locked Lean/mathlib
environment, and executes the unchanged source gate, kernel replay, axiom audit
and false-proof controls. Runtime versions and executable hashes are captured.
A workflow definition is not an installation receipt: only an observed successful
run licenses a statement that these tools were installed and executed there.

**Cross-repository receipts:** the unchanged primary gate's `repository` field
records the Actions execution host (`main`), while `checked_commit` belongs to the
checked-out source (`Math-`). Do not pair those fields into a false source URL.
`execution-context.json` separately binds source repository/commit/remote,
execution repository/commit/run and the exact original receipt SHA-256. The binder
checks the live checkout identity and every recorded log hash. Original receipts
are preserved, not rewritten to conceal this distinction. Read the companion
context whenever replay is hosted in a different repository.

## Review and merge

The active integration claim covers branch refresh, readiness changes and merge.
Use the relevant PR's current disposition for ownership and outstanding findings;
follow the handoff and conflict rules in the current workflow.

Read the exact diff, full reviews and unresolved threads, current tests and source
bindings. A successful static check is not a fresh Lean execution. An absent check
is not a passing check. Re-fetch the PR immediately before a merge and use
`expected_head_sha`. A changed reviewed source, hypothesis or consumed dependency
stales its affected review scope. Unchanged scopes may retain their source-bound
reviews after an explicit dependency comparison; fresh integration checks still
bind the actual tested commit.
Do not bypass branch protection or simulate another account's approval. The owner
may delegate engineering integration, but delegation does not create independent
review or satisfy a mathematical premise. Record your actual provider/session and
scope; when agents share a GitHub account, a comment must not be represented as an
independent GitHub approval.

Existing mathematical HOLD/AMENDs remain until their own evidence is reviewed.
Close genuinely superseded engineering PRs with successor links and preserved
history, not by merging obsolete code. Resolve parallel proposals by reusing their
useful components under explicit namespaces and source crosswalks rather than
landing competing registries. Do not race an active writer's branch.

## All current and future agents

Coordinate in the relevant existing PR. Closed
[main #95](https://github.com/d6g8k5htny-coder/main/issues/95) records the formal
rollout and its historical obligations; it is not a mandatory second reporting
destination. Math- #92 and main #177 established the first merged formal foundation.
Math- #93, main #178, sandbox #3 and trial #160 were separately observed proposals.
Main #178 subsequently merged at `eaf12644bdcbf6176a9e216ffb4ab9304f17de12`
during this audit. Preserve that work and reconcile interfaces rather than
silently replacing it or claiming unobserved review. Trial's 4.19.0 experiment
must not be presented as the primary 4.34.1 toolchain. Keep component evidence,
semantic alignment, source custody, and scientific acceptance separate.

An assignment or notification is OFFERED until a recipient posts a source-bound
pickup/result. Publishing this file does not prove every agent has read it.
Optional Blueprint publishing, external prover weights, additional proof-assistant
backends and an independent checker require their own explicit run receipts; they
are not prerequisites to call an isolated scalar Lean proof kernel-checked.

## Compact research deliveries

`python3 -B -S tools/research_delivery.py build --root <payload-directory>
--spec <spec.json> --output <new.zip>` builds a deterministic archive. Use
`verify <archive.zip>` to check it again, including after upload/download. It
accepts explicit file paths, byte counts and SHA-256 identities, not an automatic
workspace crawl. Its canonical handoff has a separate payload digest.

Run `python3 -B -S tools/research_delivery.py --help` for the specification and
bounds. Verification checks archive custody; it does not execute evidence or
certify its scientific claims. Link original execution receipts and unchanged
predecessor deliveries rather than duplicating their complete payloads by default.
