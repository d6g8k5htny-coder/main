# Exact-module replay wrapper: independent review

Reviewer: `/root/cbz_kernel_wrapper_review`, 29 October 2026. Scope: the new
`tools/kernel_replay/` adapter, tests, mutation harness, and retained evidence.
This is a tooling review, not a fresh mathematical verification or a full-CBZ
certificate. No mathematical source, Git state, canonical payload, or remote
service was changed by the reviewer.

## Finding P1: resolved object filename could hide private proof data — fixed

The original `resolve_olean` canonicalized the selected `.olean` before
`parts_of` enumerated its siblings. The Lean driver checked the real path, but
the unchanged official `replayFromImports` subsequently searched the lexical
filename again and selected its `.server`/`.private` siblings. An individual
`.olean` symlink could therefore bind three destination parts while replaying
only the exported presentation at the link location. A malformed private proof
could receive a green result.

Locations: `tools/kernel_replay/kernel_replay.py:41`, `:81`, `:98`, `:112`,
`:139`; `tools/kernel_replay/ExactReplay.lean:23`; installed
`src/lean/LeanChecker.lean:11`.

The author confirmed the static finding with a focused real-Lean reproduction.
`tools/kernel_replay/evidence/initial-real-symlink-replay.json` preserves that
false-green observation; it is historical defect evidence, not an acceptance
receipt. The fix rejects symlinked or noncanonical artifact/search paths before
canonicalization, including the main object, sibling parts, declared roots,
and manifest sysroot. Both manifest preparation and run-time validation reach
the guard. Static inspection confirms that the reported path mismatch is
closed. The new wrapper and real-private-body regressions exercise rejection;
the `allow-artifact-symlink` mutant removes the guard to expose the old defect.

## Reviewed behavior

- The production Lean driver imports the installed `LeanChecker` and invokes
  its exported `replayFromImports target` exactly once. It does not preimport
  the target. The official routine imports only the target's declared imports
  into its replay environment, reads exported/server/private parts, and replays
  constants from the most private available part. Production copies no kernel
  replay implementation.
- The wrapper uses one fresh `lean -j1` process per explicit module and waits
  before starting the next. Names, nonempty coverage, duplicates, exact module
  ordering/count, and exact manifest fields are checked. Prefix expansion and
  the official concurrent prefix CLI are not invoked.
- Sysroot libraries precede caller search roots for both driver import and
  replay search. The Python package-shadowing selection matches upstream
  `SearchPath.findWithExt`; the driver additionally checks the selected real
  `.olean` path. The existing checker-shadow regression covers the previously
  discovered search-order defect.
- Manifest source, compiled parts, tools, and explicit context bindings are
  revalidated before, between, and after module runs. Nonzero exit, wall
  timeout, missing/wrong completion marker, or coverage failure prevents green.
  Address-space/core limits are applied in the child; timeout kills its process
  group. Output is file-backed and only a bounded tail enters the report.
- G4 replay and G2 axiom policy remain separate. The unapproved fixture axiom
  can pass the kernel while the fixture-only G2 check rejects it; the report
  explicitly states `g2_status: NOT_ASSESSED` and `full_cbz: false`.

## Evidence reuse and mutation review

The handoff's completed 23 scoped tests were accepted as applicable evidence
for unchanged behavior. No Lean invocation, project build, kernel replay, or
test suite was rerun by this reviewer. New checks for the concrete symlink
defect and changed fixtures/tests were performed by the author, avoiding races.

The initial mutation report is retained as
`tools/kernel_replay/evidence/g0-initial-mutations.json`: 15 planned/executed,
11 killed, four survived. Three binding tests had asserted only non-green and
were masked by deliberately invalid dummy objects; they now require the
specific binding error. The initial private fixture did not truly confine its
bad proof to private data; the replacement uses the normal declaration API
under an explicitly quarantined fixture-only kernel-check bypass. These were
real test-coverage defects and the initial score must not be presented as
acceptance.

A separate static mutation-harness review found no further blocking defect.
The harness isolates mutations, requires exactly one replacement site, and
classifies runner errors, skips, empty test sets, abnormal exits, and outer
timeouts as `ERROR`, not `KILLED`. A kill means an assertion failed; retained
assertion traces must establish the intended failure rather than relying on
the scalar mutation count alone.

Final author acceptance evidence has been inspected and reused:

- `tools/kernel_replay/evidence/g0-tests.json`: 25 tests passed, zero assertion
  failures/errors/skips/expected failures/unexpected successes after the path
  guard and corrected fixtures/tests.
- `tools/kernel_replay/evidence/g0-additional-tests.json`: two additional tests
  passed for search-root/part symlinks and CLI prepare/run/source drift. The
  production adapter was unchanged, so the earlier 25 tests remain applicable:
  27 distinct passing tests in total, not a claim of a repeated 27-test run.
- `tools/kernel_replay/evidence/g0-mutations.json`: 16 planned, 16 executed,
  16 killed; no survivor, runner error, or skip. The retained assertion traces
  match the mutated obligations. The symlink, exported-only, skipped-replay,
  and preloaded-target mutants visibly receive improper green results that
  their negative tests reject. The concurrency trace shows the shared lock
  collision; the memory-limit trace shows the deliberately removed limit.

The author's final production identity is
`kernel_replay.py` SHA-256
`416b688929daff9f473ce3e0abd399d8be2cde46c12956d3e5d5d9937d914293`;
`ExactReplay.lean` SHA-256
`24e55f5e99c83a46f9f908bbc90f22292a94305294e287d9b2eccbf24e8e6b66`.
These identities and the executed evidence came from the author; this review
did not regenerate checksums or rerun verification merely for handoff.

Disposition: the one P1 finding is fixed and covered; no open blocking finding
remains in the reviewed adapter, tests, or mutation harness.

## Limits

Source and object hashes are independent bindings; G3 evidence must establish
that the source produced the object. The caller's independent owned-module
inventory must establish complete expected coverage. Only declared targets
and explicit context are bound, not an undeclared dependency closure. Upstream
unsafe/partial exclusions and axiom acceptance are unchanged. The observed
low-address-space failures remain retained; an 8 GiB fixture limit establishes
fixture operation only, not the memory requirement of a heavy project replay.
Provider-distinct review and other process gates remain separate from this
tooling review and from completed mathematical evidence.
