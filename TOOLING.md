# CBZ mission-tool registry

**State: component registry; the final federation toolchain is incomplete.**

This file lists runnable mission-relevant tools present in the Main revision
that contains this file. Its candidate source parent is
`8687f4cfe8adab1c347b50cebf619069f6d14574`, based on Main default
`299c7fad72f1079b87b2d044b2d30f2098678f29`. Each entry states purpose,
command, verification and scope. A component passing in isolation does not
complete a CBZ gate or the final G0–G11 matrix.

## Formal gate checker

- **Purpose:** run the Main formal checks, including the pinned Lean toolchain,
  exact source checks and retained negative controls documented by the formal
  contract.
- **Command:** `python -B tools/formal_gate_check.py --run-lean`
- **Verification:** follow
  [docs/FORMAL_VERIFICATION.md](docs/FORMAL_VERIFICATION.md) and
  [docs/FORMAL_REQUIRED_CHECKS.md](docs/FORMAL_REQUIRED_CHECKS.md); retain the
  actual build, kernel, axiom and negative-control output.
- **Scope:** Main's documented formal component only. It does not run the full
  federation, prove scientific alignment or substitute for G0–G11.

## G11 bounded census

- **Purpose:** report Python stubs, code-comment markers, no-op workflow steps
  and missing static documentation/test references in one tracked Git tree.
- **Commands:**

  ```sh
  python3 -B -S -m unittest tests.test_g11_census
  python3 -B -S -O -m unittest tests.test_g11_census
  python3 -B -S tools/cbz/g11_mutation_selftest.py
  python3 -B -S -O tools/cbz/g11_mutation_selftest.py
  python3 -B -S tools/cbz/g11_census.py --rev <commit> --json census.json
  ```

- **Verification:** the unit suite and mutation self-test must pass in both
  modes; a census exit of 0 means only that this implementation found no
  production finding in its declared paths and rules.
- **Scope:** static bounded triage. It does not establish runtime behavior,
  semantic placeholder absence, Layer A clearance or G11 admission. Full
  limits are in [docs/cbz/G11_AUDITOR.md](docs/cbz/G11_AUDITOR.md).

## Federation drift crawler

- **Purpose:** discover the federation from the Workspace seed and check pin,
  registry, topology and literal Lean-interface consistency.
- **Commands:**

  ```sh
  cd tests
  python3 -B -m unittest test_cbz_drift
  python3 -B -O -m unittest test_cbz_drift
  cd ..
  python3 -B -S tools/cbz_drift/crawl.py \
    --seed d6g8k5htny-coder/Universal-Law-Workspace \
    --search <checkout-directory> --fetch --at pins --json drift.json
  ```

- **Verification:** both synthetic-federation suites, including the registered
  source mutants, must pass. A current crawl must retain BLOCK and INFO output.
- **Scope:** source-level pin and interface drift under the documented parser.
  It does not elaborate Lean, expand macros, prove declarations or complete
  G2–G4. See [tools/cbz_drift/README.md](tools/cbz_drift/README.md).

## Empirical statement check (G12 component)

- **Purpose:** exact-arithmetic sampled checks for registered Lean statement
  classes and finite discrete models.
- **Commands:**

  ```sh
  python3 -B -S -m unittest tests.test_cbz_statement_fuzz -v
  python3 -B -O -S -m unittest tests.test_cbz_statement_fuzz -v
  python3 -B -S tools/cbz/statement_fuzz.py --package formal --selftest --json report.json
  ```

- **Verification:** the full registered self-test and mutation contract must
  pass in both modes; unsupported constructs remain named and fail closed.
- **Scope:** bounded empirical corroboration only. PASS and CORROBORATED are
  not kernel proof or scientific acceptance. G12 is registered context, not a
  replacement for the required final G0–G11 pass.

## Additional active bounded tools

### Architecture producer identity binding

- **Purpose:** bind producer output to current commit, repository, run,
  attempt, receipt and artifact identity strings.
- **Command:** in the hosted ephemeral runner,
  `python3 -B -S tools/architecture_check_binding.py --expected-commit "$GITHUB_SHA" --expected-repository "$GITHUB_REPOSITORY" --expected-run-id "$GITHUB_RUN_ID" --expected-run-attempt "$GITHUB_RUN_ATTEMPT" < producer-input.json > producer-binding.json`.
- **Verification:** `tests/test_architecture_binding.py`, run in both modes by
  `.github/workflows/research-architecture.yml`.
- **Scope:** validates identity strings from at most 64 KiB of stdin. It does
  not validate archive bytes, receipt contents, mathematics or scientific
  status.

### Current inventory acquisition projection

- **Purpose:** produce deterministic, all-or-nothing current
  content-association annotations without mutating the supplied inventory.
- **Command:** import `project_current_acquisitions` from
  `tools/current_inventory_acquisition.py`; there is no standalone CLI.
- **Verification:** `tests/test_current_inventory_acquisition.py`, included
  in both `test_current_site*.py` modes by
  `.github/workflows/current-site.yml`.
- **Scope:** accepts only the registered Main and Math- repositories and
  complete-file byte counts no greater than 16 MiB. It does not perform
  scientific review, applicability, acceptance or graph binding.

### Current-site publication custody

- **Purpose:** retain prepared and observed Pages receipts on the fixed
  `codex/site-publication-receipts` metadata-only branch.
- **Commands:** `.github/workflows/current-site.yml` runs
  `python3 -B -S tools/current_site_publication.py restore`, then
  `package`, `prepare` and `observe` with the workflow's bounded paths.
- **Verification:** `tests/test_current_site_publication.py`, included in
  both `test_current_site*.py` workflow modes.
- **Scope:** requires Main, `refs/heads/main`, the tested checkout and a push,
  schedule or dispatch context. It does not execute downloaded code, change
  the source branch or force-update a ref. Receipt members are bounded as
  declared by the tool.

### Dependency source bindings

- **Purpose:** record the pinned Math- path, blob, byte and digest identities
  behind dependency-atlas node fingerprints.
- **Command:**
  `python3 -B -S tools/dependency_source_bindings.py --math /path/to/Math- --check`.
- **Verification:** `tests/test_dependency_source_binding.py`.
- **Scope:** read-only Git plumbing against the Math- checkout pinned by
  `docs/site/dependency-source/PROVENANCE.json`. Digest agreement is byte
  identity only, never proof validity, independent review or scientific
  acceptance. Omitting `--check` writes the configured fixture.

### Formal source/execution context binding

- **Purpose:** bind an unchanged successful formal receipt and its log digests
  to separate Math- source and Main execution repositories.
- **Command:** the hosted replay runs
  `python3 -B -S ../operations/tools/formal_execution_context.py --source . --operations ../operations --expected-commit 867d9e34b60186ff46ab5b3ecf8ad5ba6a5cc8b4 --run-id "$GITHUB_RUN_ID"`.
- **Verification:** `tests/test_formal_execution_context.py`, run in both
  modes by `.github/workflows/release-snapshot.yml`.
- **Scope:** accepts only Math- as source and Main as execution repository,
  requires a kernel-checked receipt and writes a separate execution context.
  It is custody, not independent alignment or scientific acceptance.

### Periodic H0 pilot custody

- **Purpose:** retain bounded raw output, source identity and failure state for
  the hosted periodic H0 pilot without scientific promotion.
- **Commands:** `.github/workflows/periodic-h0-pilot.yml` runs
  `python -B -S tools/periodic_pilot_custody.py init`, `run`, and under
  `always()`, `seal`.
- **Verification:** `tests/test_periodic_pilot_custody.py` in both modes and
  `tests/test_periodic_pilot_custody_oserror.py`.
- **Scope:** requires the declared hosted-pilot context and enforces the
  source/output/log bounds in the tool. Runner identity is claimed, not a
  native API attestation; this is custody, not archive readback, independent
  review or a theorem verdict.

### Site-graph output capture

- **Purpose:** retain bounded original stdout/stderr prefixes and observed byte
  counts and hashes for the hosted site-graph control lane.
- **Command:** `.github/workflows/site-graph-projection.yml` runs
  `python3 -B -S tools/site_graph_capture.py --mode "$CONTROL_MODE" --output "$RUNNER_TEMP/site-graph-projection"`
  in normal and optimized modes.
- **Verification:** `tests/test_site_graph_capture.py`; the child suite is
  `test_site_graph*.py`.
- **Scope:** trusted hosted-runner argv only, with the documented time, stream,
  aggregate, metadata and package bounds. It is output custody, not a sandbox
  or cumulative-emission guarantee; overflow, timeout or incomplete
  collection is refused even when the child exits zero.

### Current-site native registry APIs

- **Purpose:** validate and construct native registries and current bundles
  from already admitted bytes and immutable metadata.
- **Command:** use `validate_native_registry`,
  `admit_current_native_bundle`, `build_native_registry` or
  `validate_native_progress` from
  `tools/current_site_native_registry.py`; no standalone CLI is present.
- **Verification:** `tests/test_current_site_native_registry.py` and
  `tests/test_current_site_native_base.py`, run in isolated normal and
  optimized modes by their two named hosted controls.
- **Scope:** does not acquire artifacts, rerun archive admission, establish
  custody, publish, restore files or confer scientific authority. Refusals are
  `ValueError`; registry limits are documented by the module.

### Google Drive identifier scan

- **Purpose:** locally scan added range text, in-range commit messages and
  patches, and an optional pull-request body for Drive-shaped identifiers.
- **Commands:** `python -B tools/privacy/drive_id_scan.py --base BASE --head HEAD`;
  optionally add `--body pr.txt`, or use `--body -` for stdin.
- **Verification:** `tests/test_drive_id_scan.py`.
- **Scope:** local reviewer tool `OPPRIV-SCAN-01`, not a required check. It
  changes no workflow or settings and has no scientific effect. Matches are
  redacted to four characters; exits 0, 1 and 2 mean clean, hit and invocation
  or Git failure respectively.

### Semantic-memory Task1 fixture control

- **Purpose:** produce Task1 structural reports, verify one report or perform
  the frozen benchmark admission.
- **Command:**
  `python3 -I -B -S tools/semantic_memory_experiment.py prepare-report --fixtures experiments/semantic_memory/fixtures`;
  the other admitted modes are `verify-report` and `admit-benchmark`.
- **Verification:** `tests/test_semantic_memory_benchmark.py`, run in isolated
  normal and optimized modes by
  `.github/workflows/semantic-memory-experiment.yml`.
- **Scope:** Task1 synthetic fixtures only, not semantic retrieval or
  installation. It loads only the adjacent reviewed benchmark module, with
  the source and output limits declared in the wrapper.

### Native unittest timing probe

- **Purpose:** run native `unittest` while writing an ordered, flushed JSONL
  sidecar of invocation, per-test timing and native outcomes.
- **Command:**
  `python -B -S tools/unittest_timing_probe.py NEW_JOURNAL discover ...`;
  the journal must not already exist and remaining arguments are native
  `unittest` arguments.
- **Verification:** `tests/test_architecture_unittest_timing_probe.py`,
  included in both `test_architecture_*.py` hosted modes.
- **Scope:** adds no timeout, retry, filter, signal handler or scientific
  verdict. Flush is not `fsync`; this is a trusted-process diagnostic, not
  hostile-test containment or attestation.

## Restored bounded replay controls

- **Purpose:** rerun the retained Python validator and three registered mutants
  in both interpreter modes for the preserved additional kernel evidence.
- **Command:** from `cbz/evidence/kernel/additional/replay-tool/`, run
  `python3.11 -B -S run_mutations.py --output <new-directory>`.
- **Verification:** preserve the generated result and logs; compare their
  source binding with the README and retained historical outcome.
- **Scope:** this validates the documented Python replay tool. It neither
  authenticates a historical execution nor performs a Lean kernel replay.

## Current-tree G11 residue retained

The workflow-scope defect recorded by the preceding canonical-surface change
was repaired and merged normally as Main #762,
`c4db7562640817ab5f23119571030851b964472c`. Its exact-head controls passed
39/39 in both modes and killed 76/76 designated mutants in both modes,
including a new adverse control that removes the step-scope guard.

A current-tree census was then run in both modes:

```sh
python3 -B -S tools/cbz/g11_census.py --rev HEAD --json /tmp/g11.json
python3 -B -S -O tools/cbz/g11_census.py --rev HEAD --json /tmp/g11-opt.json
```

At #762's qualified head, both modes produced byte-identical reports
(SHA-256
`86b7139fc1c2d308c1746132e1d3dd070fc6c393c4e0ba9a1aa0223d35c6feb2`).
The seven false `defaults.run` findings were absent, while 55 documentation or
static-exercise findings remained. Main #763 closed the seven
documentation-only gaps. The subsequent review-repaired attribution source
`8687f4cfe8adab1c347b50cebf619069f6d14574` (parent
`adb3dc69e1d29ab78524b98f151207db93580bec`, tree
`d3b4b4b5acec0e3c3311afe9d4b6d649bb2a40bd`) recognizes explicit local path
bindings only when delivered to trusted Python execution/loading APIs,
reachable forwarding wrappers, or named Node child-process imports. Its blobs
and tree exactly match local qualification
`55e5638d14ecb687fb2a68f981ec3530184e9ee1`. Fresh native execution passed
201/201 controls and killed 268/268 designated mutations in both modes; the
sealed mutation-inventory SHA-256 is
`3fadcad54079e33cb6c60623cad3ec33a3fe896199eac5b188a167f8238e0fce`. The exact
normal/optimized mutation logs are byte-identical at 32,155 bytes with SHA-256
`2fdf38942365217697cffd086cd535f99329244d7575d7f06d62d7497560e050`. The exact
native normal and optimized censuses were byte-identical (SHA-256
`2d73a9cfd2d01cbccf5d73192238ab139f21d07f6e4d4e005f030ee56a8dfafc`,
56,013 bytes each),
with zero production findings and exactly 32 no-current-exercise or documentation tool
candidates; both runs exited 1. Adverse controls refuse mere path
credit from helper-mediated writes on unresolved Python branches, quoted
JavaScript import text and unreachable JavaScript URL initializers. The
function-scope URL control invokes the defining helper so scope leakage remains
independently mutation-tested. Those three repairs account for review/adverse
defects 141–143. An additional adverse control maps an
aggregate mutation through a helper parameter back to the caller argument on
an unresolved branch; a paired positive control proves local parameter
rebinding does not leak. Three additional review controls require all enclosing
JavaScript branches to select a call, require a proven first iteration for
`while`/`for` bodies, and reject process calls after direct helper
`return`/`throw`; the duplicate parameter-alias review is preserved but counted
once. Two older unresolved review findings additionally capture an augmented-
assignment target before RHS helper effects and defer JavaScript parameter
default initializers until invocation. Three older unresolved findings now
invalidate trusted Python imports after possible branch rebindings, terminate
JavaScript loop tails after direct `break`/`continue`, and reject process-call
credit inside unsupported `switch` selection. The distinct defect count is
152. Adverse controls
also refuse mere path
credit from stale assignment-target effects, unresolved comparison suffixes,
unresolved eager-comprehension effects, uninvoked JavaScript method bodies and
literal-true dead `else` arms. They also refuse mere path
credit from unknown Boolean/wrapper branches that may mutate provenance,
incomplete URL initializers, unselected JavaScript conditional arms,
non-string aggregate conversion, helper calls missing required arguments, and
module-name fallback through an enclosing closure binding. They also refuse
mere path credit after unknown conditional-expression effects, calls that
cannot bind a helper or wrapper signature, assignment-expression import
rebinding, tuple `+=` alias conflation, unbraced literal-false JavaScript
branches, concise arrow bodies, quoted URL declarations, or mutable URL-object
alias escape. Positive controls cover annotation-only imports, bounded
`unittest` lifecycle order, and positional, keyword, default, `*args` and
`**kwargs` aggregate identity. General lifecycle mechanisms and dynamic
language semantics remain outside this bounded static analyzer. They also
refuse mere path credit from multiline source-line resorting, possible loop writes, class or
repeated-helper frame leakage, subprocess shell/positional/executable
ambiguity, a shadowed JavaScript `URL` constructor, mutated URL properties and
unproven JavaScript short-circuit calls. The supported subprocess surface is
one ordinary argv positional argument with `shell` absent/false/none,
`executable` absent/none and no `**kwargs`; unsupported forms fail closed.
URL construction must resolve to the unshadowed intrinsic and remains valid
only until the binding or a rooted property may be written. Repeated Python
helper calls receive fresh local bindings, imports and nested definitions while
declared global/nonlocal effects retain their documented destinations. Adverse
controls also refuse mere path
mentions retained through exception-boundary helper/wrapper identities or
exception target bindings; unsupported literal comprehension sources; scripts
after unknown, attached command-mode, or exit-only interpreter options;
assignment-expression and nested-aggregate alias staleness; decorated
forwarding wrappers; and URL names delivered only in child-process options.
They also refuse mere path
mentions caused by stale bindings after unproven exception branches, wrapper
discovery in those branches or in empty/unconsumed comprehensions, execution
after a terminating `finally`, outer JavaScript URL provenance hidden by a
destructured lexical binding, or helper-local treatment of a `nonlocal` static
assignment. They also refuse mere path
mentions through generator-method creation, comprehension-target leakage and
unreachable Boolean operands during wrapper discovery. They also refuse mere path
mentions, fake same-named APIs, interpreter command-mode arguments, unreachable
helpers, arbitrary unknown branches and JavaScript same-line decoys outside the
selected child-process call. Filesystem-root traversal terminates and reachable
function-local process imports remain creditable. Destructured targets receive
only corresponding resolved values; unsupported shapes are left unresolved.
Reachable parameter rebinding invalidates wrapper forwarding; unresolved local
shadows do not fall back to module values; and quoted or property-only
JavaScript names are not expression evidence. Forwarded commands receive the
same interpreter command-mode filtering as direct calls, augmented assignment
replaces or clears stale static values, and property calls cannot impersonate
trusted named imports. Unknown command elements retain position, so an unknown
prefix refuses attribution while a dynamic suffix after a proven tool does not
erase the proven prefix. Boolean `or` is left-to-right, local callees cannot
inherit module wrappers, quoted JavaScript calls remain data, and resolved
starred elements, `unittest` lifecycle methods and trusted loader wrappers are
recognized. Class methods do not register as bare module wrappers. A known
conditional expression selects only its taken command branch; unknown
alternatives receive evidence only from the intersection of every branch. An
intervening JavaScript assignment invalidates an earlier URL binding. The
entire argument list of `pathlib.Path` participates in resolution, element or
slice assignment clears its root static aggregate, and a function parameter
shadowing a named JavaScript child-process import blocks attribution within
that function. Interpreter mode switches are interpreted only before the
selected script; wrapper element writes invalidate forwarding; helper-local
invalidations remain unresolved instead of inheriting module bindings; and
JavaScript default initializer references do not shadow their imported
callees. Recognized interpreter options consume their operands before script
selection, command-list element writes invalidate every resolved identity
alias, and a function parameter shadows a same-named JavaScript URL binding
within that function. Supported list-mutator calls invalidate command provenance
and all resolved aliases; same-function JavaScript declarations shadow named
child-process imports; and local URL declarations cannot escape their function
scope. These are finite static rules, not general runtime semantics. The
scanner also invalidates shared wrapper-parameter provenance through aliases,
rejects calls through a wrapper name after that binding is reassigned, and
checks JavaScript declarations in every lexically containing function. The
wrapper scanner now stops after unconditional `return`/`raise` statements, and
deferred callers carry their actual module activation point into module-wrapper
availability and reassignment checks. The
scanner also clears a previous forwarding registration when a later
same-named function definition is non-forwarding, preserves list identity
across `+=` for alias-aware mutations, and refuses JavaScript process calls in
literal-false branches or uncalled named helpers. The
scanner evaluates deferred Python bodies in activation order and requires
JavaScript helper calls to have a reachable module-rooted invocation chain.
Only callbacks passed to an imported `node:test` identity are recognized as
test roots; this restores the actual museum-conditional test path while an
arbitrary callback API remains untrusted. The temporary 30-tool result remains
historical. The
scanner also invalidates branch-written bindings at a supported `is None`
join rather than combining mutually exclusive states. An imported `node:test`
callback root requires both a reachable registration call and an unshadowed
registrar identity; helper-local or nested registrations without a reachable
module chain remain untrusted. The
scanner analyzes supported cache-guard alternatives in isolated fail-closed
binding environments and expands reachable Python helper effects at the call
position before continuing the caller. Declared Python `global` writes target
the module environment; element/slice deletion invalidates resolved command
aliases; and rebound JavaScript helpers do not provide reachability. The
scanner also invalidates wrapper forwarding on deletion, excludes declared
globals from helper-local static shadows, and resolves JavaScript helper calls
against the nearest lexical declaration. Each rule is covered by a dedicated
control and designated mutant. The
scanner also applies Python argument helper effects before enclosing-call
attribution, rejects reassigned Python helper bodies, and validates deferred
JavaScript module URL bindings at reachable activation points. The
scanner captures each Python argument's static value at evaluation while
invalidating shared list objects after later mutation, and evaluates assignment
RHS helpers before target rebinding. It likewise captures list and tuple
elements in left-to-right evaluation order and recognizes module-level
JavaScript lexical-block declarations as shadows of imported callees. Captured
sequence values also propagate through supported concatenation; ordinary
helper traversal stops after definite termination, including selected literal
branches; and destructured lexical declarations participate in JavaScript
callee shadowing. Named concatenation operands retain their evaluation-time
values across later operand effects; selected branch traversal stops after
definite termination inside the branch; and semicolonless lexical declarations
shadow imported callees under ASI. Repeated sequential Python helper calls are
each expanded while active-call identity bounds recursion; declared `global`
reassignment invalidates the module helper used by later calls; and literal
Boolean short-circuiting excludes an unreachable operand. Literal conditional
expressions traverse only a proven selected arm, while coroutine creation does
not execute an async helper body; general coroutine activation remains outside
the bounded scanner. Async forwarding wrappers and generator functions are not
executed by object creation; comprehension bodies require proven iteration,
generator expressions require consumption, chained comparisons respect a
proven false prefix, and lexical JavaScript URL bindings remain block-bounded. The
stricter source newly exposes `tools/atlas_formal_source_bridge.py`; the former
28-tool result remains historical and is not rebound. Static evidence is not
runtime proof. Exact dormant dispositions and the final orchestrator remain
unfinished; G11 is FAIL.
Forwarding-wrapper conditional expressions now follow only a proven selected
arm. Declared `global`/`nonlocal` wrapper reassignments target their destination
scope in execution order; lambda creation evaluates defaults but not its body;
and declared `nonlocal` helper reassignment invalidates the nearest enclosing
active helper. Dedicated adverse controls and mutants cover these finite rules.
Wrapper forwarding is registered only after call arguments have applied their
effects; chained comparisons stop after a proven false prefix in wrapper
discovery; and exception handlers/`else` bodies remain uncredited without
proven exception flow while `try`/`finally` bodies are retained.

## Missing final orchestrator

No documented command in the current default runs G0 through G11 over one
common federation snapshot and emits one source-bound receipt. Therefore tool
criterion 9, cold-start criterion 11 and the final-pass requirement remain
unawarded. A future orchestrator must fail closed, preserve every failure,
record exact sources and tool versions, and carry its own adverse tests and
mutations before admission.
