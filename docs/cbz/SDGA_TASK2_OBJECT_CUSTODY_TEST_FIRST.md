# SDGA Task A object-custody review fix: test-first checkpoint

Date: 2026-11-10 UTC. No helper implementation or execution is claimed.

## Accepted current-scope defect

At source head ae6543d04e6dca4514db4ebceec7a77537101407 / tree
0381befeae754eab68805708663f6d8b4edca885, the actual launcher measures object
digests while constructing requests but does not remeasure requested artifacts
after observations or at final acceptance. Repository source remeasurement and
repeated request digest fields do not establish object custody. This violates
the current Task A before/after requirement and is not parked as future work.

The old six-method/39-subcase and pure12 results remain source-bound observations
with this gap unresolved. The optional immutable record
27af7e32ef61ca21b705c02544bda2fb1f163079 may be retained at
cbz/evidence/sdga-task-a-20261110/pre-custody-control-observation.json.
It is not rebound to this checkpoint or represented as post-object verification.

## Exact staged test-only source

- tests/sdga_object_custody_contract.py:
  a9e147e246abfacf9561d286aef65086dedd2c66.
- .github/workflows/sdga-component-contract.yml:
  f22987c60144ae7db6202cae6085402ca2c2f947.
- Baseline launcher d7f72edb1b7dac6458c568e9411edf21f2703476 remains unchanged.
- Component336a35a3, Safe35c407e8, Controlsae25a1bd, the two pure functions,
  original six/39 assertions, all executor labels, pure12 assertions and pinned
  native setup remain unchanged.

The workflow adds only two sequential Python-only observation steps, a separate
original-custody-record printing step and four artifact paths. Removing exactly
those additions reconstructs workflow49580c9d byte-for-byte. Neither existing
native nor pure command/assertion is replaced or disabled.

## Separate method/route census and actual I/O

Three explicit Python-only unittest methods cover six routes:

1. intact: actual child reads unchanged OLean-labelled/IR-labelled/request byte
   files and reports its manifest; acceptance is required. The same method also
   runs an unchanged actual child followed by an intentional observer ValueError,
   requiring the original error plus retained after-manifest.
2. changed-before: actual request bytes differ before context entry; refuse
   before launching a child and never reset the expected snapshot.
3. changed-after: parent changes actual IR-labelled bytes only after the fixed
   child reports its read-ready handshake; refuse after the observation.
4. missing-after: parent unlinks the actual OLean-labelled file after that
   handshake; refuse after the observation.
5. expected-missing: deliberate absence is established before taking the
   expected snapshot; stable intentional absence is accepted.
6. appeared-after: an intentionally missing file appears after the child's
   handshake; refuse without rewriting its expected-missing state.

All files are tiny real I/O byte fixtures, explicitly not genuine Lean objects
or Lean semantic-adverse evidence. The fixed child records original stdout,
stderr, exit and read timing. Parent mutation timing must follow the child-read
timing. After a successful helper implementation these six routes use six
actual fixed child invocations (the pre-entry route launches none; the intact
method uses two). They remain entirely outside the six-method/39-subcase census.

## Required narrow shared helper API, initially absent

The new source requires two public functions in the existing launcher:

- measure_objects(paths): independently measure each normalized absolute path;
  existing rows contain exists=true, bytes and SHA-256; actual missing rows are
  exactly exists=false. Unreadable/unmeasurable files must be errors rather than
  missing states.
- object_custody(paths, expected_manifest, evidence_path): a context manager
  that compares actual before and after observations against the immutable
  expected snapshot and writes a JSON record containing before, after, drift,
  status and observation_error. Before mismatch prevents body/child launch.
  After measurement occurs on every exit; original body failure is propagated
  and retained alongside custody failure. Normal-body drift raises RuntimeError
  beginning OBJECT_DRIFT:. Expected snapshots must never be silently reset.

The existing d7f launcher supplies neither function. Each method first asserts
both helpers exist; actual absence must produce three unittest failures,
zero errors/skips, all six routes not_run and zero child observations per mode.
That is implementation-absence RED only, with no semantic-negative or
mutation-kill credit. Source inspection/readback does not claim this happened.

## Later bounded integration, only after root reads actual RED

After root independently reads the published actual RED, implement those
helpers and wrap the existing fixed inventory/native-control/CLI observations.
Preserve measured request bytes as well as all requested OLean/IR paths.
Baseline measurements must agree with the request's digests; no snapshot reset
may hide earlier drift. Deliberate copy mutations and their request writes
precede separate explicit expected snapshots, including missing-server absence.

Maintain an immutable per-path expectation registry; repeated binding must
agree rather than overwrite. Before-check failure blocks native launch.
Finally after-checks retain original native outcomes even on child failure.
A final registry remeasurement and custody-failure census must actually block
launcher acceptance for unexplained changed, missing or newly appeared files.
Existing source measurements remain separate. Original native six/39 and
pure12 assertions/counts/executors stay unchanged.

Correct only the cited source path in
docs/cbz/SDGA_TASK2_PURE_DEPENDENCY_CONTRACT.md from
src/Lean/Parser/ParseImportsFast.lean to src/Lean/Elab/ParseImportsFast.lean
in the implementation phase. No change to either pure API follows.

Scientific effect NONE; classification/qualification BLOCKED;
execution_authenticated=false. No Task B/C/D, whole-SDGA, general corruption
or hostile-host certification credit is available.
