# SDGA Task A requested-object custody: bounded review fix

Date: 2026-11-10 UTC. This document records source implementation and the
preceding observed RED. Native execution is established only by its separately
bound current-source run records.

## Accepted defect and observed RED

Final branch review found a current Task A defect: requested OLean/IR digests
were captured at request construction but not remeasured after observations or
at final acceptance. Source hashes and repeated request digests did not close
that gap. Old native six/39 results are not relabelled as object-custody evidence.

Root independently read run42752944470/attempt1/job129620599585. Native checkout
2a1639c474d1a8f579d3dc90a5246ac13a62384e has exact tree
fbfc43fb2b26e3f89ce4f23986f7f86d2f2ec0ff and ordered parents
[97663b478822848db5262b2fffcbacd37d6231b4,
d3295713a2a9511ecbe24c3699999c820f5503fc].
Both Python modes ran three new custody methods: three failures for absent
measure_objects/object_custody, zero errors/skips, six routes not_run and zero
child observations. The unchanged native six methods/39 subcases and pure12
methods passed in each mode; source drift was empty. This was implementation
absence only, never semantic-negative or mutation-kill credit.

Twelve exact printed original-record copies are retained in
cbz/evidence/sdga-task-a-20261110/custody-implementation-absence.json, blob
21e586faeffa7cdac1e8f14f590e11b6a4c5ef4f. Their UTF-8 extents match native
headers; SHA-256 values are native declarations, not independently authenticated.
No ZIP extraction or rebinding to this implementation is claimed.

## Exact source and preserved contracts

- tests/sdga_component_contract.py implementation:
  9921a3f91c3778115b91df3b1a0d177fc3ce2483.
- The independent test remains exactly
  a9e147e246abfacf9561d286aef65086dedd2c66: three methods/six routes,
  outside the native six/39 census.
- Workflow remains exactly f22987c60144ae7db6202cae6085402ca2c2f947.
- Component336a35a3, Controlsae25a1bd, Safe35c407e8, pure APIs/tests and all
  existing six-method/39-subcase assertions, IDs and executor labels are
  unchanged. Extracted original assertion lines and CASES agree exactly.
- Pure dependency contract blob 9bc59482c62ee15f66f857c5730acf11f05f21ac changes
  only the cited path Parser/ParseImportsFast.lean to Elab/ParseImportsFast.lean.

Exact UTF-8 blob readbacks were verified. No filesystem/local executor,
manual rerun, proof change or Task B/C production change was used.

## Actual-byte helper and immutable binding

measure_objects independently reads normalized absolute paths. Existing rows
contain actual byte length and SHA-256. Only FileNotFoundError becomes an
explicit missing state; unreadable files/directories and other measurement
errors fail rather than masquerade as missing.

object_custody compares measured before and after manifests to a copied expected
snapshot. Before mismatch prevents body/child launch. Its finally path measures
after every exit, writes both manifests, drift, measurement errors and the
original observation error, and raises OBJECT_DRIFT on normal-body drift.
A pending original exception remains the raised exception, with any subsequent
evidence-write exception retained as its cause. An evidence-write failure is
blocking; no unavailable evidence receives success credit.

bind_request captures request bytes and every requested OLean/IR path after the
request is written. It verifies those request bytes against the supplied record
and each present actual part against its declared digest. The only permitted
missing state is an explicitly declared, actually absent adverse-copy path.
Repeated binding cannot change an existing path's expected state. Request
bindings, declared part digests, intentional absence and exact expected snapshots
are retained rather than overwritten.

C13 captures each separate copy only after missing/reorder/truncation mutations
and its request write. Wrong-resolved-path and unknown-field requests receive
their own bindings after their deliberate writes. Thus intended adverse states
remain stable expected states; any later unexplained change still fails.

## Fixed observations and final acceptance

The existing inventory, native_case and CLI observations use observe_lean and
the same object_custody helper. The fixed source covers eleven observations and
six request bindings. With the current five-part fixture requests these bind
31 paths (request bytes plus OLean/IR), including one explicitly missing server
part. The original native stdout/stderr, exit, timeout and launch outcomes remain
separately retained and linked from every custody record.

Any custody failure prevents further guarded native scheduling. Every
observation retains its own expected/before/after state and original native
outcome. The source-identities record includes the complete object-custody
summary, and individual records plus the summary remain in the external output
directory/artifact. The new independent test source is included in native source
before/after bindings as well as its own regression receipt.

finalize_objects remeasures the complete immutable registry. Unexplained changed,
missing or appeared files, measurement errors, prior custody failures, incomplete
eleven-observation/six-request census or empty setup cannot satisfy its pass
status. The launcher acceptance expression requires that status in addition to
all unchanged native six/39, source-drift and cleanup conditions.

## Validation obligations and explicit limits

Root authorized implementation only after independently reading the actual RED.
The intended current-source evidence is the unchanged native six/39 and pure12
suites plus the independent three-method/six-route custody suite, all in normal
and optimized modes, without skips or source drift. Green runtime records must
show actual per-observation before/after equality and the final registry result.

The independent suite uses real byte fixtures and child-read handshakes. It
covers pre-entry changed request rejection, post-read changed IR/missing OLean
rejection, stable deliberate absence, unexpected appearance, immutable expected
snapshots and original body-error propagation with after measurement. These are
Python I/O custody observations, not genuine Lean object/semantic-refusal or
global qualification results.

The final-registry post-last-observation negative branch is source-verified to
gate acceptance but is NOT runtime-exercised by a separate negative probe.
Simultaneous body failure plus drift is source-reviewed rather than a separate
runtime-negative claim. No mutation-kill credit is inferred from the initial
missing-helper RED. Before/after hashes do not certify concurrent hostile-host
mutation or arbitrary compacted object graphs.

Scientific effect NONE; classification/qualification BLOCKED;
execution_authenticated=false. Task B policy, Task C replay, complete IR and
the production 99-module observation remain outside this fix.
