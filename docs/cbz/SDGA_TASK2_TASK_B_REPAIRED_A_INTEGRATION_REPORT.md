# Task B integration onto repaired A — source only

Status STAGED_SOURCE_ONLY. No native execution, parsing/compile result, RED/GREEN,
publication or qualification claim. Scientific NONE; qualification/classification
BLOCKED; execution_authenticated false. No filesystem/executor/wait, refs, commits,
PRs, workflow runs, comments, production code or subagents were used/created.

Base actual PR797 head30ee811900b81c41b07650f68aa5b6db2f353a95,
treec048562421a3d0772bbc2bd3fe7898a9e676b086 (A on Main984).
Rechecked branch head stayed30ee811900b81c41b07650f68aa5b6db2f353a95.
Read exact repaired A launcher9921a3f91c3778115b91df3b1a0d177fc3ce2483.
Integrated proposed tests/sdga_component_contract.py:
54421c70918fdebf6b676cd8cfd49cde0c9d67b9.
This is surgically edited A9921, never wholesale replacement by old B709ff5.

Reviewed B Controls42a80b09a78ab7777469d8966c6e46ac9bc748f1 and all four
fixture blobs are unchanged:
- Fixture/BadDependency.lean aa7b27902b38fe805f49e11b71df7cc3c48ff5da
- Fixture/AxiomUse.lean 9fd2e71b4025d53b05ee3caeb1c26f7066f6ddce
- Fixture/Skipped.lean e68e3203646d082a04afd242f17821e61587892a
- Fixture/OwnedAxiom.lean c08550a94286998dd88c29a7f77794e0acfcdf0b

Protected repaired A artifacts remain exact:
- tests/sdga_object_custody_contract.py a9e147e246abfacf9561d286aef65086dedd2c66
- .github/workflows/sdga-component-contract.yml f22987c60144ae7db6202cae6085402ca2c2f947
- tools/sdga/Component.lean336a35a3b916196e4f5d90d7c2c55be63d073c90
- tools/sdga_scope.py14d4617b523c29007da4bdb95ee6ee4dc588a86c
- tests/test_sdga_component_protocol.py8809f1974998f9094a0f1239ecca56ddcf272a88
- frozen profile4c6b4ea59eb07a17ebfed536b07688c53c01eb64

Expected combined census remains10 methods/52 subcases =46 Lean+6 Python_controller.
Original A6/39 is preserved, pure12 unchanged. Separate custody3 methods/6 routes
and process cleanup remain outside that census. Every original repaired A test
method body and all four reviewed B test method bodies are exact unchanged bytes.
The entire original A case map39 is exact retained as TASK_A_CASES.

Surgical source evidence:18 uniquely anchored edits, reversed in order, recover
exact A9921 source string. The reviewed B helper/test block receives only two
I/O wrapping edits; reversing those recovers exact reviewed B709 block.
Of38 original top-level/method function blocks,32 remain exact.
Only __init__, capture_sources, finalize_objects, bootstrap, flush and main change,
for fixed B state/scheduling or fixed combined metadata/census.
A measure_objects, object_custody, bind_request, observe_lean, command,
cleanup_group, cleanup_only_preflight, native_case, accept_rows and request_for
remain byte-exact, including registry, final drift, exception retention and cleanup.
The immutable transfer manifest includes full before/after patch operations,
reverse-preservation results, independently enumerated counts and44 sourceBindings
covering existing scoped A histories/source artifacts plus B fixtures.

B integration changes:
1. A single fixed TASK_B_FIXTURES tuple schedules BadDependency, AxiomUse, Skipped,
   OwnedAxiom. Sequential compile order stays BadDependency, Safe, AxiomUse,
   Skipped, OwnedAxiom, then Component. Failed BadDependency blocks AxiomUse.
2. Successful BadDependency compile writes one fixed binding-only request and
   immediately calls unchanged A.bind_request. This records request bytes and all
   five immutable parts without inventory, observation or subcase credit.
3. Three existing B native inventory request builders write then immediately call
   A.bind_request. Inventory subprocesses now call A.observe_lean with those
   exact request paths. AxiomUse also includes the BadDependency binding as its
   actual imported dependency. Skipped imports core Init; OwnedAxiom has no
   imports, so neither claims that unrelated dependency.
4. Controls C04/C05/C06 and four separate C07 calls now call A.observe_lean.
   C04 guards AxiomUse plus BadDependency; C05 guards Safe; C06 guards OwnedAxiom;
   each C07 call guards Skipped. No request/file measurement is reset at observation.
5. Existing fixture rglob source capture includes all four B sources; protected A
   custody-test capture remains. Combined fixed final census is21 observations,
   10 request bindings and55 paths. Expected values are literals from the
   independent enumeration below, never assigned from observed values.

Independent source enumeration:
A11: inventory; C01; C02; C03; C12; C13 missing-server/reordered-parts/
truncated-private; C15 wrong-resolved-path; CLI unknown-mode/unknown-field.
B10: three actual module inventories; Controls C04/C05/C06; four C07 calls.
Total21. BadDependency binding-only adds zero observations.
A6 requests are baseline, three C13, wrong-path, unknown-field.
B4 requests are BadDependency, AxiomUse, OwnedAxiom, Skipped. Total10.
A unique paths31 =6 request files+5 Safe roots×5 parts.
The roots are owned and four copied adverse roots (three C13 plus wrong-path);
unknown-field reuses baseline parts. B24 =4 request files+4 unique roots×5 parts.
Total55. Parts are .olean, .olean.server, .olean.private, .ir.sig, .ir.
Only existing C13 missing-server uses intentionally_missing. No new absence
exception is introduced; one intentionally absent path remains in the census.

Semantic failure versus custody:
A.observe_lean wraps and returns the native command result; a healthy subprocess
returning nonzero still records unchanged file custody. C07 missing production
guard therefore remains a nonpassing specific native helper-absence observation,
not fabricated OBJECT_DRIFT. accept_rows remains outside custody and preserves
native failed rows/streams. Both C07 guard invocations remain independent.
Actual production guard is still absent; no production implementation was staged.
The approved Controls typed adapter automatically invokes exact future
SDGA.guardReplayConstant : ConstantInfo → Except AuditError Unit when present.
Future replayOwned must share it; Task C kernel replay is outside this change.

Next authorized test-first command, after root sees repaired A actual GREEN and
scoped source review, from a fresh checkout and nonexistent output directory:
python3 -B -S tests/sdga_component_contract.py --output "$RUNNER_TEMP/sdga-task-b-red-normal" -v
Then repeat with -O and a different fresh output directory when root authorizes.
Retain actual helper absence or semantic failure; do not manufacture strictAxioms
RED by removing existing correct behavior. No execution occurred for this report.
Root owns publication, hosted execution authorization and subsequent minimal fix.
