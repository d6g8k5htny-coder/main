# Task C R1 public replay CLI contract — test-only source stage

Status: STAGED_FOR_ROOT_SOURCE_REVIEW. No R1 ref update, publication, hosted run or production implementation has occurred. Root explicitly requires source read/GO before publication and actual R1 CLI unsupported evidence before kernel implementation.

## Concrete coverage gap and retained R0

R0 at PR807 head fa0deec246fa603d7c45aae27df7684de35538ee exercised actual shared replayOwned. Those controls could eventually all pass while public runReplay still refused NOT_IMPLEMENTED. Root therefore requires one positive public CLI probe in this same milestone. This is a behavioral integration obligation: the real CLI must load the actual request, construct the actual direct-import base, call the same shared replay helper and report every owned Name only after actual kernel replay succeeds.

Actual R0 run42759906446/attempt1/job129642895238 has signed native46b8079aafd5b4d784174c0b680a43ae57e1c4a6, exact treefde944db2b70af4abbd30ec775e7179a192b4084 and ordered parents[B4fe4369aecdaf0b3af02f3f77cb91dfbbd3355cd,Cfa0deec246fa603d7c45aae27df7684de35538ee]. Root independently read it. Both modes: 13 methods, ten successes/three failures, 61 unique declared IDs=56pass+3fail+2not_run (59 observed), zero errors/skips. All original A/B52 passed. C08/full-safe-replay and C10/good-base-setup retained exact production NOT_IMPLEMENTED; C09/kernel-type-rejection failed its precise wrong-cause assertion on NOT_IMPLEMENTED. C10's remaining two subcases were not executed. Four C prerequisites (fresh base absence twice, actual source theorem CI, singleton resolved refs) passed. This is implementation-absence RED, without native kernel success/type-rejection/duplicateBase or mutation-kill credit.

Both modes also passed fixed24observations/10bindings/55paths/final registry, pure12, separate custody3methods/6routes and cleanup; source/object drift empty. All35 recorded command intervals per mode were sequential, no non-cleanup timeout or unconfirmed owned-process cleanup occurred. LEAN_NUM_THREADS=2, lean --threads=2, native process limit180s and workflow limit25min were unchanged. Reported maxrss is cumulative RUSAGE_CHILDREN, not a per-process or concurrent memory measurement.

Complete log blob4c1798d5b0a0b32852a05382479629394c30f1b3 is 1,410,858UTF8bytes with SHA2562280ad23ca10814e5b62e0982326467f4a4d5c90c444cc583f11213e7e3c3f45. All210 runner-printed original blocks independently matched declared UTF8 byte counts/SHA256; packetdef3e3e95fb4d7edeca76695d696a5a91fc00a3c retains them exactly. Observation receipt332d5b81c11b046f6d43f3d67eb1d8898ab6acd0 binds source/run/limits. Hosted artifact13736893830 has ZIP digestd4f9cbaf1de321cc66a8ba426a7e0052c15e0801842fc9a725c923344935998b; no independent ZIP download/unpack/read is claimed. These blobs were read back byte-for-byte.

## Exact test-only change

Base launcher is approved/published28a380649337295255812e095b59652155463fb5. Seven uniquely anchored edits:
1. Initialize STATE.cli_replay as supplemental not_run, with native command, stdout_hex/stderr_hex and receipt slots.
2. Change the fixed expected observation_count24→25.
3. Change only its matching fixed length equality24→25; binding10/path55 literals stay exact.
4. Include supplemental cli_replay in source-identities through existing flush.
5. Wrap original complete C08 helper body in subTest(route="shared-helper"), then run a separate subTest(route="public-cli").
6. Include cli_replay in the final result record.
7. Require final cli_replay.status=="pass" in overall acceptance, in addition to every existing gate.

The old C08 helper body/assertions recover byte-exact after undoing only the necessary subTest indentation. No original assertion is removed. Ordinary unittest subTest failure in that first block is recorded and permits the later independent CLI block to run. The CLI uses only the original healthy baseline prerequisites and never reads helper rows as its setup. Thus current helper NOT_IMPLEMENTED cannot prevent the actual public replay command attempt.

Exact new observation:
STATE.observe_lean("cli-safe-replay", ["--run", str(BRIDGE), "replay", str(STATE.request_path)], requests=(STATE.request_path,))

The fixed request is the existing actual Safe request. There is no extra object file, request binding, fixture or module filter. The command's exit/stdout/stderr/cleanup/limits remain retained by the unchanged launcher custody/command machinery. The supplemental record copies the complete command result and stores lossless raw stdout/stderr bytes as hex before asserting success. If observe_lean itself fails after creating a command, its retained latest matching command is referenced without manufacturing returned streams. A failed record write is chained under the original pending exception; the original failure is preserved.

The public route requires exit0 and empty stderr. Its UTF8 stdout must be exactly one JSON line/object. A bounded object_pairs_hook assertion rejects duplicate JSON fields; no generic serializer or production parser is added. Exactly twelve fields are required, with no missing/extra keys:

| Field | Required value |
| --- | --- |
| schema | cbz.sdga.kernel-replay.v1 |
| module | exact STATE.request["module"] lossless wire |
| scope_id | exact actual request scope_id |
| profile_sha256 | exact actual request profile_sha256 |
| helper | SDGA.replayOwned |
| owned_count | integer equal to complete native inventory declaration count |
| replayed_names | duplicate-free string list whose exact set equals all actual native inventory Names |
| skipped_names | [] |
| scientific_effect | NONE |
| qualification | BLOCKED |
| classification_status | BLOCKED |
| execution_authenticated | Boolean false |

Only replayed_names ordering is ignored, as the contract requires an exact set. Its duplicates, missing/extra Names, incorrect types, wrong counts, stale request bindings or extra output fail. The actual R0 population has fifteen Names. No fake marker or test-owned producer is added. The probe grades the actual native stdout receipt; separate future production source review must establish that this receipt is emitted only after unchanged base.replay owned and complete all-owned success/no-skip checks, through the shared safety/base-guard helper route reviewed in the existing C design.

Supplemental cli_replay lives outside all61 declared subcase IDs and outside the existing thirteen test methods. No JUnit/unittest method is added. It is still mandatory for overall acceptance. On the unchanged Component7f baseline, public runReplay must actually return NOT_IMPLEMENTED/nonzero with original raw outcome and supplemental fail. That failure is positive-route implementation absence, never native kernel rejection credit.

## Preservation and fixed census

Controls c25cd01838c152403eba1acd39fb029a488bb21f is unchanged. C09/C10 and every A/B test body are byte-exact. Original top-level measure_objects/object_custody, bind_request, observe_lean, native_case, commands/cleanup, B dependency binding/fixtures/source capture, source histories, final object registry/error handling and every other acceptance gate remain unchanged. The only existing class methods touched are __init__, finalize_objects (two numeric literals), flush (supplemental metadata), and C08. Existing main receives only the supplemental final-result field and PASS gate. Reversing all seven changes recovers exact28a. No native syntax/compile/run result is claimed for R1.

Fixed normal complete-success custody enumeration is below. It is derived from fixed source call sites and not from an observed length. Only cli-safe-replay is added after control-C08:

1. inventory
2. control-C01
3. control-C02
4. control-C03
5. task-b-AxiomUse-inventory
6. control-C04
7. control-C05
8. task-b-OwnedAxiom-inventory
9. control-C06
10. task-b-Skipped-inventory
11. control-C07-unsafe-info
12. control-C07-partial-info
13. control-C07-unsafe-replay-skip
14. control-C07-partial-replay-skip
15. control-C08
16. cli-safe-replay
17. control-C09
18. control-C10
19. control-C12
20. control-C13-missing-server
21. control-C13-reordered-parts
22. control-C13-truncated-private
23. control-C15-wrong-resolved-path
24. cli-unknown-mode
25. cli-unknown-field

Expected total is fixed25observations, fixed10bindings and fixed55paths. Requests and path registration remain the previous ten bindings: five genuine fixture requests/parts=30paths, three C13 adverse roots=18paths, C15 wrong-path root=6paths and unknown-field request reusing Safe parts=1path; total55, including the intentionally absent server path. The CLI reuses Safe and adds no registry key. BadDependency remains binding-only and guarded inside AxiomUse observations.

## Proposed fresh composition and source limits

Fresh read-only Main is964390a94c08f00f119dc3b2e0be16180c1544e2/tree6677c6cf4744f5136c7a4c67bf4f3a74d869cff8, ordered parents[b6be588c82d2256c917b7d26d901085bc90319fb,4fe4369aecdaf0b3af02f3f77cb91dfbbd3355cd]. PR807 auto-retargeted to Main and remains Draft at old Cfa0d. Main DECISIONS is exactB4fe prefixc76f88ce0b951456395a4f1624ced8d88627b8b3.

After root source GO, create an ordinary merge child with fresh Main first parent and old Cfa0d second parent. Candidate DECISIONS retains the full fresh Main prefix, only the old C-owned append relative to B4fe, then the new R1-owned append. Refresh these guards if Main/C moves; no force or blind canonical replacement. Preserve every peer/base path.

Proposed complete C scope is five paths relative to fresh Main: existing Controls c25, revised existing launcher, existing approved design4ffc79dc9356879aa255813b5c97b3456a76313d, this minimal R1 report, and append-only DECISIONS. Relative to old Cfa0d, R1 touches only launcher, this report and the new DEC suffix.

Production Component7f89460edca71754dbce9a07de229ad39ad3e851 remains untouched with safe replayOwned/runReplay NOT_IMPLEMENTED. Workflowf22987c60144ae7db6202cae6085402ca2c2f947, custodytesta9e147e246abfacf9561d286aef65086dedd2c66, pure12/8809f1974998f9094a0f1239ecca56ddcf272a88, all fixtures, Task1JSONL/Name/classifier/assertions, frozen profile/99module/1137targets, source proofs/bounds, formal gates and earlier failures remain exact.

R1 publication and automatic hosted R1 observation are pending source GO. Root must read the actual unsupported public CLI result before production kernel work. No workflow/manual-rerun/comment/review/settings/merge action is requested here. scientific_effect=NONE; qualification/classification_status=BLOCKED; execution_authenticated=false; no fullG3/whole native/kernel/IR/finalG0-G11/CBZ/scientific acceptance claim.
