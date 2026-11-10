# Task C guarded upstream kernel replay — implementation source stage

Status: SOURCE_ONLY_PRODUCTION_CANDIDATE, pending root source read/publication GO. No candidate compile, native run, ref update or production qualification is claimed.

## Authorization and exact binding

Root independently read actual R0 and R1 before authorizing production source staging. R0run42759906446/1/job129642895238 at signed native46b8079aafd5b4d784174c0b680a43ae57e1c4a6/treefde944db2b70af4abbd30ec775e7179a192b4084 observed helper implementation absence: both modes13methods,10success/3failure,61IDs=56pass+3fail+2not_run,0errors/skips; A/B52 and24/10/55/pure12/custody3/6 passed. R1run42761796853/1/job129648973831 at signed nativee080ce31514a334ec95ab12ff5e9a4d351f99a32/treecaa1506fb3db73d08e8f9b80decea155b90fea0f, parents[Main964390a94c08f00f119dc3b2e0be16180c1544e2,C2850d98a75565a1c74815a864ce77956c07bf652], observed the independent actual public CLI after helper failure: exit1/stdoutempty/exact81byte NOT_IMPLEMENTED stderr in both modes, no timeout, cleanup confirmed, supplementalFAIL/receiptNULL. R1 had13methods,10success/3failedmethods/four failure entries, same61IDs=56pass+3fail+2not_run,0errors/skips; A/B52 and25/10/55/pure12/custody3/6 passed. All object/source drift was empty. Neither run observed successful native replay/type rejection/duplicateBase/CLI receipt; there is no mutation-kill credit.

Source base is actual protected Component7f89460edca71754dbce9a07de229ad39ad3e851, independently read as blob and at feature2850. Candidate is ca46b1ae1031aa0740a9d4366d649e156181aac4; exact three-region transfer/diff is bb36f7bec00d14314c48da975ab955002042e3b3. Reverse replacement recovers complete7f byte-for-byte.

Fresh read-only Main964390a94c08f00f119dc3b2e0be16180c1544e2/tree6677c6cf4744f5136c7a4c67bf4f3a74d869cff8 and C2850/treecaa150 were verified during this source stage. Main DECISIONS prefix is exact; candidate DEC keeps the entire fresh Main prefix, the exact existing C/R1-owned suffix, and only the new implementation suffix. Future publication must refresh Main/C guards and preserve peer changes. Proposed history retains Main first parent plus feature2850 second parent.

## Shared production/control route

replayOwned keeps its exact public type Kernel.Environment → Std.HashMap Name ConstantInfo → IO Kernel.Environment. It now:
1. Calls guardReplayInput through existing fromAudit.
2. Calls exactly unchanged base.replay owned once.
3. Calls verifyReplayOutput through fromAudit.
4. Returns that actual replayed environment.

There is no catch around upstream replay. Its actual native kernel IO.userError and Replay's own declaration wrapper propagate unchanged to direct C09. Existing runCLI/diagnostic remain byte-exact; their existing CLI-level diagnostic normalization still applies to actual public failures. No parallel prefix CLI, copied checker, unchecked environment insertion, no-op or test-owned replay behavior is introduced.

guardReplayInput validates each submitted map key is nonanonymous, equals ci.name, and is absent from base. This entire identity/absence pass occurs before upstream, so C10's bad singleton receives exact existing AuditError.duplicateBase with its lossless Name before Replay's matching-theorem shortcut can ignore the value. Every submitted CI then calls the byte-exact existing guardReplayConstant for actual isUnsafe/isPartial refusal; no name is exempt or skipped.

Pinned Replay indexes newConstants[n]! for every mutual inductive group member and constructor. guardReplayInductiveLinks checks only those required metadata lookups using safe Option/pattern matching: group contains itself; every group member is present and inductive with matching all/levelParams/numParams; every listed constructor is present, constructor-kind, and points to that group member. These are metadata/input checks protecting the unchanged upstream indexed path, not a kernel type checker. No special fixture/Name prefix or arbitrary import filter is used. Additional malformed-inductive adverse cases have source review only.

The existing strictAxioms traversal runs once from every owned map key with lookup owned[name]? <|> base.find? name. Missing lookup or CI identity mismatch remains fatal. Every reached axiom is rejected if owned, including an owned axiom named exactly propext. Dependency axioms are allowed only for exact Name literals propext, Classical.choice and Quot.sound; no namespace/string-prefix/display comparison. Task1's actual current source14d4617b523c29007da4bdb95ee6ee4dc588a86c was reread and its exact constructor-preserving allowlist matched. Task1 itself is unchanged.

## Isolated collector extension affecting existing inventory traversal

Exact existing private dependencies change: three narrow branches append DefinitionVal.all, TheoremVal.all and OpaqueVal.all. Everything else in that collector and strictAxioms/closeAxioms stays exact. Pinned Lean.Util.FoldConsts.ConstantInfo.getUsedConstantsAsSet collects ci.type and value? allowOpaque=true; when a value exists, the metadata .all group list is not collected. Pinned Declaration states these .all fields describe every member of the mutual block and are source metadata not used by the kernel. To make every explicit group link missing-fatal alongside the existing inductive/constructor/recursor links, the shared collector must include them.

Root explicitly authorized staging these three branches and required this scope limit: inventory also calls strictAxioms, so future inventory closure contents or missing-group failures can change. Its JSONL schema/Name wire/classifier/assertions are untouched, and historical outcomes remain source-bound. There is no claim of new runtime missing-group coverage: unchanged C05 controls missing type/value/inductive/recursor links, without a new group-specific subcase. No contract/census/assertion was weakened to obtain RED or GREEN.

## Complete post-replay CI observation

verifyReplayOutput requires every owned map key to resolve in the actual replayed environment and compares the complete pinned CI metadata:
- AxiomVal, DefinitionVal, TheoremVal, OpaqueVal, ConstructorVal and RecursorVal use their existing pinned derived BEq, including inherited ConstantVal fields and all record fields.
- InductiveVal has no BEq. Compare inherited name/levelParams/type through ConstantVal BEq, then all eight extra fields: numParams, numIndices, all, ctors, numNested, isRec, isUnsafe and isReflexive.
- QuotVal/QuotKind also lack BEq. Compare inherited ConstantVal, then match every exact QuotKind constructor type/ctor/lift/ind; unmatched pairs are false.
- Different CI constructors are false.

The check is metadata equality using pinned native expression/record equality, not display/hash equality, object byte identity, an AST serializer or a second type checker. Unsafe/partial input refusal plus actual complete output checks prevent any skipped Name from being reported as success. Constructor/recursor kernel validation remains upstream's unchanged postponed native equality check as well. The next authorized native run must establish actual Safe-fixture inductive/constructor/recursor success; no candidate success is asserted now. Quotient and new malformed-link cases have no new runtime coverage claim.


### Immutable pinned field and collector evidence

Exact field/source-excerpt artifact:765488374322f6e70853cc01e89117ebfa852046. It includes the complete pinned Declaration record definitions and ConstantInfo collector implementation used for this source comparison.

All Lean API evidence was read at commit5045d0056413266e57c625dcd7c365b10e377c52:
- src/Lean/Replay.lean, blob71012211da8d22961873d9ff70feadac973715a8: actual replay declaration, native addDeclCore, missing unchecked new-map indices, duplicate-theorem branch and native error behavior.
- src/Lean/Declaration.lean: ConstantVal derives BEq over name/levelParams/type; exact record fields below; InductiveVal/QuotVal/QuotKind do not derive BEq.
- src/Lean/Util/FoldConsts.lean: ci.type.getUsedConstantsAsSet plus value? allowOpaque=true; for a present value it never visits defn/thm/opaque .all. Those three Declaration records explicitly store all mutual-block members, default [name].
- src/Lean/Environment.lean: ModuleData.imports, importModules signature/private/trust0/loadExtsfalse, toKernelEnv/find?/header/module index, explicit unsafe Environment.freeRegions API.
- src/Lean/CompactedRegion.lean: compacted contents have no individual reference counters; region release is explicit unsafe CompactedRegion.free. No such free operation is added/called.
- Current Task1 source14d4617b523c29007da4bdb95ee6ee4dc588a86c: exact allowlist uses encode_name(("propext",)), encode_name(("Classical","choice")), encode_name(("Quot","sound")).

Every row below also includes inherited ConstantVal fields name, levelParams and type:

| CI variant | All additional pinned fields | Candidate comparison |
| --- | --- | --- |
| axiomInfo | isUnsafe | native derived AxiomVal BEq |
| defnInfo | value, hints, safety, all | native derived DefinitionVal BEq |
| thmInfo | value, all | native derived TheoremVal BEq |
| opaqueInfo | value, isUnsafe, all | native derived OpaqueVal BEq |
| inductInfo | numParams, numIndices, all, ctors, numNested, isRec, isUnsafe, isReflexive | each field explicitly compared; inherited ConstantVal BEq |
| ctorInfo | induct, cidx, numParams, numFields, isUnsafe | native derived ConstructorVal BEq |
| recInfo | all, numParams, numIndices, numMotives, numMinors, rules, k, isUnsafe | native derived RecursorVal BEq; each RecursorRule derives BEq over ctor/nfields/rhs |
| quotInfo | kind | inherited ConstantVal BEq plus exhaustive same-kind pairs type/type, ctor/ctor, lift/lift, ind/ind |

The candidate matches all eight CI constructors explicitly and rejects cross-kind pairs. This is exhaustive field coverage relative to those immutable pinned record definitions, not runtime evidence or universal AST/object-memory identity.

## Public runReplay and receipt causality

runReplay reads actual final ModuleData.imports, requires ordered part import arrays agree, and calls importModules imports {} (trustLevel:=0) (loadExts:=false) (level:=.private). Exact Import records/modifiers are passed unchanged. This is source-equivalent to the independent C08/C09/C10 fresh-base oracle; it does not reuse loaded.env, which already contains the target. It checks actual base trust/direct imports and target module absence in both module index and header; the shared replayOwned then checks absence of every submitted owned Name plus safety/closure.

Raw LoadedModule contains every OLean/IR CompactedRegion, and base/returned Kernel.Environment retain import header regions. No CompactedRegion.free, Environment.freeRegions, unsafe withImportModules or unchecked insertion is called. LoadedModule/final raw data remain in the runReplay scope through output construction; this bridge never frees those mappings. Direct helper callers retain the actual LoadedModule/raw regions while using CIs, as the existing controls do. This stage does not change object loading/lifetime APIs.

Only after shared replay and complete postcheck return, runReplay resolves each final source Name in the actual returned environment and uses actual CI.name to build replayed_names. The receipt is emitted once through the existing flushing emit helper with exactly these twelve fields:
schema=cbz.sdga.kernel-replay.v1; module=lossless encodeName actual request.module; scope_id=request.scopeId; profile_sha256=request.profileSha256; helper=SDGA.replayOwned; owned_count=actual loaded.owned.size; replayed_names=actual replayed CI Names for the complete final population; skipped_names=[]; scientific_effect=NONE; qualification=BLOCKED; classification_status=BLOCKED; execution_authenticated=false.

No receipt is emitted before actual replay/full postcheck. The existing R1 positive CLI probe independently validates exit0/empty stderr/one exact receipt, full Name set/count, no skipped/duplicate fields/Names and all request bindings. Its supplemental PASS remains mandatory outside61; test text is unchanged. Missing/unsupported/partial/resource-failed or wrong native-cause outcomes block, not coverage exemptions.

## Exact unchanged scope and pending native obligations

Only Component's module header, three collector branches and old replay placeholder region change. Raw loading/framing/request validation/Name wire/AuditError helpers, strict traversal function bodies, inventory JSONL body, existing public safety guard and diagnostic/runCLI are byte-exact to7f. Controls c25cd01838c152403eba1acd39fb029a488bb21f, launcher86842b2814fbceaa7d7d65b0d0088546dbc3bb06, workflowf22987c60144ae7db6202cae6085402ca2c2f947, custodytesta9e147e246abfacf9561d286aef65086dedd2c66, pure12/8809f1974998f9094a0f1239ecca56ddcf272a88 and all fixtures remain exact. No TaskD guards/IR sidecar, generic AST serializer/orchestration, arbitrary module filter, proof/bound/formal gate, frozen99module/1137target/profile or Task1 source/assertion change is introduced.

Candidate source checks are text/reverse-diff/API inspection only. No Lean elaboration, native compile, native result, actual receipt or candidate13/61 qualification has been observed. After root complete-source review and publication GO, the unchanged hosted workflow must establish both modes13methods/61subcases, all52old A/B rows, real C08 good replay, actual C09 native declTypeMismatch, real C10 duplicateBase after successful good contamination, mandatory actual public CLI success/receipt, pure12, custody3/6 and fixed25observations/10bindings/55paths/final registry without drift/skips or resource failure. Preserve first failures and original streams. Processes stay sequential; LEAN_NUM_THREADS is not a Lake child-job cap. No manual rerun or assertion replacement is authorized by this source artifact.

## Permanent original evidence proposal

Six owned evidence entries are staged alongside the source/report/DEC proposal:
- red-r0/job.log →4c1798d5b0a0b32852a05382479629394c30f1b3; 1,410,858UTF8bytes/SHA2562280ad23ca10814e5b62e0982326467f4a4d5c90c444cc583f11213e7e3c3f45.
- red-r0/originals.json →def3e3e95fb4d7edeca76695d696a5a91fc00a3c; all210runner-printed blocks and exact declared/matched byte counts/SHA256.
- red-r0/observation.json →332d5b81c11b046f6d43f3d67eb1d8898ab6acd0.
- red-r1/job.log →d179dbe624c7156288967fb6078b30a551f10e55; 1,454,841UTF8bytes/SHA25660def27b0708afb984e9f02bf0617508e43dd52d2de0db8f893ad6f600cb578e.
- red-r1/originals.json →6ad8187f85b76866773edce6988e4d0a9eb8c35b; all214runner-printed blocks and exact declared/matched byte counts/SHA256.
- red-r1/observation.json →ee18f0e736f70c67b20c8c86b21cdfbac025d178.

All paths are under cbz/evidence/sdga-task-c-20261110/. These records preserve actual old sources/causes permanently when the later approved tree is published; they do not rebind those observations to this candidate. Hosted artifacts13736893830 and13737927810 have API/log-reported ZIP digests d4f9cbaf1de321cc66a8ba426a7e0052c15e0801842fc9a725c923344935998b and7bb437cedb4ceeaf8c099c794898a87dc06edb0ec48a9fe44eaa4ab5f352c386; no independent artifact ZIP extraction/read is claimed.

scientific_effect=NONE; qualification/classification_status=BLOCKED; execution_authenticated=false. No fullG3/whole IR/native/kernel/finalG0-G11/CBZ/scientific acceptance claim.

## 2026-11-10 — initial bounded native GREEN; fresh Q/Main composition pending

Run42764126530/1/job129656474935 tested signed nativea908e4af5d8a33d0f90d1d8f8e83e9f105c66b25,
tree0abdb0aac07bb72550f34a4ea39cda4a487ff94d, parents[Main964390a9,C9607199c].
Both modes passed13methods/61unique IDs(55Lean+6Python), mandatory actual
15-Name public CLI receipt, pure12 and separate custody3/6. C08 real full-map
replay, C09 exact native Nat→True type rejection and C10 good contamination→
duplicateBase passed. All25file observations/final10bindings55paths passed;
failures/errors/skips/not_run/source-or-object drift were zero.

Root read all214exact original records and signed source/parents. Final scoped
broad review returned ReadyYES with no C/I/M finding. This bounded result is
not whole-kernel/IR coverage, final gates or scientific acceptance; existing
conservative .all/new adverse-coverage limits remain unchanged.

Retain green-r0/job.log20b0730ec172c9b2671b260824dad2ddca7e99dc,
originals.json75f502cddea83f72382313c4e3b849acf7c72d9e and
observation.jsone038858936c8d85e9a072c89d1fd4d8603a17e51 under
cbz/evidence/sdga-task-c-20261110/. All214runner byte counts/SHA256 matched.
Artifact13739705171 is hosted; no independent ZIP extraction/read is claimed.

Compose fresh Q806/Main first parent plus complete old C9607 second parent
once, preserving full peer/canonical prefixes and exact C-owned suffix relative
to old Main964/c76. Componentca46,launcher868,Controlsc25,workflowf229,
custodya9e,pure/fixtures/Task1 stay exact. Prior GREEN remains bound to its old
source; automatic fresh-composed13/61+CLI/pure12/custody3/6/25-10-55 and
current protected checks are required before Ready/merge. Q806 precedes C.
Scientific NONE; qualification/classification BLOCKED; authfalse.
