# Formal Regression Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver authentic old/candidate formal collection, complete review regression and a protected required consumer for every actually admitted research domain.

**Architecture:** Admitted T/P controls acquisition and hosted credential-free production. Offline verification preserves ABIs and whole-group review; separate publisher/write admission requires fresh context. Tasks deliver real mapping, declaration authentication, custody and held-out review.

**Tech Stack:** Python3.11+ stdlib; existing Main Lean `leanprover/lean4:v4.34.1`; separately qualified hosted Linux/runtime and native APIs. No new Python package dependency.

**Spec:** R01 `docs/superpowers/specs/2026-10-18-formal-regression-collector-design.md` plus `docs/architecture/formal-regression-contract.json`, exact accepted copies carried/pinned in this packet. Root03:16:51 admitted design for planning; fresh PLAN review remains pending. TASK-REQUIREMENT-MATRIX.json carries complete cases/dependencies; immutable historical status fields stay historical.

## Global Constraints

- Plan only: all8 ACTIVE_UNMET/org0/source exposed/science NONE/no status authority. No collector semantic approval. Standing AGENTS removes owner reapproval; technical review/current ownership/capability remain.
- Project execution is hosted/ephemeral only; no laptop Git CLI/repair. Use isolated owned source/native objects/guarded refs and protected integration, preserving unrelated Main; no global reservation/unknown writer resumption.
- Preserve B=current base, H=head/status subject, S=service merge, T=admitted control, P=policy digest, K=native checkout, M=landed merge. Verify actual parents/tree/event/control source; these identities are not interchangeable.
- Request key is SHA256(canonical `{domain:formal-regression-request/v1,context:RequestContext}`). Inventory digest is the one exact R01 core/canonical ASCII/SHA256 formula, excluding run/acquisition/attempt provenance but retaining it separately. No supplied status/checksum replaces source/native authentication.
- Keep adapter v1 stdin/four expected-context options/report, `custody: unknown`, native Main/Math schemas and existing required checks. Independently authoritative gate SHA256`a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8` cannot be changed by packet provenance.
- Adapter limits16MiB packet/4MiB capture/depth64/256nodes/2048edges. Acquisition64MiB total retained originals/decompression,1024members,4MiB each metadata response. Never split whole review groups or omit originals to fit; overflow is incomplete/non-success.
- Preserve originals/failures/corrections. Native retention>=30days; governing-custody transfer/readback precedes expiry. Denied/unknown delivery stops, no retry/guessed attempt/alternate identity; changed input gets new acquisition.
- Qualify private custody first. Public allowlist/digest excludes private paths/links/bodies/secrets; Actions artifacts are not presumed private.
- Basis stays Main6fce/tree5b3447; Root3edacfe5/a420 conserves17 dependencies plus21st workflow, source-only. Refresh current source/claims before writes; no run rebinding.

## Review Focus

1. H!=S/base drift: native context and write-boundary tests refuse stale evidence (Tasks3,7,9).
2. Forged type stdout/metaprograms/initializers: require actual declaration observation, not transcript shape (Task4).
3. Whole original reviews across downstream/deleted/context edges: retain every use/six-field ref (Task6).
4. Private/partial/cancelled acquisitions: bounded honest terminals, no second original GET or public leak (Tasks3,5).
5. Recollection/proof edits/lost selectors: stable content hash and independent semantic/native predicates (Tasks1,5,6).

## Files and shared interfaces

The eleven paths (complete responsibilities/controls in matrix):

| Path | Responsibility |
|---|---|
| `architecture/formal_regression.py` | Closed contracts, content projection, adapter/formal impact, pure decision. |
| `tools/formal_regression_collect.py` | Authenticated native transport and atomic private preservation. |
| `tools/formal_regression_native.py` | Trusted supervisor, hostile isolation, observations/disposal. |
| `tools/formal_regression_verify.py` | Offline verification and participating-write admission. |
| `tools/formal_regression_publish.py` | Narrow check publication/readback; no source/ref/settings authority. |
| `.github/workflows/formal-regression.yml` | Admitted event/acquisition/verification/publication controller. |
| `.github/workflows/formal-regression-native.yml` | Qualified preparation and credential-free producers. |
| `docs/architecture/formal-regression-policy.json` | Actual domain/control/authority/runtime/custody/publisher/enforcement policy; admission pin outside candidate. |
| `claims/formal-bindings.json` | Actual accepted links and exhaustive typed coverage. |
| `tests/test_formal_regression.py` | Frozen contract/transport/semantic/security controls. |
| `tests/test_formal_regression_enforcement.py` | Actual principal/context/protection/admission controls. |

. Additional exact paths (Task0 reconciliation): `.github/workflows/formal-regression-controls.yml` (dedicated hosted controls); `tools/formal_regression_declarations.lean` (conditional trusted observer); `tests/fixtures/formal_regression/control-plan.json` (frozen inventory); `tests/fixtures/formal_regression/inventory-conformance.json` (exact82,499B accepted synthetic data); `docs/architecture/formal-regression-contract.json` (exact accepted DATA); `docs/architecture/formal-regression-runbook.md` (activation/recovery/retention); accepted spec and this plan under `docs/superpowers/{specs,plans}`. Existing native/graph/shared-workflow edits require separate exact scope/review/pins.

`Record=dict[str,object]` denotes a closed DATA type; preserve all accepted type/field names. Internal `InventoryResult={core,formal_inventory_sha256,diagnostic_inventory_sha256,issues}`, `ImpactReport={semantic,evidence_loss,original_groups,affected_original_refs,issues}`; no native/core schema change.

Cycle C: freeze controls/method inventory; hosted assertion RED; admitted implementation; unchanged normal/`-O` GREEN; distinct originals/source/runtime/cleanup review; guarded increment. Import/setup/skip/corrupt fixtures mean HOLD, not functional RED. An API scaffold is not functional admission. Fixtures/gold stay unchanged.

Hosted commands: `python3 -B -S -m unittest discover -s tests -p test_formal_regression.py -v` and `python3 -B -O -S ...`; enforcement uses `-p test_formal_regression_enforcement.py`. Matrix lists exact task classes/cases. Preserve every result/stream/exit/original. Only verified Decision.PASS permits required success.

## Task0: Current source, actual owners and finite scopes

Files: Current source/claims/Work Events/policy; all proposed paths. No implementation before this gate.
Interfaces: Accepted design/review/disposition -> source/owner reconciliation, admitted authorities and named finite code/control/native/custody/publisher scopes.

- [ ] Read bounded current PR325/board/Work Events/native identities; identify actual product/control/runtime/authority/publisher owners before pickup. UNKNOWN requires current finite assignment through project coordination, never takeover or renewed owner permission.
- [ ] Compare current complete Main/all17 dependencies and proposed extensions with SITE/Goal1/4 active scopes; preserve every unrelated leaf. Path absence is DATA, not evidence of no active writer.
- [ ] Obtain fresh whole-PLAN review/coordinating admission; establish isolation/owned branch/rollback checkpoint and public-screen exact design/contract/conformance copies.
- [ ] Record actual M1–M7 actors/capabilities/deadlines. Continue independent tasks; missing input blocks dependent activation.

## Task1: Frozen external controls and closed inert core

Files: Core, both tests, DATA/conformance copies, control-plan/controls workflow; policy starts diagnostic/unresolved.
Interfaces: `decode_request(raw:bytes)->RequestContext`; `decode_policy(raw:bytes,expected_sha256:str)->Policy`; `request_key(context:RequestContext)->str`; `content_inventory(records:list[TargetStatement],policy:Policy,membership:Record,acceptances:Record)->InventoryResult`; `inventory_sha256(core:Record)->str`.

- [ ] Freeze matrix `ContractControls` and exact9-preimage fixtures.
- [ ] Independently review/freeze controls and qualify the dedicated hosted caller/preparation/resource/original-custody path without publisher credentials.
- [ ] Run C's normal/optimized assertion RED on the frozen original/scaffold; missing import is not that witness.
- [ ] Implement exact R01 nested forms/order/projection/null/known-absence/digests and strict decoding; never execute candidate Python/config/hooks or a supplied gate.
- [ ] Run unchanged controls GREEN and CLI/API refusals; qualify originals and distinct source review before the guarded increment. Synthetic positives retain unknown real authority/custody.

## Task2: Actual typed mappings and whole-review authority

Files: Binding/policy/core, `MappingAuthorityControls`, private authentic originals.
Interfaces: `resolve_coverage(binding:Record,base:Record,candidate:Record,policy:Policy,acceptances:list[AuthorityAcceptance])->Record`, exactly `{node_rows,target_rows,unresolved,accepted_manifest_sha256}`.

- [ ] Freeze/run all matrix `MappingAuthorityControls` RED.
- [ ] Route exact graph/native package/target/source/slice inventories to actual qualified domain authors and mapping authority. Obtain real typed connections or supported exclusions and immutable native acceptance bound to resolved graph/manifest/scope/domain. Every node/target is linked, accepted outside-domain or unresolved.
- [ ] Acquire current Math inputs under its own authority/native format when admitting that domain. Main31 SIDE24 targets and graph49 nodes never create a Math map.
- [ ] Retrieve each original review/evidence by six-field GitRef, all uses, immutable acceptance/version and actual lineage under existing policy. Unknown or same-lineage evidence stays unresolved; collector supplies no semantic verdict.
- [ ] Implement authority/coverage; run C GREEN; separately qualify real acceptance readbacks and matrix negatives.

## Task3: Native transport and private preservation

Files: Collect CLI/policy/core; `TransportControls`, `PreservationControls`.
Interfaces: `collect(request_context:RequestContext,admitted_policy:Policy,native_reader:NativeReader)->Record`, exactly `{bundle_dir,bundle_sha256,terminal,issues}`. NativeReader implements the seven accepted primitives, returning raw bytes plus NativeObservation; acquisition/principal labels alone are not authentication.

- [ ] Freeze/run matrix `TransportControls`/`PreservationControls` RED.
- [ ] Establish actual CI principal and attempt/job/artifact/supervisor joins; interactive connector access is insufficient. Preserve denied/unsupported calls without alternate identity/retry.
- [ ] Qualify a real private carrier using nonsecret write -> metadata readback -> raw-byte/hash equality, access/privacy refusals and retention transfer. Acquire no private source/review material before this passes.
- [ ] Implement no-follow bounded atomic staging outside source. Record `(repository,run,attempt,artifact_id)` ATTEMPTED before each original GET and exact outcome afterward; unknown/before-byte failure remains terminal. Changed-input recovery gets a new acquisition and preserves the old negative.
- [ ] Run C GREEN and actual transport/private-custody probes; qualify independently originals/disposal.

## Task4: Trusted declaration observation and hostile native execution

Files: Native supervisor/workflow/policy, conditional Lean observer; `NativeBoundaryControls` and actual hosted originals.
Interfaces: `produce_native(requested_K:Record,actual_native_event_context:Record,admitted_TP:Record,sandbox_contract:Record)->Record`. Observer CLI consumes `--checked-artifacts --targets --output`; closed observation `{schema_version,checked_source,compiler_context,targets,issues}` binds actual declaration/module/type identity and exact native record slices. API/provenance must be qualified before admission.

- [ ] Freeze/run actual-hosted matrix `NativeBoundaryControls` RED.
- [ ] Inspect pinned compiler/leanchecker/producer source and runner APIs in bounded hosted preparation. Implement a reviewed observer of kernel-validated declarations without candidate initializer/metaprogram execution in its trusted phase. Stop candidate work first; only this separate observer writes supervisor evidence. Verify target/module/printable-type binding.
- [ ] Demonstrate legitimate/type-changed declarations and rejected forged stdout/initializers. Shape, same-process nonce/FD or isolation alone is insufficient. Unsupported actual API means preserved probe/HOLD `DECLARATION_AUTHENTICITY_UNAVAILABLE`, requiring a reviewed usable observer/compiler release, never an auth flag.
- [ ] Qualify T-pinned Lean/runtime/dependency/preparation originals; existing Python/Main install evidence is insufficient. Workers:`--pull=never`, no network/credentials/socket/sensitive mounts; read-only source/control/toolchain, nonroot, caps dropped/no-new-privileges, bounded output.
- [ ] Probe proposed2CPU/4GiB/256pids/1800s;45min=1800worker+300prep/inspection+120disposal+300custody+180reserve. Unsupported fit/capacity is HOLD, not inflation;64MiB retained total across contexts.
- [ ] Execute fresh B/S normal/optimized supervisor contexts. Preserve all native logs/receipts and protected K/event/T/attempt/command/exit/resource/disposal/source-after observations; distinct qualification precedes M3/native PASS.

## Task5: Authentic two-sided bundles and content inventories

Files: Core/collect/native/controller files; `CollectionControls`.
Interfaces: `assemble_bundle(request:RequestContext,policy:Policy,observations:list[NativeObservation],native_records:list[TargetStatement],authority:Record)->Bundle`; collect CLI`--request --policy --expected-policy-sha256 --output`, exact-once options/no secret arguments.

- [ ] Freeze/run matrix `CollectionControls` RED.
- [ ] Acquire actual B/S sources/native originals/reviews once; preserve Main origin/copy and Math parent resolution. Reuse only qualified same-context captures, never reconstruct execution.
- [ ] Implement lossless framing, actual compiler/dependency/scope/import projection and R01 inventories. Preserve raw records/refs; unavailable types are unresolved, not semantic null deltas.
- [ ] Hash/fsync/seal/preserve/read back the complete Bundle/GraphDescriptor/terminal even for pre-worker failures. A source-content store may deduplicate explicit source identities; distinct native execution/attempt originals must all remain preserved.
- [ ] Run C GREEN plus an authentic unchanged B/S acquisition and actual negative/partial cases. Recollection preserves request/content hashes while separate custody advances; it creates no kernel/alignment currency.

## Task6: Complete impact and actual detached revalidation

Files: Core/tests, authentic review records in governed custody.
Interfaces: `compare(base:InventoryResult,candidate:InventoryResult,adapter_reports:list[Record],graphs:Record,authority:Record)->ImpactReport`; `validate_revalidation(record:Revalidation,bundle:Bundle,authority:Record)->Record`, exactly `{accepted,unresolved_original_groups,issues}`. Adapter CLI retains four once-only expected options and labeled context; original packets/reports remain unchanged.

- [ ] Freeze/run matrix `ImpactControls` RED.
- [ ] Implement separate semantic/evidence channels and pinned gate union traversal of every old/new edge. Original graphs remain original; hold clones are derived. Select refs from authentic OLD links, retaining deletions and all surviving group uses.
- [ ] Validate complete detached context/groups/refs/native authority against R01; preserve reviewed_graph/reviewed_bindings and6112554711. New commit/native success alone cannot revalidate.
- [ ] Obtain real independently authored/accepted revalidation for an affected group, retaining originals/replacement acceptance. Missing review is HOLD. Task10 uses a different unfitted transition.
- [ ] Run C GREEN and distinct whole-group/native-ABI conservation review over authentic original/candidate reports.

## Task7: Independently admitted T/P controller and all-event coverage

Files: Controller/native/controls workflows, policy/runbook; `ControllerControls`.
Interfaces: Emits accepted RequestContext/Acquisition/Bundle; effective controller/producer/workflow/dependencies equal externally admitted T/P, candidate edits remain data.

- [ ] Freeze/run matrix `ControllerControls` RED.
- [ ] Admit immutable T and an actual installed route proving effective workflow source. Protected control refs require native readback; dispatch-by-SHA is not presumed. Metadata-only launchers cannot grant credentials; independent credential/publisher binding verifies actual T/job source, not YAML self-attestation.
- [ ] Implement separated trusted acquisition, tokenless production, offline verification and narrow publication; bind each native job/attempt/artifact. No credentialed candidate execution or ambiguous matrix output.
- [ ] Connect all listed PR events without path filters; base pushes invalidate B/S, evidence/review updates extend only unchanged source, metadata updates recheck context/idempotency. Main M is separate; branch pushes diagnostic; scope remains per domain.
- [ ] Run C GREEN and actual installed-route event/attempt/tamper/early-failure controls; independently admit exact effective T/P. A merge or Actions app15368 does not establish trusted control execution.

## Task8: Offline decision and actual narrow publisher

Files: Core/verify/publish/policy; `DecisionControls`, `PublisherControls`.
Interfaces: `verify(bundle:Bundle,admitted_policy:Policy,fresh_native_context:Record)->Decision`; verify CLI`verify --bundle --policy --expected-policy-sha256 --native-context --output`. `publish(sealed_PASS:Decision,final_native_context:RequestContext,admitted_installation:Record)->NativeObservation`; publish CLI`--decision --policy --expected-policy-sha256 --native-context --output`. Inject admitted native transport, never candidate URLs.

- [ ] Freeze/run matrix `DecisionControls`/`PublisherControls` RED.
- [ ] Implement independent raw-member/provenance/core recomputation and all Decision predicates. No-change requires full authoritative coverage/custody and current native positives; adapter exit0 or a diagnostic report is insufficient.
- [ ] Qualify actual app/installation/principal/effective check-only permissions, separate from acquisition/workers. No contents/ref/PR/settings authority; name/app-ID is insufficient.
- [ ] Publish recomputed sealed PASS only after final tuple equality; mark one attempt, then read back exact app/installation/check/body/conclusion at H with S/key explicit. Unknown delivery stays HOLD; public projection has separate allowlist/hash.
- [ ] Run C GREEN plus actual canary legitimate/wrong-principal/stale probes; independently qualify originals and production-scope equivalence.

## Task9: Protected final-write enforcement and conditional activation

Files: Verify`admit` mode, enforcement tests, policy/runbook; protection/final-write integration needs separately named actual scope.
Interfaces: `admit(request:RequestContext,decision:Decision,fresh_native_context:Record,check_readback:NativeObservation,admission_capability:Record)->Record`, exactly `{admitted,reasons,observed_context}`; consumed at the participating write boundary, never a permanent permission token.

- [ ] Freeze/run matrix canary `EnforcementControls` RED; no unsafe production merge.
- [ ] Read current rules/app/check/conversation/write routes. Add formal-regression via qualified operator/native full-rule readback, preserving verify/public-intake/public-shop and no bypass.
- [ ] Implement fresh report/check/B/H/S/T/P/source consumption and real drift refusal. Expected-H alone does not enforce expected-B. Probe actual race behavior; separately qualify any queue/other merge mode's real parents/events.
- [ ] Prove rejection on claimed routes; disclose unconstrained routes. Missing race-control capability keeps diagnostic/non-success/HOLD; a fresh GET is not atomic enforcement.
- [ ] Run C GREEN and non-destructive production negatives/readbacks; independent policy/install/protection/route review precedes blocking activation. Main does not certify Math/all writers.

## Task10: Unfitted real transition and actual nonauthor review

Files: Private held-out source/review/capture/report originals, enforcement tests/runbook; no scientific registers.
Interfaces: Real B/S/mapping/old whole review -> ImpactReport -> authentic Revalidation/AuthorityAcceptance -> protected Decision/publication/admission.

- [ ] Have a qualified owner seal an unfitted real statement/premise transition, downstream original review and selection lineage; synthetic fixtures/Main31 cannot substitute.
- [ ] Collect genuine sources/declarations/types, complete union impact/exact refs and real proof/unscoped/selector/deleted-edge calibration; execution failure is not semantic change.
- [ ] Obtain actual nonauthor mathematical/alignment revalidation and native acceptance for every surviving group use at exact context. Retain originals/exposure/lineage; acquire a missing reviewer/capture/authority through project coordination, never self-approve.
- [ ] Run the real transition through private readback/verifier/required publisher/protected admission; retain exact positive/negative evidence.
- [ ] Obtain distinct whole-source/security/evidence qualification and scoped M7 acceptance. Domain reviewers handle science; implementation promotes none.

## Task11: Same validated artifact, domain rollout and recovery

Files: GraphDescriptor/collector/controller/policy/runbook; SSG paths remain with actual separately reconciled Goal1/4 owner.
Interfaces: Exact descriptor fields`carrier_graph_ref,upstream_provenance_ref,gate_sha256,exporter_source_refs,graph_artifact_ref,graph_sha256,regression_report_sha256`; S/native run/attempt remain bound in custody. Deliver original exporter-compatible graph bytes, not a hold clone/re-export from another source.

- [ ] With actual bundle/SSG owner, qualify same-graph acceptance and wrong artifact/run/attempt/graph/report/S refusals using exact pins/native build readback. Keep SSG open until paired consumer evidence.
- [ ] Roll out each domain after its own full gates; exercise ordinary/fork/metadata/base-update/failed/cancelled/missing/unchanged events, not one Main GREEN.
- [ ] At M verify parents/tree against S and observe natural Main push separately. Recheck source/claims/checks before protected integration; conserve SITE/RAG/unrelated leaves and negatives.
- [ ] Exercise denied/unknown delivery, changed-input acquisition, retention transfer and last-admitted-T/P rollback without stale success/old-output loss. Unsafe enforcement stays non-success; temporary policy rollback requires named technical disposition/native readback, no silent bypass. Retain rules/reports.
- [ ] Obtain independent whole-branch/source/evidence review and per-domain acceptance/recovery records. Release only completed scopes; full8 additionally requires the other seven requirements and actual same-artifact SSG.

**Execution handoff:** Fresh whole-PLAN review/admission precedes finite implementer/reviewer assignments. No user reapproval; current source, unchanged controls and qualified originals govern.
