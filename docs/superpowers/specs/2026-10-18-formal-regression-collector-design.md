# Source-bound formal regression collector and required consumer

Draft R01 for fresh whole-spec review, 18 October 2026. Original author: OpenAI/Codex `/root/native_object_conservation_1017`, source exposed, org0. Root assigned this design-only continuation at PR325/6243925080 (00:52:16 UTC), the separate extension promised by6069575569, not an older accepted collector handoff. No implementation plan, code or activation. Science NONE/authority false; all8 ACTIVE_UNMET.

## Intent and established boundary

Complete original Task2: preserve actual base/candidate reports for every PR, consume them at an exact-candidate required gate, and qualify actual formal-change controls/nonauthor review. Preserve existing graph, reviews, negatives and independent evidence axes; create no global graph/status/ownership register. Requirements1/4 still need the same validated graph artifact consumed by the SSG. This design supplies that interface, not its unfinished integration.

The frozen source basis is Main `6fce9b8c1a9e723dd4983bfc0d2ebeede09c87f8`, tree `5b3447fa17a4f6596ad585da81cd249b3906404d`. Later Main changes require a new source qualification. Existing adapter76502B/4b73b4a7 and CLI1509B/ad759f3f validate supplied captures, not authentic collection. Their exact v1 ABI stays unchanged. The graph gate remains independently pinned to SHA256 `a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8`; packet provenance cannot change executable authority. Main/Math native formats stay distinct. Main's31 SIDE24 targets supply no graph-node mapping. Missing real mappings and review custody therefore remain unresolved and prevent blocking-mode success. Current carrier is `docs/site/dependency-source/GRAPH.json`:49nodes/55edges, dated Math origin `7858329974e28be79f29b22644370084ff43da4f`, not current Math.

## Alternatives and selected approach

| Approach | Operation and tradeoff | Decision |
|---|---|---|
| Trusted controller, isolated producers, separate publisher | Default-trusted metadata code acquires native sources; reviewed isolated producers execute candidate data without credentials; independent verifier/publisher supplies a new required condition. Needs actual transport, publisher and freshness-enforcement capabilities. | Selected; these capabilities are not presumed installed. |
| Candidate-owned workflow inside existing `verify` | Add packet/report dependencies to current PR workflow. Simple, but PRs can edit workflow, verifier and dependency code. Reusable names/Actions app15368 do not establish trusted execution. | Rejected as trust boundary; fixture discovery remains useful. |
| Detached owner packets/manual admission | Authenticate each packet manually, then use current adapter. Useful changed-input recovery, but no automatic every-PR collection/enforcement. | Recovery only; never automatic successful fallback. |

Selected required condition: existing checks AND distinct `formal-regression`. Preserve `verify`, `public-intake`, `public-shop` and their meanings. Later workflow/publisher/protection edits require exact-scope coordination; proposed names transfer no peer ownership.

## Roles and immutable context

The trusted controller reads metadata and source only. It never executes a PR checkout. Its effective executable/control revision `T` and policy digest `P` are independently admitted outside the PR. A candidate change to those files is data until a separately reviewed control release is admitted. Even a newer default branch does not silently replace `T` or `P`.

Request context binds repository/PR, base repository/ref/commit/tree `B`, head repository/ref/commit/tree `H`, service checked-merge commit/tree `S` and parents, admitted control revision `T` and policy digest `P`. `B` is current target tip, not fork point. Required comparison is `B -> S`; retain `H` separately as attribution/native check status subject. A run `head_sha`, target-event `GITHUB_SHA`, or `H` is never silently equated to tested `S`. Event/collection identity and native observations are acquisition metadata, separate from stable source context.

Require native readback of the configured `B/H/S` parent relationship and exact tested tree. Missing/conflicting service merge is `UNRESOLVED_CANDIDATE`, never head-only fallback. Optional head/fork-point comparisons are diagnostic. Different service merge semantics need separate review and an actual capability witness.

Producer context `N` separately records repository, checked commit `K`, run head, workflow/control source, job, run ID/attempt, purpose and conclusion. Each capture retains its context. Supervisor records requested `K` separately from native event/control SHA; changing environment values cannot turn an event at `T/B` into native proof of checkout `S`. A Main run cannot refresh Math evidence; equal bytes or a new sidecar cannot rebind an old receipt from `K` to `S`. Landed merge `M` is a new observation: verify parents/tree against `S`, retain the original checked identity and qualify any Main-push run separately; tree equality does not relabel an `S` run as `M`.

All PR open/reopen/synchronize/base-edit/ready events request a report, without path filters. Metadata-only updates reuse immutable captures only after fresh context checks, not execution replay. Base pushes invalidate affected outstanding contexts; producer completion/review updates may extend evidence for unchanged context. Main push is separate postmerge observation, never retroactive admission. Branch pushes supply diagnostic evidence only. Coverage is per admitted repository/domain: initial Main rollout does not establish Math/all-repository enforcement. Current collector/product ownership remains UNKNOWN; this assignment owns only the draft.

## Collection contract and source membership

`collect(request, policy, native_reader) -> bundle` requires authenticated commit/tree/blob and PR/merge reads, attempt-specific job/artifact identity, one bounded original-artifact read, review retrieval and preservation readback. Agent-connector availability proves no CI transport capability. Missing/denied capabilities preserve non-success; no alternate identity, retry, guessed attempt or PR-selected URL.

Save complete native responses before parsing, with request/time/principal-scope labels and byte digest; exclude tokens/access URLs. Resolve trusted-policy/typed paths. Verify bytes/SHA/Git-blob framing, repository/commit/path membership, native mode/tree, and repeated identities. Reject duplicate paths/keys, symlinks/special files/submodules, escapes and unsupported LFS. Reconstruct complete trees where possible; distinguish verified tree/blob hash from authenticated service commit-to-tree assertion. Normalized metadata alone does not reconstruct a signed commit hash.

Inventory relevant package members, including manifest-bound/unbound files, scope/imports, toolchain/locks, producer/control source and informal origin/local copy. Preserve Main origin versus checked-copy roles. Resolve Math file keys against its manifest parent without rewriting them or adding Main rows. Missing/unsupported source is availability loss; no same-title/target/hash substitute.

Use the existing admitted validated graph, its carrier GitRef, dated upstream provenance and independently pinned gate. Qualify carrier/upstream membership separately; preserve original nodes/edges and dated origin. Emit existing exporter-compatible `graph.json` and bind its bytes. A changed graph/provenance pin requires source review; private evidence custody and public export remain separate.

## Authoritative mappings and actual statement inventory

Proposed `claims/formal-bindings.json` links each graph node explicitly to native format/package/target/module, known informal source/slices and exact original alignment GitRefs. Snapshot-relative references carry path/blob/SHA/bytes plus `base`/`candidate` role; collector resolves actual `B/S` commit. External origins use full immutable refs. External acceptance binds resolved manifest/graph/source identities, avoiding self-referential commit hashes. Candidate links require admitted mapping-authority acceptance; merged files, titles/shared hashes do not authorize links.

Coverage enumerates every admitted node/target as linked, explicitly outside domain with accepted scope evidence, or unresolved. Empty/missing/ambiguous/lost mappings fail coverage. Candidate-only exclusion is unresolved; Main's31 targets cannot certify global scope. Current authoritative mapping/acceptance is absent: diagnostic output must expose this and blocking success is unavailable.

Collect both sides' exact manifest rows, modules, scope, dependency/toolchain/compiler/options, and target records from original qualified `elaborated-types.log`. Use lossless target-header/indented-continuation framing; retain whole log plus every record's byte offsets/hash. Require one complete record per native target, no foreign/duplicate/ambiguous output and admitted print options. Main uses `pp.explicit true`. Printed type data is not semantic normal form/equivalence, and a parser cannot authenticate `#check`-shaped text.

Authenticate producer source/runtime before crediting observed statement data. Old native evidence binds `B`, candidate binds `S`; an older checked identity remains explicit and unresolved for that side. Byte conservation is not fresh execution. Changed toolchain/options/dependency context makes comparison unknown and requires revalidation even when printed text matches.

Comparable changed type bytes yield conservative `formal_statement_changed`; target addition/deletion, consumed-premise, scope/manifest-contract changes also seed revalidation. A changed module with equal type records is `formal_source_changed_unscoped` unless accepted body/declaration selectors and unchanged statements/context establish proof-only change. Whole-file hashes never become statement slices; equal types never certify informal alignment. Preserve6112554711: informal selector-only edit/loss can hold alignment without asserting failed/stale kernel evidence by itself.

## Native production and custody qualification

Native workflow verifies `B/H/S`, uses admitted control/producer `T` and separately prepared pinned runtime. Candidate config/source is untrusted. Compile/build runs only in a disposable hosted sandbox with reviewed nonroot/no-network/resource/cleanup controls, read-only source/control/toolchain, separate bounded output and no repository/cloud/publisher credentials. PR workflow/tests/hooks/setup never execute in privileged controller jobs. A candidate producer script cannot attest itself.

Trusted supervisor records actual source, commands/executable pins, exits and runtime bounds outside candidate-writable storage; stops/removes execution before collection. Reuse Main producer semantics/format through a reviewed supervisor adaptation, not an assumption the existing script safely handles arbitrary packages unchanged. Real adversarial controls must address metaprogramming, output injection and build config. Sandbox/stdout/artifact presence proves neither perfect protection nor semantic review.

Join native run/attempt/job/artifact/member identity, original ZIP/member bytes, receipts/logs and supervisor observations. Current Main receipt does not independently check an attempt field; required binder adds environment/output context. Qualify attempt through actual attempt-specific artifact/job/supervisor custody or mark `ATTEMPT_UNKNOWN` and refuse positive execution. Never infer it from name/current metadata/adapter expectation. Preserve failed/skipped/cancelled/timed-out/missing/negative-control records; only genuine successful positive checks support their native target kernel axis.

## Regression, originals and review revalidation

Keep unchanged adapter ABI/reports and `custody: unknown`. Produce historical-original and candidate projections using old/candidate formal records respectively and explicit expected native context. A single context is not cross-run/repository currency. Whole original groups must not be partitioned to hide uses. Math's absent Main-style bridge remains unknown; the wrapper does not promote adapter kernel axes.

Add a separate formal-impact projection because adapter fingerprints do not compare target types. Feed the same pinned `reverse_impact_between` complete per-node semantic fingerprints: accepted links, type/context content and scope/manifest contract. Track execution/availability loss through a separate evidence-impact channel. Preserve raw identities; run-ID change alone is not statement change. Normalize paired snapshot commit advancement only as current adapter permits. Validate actual graphs; traverse all old/new edges, including deleted/context edges. Original graphs remain unchanged; any hold-marked clone is derived only.

Union adapter/formal-impact nodes/reasons; retain deleted/added nodes, removed edges, unscoped changes and unknown mappings. Select affected original reviews only from authenticated old explicit links. Group by full original GitRef. Hold every surviving use of that whole record, including downstream uses; unmapped uses stay unresolved. Fresh run success never refreshes alignment. Execution-only loss fails the native/gate predicate but does not by itself invent a statement delta, stale semantic alignment or affected-review ref; preserve the existing independent-axis contract.

Retrieve original reviews/evidence by exact identity and authenticate membership/native acceptance/version. Mutable comments need acquired body/update hashes, stable final readback and an accepted immutable review record; ID alone is not immutable content. Native GitHub accounts do not prove model/provider/agent lineage: existing review authority must actually accept that lineage/evidence. Preserve exposure/org0.

Detached `formal-regression-revalidation/v1` binds stable request key, new graph ref, accepted mapping and formal-statement inventory digests (the one closed DATA.formal_inventory_hash_contract core/SHA256, with declarative preimages in INVENTORY-CONFORMANCE.json), manifests/scopes, every original GitRef and successor review/evidence refs. Require all surviving whole-group uses and actual accepted revalidation or explicit unresolved scope. Preserve native alignment schemas and reviewed_graph/reviewed_bindings rules. Authority's native acceptance binds exact detached bytes/context; newer commits or candidate `ACCEPTED` strings are insufficient.

Detached review commits refer to existing `S` without changing `H/S`. Preserve stale originals/affected refs after successor coverage; outer decisions link qualified replacements without erasing history. Collector never performs semantic review, invents reviewer PASS, rewrites originals or dispatches a mathematical verdict.

## Report, verifier and required decision

Bundle binds request/control/policy; raw inventory/membership/custody; accepted mapping/coverage; old/new types; unchanged adapter packets/reports; formal impact; original/successor reviews; exact graph artifact; retention/readback and issues. Index hashes every member and role; originals stay byte-exact. Stable request key hashes canonical domain `formal-regression-request/v1` and repository/PR/B/H/S/T/P plus merge mode. Exclude event/acquisition IDs, timestamps and evidence revisions: evidence may extend without moving source context. Different B/H/S/T/P means a new key.

`verify(bundle, admitted_policy, fresh_native_context) -> decision` uses only admitted verifier/adapter/gate bytes, rehashes members and recomputes projections. No supplied success/checksum/needs-output substitutes for custody. Authenticate bundle provenance against the admitted acquisition workflow/job/attempt and native artifact; a self-declared principal label cannot establish custody. PASS requires complete authoritative coverage, authentic source/types/native context, supported producer/attempt, current candidate binding, authenticated whole-group revalidation, successful preservation/readback and exact final context. Missing/failed/skipped/stale, malformed/partial/unknown/oversized input is non-success. Adapter exit0 with stale/unknown data is not gate success.

No-change PASS still requires complete authoritative coverage/custody and independently required current native checks. Before those exist, no-delta diagnostics remain unresolved. Decisions: PASS/FAIL/UNRESOLVED/SUPERSEDED; only PASS permits successful required result. Preserve historical adapter axes and scientific NONE.

Separate read-only acquisition from narrow check-publisher principal. Publisher executes no candidate code, accepts sealed recomputed PASS only after final native B/H/S/T/P equality, and reads back exact check identity/body/conclusion. Proposed name `formal-regression`; actual app/installation is an absent activation input, never presumed15368. Publish at native status subjectH, with explicit checkedS/request key. Capability tests must establish platform head/merge evaluation; H publication never asserts H==S.

Check success is as-of publication, not atomic later-merge authority. Participating merge admission re-consumes the sealed report and freshly verifies B/H/S, check/P/T/source guards at its write boundary; drift refuses, never rebinds. Cached strict rules are not current enforcement proof. Name/app binding cannot constrain alternate/bypass writers or every base race. Blocking activation needs actual stale/spoof enforcement probes and an admitted final-write path. Unconstrained routes keep enforcement incomplete; diagnostic collection may operate. No completed lease broker/global Main lock is invented.

## Preservation, privacy and recovery

Use fresh bounded acquisition directory, complete sealed members/index/report, fsync and atomic promotion outside source. Failure never replaces prior qualified output. Retain adapter limits16MiB packet/4MiB capture/256nodes/2048edges. Acquisition caps64MiB retained originals/decompression,1024members and4MiB per metadata response, checked before allocation. Oversize/truncated acquisition is preserved as an explicit negative, never evidence completeness; greater capacity needs reviewed policy, not splitting away whole-review coverage.

Preserve native artifact IDs, ZIP/member hashes and actual upload/readback. Minimum retention30days is a deadline, not permanence; long-term acceptance needs existing governing-custody transfer/readback before expiry. Unavailable storage/disclosure capability blocks that claim. Every PR acquisition, including refusal/incomplete cases, preserves a bounded terminal report; failed preservation prevents success. Recollection is a new acquisition, not reconstruction of lost execution.

Public output is allowlisted public-relative paths/opaque IDs, permitted hashes/context, reasons/scope/freshness. Exclude credentials, private paths/access links, raw private source/review bodies and secret logs. Originals stay in admitted private custody; allowlist rather than cleanup regex. Public projection has its own digest; private digests do not authenticate unavailable contents.

Refusal/denial/unknown delivery stops without blind retry. Changed-input recovery gets a new acquisition ID and preserves negatives. Duplicate delivery is idempotent for same source/evidence; evidence changes create sealed revisions. Cancelled/missing output never succeeds. Retain actual stop/remove/source-after custody before cleanup; missing cleanup observation is separately unresolved.

## Proposed file and interface boundaries

These are prospective paths only, requiring later exact-file ownership reconciliation. No files below are created by this draft except this private proposed spec.

| Proposed repository path | Responsibility |
|---|---|
| `architecture/formal_regression.py` | Closed bundle/mapping/revalidation contracts, statement framing, pinned adapter/gate projections and conservative policy decision; no network or credentials. |
| `tools/formal_regression_collect.py` | Trusted native-reader transport, membership acquisition and atomic bundle preservation. |
| `tools/formal_regression_native.py` | Trusted supervisor adapter for isolated native production and external command/source/runtime observations; no publisher credentials. |
| `tools/formal_regression_verify.py` | Offline recomputation and final-context admission interface; no publication authority. |
| `tools/formal_regression_publish.py` | Separate narrow check publisher/readback; rejects unsealed, stale or non-PASS input. |
| `.github/workflows/formal-regression.yml` | Trusted metadata/event controller and separated acquisition/verification/publication jobs. |
| `.github/workflows/formal-regression-native.yml` | Trusted supervisor/preparation and credential-free ephemeral native production; source-owner review before use. |
| `docs/architecture/formal-regression-policy.json` | Public-safe reviewed policy: admitted repositories/domains, source/control/gate/producer pins, reviewers/mapping authority, limits, custody/disclosure and enforcement mode. Its admission pin lives outside candidate data. |
| `claims/formal-bindings.json` | Explicit versioned links/coverage; actual authoritative records are not supplied by this draft. |
| `tests/test_formal_regression.py` | External contract/security/negative controls against the proposed CLIs and preserved originals. |
| `tests/test_formal_regression_enforcement.py` | Hosted actual-context/publisher/protection/admission controls, separately qualified from fixture tests. |

Existing adapter/gate/exporter, native schemas, formal sources/reviews, shared workflows and scientific registers remain dependencies. Required changes need exact-scope extension/review. Graph descriptor exposes original provenance/exporter pins and exact graph/report digests for the future same-artifact SSG consumer; no replacement graph.

## Evidence required before closure

Hosted controls: forged self-consistent membership/receipts; missing modes/tree members/identity collisions; privacy/path/archive/resource attacks; PR workflow/verifier tampering; failed/skipped/missing/stale/ambiguous-attempt producers; shell/output failure; stale B/H/S/T/P/wrong app/status subject; deleted/context-edge impact; partial whole-review revalidation; unknown/lost mappings; actual type versus proof/unscoped deltas; toolchain mismatch; atomic preservation/cleanup failure; and legitimate unchanged positives. Synthetic controls are not authentic capture or semantic review.

After source review and genuine hosted RED/GREEN, use a held-out actual transition not fitted to implementation: native old/new sources/types, authoritative link, changed formal statement/consumed premise, downstream whole original review, complete union impact/exact refs, real independently authored authenticated revalidation, preserved report and protected consumer/admission result. Include proof-only/unscoped calibration and deleted edge. Existing qualified reviewers assess mathematical validity/alignment; this author supplies no real witness, mapping or review PASS.

Absent activation inputs: accepted resolved mappings/coverage and authority; original review/evidence identities and revalidation authority; admitted control/producer/runtime revisions; working attempt-specific transport/artifact/private custody; narrow publisher installation; current protection/final-admission configuration with actual spoof/stale/missing-producer rejection; and held-out real transition. Defined preactivation behavior is diagnostic preservation/non-success, not another handoff search or accepted implementation. Root retains whole-spec review, later plan/source/runtime/integration/all8 decisions. Author reserves no later role.
R01 author `/root/req8_inventory_contract_revision_0133` addresses only review R1 under PR325/6244260092. Predecessor remains immutable; fresh technical review is pending.
