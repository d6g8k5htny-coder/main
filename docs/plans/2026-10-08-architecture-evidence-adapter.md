# Architecture Evidence Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate inert evidence packets, preserve their original records, and report independent evidence dimensions and loss-only downstream revalidation without authenticating packet custody or promoting science.

**Architecture:** A stdlib CLI consumes one closed-schema packet on stdin and emits one deterministic report on stdout. Separate adapters retain the main and Math formal formats and existing retrofit records; the independently pinned graph gate supplies graph validation and old/new union reverse-impact semantics. Actual Git capture, native artifact readback, and formal producer custody remain a separately claimed follow-up.

**Tech Stack:** Python stdlib, JSON/base64/SHA-256/Git blob identities, unittest subprocess controls, existing pinned `hard_gate.py`, hosted ephemeral testing in both normal and optimized interpreter modes.

**Spec:** [Research architecture design, sections 1/4 and 8](../superpowers/specs/2026-10-08-research-architecture-design.md); [original Task 2](../superpowers/plans/2026-10-08-research-architecture-plan.md). This bounded plan replaces that task's proposed module layout and internal four-argument interface with the agreed packet/CLI ABI below.

## Global Constraints

- Starting checkout: main `ad4509d58aea5b1b1ac32dbea7f2be3b591a1345`; worktree `project-evidence-adapter`; coordination pickup main307/6069575569.
- This increment owns new adapter, CLI, tests, and this plan only. Formal workflows, manifests, Lean sources, existing reviews, site assets, and other owners' branches are outside the edit scope.
- Drive/R17 governs research records; GitHub executes and reviews. Scientific effect is always `NONE`; scientific status authority is always exact boolean `false`.
- Source availability, kernel verification, computation, alignment, and review remain independent. Missing evidence is not a pass. Profiles are derived reports, not another scientific-status register.
- Preserve complete original node/edge/review/receipt records. Report holds separately; never rewrite classifications, controlling flags, review verdicts, or original receipts to match the report.
- Input hashes establish internal byte consistency only. No input `authenticated`, capture flag, signature-looking string, or self-consistent native metadata can authorize external custody. Global and every axis custody remain `unknown`.
- Execute no packet-supplied Python, gate, Lean, shell, or test code; fetch no packet-selected URL/path. Trusted gate bytes must match `a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8` before loading their verified buffer.
- No project Python execution on the owner laptop. Observe genuine hosted missing-CLI RED before implementation; retain exact source, normal/optimized logs, failures, and artifacts. Standing owner authorization requires no new permission checkpoint.
- The corrected external suite is frozen at 35 methods in `tests/test_architecture_evidence_adapter.py`, SHA-256 `397581308cc517545b5e0e8a1716ce696a14aa7e9f885f61cd07dd46dcb7de63`. Its native-contract corrections below are the current test authority; the earlier 32-method draft is superseded. The implementation owner creates only the two new adapter/CLI files after the integration owner records genuine hosted absent-CLI RED for these corrected bytes.

## Review Focus

- Self-consistent forged Git/run identities remain internally checkable but externally unverified; Task 1 pins this distinction.
- Deep, oversized, duplicate-key, nonfinite, type-coerced, or contradictory captures fail without success JSON; Task 1 pins parser limits and exact types.
- A proof-body edit cannot become a statement edit; absent or incomplete declared slices yield unresolved coverage; Task 3 pins both cases.
- Partial alignment, same-provider lineage, and negative-control runs cannot become whole-manifest alignment or positive kernel execution; Task 2 pins the distinctions.
- Removed dependencies, deleted nodes, context edges, and multiple original alignment identities remain visible in union impact; Task 3 pins exact traversal and affected-record identities.

---

## Agreed v1 ABI

Create `architecture/evidence_adapter.py`, `tools/architecture_evidence_adapter.py`, and `tests/test_architecture_evidence_adapter.py`. Synthetic fixtures live inside the test module; no separate fixture files are required. The CLI requires each of `--expected-repository`, `--expected-commit`, `--expected-run-id`, and `--expected-run-attempt` exactly once. They compare declared native metadata; they do not attest capture. The module exposes `adapt_packet(packet: dict, expected_context: dict[str, str]) -> dict`; the CLI exposes `main(argv: list[str] | None = None) -> int`.

Stdin is one UTF-8 JSON document, at most 16 MiB, maximum nesting 64. Every decoded capture is at most 4 MiB. Reject duplicate keys, nonfinite numbers including exponent overflow, trailing documents, unsupported wrapper keys, invalid base64, and nonexact identity/slice types. The tests define the original native document fields supported by each format; no invented merged native schema is accepted.

The exact packet is `{schema_version: 1, old: Snapshot, new: Snapshot, formal_records: [Formal], retrofit_records: [Retrofit]}`:

| Type | Exact wrapper fields |
|---|---|
| `Snapshot` | `graph: GitCapture`, `bindings: [Binding]` |
| `Binding` | `node`, `source: GitCapture or null`, `statement_slice: Slice or null`, `proof_slice: Slice or null` |
| `Slice` | `start_byte`, `end_byte`, `sha256` |
| `GitCapture` | `ref: GitRef`, `raw_base64` |
| `GitRef` | `repository`, `commit`, `path`, `git_blob`, `sha256`, `bytes` |
| `RunCapture` | `ref: RunRef`, `raw_base64` |
| `RunRef` | `repository`, `checked_commit`, `run_id`, `run_attempt`, `artifact_id`, `member_path`, `sha256`, `bytes` |
| `Formal` | `id`, `format`, `manifest: GitCapture`, `scope: GitCapture`, `source_files: [GitCapture]`, `receipt: RunCapture or null`, `logs: [RunCapture]`, `alignment: GitCapture or null`, `native_run`, `node_targets: [{node, target}]` |
| `native_run` | `repository`, `checked_commit`, `run_head_sha`, `run_id`, `run_attempt`, `purpose`, `conclusion`, `expected_conclusion` |
| `Retrofit` | `id`, `record: GitCapture`, `node_records: [{node, record_id}]` |

`format` is exactly `main-formal-gate/v1` or `math-formal-gate/v1`. SHA-256 is lowercase 64 hex; Git commit/blob is lowercase 40 hex; counts and offsets are exact integers; native run/attempt/artifact IDs are positive ASCII decimals without leading zeros. Paths are normalized safe relative paths. Verify raw byte count, SHA-256, and Git blob framing/hash against each capture before parsing embedded documents. Slices bind byte offsets and hashes in their explicit source capture and are **declared byte slices only**, not a semantic Lean parser or coverage certificate.

### Resolved native contracts

All Git captures share one packet-wide immutable-identity registry keyed by `(repository, commit, path)`. Repeated captures of that identity must have identical verified raw bytes, byte count, SHA-256, and Git blob identity. Conflicts are a refusal even when the captures occur in different graphs, bindings, scopes, source lists, alignment records, or retrofit records. Consistency within each wrapper alone is insufficient.

Main's manifest `sources` rows identify the original informal source separately from the checked main repository's `local_copy`. The origin can be an older Math commit and path while the local copy lives at a current main path such as `formal/sources/...`. Compare the node's explicit source binding to the complete origin identity, then independently verify the local copy against the native `files` hash and checked main repository/commit. Keep both roles and their original fields; never demand that the origin use main's checked commit, or join the node by the local-copy path. A mismatch in the origin join leaves kernel applicability unknown; contradictory capture or file identities are a refusal.

Math's native `files` keys are relative to the captured manifest's parent directory. Derive a safe root-relative Git lookup path by joining that parent with each native key, and match the resulting path to `source_files`. For example, a manifest at `TEST/formal/manifest.json` with key `SCOPE.md` binds the capture at `TEST/formal/SCOPE.md`. The retained manifest still contains its original `SCOPE.md` key. This lookup rule does not add main-only source rows, receipt fields, or a node-to-target bridge to Math.

Math alignment keeps its source-native `author`, `reviewer`, `proposal_authors`, and target declarations. Normalize lineage comparison values with split-whitespace joining and case folding; compare the reviewer with the primary author and every declared proposal author for provider, family, and agent. Empty or placeholder lineage, a matching normalized lineage component, and invalid proposer target scope cannot become current alignment. Each proposer target declaration must be a valid unique subset of the complete native target inventory. Valid partial coverage and unresolved lineage stay preserved as stale/unknown evidence, with no new accepted-review marker or organizational-independence credit. Preserve all original native annotations, including main's meaning, coordination, author, unbound-file, and negative-control fields, rather than rewriting either format into a merged schema.

The exact report has `schema_version`, `scientific_effect`, `scientific_status_authority`, `custody`, `original_packet`, `dimensions`, `formal_summary`, and `regression`. `original_packet` preserves the complete typed input, including original base64 bytes. Sort only derived lists/maps; do not reorder retained native arrays or alter retained native fields.

- `dimensions` covers exactly surviving new graph nodes. Each has `source`, `review`, `kernel`, `computation`, and `alignment`, each exactly `{record_state, applicability, reasons, evidence_ids, custody}`. `record_state` uses `recorded`, `not_recorded`, `unknown`, `not_applicable`; applicability uses `current`, `stale`, `unknown`, `not_applicable`. Applicability describes packet-internal consistency only; custody always remains `unknown`.
- Every `formal_summary` row has exactly `{id, format, manifest_targets, node_targets, retained_manifest, retained_receipt, retained_alignment}`. Preserve the original decoded documents and original node-target bindings; `manifest_targets` is the complete sorted native inventory. Null receipts/alignment stay null.
- `regression` has exactly `{changed_nodes, impacted_nodes, statement_changed, proof_changed, unscoped_source_changed, traversal_edges, revalidation_required, affected_alignment_records}`. `revalidation_required` is sorted `[{node, reasons}]` for affected surviving new nodes. `affected_alignment_records` contains exact original alignment `GitRef` objects, deduplicated by full identity and sorted by canonical JSON. Select them only from explicit mapped impacted nodes or stale whole-manifest/scope contracts; unknown mappings never invent an affected record.
- Malformed or contradictory packets exit nonzero, write refusal detail to stderr, and emit no success report. Valid unknown/stale/negative evidence produces a report with exit zero. Output is deterministic and cannot mutate retained records.

## Task 1: Bounded packet integrity and preservation

**Files:** Create the adapter module, CLI, and external CLI test module listed above. Tests launch the CLI as a subprocess, propagate `-O`/`-OO`, and use `-B -S`; they do not import implementation code.

**Interfaces:** Consume the complete v1 packet and expected context. Produce exact typed captures and a deterministic report envelope, preserving all originals with custody `unknown`.

- [ ] Write subprocess controls for valid deterministic output, preservation, every required CLI argument, matching/changed declared native context, and refusal of any invented capture-auth override. Fixtures are labeled synthetic `TEST.A/B/C` with `TEST.Formal.alpha/beta`; they make no actual capture or execution claim.
- [ ] Add controls for duplicate/nonfinite/deep/oversized stdin, malformed base64, decoded capture limit, byte/SHA/blob mismatch, unsafe paths, offset/hash/type errors, unknown wrapper keys, and multiple documents. A bool must not pass an integer check.
- [ ] Pin packet-wide immutable Git identity consistency, including conflicts crossing graph/source/formal capture roles; do not limit collision checks to a single record or list.
- [ ] Publish the tests-only candidate and run `python -B -S -m unittest discover -s tests -p test_architecture_evidence_adapter.py -v` and the same command with `-O` in the hosted restricted lane. RED must explicitly identify the absent CLI; an import crash, zero discovery, or unrelated failure does not satisfy this step. Root records exact candidate/run/artifact identities and original logs.
- [ ] After observed RED, implement bounded strict parsing, inert capture verification, `adapt_packet`, and the stdout-only CLI. Load only independently pinned repository gate bytes; never a gate supplied in the packet.
- [ ] Repeat those hosted commands on the exact candidate. Require all controls executed, no skipped tests, and identical discovery in both modes. Freeze and commit this independently reviewable packet boundary through the integration owner.

## Task 2: Native evidence adapters and independent dimensions

**Files:** Extend `architecture/evidence_adapter.py` and `tests/test_architecture_evidence_adapter.py`; CLI ABI remains unchanged.

**Interfaces:** Consume verified captures, native documents, explicit target bindings, and expected context from Task 1. Produce all five axes and retained native formal summaries without new authority fields.

- [ ] Pin main's dict-target inventory, source rows, file hashes, original receipt/log bindings, and Math's distinct list-target inventory in external controls. Source files must cover manifest file hashes; original receipt/log axiom inventories must name only native targets and preserve permitted axioms. Incomplete positive inventories remain unresolved and cannot produce current kernel applicability.
- [ ] Verify main origin and checked local-copy identities separately, including a Math origin at a historical commit with a current main local copy. Rebase Math native relative file keys only for safe source-file lookup and retain all native key spellings and annotations unchanged.
- [ ] Add valid failed-build controls with null receipt and retain failed/skipped/cancelled/negative records. Positive kernel applicability requires `purpose == "check"` and actual `conclusion == "success"`; matching caller-declared `expected_conclusion == "failure"` cannot make a negative control positive.
- [ ] Pin explicit main target joins: node target, node statement slice, exact manifest source identity `(repository, commit, path, bytes, SHA-256)`, and exact informal-anchor bytes within the declared slice. Unknown mappings stay unresolved. Math's native format lacks this main source-row bridge; preserve its inventory/receipt without fabricating an equivalent join.
- [ ] Add whole-manifest alignment controls: exact manifest and scope hash bindings, unique full target coverage, accepted disposition, and distinct declared author/reviewer provider/family/agent. Partial/nonaccepted/stale records remain preserved with stale/unknown applicability; lineage strings establish no authenticated review or organizational independence.
- [ ] For Math, validate every proposal author's normalized lineage and unique native-target subset, including a conflict in a nonfirst proposer, whitespace/case collisions, placeholder agent values, and foreign target scope. This extends native contract checking without authenticating any declared lineage.
- [ ] Add independent source/review/computation controls against original retrofit records. Changing one axis must not award another; graph classification, review text, matching hash, or green native metadata cannot manufacture kernel, computation, alignment, or scientific acceptance.
- [ ] Observe the corresponding hosted RED controls before implementing the two native adapters and retrofit projection. Rerun the complete adapter suite in both modes, retain originals and review this unit before committing through the integration owner.

## Task 3: Loss-only union impact and exact alignment identities

**Files:** Extend the same adapter and external CLI tests. Existing gate/source/review files remain unchanged.

**Interfaces:** Consume both validated graphs, their explicit source/slice bindings, and Task 2 dimensions/native identities. Produce separate regression lists and exact original affected alignment refs; preserve original graphs unchanged.

- [ ] Write changed-statement, proof-body-only, missing-slice, unscoped-source change, availability-loss, failed/skipped/missing receipt, stale alignment, removed-edge, deleted-node, added-node, context-edge, and unknown-mapping controls. Include full-record nested type changes that ordinary Python equality can hide.
- [ ] Write the original alignment-identity control: old/new union impacts a downstream node, and `affected_alignment_records` names its original repository/commit/path/blob/SHA/bytes exactly; canonical sorting/dedup cannot merge different original identities. Unmapped nodes yield no guessed review ref.
- [ ] Observe hosted RED, then implement canonical full-record comparisons and pinned union reverse impact. The gate uses **all** old/new `(from, to)` edges, including deleted/context edges; required-edge cycle validation remains a separate rule. Only surviving new nodes appear in revalidation proposals.
- [ ] Keep stable reason codes: `node_record_changed`, `dependency_edges_changed`, `graph_context_changed`, `statement_changed`, `proof_body_changed`, `source_changed_unscoped`, `dependency_changed`; axis reasons include `manifest_binding_changed`, `scope_binding_changed`, `target_coverage_changed`, `alignment_not_accepted`, `lineage_not_distinct`, `native_run_not_success`, `native_run_negative_control`, `native_context_changed`, `missing_statement_slice`, `missing_target_source_binding`.
- [ ] On a proof-only edit, invalidate affected kernel evidence and explicitly byte-bound reviews. Whole-manifest alignment can stale when manifest bytes change; report that binding change without adding `statement_changed` if its declared statement slice is unchanged. A fresh successful kernel record cannot automatically refresh alignment.
- [ ] Run the complete hosted suite in normal and optimized modes on frozen bytes, obtain fresh nonauthor source review, reconcile findings, and preserve the exact report/artifact. Root verifies required current-candidate checks and performs protected integration; this plan author runs no code or Git writes.

## Completion boundary and follow-ups

This increment is complete only when the exact CLI ABI and every substantive control pass in hosted execution with original records preserved and reviewed. That establishes packet validation and regression reporting, not authentic Git capture, native artifact custody, actual semantic alignment, or all-eight architecture completion.

The current dated 49-node graph has no explicit node-to-Lean mapping. A shared informal-source hash does not make the full graph claim kernel checked; main's SIDE24 targets explicitly retain narrower scopes. Synthetic fixtures cannot fill that gap.

Separately claimed work must acquire exact Git `commit:path` bytes and original native run artifacts, verify archive digests with a fail-closed readback, bind producer identities/current native context, and retain original manifests/receipts/logs. The existing formal upload lacks artifact ID/digest/name outputs and its name omits attempt; the original receipt lacks attempt. Do not edit those producers here or invent authenticated capture from this packet. Base/candidate real-input collection, every-PR required enforcement of this new adapter, SSG artifact consumption, and actual Math transition integration remain unimplemented until separately tested and reconciled with their current owners.
