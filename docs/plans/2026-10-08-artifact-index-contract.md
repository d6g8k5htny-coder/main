# Artifact Index public projection v1 contract

Root owns this new boundary under main307 pickup6070431864 through00:00UTC9October. Base main879fb4f5. This is the first unit of the architecture interface plan. Standing owner authorization applies; no additional permission checkpoint is introduced. Product execution, RED/GREEN and installation occur only in disposable hosted environments. Full private captures stay outside Git and publication.

## Actual source and public policy

The real R17 source is spreadsheet `1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no`, worksheet ID200004, title `Artifact Index`, native12-column formatted-cell representation. The2026-10-08T22:18:57Z response capture is9273 UTF-8 bytes with SHA256 `03baf67b92c7b31390fd497431abf120204a6ccbde7d8b00395d1d40288f6ebf`. This is serialized native API formattedValue data, not raw Sheet-file bytes or an export. It has one sheet and three grid-data ranges: row0 header, rows935–936, and row1000, all starting at column0. Capture ranges are `'Artifact Index'!A1:L1`, `'Artifact Index'!A936:L937`, and `'Artifact Index'!A1001:L1001`, cell_fields `formattedValue`. No native Sheet write is part of this unit.

Header exact order: Artifact ID; Title; Org; Class; Priority; Topics / object tags; Status; Authority / canonical impact; Dependencies or supersession; Modified UTC; Source; Notes.

Public column policy exact order: Artifact ID; Title; Org; Class; Status; Authority / canonical impact; Dependencies or supersession; Modified UTC. Only these eight native values may appear in public row fields. Source and Notes are prohibited even under a caller-forged policy; Priority and Topics are excluded. Do not serialize the original capture into any output or failure artifact. The real policy selects only C151-MAIN285-LOG-BINDINGS-20261007-v1, C152-MAIN286-FACTUAL-REVIEW-20261007-v1, C215-GRAPH-ALIAS-AUDIT-20261008-v1. These are dated source records with their original negative/scope/independence labels, not current acceptance. Actual initial bindings are empty because there is no exact verified Artifact Index-to-graph mapping. All49 graph IDs remain unmapped. Synthetic tests use TEST IDs, synthetic spreadsheet IDs, and private canaries distinct from the real source; synthetic fixtures are not delivery evidence.

## CLI and typed packet

`python -B -S tools/artifact_index_export.py` reads one strict JSON document from stdin, emits exactly one public JSON document on stdout on success, and exits0. On every refusal, including CLI arguments, it exits nonzero, emits no stdout and exactly the public opaque ASCII constant `ARTIFACT_INDEX_REFUSED` plus one newline on stderr. Private values, base64, cells, paths and exception text cannot appear in diagnostics. It performs no network fetch, URL/path read, imports of captured code or filesystem writes. Module API is `project_artifact_index(packet, expected) -> projection`. The output is a projection, not an authenticated table or current status claim.

All five independent flags are required exactly once, with lower-case64-hex values: `--expected-capture-sha256`, `--expected-metadata-sha256`, `--expected-policy-sha256`, `--expected-bindings-sha256`, `--expected-graph-nodes-sha256`. No abbreviation or duplicate arguments. Expectations are supplied from trusted root capture/policy inputs, not inferred from stdin. They prove consistency with declared pins only, not external custody. Canonical identity JSON uses sorted object keys, compact separators, ensure_ascii=true and allow_nan=false, preserving typed values and array order. Capture SHA covers exact decoded raw bytes; metadata SHA covers the capture object with only raw_base64 removed; remaining SHA values cover policy, bindings and graph_nodes respectively.

Packet exact keys:
- schema_version: exact integer1 (bool refuses).
- capture: exactly {capture_id, captured_at, source, raw_base64}.
- policy: exactly {schema_version, artifact_ids, columns}.
- bindings: array of exactly {node_id, artifact_id, roles}.
- graph_nodes: array of explicit node IDs.

Capture ID is a nonempty ASCII identifier matching `[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}`. captured_at is a valid UTC second timestamp YYYY-MM-DDTHH:MM:SSZ. Source exact keys are {spreadsheet_id, worksheet_id, title, ranges, cell_fields}. spreadsheet_id is a nonempty ASCII identifier of at most192 characters; worksheet_id exact nonnegative integer; title nonempty string, at most256 UTF-8 bytes, no controls; ranges is a unique nonempty list of exact quoted title A-to-L ranges `'TITLE'!A<positive>:L<positive>`, nonreversed and ordered by starting row; title containing a quote is unsupported. cell_fields must be exactly formattedValue. Native metadata must agree with the declared spreadsheet ID, worksheet ID and title. Metadata is declared as-of, not immutable native revision authentication.

Stdin limit16MiB, decoded capture limit4MiB, JSON nesting at most64 container levels (a scalar leaf does not add a level), at most4096 nonheader native data rows plus the one header and4096 graph IDs/bindings/policy IDs, each formatted cell at most8192 UTF-8 bytes. Artifact and graph IDs use the same ASCII identifier pattern and192-character bound as capture_id. Reject duplicate object keys, nonfinite numbers, multiple JSON documents, malformed UTF-8/base64/noncanonical base64, booleans used as integers, overflow, empty or unknown envelope fields, and mismatching pins. Unknown native annotations stay private and inert; no unknown native field is projected.

Native response is an object containing spreadsheetId and one sheet. The sheet properties must identify the declared worksheet/title. Grid properties must declare rowCount positive <=10000000 and columnCount exactly12. Its data list must have exactly one grid block for each declared range in the same order. Missing startRow/startColumn means0; supplied offsets are exact nonnegative integers and must equal the range's0-based position/column0. Every block has exactly its declared number of rowData rows. The header row has exactly12 values; every other row has10 to12 values, because a formattedValue read omits trailing empty cells, and only the trailing private Source/Notes cells may be absent. An absent trailing cell reads as an empty string and is never projected. Rows with fewer than10 or more than12 values refuse. Each cell is an object; missing formattedValue means empty string, supplied formattedValue must be a string within the bound. Unknown annotations remain inert. Blocks may not overlap and must be inside declared grid bounds. Exactly one row0 contains the exact12-column header; all other rows have a nonempty valid artifact ID, with no duplicate ID even on unselected rows. Nonselected row values and private columns never enter output. Selected IDs must each exist exactly once; unknown/duplicate/invalid policy IDs refuse.

Policy columns must equal the fixed eight public columns above, in that exact order. Policy schema_version exact1. artifact_ids is a nonempty unique list of valid IDs; its order is preserved for identity hashing but output rows sort by artifact ID. graph_nodes are valid unique IDs. Bindings require known graph ID and selected artifact ID. Roles are a nonempty unique list drawn from source, review, kernel, computation, alignment, teaching-source; no inferred acceptance or title join. Duplicate node/artifact pairs refuse, even with different roles. Explicit valid bindings sort by node_id, artifact_id; their roles sort lexically. Unknown mappings refuse; absent mappings remain unmapped.

## Exact public output

Projection exact root fields:
`{schema_version:1,scientific_effect:"NONE",scientific_status_authority:false,custody:"unknown",source,identities,columns,rows,bindings,unmapped_nodes}`.

source exactly `{capture_id,captured_at,spreadsheet_id,worksheet_id,worksheet_title,ranges,cell_fields,representation:"native-api-formattedValue-json"}`; preserve declared metadata/range order only. It contains no source URL, original packet, native annotations or raw capture bytes. identities exactly `{capture_sha256,metadata_sha256,policy_sha256,bindings_sha256,graph_nodes_sha256}`, matching the independently supplied pins. columns equals the eight public column names.

Every row exactly `{artifact_id,source_row,fields}`: source_row is original1-based native row, fields is an object with exactly the eight column names and original formatted strings. Do not interpret status, authority, review prose, URLs or scope into another evidence axis. Empty fields stay empty. Every binding exactly {node_id,artifact_id,roles} with explicit canonical sorting. unmapped_nodes is the sorted graph IDs with no binding. Success JSON is deterministic ASCII with sorted object keys and one terminal newline. No output timestamp, original private input, accepted marker or inferred formal join.

## External controls and completion

Tests launch the CLI as a subprocess with -B -S and propagate the active optimized mode. Before implementation, actual hosted RED must identify the missing CLI, not merely fail discovery/import/setup. Cover valid deterministic preservation and typing; multiple native ranges/default offsets/blank cells; every independently pinned identity; each required/duplicate/invalid flag; duplicate/nonfinite/deep/oversized JSON and canonical base64; native source/header/range/row/grid/type contradictions; unknown/duplicate artifact IDs and bindings; six explicit roles; unknown graph IDs; missing mappings; forged columns; private Source/Notes and unselected-row canaries absent from public stdout and refusal stderr; inert code/URL strings never fetched or executed. Positive selected public strings may themselves contain URLs/code as inert escaped data; omission is determined by explicit field/row policy.

Keep fixtures synthetic and compact. Do not copy any real private capture into tests or the public repository. Complete this unit only after exact candidate hosted normal/optimized controls pass with full discovery, original failure/success archives and nonauthor source/privacy review. Real selected source projection/readback follows that tested parser; the capture SHA alone does not authenticate a Sheet revision. Native graph archive readback, evidence-public projection, Astro/build/browser/deployment and all other architecture outcomes remain separate work.

## Proposed v2 extension: required node_id <-> artifact_path+sha binding (GB-5, proposal only)

Status: PROPOSAL under GRAPH-BIND-01 (main#475 directive 6363671135, slice GB-5), amended per review 522/5627565070 (agent 14) and questions from agent 7 (522/6363777423) and agent 1. Doc-only. It changes no code, workflow, CI, required check or setting. v1 behavior above stays in force until a separate tested implementation lands with nonauthor review. Wiring any check into CI needs Dylan's approval, routed through CoS. Scientific effect NONE. A binding records where an artifact lives at a pinned revision. It is never acceptance, review or discharge evidence, and math stays fail-closed. Every default below is fail-closed: when something is unclear, report it as not bound.

### Motivation

v1 bindings join graph nodes to Artifact Index rows only by `artifact_id`. `architecture/public-artifact-index-bindings.json` is currently `[]`, and all graph IDs are unmapped. Nothing records which repository file at which exact bytes backs a node, so no checker can report unbound or stale items.

### Location and root shape

- v2 lives in a **new file**, `architecture/graph-artifact-bindings.v2.json`. The v1 file `architecture/public-artifact-index-bindings.json` stays untouched, so v1 readers (`architecture/artifact_index.py`, `tools/evidence_path.py`) keep working.
- The root is exactly `{"schema_version": 2, "node_source": {...}, "scope": [...], "bindings": [...]}`. Unknown, missing or duplicate keys refuse.
  - `scope` is a sorted, unique list of reviewed repo-relative include globs. It defines the universe of in-scope artifacts. It is never inferred. Changing it is a reviewed doc/data change.
  - `node_source` names the node universe: `{"kind": "graph-readback" | "site-projection", "repository", "path", "commit"}`, pinned to a 40-hex commit. `node_id` values and the "unbound node" set come only from this one declared export at that pin. GB-6 and GB-7 must not choose their own sources.

### Binding record

Each entry in `bindings` is exactly this object:

- `node_id` (**required**): an ID present in `node_source`. Same ASCII pattern and 192-character bound as v1 IDs. There is no non-node subject.
- `kind` (**required**): one of `file`, `tree`, `historical`, `absent`, `external_unverified` (see below).
- `repository`: `owner/name`.
- `artifact_path`: a repo-relative POSIX path. It must be non-empty, contain no leading `/`, have no `.` or `..` segments, use no backslashes or control characters, and be at most 1024 UTF-8 bytes.
- `commit`: 40-hex lowercase. **Provenance only:** the commit where the pin was taken.
- `blob_sha`: 40-hex lowercase git object SHA. For `file`/`historical` it is the blob; for `tree` it is the tree object.
- `sha256`: 64-hex lowercase SHA-256 of the exact file bytes. Required for `file` and `historical`, and omitted for `tree`.
- `artifact_id` (**optional**): omitted unless the ID is already selected by `public-artifact-index-policy.json` (currently C151, C152, C215 only). `artifact_index.py` refuses anything else, so binding a file never requires a policy change.
- `roles`: a non-empty, unique, lexically sorted list drawn from the v1 role set.
- `reason`, `last_seen_commit`, `url`: only for the kinds that require them (below).

**Compact reference form** (for prose, logs and check output; the JSON record is authoritative): `owner/repo:path@commit#blob`, e.g. `d6g8k5htny-coder/main:architecture/artifact_index.py@7e9a532c13e3214bcb977a47372ad75c2c1b7b75#<40-hex blob>`. Use full 40-hex values in records, never abbreviations.

A binding is unique per (`node_id`, `repository`, `artifact_path`). One node may bind several artifacts, and one artifact may back several nodes. Records sort by `node_id`, then `repository`, then `artifact_path`.

### Evaluation (all states at the checked ref)

The checker evaluates every record at one **checked ref** (for example the current main commit), never at the record's own `commit`:

- **bound**: `kind: file`, the path exists at the checked ref, its blob equals `blob_sha` and its bytes hash to `sha256`. (`kind: tree`: the tree object at the path equals `blob_sha`.)
- **stale**: the path exists at the checked ref but its blob or tree differs. Reported, never silently repinned.
- **missing**: the path does not exist at the checked ref.
- **historical**: a `kind: historical` record (see Q1). Reported separately, never counted as bound.
- **absent**: a `kind: absent` record. Counted with missing, never bound.
- **external_unverified**: a `kind: external_unverified` record. Never bound.
- **unbound node**: a node in `node_source` that has no record of its own.
- **unbound artifact**: a file matched by `scope` that no `file` record names.

In report mode the check lists all states and exits 0. In enforce mode it exits nonzero on anything other than bound. Enforce mode is only used once Dylan approves CI wiring. Output is deterministic sorted ASCII JSON, matching v1 output rules.

### Fail-closed answers to open questions

1. **Deleted file at an old commit** (e.g. `STATUS_RN_UNIF.md@a3e42254`): allowed only as `kind: historical`, pinned by `commit`, `blob_sha` and `sha256`. It's reported as historical and never as current-bound. *Why:* the bytes are still verifiable from git history, but a node that only resolves to deleted content isn't backed by the current project.
2. **Directory as a source**: not as a loose path. Bind each file, or use `kind: tree` with an explicit tree `blob_sha`. *Why:* a path with no hash can change silently, and a tree SHA pins every byte below it.
3. **URL or `external:` sources**: never bound. Use `kind: external_unverified` with `url`, until a pinned hash or an archived copy exists in a repo, which can then be bound as `file`. *Why:* the checker does no network fetch and can't verify remote bytes.
4. **Inherited bindings from linked nodes**: not allowed. Every node needs its own record, and a node without one is unbound. *Why:* inheritance would hide gaps and let one binding stand in for many unverified claims.
5. **Absent files**: yes, explicit `kind: absent` with a required `reason` and `last_seen_commit`. It counts as missing, never bound. *Why:* it records known gaps on purpose without faking a pin.
6. **Files with no graph node** (agent 7 Q1): keep `node_id` required. No non-node subjects and no new Artifact Index rows. Such files stay reported as **unbound artifact**. A GB-7 lane binds a file only once a real node exists for it. Creating nodes is a separate, reviewed graph change, not part of a binding PR.
7. **Do `tools/` files need nodes?** (agent 7 Q2): no. `tools/` is excluded from the default `scope`, while `architecture/` stays in scope. Tool outputs, or the nodes they serve, are bound where those exist. A specific tool joins `scope` only through a reviewed scope change, when it is itself a public artifact (for example an export CLI cited from a node page). *Why:* this keeps unbound counts meaningful and avoids inventing nodes.

### Migration

1. v1 `{node_id, artifact_id, roles}` records remain valid in the v1 file. The v2 file rejects v1-shaped records.
2. Pins come from a reviewed source commit, never from the working tree of an unreviewed head.
3. Per-lane binding PRs (GB-7+) populate the v2 file only after GB-1..GB-4 inventories report, one PR per area.

### Non-goals

No inference of bindings from titles or prose, no inheritance, no network fetch, no Sheet writes, no node creation and no change to required checks. Binding status does not change scientific status.
