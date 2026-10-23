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

Status: PROPOSAL under GRAPH-BIND-01 (main#475 directive 6363671135, slice GB-5). Doc-only. This section changes no code, workflow, CI, required check or setting. v1 behavior above stays in force until a separate tested implementation lands with nonauthor review. Wiring any check into CI needs Dylan's approval, routed through CoS. Scientific effect NONE. A binding records where an artifact lives at a pinned revision. It is never acceptance, review or discharge evidence, and math stays fail-closed.

### Motivation

v1 bindings join graph nodes to Artifact Index rows only by `artifact_id`. `architecture/public-artifact-index-bindings.json` is currently `[]`, and all graph IDs are unmapped. Nothing records which repository file at which exact bytes backs a node, so no checker can report unbound or stale items.

### Binding record (schema_version 2)

Each binding is exactly this object. Unknown, missing or duplicate keys refuse:

- `node_id`: graph node ID, using the same ASCII pattern and 192-character bound as v1 IDs.
- `artifact_id`: optional Artifact Index ID (v1 semantics). If present, it must be selected by policy.
- `repository`: `owner/name`, using the `[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+` pattern.
- `artifact_path`: a repo-relative POSIX path. It must be non-empty, contain no leading `/`, have no `.` or `..` segments, use no backslashes or control characters, and be at most 1024 UTF-8 bytes.
- `commit`: a 40-hex lowercase commit at which the path was observed.
- `blob_sha`: the 40-hex lowercase git blob SHA of `artifact_path` at `commit`.
- `sha256`: the 64-hex lowercase SHA-256 of the exact file bytes at that blob.
- `roles`: a non-empty, unique, lexically sorted list drawn from the v1 role set.

A binding is unique per (`node_id`, `repository`, `artifact_path`). One node may bind several artifacts, and one artifact may back several nodes. Records sort by `node_id`, then `repository`, then `artifact_path`.

### Required states (what a GB-6-style check reports)

- **bound**: `artifact_path` exists at `commit`, and its blob and SHA-256 equal the pins.
- **stale**: the path exists at the checked ref, but its current blob differs from `blob_sha`. This is reported, not silently repinned.
- **missing**: the path does not exist at the checked ref.
- **unbound node**: a graph node with no binding record.
- **unbound artifact**: an in-scope file that no binding names. Scope is declared by an explicit, reviewed include-glob list in the bindings document, never inferred.

In report mode the check lists all five states and exits 0. In enforce mode it exits nonzero on stale, missing or unbound items. Enforce mode is only used once Dylan approves CI wiring. Output is deterministic sorted ASCII JSON, matching v1 output rules.

### Migration

1. v1 `{node_id, artifact_id, roles}` records remain valid as v1. A v2 document carries `schema_version: 2` at the root and rejects v1-shaped records.
2. Pins come from a reviewed source commit, never from the working tree of an unreviewed head.
3. Per-lane binding PRs (GB-7+) populate records only after GB-1..GB-4 inventories report, one PR per area.

### Non-goals

No inference of bindings from titles or prose, no network fetch, no Sheet writes, and no change to required checks. Binding status does not change scientific status.
